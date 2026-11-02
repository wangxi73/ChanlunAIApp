import bisect
import json
import pickle
from pathlib import Path
from universe import get_universes
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR
from run_backtest_10y_50_50 import get_daily_evaluated
from market_regime_v2 import classify_market_regime

def step_subaccount_v2(state, universe, indexed_data, bench_day_bars, dt, eval_cache,
                       base_stock_cap=250000.0, base_etf_cap=300000.0, base_min_cash=150000.0,
                       regime_mode="dual_factor", use_ratchet_stop=True):
    holdings = state["holdings"]
    cash = state["cash"]

    # 1. Update holdings price
    for sym, h in holdings.items():
        snap = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
        if snap and snap["lastPrice"] > 0:
            h["last_price"] = snap["lastPrice"]

    day_snaps = {}
    for sym, _, _ in universe:
        s = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
        if s:
            day_snaps[sym] = s

    evaluated = get_daily_evaluated(universe, day_snaps, holdings, indexed_data, dt, eval_cache)
    eval_map = {e["symbol"]: e for e in evaluated}

    # 2. Classify today's market regime (BULL / RANGE / WEAK / CRASH)
    regime, r_cfg, r_meta = classify_market_regime(bench_day_bars, day_snaps, evaluated, mode=regime_mode)
    state["regime_counts"][regime] = state["regime_counts"].get(regime, 0) + 1

    # 3. SELL check (with optional Ratchet Hard Stop & Crash Tightening)
    for sym in list(holdings.keys()):
        h = holdings[sym]
        ev = eval_map.get(sym)
        if not ev:
            continue
        p_m15 = ev["periods"].get("15m")
        if not p_m15:
            continue
        zs15 = p_m15.get("lastZs")
        bi15 = p_m15.get("lastBi")
        div15 = p_m15.get("divRatio")

        sell_reason = None
        # 在熊市(WEAK/CRASH)下锁死建仓初始止损位(绝不允许随新低位中枢下移深套)；在牛市/结构市保持原版缠论DD结构止损防洗盘
        if use_ratchet_stop and regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
            sell_reason = f"[V2熊市锁死止损] {h['name']}({sym})在{r_cfg['label']}下跌破初始止损位({h['init_stop']})果断离场"
        elif zs15 and ev["price"] < zs15["dd"]:
            sell_reason = f"[纯量化规则] {h['name']}({sym})跌破15m中枢极值DD({zs15['dd']})触发结构破位止损"
        elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
            sell_reason = f"[纯量化规则] {h['name']}({sym})出现15m向上笔盘整顶背驰一卖(面积比={div15})"

        if sell_reason:
            price = ev["price"]
            shares = h["shares"]
            gross = shares * price
            fee = max(gross * 0.0005, 5.0)
            net = gross - fee
            cash += net
            pnl_amt = net - (shares * h["avg_cost"])
            pnl_pct = (price - h["avg_cost"]) / h["avg_cost"] * 100.0
            state["closed_trades"].append({
                "symbol": sym, "name": h["name"], "buy_date": h["buy_date"],
                "sell_date": dt, "shares": shares, "buy_price": h["avg_cost"],
                "sell_price": price, "pnl_amt": round(pnl_amt, 2),
                "pnl_pct": round(pnl_pct, 2), "reason": sell_reason,
                "buy_regime": h.get("buy_regime", "UNKNOWN")
            })
            state["trades"].append({
                "date": dt, "action": "SELL", "symbol": sym, "name": h["name"],
                "shares": shares, "price": price, "amount": round(gross, 2),
                "fee": round(fee, 2), "regime": regime, "reason": sell_reason
            })
            del holdings[sym]

    # 4. Dynamic Position & Trade Frequency BUY Valves based on Market Regime
    mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
    cur_equity = cash + mkt_val_now
    max_allowed_mkt_val = cur_equity * r_cfg["max_exposure_pct"]
    remaining_exposure_room = max(0.0, max_allowed_mkt_val - mkt_val_now)

    daily_limit = r_cfg["daily_buy_limit"]
    min_score = r_cfg["min_score"]
    pos_scale = r_cfg["single_pos_scale"]
    eff_min_cash = base_min_cash * max(pos_scale, 0.5)

    budget = min(cash, remaining_exposure_room)
    buy_count = 0

    if daily_limit > 0 and remaining_exposure_room >= eff_min_cash:
        for cand in evaluated:
            if buy_count >= daily_limit:
                break
            sym = cand["symbol"]
            if sym in holdings:
                continue
            if cand["score"] < min_score or budget < eff_min_cash or cand["price"] <= 0:
                continue

            target_amt = (base_stock_cap if cand["kind"] == "stock" else base_etf_cap) * pos_scale
            alloc = min(budget * 0.95, target_amt, remaining_exposure_room)
            lots = int(alloc // (cand["price"] * 100.0))
            shares = lots * 100
            if shares < 100:
                continue

            p_m15 = cand["periods"].get("15m")
            zs15 = p_m15.get("lastZs") if p_m15 else None
            raw_stop = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
            # Cap maximum initial stop distance by regime max_stop_pct
            floor_stop = round(cand["price"] * (1.0 - r_cfg["max_stop_pct"]), 3)
            stop_loss = max(raw_stop, floor_stop) if use_ratchet_stop else raw_stop

            gross = shares * cand["price"]
            fee = max(gross * 0.00025, 5.0)
            total_cost = gross + fee
            if total_cost > cash:
                continue

            cash -= total_cost
            budget -= gross
            remaining_exposure_room -= gross
            buy_count += 1
            holdings[sym] = {
                "name": cand["name"], "shares": shares,
                "avg_cost": round(total_cost / shares, 4),
                "last_price": cand["price"], "buy_date": dt,
                "stop_loss": stop_loss, "init_stop": raw_stop, "buy_regime": regime
            }
            state["trades"].append({
                "date": dt, "action": "BUY", "symbol": sym, "name": cand["name"],
                "shares": shares, "price": cand["price"], "amount": round(gross, 2),
                "fee": round(fee, 2), "stop_loss": stop_loss, "score": cand["score"],
                "regime": regime, "reason": f"[{r_cfg['label']}] 买入{cand['name']}({sym}) (分={cand['score']}, 仓位系数={pos_scale})"
            })

    state["cash"] = cash
    mkt_val = sum(h["shares"] * h["last_price"] for h in holdings.values())
    state["equity"] = round(cash + mkt_val, 2)
    return r_meta

def simulate_v2_portfolio(name, w_hs, hs300_u, all_a_u, indexed_data, trade_dates, bench_map, eval_cache,
                          initial_capital=1000000.0, rebalance=True, regime_mode="dual_factor", use_ratchet_stop=True):
    w_aa = round(1.0 - w_hs, 2)
    scale = initial_capital / 100000.0
    s_hs = {"cash": initial_capital * w_hs, "holdings": {}, "trades": [], "closed_trades": [], "equity": initial_capital * w_hs, "regime_counts": {}}
    s_aa = {"cash": initial_capital * w_aa, "holdings": {}, "trades": [], "closed_trades": [], "equity": initial_capital * w_aa, "regime_counts": {}}

    hs_stock_cap = 25000.0 * scale * w_hs if w_hs > 0 else 0.0
    hs_etf_cap = 30000.0 * scale * w_hs if w_hs > 0 else 0.0
    hs_min_cash = 15000.0 * scale * w_hs if w_hs > 0 else 1e9

    aa_stock_cap = 25000.0 * scale * w_aa if w_aa > 0 else 0.0
    aa_etf_cap = 30000.0 * scale * w_aa if w_aa > 0 else 0.0
    aa_min_cash = 15000.0 * scale * w_aa if w_aa > 0 else 1e9

    init_bench = bench_map[trade_dates[0]]
    bench_idx = indexed_data["sh000300"]
    history = []

    for i, dt in enumerate(trade_dates):
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]

        r_meta = None
        if w_hs > 0:
            r_meta = step_subaccount_v2(
                s_hs, hs300_u, indexed_data, bench_day_slice, dt, eval_cache,
                base_stock_cap=hs_stock_cap, base_etf_cap=hs_etf_cap, base_min_cash=hs_min_cash,
                regime_mode=regime_mode, use_ratchet_stop=use_ratchet_stop
            )
        if w_aa > 0:
            r_meta = step_subaccount_v2(
                s_aa, all_a_u, indexed_data, bench_day_slice, dt, eval_cache,
                base_stock_cap=aa_stock_cap, base_etf_cap=aa_etf_cap, base_min_cash=aa_min_cash,
                regime_mode=regime_mode, use_ratchet_stop=use_ratchet_stop
            )

        if rebalance and 0.0 < w_hs < 1.0 and i + 1 < len(trade_dates):
            cur_ym = dt[:7]
            nxt_ym = trade_dates[i + 1][:7]
            if cur_ym.endswith("-10") and cur_ym != nxt_ym:
                tot_eq = s_hs["equity"] + s_aa["equity"]
                target_hs = tot_eq * w_hs
                diff_hs = target_hs - s_hs["equity"]
                if diff_hs > 0:
                    xfer = min(diff_hs, s_aa["cash"])
                    s_aa["cash"] -= xfer
                    s_hs["cash"] += xfer
                elif diff_hs < 0:
                    xfer = min(-diff_hs, s_hs["cash"])
                    s_hs["cash"] -= xfer
                    s_aa["cash"] += xfer
                s_hs["equity"] = round(s_hs["cash"] + sum(h["shares"] * h["last_price"] for h in s_hs["holdings"].values()), 2)
                s_aa["equity"] = round(s_aa["cash"] + sum(h["shares"] * h["last_price"] for h in s_aa["holdings"].values()), 2)

        tot_val = round(s_hs["equity"] + s_aa["equity"], 2)
        tot_ret = round((tot_val - initial_capital) / initial_capital * 100.0, 2)
        hs300_lvl = bench_map[dt]
        hs300_ret = round((hs300_lvl - init_bench) / init_bench * 100.0, 2)
        history.append({
            "date": dt, "portfolio_value": tot_val,
            "cash": round(s_hs["cash"] + s_aa["cash"], 2),
            "holdings_count": len(s_hs["holdings"]) + len(s_aa["holdings"]),
            "regime": r_meta["regime"] if r_meta else "RANGE",
            "portfolio_return_pct": tot_ret, "hs300_level": hs300_lvl,
            "hs300_return_pct": hs300_ret, "alpha_pct": round(tot_ret - hs300_ret, 2)
        })

    res = {
        "account_name": name, "initial_capital": initial_capital,
        "final_value": history[-1]["portfolio_value"],
        "regime_counts": s_aa["regime_counts"] if w_aa > 0 else s_hs["regime_counts"],
        "history": history, "trades": s_hs["trades"] + s_aa["trades"],
        "closed_trades": s_hs["closed_trades"] + s_aa["closed_trades"]
    }
    m = compute_metrics(res)
    pk = max(history, key=lambda x: x["portfolio_value"])
    tr = min(history, key=lambda x: x["portfolio_value"])
    m["peak_value"] = pk["portfolio_value"]
    m["peak_return_pct"] = pk["portfolio_return_pct"]
    m["peak_date"] = pk["date"]
    m["trough_value"] = tr["portfolio_value"]
    m["trough_return_pct"] = tr["portfolio_return_pct"]
    m["trough_date"] = tr["date"]
    res["metrics"] = m
    return res
