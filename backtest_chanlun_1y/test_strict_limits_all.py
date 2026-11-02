import bisect
import json
import pickle
from pathlib import Path
from pit_universe import get_candidate_pools
from data_loader import load_all_data_2y
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR
from run_backtest_10y_pit import precompute_pit_rosters, with_held_metas, KCB_ACTIVE_DATE
from run_backtest_10y_50_50 import get_daily_evaluated
from limit_Execution_filter import is_limit_up_close, is_limit_down_close
import market_regime_v2

base = Path(__file__).resolve().parent
EVAL_CACHE_FILE = base / "cache_10y_eval.pkl"

def get_day_bar_at(indexed_data, sym, dt):
    idx_e = indexed_data.get(sym)
    if not idx_e:
        return None
    pos = bisect.bisect_right(idx_e["day_dates"], dt) - 1
    if pos >= 0 and idx_e["day_dates"][pos] == dt:
        return idx_e["day"][pos]
    return None

def step_subaccount_strict(state, universe, indexed_data, bench_day_bars, dt, eval_cache,
                           base_stock_cap=250000.0, base_etf_cap=300000.0, base_min_cash=150000.0,
                           regime_mode="dual_factor", use_ratchet_stop=True, enforce_limits=True,
                           route_fn=None):
    holdings = state["holdings"]
    cash = state["cash"]

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

    regime, r_cfg, r_meta = market_regime_v2.classify_market_regime(bench_day_bars, day_snaps, evaluated, mode=regime_mode)
    state["regime_counts"][regime] = state["regime_counts"].get(regime, 0) + 1

    # 1. SELL check (with Limit-Down No-Sell & Next-Day Forced Queue)
    for sym in list(holdings.keys()):
        h = holdings[sym]
        ev = eval_map.get(sym)
        snap = day_snaps.get(sym)
        if not ev and not h.get("pending_forced_sell"):
            continue

        sell_reason = h.get("pending_sell_reason")
        if not sell_reason and ev:
            p_m15 = ev["periods"].get("15m")
            if p_m15:
                zs15 = p_m15.get("lastZs")
                bi15 = p_m15.get("lastBi")
                div15 = p_m15.get("divRatio")
                if use_ratchet_stop and regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
                    sell_reason = f"[熊市锁死止损] {h['name']}({sym})跌破初始止损位({h['init_stop']})"
                elif zs15 and ev["price"] < zs15["dd"]:
                    sell_reason = f"[结构破位止损] {h['name']}({sym})跌破15m中枢DD({zs15['dd']})"
                elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                    sell_reason = f"[15m顶背驰一卖] {h['name']}({sym})面积比={div15}"

        if sell_reason:
            dbar = get_day_bar_at(indexed_data, sym, dt)
            if enforce_limits and is_limit_down_close(sym, dt, snap, dbar):
                # 封死跌停卖不出！标记强制顺延至次日卖出
                h["pending_forced_sell"] = True
                h["pending_sell_reason"] = sell_reason + f" (注:{dt}封跌停无法卖出,顺延次日强平)"
                state["limit_stats"]["blocked_limit_down_sells"] += 1
                continue

            price = h["last_price"]
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
                "pnl_pct": round(pnl_pct, 2), "reason": sell_reason
            })
            state["trades"].append({
                "date": dt, "action": "SELL", "symbol": sym, "name": h["name"],
                "shares": shares, "price": price, "amount": round(gross, 2),
                "fee": round(fee, 2), "regime": regime, "reason": sell_reason
            })
            del holdings[sym]

    # 2. BUY check (with Limit-Up No-Buy Filter)
    if route_fn is not None:
        ranked_cands = sorted(evaluated, key=lambda c: route_fn(c, regime), reverse=True)
    else:
        ranked_cands = evaluated

    mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
    cur_equity = cash + mkt_val_now
    remaining_room = max(0.0, cur_equity * r_cfg["max_exposure_pct"] - mkt_val_now)

    daily_limit = r_cfg["daily_buy_limit"]
    min_score = r_cfg["min_score"]
    pos_scale = r_cfg["single_pos_scale"]
    eff_min_cash = base_min_cash * max(pos_scale, 0.5)
    budget = min(cash, remaining_room)
    buy_count = 0

    if daily_limit > 0 and remaining_room >= eff_min_cash:
        for cand in ranked_cands:
            if buy_count >= daily_limit:
                break
            sym = cand["symbol"]
            if sym in holdings:
                continue
            if cand["score"] < min_score or budget < eff_min_cash or cand["price"] <= 0:
                continue

            snap = day_snaps.get(sym)
            dbar = get_day_bar_at(indexed_data, sym, dt)
            if enforce_limits and is_limit_up_close(sym, dt, snap, dbar):
                # 收盘封死涨停买不进！直接跳过顺延下一只
                state["limit_stats"]["blocked_limit_up_buys"] += 1
                continue

            target_amt = (base_stock_cap if cand["kind"] == "stock" else base_etf_cap) * pos_scale
            alloc = min(budget * 0.95, target_amt, remaining_room)
            lots = int(alloc // (cand["price"] * 100.0))
            shares = lots * 100
            if shares < 100:
                continue

            p_m15 = cand["periods"].get("15m")
            zs15 = p_m15.get("lastZs") if p_m15 else None
            raw_stop = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
            floor_stop = round(cand["price"] * (1.0 - r_cfg["max_stop_pct"]), 3)
            stop_loss = max(raw_stop, floor_stop) if use_ratchet_stop else raw_stop

            gross = shares * cand["price"]
            fee = max(gross * 0.00025, 5.0)
            total_cost = gross + fee
            if total_cost > cash:
                continue

            cash -= total_cost
            budget -= gross
            remaining_room -= gross
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
                "regime": regime, "reason": f"[{r_cfg['label']}] 买入{cand['name']}({sym}) (分={cand['score']})"
            })

    state["cash"] = cash
    mkt_val = sum(h["shares"] * h["last_price"] for h in holdings.values())
    state["equity"] = round(cash + mkt_val, 2)
    return r_meta

def run_v2_2_with_limits(trade_dates, rosters, fmap, indexed, bmap, eval_c, enforce_limits=True, use_stop=True):
    market_regime_v2._BENCH_CACHE.clear()
    w_hs, w_cyb, w_kcb = 0.20, 0.50, 0.30
    init_cyb_w = 0.80
    is_pre_kcb = trade_dates[0] < KCB_ACTIVE_DATE
    s_hs = {"cash": 200000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 200000.0, "regime_counts": {}, "limit_stats": {"blocked_limit_up_buys": 0, "blocked_limit_down_sells": 0}}
    s_cyb = {"cash": 800000.0 if is_pre_kcb else 500000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 800000.0 if is_pre_kcb else 500000.0, "regime_counts": {}, "limit_stats": {"blocked_limit_up_buys": 0, "blocked_limit_down_sells": 0}}
    s_kcb = {"cash": 0.0 if is_pre_kcb else 300000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 0.0 if is_pre_kcb else 300000.0, "regime_counts": {}, "limit_stats": {"blocked_limit_up_buys": 0, "blocked_limit_down_sells": 0}}
    kcb_seeded = not is_pre_kcb
    kcb_target_transfer = 0.0
    bench_idx = indexed["sh000300"]
    hist = []

    for i, dt in enumerate(trade_dates):
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        sl = bench_idx["day"][max(0, b_pos - 250):b_pos]
        hs_u, cyb_u, kcb_u = rosters[dt]
        kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)

        if kcb_active and not kcb_seeded:
            comb_eq = s_cyb["equity"] + s_kcb["equity"]
            kcb_target_transfer = comb_eq * (w_kcb / init_cyb_w)
            kcb_seeded = True
        if kcb_active and kcb_target_transfer > 100.0 and s_cyb["cash"] > 100.0:
            xfer = min(kcb_target_transfer, s_cyb["cash"])
            s_cyb["cash"] -= xfer; s_kcb["cash"] += xfer; kcb_target_transfer -= xfer
            s_cyb["equity"] = round(s_cyb["cash"] + sum(h["shares"]*h["last_price"] for h in s_cyb["holdings"].values()), 2)
            s_kcb["equity"] = round(s_kcb["cash"] + sum(h["shares"]*h["last_price"] for h in s_kcb["holdings"].values()), 2)

        cur_w_cyb = w_cyb if kcb_active else init_cyb_w
        cur_w_kcb = w_kcb if kcb_active else 0.0

        step_subaccount_strict(s_hs, with_held_metas(hs_u, s_hs["holdings"], fmap), indexed, sl, dt, eval_c, 50000.0, 60000.0, 30000.0, use_ratchet_stop=use_stop, enforce_limits=enforce_limits)
        step_subaccount_strict(s_cyb, with_held_metas(cyb_u, s_cyb["holdings"], fmap), indexed, sl, dt, eval_c, 250000.0*cur_w_cyb, 300000.0*cur_w_cyb, 150000.0*cur_w_cyb, use_ratchet_stop=use_stop, enforce_limits=enforce_limits)
        if cur_w_kcb > 0:
            step_subaccount_strict(s_kcb, with_held_metas(kcb_u, s_kcb["holdings"], fmap), indexed, sl, dt, eval_c, 75000.0, 90000.0, 45000.0, use_ratchet_stop=use_stop, enforce_limits=enforce_limits)

        if i + 1 < len(trade_dates) and dt[:7].endswith("-10") and dt[:7] != trade_dates[i+1][:7]:
            tot_eq = s_hs["equity"] + s_cyb["equity"] + s_kcb["equity"]
            subs = [(s_hs, tot_eq*w_hs), (s_cyb, tot_eq*cur_w_cyb), (s_kcb, tot_eq*cur_w_kcb)]
            surplus = 0.0; defs = []
            for sub, tgt in subs:
                diff = sub["equity"] - tgt
                if diff > 0:
                    g = min(diff, sub["cash"]); sub["cash"] -= g; surplus += g
                elif diff < 0:
                    defs.append((sub, -diff))
            tdef = sum(d for _, d in defs)
            if tdef > 0 and surplus > 0:
                for sub, d in defs: sub["cash"] += surplus * (d / tdef)
            for sub, _ in subs:
                sub["equity"] = round(sub["cash"] + sum(h["shares"]*h["last_price"] for h in sub["holdings"].values()), 2)

        tot = round(s_hs["equity"] + s_cyb["equity"] + s_kcb["equity"], 2)
        hist.append({"date": dt, "portfolio_value": tot, "portfolio_return_pct": round((tot-1e6)/1e6*100, 2), "hs300_level": bmap[dt], "hs300_return_pct": round((bmap[dt]-bmap[trade_dates[0]])/bmap[trade_dates[0]]*100, 2), "alpha_pct": 0.0})

    m = compute_metrics({"initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"], "history": hist, "trades": s_hs["trades"]+s_cyb["trades"]+s_kcb["trades"], "closed_trades": s_hs["closed_trades"]+s_cyb["closed_trades"]+s_kcb["closed_trades"]})
    lu_blk = s_hs["limit_stats"]["blocked_limit_up_buys"] + s_cyb["limit_stats"]["blocked_limit_up_buys"] + s_kcb["limit_stats"]["blocked_limit_down_sells"] if False else (s_hs["limit_stats"]["blocked_limit_up_buys"] + s_cyb["limit_stats"]["blocked_limit_up_buys"] + s_kcb["limit_stats"]["blocked_limit_up_buys"])
    ld_blk = s_hs["limit_stats"]["blocked_limit_down_sells"] + s_cyb["limit_stats"]["blocked_limit_down_sells"] + s_kcb["limit_stats"]["blocked_limit_down_sells"]
    return m, hist, lu_blk, ld_blk

if __name__ == "__main__":
    hs_pool, cyb_pool, kcb_pool = get_candidate_pools()
    all_syms = list({m[0] for m in (hs_pool + cyb_pool + kcb_pool)} | {"sh000300"})

    # 1. Test 1Y
    raw_2y = load_all_data_2y(all_syms, max_workers=12)
    cache_1y_dir = base / "cache"
    for s in all_syms:
        cf1 = cache_1y_dir / f"{s}.pkl"
        if cf1.exists():
            try:
                d1 = pickle.load(open(cf1, "rb"))
                if d1.get("m15"): raw_2y[s] = d1
            except Exception: pass
    idx_1y = prepare_indexed_data(raw_2y)
    bench_1y = idx_1y["sh000300"]["day"]
    bmap_1y = {b["time"]: b["close"] for b in bench_1y}
    td_1y = [b["time"] for b in bench_1y if "2025-10-29" <= b["time"] <= "2026-10-29"]
    rosters_1y, fmap = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_1y, td_1y)
    eval_1y = {}
    for enf in [False, True]:
        m, h, lu, ld = run_v2_2_with_limits(td_1y, rosters_1y, fmap, idx_1y, bmap_1y, eval_1y, enforce_limits=enf, use_stop=True)
        pk = max(h, key=lambda x: x["portfolio_value"])
        print(f"[1Y V2.2 20:50:30] 严苛涨跌停限制={str(enf):5s} | 期末:¥{m['final_value']/1e4:6.2f}w ({m['total_return_pct']:+6.2f}%) | 最高:¥{pk['portfolio_value']/1e4:6.2f}w | 最大回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:4.2f} | 拦截涨停买:{lu:2d}笔, 拦截跌停卖:{ld:2d}笔")

    # 2. Test 10Y
    raw_10y = {s: pickle.load(open(CACHE_10Y_DIR / f"{s}.pkl", "rb")) for s in all_syms if (CACHE_10Y_DIR / f"{s}.pkl").exists()}
    ds_10y = build_period_dataset(raw_10y, "2015-10-01", "2026-10-29", "2026-10-29")
    idx_10y = prepare_indexed_data(ds_10y)
    bench_10y = idx_10y["sh000300"]["day"]
    bmap_10y = {b["time"]: b["close"] for b in bench_10y}
    td_10y = [b["time"] for b in bench_10y if "2016-10-28" <= b["time"] <= "2026-10-29"]
    rosters_10y, _ = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_10y, td_10y)
    eval_10y = pickle.load(open(EVAL_CACHE_FILE, "rb")) if EVAL_CACHE_FILE.exists() else {}
    for enf in [False, True]:
        m, h, lu, ld = run_v2_2_with_limits(td_10y, rosters_10y, fmap, idx_10y, bmap_10y, eval_10y, enforce_limits=enf, use_stop=False)
        pk = max(h, key=lambda x: x["portfolio_value"])
        print(f"[10Y V2.2 20:50:30] 严苛涨跌停限制={str(enf):5s} | 期末:¥{m['final_value']/1e4:6.2f}w ({m['total_return_pct']:+6.2f}%) | 最高:¥{pk['portfolio_value']/1e4:6.2f}w | 最大回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:4.2f} | 拦截涨停买:{lu:3d}笔, 拦截跌停卖:{ld:2d}笔")
