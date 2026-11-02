import bisect
import json
import math
import pickle
from pathlib import Path
from universe import get_universes
from chanlun_engine import evaluate_asset, get_stage1_candidates
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR

def get_daily_evaluated(universe, day_snaps, holdings, indexed_data, dt, eval_cache):
    stage1_selected = get_stage1_candidates(universe, day_snaps, holdings, top_k=44)
    evaluated = []
    for meta in stage1_selected:
        sym = meta[0]
        ck = (sym, dt)
        if ck in eval_cache:
            ev = eval_cache[ck]
            if ev is not None:
                evaluated.append(ev)
            continue
        idx_e = indexed_data.get(sym)
        if not idx_e:
            eval_cache[ck] = None
            continue
        i_d = bisect.bisect_right(idx_e["day_dates"], dt)
        i_60 = bisect.bisect_right(idx_e["m60_dates"], dt)
        i_15 = bisect.bisect_right(idx_e["m15_dates"], dt)
        d_bars = idx_e["day"][max(0, i_d - 250):i_d]
        m60_bars = idx_e["m60"][max(0, i_60 - 280):i_60]
        m15_bars = idx_e["m15"][max(0, i_15 - 280):i_15]
        if not d_bars or not m15_bars:
            eval_cache[ck] = None
            continue
        ev = evaluate_asset(meta, d_bars, m60_bars, m15_bars, day_snaps.get(sym))
        eval_cache[ck] = ev
        evaluated.append(ev)

    evaluated.sort(key=lambda x: (
        x["score"],
        x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0,
        x["snapshot"]["amountYi"] if x["snapshot"] else 0.0
    ), reverse=True)
    return evaluated

def step_subaccount(state, universe, indexed_data, dt, eval_cache, dynamic_sizing=False):
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

    # 1. SELL check
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
        stop_ref = round(ev["price"] * 0.96, 3)

        if zs15 and ev["price"] < zs15["dd"]:
            stop_ref = zs15["dd"]
            sell_reason = f"[纯量化规则] {h['name']}({sym})跌破15m中枢极值DD({zs15['dd']})触发结构破位止损，卖出全部{h['shares']}股。"
        elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
            if zs15:
                stop_ref = round(zs15["zd"] * 0.96, 3)
            sell_reason = f"[纯量化规则] {h['name']}({sym})出现15m向上笔盘整顶背驰一卖(面积比={div15})，卖出{h['shares']}股。"

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
                "pnl_pct": round(pnl_pct, 2), "reason": sell_reason
            })
            state["trades"].append({"date": dt, "action": "SELL", "symbol": sym, "shares": shares, "price": price, "amount": round(gross, 2), "fee": round(fee, 2)})
            del holdings[sym]

    # 2. BUY check
    mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
    cur_equity = cash + mkt_val_now
    if dynamic_sizing:
        stock_cap = cur_equity * 0.25
        etf_cap = cur_equity * 0.30
        min_cash = cur_equity * 0.15
    else:
        stock_cap = 125000.0
        etf_cap = 150000.0
        min_cash = 75000.0

    budget = cash
    buy_count = 0
    for cand in evaluated:
        if buy_count >= 2:
            break
        sym = cand["symbol"]
        if sym in holdings:
            continue
        if cand["score"] < 45 or budget < min_cash or cand["price"] <= 0:
            continue
        target_amt = stock_cap if cand["kind"] == "stock" else etf_cap
        alloc = min(budget * 0.95, target_amt)
        lots = int(alloc // (cand["price"] * 100.0))
        shares = lots * 100
        if shares < 100:
            continue
        p_m15 = cand["periods"].get("15m")
        zs15 = p_m15.get("lastZs") if p_m15 else None
        stop_loss = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
        gross = shares * cand["price"]
        fee = max(gross * 0.00025, 5.0)
        total_cost = gross + fee
        if total_cost > cash:
            continue
        cash -= total_cost
        budget -= gross
        buy_count += 1
        holdings[sym] = {
            "name": cand["name"], "shares": shares,
            "avg_cost": round(total_cost / shares, 4),
            "last_price": cand["price"], "buy_date": dt
        }
        state["trades"].append({"date": dt, "action": "BUY", "symbol": sym, "shares": shares, "price": cand["price"], "amount": round(gross, 2), "fee": round(fee, 2)})

    state["cash"] = cash
    mkt_val = sum(h["shares"] * h["last_price"] for h in holdings.values())
    state["equity"] = round(cash + mkt_val, 2)

def run_50_50_simulation(mode_name, hs300_u, all_a_u, indexed_data, trade_dates, bench_map, eval_cache, annual_rebalance=False, dynamic_sizing=False):
    initial_capital = 1000000.0
    s_hs = {"cash": 500000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 500000.0}
    s_aa = {"cash": 500000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 500000.0}
    init_bench = bench_map[trade_dates[0]]
    history = []

    for i, dt in enumerate(trade_dates):
        step_subaccount(s_hs, hs300_u, indexed_data, dt, eval_cache, dynamic_sizing=dynamic_sizing)
        step_subaccount(s_aa, all_a_u, indexed_data, dt, eval_cache, dynamic_sizing=dynamic_sizing)

        # Annual 50/50 rebalance at end of October each year
        if annual_rebalance and i + 1 < len(trade_dates):
            cur_ym = dt[:7]
            nxt_ym = trade_dates[i + 1][:7]
            if cur_ym.endswith("-10") and cur_ym != nxt_ym:
                tot_eq = s_hs["equity"] + s_aa["equity"]
                target_eq = tot_eq * 0.5
                diff_hs = target_eq - s_hs["equity"]
                if diff_hs > 0:
                    # transfer cash from s_aa to s_hs
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
        alpha = round(tot_ret - hs300_ret, 2)
        history.append({
            "date": dt, "portfolio_value": tot_val,
            "hs300_sub_value": s_hs["equity"], "alla_sub_value": s_aa["equity"],
            "cash": round(s_hs["cash"] + s_aa["cash"], 2),
            "holdings_count": len(s_hs["holdings"]) + len(s_aa["holdings"]),
            "portfolio_return_pct": tot_ret, "hs300_level": hs300_lvl,
            "hs300_return_pct": hs300_ret, "alpha_pct": alpha
        })

    all_trades = s_hs["trades"] + s_aa["trades"]
    all_closed = s_hs["closed_trades"] + s_aa["closed_trades"]
    res = {
        "account_name": mode_name, "initial_capital": initial_capital,
        "final_value": history[-1]["portfolio_value"],
        "final_hs300_sub": s_hs["equity"], "final_alla_sub": s_aa["equity"],
        "history": history, "trades": all_trades, "closed_trades": all_closed
    }
    res["metrics"] = compute_metrics(res)
    return res

if __name__ == "__main__":
    hs300_u, all_a_u = get_universes()
    cache_dir = Path(__file__).resolve().parent / "cache"
    cached_syms = {p.stem for p in cache_dir.glob("*.pkl")}
    if cached_syms:
        all_a_u = [m for m in all_a_u if m[0] in cached_syms]
    all_syms = ["sh000300"] + [m[0] for m in all_a_u]

    raw_cache = {}
    for s in all_syms:
        cf = CACHE_10Y_DIR / f"{s}.pkl"
        if cf.exists():
            with open(cf, "rb") as f:
                raw_cache[s] = pickle.load(f)

    dataset = build_period_dataset(raw_cache, "2015-10-01", "2026-10-29", "2026-10-29")
    indexed = prepare_indexed_data(dataset)
    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2016-10-28" <= b["time"] <= "2026-10-29"]

    eval_cache = {}
    print("Running 50/50 Split Mode 1 (Independent 50w + 50w, fixed 12.5w/trade)...")
    res_ind = run_50_50_simulation("方案①：50万沪深300 + 50万全A (独立不调仓·单笔12.5万)", hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache, annual_rebalance=False, dynamic_sizing=False)

    print("Running 50/50 Split Mode 2 (Annual 50/50 Rebalance, fixed 12.5w/trade)...")
    res_reb = run_50_50_simulation("方案②：50万沪深300 + 50万全A (每年末50:50再平衡·单笔12.5万)", hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache, annual_rebalance=True, dynamic_sizing=False)

    print("Running 50/50 Split Mode 3 (Annual 50/50 Rebalance + Dynamic 25% sizing)...")
    res_dyn = run_50_50_simulation("方案③：50万沪深300 + 50万全A (每年末再平衡+动态仓位复利)", hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache, annual_rebalance=True, dynamic_sizing=True)

    out_path = Path(__file__).resolve().parent / "backtest_results_10y_50_50.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"independent": res_ind, "rebalanced": res_reb, "dynamic_rebalanced": res_dyn}, f, ensure_ascii=False, indent=2)

    for k, r in [("Independent", res_ind), ("Rebalanced", res_reb), ("Dynamic+Reb", res_dyn)]:
        pk = max(r["history"], key=lambda x: x["portfolio_value"])
        tr = min(r["history"], key=lambda x: x["portfolio_value"])
        print(f"{k}: Final=¥{r['final_value']:,.2f} (HS300_sub=¥{r['final_hs300_sub']:,.2f}, AllA_sub=¥{r['final_alla_sub']:,.2f}) | Peak=¥{pk['portfolio_value']:,.2f} ({pk['date']}) | Trough=¥{tr['portfolio_value']:,.2f} ({tr['date']})")
        print("  Metrics:", json.dumps(r["metrics"], ensure_ascii=False))
