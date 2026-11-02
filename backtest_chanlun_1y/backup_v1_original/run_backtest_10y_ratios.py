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
from run_backtest_10y_50_50 import get_daily_evaluated

def step_sub(state, universe, indexed_data, dt, eval_cache, stock_cap, etf_cap, min_cash):
    if state["equity"] <= 100.0 and not state["holdings"]:
        return
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
        if zs15 and ev["price"] < zs15["dd"]:
            sell_reason = "DD_STOP"
        elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
            sell_reason = "15M_S1"

        if sell_reason:
            price = ev["price"]
            shares = h["shares"]
            gross = shares * price
            fee = max(gross * 0.0005, 5.0)
            net = gross - fee
            cash += net
            pnl_amt = net - (shares * h["avg_cost"])
            pnl_pct = (price - h["avg_cost"]) / h["avg_cost"] * 100.0
            state["closed_trades"].append({"pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2)})
            state["trades"].append({"action": "SELL"})
            del holdings[sym]

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
        state["trades"].append({"action": "BUY"})

    state["cash"] = cash
    mkt_val = sum(h["shares"] * h["last_price"] for h in holdings.values())
    state["equity"] = round(cash + mkt_val, 2)

def run_ratio_sim(w_hs, hs300_u, all_a_u, indexed_data, trade_dates, bench_map, eval_cache, rebalance=True):
    initial_capital = 1000000.0
    w_aa = round(1.0 - w_hs, 2)
    s_hs = {"cash": initial_capital * w_hs, "holdings": {}, "trades": [], "closed_trades": [], "equity": initial_capital * w_hs}
    s_aa = {"cash": initial_capital * w_aa, "holdings": {}, "trades": [], "closed_trades": [], "equity": initial_capital * w_aa}

    hs_stock_cap = max(250000.0 * w_hs, 25000.0) if w_hs > 0 else 0.0
    hs_etf_cap = max(300000.0 * w_hs, 30000.0) if w_hs > 0 else 0.0
    hs_min_cash = max(150000.0 * w_hs, 15000.0) if w_hs > 0 else 1e9

    aa_stock_cap = max(250000.0 * w_aa, 25000.0) if w_aa > 0 else 0.0
    aa_etf_cap = max(300000.0 * w_aa, 30000.0) if w_aa > 0 else 0.0
    aa_min_cash = max(150000.0 * w_aa, 15000.0) if w_aa > 0 else 1e9

    init_bench = bench_map[trade_dates[0]]
    history = []

    for i, dt in enumerate(trade_dates):
        if w_hs > 0:
            step_sub(s_hs, hs300_u, indexed_data, dt, eval_cache, hs_stock_cap, hs_etf_cap, hs_min_cash)
        if w_aa > 0:
            step_sub(s_aa, all_a_u, indexed_data, dt, eval_cache, aa_stock_cap, aa_etf_cap, aa_min_cash)

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
            "hs300_sub": s_hs["equity"], "alla_sub": s_aa["equity"],
            "portfolio_return_pct": tot_ret, "hs300_level": hs300_lvl,
            "hs300_return_pct": hs300_ret, "alpha_pct": round(tot_ret - hs300_ret, 2)
        })

    res = {
        "ratio_label": f"{int(round(w_hs*100))}% : {int(round(w_aa*100))}%",
        "w_hs300": w_hs, "w_alla": w_aa,
        "initial_capital": initial_capital, "final_value": history[-1]["portfolio_value"],
        "final_hs300_sub": s_hs["equity"], "final_alla_sub": s_aa["equity"],
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
    ratios = [round(1.0 - i * 0.1, 1) for i in range(11)]
    reb_results = []
    ind_results = []

    for w_hs in ratios:
        lbl = f"{int(round(w_hs*100))}:{int(round((1-w_hs)*100))}"
        r_reb = run_ratio_sim(w_hs, hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache, rebalance=True)
        r_ind = run_ratio_sim(w_hs, hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache, rebalance=False)
        m_r = r_reb["metrics"]
        m_i = r_ind["metrics"]
        print(f"Ratio {lbl:>7} | [再平衡] 期末:¥{m_r['final_value']/1e4:6.2f}w ({m_r['total_return_pct']:+6.2f}%) 最高:¥{m_r['peak_value']/1e4:6.2f}w 回撤:-{m_r['max_drawdown_pct']:5.2f}% 夏普:{m_r['sharpe_ratio']:4.2f} | [不调仓] 期末:¥{m_i['final_value']/1e4:6.2f}w ({m_i['total_return_pct']:+6.2f}%) 最高:¥{m_i['peak_value']/1e4:6.2f}w 回撤:-{m_i['max_drawdown_pct']:5.2f}%")
        # Strip daily history when saving summary to keep JSON compact, keep key series
        reb_results.append({"ratio": lbl, "w_hs300": w_hs, "metrics": m_r, "final_hs300_sub": r_reb["final_hs300_sub"], "final_alla_sub": r_reb["final_alla_sub"], "history": r_reb["history"]})
        ind_results.append({"ratio": lbl, "w_hs300": w_hs, "metrics": m_i, "final_hs300_sub": r_ind["final_hs300_sub"], "final_alla_sub": r_ind["final_alla_sub"]})

    out_path = Path(__file__).resolve().parent / "backtest_results_10y_ratios.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rebalanced": reb_results, "independent": ind_results}, f, ensure_ascii=False, indent=2)
