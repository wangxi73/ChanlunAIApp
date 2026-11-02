import bisect
import json
import pickle
from pathlib import Path
from sector_universes import get_three_sector_universes
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR
from run_backtest_v2_regime import step_subaccount_v2
import market_regime_v2

EVAL_CACHE_FILE = Path(__file__).resolve().parent / "cache_10y_eval.pkl"
BENCH_CACHE_FILE = Path(__file__).resolve().parent / "cache_10y_bench_feats.pkl"
KCB_ACTIVE_DATE = "2019-08-26"  # 科创板首批25家上市满1个月具备完整缠论笔/中枢结构

def simulate_3sector_portfolio(w_hs, w_cyb, w_kcb, hs300_u, cyb_u, kcb_u, indexed_data, trade_dates, bench_map, eval_cache,
                               initial_capital=1000000.0, rebalance=True, use_v2_regime=True):
    scale = initial_capital / 100000.0
    # 2016-10-28 起步时无科创板，科创板配比 w_kcb 自动由创业板替代 (eff_cyb = w_cyb + w_kcb)
    init_cyb_w = round(w_cyb + w_kcb, 4)
    s_hs = {"cash": initial_capital * w_hs, "holdings": {}, "trades": [], "closed_trades": [], "equity": initial_capital * w_hs, "regime_counts": {}}
    s_cyb = {"cash": initial_capital * init_cyb_w, "holdings": {}, "trades": [], "closed_trades": [], "equity": initial_capital * init_cyb_w, "regime_counts": {}}
    s_kcb = {"cash": 0.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 0.0, "regime_counts": {}}

    kcb_seeded = False
    kcb_target_transfer = 0.0
    init_bench = bench_map[trade_dates[0]]
    bench_idx = indexed_data["sh000300"]
    history = []

    if not use_v2_regime:
        for k in market_regime_v2.REGIME_CONFIGS:
            market_regime_v2.REGIME_CONFIGS[k].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    else:
        market_regime_v2.REGIME_CONFIGS["BULL"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
        market_regime_v2.REGIME_CONFIGS["RANGE"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
        market_regime_v2.REGIME_CONFIGS["WEAK"].update({"max_exposure_pct": 0.70, "daily_buy_limit": 1, "min_score": 60, "single_pos_scale": 0.60})
        market_regime_v2.REGIME_CONFIGS["CRASH"].update({"max_exposure_pct": 0.0, "daily_buy_limit": 0, "min_score": 999, "single_pos_scale": 0.0})

    for i, dt in enumerate(trade_dates):
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]
        kcb_active = (dt >= KCB_ACTIVE_DATE) and (w_kcb > 0)

        # 当科创板正式启用后，从替代它的创业板子账户中按 w_kcb/(w_cyb+w_kcb) 拆分资金至科创板子账户
        if kcb_active and init_cyb_w > 0:
            if not kcb_seeded:
                comb_eq = s_cyb["equity"] + s_kcb["equity"]
                kcb_target_transfer = comb_eq * (w_kcb / init_cyb_w)
                kcb_seeded = True
            if kcb_target_transfer > 100.0 and s_cyb["cash"] > 100.0:
                xfer = min(kcb_target_transfer, s_cyb["cash"])
                s_cyb["cash"] -= xfer
                s_kcb["cash"] += xfer
                kcb_target_transfer -= xfer
                s_cyb["equity"] = round(s_cyb["cash"] + sum(h["shares"] * h["last_price"] for h in s_cyb["holdings"].values()), 2)
                s_kcb["equity"] = round(s_kcb["cash"] + sum(h["shares"] * h["last_price"] for h in s_kcb["holdings"].values()), 2)

        cur_w_cyb = w_cyb if kcb_active else init_cyb_w
        cur_w_kcb = w_kcb if kcb_active else 0.0

        if w_hs > 0:
            step_subaccount_v2(
                s_hs, hs300_u, indexed_data, bench_day_slice, dt, eval_cache,
                base_stock_cap=max(25000.0 * scale * w_hs, 25000.0),
                base_etf_cap=max(30000.0 * scale * w_hs, 30000.0),
                base_min_cash=max(15000.0 * scale * w_hs, 15000.0),
                regime_mode="dual_factor", use_ratchet_stop=False
            )
        if cur_w_cyb > 0 or s_cyb["holdings"]:
            step_subaccount_v2(
                s_cyb, cyb_u, indexed_data, bench_day_slice, dt, eval_cache,
                base_stock_cap=max(25000.0 * scale * cur_w_cyb, 25000.0) if cur_w_cyb > 0 else 0.0,
                base_etf_cap=max(30000.0 * scale * cur_w_cyb, 30000.0) if cur_w_cyb > 0 else 0.0,
                base_min_cash=max(15000.0 * scale * cur_w_cyb, 15000.0) if cur_w_cyb > 0 else 1e9,
                regime_mode="dual_factor", use_ratchet_stop=False
            )
        if cur_w_kcb > 0:
            step_subaccount_v2(
                s_kcb, kcb_u, indexed_data, bench_day_slice, dt, eval_cache,
                base_stock_cap=max(25000.0 * scale * cur_w_kcb, 25000.0),
                base_etf_cap=max(30000.0 * scale * cur_w_kcb, 30000.0),
                base_min_cash=max(15000.0 * scale * cur_w_kcb, 15000.0),
                regime_mode="dual_factor", use_ratchet_stop=False
            )

        # 每年 10 月底三账户动态再平衡 (科创板启用前按 w_hs : w_cyb+w_kcb : 0 再平衡；启用后按 w_hs : w_cyb : w_kcb 再平衡)
        if rebalance and i + 1 < len(trade_dates):
            cur_ym = dt[:7]
            nxt_ym = trade_dates[i + 1][:7]
            if cur_ym.endswith("-10") and cur_ym != nxt_ym:
                tot_eq = s_hs["equity"] + s_cyb["equity"] + s_kcb["equity"]
                t_hs = tot_eq * w_hs
                t_cyb = tot_eq * cur_w_cyb
                t_kcb = tot_eq * cur_w_kcb
                kcb_target_transfer = 0.0

                # Pool available cash for rebalancing transfers
                subs = [(s_hs, t_hs), (s_cyb, t_cyb), (s_kcb, t_kcb)]
                surplus_pool = 0.0
                deficits = []
                for sub, tgt in subs:
                    diff = sub["equity"] - tgt
                    if diff > 0:
                        give = min(diff, sub["cash"])
                        sub["cash"] -= give
                        surplus_pool += give
                    elif diff < 0:
                        deficits.append((sub, -diff))
                tot_def = sum(d for _, d in deficits)
                if tot_def > 0 and surplus_pool > 0:
                    for sub, d in deficits:
                        sub["cash"] += surplus_pool * (d / tot_def)
                for sub, _ in subs:
                    sub["equity"] = round(sub["cash"] + sum(h["shares"] * h["last_price"] for h in sub["holdings"].values()), 2)

        tot_val = round(s_hs["equity"] + s_cyb["equity"] + s_kcb["equity"], 2)
        tot_ret = round((tot_val - initial_capital) / initial_capital * 100.0, 2)
        hs300_lvl = bench_map[dt]
        hs300_ret = round((hs300_lvl - init_bench) / init_bench * 100.0, 2)
        history.append({
            "date": dt, "portfolio_value": tot_val,
            "hs300_sub": s_hs["equity"], "cyb_sub": s_cyb["equity"], "kcb_sub": s_kcb["equity"],
            "portfolio_return_pct": tot_ret, "hs300_level": hs300_lvl,
            "hs300_return_pct": hs300_ret, "alpha_pct": round(tot_ret - hs300_ret, 2)
        })

    lbl = f"{int(round(w_hs*100))}:{int(round(w_cyb*100))}:{int(round(w_kcb*100))}"
    res = {
        "ratio": lbl, "w_hs300": w_hs, "w_cyb": w_cyb, "w_kcb": w_kcb,
        "initial_capital": initial_capital, "final_value": history[-1]["portfolio_value"],
        "final_hs300_sub": s_hs["equity"], "final_cyb_sub": s_cyb["equity"], "final_kcb_sub": s_kcb["equity"],
        "history": history, "trades": s_hs["trades"] + s_cyb["trades"] + s_kcb["trades"],
        "closed_trades": s_hs["closed_trades"] + s_cyb["closed_trades"] + s_kcb["closed_trades"]
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
    base = Path(__file__).resolve().parent
    hs300_u, cyb_u, kcb_u = get_three_sector_universes()
    all_syms = list({m[0] for m in (hs300_u + cyb_u + kcb_u)} | {"sh000300"})

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

    eval_cache = pickle.load(open(EVAL_CACHE_FILE, "rb")) if EVAL_CACHE_FILE.exists() else {}
    if BENCH_CACHE_FILE.exists():
        market_regime_v2._BENCH_CACHE.update(pickle.load(open(BENCH_CACHE_FILE, "rb")))

    ratios_3sector = [
        (1.0, 0.0, 0.0),
        (0.8, 0.1, 0.1),
        (0.7, 0.2, 0.1),
        (0.6, 0.25, 0.15),
        (0.6, 0.2, 0.2),
        (0.5, 0.3, 0.2),
        (0.5, 0.25, 0.25),
        (0.4, 0.4, 0.2),
        (0.4, 0.3, 0.3),
        (0.4, 0.2, 0.4),
        (0.3, 0.4, 0.3),
        (0.3, 0.35, 0.35),
        (0.2, 0.5, 0.3),
        (0.2, 0.4, 0.4),
        (0.0, 0.6, 0.4),
        (0.0, 1.0, 0.0),
    ]

    v2_results = []
    v1_results = []

    print("Scanning 16 Three-Sector (HS300 : ChiNext : STAR) allocations over 10 years...")
    for w_hs, w_cyb, w_kcb in ratios_3sector:
        r_v2 = simulate_3sector_portfolio(w_hs, w_cyb, w_kcb, hs300_u, cyb_u, kcb_u, indexed, trade_dates, bench_map, eval_cache, rebalance=True, use_v2_regime=True)
        r_v1 = simulate_3sector_portfolio(w_hs, w_cyb, w_kcb, hs300_u, cyb_u, kcb_u, indexed, trade_dates, bench_map, eval_cache, rebalance=True, use_v2_regime=False)
        m2 = r_v2["metrics"]
        m1 = r_v1["metrics"]
        lbl = r_v2["ratio"]
        print(f"配比 {lbl:>9s} | [V2牛熊控仓] 期末:¥{m2['final_value']/1e4:6.2f}w ({m2['total_return_pct']:+6.2f}%) 最高:¥{m2['peak_value']/1e4:6.2f}w 最低:¥{m2['trough_value']/1e4:5.2f}w 回撤:-{m2['max_drawdown_pct']:5.2f}% 夏普:{m2['sharpe_ratio']:4.2f} | [V1原版] 期末:¥{m1['final_value']/1e4:6.2f}w 回撤:-{m1['max_drawdown_pct']:5.2f}%")
        v2_results.append({"ratio": lbl, "w_hs300": w_hs, "w_cyb": w_cyb, "w_kcb": w_kcb, "metrics": m2, "final_subs": [r_v2["final_hs300_sub"], r_v2["final_cyb_sub"], r_v2["final_kcb_sub"]], "history": r_v2["history"]})
        v1_results.append({"ratio": lbl, "w_hs300": w_hs, "w_cyb": w_cyb, "w_kcb": w_kcb, "metrics": m1, "final_subs": [r_v1["final_hs300_sub"], r_v1["final_cyb_sub"], r_v1["final_kcb_sub"]]})

    with open(EVAL_CACHE_FILE, "wb") as f:
        pickle.dump(eval_cache, f)

    with open(base / "backtest_results_10y_3sectors.json", "w", encoding="utf-8") as f:
        json.dump({"v2_regime": v2_results, "v1_original": v1_results}, f, ensure_ascii=False, indent=2)
