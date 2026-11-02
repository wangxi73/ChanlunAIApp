import bisect
import json
import pickle
from pathlib import Path
from pit_universe import get_candidate_pools, select_pit_universe_at_date
from data_loader import load_all_data_2y
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_v2_regime import step_subaccount_v2
from run_backtest_10y_pit import precompute_pit_rosters, with_held_metas
import market_regime_v2

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    hs_pool, cyb_pool, kcb_pool = get_candidate_pools()
    all_syms = list({m[0] for m in (hs_pool + cyb_pool + kcb_pool)} | {"sh000300"})

    print(f"Loading 15m/60m/daily K-line data for {len(all_syms)} symbols...")
    raw_data = load_all_data_2y(all_syms, max_workers=12)
    indexed = prepare_indexed_data(raw_data)

    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2025-10-29" <= b["time"] <= "2026-10-29"]
    print(f"Running 1Y Backtest ({trade_dates[0]} -> {trade_dates[-1]}, {len(trade_dates)} days) with 1,000,000 RMB [20% HS300 : 50% CYB : 30% KCB]...")

    rosters_by_date, full_meta_map = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, indexed, trade_dates)

    def run_1y(use_v2_regime=True):
        market_regime_v2._BENCH_CACHE.clear()
        if not use_v2_regime:
            for k in market_regime_v2.REGIME_CONFIGS:
                market_regime_v2.REGIME_CONFIGS[k].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
        else:
            market_regime_v2.REGIME_CONFIGS["BULL"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
            market_regime_v2.REGIME_CONFIGS["RANGE"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
            market_regime_v2.REGIME_CONFIGS["WEAK"].update({"max_exposure_pct": 0.70, "daily_buy_limit": 1, "min_score": 60, "single_pos_scale": 0.60})
            market_regime_v2.REGIME_CONFIGS["CRASH"].update({"max_exposure_pct": 0.0, "daily_buy_limit": 0, "min_score": 999, "single_pos_scale": 0.0})

        initial_capital = 1000000.0
        w_hs, w_cyb, w_kcb = 0.20, 0.50, 0.30
        scale = 10.0

        s_hs = {"cash": 200000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 200000.0, "regime_counts": {}}
        s_cyb = {"cash": 500000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 500000.0, "regime_counts": {}}
        s_kcb = {"cash": 300000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 300000.0, "regime_counts": {}}

        init_bench = bench_map[trade_dates[0]]
        bench_idx = indexed["sh000300"]
        eval_cache = {}
        history = []

        for dt in trade_dates:
            b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
            bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]
            hs_u, cyb_u, kcb_u = rosters_by_date[dt]

            r_meta = step_subaccount_v2(
                s_hs, with_held_metas(hs_u, s_hs["holdings"], full_meta_map), indexed, bench_day_slice, dt, eval_cache,
                base_stock_cap=25000.0 * scale * w_hs, base_etf_cap=30000.0 * scale * w_hs, base_min_cash=15000.0 * scale * w_hs,
                regime_mode="dual_factor", use_ratchet_stop=use_v2_regime
            )
            step_subaccount_v2(
                s_cyb, with_held_metas(cyb_u, s_cyb["holdings"], full_meta_map), indexed, bench_day_slice, dt, eval_cache,
                base_stock_cap=25000.0 * scale * w_cyb, base_etf_cap=30000.0 * scale * w_cyb, base_min_cash=15000.0 * scale * w_cyb,
                regime_mode="dual_factor", use_ratchet_stop=use_v2_regime
            )
            step_subaccount_v2(
                s_kcb, with_held_metas(kcb_u, s_kcb["holdings"], full_meta_map), indexed, bench_day_slice, dt, eval_cache,
                base_stock_cap=25000.0 * scale * w_kcb, base_etf_cap=30000.0 * scale * w_kcb, base_min_cash=15000.0 * scale * w_kcb,
                regime_mode="dual_factor", use_ratchet_stop=use_v2_regime
            )

            tot_val = round(s_hs["equity"] + s_cyb["equity"] + s_kcb["equity"], 2)
            tot_ret = round((tot_val - initial_capital) / initial_capital * 100.0, 2)
            hs300_lvl = bench_map[dt]
            hs300_ret = round((hs300_lvl - init_bench) / init_bench * 100.0, 2)
            history.append({
                "date": dt, "portfolio_value": tot_val,
                "hs300_sub": s_hs["equity"], "cyb_sub": s_cyb["equity"], "kcb_sub": s_kcb["equity"],
                "cash": round(s_hs["cash"] + s_cyb["cash"] + s_kcb["cash"], 2),
                "holdings_count": len(s_hs["holdings"]) + len(s_cyb["holdings"]) + len(s_kcb["holdings"]),
                "regime": r_meta["regime"],
                "portfolio_return_pct": tot_ret, "hs300_level": hs300_lvl,
                "hs300_return_pct": hs300_ret, "alpha_pct": round(tot_ret - hs300_ret, 2)
            })

        res = {
            "initial_capital": initial_capital, "final_value": history[-1]["portfolio_value"],
            "sub_accounts": {
                "hs300": {"initial": 200000.0, "final": s_hs["equity"], "return_pct": round((s_hs["equity"]-200000.0)/200000.0*100, 2), "closed_trades": s_hs["closed_trades"], "holdings": s_hs["holdings"]},
                "cyb": {"initial": 500000.0, "final": s_cyb["equity"], "return_pct": round((s_cyb["equity"]-500000.0)/500000.0*100, 2), "closed_trades": s_cyb["closed_trades"], "holdings": s_cyb["holdings"]},
                "kcb": {"initial": 300000.0, "final": s_kcb["equity"], "return_pct": round((s_kcb["equity"]-300000.0)/300000.0*100, 2), "closed_trades": s_kcb["closed_trades"], "holdings": s_kcb["holdings"]},
            },
            "regime_counts": s_cyb["regime_counts"],
            "history": history,
            "trades": s_hs["trades"] + s_cyb["trades"] + s_kcb["trades"],
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

    res_v2 = run_1y(use_v2_regime=True)
    res_v1 = run_1y(use_v2_regime=False)

    out_path = base / "backtest_results_v2_1y_20_50_30.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"v2_latest": res_v2, "v1_no_regime": res_v1}, f, ensure_ascii=False, indent=2)

    for label, r in [("最新模型 V2 (20:50:30 + 牛熊控仓 + 当年真实池)", res_v2), ("对照 V1 (20:50:30 无牛熊控仓)", res_v1)]:
        m = r["metrics"]
        subs = r["sub_accounts"]
        print(f"=== {label} ===")
        print(f"  期末总净值: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | Alpha: {m['alpha_pct']:+.2f}% (沪深300: {m['hs300_return_pct']:+.2f}%)")
        print(f"  最高净值  : ¥{m['peak_value']:,.2f} ({m['peak_return_pct']:+.2f}%, {m['peak_date']}) | 最低净值: ¥{m['trough_value']:,.2f} ({m['trough_return_pct']:+.2f}%, {m['trough_date']})")
        print(f"  最大回撤  : -{m['max_drawdown_pct']:.2f}% ({m['max_drawdown_date']}) | 夏普比率: {m['sharpe_ratio']:.2f} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f}")
        print(f"  子账户拆解: 沪深300(20w)->¥{subs['hs300']['final']:,.2f}({subs['hs300']['return_pct']:+.2f}%) | 创业板(50w)->¥{subs['cyb']['final']:,.2f}({subs['cyb']['return_pct']:+.2f}%) | 科创板(30w)->¥{subs['kcb']['final']:,.2f}({subs['kcb']['return_pct']:+.2f}%)")
        print(f"  环境分布  : {r['regime_counts']}")
