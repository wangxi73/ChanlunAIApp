import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from universe import get_universes
from run_backtest import prepare_indexed_data
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR
from run_backtest_10y_ratios import run_ratio_sim
from run_backtest_v2_regime import simulate_v2_portfolio
import market_regime_v2

EVAL_CACHE_FILE = Path(__file__).resolve().parent / "cache_10y_eval.pkl"
BENCH_CACHE_FILE = Path(__file__).resolve().parent / "cache_10y_bench_feats.pkl"

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    hs300_u, all_a_u = get_universes()
    cached_syms = {p.stem for p in (base / "cache").glob("*.pkl")}
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

    eval_cache = pickle.load(open(EVAL_CACHE_FILE, "rb")) if EVAL_CACHE_FILE.exists() else {}
    if BENCH_CACHE_FILE.exists():
        market_regime_v2._BENCH_CACHE.update(pickle.load(open(BENCH_CACHE_FILE, "rb")))

    print("1. Running V1 Original (40% HS300 + 60% All-A, Annual Rebalance)...")
    res_v1 = run_ratio_sim(0.4, hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache, rebalance=True)

    print("2. Running V2-Dual (Dual-Factor Regime Control, 40% HS300 + 60% All-A)...")
    res_v2_dual = simulate_v2_portfolio(
        "V2新算法(双因子牛熊控仓)：40%沪深300 + 60%全A", 0.4,
        hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache,
        initial_capital=1000000.0, rebalance=True, regime_mode="dual_factor", use_ratchet_stop=False
    )

    print("3. Running V2-Index (Index-Only Regime Control, 40% HS300 + 60% All-A)...")
    res_v2_idx = simulate_v2_portfolio(
        "V2新算法(纯大盘缠论控仓)：40%沪深300 + 60%全A", 0.4,
        hs300_u, all_a_u, indexed, trade_dates, bench_map, eval_cache,
        initial_capital=1000000.0, rebalance=True, regime_mode="index_only", use_ratchet_stop=False
    )

    with open(base / "backtest_results_v2_10y_40_60.json", "w", encoding="utf-8") as f:
        json.dump({"v1_original": res_v1, "v2_dual": res_v2_dual, "v2_index": res_v2_idx}, f, ensure_ascii=False, indent=2)

    for label, r in [("V1原版 (无环境控仓)", res_v1), ("V2双因子牛熊控仓", res_v2_dual), ("V2纯大盘缠论控仓", res_v2_idx)]:
        m = r["metrics"]
        print(f"{label:20s} | 期末:¥{m['final_value']/1e4:6.2f}w ({m['total_return_pct']:+6.2f}%) | 最高:¥{m['peak_value']/1e4:6.2f}w ({m['peak_date']}) | 最低:¥{m['trough_value']/1e4:6.2f}w ({m['trough_date']}) | 最大回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:4.2f} | 胜率:{m['win_rate_pct']:5.2f}% | 盈亏比:{m['profit_loss_ratio']:4.2f}")
