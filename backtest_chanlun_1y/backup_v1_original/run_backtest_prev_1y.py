import json
from pathlib import Path
from universe import get_universes
from data_loader import load_all_data_2y
from run_backtest import prepare_indexed_data, simulate_account, compute_metrics

if __name__ == "__main__":
    hs300_u, all_a_u = get_universes()
    # Read exact symbols from previous cache so the universe is identical
    cache_dir = Path(__file__).resolve().parent / "cache"
    cached_syms = {p.stem for p in cache_dir.glob("*.pkl")}
    if cached_syms:
        all_a_u = [m for m in all_a_u if m[0] in cached_syms]
    all_syms = ["sh000300"] + [m[0] for m in all_a_u]
    print(f"Fetching 2Y K-line data for {len(all_syms)} symbols...")
    raw_data = load_all_data_2y(all_syms, max_workers=12)
    indexed = prepare_indexed_data(raw_data)

    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2024-10-29" <= b["time"] <= "2025-10-29"]
    print(f"Running backtest over {len(trade_dates)} trading days ({trade_dates[0]} -> {trade_dates[-1]})...")

    res_hs300 = simulate_account(
        "quant_hs300_prev1y", "账户①：无AI介入 · 仅限沪深300 (¥100,000)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map
    )
    res_all_a = simulate_account(
        "quant_all_a_prev1y", "账户②：无AI介入 · 全市场沪深A股 (¥100,000)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map
    )

    m_hs300 = compute_metrics(res_hs300)
    m_all_a = compute_metrics(res_all_a)
    res_hs300["metrics"] = m_hs300
    res_all_a["metrics"] = m_all_a

    out_path = Path(__file__).resolve().parent / "backtest_results_prev1y.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"quant_hs300": res_hs300, "quant_all_a": res_all_a}, f, ensure_ascii=False, indent=2)

    print("Prev-1Y HS300 Metrics:", json.dumps(m_hs300, ensure_ascii=False))
    print("Prev-1Y All-A Metrics:", json.dumps(m_all_a, ensure_ascii=False))
