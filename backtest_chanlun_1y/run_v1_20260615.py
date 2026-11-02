import json
import pickle
from pathlib import Path
from universe import get_universes
from run_backtest import prepare_indexed_data, simulate_account, compute_metrics
from run_backtest_5y_100w import simulate_account_100w

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    hs300_u, all_a_u = get_universes()
    cache_dir = base / "cache"
    cached_syms = {p.stem for p in cache_dir.glob("*.pkl")}
    if cached_syms:
        all_a_u = [m for m in all_a_u if m[0] in cached_syms]
    all_syms = ["sh000300"] + [m[0] for m in all_a_u]

    raw_data = {}
    for s in all_syms:
        cf = cache_dir / f"{s}.pkl"
        if cf.exists():
            with open(cf, "rb") as f:
                raw_data[s] = pickle.load(f)

    indexed = prepare_indexed_data(raw_data)
    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2026-06-15" <= b["time"] <= "2026-10-29"]
    print(f"Running V1.0 Original Model over {len(trade_dates)} trading days ({trade_dates[0]} -> {trade_dates[-1]})...")

    # 1. 100w capital (10x proportional scaling: single trade cap 25w, min cash 15w)
    r_hs300_100w = simulate_account_100w(
        "v1_quant_hs300_100w", "一代V1.0 账户①：仅限沪深300 (100万本金)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map, scale_factor=10.0
    )
    r_alla_100w = simulate_account_100w(
        "v1_quant_all_a_100w", "一代V1.0 账户②：全市场沪深A股 (100万本金)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map, scale_factor=10.0
    )

    # 2. Original 10w capital (single trade cap 2.5w, min cash 1.5w)
    r_hs300_10w = simulate_account(
        "v1_quant_hs300_10w", "一代V1.0 账户①：仅限沪深300 (原版10万本金)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map
    )
    r_alla_10w = simulate_account(
        "v1_quant_all_a_10w", "一代V1.0 账户②：全市场沪深A股 (原版10万本金)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map
    )

    for r in (r_hs300_100w, r_alla_100w, r_hs300_10w, r_alla_10w):
        m = compute_metrics(r)
        pk = max(r["history"], key=lambda x: x["portfolio_value"])
        tr = min(r["history"], key=lambda x: x["portfolio_value"])
        m["peak_value"] = pk["portfolio_value"]
        m["peak_return_pct"] = pk["portfolio_return_pct"]
        m["peak_date"] = pk["date"]
        m["trough_value"] = tr["portfolio_value"]
        m["trough_return_pct"] = tr["portfolio_return_pct"]
        m["trough_date"] = tr["date"]
        r["metrics"] = m
        print(f"=== {r['account_name']} ===")
        print(f"  期末总净值: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | Alpha: {m['alpha_pct']:+.2f}% (同期沪深300: {m['hs300_return_pct']:+.2f}%)")
        print(f"  最高净值  : ¥{m['peak_value']:,.2f} ({m['peak_return_pct']:+.2f}%, {m['peak_date']}) | 最低净值: ¥{m['trough_value']:,.2f} ({m['trough_return_pct']:+.2f}%, {m['trough_date']})")
        print(f"  最大回撤  : -{m['max_drawdown_pct']:.2f}% ({m['max_drawdown_date']}) | 夏普: {m['sharpe_ratio']:.2f} | 订单: {m['total_orders']} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f}")

    for d_target in ["2026-06-15", "2026-06-25", "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30", "2026-10-29"]:
        x1 = [pt for pt in r_hs300_100w["history"] if pt["date"] <= d_target][-1]
        x2 = [pt for pt in r_alla_100w["history"] if pt["date"] <= d_target][-1]
        print(f"{x1['date']} | 沪深300:{x1['hs300_level']:.2f}({x1['hs300_return_pct']:+.2f}%) | V1账户①(沪深300):¥{x1['portfolio_value']/1e4:6.2f}w({x1['portfolio_return_pct']:+.2f}%) | V1账户②(全A):¥{x2['portfolio_value']/1e4:6.2f}w({x2['portfolio_return_pct']:+.2f}%)")

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    h1 = r_hs300_100w["history"]
    h2 = r_alla_100w["history"]
    h1_10 = r_hs300_10w["history"]
    h2_10 = r_alla_10w["history"]

    dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h1]
    v1_100 = [x["portfolio_value"] / 10000.0 for x in h1]
    v2_100 = [x["portfolio_value"] / 10000.0 for x in h2]
    v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h1]

    def calc_dd(hist):
        peak = hist[0]["portfolio_value"]
        dds = []
        for x in hist:
            v = x["portfolio_value"]
            if v > peak:
                peak = v
            dds.append(-(peak - v) / peak * 100.0)
        return dds

    dd1 = calc_dd(h1)
    dd2 = calc_dd(h2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.0), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

    ax1.plot(dates, v1_100, label=f"一代V1.0 账户①：无AI·仅限沪深300 [100万本金] (最高 ¥116.45万 / 期末 ¥{v1_100[-1]:.2f}万, {h1[-1]['portfolio_return_pct']:+.2f}%)", color="#38bdf8", linewidth=2.3)
    ax1.plot(dates, v2_100, label=f"一代V1.0 账户②：无AI·全市场A股 [100万本金] (最高 ¥123.02万 / 期末 ¥{v2_100[-1]:.2f}万, {h2[-1]['portfolio_return_pct']:+.2f}%)", color="#f59e0b", linewidth=2.2)
    ax1.plot(dates, v_bench, label=f"业绩基准：沪深300等值净值 (期末 ¥{v_bench[-1]:.2f}万, {h1[-1]['hs300_return_pct']:+.2f}%)", color="#94a3b8", linewidth=1.5, linestyle="--")
    ax1.axhline(100.0, color="#64748b", linewidth=1)
    ax1.set_title("ChanlunAIApp 一代原版模型 (V1.0) 2026-06-15 至 2026-10-29 回测净值对比 (100万本金)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
    ax1.set_ylabel("账户总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
    leg1 = ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
    for text in leg1.get_texts():
        text.set_color("#f8fafc")

    ax2.fill_between(dates, dd1, 0, color="#38bdf8", alpha=0.25, label=f"一代账户① (沪深300) 最大回撤 (-{r_hs300_100w['metrics']['max_drawdown_pct']:.2f}%)")
    ax2.plot(dates, dd1, color="#38bdf8", linewidth=1.2)
    ax2.fill_between(dates, dd2, 0, color="#f59e0b", alpha=0.22, label=f"一代账户② (全市场A股) 最大回撤 (-{r_alla_100w['metrics']['max_drawdown_pct']:.2f}%)")
    ax2.plot(dates, dd2, color="#f59e0b", linewidth=1.2)
    ax2.set_ylabel("动态回撤 (%)", color="#e2e8f0", fontsize=11)
    ax2.set_xlabel("日期 (2026年6月15日 ~ 10月29日)", color="#e2e8f0", fontsize=11)
    leg2 = ax2.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
    for text in leg2.get_texts():
        text.set_color("#f8fafc")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_img = base / "chanlun_quant_backtest_v1_20260615.png"
    plt.savefig(out_img, dpi=180)
    print("Saved chart to", out_img)
