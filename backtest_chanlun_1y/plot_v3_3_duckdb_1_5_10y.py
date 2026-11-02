import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_v3_3_duckdb_1_5_10y.json", "r", encoding="utf-8") as f:
    data = json.load(f)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14.5, 11))
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

specs = [
    ("V3.3-B (15%动态扩容 + 60m脉冲过滤)", "#f59e0b", 2.3),
    ("V3.3-A (15%动态扩容 + 日线透支过滤)", "#10b981", 2.1),
    ("V3.3-C (12%动态扩容 + 日线透支过滤·均衡)", "#38bdf8", 1.9),
]

for ax, h_key, title_str, loc_interval in [
    (ax1, "1Y", "① 1年期回测 (2025-10-30 至 2026-10-29 · 242交易日)", "month"),
    (ax2, "5Y", "② 5年期回测 (2021-10-29 至 2026-10-29 · 1212交易日)", "6month"),
    (ax3, "10Y", "③ 10年期回测 (2016-10-28 至 2026-10-29 · 2429交易日)", "year"),
]:
    h_dict = data[h_key]
    ref_hist = h_dict[specs[0][0]]["history"]
    dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in ref_hist]
    bench_vals = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in ref_hist]

    for v_name, col, lw in specs:
        obj = h_dict[v_name]
        m = obj["metrics"]
        vals = [x["portfolio_value"] / 10000.0 for x in obj["history"]]
        ax.plot(
            dates, vals, color=col, linewidth=lw,
            label=f"{v_name} (期末 ¥{m['final_value']/1e4:.2f}万, {m['total_return_pct']:+.2f}% | 最高 ¥{m['peak_value']/1e4:.2f}万 | 回撤 -{m['max_drawdown_pct']:.2f}% | 夏普 {m['sharpe_ratio']:.2f})"
        )
    ax.plot(dates, bench_vals, color="#94a3b8", linewidth=1.4, linestyle="--", label=f"沪深300基准等值净值 (期末 ¥{bench_vals[-1]:.2f}万, {ref_hist[-1]['hs300_return_pct']:+.2f}%)")
    ax.axhline(100.0, color="#64748b", linewidth=1)
    ax.set_title(f"ChanlunAIApp V3.3 模型 [DuckDB unified_ratio_qfq] — {title_str}", color="#f8fafc", fontsize=12.5, fontweight="bold", pad=8)
    ax.set_ylabel("净值 (万元)", color="#e2e8f0", fontsize=10.5)
    leg = ax.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.2)
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    if loc_interval == "month":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    elif loc_interval == "6month":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))

ax3.set_xlabel("交易日期", color="#e2e8f0", fontsize=11)
plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v3_3_duckdb_1_5_10y.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
