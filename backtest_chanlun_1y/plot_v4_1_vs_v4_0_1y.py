import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from run_v4_0_1y import load_v4_1y_duckdb
from ablation_v40_to_v41 import run_configurable_v4

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_v4_0_1y.json", "r", encoding="utf-8") as f:
    d40 = json.load(f)
with open(base / "backtest_v4_1_1y.json", "r", encoding="utf-8") as f:
    d41 = json.load(f)

indexed, rosters_dual, fmap, trade_dates = load_v4_1y_duckdb()
ec = {}

# Run key V4.1 progressive builds so we have their full daily equity curves for the comparison chart
from run_backtest import compute_metrics
def get_curve(lbl, flg):
    bench_idx = indexed["sh000300"]
    bmap = {b["time"]: b["close"] for b in bench_idx["day"]}
    init_b = bmap[trade_dates[0]]
    # modify run_configurable_v4 to return history too
    m = run_configurable_v4(lbl, rosters_dual, fmap, indexed, trade_dates, flg, ec)
    return m

m_v40 = d40["V4.0-A (零人工先验 · 15%净值动态复利)"]["metrics"]
h_v40 = d40["V4.0-A (零人工先验 · 15%净值动态复利)"]["history"]
h_v41_full = d41["V4.1-Dual (双通道主板+月度缓冲双创 · 12%复利)"]["history"]
m_v41_full = d41["V4.1-Dual (双通道主板+月度缓冲双创 · 12%复利)"]["metrics"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_v40]
v_40 = [x["portfolio_value"] / 10000.0 for x in h_v40]
v_41_full = [x["portfolio_value"] / 10000.0 for x in h_v41_full]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_v40]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9.0), gridspec_kw={"height_ratios": [1.6, 1.3]})
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, v_40, color="#38bdf8", linewidth=2.2, linestyle="--", label=f"V4.0 基线 [15:00全天K线信号 · 无分批锁利] (期末 ¥{v_40[-1]:.2f}万, +10.16% | 胜率 48.2% | 盈亏比 1.24)")
ax1.plot(dates, v_41_full, color="#10b981", linewidth=2.3, label=f"V4.1 全模块严苛版 [14:45物理隔离+Z-Score+0.25%限额4bps滑点] (期末 ¥{v_41_full[-1]:.2f}万, -13.70% | 盈亏比提升至 1.32~1.86)")
ax1.plot(dates, v_bench, color="#94a3b8", linewidth=1.6, linestyle=":", label=f"业绩基准：沪深300指数等值净值 (期末 ¥{v_bench[-1]:.2f}万, -14.18%)")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp V4.1 vs V4.0 最近1年 (2025-10-30 ~ 2026-10-29) 净值走势对比", color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

# Subplot 2: Bar chart of the 6-module ablation bridge between V4.0 and V4.1
labels = [
    "V4.0 基线\n(Dual 15%)",
    "V4.0 +\nM5(60m减半锁利)",
    "V4.0 +\nM4(月度缓冲池)",
    "V4.0 +\nM6(0.25%限额+滑点)",
    "V4.1 严苛隔离\nM1+M4+M5+M6",
    "V4.1 全模块\n(含M2+M3纯Z-Score)",
    "沪深300\n基准"
]
ret_vals = [10.16, 14.76, 10.84, 8.15, 3.87, -11.38, -14.18]
pl_vals = [1.24, 1.39, 1.31, 1.24, 1.45, 1.32, 0.0]
wr_vals = [48.21, 53.08, 53.24, 47.81, 50.51, 47.91, 0.0]

x = np.arange(len(labels))
colors = ["#38bdf8", "#f59e0b", "#a855f7", "#22d3ee", "#10b981", "#6366f1", "#64748b"]
bars = ax2.bar(x, ret_vals, width=0.52, color=colors, alpha=0.88, edgecolor="#f8fafc", linewidth=0.8)
ax2.axhline(0.0, color="#cbd5e1", linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, color="#f8fafc", fontsize=9.5)
ax2.set_ylabel("1年累计收益率 (%)", color="#e2e8f0", fontsize=11)
ax2.set_title("V4.0 -> V4.1 六大升级模块单因子消融归因 (展示各模块对收益率、胜率与盈亏比的真实贡献)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
ax2.set_ylim(-22, 23)

for i, b in enumerate(bars):
    val = ret_vals[i]
    y_pos = val + 1.0 if val >= 0 else val - 2.8
    txt = f"{val:+.2f}%"
    if pl_vals[i] > 0:
        txt += f"\n(胜{wr_vals[i]:.0f}%/盈亏比{pl_vals[i]:.2f})"
    ax2.text(b.get_x() + b.get_width() / 2.0, y_pos, txt, ha="center", va="bottom" if val >= 0 else "top", color="#f8fafc", fontsize=8.5, fontweight="bold")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v4_1_vs_v4_0_1y.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
