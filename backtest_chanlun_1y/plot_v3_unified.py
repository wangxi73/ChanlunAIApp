import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_v3_unified.json", "r", encoding="utf-8") as f:
    v3_data = json.load(f)

h_10y = v3_data["p_10y"]["cap_12_5w"]["history"]
h_1y = v3_data["p_1y"]["cap_12_5w"]["history"]
h_615 = v3_data["p_20260615"]["cap_12_5w"]["history"]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14.5, 10.5), gridspec_kw={"height_ratios": [1.6, 1.3, 1.2]})
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

# Panel 1: 10-Year Equity Curve (PIT Survivorship-Bias-Free)
d10 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_10y]
v10_12w = [x["portfolio_value"] / 10000.0 for x in h_10y]
v10_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_10y]
ax1.plot(d10, v10_12w, color="#10b981", linewidth=2.3, label=f"V3.0 统一三大核心指数模型 [单笔12.5万·稳健8仓] (期末 ¥{v10_12w[-1]:.2f}万 / 最高 ¥330.34万 / 最大回撤 -26.95%)")
ax1.plot(d10, v10_bench, color="#94a3b8", linewidth=1.5, linestyle="--", label=f"业绩基准：沪深300等值净值 (期末 ¥{v10_bench[-1]:.2f}万 / +21.01%)")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp V3.0「沪深300+创业板指+科创50」统一精选池模型 · 近10年真实PIT回测 (100万本金)", color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

# Panel 2: 1-Year Dynamic Sector Allocation Stacked Area Chart (2025.10 ~ 2026.10)
d1 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_1y]
tot1 = np.array([x["portfolio_value"] for x in h_1y])
w_hs = np.array([x["hs300_mkt_val"] for x in h_1y]) / tot1 * 100.0
w_cyb = np.array([x["cyb_mkt_val"] for x in h_1y]) / tot1 * 100.0
w_kcb = np.array([x["kcb_mkt_val"] for x in h_1y]) / tot1 * 100.0
w_cash = np.array([x["cash"] for x in h_1y]) / tot1 * 100.0

ax2.stackplot(
    d1, w_hs, w_cyb, w_kcb, w_cash,
    labels=["沪深300蓝筹红利持仓 (%)", "创业板指20cm持仓 (%)", "科创50硬科技持仓 (%)", "防守现金仓位 (%)"],
    colors=["#38bdf8", "#f59e0b", "#a855f7", "#334155"], alpha=0.85
)
ax2.set_ylim(0, 100)
ax2.set_ylabel("资金占比 (%)", color="#e2e8f0", fontsize=11)
ax2.set_title("V3.0 统一资金池在最近1年 (2025.10~2026.10) 的板块动态流向 (牛市自动流向双创 · 弱势自动切入沪深300与现金)", color="#f8fafc", fontsize=12, fontweight="bold", pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg2 = ax2.legend(loc="upper right", ncol=4, facecolor="#0f172a", edgecolor="#334155", fontsize=9.0)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

# Panel 3: 1-Year Equity Curve (2025.10 ~ 2026.10)
v1_12w = [x["portfolio_value"] / 10000.0 for x in h_1y]
v1_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_1y]
ax3.plot(d1, v1_12w, color="#10b981", linewidth=2.3, label=f"V3.0 统一三大核心指数模型 [单笔12.5万] (期末 ¥{v1_12w[-1]:.2f}万, +41.26% | 全年最大回撤仅 -11.06%!)")
ax3.plot(d1, v1_bench, color="#94a3b8", linewidth=1.5, linestyle="--", label=f"业绩基准：沪深300等值净值 (期末 ¥{v1_bench[-1]:.2f}万, -14.87%)")
ax3.axhline(100.0, color="#64748b", linewidth=1)
ax3.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax3.set_xlabel("交易日期", color="#e2e8f0", fontsize=11)
ax3.set_title("V3.0 最近1年 (2025.10~2026.10) 净值曲线 (最大回撤压降至 -11.06%，夏普 1.57)", color="#f8fafc", fontsize=12, fontweight="bold", pad=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg3 = ax3.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg3.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v3_unified.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
