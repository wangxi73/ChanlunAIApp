import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_v3_2_mainline.json", "r", encoding="utf-8") as f:
    d32 = json.load(f)
with open(base / "backtest_results_v3_unified.json", "r", encoding="utf-8") as f:
    d30 = json.load(f)

h32_10y_25 = d32["p_10y"]["cap_25w"]["history"]
h32_10y_16 = d32["p_10y"]["cap_16w"]["history"]
h30_10y_12 = d30["p_10y"]["cap_12_5w"]["history"]
h32_1y_25 = d32["p_1y"]["cap_25w"]["history"]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14.5, 10.5), gridspec_kw={"height_ratios": [1.6, 1.3, 1.2]})
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

d10 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h32_10y_25]
v32_25 = [x["portfolio_value"] / 10000.0 for x in h32_10y_25]
v32_16 = [x["portfolio_value"] / 10000.0 for x in h32_10y_16]
v30_12 = [x["portfolio_value"] / 10000.0 for x in h30_10y_12]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h32_10y_25]

ax1.plot(d10, v32_25, color="#f59e0b", linewidth=2.4, label=f"V3.2 实验版 [单笔25万·4仓] (期末 ¥{v32_25[-1]:.2f}万, +506.82% | 最低谷底 ¥94.18万 / 夏普 0.86)")
ax1.plot(d10, v32_16, color="#10b981", linewidth=2.1, label=f"V3.2 实验版 [单笔16万·6仓] (期末 ¥{v32_16[-1]:.2f}万, +302.91% | 最高 ¥436.25万 / 夏普 0.71)")
ax1.plot(d10, v30_12, color="#38bdf8", linewidth=1.8, linestyle="-.", label=f"V3.0 稳健版 [单笔12.5万·8仓] (期末 ¥{v30_12[-1]:.2f}万, +210.42% | 最大回撤 -26.95%)")
ax1.plot(d10, v_bench, color="#94a3b8", linewidth=1.4, linestyle="--", label="沪深300基准等值净值 (期末 ¥121.01万, +21.01%)")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp V3.2 实验版 (牛市动态主线识别 + 熊市分级红利避险) 近10年真实PIT+严苛涨跌停回测", color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

# Panel 2: 10-Year Dynamic Sector Allocation of V3.2
tot10 = np.array([x["portfolio_value"] for x in h32_10y_25])
w_hs = np.array([x["hs300_mkt_val"] for x in h32_10y_25]) / tot10 * 100.0
w_cyb = np.array([x["cyb_mkt_val"] for x in h32_10y_25]) / tot10 * 100.0
w_kcb = np.array([x["kcb_mkt_val"] for x in h32_10y_25]) / tot10 * 100.0
w_cash = np.array([x["cash"] for x in h32_10y_25]) / tot10 * 100.0

ax2.stackplot(
    d10, w_hs, w_cyb, w_kcb, w_cash,
    labels=["沪深300(含T1/T2红利盾牌) (%)", "创业板主线持仓 (%)", "科创板主线持仓 (%)", "防守现金 (%)"],
    colors=["#38bdf8", "#f59e0b", "#a855f7", "#334155"], alpha=0.85
)
ax2.set_ylim(0, 100)
ax2.set_ylabel("资金占比 (%)", color="#e2e8f0", fontsize=11)
ax2.set_title("V3.2 近10年动态主线切换与熊市分级避险仓位分布 (牛市自动识别主线 · 深度熊市压仓55%并锚定T1红利盾牌)", color="#f8fafc", fontsize=12, fontweight="bold", pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
leg2 = ax2.legend(loc="upper right", ncol=4, facecolor="#0f172a", edgecolor="#334155", fontsize=9.0)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

# Panel 3: 1-Year Equity Curve (2025.10 ~ 2026.10)
d1 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h32_1y_25]
v1_25 = [x["portfolio_value"] / 10000.0 for x in h32_1y_25]
v1_b = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h32_1y_25]
ax3.plot(d1, v1_25, color="#f59e0b", linewidth=2.3, label=f"V3.2 实验版 [单笔25万] 最近1年净值 (期末 ¥{v1_25[-1]:.2f}万, +39.41% | 相比V3.1同口径提升 +35.51万!)")
ax3.plot(d1, v1_b, color="#94a3b8", linewidth=1.4, linestyle="--", label="沪深300基准等值净值 (-14.87%)")
ax3.axhline(100.0, color="#64748b", linewidth=1)
ax3.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax3.set_xlabel("交易日期", color="#e2e8f0", fontsize=11)
ax3.set_title("V3.2 最近1年 (2025.10~2026.10) 净值曲线 (三大板块全面正贡献：沪深300 +4.88万 / 创业板 +21.80万 / 科创板 +12.32万)", color="#f8fafc", fontsize=11.8, fontweight="bold", pad=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg3 = ax3.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg3.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v3_2_mainline.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
