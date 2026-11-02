import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_v3_3_ultimate.json", "r", encoding="utf-8") as f:
    d33 = json.load(f)
with open(base / "backtest_results_v3_2_mainline.json", "r", encoding="utf-8") as f:
    d32 = json.load(f)
with open(base / "backtest_results_v3_unified.json", "r", encoding="utf-8") as f:
    d30 = json.load(f)

h33_15 = d33["p_10y"]["dyn_15pct"]["history"]
h33_12 = d33["p_10y"]["dyn_12pct"]["history"]
h32_25 = d32["p_10y"]["cap_25w"]["history"]
h30_12 = d30["p_10y"]["cap_12_5w"]["history"]
h33_1y = d33["p_1y"]["dyn_15pct"]["history"]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14.5, 10.5), gridspec_kw={"height_ratios": [1.7, 1.15, 1.2]})
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

d10 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h33_15]
v33_15 = [x["portfolio_value"] / 10000.0 for x in h33_15]
v33_12 = [x["portfolio_value"] / 10000.0 for x in h33_12]
v32_25 = [x["portfolio_value"] / 10000.0 for x in h32_25]
v30_12 = [x["portfolio_value"] / 10000.0 for x in h30_12]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h33_15]

ax1.plot(d10, v33_15, color="#f59e0b", linewidth=2.4, label=f"V3.3 终极版 [15%动态复利扩容 + 60m透支过滤] (最高 ¥1,039.47万 / 期末 ¥{v33_15[-1]:.2f}万, +700.24%)")
ax1.plot(d10, v32_25, color="#10b981", linewidth=2.0, label=f"V3.2 主线版 [固定单笔25万] (最高 ¥658.51万 / 期末 ¥{v32_25[-1]:.2f}万, +506.82%)")
ax1.plot(d10, v33_12, color="#a855f7", linewidth=1.9, label=f"V3.3 均衡版 [12%动态复利扩容] (最高 ¥627.03万 / 期末 ¥{v33_12[-1]:.2f}万, +406.21%)")
ax1.plot(d10, v30_12, color="#38bdf8", linewidth=1.7, linestyle="-.", label=f"V3.0 稳健版 [固定单笔12.5万] (最高 ¥330.34万 / 期末 ¥{v30_12[-1]:.2f}万, +210.42%)")
ax1.plot(d10, v_bench, color="#94a3b8", linewidth=1.4, linestyle="--", label="沪深300基准等值净值 (期末 ¥121.01万, +21.01%)")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp V3.3 终极版 vs V3.2 / V3.0 近10年真实PIT+严苛涨跌停回测对比 (100万本金)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=10)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

u_33 = [(1.0 - x["cash"] / x["portfolio_value"]) * 100.0 for x in h33_15]
u_32 = [(1.0 - x["cash"] / x["portfolio_value"]) * 100.0 for x in h32_25]
k = 20
u_33_ma = np.convolve(u_33, np.ones(k)/k, mode="same")
u_32_ma = np.convolve(u_32, np.ones(k)/k, mode="same")

ax2.plot(d10, u_33_ma, color="#f59e0b", linewidth=2.0, label="V3.3 [15%净值动态扩容] 20日平均仓位利用率 (%) — 后期净值500万+仍保持 75%~90% 满负荷复利")
ax2.plot(d10, u_32_ma, color="#38bdf8", linewidth=1.8, linestyle="--", label="V3.2 [固定单笔25万] 20日平均仓位利用率 (%) — 后期因25万上限锁死导致利用率跌至 35%")
ax2.set_ylim(0, 105)
ax2.set_ylabel("资金利用率 (%)", color="#e2e8f0", fontsize=11)
ax2.set_title("为什么 V3.3 十年能多赚 193.4 万？—— 解决后期净值增长后的「64%现金闲置瓶颈」", color="#f8fafc", fontsize=12, fontweight="bold", pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
leg2 = ax2.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

d1 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h33_1y]
v1_15 = [x["portfolio_value"] / 10000.0 for x in h33_1y]
v1_b = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h33_1y]
ax3.plot(d1, v1_15, color="#10b981", linewidth=2.3, label=f"V3.3 终极版 [15%动态扩容] 最近1年净值 (期末 ¥{v1_15[-1]:.2f}万, +33.94% | 最大回撤降至 -16.10% | 三大板块全盈利: 沪+11.0万/创+9.9万/科+12.6万)")
ax3.plot(d1, v1_b, color="#94a3b8", linewidth=1.4, linestyle="--", label="沪深300基准等值净值 (-14.87%)")
ax3.axhline(100.0, color="#64748b", linewidth=1)
ax3.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax3.set_xlabel("交易日期", color="#e2e8f0", fontsize=11)
ax3.set_title("V3.3 最近1年 (2025.10~2026.10) 净值曲线 (三大板块均衡盈利，最大回撤从 V3.2 的 -19.40% 降至 -16.10%)", color="#f8fafc", fontsize=11.8, fontweight="bold", pad=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg3 = ax3.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.3)
for text in leg3.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v3_3_ultimate.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
