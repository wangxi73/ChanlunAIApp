import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_10y_3sectors.json", "r", encoding="utf-8") as f:
    data = json.load(f)

v2_list = data["v2_regime"]
v1_list = data["v1_original"]

labels = [r["ratio"] for r in v2_list]
v2_final = [r["metrics"]["final_value"] / 10000.0 for r in v2_list]
v2_peak = [r["metrics"]["peak_value"] / 10000.0 for r in v2_list]
v1_final = [r["metrics"]["final_value"] / 10000.0 for r in v1_list]
v2_dd = [r["metrics"]["max_drawdown_pct"] for r in v2_list]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9.5), gridspec_kw={"height_ratios": [1.15, 1.15]})
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

x = np.arange(len(labels))
width = 0.27
b1 = ax1.bar(x - width, v2_peak, width, label="V2牛熊控仓 · 10年最高净值(万)", color="#f59e0b", alpha=0.9)
b2 = ax1.bar(x, v2_final, width, label="V2牛熊控仓 · 10年期末净值(万)", color="#10b981", alpha=0.95)
b3 = ax1.bar(x + width, v1_final, width, label="V1原版无控仓 · 10年期末净值(万)", color="#38bdf8", alpha=0.82)

ax1_dd = ax1.twinx()
ax1_dd.plot(x, v2_dd, color="#f43f5e", marker="o", linewidth=2.0, label="V2最大回撤 (%) [右轴]")
ax1_dd.set_ylabel("最大回撤 (%)", color="#f43f5e", fontsize=11)
ax1_dd.tick_params(colors="#f43f5e")
ax1_dd.set_ylim(15, 55)
ax1_dd.spines["top"].set_visible(False)

ax1.set_xticks(x)
ax1.set_xticklabels([f"沪:创:科\n{l}" for l in labels], fontsize=8.5)
ax1.set_ylabel("账户总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
ax1.set_ylim(0, 960)
ax1.set_title("ChanlunAIApp 近10年 (2016.10~2026.10)「沪深300 : 创业板 : 科创板」16组配比全景寻优 (100万本金)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)

for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 10, f"{h:.0f}w", ha="center", va="bottom", color="#10b981", fontsize=7.8, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_dd.get_legend_handles_labels()
leg1 = ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

v2_map = {r["ratio"]: r for r in v2_list}
dates = [datetime.strptime(pt["date"], "%Y-%m-%d") for pt in v2_list[0]["history"]]
rep_specs = [
    ("20:50:30", "#f59e0b", 2.4, "20%沪深300 : 50%创业板 : 30%科创板 [夏普冠军 0.97]"),
    ("30:40:30", "#10b981", 2.2, "30%沪深300 : 40%创业板 : 30%科创板 [三足均衡最优]"),
    ("40:40:20", "#38bdf8", 2.2, "40%沪深300 : 40%创业板 : 20%科创板 [最低回撤 -24.35%]"),
    ("50:30:20", "#a855f7", 1.9, "50%沪深300 : 30%创业板 : 20%科创板 [稳健对半标杆]"),
    ("0:60:40", "#ec4899", 1.7, "0%沪深300 : 60%创业板 : 40%科创板 [纯双创进攻]"),
    ("100:0:0", "#94a3b8", 1.5, "100%沪深300 : 0%创业板 : 0%科创板 [纯蓝筹基准]"),
]
for key, col, lw, desc in rep_specs:
    obj = v2_map[key]
    m = obj["metrics"]
    vals = [pt["portfolio_value"] / 10000.0 for pt in obj["history"]]
    ax2.plot(dates, vals, color=col, linewidth=lw, label=f"{desc} (期末 ¥{m['final_value']/1e4:.1f}万 / 最高 ¥{m['peak_value']/1e4:.1f}万 / 回撤 -{m['max_drawdown_pct']:.2f}%)")

ax2.set_title("代表性「沪深300 : 创业板 : 科创板」配比方案近10年净值曲线 (V2牛熊控仓 + 无科创板期由创业板替代)", color="#f8fafc", fontsize=12.5, fontweight="bold", pad=10)
ax2.set_ylabel("组合总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易年份", color="#e2e8f0", fontsize=11)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
leg2 = ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_10y_3sectors.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
