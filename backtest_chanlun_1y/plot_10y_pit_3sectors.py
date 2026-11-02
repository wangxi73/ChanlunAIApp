import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_10y_pit_3sectors.json", "r", encoding="utf-8") as f:
    pit_data = json.load(f)

v2_list = pit_data["pit_v2_regime"]
v1_list = pit_data["pit_v1_original"]

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
b1 = ax1.bar(x - width, v2_peak, width, label="当年真实股票池 [V2牛熊控仓] · 10年最高净值(万)", color="#f59e0b", alpha=0.9)
b2 = ax1.bar(x, v2_final, width, label="当年真实股票池 [V2牛熊控仓] · 10年期末净值(万)", color="#10b981", alpha=0.95)
b3 = ax1.bar(x + width, v1_final, width, label="当年真实股票池 [V1原版无控仓] · 10年期末净值(万)", color="#38bdf8", alpha=0.82)

ax1_dd = ax1.twinx()
ax1_dd.plot(x, v2_dd, color="#f43f5e", marker="o", linewidth=2.0, label="当年真实股票池 [V2] 最大回撤 (%) [右轴]")
ax1_dd.set_ylabel("最大回撤 (%)", color="#f43f5e", fontsize=11)
ax1_dd.tick_params(colors="#f43f5e")
ax1_dd.set_ylim(30, 60)
ax1_dd.spines["top"].set_visible(False)

ax1.set_xticks(x)
ax1.set_xticklabels([f"沪:创:科\n{l}" for l in labels], fontsize=8.5)
ax1.set_ylabel("账户总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
ax1.set_ylim(0, 620)
ax1.set_title("ChanlunAIApp「逐年当年真实股票池 (无幸存者偏差 PIT)」近10年16组配比回测寻优 (100万本金)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)

for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 7, f"{h:.0f}w", ha="center", va="bottom", color="#10b981", fontsize=8, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_dd.get_legend_handles_labels()
leg1 = ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.3)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

v2_map = {r["ratio"]: r for r in v2_list}
dates = [datetime.strptime(pt["date"], "%Y-%m-%d") for pt in v2_list[0]["history"]]
rep_specs = [
    ("20:50:30", "#f59e0b", 2.3, "20%沪深300 : 50%创业板 : 30%科创板 [进取夏普冠军 0.68]"),
    ("30:35:35", "#10b981", 2.2, "30%沪深300 : 35%创业板 : 35%科创板 [三足均衡最优]"),
    ("40:20:40", "#38bdf8", 2.1, "40%沪深300 : 20%创业板 : 40%科创板 [稳健高收益甜点]"),
    ("50:25:25", "#a855f7", 2.0, "50%沪深300 : 25%创业板 : 25%科创板 [最低回撤 -37.95%]"),
    ("0:100:0", "#ef4444", 1.6, "0%沪深300 : 100%创业板 : 0%科创板 [纯创业板(回撤-50.2%)]"),
    ("100:0:0", "#94a3b8", 1.5, "100%沪深300 : 0%创业板 : 0%科创板 [纯沪深300基准]"),
]
for key, col, lw, desc in rep_specs:
    obj = v2_map[key]
    m = obj["metrics"]
    vals = [pt["portfolio_value"] / 10000.0 for pt in obj["history"]]
    ax2.plot(dates, vals, color=col, linewidth=lw, label=f"{desc} (期末 ¥{m['final_value']/1e4:.1f}万 / 最高 ¥{m['peak_value']/1e4:.1f}万 / 回撤 -{m['max_drawdown_pct']:.2f}%)")

ax2.axhline(100.0, color="#64748b", linewidth=1, linestyle="--")
ax2.set_title("逐年当年真实股票池 (Point-in-Time) 代表性配比近10年净值走势 (含历史暴雷/衰落权重股)", color="#f8fafc", fontsize=12.5, fontweight="bold", pad=10)
ax2.set_ylabel("组合总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易年份", color="#e2e8f0", fontsize=11)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
leg2 = ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.3)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_10y_pit_3sectors.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)

h_map = {r["ratio"]: {x["date"]: x for x in r["history"]} for r in v2_list}
all_d = [x["date"] for x in v2_list[0]["history"]]
targets = ["2016-10-28"] + [f"{y}-10-31" for y in range(2017, 2026)] + ["2026-10-29"]
cps = [[d for d in all_d if d <= t][-1] for t in targets]
for i in range(1, len(cps)):
    d0, d1 = cps[i-1], cps[i]
    br = (h_map["50:25:25"][d1]["hs300_level"] - h_map["50:25:25"][d0]["hs300_level"]) / h_map["50:25:25"][d0]["hs300_level"] * 100.0
    row = []
    for k in ["50:25:25", "40:20:40", "30:35:35", "20:50:30"]:
        v0 = h_map[k][d0]["portfolio_value"]
        v1 = h_map[k][d1]["portfolio_value"]
        r = (v1 - v0) / v0 * 100.0
        row.append(f"¥{v1/1e4:6.2f}w({r:+6.2f}%)")
    print(f"Y{i:02d} ({d0}..{d1}) | HS300:{br:+6.2f}% | 50:25:25->{row[0]} | 40:20:40->{row[1]} | 30:35:35->{row[2]} | 20:50:30->{row[3]}")
