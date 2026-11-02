import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base_dir = Path(__file__).resolve().parent
with open(base_dir / "backtest_results_10y_ratios.json", "r", encoding="utf-8") as f:
    data = json.load(f)

reb = data["rebalanced"]
ind = data["independent"]

labels = [r["ratio"] for r in reb]
reb_final = [r["metrics"]["final_value"] / 10000.0 for r in reb]
reb_peak = [r["metrics"]["peak_value"] / 10000.0 for r in reb]
ind_final = [r["metrics"]["final_value"] / 10000.0 for r in ind]
reb_dd = [r["metrics"]["max_drawdown_pct"] for r in reb]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.15, 1.15]})
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

x = np.arange(len(labels))
width = 0.27
b1 = ax1.bar(x - width, reb_peak, width, label="每年末动态再平衡 · 10年最高净值(万)", color="#f59e0b", alpha=0.9)
b2 = ax1.bar(x, reb_final, width, label="每年末动态再平衡 · 10年期末净值(万)", color="#10b981", alpha=0.95)
b3 = ax1.bar(x + width, ind_final, width, label="初始分配不调仓 · 10年期末净值(万)", color="#38bdf8", alpha=0.85)
ax1.axhline(100.0, color="#94a3b8", linestyle="--", linewidth=1, label="初始本金 (100万)")
ax1.set_xticks(x)
ax1.set_xticklabels([f"沪深300:全A\n{l}" for l in labels], fontsize=9.5)
ax1.set_ylabel("账户净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
ax1.set_title("ChanlunAIApp 近10年 (2016.10~2026.10) 11种「沪深300 : 全A」资金配比方案全景对比", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 6, f"{h:.0f}w", ha="center", va="bottom", color="#10b981", fontsize=8.5, fontweight="bold")
for bar in b1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 6, f"{h:.0f}w", ha="center", va="bottom", color="#f59e0b", fontsize=8)
ax1.set_ylim(0, 480)
leg1 = ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

# Subplot 2: Equity curves of top representative ratios (Rebalanced)
rep_ratios = [
    ("40:60", "#10b981", 2.4, "40%沪深300 : 60%全A [收益冠军]"),
    ("80:20", "#38bdf8", 2.1, "80%沪深300 : 20%全A [稳健夏普冠军]"),
    ("90:10", "#a855f7", 1.8, "90%沪深300 : 10%全A"),
    ("50:50", "#f59e0b", 1.9, "50%沪深300 : 50%全A [均衡配置]"),
    ("100:0", "#e2e8f0", 1.6, "100%沪深300 : 0%全A [纯蓝筹]"),
    ("0:100", "#ef4444", 1.5, "0%沪深300 : 100%全A [纯全A]"),
]
reb_map = {r["ratio"]: r for r in reb}
dates = [datetime.strptime(pt["date"], "%Y-%m-%d") for pt in reb[0]["history"]]
for r_key, col, lw, desc in rep_ratios:
    r_obj = reb_map[r_key]
    vals = [pt["portfolio_value"] / 10000.0 for pt in r_obj["history"]]
    m = r_obj["metrics"]
    ax2.plot(dates, vals, color=col, linewidth=lw, label=f"{desc} (期末 ¥{m['final_value']/1e4:.1f}万 / 最高 ¥{m['peak_value']/1e4:.1f}万)")

ax2.axhline(100.0, color="#64748b", linewidth=1, linestyle="--")
ax2.set_title("代表性「沪深300 : 全A」配比方案近10年净值曲线对比 (每年末动态再平衡)", color="#f8fafc", fontsize=12.5, fontweight="bold", pad=10)
ax2.set_ylabel("组合总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易年份", color="#e2e8f0", fontsize=11)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
leg2 = ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base_dir / "chanlun_quant_backtest_10y_ratios.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
