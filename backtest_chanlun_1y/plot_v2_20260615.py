import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_v2_20260615.json", "r", encoding="utf-8") as f:
    data = json.load(f)

v2_stop = data["v2_ratchet_stop"]
v2 = data["v2"]
v1 = data["v1"]

h_s = v2_stop["history"]
h_2 = v2["history"]
h_1 = v1["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_s]
val_s = [x["portfolio_value"] / 10000.0 for x in h_s]
val_2 = [x["portfolio_value"] / 10000.0 for x in h_2]
val_1 = [x["portfolio_value"] / 10000.0 for x in h_1]
val_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_s]

ret_hs = [(x["hs300_sub"] - 200000.0) / 200000.0 * 100.0 for x in h_s]
ret_cyb = [(x["cyb_sub"] - 500000.0) / 500000.0 * 100.0 for x in h_s]
ret_kcb = [(x["kcb_sub"] - 300000.0) / 300000.0 * 100.0 for x in h_s]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.2), gridspec_kw={"height_ratios": [1.8, 1.2]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, val_s, label=f"V2+熊市锁死止损 [20:50:30] (最高 ¥115.02万 / 期末 ¥{val_s[-1]:.2f}万, {h_s[-1]['portfolio_return_pct']:+.2f}%)", color="#10b981", linewidth=2.4)
ax1.plot(dates, val_2, label=f"V2牛熊控仓 [20:50:30] (最高 ¥115.02万 / 期末 ¥{val_2[-1]:.2f}万, {h_2[-1]['portfolio_return_pct']:+.2f}%)", color="#38bdf8", linewidth=1.9)
ax1.plot(dates, val_1, label=f"V1无控仓 [20:50:30] (最高 ¥115.02万 / 期末 ¥{val_1[-1]:.2f}万, {h_1[-1]['portfolio_return_pct']:+.2f}%)", color="#f59e0b", linewidth=1.7, linestyle="-.")
ax1.plot(dates, val_bench, label=f"业绩基准：沪深300等值净值 (期末 ¥{val_bench[-1]:.2f}万, {h_s[-1]['hs300_return_pct']:+.2f}%)", color="#94a3b8", linewidth=1.5, linestyle="--")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp 2026-06-15 至 2026-10-29 高位见顶回落期专项回测 (100万本金 · 20:50:30)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
ax1.set_ylabel("账户总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

ax2.plot(dates, ret_hs, label=f"沪深300子账户 (20万本金 -> 期末 ¥19.21万, {ret_hs[-1]:+.2f}%)", color="#38bdf8", linewidth=2.0)
ax2.plot(dates, ret_cyb, label=f"创业板子账户 (50万本金 -> 期末 ¥45.22万, {ret_cyb[-1]:+.2f}%)", color="#10b981", linewidth=2.0)
ax2.plot(dates, ret_kcb, label=f"科创板子账户 (30万本金 -> 期末 ¥21.36万, {ret_kcb[-1]:+.2f}%)", color="#a855f7", linewidth=2.0)
ax2.axhline(0.0, color="#64748b", linewidth=1)
ax2.set_ylabel("子账户累计涨跌 (%)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("日期 (2026年6月15日 ~ 10月29日)", color="#e2e8f0", fontsize=11)
ax2.set_title("三大板块子账户在 2026.06~10 高位调整期的独立表现拆解", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
leg2 = ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=14))
plt.xticks(rotation=0)
plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v2_20260615.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)

for d_target in ["2026-06-15", "2026-06-25", "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30", "2026-10-29"]:
    x_s = [pt for pt in h_s if pt["date"] <= d_target][-1]
    x_2 = [pt for pt in h_2 if pt["date"] <= d_target][-1]
    x_1 = [pt for pt in h_1 if pt["date"] <= d_target][-1]
    print(f"{x_s['date']} ({x_s['regime']:5s}) | 沪深300:{x_s['hs300_level']:.2f}({x_s['hs300_return_pct']:+.2f}%) | V2+硬止损:¥{x_s['portfolio_value']/1e4:6.2f}w({x_s['portfolio_return_pct']:+.2f}%) | V2:¥{x_2['portfolio_value']/1e4:6.2f}w({x_2['portfolio_return_pct']:+.2f}%) | V1:¥{x_1['portfolio_value']/1e4:6.2f}w({x_1['portfolio_return_pct']:+.2f}%)")
