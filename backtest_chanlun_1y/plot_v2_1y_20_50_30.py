import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_v2_1y_20_50_30.json", "r", encoding="utf-8") as f:
    data = json.load(f)

v2 = data["v2_latest"]
v1 = data["v1_no_regime"]
h2 = v2["history"]
h1 = v1["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h2]
val_v2 = [x["portfolio_value"] / 10000.0 for x in h2]
val_v1 = [x["portfolio_value"] / 10000.0 for x in h1]
val_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h2]

ret_hs_sub = [(x["hs300_sub"] - 200000.0) / 200000.0 * 100.0 for x in h2]
ret_cyb_sub = [(x["cyb_sub"] - 500000.0) / 500000.0 * 100.0 for x in h2]
ret_kcb_sub = [(x["kcb_sub"] - 300000.0) / 300000.0 * 100.0 for x in h2]

def calc_dd(hist):
    peak = hist[0]["portfolio_value"]
    dds = []
    for x in hist:
        v = x["portfolio_value"]
        if v > peak:
            peak = v
        dds.append(-(peak - v) / peak * 100.0)
    return dds

dd_v2 = calc_dd(h2)
dd_v1 = calc_dd(h1)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2.0, 1.3, 1.0]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, val_v2, label=f"最新模型 V2 [20:50:30 + 牛熊控仓] (最高 ¥208.25万 / 期末 ¥{val_v2[-1]:.2f}万, +{h2[-1]['portfolio_return_pct']:.2f}%)", color="#10b981", linewidth=2.4)
ax1.plot(dates, val_v1, label=f"对照模型 V1 [20:50:30 无控仓] (最高 ¥211.58万 / 期末 ¥{val_v1[-1]:.2f}万, +{h1[-1]['portfolio_return_pct']:.2f}%)", color="#f59e0b", linewidth=1.8, linestyle="-.")
ax1.plot(dates, val_bench, label=f"业绩基准：沪深300等值净值 (期末 ¥{val_bench[-1]:.2f}万, {h2[-1]['hs300_return_pct']:+.2f}%)", color="#94a3b8", linewidth=1.5, linestyle="--")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp 最新V2模型「20%沪深300 : 50%创业板 : 30%科创板」最近1年回测 (100万本金 · 2025.10~2026.10)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

ax2.plot(dates, ret_cyb_sub, label=f"创业板子账户 (50万本金 -> 期末 ¥88.81万, +{ret_cyb_sub[-1]:.2f}%)", color="#f59e0b", linewidth=2.0)
ax2.plot(dates, ret_kcb_sub, label=f"科创板子账户 (30万本金 -> 期末 ¥39.87万, +{ret_kcb_sub[-1]:.2f}%)", color="#a855f7", linewidth=2.0)
ax2.plot(dates, ret_hs_sub, label=f"沪深300子账户 (20万本金 -> 期末 ¥24.55万, +{ret_hs_sub[-1]:.2f}%)", color="#38bdf8", linewidth=2.0)
ax2.axhline(0.0, color="#64748b", linewidth=1)
ax2.set_ylabel("子账户收益率 (%)", color="#e2e8f0", fontsize=11)
ax2.set_title("三大板块子账户独立收益率拆解 (20万沪深300 + 50万创业板 + 30万科创板)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
leg2 = ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

ax3.fill_between(dates, dd_v1, 0, color="#f59e0b", alpha=0.22, label=f"无控仓 V1 最大回撤 (-{v1['metrics']['max_drawdown_pct']:.2f}%)")
ax3.plot(dates, dd_v1, color="#f59e0b", linewidth=1.1)
ax3.fill_between(dates, dd_v2, 0, color="#10b981", alpha=0.28, label=f"最新模型 V2 最大回撤 (-{v2['metrics']['max_drawdown_pct']:.2f}% | 7~10月回调少亏13.07万)")
ax3.plot(dates, dd_v2, color="#10b981", linewidth=1.3)
ax3.set_ylabel("动态回撤 (%)", color="#e2e8f0", fontsize=11)
ax3.set_xlabel("交易月份", color="#e2e8f0", fontsize=11)
leg3 = ax3.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg3.get_texts():
    text.set_color("#f8fafc")

ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=0)
plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v2_1y_20_50_30.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)

for i in range(len(h2)):
    cur_m = h2[i]["date"][:7]
    nxt_m = h2[i+1]["date"][:7] if i+1 < len(h2) else None
    if cur_m != nxt_m:
        x = h2[i]
        x1 = h1[i]
        print(f"{x['date']} | HS300:{x['hs300_level']:.2f}({x['hs300_return_pct']:+.2f}%) | V2总净值:¥{x['portfolio_value']/1e4:6.2f}w({x['portfolio_return_pct']:+6.2f}%) [沪:{x['hs300_sub']/1e4:.1f}w 创:{x['cyb_sub']/1e4:.1f}w 科:{x['kcb_sub']/1e4:.1f}w] | V1无控仓:¥{x1['portfolio_value']/1e4:6.2f}w | 环境:{x['regime']}")
