import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base_dir = Path(__file__).resolve().parent
with open(base_dir / "backtest_results_10y_50_50.json", "r", encoding="utf-8") as f:
    data = json.load(f)

h_ind = data["independent"]["history"]
h_reb = data["rebalanced"]["history"]
h_dyn = data["dynamic_rebalanced"]["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_ind]
v_reb = [x["portfolio_value"] / 10000.0 for x in h_reb]
v_dyn = [x["portfolio_value"] / 10000.0 for x in h_dyn]
v_ind = [x["portfolio_value"] / 10000.0 for x in h_ind]
v_reb_hs = [x["hs300_sub_value"] / 10000.0 for x in h_reb]
v_reb_aa = [x["alla_sub_value"] / 10000.0 for x in h_reb]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_ind]

def calc_dd_series(hist):
    peak = hist[0]["portfolio_value"]
    dds = []
    for x in hist:
        v = x["portfolio_value"]
        if v > peak:
            peak = v
        dds.append(-(peak - v) / peak * 100.0)
    return dds

dd_reb = calc_dd_series(h_reb)
dd_ind = calc_dd_series(h_ind)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, v_reb, label=f"方案② 50/50 每年末再平衡 [单笔12.5万] (最高 ¥401.69万 / 期末 ¥{v_reb[-1]:.2f}万, +{h_reb[-1]['portfolio_return_pct']:.2f}%)", color="#10b981", linewidth=2.4)
ax1.plot(dates, v_dyn, label=f"方案③ 50/50 每年末再平衡+动态仓位 (最高 ¥330.34万 / 期末 ¥{v_dyn[-1]:.2f}万, +{h_dyn[-1]['portfolio_return_pct']:.2f}%)", color="#38bdf8", linewidth=1.9)
ax1.plot(dates, v_ind, label=f"方案① 50/50 独立不调仓 [单笔12.5万] (最高 ¥253.52万 / 期末 ¥{v_ind[-1]:.2f}万, +{h_ind[-1]['portfolio_return_pct']:.2f}%)", color="#f59e0b", linewidth=1.9)
ax1.plot(dates, v_bench, label=f"业绩基准：沪深300等值净值 (期末 ¥{v_bench[-1]:.2f}万, +{h_ind[-1]['hs300_return_pct']:.2f}%)", color="#94a3b8", linewidth=1.5, linestyle="--")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp 100万本金「50%沪深300 + 50%全A」近10年回测对比 (2016-10-28 至 2026-10-29)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
ax1.set_ylabel("组合总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

ax2.fill_between(dates, dd_reb, 0, color="#10b981", alpha=0.22, label=f"方案② 每年末再平衡 最大回撤 (-{data['rebalanced']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.plot(dates, dd_reb, color="#10b981", linewidth=1.2)
ax2.fill_between(dates, dd_ind, 0, color="#f59e0b", alpha=0.18, label=f"方案① 独立不调仓 最大回撤 (-{data['independent']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.plot(dates, dd_ind, color="#f59e0b", linewidth=1.1)
ax2.set_ylabel("动态回撤 (%)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易年份", color="#e2e8f0", fontsize=11)
leg2 = ax2.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=0)
plt.tight_layout()
out_img = base_dir / "chanlun_quant_backtest_10y_50_50.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
