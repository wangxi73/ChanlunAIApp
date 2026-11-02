import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base_dir = Path(__file__).resolve().parent
with open(base_dir / "backtest_results_5y_100w.json", "r", encoding="utf-8") as f:
    data = json.load(f)

p_hs300 = data["proportional_25w"]["quant_hs300"]["history"]
p_alla = data["proportional_25w"]["quant_all_a"]["history"]
d_hs300 = data["diversified_2_5w"]["quant_hs300"]["history"]
d_alla = data["diversified_2_5w"]["quant_all_a"]["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in p_hs300]
val_p_alla = [x["portfolio_value"] / 10000.0 for x in p_alla]
val_d_alla = [x["portfolio_value"] / 10000.0 for x in d_alla]
val_p_hs300 = [x["portfolio_value"] / 10000.0 for x in p_hs300]
val_d_hs300 = [x["portfolio_value"] / 10000.0 for x in d_hs300]
val_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in p_hs300]

def calc_dd_series(hist):
    peak = hist[0]["portfolio_value"]
    dds = []
    for x in hist:
        v = x["portfolio_value"]
        if v > peak:
            peak = v
        dds.append(-(peak - v) / peak * 100.0)
    return dds

dd_p_alla = calc_dd_series(p_alla)
dd_p_hs300 = calc_dd_series(p_hs300)
dd_d_alla = calc_dd_series(d_alla)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, val_p_alla, label=f"账户② 全市场A股 [单笔25万·集中] (期末 ¥{val_p_alla[-1]:.2f}万 / +{p_alla[-1]['portfolio_return_pct']:.2f}%)", color="#f59e0b", linewidth=2.2)
ax1.plot(dates, val_d_alla, label=f"账户② 全市场A股 [单笔2.5万·分散] (期末 ¥{val_d_alla[-1]:.2f}万 / +{d_alla[-1]['portfolio_return_pct']:.2f}%)", color="#10b981", linewidth=2.0)
ax1.plot(dates, val_p_hs300, label=f"账户① 仅限沪深300 [单笔25万·集中] (期末 ¥{val_p_hs300[-1]:.2f}万 / +{p_hs300[-1]['portfolio_return_pct']:.2f}%)", color="#38bdf8", linewidth=2.0)
ax1.plot(dates, val_d_hs300, label=f"账户① 仅限沪深300 [单笔2.5万·分散] (期末 ¥{val_d_hs300[-1]:.2f}万 / +{d_hs300[-1]['portfolio_return_pct']:.2f}%)", color="#a855f7", linewidth=1.6, linestyle="-.")
ax1.plot(dates, val_bench, label=f"业绩基准：沪深300等值净值 (期末 ¥{val_bench[-1]:.2f}万 / {p_hs300[-1]['hs300_return_pct']:+.2f}%)", color="#94a3b8", linewidth=1.6, linestyle="--")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp 无AI介入双模拟账户 近5年长周期回测 (初始资金100万 · 2021-10-29 至 2026-10-29)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
ax1.set_ylabel("账户总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

ax2.fill_between(dates, dd_p_alla, 0, color="#f59e0b", alpha=0.22, label=f"账户② [单笔25万] 最大回撤 (-{data['proportional_25w']['quant_all_a']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.plot(dates, dd_p_alla, color="#f59e0b", linewidth=1.1)
ax2.fill_between(dates, dd_p_hs300, 0, color="#38bdf8", alpha=0.22, label=f"账户① [单笔25万] 最大回撤 (-{data['proportional_25w']['quant_hs300']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.plot(dates, dd_p_hs300, color="#38bdf8", linewidth=1.1)
ax2.plot(dates, dd_d_alla, color="#10b981", linewidth=1.4, label=f"账户② [单笔2.5万分散] 最大回撤 (-{data['diversified_2_5w']['quant_all_a']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.set_ylabel("动态回撤 (%)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易年份", color="#e2e8f0", fontsize=11)
leg2 = ax2.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=0)
plt.tight_layout()
out_img = base_dir / "chanlun_quant_backtest_5y_100w.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
