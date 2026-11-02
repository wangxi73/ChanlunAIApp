import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base_dir = Path(__file__).resolve().parent
with open(base_dir / "backtest_results_2019_2020.json", "r", encoding="utf-8") as f:
    data = json.load(f)

h_hs300 = data["quant_hs300"]["history"]
h_alla = data["quant_all_a"]["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_hs300]
ret_hs300_acct = [x["portfolio_return_pct"] for x in h_hs300]
ret_alla_acct = [x["portfolio_return_pct"] for x in h_alla]
ret_bench = [x["hs300_return_pct"] for x in h_hs300]

def calc_dd_series(hist):
    peak = hist[0]["portfolio_value"]
    dds = []
    for x in hist:
        v = x["portfolio_value"]
        if v > peak:
            peak = v
        dds.append(-(peak - v) / peak * 100.0)
    return dds

dd_hs300 = calc_dd_series(h_hs300)
dd_alla = calc_dd_series(h_alla)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, ret_hs300_acct, label=f"账户① 无AI·仅限沪深300 ({ret_hs300_acct[-1]:+.2f}%)", color="#38bdf8", linewidth=2.2)
ax1.plot(dates, ret_alla_acct, label=f"账户② 无AI·全市场A股 ({ret_alla_acct[-1]:+.2f}%)", color="#f59e0b", linewidth=2.2)
ax1.plot(dates, ret_bench, label=f"业绩基准：沪深300指数 ({ret_bench[-1]:+.2f}%)", color="#94a3b8", linewidth=1.6, linestyle="--")
ax1.axhline(0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp 无AI介入双模拟账户 历史回测 (2019-04-01 至 2020-04-01 震荡与疫情冲击周期)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
ax1.set_ylabel("累计收益率 (%)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=10.5)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

ax2.fill_between(dates, dd_hs300, 0, color="#38bdf8", alpha=0.28, label=f"账户① 最大回撤 (-{data['quant_hs300']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.plot(dates, dd_hs300, color="#38bdf8", linewidth=1.2)
ax2.fill_between(dates, dd_alla, 0, color="#f59e0b", alpha=0.28, label=f"账户② 最大回撤 (-{data['quant_all_a']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.plot(dates, dd_alla, color="#f59e0b", linewidth=1.2)
ax2.set_ylabel("动态回撤 (%)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易日期", color="#e2e8f0", fontsize=11)
leg2 = ax2.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=10)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=0)
plt.tight_layout()
out_img = base_dir / "chanlun_quant_backtest_2019_2020.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
