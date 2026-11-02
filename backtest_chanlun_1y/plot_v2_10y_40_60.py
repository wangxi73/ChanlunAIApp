import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_v2_10y_40_60.json", "r", encoding="utf-8") as f:
    data = json.load(f)

h_v1 = data["v1_original"]["history"]
h_v2_dual = data["v2_dual"]["history"]
h_v2_idx = data["v2_index"]["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_v1]
v_v1 = [x["portfolio_value"] / 10000.0 for x in h_v1]
v_v2_dual = [x["portfolio_value"] / 10000.0 for x in h_v2_dual]
v_v2_idx = [x["portfolio_value"] / 10000.0 for x in h_v2_idx]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_v1]

def calc_dd(hist):
    peak = hist[0]["portfolio_value"]
    dds = []
    for x in hist:
        v = x["portfolio_value"]
        if v > peak:
            peak = v
        dds.append(-(peak - v) / peak * 100.0)
    return dds

dd_v1 = calc_dd(h_v1)
dd_v2 = calc_dd(h_v2_dual)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"height_ratios": [2.3, 1]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, v_v2_dual, label=f"V2新算法 [双因子牛熊控仓] (最高 ¥477.31万 / 期末 ¥{v_v2_dual[-1]:.2f}万, +{h_v2_dual[-1]['portfolio_return_pct']:.2f}%)", color="#10b981", linewidth=2.4)
ax1.plot(dates, v_v2_idx, label=f"V2新算法 [纯大盘缠论控仓] (最高 ¥476.37万 / 期末 ¥{v_v2_idx[-1]:.2f}万, +{h_v2_idx[-1]['portfolio_return_pct']:.2f}%)", color="#38bdf8", linewidth=1.7, linestyle="-.")
ax1.plot(dates, v_v1, label=f"V1原版算法 [无环境控仓] (最高 ¥431.71万 / 期末 ¥{v_v1[-1]:.2f}万, +{h_v1[-1]['portfolio_return_pct']:.2f}%)", color="#f59e0b", linewidth=2.0)
ax1.plot(dates, v_bench, label=f"业绩基准：沪深300等值净值 (期末 ¥{v_bench[-1]:.2f}万, +{h_v1[-1]['hs300_return_pct']:.2f}%)", color="#94a3b8", linewidth=1.5, linestyle="--")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp V2牛熊环境控仓新算法 vs V1原版 近10年回测对比 (40%沪深300+60%全A · 100万本金)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=12)
ax1.set_ylabel("账户总净值 (万元 RMB)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

ax2.fill_between(dates, dd_v1, 0, color="#f59e0b", alpha=0.22, label=f"V1原版 最大回撤 (-{data['v1_original']['metrics']['max_drawdown_pct']:.2f}%)")
ax2.plot(dates, dd_v1, color="#f59e0b", linewidth=1.1)
ax2.fill_between(dates, dd_v2, 0, color="#10b981", alpha=0.25, label=f"V2新算法 最大回撤 (-{data['v2_dual']['metrics']['max_drawdown_pct']:.2f}% | 降低11.82%)")
ax2.plot(dates, dd_v2, color="#10b981", linewidth=1.3)
ax2.set_ylabel("动态回撤 (%)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易年份", color="#e2e8f0", fontsize=11)
leg2 = ax2.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.8)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=0)
plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v2_10y_40_60.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)

m_v1 = {x["date"]: x for x in h_v1}
m_v2 = {x["date"]: x for x in h_v2_dual}
all_d = [x["date"] for x in h_v1]
targets = ["2016-10-28"] + [f"{y}-10-31" for y in range(2017, 2026)] + ["2026-10-29"]
cps = [[d for d in all_d if d <= t][-1] for t in targets]
for i in range(1, len(cps)):
    d0, d1 = cps[i-1], cps[i]
    br = (m_v1[d1]["hs300_level"] - m_v1[d0]["hs300_level"]) / m_v1[d0]["hs300_level"] * 100.0
    v1_e, v2_e = m_v1[d1]["portfolio_value"], m_v2[d1]["portfolio_value"]
    r1 = (v1_e - m_v1[d0]["portfolio_value"]) / m_v1[d0]["portfolio_value"] * 100.0
    r2 = (v2_e - m_v2[d0]["portfolio_value"]) / m_v2[d0]["portfolio_value"] * 100.0
    print(f"Y{i:02d} ({d0}..{d1}) | HS300:{br:+6.2f}% | V1:¥{v1_e/1e4:6.2f}w ({r1:+6.2f}%) | V2:¥{v2_e/1e4:6.2f}w ({r2:+6.2f}%) | V2超额:{r2-r1:+6.2f}%")
