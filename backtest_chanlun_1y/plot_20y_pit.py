import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_results_20y_pit.json", "r", encoding="utf-8") as f:
    data = json.load(f)

r_33_m60 = data["v3_3_m60"]
r_33_day = data["v3_3_day"]
r_32_fix = data["v3_2_fixed"]
r_30_fix = data["v3_0_steady"]

h_33_m60 = r_33_m60["history"]
h_33_day = r_33_day["history"]
h_32_fix = r_32_fix["history"]
h_30_fix = r_30_fix["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_33_m60]
v_33_m60 = [x["portfolio_value"] / 10000.0 for x in h_33_m60]
v_33_day = [x["portfolio_value"] / 10000.0 for x in h_33_day]
v_32_fix = [x["portfolio_value"] / 10000.0 for x in h_32_fix]
v_30_fix = [x["portfolio_value"] / 10000.0 for x in h_30_fix]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_33_m60]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1.8, 1.3, 1.15]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

# Panel 1: 20-Year Equity Curves
ax1.plot(dates, v_33_m60, color="#f59e0b", linewidth=2.3, label=f"V3.3 终极进取版 [15%动态复利 + 60m脉冲过滤] (期末 ¥{v_33_m60[-1]:.1f}万, +2254.97% | 最高 ¥2,888.3万)")
ax1.plot(dates, v_33_day, color="#ec4899", linewidth=2.0, label=f"V3.3 终极均衡版 [15%动态复利 + 日线透支过滤] (期末 ¥{v_33_day[-1]:.1f}万, +1930.61% | 最高 ¥2,457.1万)")
ax1.plot(dates, v_32_fix, color="#10b981", linewidth=2.1, label=f"V3.2 主线标准版 [固定单笔25万] (期末 ¥{v_32_fix[-1]:.1f}万, +810.25% | 最大回撤 -37.51% / 夏普 0.70)")
ax1.plot(dates, v_30_fix, color="#38bdf8", linewidth=2.0, label=f"V3.0 稳健防守版 [固定单笔12.5万·8仓] (期末 ¥{v_30_fix[-1]:.1f}万, +443.22% | 20年最大回撤仅 -23.28%! 夏普 0.75)")
ax1.plot(dates, v_bench, color="#94a3b8", linewidth=1.4, linestyle="--", label=f"业绩基准：沪深300等值净值 (期末 ¥{v_bench[-1]:.1f}万, +179.47%)")
ax1.axvline(datetime(2010, 3, 1), color="#f59e0b", linestyle=":", alpha=0.6, linewidth=1.2)
ax1.axvline(datetime(2019, 8, 26), color="#a855f7", linestyle=":", alpha=0.6, linewidth=1.2)
ax1.text(datetime(2007, 3, 1), 2500, "第一时代(2006~2010)\n纯沪深300/主板时代", color="#94a3b8", fontsize=9)
ax1.text(datetime(2012, 6, 1), 2500, "第二时代(2010.03~2019.08)\n沪深300 + 创业板(10cm)双轨时代", color="#f59e0b", fontsize=9)
ax1.text(datetime(2020, 6, 1), 2500, "第三时代(2019.08~2026.10)\n沪深300 + 创业板(20cm) + 科创板(20cm)", color="#a855f7", fontsize=9)
ax1.set_title("ChanlunAIApp 近20年 (2006-10-30 至 2026-10-29) 跨三时代真实PIT+严苛涨跌停回测全景对比 (100万本金)", color="#f8fafc", fontsize=13.5, fontweight="bold", pad=10)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.2)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

# Panel 2: Log-Scale Comparison of V3.0 Steady vs V3.2 Fixed vs Benchmark (highlighting drawdown stability across 2008, 2015, 2018, 2022)
ax2.plot(dates, v_32_fix, color="#10b981", linewidth=2.2, label="V3.2 主线版 [固定25万] (20年 9.1倍，穿越2008/2015/2018/2022四轮大股灾，最大回撤 -37.51%)")
ax2.plot(dates, v_30_fix, color="#38bdf8", linewidth=2.2, label="V3.0 稳健版 [固定12.5万] (20年 5.4倍，20年最大回撤仅 -23.28%，从未跌破初始100万本金!)")
ax2.plot(dates, v_bench, color="#94a3b8", linewidth=1.4, linestyle="--", label="沪深300基准 (2008年最大回撤 -72.3%，2015年最大回撤 -46.7%)")
ax2.set_ylabel("稳健组净值 (万元)", color="#e2e8f0", fontsize=11)
ax2.set_title("固定单笔组 (V3.2 与 V3.0) 20年稳健复利曲线放大图 (重点看 2008金融危机 与 2015千股跌停股灾 的抗跌表现)", color="#f8fafc", fontsize=11.8, fontweight="bold", pad=8)
leg2 = ax2.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.3)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

# Panel 3: 20-Year Three-Era Sector Allocation Stacked Area Chart
tot20 = np.array([x["portfolio_value"] for x in h_33_day])
w_hs = np.array([x["hs300_mkt_val"] for x in h_33_day]) / tot20 * 100.0
w_cyb = np.array([x["cyb_mkt_val"] for x in h_33_day]) / tot20 * 100.0
w_kcb = np.array([x["kcb_mkt_val"] for x in h_33_day]) / tot20 * 100.0
w_cash = np.array([x["cash"] for x in h_33_day]) / tot20 * 100.0

ax3.stackplot(
    dates, w_hs, w_cyb, w_kcb, w_cash,
    labels=["沪深300(含T1/T2红利盾牌) (%)", "创业板主线持仓 (2010.03起) (%)", "科创板主线持仓 (2019.08起) (%)", "防守现金 (%)"],
    colors=["#38bdf8", "#f59e0b", "#a855f7", "#334155"], alpha=0.85
)
ax3.set_ylim(0, 100)
ax3.set_ylabel("资金占比 (%)", color="#e2e8f0", fontsize=11)
ax3.set_xlabel("交易年份 (2006 ~ 2026)", color="#e2e8f0", fontsize=11)
ax3.set_title("20年三阶段历史演进仓位分布 (2006-2010纯沪深300 -> 2010创业板加入 -> 2019科创板加入)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.xaxis.set_major_locator(mdates.YearLocator(2))
leg3 = ax3.legend(loc="upper right", ncol=4, facecolor="#0f172a", edgecolor="#334155", fontsize=8.8)
for text in leg3.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_20y_pit.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)

# Print 20-year year-by-year table
m_33_m60 = {x["date"]: x for x in h_33_m60}
m_33_day = {x["date"]: x for x in h_33_day}
m_32 = {x["date"]: x for x in h_32_fix}
m_30 = {x["date"]: x for x in h_30_fix}
all_d = [x["date"] for x in h_33_m60]
targets = ["2006-10-30"] + [f"{y}-10-31" for y in range(2007, 2026)] + ["2026-10-29"]
cps = [[d for d in all_d if d <= t][-1] for t in targets]

for i in range(1, len(cps)):
    d0, d1 = cps[i-1], cps[i]
    br = (m_33_m60[d1]["hs300_level"] - m_33_m60[d0]["hs300_level"]) / m_33_m60[d0]["hs300_level"] * 100.0
    v_a0, v_a1 = m_33_m60[d0]["portfolio_value"], m_33_m60[d1]["portfolio_value"]
    v_b0, v_b1 = m_33_day[d0]["portfolio_value"], m_33_day[d1]["portfolio_value"]
    v_c0, v_c1 = m_32[d0]["portfolio_value"], m_32[d1]["portfolio_value"]
    v_d0, v_d1 = m_30[d0]["portfolio_value"], m_30[d1]["portfolio_value"]
    ra = (v_a1 - v_a0) / v_a0 * 100.0
    rb = (v_b1 - v_b0) / v_b0 * 100.0
    rc = (v_c1 - v_c0) / v_c0 * 100.0
    rd = (v_d1 - v_d0) / v_d0 * 100.0
    print(f"Y{i:02d} ({d0[:7]}..{d1[:7]}) | 沪深300:{br:+6.2f}% | V3.3(60m):¥{v_a1/1e4:7.1f}w({ra:+6.2f}%) | V3.3(日线):¥{v_b1/1e4:7.1f}w({rb:+6.2f}%) | V3.2(固25w):¥{v_c1/1e4:6.1f}w({rc:+6.2f}%) | V3.0(固12.5w):¥{v_d1/1e4:6.1f}w({rd:+6.2f}%)")
