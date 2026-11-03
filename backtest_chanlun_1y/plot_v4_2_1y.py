import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_v4_0_1y.json", "r", encoding="utf-8") as f:
    d40 = json.load(f)
with open(base / "backtest_v4_1_1y.json", "r", encoding="utf-8") as f:
    d41 = json.load(f)
with open(base / "backtest_v4_2_1y.json", "r", encoding="utf-8") as f:
    d42 = json.load(f)

r_42a = d42["V4.2-A 纯流动性版 (14:45物理隔离+月度缓冲+60m锁利 · 15%复利)"]
r_42c = d42["V4.2-C 实战双通道版 (14:45物理隔离+月度缓冲+60m锁利 · 15%复利)"]
r_42d = d42["V4.2-D 实战双通道版 (14:45物理隔离+月度缓冲+60m锁利 · 12%复利)"]
r_40 = d40["V4.0-A (零人工先验 · 15%净值动态复利)"]
r_41 = d41["V4.1-Dual (双通道主板+月度缓冲双创 · 12%复利)"]

h_42c = r_42c["history"]
dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_42c]
v_42c = [x["portfolio_value"] / 10000.0 for x in h_42c]
v_42a = [x["portfolio_value"] / 10000.0 for x in r_42a["history"]]
v_42d = [x["portfolio_value"] / 10000.0 for x in r_42d["history"]]
v_40 = [x["portfolio_value"] / 10000.0 for x in r_40["history"]]
v_41 = [x["portfolio_value"] / 10000.0 for x in r_41["history"]]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_42c]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9.4), gridspec_kw={"height_ratios": [1.7, 1.2]})
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

dt_crash_start = datetime.strptime("2026-07-02", "%Y-%m-%d")
dt_crash_end = datetime.strptime("2026-08-03", "%Y-%m-%d")
ax1.axvspan(dt_crash_start, dt_crash_end, color="#ef4444", alpha=0.14, label="[CRASH保护区] 7月高位急跌连续19个交易日 100% 空仓持币避险")
ax2.axvspan(dt_crash_start, dt_crash_end, color="#ef4444", alpha=0.18)

m_42c, m_42a, m_42d = r_42c["metrics"], r_42a["metrics"], r_42d["metrics"]
dates_40 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in r_40["history"]]
dates_41 = [datetime.strptime(x["date"], "%Y-%m-%d") for x in r_41["history"]]
ax1.plot(dates, v_42c, color="#10b981", linewidth=2.7, label=f"V4.2-C 实战双通道·15%复利 [ratio_factor真实手数校验] (期末 ¥{v_42c[-1]:.2f}万, {m_42c['total_return_pct']:+.2f}% | 回撤 -{m_42c['max_drawdown_pct']:.2f}% | 夏普 {m_42c['sharpe_ratio']:.2f} | 胜率 {m_42c['win_rate_pct']:.1f}%)")
ax1.plot(dates, v_42a, color="#f59e0b", linewidth=2.2, label=f"V4.2-A 纯流动性·15%复利 [ratio_factor真实手数校验] (期末 ¥{v_42a[-1]:.2f}万, {m_42a['total_return_pct']:+.2f}% | 胜率 {m_42a['win_rate_pct']:.1f}% | 盈亏比 {m_42a['profit_loss_ratio']:.2f})")
ax1.plot(dates, v_42d, color="#a855f7", linewidth=1.8, linestyle="-.", label=f"V4.2-D 实战双通道·12%复利 [ratio_factor真实手数校验] (期末 ¥{v_42d[-1]:.2f}万, {m_42d['total_return_pct']:+.2f}% | 回撤 -{m_42d['max_drawdown_pct']:.2f}% | 胜率 {m_42d['win_rate_pct']:.1f}%)")
ax1.plot(dates_40, v_40, color="#38bdf8", linewidth=1.7, linestyle="--", label=f"V4.0 基线 [DualChannel · 未做14:45物理隔离] (期末 ¥{v_40[-1]:.2f}万, +10.16% | 胜率 48.2% | 盈亏比 1.24)")
ax1.plot(dates_41, v_41, color="#6366f1", linewidth=1.4, linestyle=":", label=f"V4.1 实验版 [含纯Z-Score平滑状态机] (期末 ¥{v_41[-1]:.2f}万, -13.70%)")
ax1.plot(dates, v_bench, color="#cbd5e1", linewidth=1.6, linestyle=":", label=f"业绩基准：沪深300指数等值净值 (期末 ¥{v_bench[-1]:.2f}万, -14.18%)")
ax1.axhline(100.0, color="#64748b", linewidth=1)

ax1.annotate(
    "7月高位急跌触发 CRASH 锁死\n07-08~08-03 连续19日 100%空仓\n净值水平横住 ¥117.75万 躲过暴跌",
    xy=(datetime.strptime("2026-07-17", "%Y-%m-%d"), 117.75),
    xytext=(datetime.strptime("2026-07-05", "%Y-%m-%d"), 126.5),
    arrowprops=dict(facecolor="#f8fafc", edgecolor="#f8fafc", arrowstyle="->", lw=1.3),
    fontsize=9.0, fontweight="bold", color="#fef08a",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#ef4444", alpha=0.92)
)

ax1.set_title("ChanlunAIApp V4.2 实战定型版 (完成全部3轮独立审计修复 · 100%强制14:45物理隔离) 最近1年净值对比", color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8.8)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

hs_pct = np.array([x["hs300_mkt_val"] / x["portfolio_value"] * 100.0 for x in h_42c])
cyb_pct = np.array([x["cyb_mkt_val"] / x["portfolio_value"] * 100.0 for x in h_42c])
kcb_pct = np.array([x["kcb_mkt_val"] / x["portfolio_value"] * 100.0 for x in h_42c])
cash_pct = np.maximum(0.0, 100.0 - hs_pct - cyb_pct - kcb_pct)

ax2.stackplot(
    dates, hs_pct, cyb_pct, kcb_pct, cash_pct,
    labels=["主板/动态T1低波盾牌 (%)", "创业板月度缓冲池 (%)", "科创板月度缓冲池 (%)", "防守空仓现金 (%)"],
    colors=["#3b82f6", "#f59e0b", "#a855f7", "#64748b"], alpha=0.88
)
ax2.plot(dates, 100.0 - cash_pct, color="#f8fafc", linewidth=1.0, alpha=0.85, label="总股票仓位线 (%)")
ax2.text(
    datetime.strptime("2026-07-18", "%Y-%m-%d"), 50.0,
    "[7月 CRASH 保护]\n100% 现金空仓避险\n(股票仓位 = 0%)",
    ha="center", va="center", color="#fef08a", fontsize=9.0, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#ef4444", alpha=0.92)
)

ax2.set_ylim(0, 100)
ax2.set_ylabel("资金占比 (%)", color="#e2e8f0", fontsize=10.5)
ax2.set_title("V4.2-C (14:45物理隔离 · 15%复利) 动态仓位流向 (创业板净赚 +16.45万, 科创板净赚 +11.87万, 盈亏比 1.75)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg2 = ax2.legend(loc="upper right", ncol=5, facecolor="#0f172a", edgecolor="#334155", fontsize=8.8)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v4_2_1y.png"
plt.savefig(out_img, dpi=180)
print("Saved Round-3 audited chart to", out_img)
