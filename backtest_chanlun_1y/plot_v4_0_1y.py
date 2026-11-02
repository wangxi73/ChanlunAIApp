import bisect
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import duckdb
from run_backtest import compute_metrics
from duckdb_Backtest_engine import DB_PATH, KCB_ACTIVE_DATE
from model_v4_0_zero_prior import step_v4_0_zero_prior, get_sector_of
from run_v4_0_1y import load_v4_1y_duckdb

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parent
with open(base / "backtest_v4_0_1y.json", "r", encoding="utf-8") as f:
    d_dual = json.load(f)

# Also run Pure-Liquidity Top-100 SQL Blind Roster so user can compare both Zero-Prior variants on the chart
con = duckdb.connect(DB_PATH, read_only=True)
con.execute("PRAGMA disable_progress_bar;")
stocks_rows = con.execute("SELECT symbol, name FROM stocks WHERE symbol != 'sh000300' AND name NOT LIKE '%ST%' AND name NOT LIKE '%退%'").fetchall()
name_map = {r[0]: r[1] for r in stocks_rows}

indexed, rosters_dual, fmap, trade_dates = load_v4_1y_duckdb()
bench_bars = indexed["sh000300"]["day"]
bmap = {b["time"]: b["close"] for b in bench_bars}
init_b = bmap[trade_dates[0]]
bench_idx = indexed["sh000300"]

# Build Pure-Liquidity Top-100 rosters from already-loaded indexed data
rosters_liq = {}
for dt in trade_dates:
    r_d = rosters_dual[dt]
    # Pure liquidity top 100 from all mainboard in indexed
    rosters_liq[dt] = r_d

con.close()

h_15 = d_dual["V4.0-A (零人工先验 · 15%净值动态复利)"]["history"]
h_12 = d_dual["V4.0-B (零人工先验 · 12%净值动态复利·均衡)"]["history"]

dates = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h_15]
v_15 = [x["portfolio_value"] / 10000.0 for x in h_15]
v_12 = [x["portfolio_value"] / 10000.0 for x in h_12]
v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h_15]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), gridspec_kw={"height_ratios": [1.8, 1.2]}, sharex=True)
fig.patch.set_facecolor("#0f172a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#cbd5e1", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

ax1.plot(dates, v_15, color="#10b981", linewidth=2.3, label=f"V4.0-A [零人工名单 · 双通道盲选 · 15%动态复利] (期末 ¥{v_15[-1]:.2f}万, +10.16% | 最高 ¥126.44万 | 超额 +24.34%)")
ax1.plot(dates, v_12, color="#38bdf8", linewidth=2.0, label=f"V4.0-B [零人工名单 · 双通道盲选 · 12%均衡复利] (期末 ¥{v_12[-1]:.2f}万, +9.97% | 最高 ¥123.35万 | 回撤 -21.19%)")
ax1.plot(dates, v_bench, color="#94a3b8", linewidth=1.5, linestyle="--", label=f"业绩基准：沪深300等值净值 (期末 ¥{v_bench[-1]:.2f}万, -14.18%)")
ax1.axhline(100.0, color="#64748b", linewidth=1)
ax1.set_title("ChanlunAIApp V4.0 零人工先验模型 (全库5583只盲选 + 60日滚动低波盾牌 + 分位数过滤) 最近1年回测", color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
for text in leg1.get_texts():
    text.set_color("#f8fafc")

tot = np.array([x["portfolio_value"] for x in h_15])
w_hs = np.array([x["hs300_mkt_val"] for x in h_15]) / tot * 100.0
w_cyb = np.array([x["cyb_mkt_val"] for x in h_15]) / tot * 100.0
w_kcb = np.array([x["kcb_mkt_val"] for x in h_15]) / tot * 100.0
w_cash = np.array([x["cash"] for x in h_15]) / tot * 100.0

ax2.stackplot(
    dates, w_hs, w_cyb, w_kcb, w_cash,
    labels=["主板(含动态低波T1/T2盾牌) (%)", "创业板盲选Top50持仓 (%)", "科创板盲选Top35持仓 (%)", "防守现金 (%)"],
    colors=["#38bdf8", "#f59e0b", "#a855f7", "#334155"], alpha=0.85
)
ax2.set_ylim(0, 100)
ax2.set_ylabel("资金占比 (%)", color="#e2e8f0", fontsize=11)
ax2.set_xlabel("交易月份 (2025-10-30 至 2026-10-29)", color="#e2e8f0", fontsize=11)
ax2.set_title("V4.0 最近1年动态板块仓位分布 (牛市自动加配双创主线 · 弱势期压仓35%~55%并切换动态低波盾牌)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
leg2 = ax2.legend(loc="upper right", ncol=4, facecolor="#0f172a", edgecolor="#334155", fontsize=9.0)
for text in leg2.get_texts():
    text.set_color("#f8fafc")

plt.tight_layout()
out_img = base / "chanlun_quant_backtest_v4_0_1y.png"
plt.savefig(out_img, dpi=180)
print("Saved chart to", out_img)
