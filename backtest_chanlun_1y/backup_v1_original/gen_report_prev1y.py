import json
from pathlib import Path

base = Path(__file__).resolve().parent
with open(base / "backtest_results_prev1y.json", "r", encoding="utf-8") as f:
    data = json.load(f)

m1 = data["quant_hs300"]["metrics"]
m2 = data["quant_all_a"]["metrics"]
h1 = data["quant_hs300"]["history"]
h2 = data["quant_all_a"]["history"]

lines = [
    "# ChanlunAIApp 无AI介入双模拟账户 前1年回测报告 (2024-10-29 至 2025-10-29)",
    "",
    "## 一、核心回测绩效对比 (2024-10-29 ~ 2025-10-29)",
    "",
    "| 指标 | 账户①：无AI介入 · 仅限沪深300 | 账户②：无AI介入 · 全市场沪深A股(两级漏斗) | 沪深300基准 (sh000300) |",
    "| :--- | :---: | :---: | :---: |",
    f"| 回测区间 | {m1['start_date']} ~ {m1['end_date']} (244交易日) | {m2['start_date']} ~ {m2['end_date']} (244交易日) | 2024-10-29 ~ 2025-10-29 |",
    f"| 初始模拟资金 | ¥100,000.00 | ¥100,000.00 | {h1[0]['hs300_level']:.2f} 点 |",
    f"| 期末总净值 | **¥{m1['final_value']:,.2f}** | **¥{m2['final_value']:,.2f}** | {h1[-1]['hs300_level']:.2f} 点 |",
    f"| 累计收益率 | **{m1['total_return_pct']:+.2f}%** | **{m2['total_return_pct']:+.2f}%** | {m1['hs300_return_pct']:+.2f}% |",
    f"| 超额收益 (Alpha) | **{m1['alpha_pct']:+.2f}%** | **{m2['alpha_pct']:+.2f}%** | 0.00% |",
    f"| 最大回撤 (Max DD) | -{m1['max_drawdown_pct']:.2f}% ({m1['max_drawdown_date']}) | -{m2['max_drawdown_pct']:.2f}% ({m2['max_drawdown_date']}) | -14.26% |",
    f"| 夏普比率 (Sharpe) | {m1['sharpe_ratio']:.2f} | {m2['sharpe_ratio']:.2f} | - |",
    f"| 累计成交笔数 (买/卖) | {m1['total_orders']} 笔 ({m1['buy_orders']}买 / {m1['sell_orders']}卖) | {m2['total_orders']} 笔 ({m2['buy_orders']}买 / {m2['sell_orders']}卖) | - |",
    f"| 完整平仓胜率 | {m1['win_rate_pct']:.2f}% ({m1['win_count']}胜 / {m1['loss_count']}负) | {m2['win_rate_pct']:.2f}% ({m2['win_count']}胜 / {m2['loss_count']}负) | - |",
    f"| 平均盈利 / 平均亏损 | +{m1['avg_win_pct']:.2f}% / -{m1['avg_loss_pct']:.2f}% | +{m2['avg_win_pct']:.2f}% / -{m2['avg_loss_pct']:.2f}% | - |",
    f"| 盈亏比 (P/L Ratio) | {m1['profit_loss_ratio']:.2f} | {m2['profit_loss_ratio']:.2f} | - |",
    "",
    "## 二、月末净值与累计收益轨迹 (2024-10 ~ 2025-10)",
    "",
    "| 月份 (月末交易日) | 沪深300点位 (累计涨跌) | 账户① 净值 (累计收益率 / Alpha) | 账户② 净值 (累计收益率 / Alpha) |",
    "| :---: | :---: | :---: | :---: |"
]

month_end_indices = []
for i in range(len(h1)):
    cur_m = h1[i]["date"][:7]
    nxt_m = h1[i + 1]["date"][:7] if i + 1 < len(h1) else None
    if cur_m != nxt_m:
        month_end_indices.append(i)

for idx in month_end_indices:
    r1 = h1[idx]
    r2 = h2[idx]
    lines.append(
        f"| {r1['date']} | {r1['hs300_level']:.2f} ({r1['hs300_return_pct']:+.2f}%) | "
        f"¥{r1['portfolio_value']:,.2f} ({r1['portfolio_return_pct']:+.2f}% / {r1['alpha_pct']:+.2f}%) | "
        f"¥{r2['portfolio_value']:,.2f} ({r2['portfolio_return_pct']:+.2f}% / {r2['alpha_pct']:+.2f}%) |"
    )

with open(base / "backtest_report_prev1y.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

for k in ["quant_hs300", "quant_all_a"]:
    ct = data[k]["closed_trades"]
    s1 = [t for t in ct if "顶背驰一卖" in t["reason"]]
    dd = [t for t in ct if "跌破15m中枢" in t["reason"]]
    s1_w = sum(1 for t in s1 if t["pnl_amt"] > 0)
    dd_w = sum(1 for t in dd if t["pnl_amt"] > 0)
    s1_avg = sum(t["pnl_pct"] for t in s1) / len(s1) if s1 else 0
    dd_avg = sum(t["pnl_pct"] for t in dd) / len(dd) if dd else 0
    top3 = sorted(ct, key=lambda x: x["pnl_amt"], reverse=True)[:3]
    bot3 = sorted(ct, key=lambda x: x["pnl_amt"])[:3]
    print(f"=== {k} ===")
    print(f"  15m顶背驰一卖: {len(s1)}笔, 胜率={s1_w/len(s1)*100:.1f}%, 均益={s1_avg:+.2f}%")
    print(f"  跌破15m中枢DD止损: {len(dd)}笔, 胜率={dd_w/len(dd)*100:.1f}%, 均益={dd_avg:+.2f}%")
    print("  Top 3:", [(t["name"], t["buy_date"], t["sell_date"], t["pnl_pct"], t["pnl_amt"]) for t in top3])
    print("  Bot 3:", [(t["name"], t["buy_date"], t["sell_date"], t["pnl_pct"], t["pnl_amt"]) for t in bot3])
