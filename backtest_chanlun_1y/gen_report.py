import json
from pathlib import Path

base = Path(__file__).resolve().parent
with open(base / "backtest_results_1y.json", "r", encoding="utf-8") as f:
    data = json.load(f)

m1 = data["quant_hs300"]["metrics"]
m2 = data["quant_all_a"]["metrics"]

lines = [
    "# ChanlunAIApp 无AI介入双模拟账户 1年期回测报告 (2025-10-29 至 2026-10-29)",
    "",
    "## 一、核心回测绩效对比",
    "",
    "| 指标 | 账户①：无AI介入 · 仅限沪深300 | 账户②：无AI介入 · 全市场沪深A股(两级漏斗) | 沪深300基准 (sh000300) |",
    "| :--- | :---: | :---: | :---: |",
    f"| 回测区间 | {m1['start_date']} ~ {m1['end_date']} (243交易日) | {m2['start_date']} ~ {m2['end_date']} (243交易日) | 2025-10-29 ~ 2026-10-29 |",
    f"| 初始模拟资金 | ¥100,000.00 | ¥100,000.00 | 4747.84 点 |",
    f"| 期末总净值 | **¥{m1['final_value']:,.2f}** | **¥{m2['final_value']:,.2f}** | 4041.84 点 |",
    f"| 累计收益率 | **{m1['total_return_pct']:+.2f}%** | **{m2['total_return_pct']:+.2f}%** | {m1['hs300_return_pct']:+.2f}% |",
    f"| 超额收益 (Alpha) | **{m1['alpha_pct']:+.2f}%** | **{m2['alpha_pct']:+.2f}%** | 0.00% |",
    f"| 最大回撤 (Max DD) | -{m1['max_drawdown_pct']:.2f}% ({m1['max_drawdown_date']}) | -{m2['max_drawdown_pct']:.2f}% ({m2['max_drawdown_date']}) | -18.42% |",
    f"| 夏普比率 (Sharpe) | {m1['sharpe_ratio']:.2f} | {m2['sharpe_ratio']:.2f} | - |",
    f"| 累计成交笔数 (买/卖) | {m1['total_orders']} 笔 ({m1['buy_orders']}买 / {m1['sell_orders']}卖) | {m2['total_orders']} 笔 ({m2['buy_orders']}买 / {m2['sell_orders']}卖) | - |",
    f"| 完整平仓胜率 | {m1['win_rate_pct']:.2f}% ({m1['win_count']}胜 / {m1['loss_count']}负) | {m2['win_rate_pct']:.2f}% ({m2['win_count']}胜 / {m2['loss_count']}负) | - |",
    f"| 平均盈利 / 平均亏损 | +{m1['avg_win_pct']:.2f}% / -{m1['avg_loss_pct']:.2f}% | +{m2['avg_win_pct']:.2f}% / -{m2['avg_loss_pct']:.2f}% | - |",
    f"| 盈亏比 (P/L Ratio) | {m1['profit_loss_ratio']:.2f} | {m2['profit_loss_ratio']:.2f} | - |",
    "",
    "## 二、月末净值与累计收益轨迹",
    "",
    "| 月份 (月末交易日) | 沪深300点位 (累计涨跌) | 账户① 净值 (累计收益率 / Alpha) | 账户② 净值 (累计收益率 / Alpha) |",
    "| :---: | :---: | :---: | :---: |"
]

h1 = data["quant_hs300"]["history"]
h2 = data["quant_all_a"]["history"]
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

with open(base / "backtest_report_1y.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("Generated markdown report successfully.")
