import json
from pathlib import Path

base = Path(__file__).resolve().parent
with open(base / "backtest_results_v2_1y_20_50_30.json", "r", encoding="utf-8") as f:
    data = json.load(f)

v2 = data["v2_latest"]
subs = v2["sub_accounts"]
ct_map = {}
for sk, sname in [("hs300", "沪深300"), ("cyb", "创业板"), ("kcb", "科创板")]:
    for c in subs[sk]["closed_trades"]:
        if c["sell_date"] >= "2026-09-01":
            ct_map[(c["symbol"], c["sell_date"], c["shares"])] = (sname, c)

trades = [t for t in v2["trades"] if t["date"] >= "2026-09-01"]
trades.sort(key=lambda x: (x["date"], 0 if x["action"] == "SELL" else 1))

lines = [
    "# 最新V2模型 (20%沪深300 : 50%创业板 : 30%科创板 · 100万本金) 最近两个月逐笔交易全记录",
    "",
    "| # | 日期 | 环境 | 板块 | 方向 | 标的(代码) | 股数 | 成交价 | 成交金额 | 平仓净盈亏 (收益率) / 建仓止损与评分 | 缠论触发依据 |",
    "| :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |"
]

for idx, t in enumerate(trades, 1):
    dt, act, sym, nm = t["date"], t["action"], t["symbol"], t["name"]
    sh, pr, amt, fee, reg, rsn = t["shares"], t["price"], t["amount"], t["fee"], t["regime"], t["reason"]
    sec = "科创板" if (sym.startswith("sh688") or sym == "sh588000") else ("创业板" if (sym.startswith("sz30") or sym == "sz159915") else "沪深300")
    if act == "SELL":
        key = (sym, dt, sh)
        if key in ct_map:
            _, c = ct_map[key]
            bd, bp, pa, pp = c["buy_date"], c["buy_price"], c["pnl_amt"], c["pnl_pct"]
            info = f"**¥{pa:+,.2f} ({pp:+.2f}%)** [{bd}@¥{bp:.2f}]"
        else:
            info = "-"
        act_str = "**卖出**"
    else:
        sl = t.get("stop_loss", 0.0)
        sc = t.get("score", 0)
        info = f"评分:{sc} / 止损:¥{sl:.2f}"
        act_str = "买入"
    lines.append(f"| {idx:03d} | {dt} | `{reg}` | {sec} | {act_str} | {nm}(`{sym}`) | {sh} | ¥{pr:.2f} | ¥{amt:,.2f} | {info} | {rsn} |")

out_md = base / "trades_2026_09_10_v2.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("Saved", len(trades), "trades to", out_md)
