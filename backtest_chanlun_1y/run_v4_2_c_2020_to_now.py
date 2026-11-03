import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from run_backtest import compute_metrics
from duckdb_Backtest_engine import DB_PATH
from model_v4_2_production import (
    build_v4_1_asymmetric_buffered_rosters,
    step_v4_2_production,
    get_sector_of
)

base = Path(__file__).resolve().parent


def load_v4_2_c_2020_duckdb():
    t0 = time.time()
    # 使用内存表缓存 2019-01-01 以来日线，加速 83 个月度/半年度调样点的滚动 SQL 计算
    mem_con = duckdb.connect(":memory:")
    mem_con.execute("PRAGMA disable_progress_bar;")
    mem_con.execute(f"ATTACH '{DB_PATH}' AS lexar (READ_ONLY);")
    print("1. Caching 2019~2026 daily K-lines into memory for 83-month rolling roster construction...")
    mem_con.execute("CREATE TABLE stocks AS SELECT * FROM lexar.stocks;")
    mem_con.execute("""
        CREATE TABLE kline_1d AS
        SELECT symbol, date, open, high, low, close, volume, amount, hfq_factor, ratio_factor
        FROM lexar.kline_1d
        WHERE date >= '2019-01-01' AND close > 0;
    """)

    bench_rows = mem_con.execute("""
        SELECT date, open, high, low, close, volume, amount
        FROM kline_1d
        WHERE symbol = 'sh000300' AND date >= '2019-01-01' AND close > 0
        ORDER BY date
    """).fetchall()
    all_dates = [r[0] for r in bench_rows]
    trade_dates = [d for d in all_dates if d >= "2020-01-02"]

    print(f"2. Building V4.2-C Asymmetric Buffered Rosters across {len(trade_dates)} trading days ({trade_dates[0]} ~ {trade_dates[-1]})...")
    rosters_by_date, needed_syms, full_meta_map = build_v4_1_asymmetric_buffered_rosters(
        mem_con, trade_dates, use_dual_channel_hs=True
    )

    sym_list_sql = ",".join(f"'{s}'" for s in sorted(needed_syms))
    print(f"3. Loading 1d/60m/15m/5m K-lines for {len(needed_syms)} symbols ({time.time()-t0:.1f}s)...")

    rows_1d = mem_con.execute(f"""
        SELECT symbol, date, open, high, low, close, volume, amount, hfq_factor, ratio_factor
        FROM kline_1d
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-01-01' AND close > 0
        ORDER BY symbol, date
    """).fetchall()

    bdom_rows = mem_con.execute(f"""
        SELECT symbol, date,
               SUM(volume * CASE WHEN high > low THEN (close - low) / (high - low) WHEN close >= open THEN 0.55 ELSE 0.45 END) AS b_vol,
               SUM(volume) AS tot_vol
        FROM lexar.kline_15m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-11-01'
        GROUP BY symbol, date
    """).fetchall()
    bdom_map = {(s, d): round((2.0 * bv - tv) / tv * 100.0, 1) for s, d, bv, tv in bdom_rows if tv and tv > 0}

    rows_60m = mem_con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume
        FROM lexar.kline_60m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-06-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()

    rows_15m = mem_con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume, amount
        FROM lexar.kline_15m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-11-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()

    rows_5m_1450 = mem_con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume, amount
        FROM lexar.kline_5m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2020-01-01'
          AND (datetime LIKE '%14:50' OR datetime LIKE '%14:50:00') AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()
    mem_con.close()

    indexed = {s: {
        "day": [], "day_dates": [], "day_by_date": {},
        "m60_tuples": [], "m60_dates": [],
        "m15_tuples": [], "m15_dates": [],
        "m5_1450_by_date": {},
        "snaps": {}
    } for s in needed_syms}

    for sym, dt, o, h, l, c, v, amt, hfq_f, rf_f in rows_1d:
        e = indexed[sym]
        prev_c = e["day"][-1]["close"] if e["day"] else o
        pct = ((c - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
        true_amt = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0) else amt
        rf = float(rf_f) if (rf_f and rf_f > 0) else 1.0
        raw_p = round(c / rf, 4)
        bar = {
            "symbol": sym, "time": dt, "open": o, "high": h, "low": l, "close": c,
            "ratio_factor": rf, "raw_price": raw_p, "volume": v, "amount": true_amt,
            "preClose": prev_c, "pctChange": pct
        }
        e["day"].append(bar)
        e["day_dates"].append(dt)
        e["day_by_date"][dt] = bar
        if sym != "sh000300":
            amt_yi = round((true_amt if true_amt and true_amt > 0 else v * c) / 1e8, 2)
            e["snaps"][dt] = {
                "symbol": sym, "lastPrice": c, "preClose": prev_c,
                "pctChange": round(pct, 2), "amountYi": amt_yi,
                "buyerDom": bdom_map.get((sym, dt), 0.0),
            }

    for sym, dt, dtm, o, h, l, c, v in rows_60m:
        e = indexed[sym]
        e["m60_tuples"].append((dtm, o, h, l, c, v))
        e["m60_dates"].append(dt)

    for sym, dt, dtm, o, h, l, c, v, amt in rows_15m:
        e = indexed[sym]
        true_amt15 = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0 and amt and amt > v * c * 10) else amt
        e["m15_tuples"].append((dtm, o, h, l, c, v, true_amt15))
        e["m15_dates"].append(dt)

    for sym, dt, dtm, o, h, l, c, v, amt in rows_5m_1450:
        e = indexed[sym]
        true_amt5 = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0 and amt and amt > v * c * 10) else amt
        e["m5_1450_by_date"][dt] = (dtm, o, h, l, c, v, true_amt5)

    print(f"4. Loaded full 2020~2026 dataset ({len(needed_syms)} symbols) in {time.time() - t0:.2f}s!")
    return indexed, rosters_by_date, full_meta_map, trade_dates


def main():
    indexed, rosters_by_date, fmap, trade_dates = load_v4_2_c_2020_duckdb()
    bench_idx = indexed["sh000300"]
    bmap = {b["time"]: b["close"] for b in bench_idx["day"]}
    init_b = bmap[trade_dates[0]]

    st = {
        "cash": 1000000.0, "holdings": {}, "trades": [], "closed_trades": [],
        "equity": 1000000.0, "regime_history": [],
        "stats": {"limit_up_blocked": 0, "limit_down_two_step": 0, "fx_protect_blocked": 0}
    }
    hist = []
    t_sim = time.time()
    for idx_d, dt in enumerate(trade_dates):
        # 每日独立 eval_cache，避免 1650+ 交易日累积切片占满内存
        day_eval_cache = {}
        step_v4_2_production(
            st, rosters_by_date[dt], fmap, indexed, bench_idx, dt, day_eval_cache,
            dyn_sizing_pct=0.15, strict_1445_isolation=True
        )
        hs_v = sum(h["shares"] * h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s) == "HS300")
        cyb_v = sum(h["shares"] * h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s) == "CYB")
        kcb_v = sum(h["shares"] * h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s) == "KCB")
        tot = st["equity"]
        tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        hist.append({
            "date": dt, "portfolio_value": tot, "cash": round(st["cash"], 2),
            "hs300_mkt_val": round(hs_v, 2), "cyb_mkt_val": round(cyb_v, 2), "kcb_mkt_val": round(kcb_v, 2),
            "holdings_count": len(st["holdings"]), "regime_label": st["regime_history"][-1][1],
            "portfolio_return_pct": tot_r, "hs300_level": bmap[dt],
            "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)
        })
        if (idx_d + 1) % 250 == 0 or idx_d == len(trade_dates) - 1:
            print(f"  Progress: {idx_d+1}/{len(trade_dates)} ({dt}) | Equity: ¥{tot:,.2f} ({tot_r:+.2f}%) | HS300: {b_r:+.2f}% | Elapsed: {time.time()-t_sim:.1f}s")

    sec_stats = {"沪深300": [0, 0.0], "创业板": [0, 0.0], "科创板": [0, 0.0]}
    exit_reasons = {"主升浪减半锁利": 0, "半仓移动止盈": 0, "防守锁死止损": 0, "冻结/结构破位止损": 0, "顶背驰一卖": 0, "早盘两步法": 0}
    for c in st["closed_trades"]:
        sec_lbl = {"HS300": "沪深300", "CYB": "创业板", "KCB": "科创板"}[get_sector_of(c["symbol"])]
        sec_stats[sec_lbl][0] += 1
        sec_stats[sec_lbl][1] = round(sec_stats[sec_lbl][1] + c["pnl_amt"], 2)
        rsn = c.get("reason", "")
        for k in exit_reasons:
            if k in rsn:
                exit_reasons[k] += 1

    res = {
        "account_name": "V4.2-C 实战双通道版 (2020~2026全周期 · 14:45隔离+ratio_factor真实手数 · 15%复利)",
        "initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"],
        "final_cash": round(st["cash"], 2), "final_holdings": st["holdings"],
        "sector_pnl_attribution": sec_stats, "exit_reasons": exit_reasons, "stats": st["stats"],
        "history": hist, "trades": st["trades"], "closed_trades": st["closed_trades"]
    }
    m = compute_metrics(res)
    pk = max(hist, key=lambda x: x["portfolio_value"])
    tr = min(hist, key=lambda x: x["portfolio_value"])
    m["peak_value"] = pk["portfolio_value"]; m["peak_return_pct"] = pk["portfolio_return_pct"]; m["peak_date"] = pk["date"]
    m["trough_value"] = tr["portfolio_value"]; m["trough_return_pct"] = tr["portfolio_return_pct"]; m["trough_date"] = tr["date"]
    years = len(trade_dates) / 243.0
    cagr = ((m["final_value"] / 1e6) ** (1.0 / years) - 1.0) * 100.0
    m["cagr_pct"] = round(cagr, 2)
    m["calmar_ratio"] = round(cagr / max(0.01, m["max_drawdown_pct"]), 2)
    res["metrics"] = m

    # 逐年分解
    by_year_hist = defaultdict(list)
    for h in hist:
        by_year_hist[h["date"][:4]].append(h)

    by_year_sec = defaultdict(lambda: {"HS300": [0, 0.0], "CYB": [0, 0.0], "KCB": [0, 0.0]})
    for c in st["closed_trades"]:
        yr = c["sell_date"][:4]
        sec = get_sector_of(c["symbol"])
        by_year_sec[yr][sec][0] += 1
        by_year_sec[yr][sec][1] += c["pnl_amt"]

    yearly_summary = {}
    prev_eq = 1e6
    prev_b = init_b
    print("\n================ 2020 ~ 2026 V4.2-C 逐年绩效与板块盈亏归因 ================")
    for yr in sorted(by_year_hist.keys()):
        yh = by_year_hist[yr]
        end_eq = yh[-1]["portfolio_value"]
        end_b = yh[-1]["hs300_level"]
        yr_ret = (end_eq - prev_eq) / prev_eq * 100.0
        b_ret = (end_b - prev_b) / prev_b * 100.0
        alpha = yr_ret - b_ret
        # 年内最大回撤
        run_pk = prev_eq
        max_dd_yr = 0.0
        for x in yh:
            v = x["portfolio_value"]
            if v > run_pk:
                run_pk = v
            dd = (run_pk - v) / run_pk * 100.0
            if dd > max_dd_yr:
                max_dd_yr = dd
        sc = by_year_sec[yr]
        yearly_summary[yr] = {
            "start_equity": round(prev_eq, 2), "end_equity": round(end_eq, 2),
            "return_pct": round(yr_ret, 2), "hs300_return_pct": round(b_ret, 2),
            "alpha_pct": round(alpha, 2), "max_drawdown_pct": round(max_dd_yr, 2),
            "hs_trades": sc["HS300"][0], "hs_pnl": round(sc["HS300"][1], 2),
            "cyb_trades": sc["CYB"][0], "cyb_pnl": round(sc["CYB"][1], 2),
            "kcb_trades": sc["KCB"][0], "kcb_pnl": round(sc["KCB"][1], 2),
        }
        print(
            f"  {yr}: 净值 ¥{prev_eq/1e4:6.2f}万 -> ¥{end_eq/1e4:6.2f}万 | 收益:{yr_ret:+6.2f}% (沪深300:{b_ret:+6.2f}%, Alpha:{alpha:+6.2f}%) | 年内回撤:-{max_dd_yr:5.2f}% | "
            f"主板:{sc['HS300'][1]:+9,.0f}({sc['HS300'][0]}笔) 创业板:{sc['CYB'][1]:+9,.0f}({sc['CYB'][0]}笔) 科创板:{sc['KCB'][1]:+9,.0f}({sc['KCB'][0]}笔)"
        )
        prev_eq = end_eq
        prev_b = end_b

    res["yearly_summary"] = yearly_summary

    print(f"\n================ {res['account_name']} 全区间汇总 ================")
    print(f"  回测区间  : {trade_dates[0]} ~ {trade_dates[-1]} ({len(trade_dates)} 交易日, 约 {years:.2f} 年)")
    print(f"  期末总净值: ¥{m['final_value']:,.2f} (累计 {m['total_return_pct']:+.2f}%, 年化 CAGR {m['cagr_pct']:+.2f}%)")
    print(f"  同期基准  : 沪深300 {m['hs300_return_pct']:+.2f}% | 超额 Alpha: {m['alpha_pct']:+.2f}%")
    print(f"  最高净值  : ¥{m['peak_value']:,.2f} ({m['peak_return_pct']:+.2f}%, {m['peak_date']}) | 最低净值: ¥{m['trough_value']:,.2f} ({m['trough_date']})")
    print(f"  最大回撤  : -{m['max_drawdown_pct']:.2f}% ({m['max_drawdown_date']}) | 夏普比率: {m['sharpe_ratio']:.2f} | 卡玛比率: {m['calmar_ratio']:.2f}")
    print(f"  交易统计  : 总平仓 {len(st['closed_trades'])} 笔 | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f} (均盈{m['avg_win_pct']:+.2f}% / 均亏{m['avg_loss_pct']:+.2f}%)")
    print(f"  板块总贡献: {sec_stats}")
    print(f"  出场归因  : {exit_reasons} | 微观拦截: {st['stats']}")

    with open(base / "backtest_v4_2_c_2020_2026.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # 绘制 2020 ~ 2026 净值曲线与仓位流向图
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    dates_dt = [datetime.strptime(x["date"], "%Y-%m-%d") for x in hist]
    v_port = [x["portfolio_value"] / 10000.0 for x in hist]
    v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in hist]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9.6), gridspec_kw={"height_ratios": [1.75, 1.2]})
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

    ax1.plot(dates_dt, v_port, color="#10b981", linewidth=2.4,
             label=f"V4.2-C 实战双通道·15%复利 (期末 ¥{v_port[-1]:.2f}万, {m['total_return_pct']:+.2f}% | 年化 {m['cagr_pct']:+.2f}% | 最大回撤 -{m['max_drawdown_pct']:.2f}% | 夏普 {m['sharpe_ratio']:.2f})")
    ax1.plot(dates_dt, v_bench, color="#cbd5e1", linewidth=1.6, linestyle=":",
             label=f"业绩基准：沪深300指数等值净值 (期末 ¥{v_bench[-1]:.2f}万, {m['hs300_return_pct']:+.2f}% | 超额 Alpha {m['alpha_pct']:+.2f}%)")
    ax1.axhline(100.0, color="#64748b", linewidth=1)

    ax1.set_title(f"ChanlunAIApp V4.2-C 实战双通道版 2020~2026 全周期实盘级回测 ({trade_dates[0]} ~ {trade_dates[-1]}, {len(trade_dates)} 交易日)",
                  color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=9.5)
    for text in leg1.get_texts():
        text.set_color("#f8fafc")

    hs_pct = np.array([x["hs300_mkt_val"] / x["portfolio_value"] * 100.0 for x in hist])
    cyb_pct = np.array([x["cyb_mkt_val"] / x["portfolio_value"] * 100.0 for x in hist])
    kcb_pct = np.array([x["kcb_mkt_val"] / x["portfolio_value"] * 100.0 for x in hist])
    cash_pct = np.maximum(0.0, 100.0 - hs_pct - cyb_pct - kcb_pct)

    ax2.stackplot(
        dates_dt, hs_pct, cyb_pct, kcb_pct, cash_pct,
        labels=["主板/动态T1低波盾牌 (%)", "创业板月度缓冲池 (%)", "科创板月度缓冲池 (%)", "防守空仓现金 (%)"],
        colors=["#3b82f6", "#f59e0b", "#a855f7", "#64748b"], alpha=0.88
    )
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("资金占比 (%)", color="#e2e8f0", fontsize=10.5)
    ax2.set_title("2020~2026 V4.2-C 三大通道动态仓位与空仓防守流向图", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    leg2 = ax2.legend(loc="upper right", ncol=4, facecolor="#0f172a", edgecolor="#334155", fontsize=8.8)
    for text in leg2.get_texts():
        text.set_color("#f8fafc")

    plt.tight_layout()
    out_img = base / "chanlun_v4_2_c_2020_2026.png"
    plt.savefig(out_img, dpi=180)
    print(f"Saved 2020~2026 chart to {out_img}")


if __name__ == "__main__":
    main()
