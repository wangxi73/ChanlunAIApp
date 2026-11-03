import bisect
import json
import time
from pathlib import Path
import duckdb
from run_backtest import compute_metrics
from duckdb_Backtest_engine import DB_PATH
from model_v4_0_zero_prior import (
    build_v4_sql_blind_rosters, step_v4_0_zero_prior,
    get_sector_of, compute_dynamic_defensive_tiers
)

base = Path(__file__).resolve().parent

def load_v4_1y_duckdb():
    t0 = time.time()
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA disable_progress_bar;")

    bench_rows = con.execute("""
        SELECT date, open, high, low, close, volume, amount
        FROM kline_1d
        WHERE symbol = 'sh000300' AND date >= '2024-06-01' AND close > 0
        ORDER BY date
    """).fetchall()
    all_dates = [r[0] for r in bench_rows]
    trade_dates = [d for d in all_dates if d >= "2025-10-30"]

    # Find semi-annual rebalance dates for 1Y (initial date + end of Dec 2025 + end of Jun 2026)
    reb_dates = [trade_dates[0]]
    for i in range(len(trade_dates) - 1):
        cur_ym, nxt_ym = trade_dates[i][:7], trade_dates[i + 1][:7]
        if cur_ym != nxt_ym and (cur_ym.endswith("-06") or cur_ym.endswith("-12")):
            reb_dates.append(trade_dates[i])

    print(f"1. Running SQL blind selection across all 5,583 stocks at rebalance dates {reb_dates}...")
    rosters_at_reb, needed_syms, full_meta_map = build_v4_sql_blind_rosters(con, reb_dates)

    rosters_by_date = {}
    cur_r = rosters_at_reb[trade_dates[0]]
    for d in trade_dates:
        rosters_by_date[d] = cur_r
        if d in rosters_at_reb:
            cur_r = rosters_at_reb[d]

    sym_list_sql = ",".join(f"'{s}'" for s in sorted(needed_syms))
    print(f"2. Loading 1d/60m/15m K-lines from DuckDB for {len(needed_syms)} SQL-selected symbols ({time.time()-t0:.1f}s)...")

    bdom_rows = con.execute(f"""
        SELECT symbol, date,
               SUM(volume * CASE WHEN high > low THEN (close - low)/(high - low) WHEN close >= open THEN 0.55 ELSE 0.45 END) AS b_vol,
               SUM(volume) AS tot_vol
        FROM kline_15m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2024-06-01'
        GROUP BY symbol, date
    """).fetchall()
    bdom_map = {(sym, dt): round((2.0 * bv - tv) / tv * 100.0, 1) for sym, dt, bv, tv in bdom_rows if tv and tv > 0}

    rows_1d = con.execute(f"""
        SELECT symbol, date, open, high, low, close, volume, amount, ratio_factor
        FROM kline_1d
        WHERE symbol IN ({sym_list_sql}) AND date >= '2024-06-01' AND close > 0
        ORDER BY symbol, date
    """).fetchall()

    rows_60m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume
        FROM kline_60m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2025-01-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()

    rows_15m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume
        FROM kline_15m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2025-08-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()
    con.close()

    indexed = {s: {
        "day": [], "day_dates": [], "day_by_date": {},
        "m60_tuples": [], "m60_dates": [],
        "m15_tuples": [], "m15_dates": [],
        "snaps": {}
    } for s in needed_syms}

    for sym, dt, o, h, l, c, v, amt, rf_f in rows_1d:
        e = indexed[sym]
        prev_c = e["day"][-1]["close"] if e["day"] else o
        pct = ((c - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
        rf = float(rf_f) if (rf_f and rf_f > 0) else 1.0
        bar = {"symbol": sym, "time": dt, "open": o, "high": h, "low": l, "close": c, "ratio_factor": rf, "raw_price": round(c / rf, 4), "volume": v, "amount": amt, "preClose": prev_c, "pctChange": pct}
        e["day"].append(bar)
        e["day_dates"].append(dt)
        e["day_by_date"][dt] = bar
        if sym != "sh000300":
            true_amt = (amt / 100.0) if (sym.startswith('sh688') and int(v) % 100 == 0) else amt
            amt_yi = (true_amt / 1e8) if (true_amt and true_amt > 0) else ((v * c) / 1e8)
            e["snaps"][dt] = {
                "symbol": sym, "lastPrice": c, "preClose": prev_c,
                "pctChange": round(pct, 2), "amountYi": round(amt_yi, 2),
                "buyerDom": bdom_map.get((sym, dt), 0.0)
            }

    for sym, dt, dtm, o, h, l, c, v in rows_60m:
        e = indexed[sym]
        e["m60_tuples"].append((dtm, o, h, l, c, v))
        e["m60_dates"].append(dt)

    for sym, dt, dtm, o, h, l, c, v in rows_15m:
        e = indexed[sym]
        e["m15_tuples"].append((dtm, o, h, l, c, v))
        e["m15_dates"].append(dt)

    print(f"Loaded V4.0 1Y DuckDB dataset in {time.time() - t0:.2f}s!")
    return indexed, rosters_by_date, full_meta_map, trade_dates

if __name__ == "__main__":
    indexed, rosters_by_date, fmap, trade_dates = load_v4_1y_duckdb()
    bench_bars = indexed["sh000300"]["day"]
    bmap = {b["time"]: b["close"] for b in bench_bars}
    init_b = bmap[trade_dates[0]]
    bench_idx = indexed["sh000300"]

    # Show sample blind-selected rosters & dynamic T1 shield on latest date
    last_dt = trade_dates[-1]
    r_last = rosters_by_date[last_dt]
    b_pos_last = bisect.bisect_right(bench_idx["day_dates"], last_dt)
    sl_last = bench_idx["day"][max(0, b_pos_last - 250):b_pos_last]
    dyn_t1, dyn_t2 = compute_dynamic_defensive_tiers(r_last["hs"], indexed, sl_last, last_dt)
    t1_names = [fmap[s][1] for s in list(dyn_t1)[:12] if s in fmap]
    print(f"[{last_dt}] SQL盲选主板Top10: {[m[1] for m in r_last['hs'][1:11]]}")
    print(f"[{last_dt}] SQL盲选创业板Top8: {[m[1] for m in r_last['cyb'][1:9]]}")
    print(f"[{last_dt}] SQL盲选科创板Top8: {[m[1] for m in r_last['kcb'][1:9]]}")
    print(f"[{last_dt}] 60日滚动低波抗跌自动认领的动态T1盾牌(前12): {t1_names}")

    eval_cache = {}
    out_1y = {}
    for v_name, dyn_pct in [
        ("V4.0-A (零人工先验 · 15%净值动态复利)", 0.15),
        ("V4.0-B (零人工先验 · 12%净值动态复利·均衡)", 0.12),
    ]:
        st = {"cash": 1000000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 1000000.0}
        hist = []
        for dt in trade_dates:
            b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
            sl = bench_idx["day"][max(0, b_pos - 250):b_pos]
            step_v4_0_zero_prior(st, rosters_by_date[dt], fmap, indexed, sl, dt, eval_cache, dyn_sizing_pct=dyn_pct)

            hs_v = sum(h["shares"]*h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s)=="HS300")
            cyb_v = sum(h["shares"]*h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s)=="CYB")
            kcb_v = sum(h["shares"]*h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s)=="KCB")
            tot = st["equity"]
            tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
            b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
            hist.append({
                "date": dt, "portfolio_value": tot, "cash": round(st["cash"], 2),
                "hs300_mkt_val": round(hs_v, 2), "cyb_mkt_val": round(cyb_v, 2), "kcb_mkt_val": round(kcb_v, 2),
                "holdings_count": len(st["holdings"]),
                "portfolio_return_pct": tot_r, "hs300_level": bmap[dt],
                "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)
            })

        sec_stats = {"沪深300": [0, 0.0], "创业板": [0, 0.0], "科创板": [0, 0.0]}
        for c in st["closed_trades"]:
            sec_lbl = {"HS300": "沪深300", "CYB": "创业板", "KCB": "科创板"}[get_sector_of(c["symbol"])]
            sec_stats[sec_lbl][0] += 1
            sec_stats[sec_lbl][1] = round(sec_stats[sec_lbl][1] + c["pnl_amt"], 2)

        res = {
            "account_name": v_name, "initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"],
            "final_cash": round(st["cash"], 2), "final_holdings": st["holdings"],
            "sector_pnl_attribution": sec_stats, "history": hist,
            "trades": st["trades"], "closed_trades": st["closed_trades"]
        }
        m = compute_metrics(res)
        pk = max(hist, key=lambda x: x["portfolio_value"])
        tr = min(hist, key=lambda x: x["portfolio_value"])
        m["peak_value"] = pk["portfolio_value"]; m["peak_return_pct"] = pk["portfolio_return_pct"]; m["peak_date"] = pk["date"]
        m["trough_value"] = tr["portfolio_value"]; m["trough_return_pct"] = tr["portfolio_return_pct"]; m["trough_date"] = tr["date"]
        res["metrics"] = m
        out_1y[v_name] = res
        print(f"\n=== {v_name} ({trade_dates[0]} ~ {trade_dates[-1]}, {len(trade_dates)} 交易日) ===")
        print(f"  期末总净值: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | 沪深300: {m['hs300_return_pct']:+.2f}% | 超额Alpha: {m['alpha_pct']:+.2f}%")
        print(f"  最高净值  : ¥{m['peak_value']:,.2f} ({m['peak_return_pct']:+.2f}%, {m['peak_date']}) | 最低净值: ¥{m['trough_value']:,.2f} ({m['trough_date']})")
        print(f"  最大回撤  : -{m['max_drawdown_pct']:.2f}% ({m['max_drawdown_date']}) | 夏普比率: {m['sharpe_ratio']:.2f} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f}")
        print(f"  三大板块平仓盈亏贡献: {sec_stats}")

    with open(base / "backtest_v4_0_1y.json", "w", encoding="utf-8") as f:
        json.dump(out_1y, f, ensure_ascii=False, indent=2)
