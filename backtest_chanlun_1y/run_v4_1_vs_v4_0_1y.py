import json
import time
from pathlib import Path
import duckdb
from run_backtest import compute_metrics
from duckdb_Backtest_engine import DB_PATH
from model_v4_1_institutional import (
    build_v4_1_asymmetric_buffered_rosters,
    step_v4_1_institutional,
    get_sector_of
)

base = Path(__file__).resolve().parent

def load_v4_1_1y_duckdb(use_dual_channel_hs: bool = True):
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

    print(f"1. Building V4.1 Asymmetric Buffered Rosters (Mainboard Semi-Annual + ChiNext/STAR Monthly, DualHS={use_dual_channel_hs})...")
    rosters_by_date, needed_syms, full_meta_map = build_v4_1_asymmetric_buffered_rosters(
        con, trade_dates, use_dual_channel_hs=use_dual_channel_hs
    )

    sym_list_sql = ",".join(f"'{s}'" for s in sorted(needed_syms))
    print(f"2. Loading 1d/60m/15m K-lines from DuckDB for {len(needed_syms)} symbols ({time.time()-t0:.1f}s)...")

    rows_1d = con.execute(f"""
        SELECT symbol, date, open, high, low, close, volume, amount
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

    for sym, dt, o, h, l, c, v, amt in rows_1d:
        e = indexed[sym]
        prev_c = e["day"][-1]["close"] if e["day"] else o
        pct = ((c - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
        true_amt = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0) else amt
        bar = {"symbol": sym, "time": dt, "open": o, "high": h, "low": l, "close": c, "volume": v, "amount": true_amt, "preClose": prev_c, "pctChange": pct}
        e["day"].append(bar)
        e["day_dates"].append(dt)
        e["day_by_date"][dt] = bar

    for sym, dt, dtm, o, h, l, c, v in rows_60m:
        e = indexed[sym]
        e["m60_tuples"].append((dtm, o, h, l, c, v))
        e["m60_dates"].append(dt)

    for sym, dt, dtm, o, h, l, c, v in rows_15m:
        e = indexed[sym]
        e["m15_tuples"].append((dtm, o, h, l, c, v))
        e["m15_dates"].append(dt)

    print(f"Loaded V4.1 1Y DuckDB dataset in {time.time() - t0:.2f}s!")
    return indexed, rosters_by_date, full_meta_map, trade_dates

def run_v4_1_1y_variant(v_name, rosters_by_date, fmap, indexed, trade_dates, eval_cache, dyn_pct=0.15):
    bench_idx = indexed["sh000300"]
    bmap = {b["time"]: b["close"] for b in bench_idx["day"]}
    init_b = bmap[trade_dates[0]]

    st = {
        "cash": 1000000.0, "holdings": {}, "trades": [], "closed_trades": [],
        "equity": 1000000.0, "regime_history": [],
        "stats": {"limit_up_blocked": 0, "limit_down_two_step": 0, "fx_protect_blocked": 0}
    }
    hist = []
    for dt in trade_dates:
        step_v4_1_institutional(st, rosters_by_date[dt], fmap, indexed, bench_idx, dt, eval_cache, dyn_sizing_pct=dyn_pct)
        hs_v = sum(h["shares"] * h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s) == "HS300")
        cyb_v = sum(h["shares"] * h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s) == "CYB")
        kcb_v = sum(h["shares"] * h["last_price"] for s, h in st["holdings"].items() if get_sector_of(s) == "KCB")
        tot = st["equity"]
        tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        hist.append({
            "date": dt, "portfolio_value": tot, "cash": round(st["cash"], 2),
            "hs300_mkt_val": round(hs_v, 2), "cyb_mkt_val": round(cyb_v, 2), "kcb_mkt_val": round(kcb_v, 2),
            "holdings_count": len(st["holdings"]),
            "regime_label": st["regime_history"][-1][1], "R_t": st["regime_history"][-1][2],
            "portfolio_return_pct": tot_r, "hs300_level": bmap[dt],
            "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)
        })

    sec_stats = {"沪深300": [0, 0.0], "创业板": [0, 0.0], "科创板": [0, 0.0]}
    exit_reasons = {"主升浪减半锁利": 0, "半仓移动止盈": 0, "冻结契约止损": 0, "结构破位止损": 0, "顶背驰一卖": 0, "早盘两步法": 0}
    for c in st["closed_trades"]:
        sec_lbl = {"HS300": "沪深300", "CYB": "创业板", "KCB": "科创板"}[get_sector_of(c["symbol"])]
        sec_stats[sec_lbl][0] += 1
        sec_stats[sec_lbl][1] = round(sec_stats[sec_lbl][1] + c["pnl_amt"], 2)
        rsn = c.get("reason", "")
        for k in exit_reasons:
            if k in rsn:
                exit_reasons[k] += 1

    res = {
        "account_name": v_name, "initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"],
        "final_cash": round(st["cash"], 2), "final_holdings": st["holdings"],
        "sector_pnl_attribution": sec_stats, "exit_reasons": exit_reasons, "stats": st["stats"],
        "history": hist, "trades": st["trades"], "closed_trades": st["closed_trades"]
    }
    m = compute_metrics(res)
    pk = max(hist, key=lambda x: x["portfolio_value"])
    tr = min(hist, key=lambda x: x["portfolio_value"])
    m["peak_value"] = pk["portfolio_value"]; m["peak_return_pct"] = pk["portfolio_return_pct"]; m["peak_date"] = pk["date"]
    m["trough_value"] = tr["portfolio_value"]; m["trough_return_pct"] = tr["portfolio_return_pct"]; m["trough_date"] = tr["date"]
    res["metrics"] = m
    return res

if __name__ == "__main__":
    # 1. Run V4.1 with Dual-Channel Mainboard + Monthly Buffered ChiNext/STAR
    indexed_dual, rosters_dual, fmap_dual, trade_dates = load_v4_1_1y_duckdb(use_dual_channel_hs=True)
    eval_cache_dual = {}
    res_dual_15 = run_v4_1_1y_variant("V4.1-Dual (双通道主板+月度缓冲双创 · 15%复利)", rosters_dual, fmap_dual, indexed_dual, trade_dates, eval_cache_dual, dyn_pct=0.15)
    res_dual_12 = run_v4_1_1y_variant("V4.1-Dual (双通道主板+月度缓冲双创 · 12%复利)", rosters_dual, fmap_dual, indexed_dual, trade_dates, eval_cache_dual, dyn_pct=0.12)

    # 2. Run V4.1 with Pure-Liquidity Mainboard + Monthly Buffered ChiNext/STAR
    indexed_liq, rosters_liq, fmap_liq, _ = load_v4_1_1y_duckdb(use_dual_channel_hs=False)
    eval_cache_liq = {}
    res_liq_15 = run_v4_1_1y_variant("V4.1-PureLiq (纯流动性主板+月度缓冲双创 · 15%复利)", rosters_liq, fmap_liq, indexed_liq, trade_dates, eval_cache_liq, dyn_pct=0.15)
    res_liq_12 = run_v4_1_1y_variant("V4.1-PureLiq (纯流动性主板+月度缓冲双创 · 12%复利)", rosters_liq, fmap_liq, indexed_liq, trade_dates, eval_cache_liq, dyn_pct=0.12)

    out_v41 = {
        res_dual_15["account_name"]: res_dual_15,
        res_dual_12["account_name"]: res_dual_12,
        res_liq_15["account_name"]: res_liq_15,
        res_liq_12["account_name"]: res_liq_12,
    }
    for k, r in out_v41.items():
        m = r["metrics"]
        print(f"\n=== {k} ({trade_dates[0]} ~ {trade_dates[-1]}, {len(trade_dates)} 交易日) ===")
        print(f"  期末总净值: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | 沪深300: {m['hs300_return_pct']:+.2f}% | Alpha: {m['alpha_pct']:+.2f}%")
        print(f"  最高净值  : ¥{m['peak_value']:,.2f} ({m['peak_return_pct']:+.2f}%, {m['peak_date']}) | 最低净值: ¥{m['trough_value']:,.2f} ({m['trough_date']})")
        print(f"  最大回撤  : -{m['max_drawdown_pct']:.2f}% ({m['max_drawdown_date']}) | 夏普: {m['sharpe_ratio']:.2f} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f} (均盈{m['avg_win_pct']:+.2f}% / 均亏{m['avg_loss_pct']:+.2f}%)")
        print(f"  板块平仓盈亏贡献: {r['sector_pnl_attribution']}")
        print(f"  出场归因统计    : {r['exit_reasons']} | 微观拦截: {r['stats']}")

    with open(base / "backtest_v4_1_1y.json", "w", encoding="utf-8") as f:
        json.dump(out_v41, f, ensure_ascii=False, indent=2)
