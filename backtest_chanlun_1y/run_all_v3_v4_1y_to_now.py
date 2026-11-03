"""
ChanlunAIApp V3系列 (V3.0 ~ V3.3-C) 与 V4系列 (V4.0-A ~ V4.2-D)
去年10月底至今 (2025-10-30 ~ 2026-11-02, 244个交易日) 全模型独立回测与 7年长周期对比
"""
import bisect
import gc
import json
import time
from datetime import datetime
from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import market_regime_v2
from run_backtest import compute_metrics
from duckdb_Backtest_engine import DB_PATH, KCB_ACTIVE_DATE
from pit_universe_20y import get_20y_candidate_pools
from model_v3_2_mainline import get_sector_of
from model_v4_0_zero_prior import build_v4_sql_blind_rosters, step_v4_0_zero_prior
from model_v4_1_institutional import build_v4_1_asymmetric_buffered_rosters, step_v4_1_institutional
from model_v4_2_production import step_v4_2_production
from run_all_v3_v4_2020_to_now import build_v3_pit_rosters, step_v3_series

base = Path(__file__).resolve().parent


def load_indexed_1y(sym_set, need_5m=False):
    t0 = time.time()
    sym_list_sql = ",".join(f"'{s}'" for s in sorted(sym_set))
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA disable_progress_bar;")

    indexed = {s: {
        "day": [], "day_dates": [], "day_by_date": {},
        "m60_tuples": [], "m60_dates": [],
        "m15_tuples": [], "m15_dates": [],
        "m5_1450_by_date": {},
        "snaps": {}
    } for s in sym_set}

    bdom_rows = con.execute(f"""
        SELECT symbol, date,
               SUM(volume * CASE WHEN high > low THEN (close - low) / (high - low) WHEN close >= open THEN 0.55 ELSE 0.45 END) AS b_vol,
               SUM(volume) AS tot_vol
        FROM kline_15m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2025-09-01'
        GROUP BY symbol, date
    """).fetchall()
    bdom_map = {(s, d): round((2.0 * bv - tv) / tv * 100.0, 1) for s, d, bv, tv in bdom_rows if tv and tv > 0}
    del bdom_rows

    rows_1d = con.execute(f"""
        SELECT symbol, date, open, high, low, close, volume, amount, hfq_factor, ratio_factor
        FROM kline_1d
        WHERE symbol IN ({sym_list_sql}) AND date >= '2024-06-01' AND close > 0
        ORDER BY symbol, date
    """).fetchall()
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
    del rows_1d, bdom_map

    rows_60m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume
        FROM kline_60m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2025-05-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()
    for sym, dt, dtm, o, h, l, c, v in rows_60m:
        e = indexed[sym]
        e["m60_tuples"].append((dtm, o, h, l, c, v))
        e["m60_dates"].append(dt)
    del rows_60m

    rows_15m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume, amount
        FROM kline_15m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2025-09-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()
    for sym, dt, dtm, o, h, l, c, v, amt in rows_15m:
        e = indexed[sym]
        true_amt15 = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0 and amt and amt > v * c * 10) else amt
        e["m15_tuples"].append((dtm, o, h, l, c, v, true_amt15))
        e["m15_dates"].append(dt)
    del rows_15m

    if need_5m:
        rows_5m = con.execute(f"""
            SELECT symbol, date, datetime, open, high, low, close, volume, amount
            FROM kline_5m
            WHERE symbol IN ({sym_list_sql}) AND date >= '2025-10-01'
              AND (datetime LIKE '%14:50' OR datetime LIKE '%14:50:00') AND close > 0
            ORDER BY symbol, datetime
        """).fetchall()
        for sym, dt, dtm, o, h, l, c, v, amt in rows_5m:
            e = indexed[sym]
            true_amt5 = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0 and amt and amt > v * c * 10) else amt
            e["m5_1450_by_date"][dt] = (dtm, o, h, l, c, v, true_amt5)
        del rows_5m

    con.close()
    gc.collect()
    print(f"   Loaded 1Y data for {len(sym_set)} symbols in {time.time()-t0:.1f}s")
    return indexed


def pack_1y_result(k, st, hist):
    res = {"account_name": k, "initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"], "history": hist, "trades": st["trades"], "closed_trades": st["closed_trades"]}
    m = compute_metrics(res)
    m["closed_count"] = len(st["closed_trades"])
    sec_pnl = {"HS300": 0.0, "CYB": 0.0, "KCB": 0.0}
    for c in st["closed_trades"]:
        sec_pnl[get_sector_of(c["symbol"])] += c["pnl_amt"]
    m["sector_pnl"] = {s: round(v, 2) for s, v in sec_pnl.items()}
    return {"metrics": m, "history": hist}


def main():
    t0 = time.time()
    summary_1y = {}

    # 1. Build rosters in memory
    mem_con = duckdb.connect(":memory:")
    mem_con.execute("PRAGMA disable_progress_bar;")
    mem_con.execute(f"ATTACH '{DB_PATH}' AS lexar (READ_ONLY);")
    mem_con.execute("CREATE TABLE stocks AS SELECT * FROM lexar.stocks;")
    mem_con.execute("""
        CREATE TABLE kline_1d AS
        SELECT symbol, date, open, high, low, close, volume, amount, hfq_factor, ratio_factor
        FROM lexar.kline_1d
        WHERE date >= '2024-06-01' AND close > 0;
    """)
    all_dates = [r[0] for r in mem_con.execute("SELECT DISTINCT date FROM kline_1d WHERE symbol='sh000300' AND date >= '2025-10-30' ORDER BY date").fetchall()]
    trade_dates = all_dates

    r_v40, syms_v40, fmap_v40 = build_v4_sql_blind_rosters(mem_con, trade_dates)
    r_v41_pure, syms_v41_p, fmap_v41_p = build_v4_1_asymmetric_buffered_rosters(mem_con, trade_dates, use_dual_channel_hs=False)
    r_v41_dual, syms_v41_d, fmap_v41_d = build_v4_1_asymmetric_buffered_rosters(mem_con, trade_dates, use_dual_channel_hs=True)
    mem_con.close()

    hs_pool, cyb_pool, kcb_pool = get_20y_candidate_pools()
    v3_syms = {m[0] for m in (hs_pool + cyb_pool + kcb_pool)}
    all_syms = v3_syms | syms_v40 | syms_v41_p | syms_v41_d | {"sh000300"}

    indexed = load_indexed_1y(all_syms, need_5m=True)
    bench_idx = indexed["sh000300"]
    bmap = {b["time"]: b["close"] for b in bench_idx["day"]}
    init_b = bmap[trade_dates[0]]
    r_v3, fmap_v3 = build_v3_pit_rosters(indexed, trade_dates, hs_pool, cyb_pool, kcb_pool)

    model_keys = [
        "V3.0 稳健8仓分散版",
        "V3.1 分层保送4仓集中版",
        "V3.2 动态主线+分级红利盾牌",
        "V3.3-A 终极版 (15%扩容+日线过滤)",
        "V3.3-B 终极版 (15%扩容+60m脉冲过滤)",
        "V3.3-C 终极版 (12%稳健扩容+日线过滤)",
        "V4.0-A 零先验SQL盲选 (15%复利)",
        "V4.0-B 零先验SQL盲选 (12%复利)",
        "V4.1-Dual 机构实验版 (双通道·平滑状态机)",
        "V4.1-PureLiq 机构实验版 (纯流动·平滑状态机)",
        "V4.2-A 实战纯流动版 (14:45隔离·15%复利)",
        "V4.2-B 实战纯流动版 (14:45隔离·12%复利)",
        "V4.2-C 实战双通道版 (14:45隔离·15%复利)",
        "V4.2-D 实战双通道版 (14:45隔离·12%复利)",
    ]
    states = {k: {
        "cash": 1000000.0, "holdings": {}, "trades": [], "closed_trades": [],
        "equity": 1000000.0, "regime_history": [],
        "stats": {"limit_up_blocked": 0, "limit_down_two_step": 0, "fx_protect_blocked": 0}
    } for k in model_keys}
    hist_map = {k: [] for k in model_keys}

    t_sim = time.time()
    for idx_d, dt in enumerate(trade_dates):
        day_cache = {}
        market_regime_v2._BENCH_CACHE.clear()
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        bench_sl = bench_idx["day"][max(0, b_pos - 250):b_pos]
        r3_dt = r_v3[dt]
        k_act = (dt >= KCB_ACTIVE_DATE) and (len(r3_dt["kcb"]) >= 8)
        seen_u3 = set(); u3_metas = []
        for m in (r3_dt["hs"] + r3_dt["cyb"] + (r3_dt["kcb"] if k_act else [])):
            if m[0] not in seen_u3:
                u3_metas.append(m); seen_u3.add(m[0])

        # V3 Series
        step_v3_series(states["V3.0 稳健8仓分散版"], u3_metas, None, fmap_v3, indexed, bench_sl, dt, day_cache, base_stock_cap=125000.0, dyn_sizing_pct=None, route_mode="v3_0")
        step_v3_series(states["V3.1 分层保送4仓集中版"], None, r3_dt, fmap_v3, indexed, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=None, route_mode="v3_1")
        step_v3_series(states["V3.2 动态主线+分级红利盾牌"], None, r3_dt, fmap_v3, indexed, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=None, route_mode="v3_2")
        step_v3_series(states["V3.3-A 终极版 (15%扩容+日线过滤)"], None, r3_dt, fmap_v3, indexed, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=0.15, route_mode="v3_3_day")
        step_v3_series(states["V3.3-B 终极版 (15%扩容+60m脉冲过滤)"], None, r3_dt, fmap_v3, indexed, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=0.15, route_mode="v3_3_m60")
        step_v3_series(states["V3.3-C 终极版 (12%稳健扩容+日线过滤)"], None, r3_dt, fmap_v3, indexed, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=0.12, route_mode="v3_3_day")

        # V4.0 Series
        step_v4_0_zero_prior(states["V4.0-A 零先验SQL盲选 (15%复利)"], r_v40[dt], fmap_v40, indexed, bench_sl, dt, day_cache, dyn_sizing_pct=0.15)
        step_v4_0_zero_prior(states["V4.0-B 零先验SQL盲选 (12%复利)"], r_v40[dt], fmap_v40, indexed, bench_sl, dt, day_cache, dyn_sizing_pct=0.12)

        # V4.1 & V4.2 Series
        step_v4_1_institutional(states["V4.1-Dual 机构实验版 (双通道·平滑状态机)"], r_v41_dual[dt], fmap_v41_d, indexed, bench_idx, dt, day_cache, dyn_sizing_pct=0.12)
        step_v4_1_institutional(states["V4.1-PureLiq 机构实验版 (纯流动·平滑状态机)"], r_v41_pure[dt], fmap_v41_p, indexed, bench_idx, dt, day_cache, dyn_sizing_pct=0.12)
        step_v4_2_production(states["V4.2-A 实战纯流动版 (14:45隔离·15%复利)"], r_v41_pure[dt], fmap_v41_p, indexed, bench_idx, dt, day_cache, dyn_sizing_pct=0.15, strict_1445_isolation=True)
        step_v4_2_production(states["V4.2-B 实战纯流动版 (14:45隔离·12%复利)"], r_v41_pure[dt], fmap_v41_p, indexed, bench_idx, dt, day_cache, dyn_sizing_pct=0.12, strict_1445_isolation=True)
        step_v4_2_production(states["V4.2-C 实战双通道版 (14:45隔离·15%复利)"], r_v41_dual[dt], fmap_v41_d, indexed, bench_idx, dt, day_cache, dyn_sizing_pct=0.15, strict_1445_isolation=True)
        step_v4_2_production(states["V4.2-D 实战双通道版 (14:45隔离·12%复利)"], r_v41_dual[dt], fmap_v41_d, indexed, bench_idx, dt, day_cache, dyn_sizing_pct=0.12, strict_1445_isolation=True)

        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        for k in model_keys:
            tot = states[k]["equity"]
            tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
            hist_map[k].append({"date": dt, "portfolio_value": tot, "portfolio_return_pct": tot_r, "hs300_level": bmap[dt], "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)})

    for k in model_keys:
        summary_1y[k] = pack_1y_result(k, states[k], hist_map[k])

    d_7y = json.load(open(base / "backtest_v3_v4_matrix_2020_2026.json", "r", encoding="utf-8"))

    print(f"\n================ 去年10月底至今 ({trade_dates[0]} ~ {trade_dates[-1]}, {len(trade_dates)}交易日) vs 7年全周期 (2020~2026) 全模型对比总表 ================")
    for k in model_keys:
        m1 = summary_1y[k]["metrics"]
        m7 = d_7y[k]["metrics"]
        print(
            f"  {k:<36} | [1Y] 收益:{m1['total_return_pct']:+6.2f}% (Alpha:{m1['alpha_pct']:+6.2f}%) 回撤:-{m1['max_drawdown_pct']:5.2f}% 夏普:{m1['sharpe_ratio']:5.2f} 胜率:{m1['win_rate_pct']:5.2f}% | "
            f"[7Y] 收益:{m7['total_return_pct']:+7.2f}% 回撤:-{m7['max_drawdown_pct']:5.2f}% 夏普:{m7['sharpe_ratio']:5.2f}"
        )

    with open(base / "backtest_v3_v4_matrix_1y_2025_2026.json", "w", encoding="utf-8") as f:
        json.dump(summary_1y, f, ensure_ascii=False, indent=2)

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    h0 = summary_1y[model_keys[0]]["history"]
    dates_dt = [datetime.strptime(x["date"], "%Y-%m-%d") for x in h0]
    v_bench = [100.0 * (1.0 + x["hs300_return_pct"] / 100.0) for x in h0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11.2), gridspec_kw={"height_ratios": [1.1, 1.1]})
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")

    v3_colors = ["#38bdf8", "#60a5fa", "#a855f7", "#f59e0b", "#10b981", "#34d399"]
    for idx_k, k in enumerate(model_keys[:6]):
        m = summary_1y[k]["metrics"]
        v = [x["portfolio_value"] / 10000.0 for x in summary_1y[k]["history"]]
        lw = 2.6 if ("V3.2" in k or "V3.0" in k) else 1.7
        ax1.plot(dates_dt, v, color=v3_colors[idx_k], linewidth=lw,
                 label=f"{k} (期末 ¥{v[-1]:.2f}万, {m['total_return_pct']:+.2f}% | 回撤 -{m['max_drawdown_pct']:.2f}% | 夏普 {m['sharpe_ratio']:.2f})")
    ax1.plot(dates_dt, v_bench, color="#cbd5e1", linewidth=1.5, linestyle=":", label=f"沪深300基准 ({h0[-1]['hs300_return_pct']:+.2f}%)")
    ax1.axhline(100.0, color="#64748b", linewidth=1)
    ax1.set_title(f"ChanlunAIApp V3系列 (V3.0 ~ V3.3-C) 近1年 ({trade_dates[0]} ~ {trade_dates[-1]}) 净值对比", color="#f8fafc", fontsize=12.5, fontweight="bold")
    ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8.5)
    for t in leg1.get_texts(): t.set_color("#f8fafc")

    v4_colors = ["#fbbf24", "#f97316", "#f43f5e", "#e11d48", "#38bdf8", "#818cf8", "#10b981", "#a855f7"]
    for idx_k, k in enumerate(model_keys[6:]):
        m = summary_1y[k]["metrics"]
        v = [x["portfolio_value"] / 10000.0 for x in summary_1y[k]["history"]]
        lw = 2.6 if ("V4.2-C" in k or "V4.0-A" in k) else 1.6
        ax2.plot(dates_dt, v, color=v4_colors[idx_k], linewidth=lw,
                 label=f"{k} (期末 ¥{v[-1]:.2f}万, {m['total_return_pct']:+.2f}% | 回撤 -{m['max_drawdown_pct']:.2f}% | 夏普 {m['sharpe_ratio']:.2f})")
    ax2.plot(dates_dt, v_bench, color="#cbd5e1", linewidth=1.5, linestyle=":", label=f"沪深300基准 ({h0[-1]['hs300_return_pct']:+.2f}%)")
    ax2.axhline(100.0, color="#64748b", linewidth=1)
    ax2.set_title(f"ChanlunAIApp V4系列 (V4.0 ~ V4.2-D) 近1年 ({trade_dates[0]} ~ {trade_dates[-1]}) 净值对比", color="#f8fafc", fontsize=12.5, fontweight="bold")
    ax2.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    leg2 = ax2.legend(loc="upper left", ncol=2, facecolor="#0f172a", edgecolor="#334155", fontsize=8.0)
    for t in leg2.get_texts(): t.set_color("#f8fafc")

    plt.tight_layout()
    out_img = base / "chanlun_v3_vs_v4_1y_2025_2026.png"
    plt.savefig(out_img, dpi=180)
    print(f"Saved V3 vs V4 1Y comparison chart to {out_img} (Total elapsed: {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
