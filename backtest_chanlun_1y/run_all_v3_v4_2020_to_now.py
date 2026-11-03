"""
ChanlunAIApp V3系列 (V3.0 ~ V3.3-C) 与 V4系列 (V4.0-A ~ V4.2-D)
2020-01-02 ~ 2026-11-02 (6.81年完整牛熊周期) 分阶段低内存高速实盘回测对比
"""
import bisect
import gc
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import market_regime_v2
from run_backtest import compute_metrics, get_stage1_candidates
from duckdb_Backtest_engine import DB_PATH, KCB_ACTIVE_DATE, calc_trade_cost, get_pit_bars_slice
from pit_universe import select_pit_universe_at_date
from pit_universe_20y import get_20y_candidate_pools
from chanlun_engine import evaluate_asset
from model_v3_2_mainline import (
    get_sector_of,
    HS300_DEFENSIVE_T1,
    HS300_DEFENSIVE_T2,
    compute_sector_mainline_boosts,
    get_bear_severity_level,
)
from model_v4_0_zero_prior import (
    build_v4_sql_blind_rosters,
    step_v4_0_zero_prior,
    is_tail_limit_up,
    is_tail_limit_down,
    resolve_limit_down_two_step,
)
from model_v4_1_institutional import (
    build_v4_1_asymmetric_buffered_rosters,
    step_v4_1_institutional,
)
from model_v4_2_production import step_v4_2_production

base = Path(__file__).resolve().parent


def load_indexed_for_symbols(sym_set, need_5m=False):
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
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-11-01'
        GROUP BY symbol, date
    """).fetchall()
    bdom_map = {(s, d): round((2.0 * bv - tv) / tv * 100.0, 1) for s, d, bv, tv in bdom_rows if tv and tv > 0}
    del bdom_rows
    gc.collect()

    rows_1d = con.execute(f"""
        SELECT symbol, date, open, high, low, close, volume, amount, hfq_factor, ratio_factor
        FROM kline_1d
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-01-01' AND close > 0
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
    gc.collect()

    rows_60m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume
        FROM kline_60m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-06-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()
    for sym, dt, dtm, o, h, l, c, v in rows_60m:
        e = indexed[sym]
        e["m60_tuples"].append((dtm, o, h, l, c, v))
        e["m60_dates"].append(dt)
    del rows_60m
    gc.collect()

    rows_15m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume, amount
        FROM kline_15m
        WHERE symbol IN ({sym_list_sql}) AND date >= '2019-11-01' AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()
    for sym, dt, dtm, o, h, l, c, v, amt in rows_15m:
        e = indexed[sym]
        true_amt15 = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0 and amt and amt > v * c * 10) else amt
        e["m15_tuples"].append((dtm, o, h, l, c, v, true_amt15))
        e["m15_dates"].append(dt)
    del rows_15m
    gc.collect()

    if need_5m:
        rows_5m = con.execute(f"""
            SELECT symbol, date, datetime, open, high, low, close, volume, amount
            FROM kline_5m
            WHERE symbol IN ({sym_list_sql}) AND date >= '2020-01-01'
              AND (datetime LIKE '%14:50' OR datetime LIKE '%14:50:00') AND close > 0
            ORDER BY symbol, datetime
        """).fetchall()
        for sym, dt, dtm, o, h, l, c, v, amt in rows_5m:
            e = indexed[sym]
            true_amt5 = (amt / 100.0) if (sym.startswith("sh688") and int(v) % 100 == 0 and amt and amt > v * c * 10) else amt
            e["m5_1450_by_date"][dt] = (dtm, o, h, l, c, v, true_amt5)
        del rows_5m
        gc.collect()

    con.close()
    print(f"   Loaded {len(sym_set)} symbols in {time.time()-t0:.1f}s")
    return indexed


def build_v3_pit_rosters(indexed, trade_dates, hs_pool, cyb_pool, kcb_pool):
    full_meta_map = {m[0]: m for m in (hs_pool + cyb_pool + kcb_pool)}
    rosters = {}
    cur_hs = select_pit_universe_at_date(hs_pool, indexed, trade_dates[0], top_n=100, min_listed_days=60)
    cur_cyb = select_pit_universe_at_date(cyb_pool, indexed, trade_dates[0], top_n=50, min_listed_days=60)
    cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, trade_dates[0], top_n=35, min_listed_days=20) if trade_dates[0] >= KCB_ACTIVE_DATE else []

    for i, dt in enumerate(trade_dates):
        if dt >= KCB_ACTIVE_DATE and not cur_kcb:
            cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, dt, top_n=35, min_listed_days=20)
        rosters[dt] = {"hs": cur_hs, "cyb": cur_cyb, "kcb": cur_kcb}
        if i < len(trade_dates) - 1:
            cur_ym, nxt_ym = dt[:7], trade_dates[i + 1][:7]
            if cur_ym != nxt_ym and (cur_ym.endswith("-06") or cur_ym.endswith("-12")):
                cur_hs = select_pit_universe_at_date(hs_pool, indexed, dt, top_n=100, min_listed_days=60)
                cur_cyb = select_pit_universe_at_date(cyb_pool, indexed, dt, top_n=50, min_listed_days=60)
                if dt >= KCB_ACTIVE_DATE:
                    cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, dt, top_n=35, min_listed_days=20)
    return rosters, full_meta_map


def step_v3_series(state, pool_metas, tiered_rosters, fmap, indexed, bench_day_slice, dt, eval_cache,
                   base_stock_cap=250000.0, dyn_sizing_pct=None, max_single_cap=2500000.0,
                   route_mode="v3_0"):
    holdings = state["holdings"]
    cash = state["cash"]

    for sym in list(holdings.keys()):
        h = holdings[sym]
        if not h.get("pending_sell_reason"):
            continue
        idx_e = indexed.get(sym)
        dbar = idx_e.get("day_by_date", {}).get(dt) if idx_e else None
        can_sell, raw_exec_p, tag = resolve_limit_down_two_step(sym, dt, dbar, idx_e)
        if not can_sell:
            h["pending_sell_reason"] = h["pending_sell_reason"] + f" -> {tag}"
            continue
        exec_p, gross, fee = calc_trade_cost("SELL", sym, h["kind"], dt, raw_exec_p, h["shares"])
        net = gross - fee
        cash += net
        pnl_amt = net - (h["shares"] * h["avg_cost"])
        pnl_pct = (exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0
        state["closed_trades"].append({
            "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
            "shares": h["shares"], "buy_price": h["avg_cost"], "sell_price": exec_p,
            "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2), "reason": f"{h['pending_sell_reason']} -> {tag}"
        })
        del holdings[sym]

    for sym, h in holdings.items():
        dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
        if dbar and dbar["close"] > 0:
            h["last_price"] = dbar["close"]

    if tiered_rosters is not None:
        hs_u, cyb_u, kcb_u = tiered_rosters["hs"], tiered_rosters["cyb"], tiered_rosters["kcb"]
        kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
        seen_a = set(); all_act = []
        for m in (hs_u + cyb_u + (kcb_u if kcb_active else [])):
            if m[0] not in seen_a:
                all_act.append(m); seen_a.add(m[0])
        for s in holdings:
            if s in fmap and s not in seen_a:
                all_act.append(fmap[s]); seen_a.add(s)
        day_snaps = {s: indexed[s]["snaps"][dt] for s, _, _ in all_act if s in indexed and dt in indexed[s]["snaps"]}
        sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=16)
        sel_cyb = get_stage1_candidates(cyb_u, day_snaps, {}, top_k=18 if kcb_active else 28)
        sel_kcb = get_stage1_candidates(kcb_u, day_snaps, {}, top_k=12) if kcb_active else []
        seen = set(); stage1 = []
        for s in holdings:
            if s in fmap and s not in seen:
                stage1.append(fmap[s]); seen.add(s)
        for m in (sel_cyb + sel_kcb + sel_hs):
            if m[0] not in seen:
                stage1.append(m); seen.add(m[0])
    else:
        seen_u = {m[0] for m in pool_metas}
        u_pool = list(pool_metas)
        for s in holdings:
            if s not in seen_u and s in fmap:
                u_pool.append(fmap[s]); seen_u.add(s)
        day_snaps = {s: indexed[s]["snaps"][dt] for s, _, _ in u_pool if s in indexed and dt in indexed[s]["snaps"]}
        stage1 = get_stage1_candidates(u_pool, day_snaps, holdings, top_k=48)

    evaluated = []
    for meta in stage1:
        sym = meta[0]
        ck = (sym, dt)
        if ck in eval_cache:
            ev = eval_cache[ck]
        else:
            idx_e = indexed.get(sym)
            if not idx_e or dt not in idx_e["day_by_date"]:
                eval_cache[ck] = None; continue
            d_bars, m60_bars, m15_bars = get_pit_bars_slice(idx_e, dt)
            if not d_bars or not m15_bars:
                eval_cache[ck] = None; continue
            ev = evaluate_asset(meta, d_bars, m60_bars, m15_bars, day_snaps.get(sym))
            p15 = ev["periods"].get("15m")
            bi15 = p15.get("lastBi") if p15 else None
            zs15 = p15.get("lastZs") if p15 else None
            ev["bot_fx_low"] = bi15["low"] if bi15 else (zs15["dd"] if zs15 else d_bars[-1]["low"])
            eval_cache[ck] = ev
        if ev is not None:
            evaluated.append(ev)

    evaluated.sort(key=lambda x: (x["score"], x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0, x["snapshot"]["amountYi"] if x["snapshot"] else 0.0), reverse=True)
    eval_map = {e["symbol"]: e for e in evaluated}
    regime, r_cfg, _ = market_regime_v2.classify_market_regime(bench_day_slice, day_snaps, evaluated, mode="dual_factor")

    for sym in list(holdings.keys()):
        h = holdings[sym]
        if h.get("pending_sell_reason"):
            continue
        ev = eval_map.get(sym)
        dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
        if not dbar or dbar["close"] <= 0 or dbar["volume"] <= 0:
            continue
        sell_rsn = None
        if ev:
            p_m15 = ev["periods"].get("15m")
            if p_m15:
                zs15 = p_m15.get("lastZs")
                bi15 = p_m15.get("lastBi")
                div15 = p_m15.get("divRatio")
                if regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
                    sell_rsn = f"[14:54熊市锁死止损] 破初始止损({h['init_stop']})"
                elif zs15 and ev["price"] < zs15["dd"]:
                    sell_rsn = f"[14:54结构破位止损] 破15m DD({zs15['dd']})"
                elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                    sell_rsn = f"[14:54顶背驰一卖] 面积比={div15}"
        if sell_rsn:
            if is_tail_limit_down(sym, dt, dbar):
                h["pending_sell_reason"] = f"{sell_rsn}({dt}尾盘14:54封跌停未成)"
                continue
            exec_p, gross, fee = calc_trade_cost("SELL", sym, h["kind"], dt, dbar["close"], h["shares"])
            net = gross - fee
            cash += net
            pnl_amt = net - (h["shares"] * h["avg_cost"])
            pnl_pct = (exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0
            state["closed_trades"].append({
                "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
                "shares": h["shares"], "buy_price": h["avg_cost"], "sell_price": exec_p,
                "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2), "reason": sell_rsn
            })
            del holdings[sym]

    sec_boosts, main_winner = compute_sector_mainline_boosts(evaluated) if route_mode in ("v3_2", "v3_3_day", "v3_3_m60") else ({"HS300": 0, "CYB": 0, "KCB": 0}, "HS300")
    bear_sev = get_bear_severity_level(bench_day_slice) if regime == "WEAK" else "NONE"

    def rank_fn(cand):
        sym = cand["symbol"]
        sec = get_sector_of(sym)
        base_s = cand["score"]
        if route_mode == "v3_0":
            is_g = sec in ("CYB", "KCB")
            adj_s = base_s + (8 if (regime == "BULL" and is_g) else (15 if (regime in ("WEAK", "CRASH") and not is_g) else 0))
        elif route_mode == "v3_1":
            is_g = sec in ("CYB", "KCB")
            adj_s = base_s + (12 if (regime in ("BULL", "RANGE") and is_g) else ((18 if not is_g else -8) - (25 if cand["price"] > 180 else 0)))
        else:
            p_day = cand["periods"].get("day")
            zs_day = p_day.get("lastZs") if p_day else None
            above_zg = bool(zs_day and p_day["close"] > zs_day["zg"])
            over_pen = 0
            if route_mode == "v3_3_day" and zs_day and zs_day["zg"] > 0:
                d_zg = (p_day["close"] - zs_day["zg"]) / zs_day["zg"]
                if d_zg > 0.38: over_pen = 25
                elif d_zg > 0.28: over_pen = 10
            elif route_mode == "v3_3_m60":
                p60 = cand["periods"].get("60m")
                zs60 = p60.get("lastZs") if p60 else None
                if zs60 and zs60["zg"] > 0 and (cand["price"] - zs60["zg"]) / zs60["zg"] > 0.25:
                    over_pen = 20
            if regime in ("BULL", "RANGE"):
                mb = sec_boosts.get(sec, 0)
                if not above_zg: mb //= 2
                eb = 4 if (regime == "BULL" and sec in ("CYB", "KCB") and mb >= 8 and above_zg) else 0
                adj_s = base_s + mb + eb - over_pen
            else:
                if bear_sev == "WEAK_DEEP":
                    if sym in HS300_DEFENSIVE_T1: adj_s = base_s + 26
                    elif sym in HS300_DEFENSIVE_T2: adj_s = base_s + 14
                    elif sec == "HS300": adj_s = base_s + 2
                    else: adj_s = base_s - (4 if (sec == main_winner and above_zg) else 16)
                    if cand["price"] > 120.0: adj_s -= 30
                else:
                    if sym in HS300_DEFENSIVE_T1: adj_s = base_s + 18
                    elif sym in HS300_DEFENSIVE_T2: adj_s = base_s + 12
                    elif sec == "HS300": adj_s = base_s + 6
                    else: adj_s = base_s - (2 if (sec == main_winner and above_zg) else 8)
                    if cand["price"] > 180.0: adj_s -= 25
                adj_s -= over_pen
        return (adj_s, cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0, cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0)

    ranked = sorted(evaluated, key=rank_fn, reverse=True)
    mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
    cur_eq = cash + mkt_val_now

    eff_max_exp = 0.55 if (route_mode in ("v3_2", "v3_3_day", "v3_3_m60") and bear_sev == "WEAK_DEEP") else r_cfg["max_exposure_pct"]
    rem_room = max(0.0, cur_eq * eff_max_exp - mkt_val_now)
    min_sc = 65 if (route_mode in ("v3_2", "v3_3_day", "v3_3_m60") and bear_sev == "WEAK_DEEP") else r_cfg["min_score"]
    pos_sc = 0.55 if (route_mode in ("v3_2", "v3_3_day", "v3_3_m60") and bear_sev == "WEAK_DEEP") else r_cfg["single_pos_scale"]

    eff_stock_cap = min(max_single_cap, max(base_stock_cap, cur_eq * dyn_sizing_pct)) if dyn_sizing_pct else base_stock_cap
    eff_etf_cap = eff_stock_cap * 1.2
    eff_min_cash = min(eff_stock_cap * 0.55, 350000.0) * max(pos_sc, 0.45)

    budget = min(cash, rem_room)
    bought = 0
    daily_limit = r_cfg["daily_buy_limit"]

    if daily_limit > 0 and rem_room >= eff_min_cash:
        for cand in ranked:
            if bought >= daily_limit or budget < eff_min_cash:
                break
            sym = cand["symbol"]
            if sym in holdings or cand["score"] < min_sc or cand["price"] <= 0:
                continue
            if route_mode in ("v3_1", "v3_2", "v3_3_day", "v3_3_m60") and regime == "WEAK" and cand["price"] > (150.0 if bear_sev == "WEAK_DEEP" else 220.0):
                continue
            dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if not dbar or dbar["close"] <= 0 or dbar["volume"] <= 0:
                continue
            if is_tail_limit_up(sym, dt, dbar):
                continue
            p15 = cand["periods"].get("15m")
            zs15 = p15.get("lastZs") if p15 else None
            raw_stop = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
            bot_fx_low = cand.get("bot_fx_low", raw_stop)
            if dbar["close"] < bot_fx_low * 0.995:
                continue

            snap = cand.get("snapshot")
            liq_cap = max(250000.0, snap["amountYi"] * 1e8 * 0.01) if (snap and snap["amountYi"] > 0) else max_single_cap
            t_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_sc, liq_cap)
            alloc = min(budget * 0.95, t_amt, rem_room)

            rf = dbar.get("ratio_factor", 1.0)
            rf = rf if (rf and rf > 0) else 1.0
            raw_price = dbar["close"] / rf
            min_board_sh = 200 if sym.startswith("sh688") else 100
            lots = int(alloc // (raw_price * 1.001 * 100.0))
            raw_shares = lots * 100
            if raw_shares < min_board_sh:
                continue
            shares = round(raw_shares / rf, 4)

            exec_p, gross, fee = calc_trade_cost("BUY", sym, cand["kind"], dt, dbar["close"], shares)
            tot_cost = gross + fee
            if tot_cost > cash:
                continue
            cash -= tot_cost
            budget -= tot_cost
            rem_room -= gross
            bought += 1
            holdings[sym] = {
                "name": cand["name"], "kind": cand["kind"], "shares": shares,
                "avg_cost": round(tot_cost / shares, 4), "last_price": dbar["close"],
                "buy_date": dt, "init_stop": raw_stop
            }

    state["cash"] = cash
    state["equity"] = round(cash + sum(h["shares"] * h["last_price"] for h in holdings.values()), 2)


def pack_model_result(k, st, hist, years):
    res = {"account_name": k, "initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"], "history": hist, "trades": st["trades"], "closed_trades": st["closed_trades"]}
    m = compute_metrics(res)
    cagr = ((m["final_value"] / 1e6) ** (1.0 / years) - 1.0) * 100.0
    m["cagr_pct"] = round(cagr, 2)
    m["calmar_ratio"] = round(cagr / max(0.01, m["max_drawdown_pct"]), 2)
    m["closed_count"] = len(st["closed_trades"])

    by_yr = defaultdict(list)
    for h in hist:
        by_yr[h["date"][:4]].append(h)
    yr_rets = {}
    prev_e = 1e6
    for yr in sorted(by_yr.keys()):
        end_e = by_yr[yr][-1]["portfolio_value"]
        yr_rets[yr] = round((end_e - prev_e) / prev_e * 100.0, 2)
        prev_e = end_e
    m["yearly_return_pct"] = yr_rets

    sec_pnl = {"HS300": 0.0, "CYB": 0.0, "KCB": 0.0}
    for c in st["closed_trades"]:
        sec_pnl[get_sector_of(c["symbol"])] += c["pnl_amt"]
    m["sector_pnl"] = {s: round(v, 2) for s, v in sec_pnl.items()}
    return {"metrics": m, "history": hist}


def main():
    t0 = time.time()
    summary_out = {}

    # ================= PHASE 1: V3 SERIES (V3.0 ~ V3.3-C) =================
    print("=== PHASE 1: Running V3 Series (V3.0 ~ V3.3-C, 310 symbols) ===")
    hs_pool, cyb_pool, kcb_pool = get_20y_candidate_pools()
    v3_syms = {m[0] for m in (hs_pool + cyb_pool + kcb_pool)} | {"sh000300"}
    indexed_v3 = load_indexed_for_symbols(v3_syms, need_5m=False)
    bench_idx = indexed_v3["sh000300"]
    bmap = {b["time"]: b["close"] for b in bench_idx["day"]}
    trade_dates = [b["time"] for b in bench_idx["day"] if b["time"] >= "2020-01-02"]
    init_b = bmap[trade_dates[0]]
    years = len(trade_dates) / 243.0

    r_v3, fmap_v3 = build_v3_pit_rosters(indexed_v3, trade_dates, hs_pool, cyb_pool, kcb_pool)
    v3_keys = [
        "V3.0 稳健8仓分散版",
        "V3.1 分层保送4仓集中版",
        "V3.2 动态主线+分级红利盾牌",
        "V3.3-A 终极版 (15%扩容+日线过滤)",
        "V3.3-B 终极版 (15%扩容+60m脉冲过滤)",
        "V3.3-C 终极版 (12%稳健扩容+日线过滤)",
    ]
    states_v3 = {k: {"cash": 1000000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 1000000.0} for k in v3_keys}
    hist_v3 = {k: [] for k in v3_keys}
    t_p1 = time.time()

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

        step_v3_series(states_v3["V3.0 稳健8仓分散版"], u3_metas, None, fmap_v3, indexed_v3, bench_sl, dt, day_cache, base_stock_cap=125000.0, dyn_sizing_pct=None, route_mode="v3_0")
        step_v3_series(states_v3["V3.1 分层保送4仓集中版"], None, r3_dt, fmap_v3, indexed_v3, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=None, route_mode="v3_1")
        step_v3_series(states_v3["V3.2 动态主线+分级红利盾牌"], None, r3_dt, fmap_v3, indexed_v3, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=None, route_mode="v3_2")
        step_v3_series(states_v3["V3.3-A 终极版 (15%扩容+日线过滤)"], None, r3_dt, fmap_v3, indexed_v3, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=0.15, route_mode="v3_3_day")
        step_v3_series(states_v3["V3.3-B 终极版 (15%扩容+60m脉冲过滤)"], None, r3_dt, fmap_v3, indexed_v3, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=0.15, route_mode="v3_3_m60")
        step_v3_series(states_v3["V3.3-C 终极版 (12%稳健扩容+日线过滤)"], None, r3_dt, fmap_v3, indexed_v3, bench_sl, dt, day_cache, base_stock_cap=250000.0, dyn_sizing_pct=0.12, route_mode="v3_3_day")

        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        for k in v3_keys:
            tot = states_v3[k]["equity"]
            tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
            hist_v3[k].append({"date": dt, "portfolio_value": tot, "portfolio_return_pct": tot_r, "hs300_level": bmap[dt], "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)})

        if (idx_d + 1) % 400 == 0 or idx_d == len(trade_dates) - 1:
            print(f"   [Phase 1 V3] Day {idx_d+1}/{len(trade_dates)} ({dt}) | V3.3-B: {hist_v3['V3.3-B 终极版 (15%扩容+60m脉冲过滤)'][-1]['portfolio_return_pct']:+.2f}% | Elapsed: {time.time()-t_p1:.1f}s")

    for k in v3_keys:
        summary_out[k] = pack_model_result(k, states_v3[k], hist_v3[k], years)

    del indexed_v3, states_v3, hist_v3, r_v3
    gc.collect()

    # ================= PHASE 2A: V4.0 SERIES (V4.0-A / V4.0-B) =================
    print("\n=== PHASE 2A: Running V4.0 Series (V4.0-A / V4.0-B) ===")
    mem_con = duckdb.connect(":memory:")
    mem_con.execute("PRAGMA disable_progress_bar;")
    mem_con.execute(f"ATTACH '{DB_PATH}' AS lexar (READ_ONLY);")
    mem_con.execute("CREATE TABLE stocks AS SELECT * FROM lexar.stocks;")
    mem_con.execute("""
        CREATE TABLE kline_1d AS
        SELECT symbol, date, open, high, low, close, volume, amount, hfq_factor, ratio_factor
        FROM lexar.kline_1d
        WHERE date >= '2019-01-01' AND close > 0;
    """)
    r_v40, syms_v40, fmap_v40 = build_v4_sql_blind_rosters(mem_con, trade_dates)
    r_v41_pure, syms_v41_p, fmap_v41_p = build_v4_1_asymmetric_buffered_rosters(mem_con, trade_dates, use_dual_channel_hs=False)
    r_v41_dual, syms_v41_d, fmap_v41_d = build_v4_1_asymmetric_buffered_rosters(mem_con, trade_dates, use_dual_channel_hs=True)
    mem_con.close()
    gc.collect()

    indexed_v40 = load_indexed_for_symbols(syms_v40 | {"sh000300"}, need_5m=False)
    bench_idx40 = indexed_v40["sh000300"]
    v40_keys = ["V4.0-A 零先验SQL盲选 (15%复利)", "V4.0-B 零先验SQL盲选 (12%复利)"]
    states_v40 = {k: {"cash": 1000000.0, "holdings": {}, "trades": [], "closed_trades": [], "equity": 1000000.0} for k in v40_keys}
    hist_v40 = {k: [] for k in v40_keys}
    t_p2a = time.time()

    for idx_d, dt in enumerate(trade_dates):
        day_cache = {}
        b_pos = bisect.bisect_right(bench_idx40["day_dates"], dt)
        bench_sl = bench_idx40["day"][max(0, b_pos - 250):b_pos]
        step_v4_0_zero_prior(states_v40["V4.0-A 零先验SQL盲选 (15%复利)"], r_v40[dt], fmap_v40, indexed_v40, bench_sl, dt, day_cache, dyn_sizing_pct=0.15)
        step_v4_0_zero_prior(states_v40["V4.0-B 零先验SQL盲选 (12%复利)"], r_v40[dt], fmap_v40, indexed_v40, bench_sl, dt, day_cache, dyn_sizing_pct=0.12)
        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        for k in v40_keys:
            tot = states_v40[k]["equity"]
            tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
            hist_v40[k].append({"date": dt, "portfolio_value": tot, "portfolio_return_pct": tot_r, "hs300_level": bmap[dt], "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)})
        if (idx_d + 1) % 400 == 0 or idx_d == len(trade_dates) - 1:
            print(f"   [Phase 2A V4.0] Day {idx_d+1}/{len(trade_dates)} ({dt}) | V4.0-A: {hist_v40['V4.0-A 零先验SQL盲选 (15%复利)'][-1]['portfolio_return_pct']:+.2f}% | Elapsed: {time.time()-t_p2a:.1f}s")

    for k in v40_keys:
        summary_out[k] = pack_model_result(k, states_v40[k], hist_v40[k], years)

    del indexed_v40, states_v40, hist_v40, r_v40
    gc.collect()

    # ================= PHASE 2B: V4.1 & V4.2 SERIES =================
    print("\n=== PHASE 2B: Running V4.1 & V4.2 Series ===")
    with open(base / "_ckpt_v3_v40.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False)

    # Reuse already-completed V4.2-C from backtest_v4_2_c_2020_2026.json!
    v42c_saved = json.load(open(base / "backtest_v4_2_c_2020_2026.json", "r", encoding="utf-8"))
    m_c = v42c_saved["metrics"]
    by_yr_c = defaultdict(list)
    for h in v42c_saved["history"]:
        by_yr_c[h["date"][:4]].append(h)
    yr_rets_c = {}
    prev_e_c = 1e6
    for yr in sorted(by_yr_c.keys()):
        end_e_c = by_yr_c[yr][-1]["portfolio_value"]
        yr_rets_c[yr] = round((end_e_c - prev_e_c) / prev_e_c * 100.0, 2)
        prev_e_c = end_e_c
    m_c["yearly_return_pct"] = yr_rets_c
    m_c["sector_pnl"] = {
        "HS300": v42c_saved["sector_pnl_attribution"]["沪深300"][1],
        "CYB": v42c_saved["sector_pnl_attribution"]["创业板"][1],
        "KCB": v42c_saved["sector_pnl_attribution"]["科创板"][1],
    }
    summary_out["V4.2-C 实战双通道版 (14:45隔离·15%复利)"] = {
        "metrics": m_c,
        "history": v42c_saved["history"],
    }
    del v42c_saved
    gc.collect()

    indexed_v41 = load_indexed_for_symbols(syms_v41_p | syms_v41_d | {"sh000300"}, need_5m=True)
    bench_idx41 = indexed_v41["sh000300"]
    v41_keys = [
        "V4.1-Dual 机构实验版 (双通道·平滑状态机)",
        "V4.1-PureLiq 机构实验版 (纯流动·平滑状态机)",
        "V4.2-A 实战纯流动版 (14:45隔离·15%复利)",
        "V4.2-B 实战纯流动版 (14:45隔离·12%复利)",
        "V4.2-D 实战双通道版 (14:45隔离·12%复利)",
    ]
    states_v41 = {k: {
        "cash": 1000000.0, "holdings": {}, "trades": [], "closed_trades": [],
        "equity": 1000000.0, "regime_history": [],
        "stats": {"limit_up_blocked": 0, "limit_down_two_step": 0, "fx_protect_blocked": 0}
    } for k in v41_keys}
    hist_v41 = {k: [] for k in v41_keys}
    t_p2b = time.time()

    for idx_d, dt in enumerate(trade_dates):
        day_cache = {}
        step_v4_1_institutional(states_v41["V4.1-Dual 机构实验版 (双通道·平滑状态机)"], r_v41_dual[dt], fmap_v41_d, indexed_v41, bench_idx41, dt, day_cache, dyn_sizing_pct=0.12)
        step_v4_1_institutional(states_v41["V4.1-PureLiq 机构实验版 (纯流动·平滑状态机)"], r_v41_pure[dt], fmap_v41_p, indexed_v41, bench_idx41, dt, day_cache, dyn_sizing_pct=0.12)
        step_v4_2_production(states_v41["V4.2-A 实战纯流动版 (14:45隔离·15%复利)"], r_v41_pure[dt], fmap_v41_p, indexed_v41, bench_idx41, dt, day_cache, dyn_sizing_pct=0.15, strict_1445_isolation=True)
        step_v4_2_production(states_v41["V4.2-B 实战纯流动版 (14:45隔离·12%复利)"], r_v41_pure[dt], fmap_v41_p, indexed_v41, bench_idx41, dt, day_cache, dyn_sizing_pct=0.12, strict_1445_isolation=True)
        step_v4_2_production(states_v41["V4.2-D 实战双通道版 (14:45隔离·12%复利)"], r_v41_dual[dt], fmap_v41_d, indexed_v41, bench_idx41, dt, day_cache, dyn_sizing_pct=0.12, strict_1445_isolation=True)

        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        for k in v41_keys:
            tot = states_v41[k]["equity"]
            tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
            hist_v41[k].append({"date": dt, "portfolio_value": tot, "portfolio_return_pct": tot_r, "hs300_level": bmap[dt], "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)})
        if (idx_d + 1) % 400 == 0 or idx_d == len(trade_dates) - 1:
            print(f"   [Phase 2B V4.1/V4.2] Day {idx_d+1}/{len(trade_dates)} ({dt}) | V4.2-A: {hist_v41['V4.2-A 实战纯流动版 (14:45隔离·15%复利)'][-1]['portfolio_return_pct']:+.2f}% | Elapsed: {time.time()-t_p2b:.1f}s")

    for k in v41_keys:
        summary_out[k] = pack_model_result(k, states_v41[k], hist_v41[k], years)

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

    print("\n================ 2020-01-02 ~ 2026-11-02 (6.81年) V3系列 vs V4系列 全模型对比总表 ================")
    for k in model_keys:
        m = summary_out[k]["metrics"]
        print(
            f"  {k:<36} | 期末:¥{m['final_value']/1e4:7.2f}万 ({m['total_return_pct']:+7.2f}%, 年化:{m['cagr_pct']:+6.2f}%) | "
            f"Alpha:{m['alpha_pct']:+7.2f}% | 最大回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:5.2f} | "
            f"胜率:{m['win_rate_pct']:5.2f}% | 盈亏比:{m['profit_loss_ratio']:4.2f}"
        )

    print("\n================ 2020 ~ 2026 各年度分年收益率矩阵 (%) ================")
    yrs_sorted = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    hdr = f"  {'模型版本':<36} | " + " | ".join(f"{y:>7}" for y in yrs_sorted)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in model_keys:
        yr_map = summary_out[k]["metrics"]["yearly_return_pct"]
        row_str = " | ".join(f"{yr_map.get(y, 0.0):+6.2f}%" for y in yrs_sorted)
        print(f"  {k:<36} | {row_str}")

    ordered_out = {k: summary_out[k] for k in model_keys}
    with open(base / "backtest_v3_v4_matrix_2020_2026.json", "w", encoding="utf-8") as f:
        json.dump(ordered_out, f, ensure_ascii=False, indent=2)

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    h0 = ordered_out[model_keys[0]]["history"]
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
        m = ordered_out[k]["metrics"]
        v = [x["portfolio_value"] / 10000.0 for x in ordered_out[k]["history"]]
        lw = 2.6 if ("V3.3-B" in k or "V3.3-A" in k) else 1.7
        ax1.plot(dates_dt, v, color=v3_colors[idx_k], linewidth=lw,
                 label=f"{k} (期末 ¥{v[-1]:.2f}万, {m['total_return_pct']:+.2f}% | 回撤 -{m['max_drawdown_pct']:.2f}% | 夏普 {m['sharpe_ratio']:.2f})")
    ax1.plot(dates_dt, v_bench, color="#cbd5e1", linewidth=1.5, linestyle=":", label="沪深300基准 (-3.26%)")
    ax1.axhline(100.0, color="#64748b", linewidth=1)
    ax1.set_title("ChanlunAIApp V3系列 (V3.0 ~ V3.3-C · 专家PIT精选池+三赛道动态主线) 2020~2026 净值对比", color="#f8fafc", fontsize=12.5, fontweight="bold")
    ax1.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    leg1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", fontsize=8.5)
    for t in leg1.get_texts(): t.set_color("#f8fafc")

    v4_colors = ["#fbbf24", "#f97316", "#f43f5e", "#e11d48", "#38bdf8", "#818cf8", "#10b981", "#a855f7"]
    for idx_k, k in enumerate(model_keys[6:]):
        m = ordered_out[k]["metrics"]
        v = [x["portfolio_value"] / 10000.0 for x in ordered_out[k]["history"]]
        lw = 2.6 if ("V4.2-C" in k or "V4.0-A" in k) else 1.6
        ax2.plot(dates_dt, v, color=v4_colors[idx_k], linewidth=lw,
                 label=f"{k} (期末 ¥{v[-1]:.2f}万, {m['total_return_pct']:+.2f}% | 回撤 -{m['max_drawdown_pct']:.2f}% | 夏普 {m['sharpe_ratio']:.2f})")
    ax2.plot(dates_dt, v_bench, color="#cbd5e1", linewidth=1.5, linestyle=":", label="沪深300基准 (-3.26%)")
    ax2.axhline(100.0, color="#64748b", linewidth=1)
    ax2.set_title("ChanlunAIApp V4系列 (V4.0 ~ V4.2-D · 全库5583只SQL零先验盲选+14:45物理隔离) 2020~2026 净值对比", color="#f8fafc", fontsize=12.5, fontweight="bold")
    ax2.set_ylabel("总净值 (万元)", color="#e2e8f0", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    leg2 = ax2.legend(loc="upper left", ncol=2, facecolor="#0f172a", edgecolor="#334155", fontsize=8.0)
    for t in leg2.get_texts(): t.set_color("#f8fafc")

    plt.tight_layout()
    out_img = base / "chanlun_v3_vs_v4_2020_2026.png"
    plt.savefig(out_img, dpi=180)
    print(f"Saved V3 vs V4 2020~2026 comparison chart to {out_img}")


if __name__ == "__main__":
    main()
