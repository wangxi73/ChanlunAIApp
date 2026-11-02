"""
V3.3 模型 1年 / 5年 / 10年 实盘级回测 (唯一数据源 DuckDB unified_ratio_qfq)
1. 唯一数据源：duckdb.connect("/Volumes/Lexar-2T/A_Share_10Y_Klines/a_share_10y.duckdb", read_only=True)
2. 严格 PIT 截断：每个交易日 dt 仅读取 date <= dt 的历史 K 线及表中自带 amount
3. 交易规则：
   - 早盘 09:15~10:00 两步法跌停分流处置 (情况A一字跌停排队 / 情况B早盘第一波反抽高点或破均线止损)
   - 尾盘 14:53:30 读取信号 -> 14:54:00 卖出释放资金 -> 14:59:45 校验缠论底分型保护位并在集合竞价 100% 买入
   - 税费：按 2023-08-28 节点分段真实印花税(前千1/后万5) + 万2.5佣金
"""
import bisect
import json
import time
from pathlib import Path
import duckdb
from chanlun_engine import evaluate_asset, get_stage1_candidates
from pit_universe import select_pit_universe_at_date
from pit_universe_20y import get_20y_candidate_pools
from run_backtest import compute_metrics
from run_backtest_10y_pit import with_held_metas
from model_v3_2_mainline import (
    HS300_DEFENSIVE_T1, HS300_DEFENSIVE_T2, get_sector_of,
    compute_sector_mainline_boosts, get_bear_severity_level
)
import market_regime_v2

DB_PATH = "/Volumes/Lexar-2T/A_Share_10Y_Klines/a_share_10y.duckdb"
KCB_ACTIVE_DATE = "2019-08-26"
base = Path(__file__).resolve().parent

def get_limit_thresh(symbol: str, dt: str) -> float:
    if symbol.startswith("sh688") or symbol == "sh588000":
        return 19.6
    if (symbol.startswith("sz30") or symbol == "sz159915") and dt >= "2020-08-24":
        return 19.6
    return 9.75

def is_tail_limit_up(symbol: str, dt: str, day_bar: dict) -> bool:
    if not day_bar or day_bar["close"] <= 0 or day_bar["preClose"] <= 0:
        return True
    thresh = get_limit_thresh(symbol, dt)
    return day_bar["pctChange"] >= thresh and day_bar["close"] >= day_bar["high"] * 0.998

def is_tail_limit_down(symbol: str, dt: str, day_bar: dict) -> bool:
    if not day_bar or day_bar["close"] <= 0 or day_bar["preClose"] <= 0:
        return True
    thresh = get_limit_thresh(symbol, dt)
    return day_bar["pctChange"] <= -thresh and day_bar["close"] <= day_bar["low"] * 1.002

def calc_trade_cost(action: str, symbol: str, kind: str, dt: str, raw_price: float, shares: int) -> tuple:
    if action == "BUY":
        exec_price = round(raw_price, 4)
        gross = exec_price * shares
        comm = max(gross * 0.00025, 5.0)
        return exec_price, gross, round(comm, 2)
    else:
        exec_price = round(raw_price * (1.0 - 0.0002), 4)
        gross = exec_price * shares
        comm = max(gross * 0.00025, 5.0)
        if kind == "etf" or symbol in ("sh510300", "sh588000", "sz159915"):
            stamp_tax = 0.0
        else:
            stamp_rate = 0.0005 if dt >= "2023-08-28" else 0.0010
            stamp_tax = gross * stamp_rate
        return exec_price, gross, round(comm + stamp_tax, 2)

def resolve_limit_down_two_step(symbol: str, dt: str, day_bar: dict, idx_e: dict) -> tuple:
    if not day_bar or day_bar["open"] <= 0 or day_bar["volume"] <= 0:
        return False, 0.0, "停牌顺延"
    pre_c = day_bar["preClose"]
    o, h, l, c = day_bar["open"], day_bar["high"], day_bar["low"], day_bar["close"]
    thresh = get_limit_thresh(symbol, dt)
    open_pct = ((o - pre_c) / pre_c * 100.0) if pre_c > 0 else 0.0

    # 第一步 (09:15~09:25)：情况A 一字跌停挂单排队
    if open_pct <= -thresh:
        if h <= l * 1.001:
            return False, 0.0, f"{dt}一字跌停封死排队未成"
        return True, o, f"{dt}情况A(竞价跌停排队翘板清仓)"

    # 第二步 (09:30~10:00)：情况B 早盘前15~30分钟分时择机反抽或破位止损
    l_pos = bisect.bisect_left(idx_e["m15_dates"], dt)
    r_pos = bisect.bisect_right(idx_e["m15_dates"], dt)
    m_bars = idx_e["m15_tuples"][l_pos:min(l_pos + 2, r_pos)] if l_pos < r_pos else []
    if m_bars:
        b1_o, b1_h, b1_l, b1_c = m_bars[0][1], m_bars[0][2], m_bars[0][3], m_bars[0][4]
        b2_h = m_bars[1][2] if len(m_bars) >= 2 else b1_h
        morning_high = max(b1_h, b2_h)
        vwap_approx = (b1_o + b1_h + b1_l + b1_c) / 4.0
        if (b1_c < b1_o and b1_h < vwap_approx * 1.003) or (b1_l < pre_c * 0.995):
            return True, round((b1_o + b1_c) / 2.0, 4), f"{dt}情况B1(早盘急杀破均线止损)"
        return True, round(b1_o + (morning_high - b1_o) * 0.75, 4), f"{dt}情况B2(早盘第一波反抽高点卖出)"
    return True, round(o + (h - o) * 0.60, 4), f"{dt}情况B2(早盘反抽卖出)"

def load_duckdb_data():
    t0 = time.time()
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA disable_progress_bar;")

    hs_pool, cyb_pool, kcb_pool = get_20y_candidate_pools()
    core_rows = con.execute("SELECT symbol, name FROM stocks WHERE in_core = TRUE").fetchall()
    db_name_map = {r[0]: r[1] for r in core_rows}

    hs_pool = [m for m in hs_pool if m[0] in db_name_map]
    cyb_pool = [m for m in cyb_pool if m[0] in db_name_map]
    kcb_pool = [m for m in kcb_pool if m[0] in db_name_map]
    full_meta_map = {m[0]: m for m in (hs_pool + cyb_pool + kcb_pool)}
    target_syms = sorted(list(full_meta_map.keys()) + ["sh000300"])
    sym_list_sql = ",".join(f"'{s}'" for s in target_syms)

    print(f"Loading unified_ratio_qfq data from DuckDB for {len(target_syms)} symbols...")
    bdom_rows = con.execute(f"""
        SELECT symbol, date,
               SUM(volume * CASE WHEN high > low THEN (close - low)/(high - low) WHEN close >= open THEN 0.55 ELSE 0.45 END) AS b_vol,
               SUM(volume) AS tot_vol
        FROM kline_15m
        WHERE symbol IN ({sym_list_sql})
        GROUP BY symbol, date
    """).fetchall()
    bdom_map = {(sym, dt): round((2.0 * bv - tv) / tv * 100.0, 1) for sym, dt, bv, tv in bdom_rows if tv and tv > 0}

    rows_1d = con.execute(f"""
        SELECT symbol, date, open, high, low, close, volume, amount, hfq_factor, ratio_factor
        FROM kline_1d
        WHERE symbol IN ({sym_list_sql}) AND close > 0
        ORDER BY symbol, date
    """).fetchall()

    rows_60m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume
        FROM kline_60m
        WHERE symbol IN ({sym_list_sql}) AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()

    rows_15m = con.execute(f"""
        SELECT symbol, date, datetime, open, high, low, close, volume
        FROM kline_15m
        WHERE symbol IN ({sym_list_sql}) AND close > 0
        ORDER BY symbol, datetime
    """).fetchall()
    con.close()

    indexed = {s: {
        "day": [], "day_dates": [], "day_by_date": {},
        "m60_tuples": [], "m60_dates": [],
        "m15_tuples": [], "m15_dates": [],
        "snaps": {}
    } for s in target_syms}

    for sym, dt, o, h, l, c, v, amt, hfq_f, rat_f in rows_1d:
        e = indexed[sym]
        prev_c = e["day"][-1]["close"] if e["day"] else o
        pct = ((c - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
        bar = {"symbol": sym, "time": dt, "open": o, "high": h, "low": l, "close": c, "volume": v, "amount": amt, "preClose": prev_c, "pctChange": pct, "hfq_factor": hfq_f, "ratio_factor": rat_f}
        e["day"].append(bar)
        e["day_dates"].append(dt)
        e["day_by_date"][dt] = bar
        if sym != "sh000300":
            amt_yi = (amt / 1e8) if (amt and amt > 0) else ((v * c) / 1e8)
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

    print(f"Loaded DuckDB unified_ratio_qfq dataset in {time.time() - t0:.2f}s!")
    return indexed, full_meta_map, hs_pool, cyb_pool, kcb_pool

def get_pit_bars_slice(idx_e: dict, dt: str):
    i_d = bisect.bisect_right(idx_e["day_dates"], dt)
    i_60 = bisect.bisect_right(idx_e["m60_dates"], dt)
    i_15 = bisect.bisect_right(idx_e["m15_dates"], dt)
    d_bars = idx_e["day"][max(0, i_d - 250):i_d]
    raw_60 = idx_e["m60_tuples"][max(0, i_60 - 280):i_60]
    raw_15 = idx_e["m15_tuples"][max(0, i_15 - 280):i_15]
    if not d_bars or not raw_15:
        return d_bars, [], []
    m60_bars = [{"time": t[0], "open": t[1], "high": t[2], "low": t[3], "close": t[4], "volume": t[5]} for t in raw_60]
    m15_bars = [{"time": t[0], "open": t[1], "high": t[2], "low": t[3], "close": t[4], "volume": t[5]} for t in raw_15]
    return d_bars, m60_bars, m15_bars

def precompute_rosters(hs_pool, cyb_pool, kcb_pool, indexed, trade_dates):
    rosters = {}
    cur_hs = select_pit_universe_at_date(hs_pool, indexed, trade_dates[0], top_n=100, min_listed_days=60)
    cur_cyb = select_pit_universe_at_date(cyb_pool, indexed, trade_dates[0], top_n=50, min_listed_days=60)
    cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, trade_dates[0], top_n=35, min_listed_days=20) if trade_dates[0] >= KCB_ACTIVE_DATE else []

    for i, dt in enumerate(trade_dates):
        if dt >= KCB_ACTIVE_DATE and not cur_kcb:
            cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, dt, top_n=35, min_listed_days=20)
        rosters[dt] = (cur_hs, cur_cyb, cur_kcb)
        if i + 1 < len(trade_dates):
            cur_ym, nxt_ym = dt[:7], trade_dates[i + 1][:7]
            if cur_ym != nxt_ym and (cur_ym.endswith("-06") or cur_ym.endswith("-12")):
                cur_hs = select_pit_universe_at_date(hs_pool, indexed, dt, top_n=100, min_listed_days=60)
                cur_cyb = select_pit_universe_at_date(cyb_pool, indexed, dt, top_n=50, min_listed_days=60)
                if dt >= KCB_ACTIVE_DATE:
                    cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, dt, top_n=35, min_listed_days=20)
    return rosters

def eval_symbol_pit(meta, indexed, dt, eval_cache):
    sym = meta[0]
    ck = (sym, dt)
    if ck in eval_cache:
        return eval_cache[ck]
    idx_e = indexed.get(sym)
    if not idx_e:
        eval_cache[ck] = None
        return None
    d_bars, m60_bars, m15_bars = get_pit_bars_slice(idx_e, dt)
    if not d_bars or not m15_bars:
        eval_cache[ck] = None
        return None
    snap = idx_e["snaps"].get(dt)
    ev = evaluate_asset(meta, d_bars, m60_bars, m15_bars, snap)
    p15 = ev["periods"].get("15m")
    bi15 = p15.get("lastBi") if p15 else None
    zs15 = p15.get("lastZs") if p15 else None
    ev["bot_fx_low"] = bi15["low"] if bi15 else (zs15["dd"] if zs15 else d_bars[-1]["low"])
    eval_cache[ck] = ev
    return ev

def simulate_v3_3(name, rosters_by_date, fmap, indexed, trade_dates, bmap, eval_cache,
                  base_stock_cap=250000.0, dyn_sizing_pct=0.15, overext_mode="day_38"):
    market_regime_v2._BENCH_CACHE.clear()
    market_regime_v2.REGIME_CONFIGS["BULL"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["RANGE"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["WEAK"].update({"max_exposure_pct": 0.70, "daily_buy_limit": 1, "min_score": 60, "single_pos_scale": 0.65})
    market_regime_v2.REGIME_CONFIGS["CRASH"].update({"max_exposure_pct": 0.0, "daily_buy_limit": 0, "min_score": 999, "single_pos_scale": 0.0})

    initial_capital = 1000000.0
    cash = initial_capital
    holdings = {}
    trades = []
    closed_trades = []
    regime_counts = {}
    mainline_wins = {"HS300": 0, "CYB": 0, "KCB": 0}
    limit_stats = {"limit_up_blocked_buys": 0, "limit_down_two_step_sells": 0, "fx_protect_blocked_buys": 0}
    init_bench = bmap[trade_dates[0]]
    bench_idx = indexed["sh000300"]
    history = []

    for dt in trade_dates:
        # --- 0. 早盘 09:15:00 ~ 10:00:00 两步法跌停分流处置 ---
        for sym in list(holdings.keys()):
            h = holdings[sym]
            if not h.get("pending_sell_reason"):
                continue
            idx_e = indexed.get(sym)
            dbar = idx_e.get("day_by_date", {}).get(dt) if idx_e else None
            can_sell, raw_exec_p, tag = resolve_limit_down_two_step(sym, dt, dbar, idx_e)
            if not can_sell:
                h["pending_sell_reason"] = h["pending_sell_reason"] + f"->{tag}"
                continue
            exec_p, gross, fee = calc_trade_cost("SELL", sym, h["kind"], dt, raw_exec_p, h["shares"])
            net = gross - fee
            cash += net
            pnl_amt = net - (h["shares"] * h["avg_cost"])
            pnl_pct = (exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0
            full_rsn = f"{h['pending_sell_reason']}->{tag}"
            closed_trades.append({
                "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
                "shares": h["shares"], "buy_price": h["avg_cost"], "sell_price": exec_p,
                "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2), "reason": full_rsn
            })
            trades.append({"date": dt, "action": "SELL", "symbol": sym, "name": h["name"], "shares": h["shares"], "price": exec_p, "amount": round(gross, 2), "fee": fee, "reason": full_rsn})
            del holdings[sym]

        for sym, h in holdings.items():
            dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if dbar and dbar["close"] > 0:
                h["last_price"] = dbar["close"]

        # --- 1. 14:53:30 读取当日缠论/AI信号 (严格 date <= dt) ---
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]
        hs_u, cyb_u, kcb_u = rosters_by_date[dt]
        kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
        all_act = with_held_metas(hs_u + cyb_u + (kcb_u if kcb_active else []), holdings, fmap)
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

        evaluated = []
        for meta in stage1:
            ev = eval_symbol_pit(meta, indexed, dt, eval_cache)
            if ev is not None:
                evaluated.append(ev)
        evaluated.sort(key=lambda x: (x["score"], x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0, x["snapshot"]["amountYi"] if x["snapshot"] else 0.0), reverse=True)
        eval_map = {e["symbol"]: e for e in evaluated}

        regime, r_cfg, _ = market_regime_v2.classify_market_regime(bench_day_slice, day_snaps, evaluated, mode="dual_factor")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # --- 2. 14:54:00 卖出触发离场的持仓并当场释放资金 ---
        for sym in list(holdings.keys()):
            h = holdings[sym]
            if h.get("pending_sell_reason"):
                continue
            ev = eval_map.get(sym)
            dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if not dbar or dbar["close"] <= 0 or dbar["volume"] <= 0 or not ev:
                continue

            p_m15 = ev["periods"].get("15m")
            if not p_m15:
                continue
            zs15 = p_m15.get("lastZs")
            bi15 = p_m15.get("lastBi")
            div15 = p_m15.get("divRatio")
            sell_rsn = None
            if regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
                sell_rsn = f"[14:54熊市锁死止损] 破初始止损({h['init_stop']})"
            elif zs15 and ev["price"] < zs15["dd"]:
                sell_rsn = f"[14:54结构破位止损] 破15m DD({zs15['dd']})"
            elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                # 确认处于向上反弹/拉升段而非日内瀑布暴跌底部的假一卖
                if dbar["pctChange"] > -3.5 or ev["price"] >= h["avg_cost"]:
                    sell_rsn = f"[14:54顶背驰一卖] 面积比={div15}"

            if sell_rsn:
                if is_tail_limit_down(sym, dt, dbar):
                    h["pending_sell_reason"] = f"{sell_rsn}({dt}尾盘14:54封跌停转次日早盘两步法)"
                    limit_stats["limit_down_two_step_sells"] += 1
                    continue
                exec_p, gross, fee = calc_trade_cost("SELL", sym, h["kind"], dt, dbar["close"], h["shares"])
                net = gross - fee
                cash += net
                pnl_amt = net - (h["shares"] * h["avg_cost"])
                pnl_pct = (exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0
                closed_trades.append({
                    "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
                    "shares": h["shares"], "buy_price": h["avg_cost"], "sell_price": exec_p,
                    "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2), "reason": sell_rsn
                })
                trades.append({"date": dt, "action": "SELL", "symbol": sym, "name": h["name"], "shares": h["shares"], "price": exec_p, "amount": round(gross, 2), "fee": fee, "reason": sell_rsn})
                del holdings[sym]

        # --- 3. 14:59:45 校验缠论底分型保护位并在集合竞价 100% 买入 ---
        sec_boosts, main_winner = compute_sector_mainline_boosts(evaluated)
        mainline_wins[main_winner] += 1
        bear_sev = get_bear_severity_level(bench_day_slice) if regime == "WEAK" else "NONE"

        # Compute growth mainline winner between CYB and KCB
        cyb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "CYB"]
        kcb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "KCB"]
        cyb_str = (sum(sorted([x["score"] for x in cyb_cands], reverse=True)[:4]) / min(4, len(cyb_cands))) if cyb_cands else 0.0
        kcb_str = (sum(sorted([x["score"] for x in kcb_cands], reverse=True)[:4]) / min(4, len(kcb_cands))) if kcb_cands else 0.0
        growth_winner = "CYB" if cyb_str >= kcb_str else "KCB"

        def v3_3_rank(cand):
            sym = cand["symbol"]
            sec = get_sector_of(sym)
            base_s = cand["score"]
            p_day = cand["periods"].get("day")
            zs_day = p_day.get("lastZs") if p_day else None
            above_zg = bool(zs_day and p_day["close"] > zs_day["zg"])
            over_pen = 0
            if overext_mode == "day_38" and zs_day and zs_day["zg"] > 0:
                d_zg = (p_day["close"] - zs_day["zg"]) / zs_day["zg"]
                if d_zg > 0.38: over_pen = 25
                elif d_zg > 0.28: over_pen = 10
            elif overext_mode == "m60_25":
                p60 = cand["periods"].get("60m")
                zs60 = p60.get("lastZs") if p60 else None
                if zs60 and zs60["zg"] > 0 and (cand["price"] - zs60["zg"]) / zs60["zg"] > 0.25:
                    over_pen = 20

            if regime in ("BULL", "RANGE"):
                # 牛市/结构市：在20cm双创赛道中聚焦当日领涨主线(第一成长主线+16分、次成长主线+10分)，严控低弹性10cm蓝筹高频损耗
                if sec == growth_winner:
                    g_boost = 16 if above_zg else 10
                elif sec in ("CYB", "KCB"):
                    g_boost = 10 if (abs(cyb_str - kcb_str) <= 6.0 and above_zg) else 6
                else:
                    g_boost = 0
                adj_s = base_s + g_boost - over_pen
            else:
                # 熊市/弱势防守期：大幅提高门槛减少交易，仅在T1终极红利盾牌与极强逆势主线龙头中严选
                if bear_sev == "WEAK_DEEP":
                    if sym in HS300_DEFENSIVE_T1 and above_zg: adj_s = base_s + 18
                    elif sym in HS300_DEFENSIVE_T2 and above_zg: adj_s = base_s + 10
                    elif sec == growth_winner and above_zg: adj_s = base_s + 8
                    else: adj_s = base_s - 15
                    if cand["price"] > 150.0: adj_s -= 30
                else:
                    if sym in HS300_DEFENSIVE_T1 and above_zg: adj_s = base_s + 14
                    elif sec == growth_winner and above_zg: adj_s = base_s + 12
                    elif sym in HS300_DEFENSIVE_T2 and above_zg: adj_s = base_s + 8
                    else: adj_s = base_s - 8
                    if cand["price"] > 200.0: adj_s -= 25
                adj_s -= over_pen
            return (adj_s, cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0, cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0)

        ranked = sorted(evaluated, key=v3_3_rank, reverse=True)
        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_eq = cash + mkt_val_now

        # 熊市严格控仓少交易 (WEAK_DEEP最高35%仓位、门槛80分；WEAK_MILD最高55%仓位、门槛75分)
        eff_max_exp = 0.35 if bear_sev == "WEAK_DEEP" else (0.55 if regime == "WEAK" else r_cfg["max_exposure_pct"])
        rem_room = max(0.0, cur_eq * eff_max_exp - mkt_val_now)
        min_sc = 80 if bear_sev == "WEAK_DEEP" else (75 if regime == "WEAK" else r_cfg["min_score"])
        pos_sc = 0.50 if bear_sev == "WEAK_DEEP" else (0.65 if regime == "WEAK" else r_cfg["single_pos_scale"])

        eff_stock_cap = min(2500000.0, max(50000.0, cur_eq * dyn_sizing_pct)) if dyn_sizing_pct else base_stock_cap
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
                if regime == "WEAK" and cand["price"] > (150.0 if bear_sev == "WEAK_DEEP" else 220.0):
                    continue
                dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
                if not dbar or dbar["close"] <= 0 or dbar["volume"] <= 0:
                    continue
                if is_tail_limit_up(sym, dt, dbar):
                    limit_stats["limit_up_blocked_buys"] += 1
                    continue

                p15 = cand["periods"].get("15m")
                zs15 = p15.get("lastZs") if p15 else None
                raw_stop = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
                bot_fx_low = cand.get("bot_fx_low", raw_stop)

                # 14:59:45 校验缠论底分型保护位
                if dbar["close"] < bot_fx_low * 0.995:
                    limit_stats["fx_protect_blocked_buys"] += 1
                    continue

                snap = cand.get("snapshot")
                liq_cap = max(250000.0, snap["amountYi"] * 1e8 * 0.01) if (snap and snap["amountYi"] > 0) else 2500000.0
                t_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_sc, liq_cap)
                alloc = min(budget * 0.95, t_amt, rem_room)
                lots = int(alloc // (dbar["close"] * 100.0))
                shares = lots * 100
                if shares < 100:
                    continue

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
                trades.append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": exec_p, "amount": round(gross, 2), "fee": fee})

        hs_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if get_sector_of(s) == "HS300")
        cyb_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if get_sector_of(s) == "CYB")
        kcb_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if get_sector_of(s) == "KCB")
        tot_val = round(cash + hs_val + cyb_val + kcb_val, 2)
        tot_ret = round((tot_val - initial_capital) / initial_capital * 100.0, 2)
        hs300_lvl = bmap[dt]
        hs300_ret = round((hs300_lvl - init_bench) / init_bench * 100.0, 2)
        history.append({
            "date": dt, "portfolio_value": tot_val, "cash": round(cash, 2),
            "hs300_mkt_val": round(hs_val, 2), "cyb_mkt_val": round(cyb_val, 2), "kcb_mkt_val": round(kcb_val, 2),
            "holdings_count": len(holdings), "regime": regime,
            "portfolio_return_pct": tot_ret, "hs300_level": hs300_lvl,
            "hs300_return_pct": hs300_ret, "alpha_pct": round(tot_ret - hs300_ret, 2)
        })

    sec_stats = {"沪深300": [0, 0.0], "创业板": [0, 0.0], "科创板": [0, 0.0]}
    for c in closed_trades:
        sec_lbl = {"HS300": "沪深300", "CYB": "创业板", "KCB": "科创板"}[get_sector_of(c["symbol"])]
        sec_stats[sec_lbl][0] += 1
        sec_stats[sec_lbl][1] = round(sec_stats[sec_lbl][1] + c["pnl_amt"], 2)

    res = {
        "account_name": name, "initial_capital": initial_capital,
        "final_value": history[-1]["portfolio_value"], "final_cash": round(cash, 2),
        "regime_counts": regime_counts, "mainline_days": mainline_wins,
        "limit_stats": limit_stats, "sector_pnl_attribution": sec_stats,
        "history": history, "trades": trades, "closed_trades": closed_trades
    }
    m = compute_metrics(res)
    pk = max(history, key=lambda x: x["portfolio_value"])
    tr = min(history, key=lambda x: x["portfolio_value"])
    m["peak_value"] = pk["portfolio_value"]; m["peak_return_pct"] = pk["portfolio_return_pct"]; m["peak_date"] = pk["date"]
    m["trough_value"] = tr["portfolio_value"]; m["trough_return_pct"] = tr["portfolio_return_pct"]; m["trough_date"] = tr["date"]
    res["metrics"] = m
    return res

if __name__ == "__main__":
    indexed, fmap, hs_pool, cyb_pool, kcb_pool = load_duckdb_data()
    bench_bars = indexed["sh000300"]["day"]
    bmap = {b["time"]: b["close"] for b in bench_bars}
    all_dates = [b["time"] for b in bench_bars if b["time"] >= "2016-10-28"]
    rosters_all = precompute_rosters(hs_pool, cyb_pool, kcb_pool, indexed, all_dates)

    eval_cache = {}
    horizons = [
        ("1Y", [d for d in all_dates if d >= "2025-10-30"]),
        ("5Y", [d for d in all_dates if d >= "2021-10-29"]),
        ("10Y", [d for d in all_dates if d >= "2016-10-28"]),
    ]

    v33_variants = [
        ("V3.3-A (15%动态扩容 + 日线透支过滤)", 250000.0, 0.15, "day_38"),
        ("V3.3-B (15%动态扩容 + 60m脉冲过滤)", 250000.0, 0.15, "m60_25"),
        ("V3.3-C (12%动态扩容 + 日线透支过滤·均衡)", 160000.0, 0.12, "day_38"),
    ]

    out_results = {}
    for h_key, td in horizons:
        print(f"\n==================== V3.3 模型回测周期: {h_key} ({td[0]} ~ {td[-1]}, {len(td)} 交易日) ====================")
        out_results[h_key] = {}
        for v_name, b_cap, d_pct, o_mode in v33_variants:
            res = simulate_v3_3(v_name, rosters_all, fmap, indexed, td, bmap, eval_cache, b_cap, d_pct, o_mode)
            m = res["metrics"]
            out_results[h_key][v_name] = res
            print(f"{v_name:40s} | 期末:¥{m['final_value']/1e4:7.2f}w ({m['total_return_pct']:+7.2f}%) | 沪深300:{m['hs300_return_pct']:+6.2f}% | Alpha:{m['alpha_pct']:+7.2f}% | 最高:¥{m['peak_value']/1e4:7.2f}w ({m['peak_date']}) | 最低:¥{m['trough_value']/1e4:6.2f}w | 回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:5.2f} | 胜率:{m['win_rate_pct']:5.2f}% | 盈亏比:{m['profit_loss_ratio']:4.2f}")
            print(f"  板块盈亏贡献: {res['sector_pnl_attribution']} | 涨跌停/分型过滤: {res['limit_stats']}")

    with open(base / "backtest_v3_3_duckdb_1_5_10y.json", "w", encoding="utf-8") as f:
        json.dump(out_results, f, ensure_ascii=False, indent=2)
    print("\nSaved V3.3 1Y/5Y/10Y results to backtest_v3_3_duckdb_1_5_10y.json")
