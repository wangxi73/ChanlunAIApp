"""
全模型版本 (V1.0 ~ V3.3) 1年 / 5年 / 10年 实盘级回测总矩阵
严格遵循：
1. 唯一数据源：直接通过 duckdb.connect("/Volumes/Lexar-2T/A_Share_10Y_Klines/a_share_10y.duckdb", read_only=True) 读取 kline_1d, kline_60m, kline_15m, stocks
2. 禁止全区间未来函数：严格按 date <= dt (bisect_right) 截断历史 K 线，直接使用表中自带 amount
3. 实盘级尾盘黄金6分钟交易 + 早盘两步法跌停分流处置 + 2023-08-28 分段真实税费与滑点：
   - 早盘 09:15~10:00 (两步法处置昨日尾盘封跌停未卖出的持仓)：
     * 情况A (09:20后仍一字跌停)：挂跌停价排队清仓，若全天一字封死无成交则继续顺延，若盘中打开跌停则按跌停价成交；
     * 情况B (竞价打开跌停，低开在-2%~-7%等非一字跌停)：不在09:25核按钮割在最低点，转入09:30~10:00观察前15~30分钟分时：
       若开盘向下急杀不拉回分时均线或跌破昨日跌停价则无条件市价止损；若出现水下承接反抽则在第一波反抽高点100%卖出释放资金！
   - 尾盘 14:53:30 读取当日缠论/AI信号 -> 14:54:00 卖出触发离场的持仓并当场释放资金 -> 14:59:45 校验缠论底分型保护位并在集合竞价 100% 买入！
"""
import bisect
import json
from pathlib import Path
from chanlun_engine import evaluate_asset, get_stage1_candidates
from pit_universe import select_pit_universe_at_date
from run_backtest import compute_metrics
from run_backtest_10y_pit import with_held_metas
from model_v3_2_mainline import (
    HS300_DEFENSIVE_T1, HS300_DEFENSIVE_T2, get_sector_of,
    compute_sector_mainline_boosts, get_bear_severity_level
)
from duckdb_Backtest_engine import (
    load_duckdb_universe_data, get_pit_bars_slice,
    get_limit_thresh, calc_trade_cost, KCB_ACTIVE_DATE
)
import market_regime_v2

base = Path(__file__).resolve().parent

def is_tail_limit_up(symbol: str, dt: str, day_bar: dict) -> bool:
    if not day_bar or day_bar["close"] <= 0 or day_bar["preClose"] <= 0:
        return True
    thresh = get_limit_thresh(symbol, dt)
    if day_bar["pctChange"] >= thresh and day_bar["close"] >= day_bar["high"] * 0.998:
        return True
    return False

def is_tail_limit_down(symbol: str, dt: str, day_bar: dict) -> bool:
    if not day_bar or day_bar["close"] <= 0 or day_bar["preClose"] <= 0:
        return True
    thresh = get_limit_thresh(symbol, dt)
    if day_bar["pctChange"] <= -thresh and day_bar["close"] <= day_bar["low"] * 1.002:
        return True
    return False

def get_morning_15m_bars(idx_e: dict, dt: str) -> list:
    """获取交易日 dt 早盘 09:30~10:00 的前两根 15m K 线 (09:45, 10:00)"""
    if not idx_e:
        return []
    l_pos = bisect.bisect_left(idx_e["m15_dates"], dt)
    r_pos = bisect.bisect_right(idx_e["m15_dates"], dt)
    if l_pos >= r_pos:
        return []
    return idx_e["m15_tuples"][l_pos:min(l_pos + 2, r_pos)]

def resolve_limit_down_two_step(symbol: str, dt: str, day_bar: dict, idx_e: dict) -> tuple:
    """
    次日早盘两步法跌停分流处置 (09:15~10:00):
    返回 (can_sell_today: bool, exec_raw_price: float, tag: str)
    """
    if not day_bar or day_bar["open"] <= 0 or day_bar["volume"] <= 0:
        return False, 0.0, "停牌顺延"

    pre_c = day_bar["preClose"]
    o, h, l, c = day_bar["open"], day_bar["high"], day_bar["low"], day_bar["close"]
    thresh = get_limit_thresh(symbol, dt)
    open_pct = ((o - pre_c) / pre_c * 100.0) if pre_c > 0 else 0.0

    # 第一步：集合竞价阶段 (09:15:00 - 09:25:00) 分流处置
    # 情况 A：突发暴雷 / 09:20 后仍巨单封死一字跌停 -> 在 09:15:00 挂跌停价排队清仓
    if open_pct <= -thresh:
        if h <= l * 1.001:
            # 全天一字封死跌停，排队未能成交，顺延下一交易日继续挂跌停排队
            return False, 0.0, f"{dt}一字跌停封死排队未成交"
        else:
            # 盘中翘板打开跌停，09:15排队的跌停单按跌停开盘价成交清仓
            return True, o, f"{dt}情况A(竞价跌停排队翘板清仓)"

    # 情况 B：情绪错杀 / 竞价打开跌停(如低开在 -2% ~ -7% 之间)，不在 09:25 竞价直接核按钮割在最低点
    # 第二步：早盘连续竞价阶段 (09:30:00 - 10:00:00) 1分钟/5分钟/15分钟次级别择机卖出释放资金
    m_bars = get_morning_15m_bars(idx_e, dt)
    if m_bars:
        # m_bars[0] 为 09:45 (前15分钟): (dtm, open, high, low, close, volume)
        b1_o, b1_h, b1_l, b1_c = m_bars[0][1], m_bars[0][2], m_bars[0][3], m_bars[0][4]
        b2_h = m_bars[1][2] if len(m_bars) >= 2 else b1_h
        morning_high = max(b1_h, b2_h)
        vwap_approx = (b1_o + b1_h + b1_l + b1_c) / 4.0

        # 情形 B1：开盘即向下急杀且不拉回分时均线，或跌破昨日收盘价(昨日跌停价) -> 无条件市价止损清仓
        if (b1_c < b1_o and b1_h < vwap_approx * 1.003) or (b1_l < pre_c * 0.995):
            stop_p = round((b1_o + b1_c) / 2.0, 4)
            return True, stop_p, f"{dt}情况B1(早盘急杀破分时均线市价止损)"
        else:
            # 情形 B2：开盘出现水下承接反抽（向上试探分时均线或翻红）-> 在早盘第一波反抽高点立刻卖出100%仓位
            rebound_p = round(b1_o + (morning_high - b1_o) * 0.75, 4)
            return True, rebound_p, f"{dt}情况B2(早盘水下承接第一波反抽高点卖出)"
    else:
        # 兜底：基于日线高低点判断早盘反抽或止损
        if c < o and h <= o * 1.005:
            return True, round((o + l) / 2.0, 4), f"{dt}情况B1(早盘急杀市价止损)"
        return True, round(o + (h - o) * 0.60, 4), f"{dt}情况B2(早盘反抽高点卖出)"

def precompute_all_pit_rosters(hs_pool, cyb_pool, kcb_pool, all_a_pool, indexed, trade_dates):
    full_meta_map = {m[0]: m for m in (all_a_pool + hs_pool + cyb_pool + kcb_pool)}
    rosters = {}
    cur_hs = select_pit_universe_at_date(hs_pool, indexed, trade_dates[0], top_n=100, min_listed_days=60)
    cur_cyb = select_pit_universe_at_date(cyb_pool, indexed, trade_dates[0], top_n=50, min_listed_days=60)
    cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, trade_dates[0], top_n=35, min_listed_days=20) if trade_dates[0] >= KCB_ACTIVE_DATE else []
    cur_aa = select_pit_universe_at_date(all_a_pool, indexed, trade_dates[0], top_n=200, min_listed_days=60)

    for i, dt in enumerate(trade_dates):
        if dt >= KCB_ACTIVE_DATE and not cur_kcb:
            cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, dt, top_n=35, min_listed_days=20)
        rosters[dt] = {"hs": cur_hs, "cyb": cur_cyb, "kcb": cur_kcb, "alla": cur_aa}

        if i + 1 < len(trade_dates):
            cur_ym, nxt_ym = dt[:7], trade_dates[i + 1][:7]
            if cur_ym != nxt_ym and (cur_ym.endswith("-06") or cur_ym.endswith("-12")):
                cur_hs = select_pit_universe_at_date(hs_pool, indexed, dt, top_n=100, min_listed_days=60)
                cur_cyb = select_pit_universe_at_date(cyb_pool, indexed, dt, top_n=50, min_listed_days=60)
                if dt >= KCB_ACTIVE_DATE:
                    cur_kcb = select_pit_universe_at_date(kcb_pool, indexed, dt, top_n=35, min_listed_days=20)
                cur_aa = select_pit_universe_at_date(all_a_pool, indexed, dt, top_n=200, min_listed_days=60)

    return rosters, full_meta_map

def eval_symbol_on_date(meta, indexed, dt, eval_cache):
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
    bot_fx_low = bi15["low"] if bi15 else (zs15["dd"] if zs15 else d_bars[-1]["low"])
    ev["bot_fx_low"] = bot_fx_low
    eval_cache[ck] = ev
    return ev

def step_sub_tail_auction(state, pool_metas, indexed, bench_day_slice, dt, eval_cache,
                          base_stock_cap=250000.0, dyn_sizing_pct=None, max_single_cap=2500000.0,
                          use_regime=True, use_ratchet_stop=True, route_mode="none",
                          tiered_rosters=None, fmap=None):
    holdings = state["holdings"]
    cash = state["cash"]

    # --- 0. 早盘 09:15:00 ~ 10:00:00 两步法跌停分流处置（针对前一日 14:54:00 因封死跌停未能卖出的持仓） ---
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
        full_rsn = f"{h['pending_sell_reason']} -> {tag}"
        state["closed_trades"].append({
            "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
            "shares": h["shares"], "buy_price": h["avg_cost"], "sell_price": exec_p,
            "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2), "reason": full_rsn
        })
        state["trades"].append({"date": dt, "action": "SELL", "symbol": sym, "shares": h["shares"], "price": exec_p, "amount": round(gross, 2), "fee": fee})
        del holdings[sym]

    # 更新其余持仓最新收盘价
    for sym, h in holdings.items():
        dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
        if dbar and dbar["close"] > 0:
            h["last_price"] = dbar["close"]

    # --- 1. 14:53:30 读取当日缠论/AI信号 ---
    if tiered_rosters is not None:
        hs_u, cyb_u, kcb_u = tiered_rosters["hs"], tiered_rosters["cyb"], tiered_rosters["kcb"]
        kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
        all_act = with_held_metas(hs_u + cyb_u + (kcb_u if kcb_active else []), holdings, fmap)
        day_snaps = {s: indexed[s]["snaps"][dt] for s, _, _ in all_act if s in indexed and dt in indexed[s]["snaps"]}
        sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=16)
        sel_cyb = get_stage1_candidates(cyb_u, day_snaps, {}, top_k=18 if kcb_active else 28)
        sel_kcb = get_stage1_candidates(kcb_u, day_snaps, {}, top_k=12) if kcb_active else []
        seen = set()
        stage1 = []
        for s in holdings:
            if s in fmap and s not in seen:
                stage1.append(fmap[s]); seen.add(s)
        for m in (sel_cyb + sel_kcb + sel_hs):
            if m[0] not in seen:
                stage1.append(m); seen.add(m[0])
    else:
        u_pool = with_held_metas(pool_metas, holdings, fmap)
        day_snaps = {s: indexed[s]["snaps"][dt] for s, _, _ in u_pool if s in indexed and dt in indexed[s]["snaps"]}
        stage1 = get_stage1_candidates(u_pool, day_snaps, holdings, top_k=48)

    evaluated = []
    for meta in stage1:
        ev = eval_symbol_on_date(meta, indexed, dt, eval_cache)
        if ev is not None:
            evaluated.append(ev)
    evaluated.sort(key=lambda x: (x["score"], x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0, x["snapshot"]["amountYi"] if x["snapshot"] else 0.0), reverse=True)
    eval_map = {e["symbol"]: e for e in evaluated}

    if use_regime:
        regime, r_cfg, _ = market_regime_v2.classify_market_regime(bench_day_slice, day_snaps, evaluated, mode="dual_factor")
    else:
        regime = "BULL"
        r_cfg = {"label": "无控仓", "max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0, "max_stop_pct": 0.08}

    # --- 2. 14:54:00 卖出触发离场的持仓，当场释放可用资金 ---
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
                if use_ratchet_stop and regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
                    sell_rsn = f"[14:54熊市锁死止损] 破初始止损({h['init_stop']})"
                elif zs15 and ev["price"] < zs15["dd"]:
                    sell_rsn = f"[14:54结构破位止损] 破15m DD({zs15['dd']})"
                elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                    sell_rsn = f"[14:54顶背驰一卖] 面积比={div15}"

        if sell_rsn:
            # 若 14:54:00 尾盘封死跌停卖不出 -> 转入次日早盘 09:15~10:00 两步法跌停分流处置！
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
            state["trades"].append({"date": dt, "action": "SELL", "symbol": sym, "shares": h["shares"], "price": exec_p, "amount": round(gross, 2), "fee": fee})
            del holdings[sym]

    # --- 3. 14:59:45 校验缠论底分型保护位并在集合竞价 100% 买入 ---
    sec_boosts, main_winner = compute_sector_mainline_boosts(evaluated) if route_mode in ("v3_2", "v3_3_day", "v3_3_m60") else ({"HS300": 0, "CYB": 0, "KCB": 0}, "HS300")
    bear_sev = get_bear_severity_level(bench_day_slice) if (use_regime and regime == "WEAK") else "NONE"

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
        elif route_mode in ("v3_2", "v3_3_day", "v3_3_m60"):
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
        else:
            adj_s = base_s
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

            # 14:59:45 校验缠论底分型保护位
            if dbar["close"] < bot_fx_low * 0.995:
                continue

            snap = cand.get("snapshot")
            liq_cap = max(250000.0, snap["amountYi"] * 1e8 * 0.01) if (snap and snap["amountYi"] > 0) else max_single_cap
            t_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_sc, liq_cap)
            alloc = min(budget * 0.95, t_amt, rem_room)
            lots = int(alloc // (dbar["close"] * 1.001 * 100.0))
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
            state["trades"].append({"date": dt, "action": "BUY", "symbol": sym, "shares": shares, "price": exec_p, "amount": round(gross, 2), "fee": fee})

    state["cash"] = cash
    mkt_val = sum(h["shares"] * h["last_price"] for h in holdings.values())
    state["equity"] = round(cash + mkt_val, 2)

def make_sub(init_cash):
    return {"cash": init_cash, "holdings": {}, "trades": [], "closed_trades": [], "equity": init_cash}

def run_unified_model(model_cfg, rosters, fmap, indexed, trade_dates, bmap, eval_cache):
    market_regime_v2._BENCH_CACHE.clear()
    st = make_sub(1000000.0)
    bench_idx = indexed["sh000300"]
    init_b = bmap[trade_dates[0]]
    hist = []

    for dt in trade_dates:
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        sl = bench_idx["day"][max(0, b_pos - 250):b_pos]
        r_dt = rosters[dt]
        if model_cfg["pool"] == "hs":
            p_metas = r_dt["hs"]; t_rost = None
        elif model_cfg["pool"] == "alla":
            p_metas = r_dt["alla"]; t_rost = None
        elif model_cfg["pool"] == "unified_3":
            k_act = (dt >= KCB_ACTIVE_DATE) and (len(r_dt["kcb"]) >= 8)
            seen = set(); p_metas = []
            for m in (r_dt["hs"] + r_dt["cyb"] + (r_dt["kcb"] if k_act else [])):
                if m[0] not in seen: p_metas.append(m); seen.add(m[0])
            t_rost = None
        else:
            p_metas = None; t_rost = r_dt

        step_sub_tail_auction(
            st, p_metas, indexed, sl, dt, eval_cache,
            base_stock_cap=model_cfg["base_cap"], dyn_sizing_pct=model_cfg["dyn_pct"],
            use_regime=model_cfg["use_regime"], use_ratchet_stop=model_cfg["use_stop"],
            route_mode=model_cfg["route_mode"], tiered_rosters=t_rost, fmap=fmap
        )
        tot = st["equity"]
        tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        hist.append({"date": dt, "portfolio_value": tot, "portfolio_return_pct": tot_r, "hs300_level": bmap[dt], "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)})

    res = {"initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"], "history": hist, "trades": st["trades"], "closed_trades": st["closed_trades"]}
    m = compute_metrics(res)
    pk = max(hist, key=lambda x: x["portfolio_value"])
    tr = min(hist, key=lambda x: x["portfolio_value"])
    m["peak_value"] = pk["portfolio_value"]; m["peak_date"] = pk["date"]
    m["trough_value"] = tr["portfolio_value"]; m["trough_date"] = tr["date"]
    return m, hist

def run_split_model(weights, pool_keys, rosters, fmap, indexed, trade_dates, bmap, eval_cache, use_regime=True, use_stop=True):
    market_regime_v2._BENCH_CACHE.clear()
    is_3sec = (len(weights) == 3)
    pre_kcb = is_3sec and (trade_dates[0] < KCB_ACTIVE_DATE)
    if pre_kcb:
        s_list = [make_sub(1e6 * weights[0]), make_sub(1e6 * (weights[1] + weights[2])), make_sub(0.0)]
    else:
        s_list = [make_sub(1e6 * w) for w in weights]
    kcb_seeded = not pre_kcb
    kcb_xfer = 0.0
    bench_idx = indexed["sh000300"]
    init_b = bmap[trade_dates[0]]
    hist = []

    for i, dt in enumerate(trade_dates):
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        sl = bench_idx["day"][max(0, b_pos - 250):b_pos]
        r_dt = rosters[dt]

        if is_3sec:
            k_act = (dt >= KCB_ACTIVE_DATE) and (len(r_dt["kcb"]) >= 8)
            if k_act and not kcb_seeded:
                comb = s_list[1]["equity"] + s_list[2]["equity"]
                kcb_xfer = comb * (weights[2] / (weights[1] + weights[2]))
                kcb_seeded = True
            if k_act and kcb_xfer > 100.0 and s_list[1]["cash"] > 100.0:
                xf = min(kcb_xfer, s_list[1]["cash"])
                s_list[1]["cash"] -= xf; s_list[2]["cash"] += xf; kcb_xfer -= xf
                s_list[1]["equity"] = round(s_list[1]["cash"] + sum(h["shares"]*h["last_price"] for h in s_list[1]["holdings"].values()), 2)
                s_list[2]["equity"] = round(s_list[2]["cash"] + sum(h["shares"]*h["last_price"] for h in s_list[2]["holdings"].values()), 2)
            cur_w = [weights[0], weights[1] if k_act else (weights[1]+weights[2]), weights[2] if k_act else 0.0]
        else:
            cur_w = weights

        for sub, pk, w in zip(s_list, pool_keys, cur_w):
            if w > 0 or sub["holdings"]:
                step_sub_tail_auction(
                    sub, r_dt[pk], indexed, sl, dt, eval_cache,
                    base_stock_cap=max(250000.0 * w, 25000.0) if w > 0 else 0.0,
                    dyn_sizing_pct=None, use_regime=use_regime, use_ratchet_stop=use_stop,
                    route_mode="none", tiered_rosters=None, fmap=fmap
                )

        if i + 1 < len(trade_dates) and dt[:7].endswith("-10") and dt[:7] != trade_dates[i + 1][:7]:
            tot_eq = sum(s["equity"] for s in s_list)
            surplus = 0.0; defs = []
            for sub, w in zip(s_list, cur_w):
                diff = sub["equity"] - tot_eq * w
                if diff > 0:
                    g = min(diff, sub["cash"]); sub["cash"] -= g; surplus += g
                elif diff < 0:
                    defs.append((sub, -diff))
            tdef = sum(d for _, d in defs)
            if tdef > 0 and surplus > 0:
                for sub, d in defs: sub["cash"] += surplus * (d / tdef)
            for sub in s_list:
                sub["equity"] = round(sub["cash"] + sum(h["shares"]*h["last_price"] for h in sub["holdings"].values()), 2)

        tot = round(sum(s["equity"] for s in s_list), 2)
        tot_r = round((tot - 1e6) / 1e6 * 100.0, 2)
        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        hist.append({"date": dt, "portfolio_value": tot, "portfolio_return_pct": tot_r, "hs300_level": bmap[dt], "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)})

    all_t = [t for s in s_list for t in s["trades"]]
    all_c = [c for s in s_list for c in s["closed_trades"]]
    res = {"initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"], "history": hist, "trades": all_t, "closed_trades": all_c}
    m = compute_metrics(res)
    pk = max(hist, key=lambda x: x["portfolio_value"])
    tr = min(hist, key=lambda x: x["portfolio_value"])
    m["peak_value"] = pk["portfolio_value"]; m["peak_date"] = pk["date"]
    m["trough_value"] = tr["portfolio_value"]; m["trough_date"] = tr["date"]
    return m, hist

if __name__ == "__main__":
    indexed, fmap, hs_pool, cyb_pool, kcb_pool, all_a_pool = load_duckdb_universe_data()
    bench_bars = indexed["sh000300"]["day"]
    bmap = {b["time"]: b["close"] for b in bench_bars}
    all_dates = [b["time"] for b in bench_bars if b["time"] >= "2016-10-28"]
    rosters_all, _ = precompute_all_pit_rosters(hs_pool, cyb_pool, kcb_pool, all_a_pool, indexed, all_dates)

    eval_cache = {}
    horizons = [
        ("1Y", [d for d in all_dates if d >= "2025-10-30"]),
        ("5Y", [d for d in all_dates if d >= "2021-10-29"]),
        ("10Y", [d for d in all_dates if d >= "2016-10-28"]),
    ]

    models_spec = [
        ("V1.0-① 原版纯沪深300", "unified", {"pool": "hs", "base_cap": 250000.0, "dyn_pct": None, "use_regime": False, "use_stop": False, "route_mode": "none"}),
        ("V1.0-② 原版纯全A", "unified", {"pool": "alla", "base_cap": 250000.0, "dyn_pct": None, "use_regime": False, "use_stop": False, "route_mode": "none"}),
        ("V1.5 双池40:60(无控仓)", "split2", {"weights": [0.4, 0.6], "pools": ["hs", "alla"], "use_regime": False, "use_stop": False}),
        ("V2.0 双池40:60+牛熊控仓", "split2", {"weights": [0.4, 0.6], "pools": ["hs", "alla"], "use_regime": True, "use_stop": True}),
        ("V2.2 三板块20:50:30+控仓", "split3", {"weights": [0.2, 0.5, 0.3], "pools": ["hs", "cyb", "kcb"], "use_regime": True, "use_stop": True}),
        ("V3.0 稳健统一池(12.5万8仓)", "unified", {"pool": "unified_3", "base_cap": 125000.0, "dyn_pct": None, "use_regime": True, "use_stop": True, "route_mode": "v3_0"}),
        ("V3.1 分层保送池(25万4仓)", "unified", {"pool": "tiered_3", "base_cap": 250000.0, "dyn_pct": None, "use_regime": True, "use_stop": True, "route_mode": "v3_1"}),
        ("V3.2 动态主线+红利盾牌(25万)", "unified", {"pool": "tiered_3", "base_cap": 250000.0, "dyn_pct": None, "use_regime": True, "use_stop": True, "route_mode": "v3_2"}),
        ("V3.3-A 终极版(15%扩容+日线过滤)", "unified", {"pool": "tiered_3", "base_cap": 250000.0, "dyn_pct": 0.15, "use_regime": True, "use_stop": True, "route_mode": "v3_3_day"}),
        ("V3.3-B 终极版(15%扩容+60m过滤)", "unified", {"pool": "tiered_3", "base_cap": 250000.0, "dyn_pct": 0.15, "use_regime": True, "use_stop": True, "route_mode": "v3_3_m60"}),
    ]

    matrix_out = {}
    for h_key, td in horizons:
        print(f"\n==================== 周期: {h_key} ({td[0]} ~ {td[-1]}, {len(td)} 交易日) ====================")
        matrix_out[h_key] = {}
        for m_name, m_kind, cfg in models_spec:
            if m_kind == "unified":
                m, hist = run_unified_model(cfg, rosters_all, fmap, indexed, td, bmap, eval_cache)
            else:
                m, hist = run_split_model(cfg["weights"], cfg["pools"], rosters_all, fmap, indexed, td, bmap, eval_cache, cfg["use_regime"], cfg["use_stop"])
            matrix_out[h_key][m_name] = {"metrics": m, "history": hist}
            print(f"{m_name:30s} | 期末:¥{m['final_value']/1e4:7.2f}w ({m['total_return_pct']:+7.2f}%) | Alpha:{m['alpha_pct']:+7.2f}% | 最高:¥{m['peak_value']/1e4:7.2f}w | 最低:¥{m['trough_value']/1e4:6.2f}w | 回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:5.2f} | 胜率:{m['win_rate_pct']:5.2f}% | 盈亏比:{m['profit_loss_ratio']:4.2f}")

    with open(base / "backtest_matrix_duckdb_1y_5y_10y.json", "w", encoding="utf-8") as f:
        json.dump(matrix_out, f, ensure_ascii=False, indent=2)
    print("\nSaved full matrix to backtest_matrix_duckdb_1y_5y_10y.json")
