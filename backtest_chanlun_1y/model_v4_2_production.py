"""
ChanlunAIApp V4.2 实战定型版模型 (Production Edition)
----------------------------------------------------
融合 V4.0 与 V4.1 经消融实验验证的全部正向实战机制：
1. [模块1] 14:53:30 物理隔离快照 (<=14:45 前15根 + 14:53准第16根) -> 14:54:00 按 14:45 价格卖出结算 -> 14:59:45 按 15:00 收盘价+滑点买入；
2. [模块2精粹] 冻结开仓 15m 中枢极值 frozen_15m_dd 与 frozen_15m_zg (防止阴跌新中枢下移止损)，保留原版敏锐的 15m 二买/三买起涨点；
3. [模块3精粹] 果断五档波动率自适应牛熊阀门 (WEAK_DEEP 果断压至 35% 仓位/80分门槛，CRASH 0开仓) + 60日滚动低波抗跌动态 T1/T2 盾牌 + 二次函数连续乖离惩罚；
4. [模块4] 非对称分频调样 (主板半年求稳 + 双创月度 60/120日均量混合求快) + 进严出宽缓冲区 (Top35进/Top65出) + 剔除近10日天量过热股；
5. [模块5] 60m 主升浪首次 15m 顶背驰仅减仓 50% 锁利，剩余半仓按 max(15m ZG, 冻结15m ZG, 保本线) 移动止盈；
6. [模块6] 尾盘 0.25% 成交额微观限额 + 3~5 bps 冲击滑点 + 早盘 09:45/10:00 收盘均价无偏结算 + 1.8倍放量80%普涨 V 型反转解锁豁免。
"""
import bisect
from chanlun_engine import evaluate_asset, get_stage1_candidates
from duckdb_Backtest_engine import KCB_ACTIVE_DATE, get_limit_thresh, get_pit_bars_slice
from model_v4_0_zero_prior import (
    get_sector_of,
    classify_v4_adaptive_regime,
    compute_dynamic_defensive_tiers,
)
from model_v4_1_institutional import (
    get_v4_1_isolated_1453_slice,
    build_v4_1_asymmetric_buffered_rosters,
    compute_v4_1_quadratic_overext_penalties,
    calc_v4_1_trade_cost,
    resolve_v4_1_unbiased_limit_down_two_step,
)

def classify_v4_2_regime_with_v_unlock(bench_day_bars: list, day_snaps: dict, evaluated_cands: list) -> tuple:
    """
    结合 V4.0 果断的五档波动率自适应刹车阀门 与 V4.1 模块6 的史诗级 V 型反转解锁豁免
    """
    regime, bear_sev, r_cfg = classify_v4_adaptive_regime(bench_day_bars, day_snaps, evaluated_cands)
    if regime in ("WEAK", "CRASH") and day_snaps and len(bench_day_bars) >= 21:
        vols = [b.get("volume", 0.0) for b in bench_day_bars]
        avg_v20 = sum(vols[-21:-1]) / 20.0
        vol_ratio = (vols[-1] / avg_v20) if avg_v20 > 0 else 1.0
        valid_sn = [s for s in day_snaps.values() if s["lastPrice"] > 0]
        breadth = (sum(1 for s in valid_sn if s["pctChange"] > 0) / len(valid_sn)) if valid_sn else 0.0
        if vol_ratio >= 1.8 and breadth >= 0.80:
            return "BULL", "NONE", {
                "max_exposure_pct": 1.0,
                "daily_buy_limit": 2,
                "min_score": 45,
                "single_pos_scale": 1.0,
                "v_reversal_unlocked": True,
            }
    return regime, bear_sev, r_cfg

def step_v4_2_production(state: dict, rosters_dt: dict, fmap: dict, indexed: dict,
                         bench_idx: dict, dt: str, eval_cache: dict,
                         dyn_sizing_pct: float = 0.15, max_single_cap: float = 2500000.0,
                         strict_1445_isolation: bool = True):
    """
    V4.2 实战定型版单日执行引擎
    """
    holdings = state["holdings"]
    cash = state["cash"]
    sold_today = set()  # 记录当日已卖出标的，严禁 14:59:45 同日反手买回(Wash Sale)
    last_exits = state.setdefault("last_exits", {})  # 审计修复1(第3轮)：记录每只股票最近一次清仓的 (dt, sell_price)

    # --- 0. 早盘 09:15:00 ~ 10:00:00 两步法跌停分流无偏处置 (模块6) ---
    for sym in list(holdings.keys()):
        h = holdings[sym]
        if not h.get("pending_sell_reason"):
            continue
        idx_e = indexed.get(sym)
        dbar = idx_e.get("day_by_date", {}).get(dt) if idx_e else None
        can_sell, raw_exec_p, tag = resolve_v4_1_unbiased_limit_down_two_step(sym, dt, dbar, idx_e)
        if not can_sell:
            h["pending_sell_reason"] = h["pending_sell_reason"] + f"->{tag}"
            continue
        sell_sh = h.get("pending_sell_shares", h["shares"])
        exec_p, gross, fee = calc_v4_1_trade_cost("SELL", sym, h["kind"], dt, raw_exec_p, sell_sh)
        net = gross - fee
        cash += net
        pnl_amt = net - (sell_sh * h["avg_cost"])
        pnl_pct = (exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0
        state["closed_trades"].append({
            "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
            "shares": sell_sh, "buy_price": h["avg_cost"], "sell_price": exec_p,
            "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2),
            "reason": f"{h['pending_sell_reason']}->{tag}"
        })
        state["trades"].append({"date": dt, "action": "SELL", "symbol": sym, "name": h["name"], "shares": sell_sh, "price": exec_p, "amount": round(gross, 2), "fee": fee})
        sold_today.add(sym)
        if sell_sh >= h["shares"]:
            last_exits[sym] = (dt, exec_p)
            del holdings[sym]
        else:
            h["shares"] -= sell_sh
            h.pop("pending_sell_reason", None)
            h.pop("pending_sell_shares", None)

    # --- 1. 14:53:30 信号读取 (模块1：物理隔离 <=14:45 + 14:53准第16根) ---
    if strict_1445_isolation:
        bench_day_slice, _, _, _ = get_v4_1_isolated_1453_slice(bench_idx, dt)
    else:
        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]

    hs_u, cyb_u, kcb_u = rosters_dt["hs"], rosters_dt["cyb"], rosters_dt["kcb"]
    kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
    all_act = []
    seen_a = set()
    for m in (hs_u + cyb_u + (kcb_u if kcb_active else [])):
        if m[0] not in seen_a:
            all_act.append(m); seen_a.add(m[0])
    for s in holdings:
        if s in fmap and s not in seen_a:
            all_act.append(fmap[s]); seen_a.add(s)

    day_snaps = {}
    for sym, _, _ in all_act:
        idx_e = indexed.get(sym)
        if not idx_e or dt not in idx_e["day_by_date"]:
            continue
        if strict_1445_isolation:
            ck_s = ("ISO1453", sym, dt)
            if ck_s not in eval_cache:
                eval_cache[ck_s] = get_v4_1_isolated_1453_slice(idx_e, dt)
            _, _, _, sn = eval_cache[ck_s]
            if sn and sn["lastPrice"] > 0:
                day_snaps[sym] = sn
        else:
            if dt in idx_e["snaps"]:
                day_snaps[sym] = idx_e["snaps"][dt]

    # Stage 1：分赛道初筛保送
    sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=16)
    sel_cyb = get_stage1_candidates(cyb_u, day_snaps, {}, top_k=18 if kcb_active else 28)
    sel_kcb = get_stage1_candidates(kcb_u, day_snaps, {}, top_k=12) if kcb_active else []
    s1_seen = set(); stage1_metas = []
    for s in holdings:
        if s in fmap and s not in s1_seen:
            stage1_metas.append(fmap[s]); s1_seen.add(s)
    for m in (sel_cyb + sel_kcb + sel_hs):
        if m[0] not in s1_seen:
            stage1_metas.append(m); s1_seen.add(m[0])

    # Stage 2：三周期缠论结构评估
    evaluated = []
    for meta in stage1_metas:
        sym = meta[0]
        ck_e = ("EV_V42", strict_1445_isolation, sym, dt)
        if ck_e in eval_cache:
            ev = eval_cache[ck_e]
        else:
            idx_e = indexed.get(sym)
            if not idx_e:
                eval_cache[ck_e] = None; continue
            if strict_1445_isolation:
                d_b, m60_b, m15_b, sn = eval_cache[("ISO1453", sym, dt)]
            else:
                d_b, m60_b, m15_b = get_pit_bars_slice(idx_e, dt)
                sn = day_snaps.get(sym)
            if not d_b or not m15_b:
                eval_cache[ck_e] = None; continue
            ev = evaluate_asset(meta, d_b, m60_b, m15_b, sn)
            p15 = ev["periods"].get("15m")
            bi15 = p15.get("lastBi") if p15 else None
            zs15 = p15.get("lastZs") if p15 else None
            ev["bot_fx_low"] = bi15["low"] if bi15 else (zs15["dd"] if zs15 else d_b[-1]["low"])
            eval_cache[ck_e] = ev
        if ev is not None:
            evaluated.append(ev)

    evaluated.sort(key=lambda x: (x["score"], x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0, x["snapshot"]["amountYi"] if x["snapshot"] else 0.0), reverse=True)
    eval_map = {e["symbol"]: e for e in evaluated}
    regime, bear_sev, r_cfg = classify_v4_2_regime_with_v_unlock(bench_day_slice, day_snaps, evaluated)
    state["regime_history"].append((dt, f"{regime}({bear_sev})"))

    # --- 2. 14:54:00 卖出结算 (模块1按14:45价结算 + 模块2冻结15m_dd防下移 + 模块5 60m主升50%减半锁利与移动止盈) ---
    for sym in list(holdings.keys()):
        h = holdings[sym]
        if h.get("pending_sell_reason"):
            continue
        ev = eval_map.get(sym)
        dbar_1500 = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
        if not ev or not dbar_1500 or dbar_1500["close"] <= 0 or dbar_1500["volume"] <= 0:
            continue

        sell_p_ref = day_snaps[sym]["lastPrice"] if (strict_1445_isolation and sym in day_snaps) else dbar_1500["close"]
        sn_now = day_snaps.get(sym) or {"pctChange": dbar_1500["pctChange"]}
        p15 = ev["periods"].get("15m")
        p60 = ev["periods"].get("60m")
        if not p15:
            continue
        zs15 = p15.get("lastZs")
        bi15 = p15.get("lastBi")
        div15 = p15.get("divRatio")
        zs60 = p60.get("lastZs") if p60 else None
        bi60 = p60.get("lastBi") if p60 else None
        div60 = p60.get("divRatio") if p60 else None

        sell_rsn = None
        sell_ratio = 1.0

        # (a) 模块5：已执行过 50% 主升减半锁利的剩余半仓 -> 按移动止盈线 (含第一半仓锁利涨幅垫保护) 跟踪
        if h.get("half_profit_locked"):
            lock_p = h.get("half_lock_price", h["avg_cost"])
            profit_cushion = h["avg_cost"] + max(0.002 * h["avg_cost"], 0.35 * max(0.0, lock_p - h["avg_cost"]))
            trail_line = max(zs15["zg"] if zs15 else 0.0, h.get("frozen_15m_zg", 0.0), round(profit_cushion, 4))
            if sell_p_ref < trail_line or (bi60 and bi60["type"] == "up" and div60 is not None and div60 < 0.65):
                sell_rsn = f"[14:54半仓移动止盈] 跌破15mZG/60m背驰({round(trail_line,3)})"
                sell_ratio = 1.0

        # (b) 模块2 & 模块5：止损与 60m 主升分批锁利判断
        if not sell_rsn:
            # 冻结开仓 15m DD 下限，绝不允许因低位构造新中枢而使止损线无限下移
            eff_dd = max(zs15["dd"] if zs15 else 0.0, h.get("frozen_15m_dd", 0.0))
            if regime in ("WEAK", "CRASH") and sell_p_ref < h["init_stop"]:
                sell_rsn = f"[14:54防守锁死止损] 跌破初始止损({round(h['init_stop'],3)})"
            elif eff_dd > 0 and sell_p_ref < eff_dd:
                sell_rsn = f"[14:54冻结/结构破位止损] 跌破15m DD({round(eff_dd,3)})"
            elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.70:
                in_60m_main = bool(
                    zs60 and sell_p_ref > zs60["zg"]
                    and bi60 and bi60["type"] == "up" and (div60 is None or div60 >= 0.75)
                )
                # 审计合规修复1：科创板(sh688)最低申报 200股，触发50%减半锁利时科创板持仓须 >= 400股（非科创板 >= 200股）
                min_half_lock_sh = 400 if sym.startswith("sh688") else 200
                if in_60m_main and (not h.get("half_profit_locked")) and h["shares"] >= min_half_lock_sh and sell_p_ref > h["avg_cost"] * 1.01:
                    sell_ratio = 0.5
                    sell_rsn = f"[14:54主升浪减半锁利] 60m主升不背驰+15m一卖({div15})减仓50%"
                else:
                    sell_ratio = 1.0
                    sell_rsn = f"[14:54顶背驰一卖] 面积比={div15}"

        if sell_rsn:
            min_lot = 200 if sym.startswith("sh688") else 100
            sell_sh = h["shares"] if sell_ratio >= 0.99 else max(min_lot, (int(h["shares"] * 0.5) // 100) * 100)
            thresh = get_limit_thresh(sym, dt)
            if sn_now["pctChange"] <= -thresh:
                h["pending_sell_reason"] = f"{sell_rsn}({dt}尾盘14:54封跌停转次日早盘两步法)"
                h["pending_sell_shares"] = sell_sh
                state["stats"]["limit_down_two_step"] += 1
                continue

            exec_p, gross, fee = calc_v4_1_trade_cost("SELL", sym, h["kind"], dt, sell_p_ref, sell_sh)
            net = gross - fee
            cash += net
            pnl_amt = net - (sell_sh * h["avg_cost"])
            pnl_pct = (exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0
            state["closed_trades"].append({
                "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
                "shares": sell_sh, "buy_price": h["avg_cost"], "sell_price": exec_p,
                "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2), "reason": sell_rsn
            })
            state["trades"].append({"date": dt, "action": "SELL", "symbol": sym, "name": h["name"], "shares": sell_sh, "price": exec_p, "amount": round(gross, 2), "fee": fee, "reason": sell_rsn})
            sold_today.add(sym)
            if sell_sh >= h["shares"]:
                last_exits[sym] = (dt, exec_p)
                del holdings[sym]
            else:
                h["shares"] -= sell_sh
                h["half_profit_locked"] = True
                h["half_lock_price"] = sell_p_ref

    # --- 3. 14:59:45 买入结算 (模块3二次函数连续乖离惩罚 + 动态低波盾牌 + 模块6 0.25%限额与3~5bps滑点) ---
    cyb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "CYB"]
    kcb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "KCB"]
    cyb_str = (sum(sorted([x["score"] for x in cyb_cands], reverse=True)[:4]) / min(4, len(cyb_cands))) if cyb_cands else 0.0
    kcb_str = (sum(sorted([x["score"] for x in kcb_cands], reverse=True)[:4]) / min(4, len(kcb_cands))) if kcb_cands else 0.0
    growth_winner = "CYB" if cyb_str >= kcb_str else "KCB"

    dyn_t1, dyn_t2 = compute_dynamic_defensive_tiers(hs_u, indexed, bench_day_slice, dt) if regime == "WEAK" else (set(), set())
    overext_map = compute_v4_1_quadratic_overext_penalties(evaluated)
    def _get_raw_board_price(sym_code, qfq_p):
        d_rec = indexed.get(sym_code, {}).get("day_by_date", {}).get(dt)
        if d_rec and d_rec.get("raw_price", 0) > 0:
            return d_rec["raw_price"]
        return qfq_p

    all_raw_prices = sorted([_get_raw_board_price(x["symbol"], x["price"]) for x in evaluated if x["price"] > 0])
    high_price_p85 = all_raw_prices[int(len(all_raw_prices) * 0.85)] if len(all_raw_prices) >= 5 else 200.0

    def v4_2_rank(cand):
        sym = cand["symbol"]
        sec = get_sector_of(sym)
        base_s = cand["score"]
        p_day = cand["periods"].get("day")
        zs_day = p_day.get("lastZs") if p_day else None
        above_zg = bool(zs_day and p_day["close"] > zs_day["zg"])
        over_pen = overext_map.get(sym, 0.0)

        if regime in ("BULL", "RANGE"):
            if sec == growth_winner:
                g_boost = 16.0 if above_zg else 10.0
            elif sec in ("CYB", "KCB"):
                g_boost = 10.0 if (abs(cyb_str - kcb_str) <= 6.0 and above_zg) else 6.0
            else:
                g_boost = 0.0
            adj_s = base_s + g_boost - over_pen
        else:
            if bear_sev == "WEAK_DEEP":
                # 审计修复3：在 WEAK_DEEP 深度空头熊市，仅允许站上日线 ZG 的动态 T1 终极低波盾牌入场，严禁买双创高贝塔
                if sym in dyn_t1 and above_zg: adj_s = base_s + 15.0
                else: adj_s = base_s - 30.0
            else:
                if sym in dyn_t1 and above_zg: adj_s = base_s + 14.0
                elif sec == growth_winner and above_zg: adj_s = base_s + 10.0
                elif sym in dyn_t2 and above_zg: adj_s = base_s + 8.0
                else: adj_s = base_s - 12.0
            adj_s -= over_pen

        sn = cand["snapshot"]
        return (round(adj_s, 4), sn["buyerDom"] if sn else -999.0, sn["amountYi"] if sn else 0.0)

    ranked = sorted(evaluated, key=v4_2_rank, reverse=True)

    mkt_val_now = sum(h["shares"] * day_snaps.get(s, {"lastPrice": h["last_price"]})["lastPrice"] for s, h in holdings.items())
    cur_eq = cash + mkt_val_now
    rem_room = max(0.0, cur_eq * r_cfg["max_exposure_pct"] - mkt_val_now)
    min_sc = r_cfg["min_score"]
    pos_sc = r_cfg["single_pos_scale"]

    eff_stock_cap = min(max_single_cap, max(50000.0, cur_eq * dyn_sizing_pct))
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
            # 审计修复1：同日已触发卖出的股票严禁在 14:59:45 反手买回
            if sym in holdings or sym in sold_today or cand["score"] < min_sc or cand["price"] <= 0:
                continue

            # 审计修复1(第3轮)：防隔日无回调高位追回 (Anti-Whipsaw Cooldown)
            # 若距上次清仓不足 5 个交易日且当前价格高于上次卖出价 (未经历充分回调洗盘)，严禁在更高位追高接盘
            if sym in last_exits:
                last_dt, last_sp = last_exits[sym]
                idx_e_exit = indexed.get(sym)
                if idx_e_exit:
                    p_now = bisect.bisect_left(idx_e_exit["day_dates"], dt)
                    p_old = bisect.bisect_left(idx_e_exit["day_dates"], last_dt)
                    if 0 < (p_now - p_old) <= 5 and cand["price"] >= last_sp * 0.985:
                        continue

            # 审计修复2：买卖同源互斥——若当前正处于未获 60m 主升豁免的 15m 顶背驰一卖态或破 15m DD 态，严禁买入建仓
            p15_c = cand["periods"].get("15m")
            p60_c = cand["periods"].get("60m")
            bi15_c = p15_c.get("lastBi") if p15_c else None
            div15_c = p15_c.get("divRatio") if p15_c else None
            zs15_c = p15_c.get("lastZs") if p15_c else None
            zs60_c = p60_c.get("lastZs") if p60_c else None
            bi60_c = p60_c.get("lastBi") if p60_c else None
            div60_c = p60_c.get("divRatio") if p60_c else None
            if zs15_c and cand["price"] < zs15_c["dd"]:
                continue
            if bi15_c and bi15_c["type"] == "up" and div15_c is not None and div15_c < 0.70:
                in_60m_main_c = bool(
                    zs60_c and cand["price"] > zs60_c["zg"]
                    and bi60_c and bi60_c["type"] == "up" and (div60_c is None or div60_c >= 0.75)
                )
                if not in_60m_main_c:
                    continue

            # 审计修复4：日度实时校验近 10 日天量过热 (10日均量 > 2.6倍120日均量 且 10日涨幅 > 25%)，消除月中20天盲区
            idx_e_c = indexed.get(sym)
            if idx_e_c:
                pos_prev = bisect.bisect_left(idx_e_c["day_dates"], dt)
                if pos_prev >= 20:
                    bars_120 = idx_e_c["day"][max(0, pos_prev - 120):pos_prev]
                    bars_10 = bars_120[-10:]
                    avg_a120 = sum((b.get("amount") or 0.0) for b in bars_120) / len(bars_120)
                    avg_a10 = sum((b.get("amount") or 0.0) for b in bars_10) / len(bars_10)
                    ret_10 = (cand["price"] - bars_10[0]["close"]) / bars_10[0]["close"] if bars_10[0]["close"] > 0 else 0.0
                    if avg_a120 > 0 and (avg_a10 > 2.6 * avg_a120) and (ret_10 > 0.25):
                        continue
            if regime == "WEAK" and _get_raw_board_price(sym, cand["price"]) > high_price_p85:
                continue
            if regime == "WEAK" and bear_sev == "WEAK_DEEP":
                if (sym not in dyn_t1) or (cand["score"] < 90):
                    continue

            dbar_1500 = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if not dbar_1500 or dbar_1500["close"] <= 0 or dbar_1500["volume"] <= 0:
                continue

            thresh = get_limit_thresh(sym, dt)
            if dbar_1500["pctChange"] >= thresh and dbar_1500["close"] >= dbar_1500["high"] * 0.998:
                state["stats"]["limit_up_blocked"] += 1
                continue

            p15 = cand["periods"].get("15m")
            zs15 = p15.get("lastZs") if p15 else None
            # 审计修复1：保证初始止损至少留出 4.5% 呼吸空间，防止 15m 二买股价贴近 ZD 时出现 -0.3% 乌龙止损线
            raw_stop = round(min(zs15["zd"] * 0.99, cand["price"] * 0.955), 4) if zs15 else round(cand["price"] * 0.955, 4)
            bot_fx_low = cand.get("bot_fx_low", raw_stop)
            if dbar_1500["close"] < bot_fx_low * 0.995:
                state["stats"]["fx_protect_blocked"] += 1
                continue

            # 模块6：0.25% 成交额微观限额 + 3~5 bps 动态冲击滑点
            sn = cand.get("snapshot")
            full_day_amt = (sn["amountYi"] * 1e8) if (sn and sn["amountYi"] > 0) else (dbar_1500.get("amount") or (dbar_1500["volume"] * dbar_1500["close"]))
            micro_liq_cap = max(150000.0, full_day_amt * 0.0025)

            t_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_sc, micro_liq_cap)
            alloc = min(budget * 0.95, t_amt, rem_room)
            # 审计修复4：按真实除权盘面交易价校验单手门槛，按等比前复权价格记账保持净值连续
            raw_board_p = _get_raw_board_price(sym, dbar_1500["close"])
            min_board_sh = 200 if sym.startswith("sh688") else 100
            if alloc < raw_board_p * min_board_sh:
                continue
            lots = int(alloc // (max(cand["price"], dbar_1500["close"]) * 100.0))
            shares = lots * 100
            # 审计合规修复1：上交所科创板(sh688)单笔委托必须 >= 200 股
            if shares < min_board_sh:
                continue

            order_ratio = (shares * dbar_1500["close"]) / max(1.0, full_day_amt)
            impact_bps = min(5.0, max(3.0, 3.0 + (order_ratio / 0.0025) * 2.0))
            exec_p, gross, fee = calc_v4_1_trade_cost("BUY", sym, cand["kind"], dt, dbar_1500["close"], shares, impact_bps=impact_bps)
            tot_cost = gross + fee
            if tot_cost > cash:
                continue

            cash -= tot_cost
            budget -= tot_cost
            rem_room -= gross
            bought += 1
            holdings[sym] = {
                "name": cand["name"], "kind": cand["kind"], "shares": shares,
                "avg_cost": round(tot_cost / shares, 4), "last_price": dbar_1500["close"],
                "buy_date": dt, "entry_regime": regime, "init_stop": raw_stop,
                "frozen_bot_fx_low": round(bot_fx_low, 4),
                "frozen_15m_zg": round(zs15["zg"], 4) if zs15 else 0.0,
                "frozen_15m_dd": round(zs15["dd"], 4) if zs15 else 0.0,
                "half_profit_locked": False,
            }
            state["trades"].append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": exec_p, "amount": round(gross, 2), "fee": fee})

    for sym, h in holdings.items():
        dbar_1500 = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
        if dbar_1500 and dbar_1500["close"] > 0:
            h["last_price"] = dbar_1500["close"]

    state["cash"] = cash
    state["equity"] = round(cash + sum(h["shares"] * h["last_price"] for h in holdings.values()), 2)
