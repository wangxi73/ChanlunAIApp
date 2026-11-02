"""
V4.0 -> V4.1 六大模块单因子与递进消融实验 (1Y: 2025-10-30 ~ 2026-10-29)
精确量化每个模块单独开启与组合开启对 V4.0 (+17.85%) 的影响
"""
import bisect
import json
from pathlib import Path
from run_backtest import compute_metrics
from run_v4_0_1y import load_v4_1y_duckdb
from model_v4_0_zero_prior import step_v4_0_zero_prior, get_sector_of
from model_v4_1_institutional import (
    get_v4_1_isolated_1453_slice,
    verify_15m_strong_bottom_and_macd_shrink,
    get_v4_1_zscore_stage1_candidates,
    compute_v4_1_zscore_shield_scores,
    compute_v4_1_continuous_regime,
    compute_v4_1_quadratic_overext_penalties,
    calc_v4_1_trade_cost,
    resolve_v4_1_unbiased_limit_down_two_step,
)
from chanlun_engine import evaluate_asset, get_stage1_candidates
from duckdb_Backtest_engine import KCB_ACTIVE_DATE, get_pit_bars_slice, get_limit_thresh, calc_trade_cost
from run_all_models_matrix import is_tail_limit_up, is_tail_limit_down, resolve_limit_down_two_step
from model_v4_0_zero_prior import classify_v4_adaptive_regime, compute_dynamic_defensive_tiers, compute_cross_sectional_overext_penalties

def run_configurable_v4(name, rosters_by_date, fmap, indexed, trade_dates, flags, eval_cache):
    """
    flags: dict with keys:
      m1_1445_iso: bool (模块1: 14:45 物理隔离快照 vs 15:00 全天K线)
      m2_strong_fx_freeze: bool (模块2: 强底分型双缩脚 + 冻结契约止损)
      m3_zscore_cont: bool (模块3: Z-score 初筛 + 连续状态机 R_t + 二次乖离)
      m5_half_profit: bool (模块5: 60m 主升浪首次 15m 一卖减半锁利 + 移动止盈)
      m6_micro_friction: bool (模块6: 0.25% 成交额限额 + 3~5bps滑点 + 早盘收盘无偏结算)
    """
    bench_idx = indexed["sh000300"]
    bmap = {b["time"]: b["close"] for b in bench_idx["day"]}
    init_b = bmap[trade_dates[0]]

    cash = 1000000.0
    holdings = {}
    trades = []
    closed_trades = []
    hist = []

    for dt in trade_dates:
        # --- 0. 早盘跌停两步法 ---
        for sym in list(holdings.keys()):
            h = holdings[sym]
            if not h.get("pending_sell_reason"):
                continue
            idx_e = indexed.get(sym)
            dbar = idx_e.get("day_by_date", {}).get(dt) if idx_e else None
            if flags.get("m6_micro_friction"):
                can_s, raw_p, tag = resolve_v4_1_unbiased_limit_down_two_step(sym, dt, dbar, idx_e)
            else:
                can_s, raw_p, tag = resolve_limit_down_two_step(sym, dt, dbar, idx_e)
            if not can_s:
                h["pending_sell_reason"] += f"->{tag}"
                continue
            sell_sh = h.get("pending_sell_shares", h["shares"])
            if flags.get("m6_micro_friction"):
                exec_p, gross, fee = calc_v4_1_trade_cost("SELL", sym, h["kind"], dt, raw_p, sell_sh)
            else:
                exec_p, gross, fee = calc_trade_cost("SELL", sym, h["kind"], dt, raw_p, sell_sh)
            net = gross - fee
            cash += net
            closed_trades.append({
                "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
                "shares": sell_sh, "buy_price": h["avg_cost"], "sell_price": exec_p,
                "pnl_amt": round(net - sell_sh * h["avg_cost"], 2),
                "pnl_pct": round((exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0, 2),
                "reason": f"{h['pending_sell_reason']}->{tag}"
            })
            trades.append({"date": dt, "action": "SELL", "symbol": sym, "shares": sell_sh, "price": exec_p, "amount": round(gross, 2), "fee": fee})
            if sell_sh >= h["shares"]:
                del holdings[sym]
            else:
                h["shares"] -= sell_sh
                h.pop("pending_sell_reason", None); h.pop("pending_sell_shares", None)

        for sym, h in holdings.items():
            dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if dbar and dbar["close"] > 0:
                h["last_price"] = dbar["close"]

        # --- 1. 信号快照 (M1 开启则用 14:45 隔离切片，关闭则用 15:00 全天切片) ---
        if flags.get("m1_1445_iso"):
            bench_day_slice, _, _, _ = get_v4_1_isolated_1453_slice(bench_idx, dt)
        else:
            b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
            bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]

        r_dt = rosters_by_date[dt]
        hs_u, cyb_u, kcb_u = r_dt["hs"], r_dt["cyb"], r_dt["kcb"]
        kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
        all_act = hs_u + cyb_u + (kcb_u if kcb_active else [])
        seen_a = {m[0] for m in all_act}
        for s in holdings:
            if s in fmap and s not in seen_a:
                all_act.append(fmap[s]); seen_a.add(s)

        day_snaps = {}
        for s, _, _ in all_act:
            if s not in indexed or dt not in indexed[s]["day_by_date"]:
                continue
            if flags.get("m1_1445_iso"):
                ck_s = ("ISO1445", s, dt)
                if ck_s not in eval_cache:
                    eval_cache[ck_s] = get_v4_1_isolated_1453_slice(indexed[s], dt)
                _, _, _, sn = eval_cache[ck_s]
                if sn and sn["lastPrice"] > 0:
                    day_snaps[s] = sn
            else:
                if dt in indexed[s]["snaps"]:
                    day_snaps[s] = indexed[s]["snaps"][dt]

        # Stage 1 初筛
        if flags.get("m3_zscore_cont"):
            stage1 = get_v4_1_zscore_stage1_candidates({"hs": hs_u, "cyb": cyb_u, "kcb": kcb_u if kcb_active else []}, day_snaps, holdings, top_k=46)
        else:
            sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=16)
            sel_cyb = get_stage1_candidates(cyb_u, day_snaps, {}, top_k=18 if kcb_active else 28)
            sel_kcb = get_stage1_candidates(kcb_u, day_snaps, {}, top_k=12) if kcb_active else []
            s1_seen = set(); stage1 = []
            for s in holdings:
                if s in fmap and s not in s1_seen:
                    stage1.append(fmap[s]); s1_seen.add(s)
            for m in (sel_cyb + sel_kcb + sel_hs):
                if m[0] not in s1_seen:
                    stage1.append(m); s1_seen.add(m[0])

        # Stage 2 评估
        evaluated = []
        for meta in stage1:
            sym = meta[0]
            ck_e = ("EV_CFG", flags.get("m1_1445_iso", False), flags.get("m2_strong_fx_freeze", False), sym, dt)
            if ck_e in eval_cache:
                ev = eval_cache[ck_e]
            else:
                idx_e = indexed.get(sym)
                if not idx_e:
                    eval_cache[ck_e] = None; continue
                if flags.get("m1_1445_iso"):
                    d_b, m60_b, m15_b, sn = eval_cache[("ISO1445", sym, dt)]
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
                ev["m60_ma5"] = round(sum(b["close"] for b in m60_b[-5:]) / min(5, len(m60_b)), 4) if m60_b else 0.0
                if flags.get("m2_strong_fx_freeze") and bi15 and bi15.get("type") == "down":
                    bc = verify_15m_strong_bottom_and_macd_shrink(m15_b, bi15)
                    # 右侧强底分型+双缩脚奖励 +8分，左侧未确认扣 -10分
                    if bc["confirmed_level"] == 2:
                        ev["score"] += 8
                    elif bc["confirmed_level"] == 0:
                        ev["score"] -= 10
                    ev["bot_fx_low"] = bc["bot_fx_low"] if bc["bot_fx_low"] > 0 else ev["bot_fx_low"]
                eval_cache[ck_e] = ev
            if ev is not None:
                evaluated.append(ev)

        evaluated.sort(key=lambda x: (x["score"], x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0, x["snapshot"]["amountYi"] if x["snapshot"] else 0.0), reverse=True)
        eval_map = {e["symbol"]: e for e in evaluated}

        if flags.get("m3_zscore_cont"):
            reg_c = compute_v4_1_continuous_regime(bench_day_slice, day_snaps)
            regime = "WEAK" if reg_c["R_t"] < 0.40 else ("CRASH" if reg_c["is_crash_lock"] else "BULL")
            bear_sev = "WEAK_DEEP" if reg_c["R_t"] < 0.28 else "WEAK_MILD"
            r_cfg = reg_c
        else:
            regime, bear_sev, r_cfg = classify_v4_adaptive_regime(bench_day_slice, day_snaps, evaluated)

        # --- 2. 14:54:00 卖出 ---
        for sym in list(holdings.keys()):
            h = holdings[sym]
            if h.get("pending_sell_reason"):
                continue
            ev = eval_map.get(sym)
            dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if not dbar or dbar["close"] <= 0 or dbar["volume"] <= 0 or not ev:
                continue
            sell_p_ref = day_snaps[sym]["lastPrice"] if (flags.get("m1_1445_iso") and sym in day_snaps) else dbar["close"]
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

            if flags.get("m5_half_profit") and h.get("half_profit_locked"):
                t_line = max(zs15["zg"] if zs15 else 0.0, h.get("frozen_15m_zg", 0.0), h["avg_cost"] * 1.002)
                if sell_p_ref < t_line or (bi60 and bi60["type"] == "up" and div60 is not None and div60 < 0.65):
                    sell_rsn = "[14:54半仓移动止盈]"

            if not sell_rsn:
                eff_dd = max(zs15["dd"] if zs15 else 0.0, h.get("frozen_15m_dd", 0.0)) if flags.get("m2_strong_fx_freeze") else (zs15["dd"] if zs15 else 0.0)
                if regime in ("WEAK", "CRASH") and sell_p_ref < h["init_stop"]:
                    sell_rsn = "[14:54熊市锁死止损]"
                elif eff_dd > 0 and sell_p_ref < eff_dd:
                    sell_rsn = "[14:54结构破位止损]"
                elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                    if dbar["pctChange"] > -3.5 or sell_p_ref >= h["avg_cost"]:
                        in_60m_main = bool(zs60 and sell_p_ref > zs60["zg"] and bi60 and bi60["type"] == "up" and (div60 is None or div60 >= 0.75))
                        if flags.get("m5_half_profit") and in_60m_main and (not h.get("half_profit_locked")) and h["shares"] >= 200 and sell_p_ref > h["avg_cost"] * 1.01:
                            sell_ratio = 0.5
                            sell_rsn = "[14:54主升浪减半锁利]"
                        else:
                            sell_rsn = "[14:54顶背驰一卖]"

            if sell_rsn:
                sell_sh = h["shares"] if sell_ratio >= 0.99 else max(100, (int(h["shares"] * 0.5) // 100) * 100)
                if is_tail_limit_down(sym, dt, dbar):
                    h["pending_sell_reason"] = f"{sell_rsn}(尾盘跌停转早盘)"
                    h["pending_sell_shares"] = sell_sh
                    continue
                if flags.get("m6_micro_friction"):
                    exec_p, gross, fee = calc_v4_1_trade_cost("SELL", sym, h["kind"], dt, sell_p_ref, sell_sh)
                else:
                    exec_p, gross, fee = calc_trade_cost("SELL", sym, h["kind"], dt, sell_p_ref, sell_sh)
                net = gross - fee
                cash += net
                closed_trades.append({
                    "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
                    "shares": sell_sh, "buy_price": h["avg_cost"], "sell_price": exec_p,
                    "pnl_amt": round(net - sell_sh * h["avg_cost"], 2),
                    "pnl_pct": round((exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0, 2), "reason": sell_rsn
                })
                trades.append({"date": dt, "action": "SELL", "symbol": sym, "shares": sell_sh, "price": exec_p, "amount": round(gross, 2), "fee": fee})
                if sell_sh >= h["shares"]:
                    del holdings[sym]
                else:
                    h["shares"] -= sell_sh
                    h["half_profit_locked"] = True

        # --- 3. 14:59:45 买入 ---
        cyb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "CYB"]
        kcb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "KCB"]
        cyb_str = (sum(sorted([x["score"] for x in cyb_cands], reverse=True)[:4]) / min(4, len(cyb_cands))) if cyb_cands else 0.0
        kcb_str = (sum(sorted([x["score"] for x in kcb_cands], reverse=True)[:4]) / min(4, len(kcb_cands))) if kcb_cands else 0.0
        growth_winner = "CYB" if cyb_str >= kcb_str else "KCB"

        dyn_t1, dyn_t2 = compute_dynamic_defensive_tiers(hs_u, indexed, bench_day_slice, dt) if regime == "WEAK" else (set(), set())
        overext_map = compute_v4_1_quadratic_overext_penalties(evaluated) if flags.get("m3_zscore_cont") else compute_cross_sectional_overext_penalties(evaluated)
        all_prices = sorted([x["price"] for x in evaluated if x["price"] > 0])
        high_price_p85 = all_prices[int(len(all_prices) * 0.85)] if len(all_prices) >= 5 else 200.0

        def rank_fn(cand):
            sym = cand["symbol"]
            sec = get_sector_of(sym)
            base_s = cand["score"]
            p_day = cand["periods"].get("day")
            zs_day = p_day.get("lastZs") if p_day else None
            above_zg = bool(zs_day and p_day["close"] > zs_day["zg"])
            over_pen = overext_map.get(sym, 0)
            if regime in ("BULL", "RANGE"):
                if sec == growth_winner: g_boost = 16 if above_zg else 10
                elif sec in ("CYB", "KCB"): g_boost = 10 if (abs(cyb_str - kcb_str) <= 6.0 and above_zg) else 6
                else: g_boost = 0
                adj_s = base_s + g_boost - over_pen
            else:
                if bear_sev == "WEAK_DEEP":
                    if sym in dyn_t1 and above_zg: adj_s = base_s + 18
                    elif sym in dyn_t2 and above_zg: adj_s = base_s + 10
                    elif sec == growth_winner and above_zg: adj_s = base_s + 8
                    else: adj_s = base_s - 15
                else:
                    if sym in dyn_t1 and above_zg: adj_s = base_s + 14
                    elif sec == growth_winner and above_zg: adj_s = base_s + 12
                    elif sym in dyn_t2 and above_zg: adj_s = base_s + 8
                    else: adj_s = base_s - 8
                if cand["price"] > high_price_p85: adj_s -= 25
                adj_s -= over_pen
            return (adj_s, cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0, cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0)

        ranked = sorted(evaluated, key=rank_fn, reverse=True)
        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_eq = cash + mkt_val_now
        rem_room = max(0.0, cur_eq * r_cfg["max_exposure_pct"] - mkt_val_now)
        min_sc = r_cfg["min_score"]
        pos_sc = r_cfg["single_pos_scale"]
        eff_stock_cap = min(2500000.0, max(50000.0, cur_eq * 0.15))
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
                if regime == "WEAK" and cand["price"] > high_price_p85:
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
                liq_mult = 0.0025 if flags.get("m6_micro_friction") else 0.01
                liq_cap = max(150000.0 if flags.get("m6_micro_friction") else 250000.0, snap["amountYi"] * 1e8 * liq_mult) if (snap and snap["amountYi"] > 0) else 2500000.0
                t_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_sc, liq_cap)
                alloc = min(budget * 0.95, t_amt, rem_room)
                lots = int(alloc // (dbar["close"] * 100.0))
                shares = lots * 100
                if shares < 100:
                    continue
                if flags.get("m6_micro_friction"):
                    exec_p, gross, fee = calc_v4_1_trade_cost("BUY", sym, cand["kind"], dt, dbar["close"], shares, impact_bps=4.0)
                else:
                    exec_p, gross, fee = calc_trade_cost("BUY", sym, cand["kind"], dt, dbar["close"], shares)
                tot_cost = gross + fee
                if tot_cost > cash:
                    continue
                cash -= tot_cost; budget -= tot_cost; rem_room -= gross; bought += 1
                holdings[sym] = {
                    "name": cand["name"], "kind": cand["kind"], "shares": shares,
                    "avg_cost": round(tot_cost / shares, 4), "last_price": dbar["close"],
                    "buy_date": dt, "init_stop": raw_stop,
                    "frozen_15m_zg": zs15["zg"] if zs15 else 0.0,
                    "frozen_15m_dd": zs15["dd"] if zs15 else 0.0,
                    "half_profit_locked": False
                }
                trades.append({"date": dt, "action": "BUY", "symbol": sym, "shares": shares, "price": exec_p, "amount": round(gross, 2), "fee": fee})

        for sym, h in holdings.items():
            dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if dbar and dbar["close"] > 0:
                h["last_price"] = dbar["close"]
        tot_v = round(cash + sum(h["shares"] * h["last_price"] for h in holdings.values()), 2)
        tot_r = round((tot_v - 1e6) / 1e6 * 100.0, 2)
        b_r = round((bmap[dt] - init_b) / init_b * 100.0, 2)
        hist.append({"date": dt, "portfolio_value": tot_v, "portfolio_return_pct": tot_r, "hs300_level": bmap[dt], "hs300_return_pct": b_r, "alpha_pct": round(tot_r - b_r, 2)})

    res = {"account_name": name, "initial_capital": 1e6, "final_value": hist[-1]["portfolio_value"], "history": hist, "trades": trades, "closed_trades": closed_trades}
    m = compute_metrics(res)
    pk = max(hist, key=lambda x: x["portfolio_value"])
    m["peak_value"] = pk["portfolio_value"]; m["peak_date"] = pk["date"]
    return m

if __name__ == "__main__":
    # Load PureLiq rosters (same as V4.0 +17.85% baseline)
    from model_v4_0_zero_prior import build_v4_sql_blind_rosters
    import duckdb
    from duckdb_Backtest_engine import DB_PATH
    # Temporarily use PureLiq for exact V4.0 baseline alignment
    indexed, rosters_dual, fmap, trade_dates = load_v4_1y_duckdb()
    eval_cache = {}

    configs = [
        ("0. V4.0 基线 (DualChannel 15%)", {}),
        ("1. +M1 (14:45 物理隔离快照)", {"m1_1445_iso": True}),
        ("2. +M2 (强底分型双缩脚+冻结15mDD)", {"m2_strong_fx_freeze": True}),
        ("3. +M3 (Z-Score初筛+连续状态机R_t+二次乖离)", {"m3_zscore_cont": True}),
        ("4. +M5 (60m主升50%减半锁利+移动止盈)", {"m5_half_profit": True}),
        ("5. +M6 (0.25%成交限额+4bps滑点+早盘无偏)", {"m6_micro_friction": True}),
        ("6. 组合 M2+M5+M6 (核心增益+微观摩擦)", {"m2_strong_fx_freeze": True, "m5_half_profit": True, "m6_micro_friction": True}),
        ("7. 组合 M1+M2+M5+M6 (+14:45物理隔离)", {"m1_1445_iso": True, "m2_strong_fx_freeze": True, "m5_half_profit": True, "m6_micro_friction": True}),
    ]
    for lbl, flg in configs:
        m = run_configurable_v4(lbl, rosters_dual, fmap, indexed, trade_dates, flg, eval_cache)
        print(f"{lbl:44s} | 期末:¥{m['final_value']/1e4:6.2f}w ({m['total_return_pct']:+6.2f}%) | 最高:¥{m['peak_value']/1e4:6.2f}w | 回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:5.2f} | 胜率:{m['win_rate_pct']:5.2f}% | 盈亏比:{m['profit_loss_ratio']:4.2f}")

    print("\n--- Testing Module 4 (Asymmetric Monthly Buffered Rosters) + Module 5 + Module 6 ---")
    from run_v4_1_vs_v4_0_1y import load_v4_1_1y_duckdb
    indexed_41_liq, rosters_41_liq, fmap_41_liq, _ = load_v4_1_1y_duckdb(use_dual_channel_hs=False)
    ec2 = {}
    for lbl, flg in [
        ("8. +M4 Only (非对称月度缓冲池 PureLiq)", {}),
        ("9. +M4+M5 (+60m主升50%减半锁利)", {"m5_half_profit": True}),
        ("10. +M4+M5+M6 (+0.25%限额+4bps滑点+早盘无偏)", {"m5_half_profit": True, "m6_micro_friction": True}),
    ]:
        m = run_configurable_v4(lbl, rosters_41_liq, fmap_41_liq, indexed_41_liq, trade_dates, flg, ec2)
        print(f"{lbl:44s} | 期末:¥{m['final_value']/1e4:6.2f}w ({m['total_return_pct']:+6.2f}%) | 最高:¥{m['peak_value']/1e4:6.2f}w | 回撤:-{m['max_drawdown_pct']:5.2f}% | 夏普:{m['sharpe_ratio']:5.2f} | 胜率:{m['win_rate_pct']:5.2f}% | 盈亏比:{m['profit_loss_ratio']:4.2f}")
