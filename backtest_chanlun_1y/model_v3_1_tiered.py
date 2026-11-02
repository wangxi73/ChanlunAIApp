"""
ChanlunAIApp V3.1 模型（分层保送漏斗 · 单笔25万/16万进取版）
-----------------------------------------------------------
核心特征：
1. 选股池：当期真实 (沪深300 + 创业板指 + 科创50) 三大核心池（无幸存者偏差 PIT）；
2. 分层保送初筛漏斗 (Stage 1 Tiered Funnel)：
   - 防止沪深300万亿大盘股凭成交额垄断初筛名额；
   - 每日固定从 沪深300保送Top-16 + 创业板指保送Top-18(无科创板时Top-28) + 科创50保送Top-12 = 共46只进入缠论三周期精算；
3. 牛熊智能路由 (Stage 2)：
   - BULL / RANGE (牛市与结构市)：20cm双创龙头 +12分进攻优先；
   - WEAK (弱势/高位急跌防守)：沪深300低波蓝筹红利 +18分避险优先，双创 -8分且过滤 >180元高价股；
   - CRASH (熊市反弹末端诱多 / 高位非背驰闪崩)：全员暂停新开仓 (0开仓)；
4. 仓位模式：支持单笔 25万 (4仓集中进取，十年PIT净值 542.57万/峰值 606.28万) 或 单笔 16万 (6仓均衡，十年PIT净值 376.21万)。
"""
import bisect
from chanlun_engine import evaluate_asset, get_stage1_candidates
from run_backtest import compute_metrics
from run_backtest_10y_pit import with_held_metas, KCB_ACTIVE_DATE
import market_regime_v2

def get_v3_1_tiered_candidates(rosters_tuple, holdings, full_meta_map, indexed_data, dt, eval_cache):
    hs_u, cyb_u, kcb_u = rosters_tuple
    kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)

    all_active = with_held_metas(hs_u + cyb_u + (kcb_u if kcb_active else []), holdings, full_meta_map)
    day_snaps = {}
    for sym, _, _ in all_active:
        s = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
        if s:
            day_snaps[sym] = s

    sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=16)
    sel_cyb = get_stage1_candidates(cyb_u, day_snaps, {}, top_k=18 if kcb_active else 28)
    sel_kcb = get_stage1_candidates(kcb_u, day_snaps, {}, top_k=12) if kcb_active else []

    seen = set()
    stage1_merged = []
    for sym in holdings:
        if sym in full_meta_map and sym not in seen:
            stage1_merged.append(full_meta_map[sym])
            seen.add(sym)
    for m in (sel_cyb + sel_kcb + sel_hs):
        if m[0] not in seen:
            stage1_merged.append(m)
            seen.add(m[0])

    evaluated = []
    for meta in stage1_merged:
        sym = meta[0]
        ck = (sym, dt)
        if ck in eval_cache:
            ev = eval_cache[ck]
            if ev is not None:
                evaluated.append(ev)
            continue
        idx_e = indexed_data.get(sym)
        if not idx_e:
            eval_cache[ck] = None
            continue
        i_d = bisect.bisect_right(idx_e["day_dates"], dt)
        i_60 = bisect.bisect_right(idx_e["m60_dates"], dt)
        i_15 = bisect.bisect_right(idx_e["m15_dates"], dt)
        d_bars = idx_e["day"][max(0, i_d - 250):i_d]
        m60_bars = idx_e["m60"][max(0, i_60 - 280):i_60]
        m15_bars = idx_e["m15"][max(0, i_15 - 280):i_15]
        if not d_bars or not m15_bars:
            eval_cache[ck] = None
            continue
        ev = evaluate_asset(meta, d_bars, m60_bars, m15_bars, day_snaps.get(sym))
        eval_cache[ck] = ev
        evaluated.append(ev)

    return day_snaps, evaluated

def run_v3_1_tiered(rosters_by_date, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
                    initial_capital=1000000.0, single_stock_cap=250000.0, single_etf_cap=300000.0, min_cash=150000.0,
                    use_ratchet_stop=True):
    market_regime_v2.REGIME_CONFIGS["BULL"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["RANGE"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["WEAK"].update({"max_exposure_pct": 0.70, "daily_buy_limit": 1, "min_score": 60, "single_pos_scale": 0.65})
    market_regime_v2.REGIME_CONFIGS["CRASH"].update({"max_exposure_pct": 0.0, "daily_buy_limit": 0, "min_score": 999, "single_pos_scale": 0.0})

    holdings = {}
    cash = initial_capital
    trades = []
    closed_trades = []
    regime_counts = {}
    init_bench = bench_map[trade_dates[0]]
    bench_idx = indexed_data["sh000300"]
    history = []

    for dt in trade_dates:
        for sym, h in holdings.items():
            snap = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
            if snap and snap["lastPrice"] > 0:
                h["last_price"] = snap["lastPrice"]

        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]
        day_snaps, evaluated = get_v3_1_tiered_candidates(rosters_by_date[dt], holdings, full_meta_map, indexed_data, dt, eval_cache)
        eval_map = {e["symbol"]: e for e in evaluated}

        regime, r_cfg, _ = market_regime_v2.classify_market_regime(bench_day_slice, day_snaps, evaluated, mode="dual_factor")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        for sym in list(holdings.keys()):
            h = holdings[sym]
            ev = eval_map.get(sym)
            if not ev:
                continue
            p_m15 = ev["periods"].get("15m")
            if not p_m15:
                continue
            zs15 = p_m15.get("lastZs")
            bi15 = p_m15.get("lastBi")
            div15 = p_m15.get("divRatio")

            sell_reason = None
            if use_ratchet_stop and regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
                sell_reason = f"[V3.1熊市锁死止损] {h['name']}({sym})在{r_cfg['label']}下跌破初始止损位({h['init_stop']})离场"
            elif zs15 and ev["price"] < zs15["dd"]:
                sell_reason = f"[纯量化规则] {h['name']}({sym})跌破15m中枢极值DD({zs15['dd']})触发结构破位止损"
            elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                sell_reason = f"[纯量化规则] {h['name']}({sym})出现15m向上笔盘整顶背驰一卖(面积比={div15})"

            if sell_reason:
                price = ev["price"]
                shares = h["shares"]
                gross = shares * price
                fee = max(gross * 0.0005, 5.0)
                net = gross - fee
                cash += net
                pnl_amt = net - (shares * h["avg_cost"])
                pnl_pct = (price - h["avg_cost"]) / h["avg_cost"] * 100.0
                closed_trades.append({
                    "symbol": sym, "name": h["name"], "buy_date": h["buy_date"],
                    "sell_date": dt, "shares": shares, "buy_price": h["avg_cost"],
                    "sell_price": price, "pnl_amt": round(pnl_amt, 2),
                    "pnl_pct": round(pnl_pct, 2), "reason": sell_reason
                })
                trades.append({"date": dt, "action": "SELL", "symbol": sym, "name": h["name"], "shares": shares, "price": price, "amount": round(gross, 2), "fee": round(fee, 2), "regime": regime, "reason": sell_reason})
                del holdings[sym]

        def route_priority(cand):
            sym = cand["symbol"]
            is_growth_20cm = sym.startswith("sz30") or sym.startswith("sh688") or sym in ("sz159915", "sh588000")
            base_s = cand["score"]
            if regime in ("BULL", "RANGE"):
                adj_s = base_s + (12 if is_growth_20cm else 0)
            else:
                adj_s = base_s + (18 if not is_growth_20cm else -8)
                if cand["price"] > 180.0:
                    adj_s -= 25
            return (
                adj_s,
                cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0,
                cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0
            )

        ranked_cands = sorted(evaluated, key=route_priority, reverse=True)
        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_equity = cash + mkt_val_now
        remaining_room = max(0.0, cur_equity * r_cfg["max_exposure_pct"] - mkt_val_now)

        daily_limit = r_cfg["daily_buy_limit"]
        min_score = r_cfg["min_score"]
        pos_scale = r_cfg["single_pos_scale"]
        eff_min_cash = min_cash * max(pos_scale, 0.5)
        budget = min(cash, remaining_room)
        buy_count = 0

        if daily_limit > 0 and remaining_room >= eff_min_cash:
            for cand in ranked_cands:
                if buy_count >= daily_limit:
                    break
                sym = cand["symbol"]
                if sym in holdings or cand["score"] < min_score or budget < eff_min_cash or cand["price"] <= 0:
                    continue
                if regime == "WEAK" and cand["price"] > 220.0:
                    continue
                target_amt = (single_stock_cap if cand["kind"] == "stock" else single_etf_cap) * pos_scale
                alloc = min(budget * 0.95, target_amt, remaining_room)
                lots = int(alloc // (cand["price"] * 100.0))
                shares = lots * 100
                if shares < 100:
                    continue
                p_m15 = cand["periods"].get("15m")
                zs15 = p_m15.get("lastZs") if p_m15 else None
                raw_stop = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
                floor_stop = round(cand["price"] * (1.0 - r_cfg["max_stop_pct"]), 3)
                stop_loss = max(raw_stop, floor_stop) if use_ratchet_stop else raw_stop
                gross = shares * cand["price"]
                fee = max(gross * 0.00025, 5.0)
                total_cost = gross + fee
                if total_cost > cash:
                    continue
                cash -= total_cost
                budget -= gross
                remaining_room -= gross
                buy_count += 1
                holdings[sym] = {
                    "name": cand["name"], "shares": shares,
                    "avg_cost": round(total_cost / shares, 4),
                    "last_price": cand["price"], "buy_date": dt,
                    "stop_loss": stop_loss, "init_stop": raw_stop, "buy_regime": regime
                }
                trades.append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": cand["price"], "amount": round(gross, 2), "fee": round(fee, 2), "stop_loss": stop_loss, "score": cand["score"], "regime": regime, "reason": f"[V3.1·{r_cfg['label']}] 买入{cand['name']}({sym}) (分={cand['score']})"})

        hs_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if not (s.startswith("sz30") or s.startswith("sh688") or s in ("sz159915", "sh588000")))
        cyb_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if s.startswith("sz30") or s == "sz159915")
        kcb_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if s.startswith("sh688") or s == "sh588000")
        tot_val = round(cash + hs_val + cyb_val + kcb_val, 2)
        tot_ret = round((tot_val - initial_capital) / initial_capital * 100.0, 2)
        hs300_lvl = bench_map[dt]
        hs300_ret = round((hs300_lvl - init_bench) / init_bench * 100.0, 2)
        history.append({
            "date": dt, "portfolio_value": tot_val, "cash": round(cash, 2),
            "hs300_mkt_val": round(hs_val, 2), "cyb_mkt_val": round(cyb_val, 2), "kcb_mkt_val": round(kcb_val, 2),
            "holdings_count": len(holdings), "regime": regime,
            "portfolio_return_pct": tot_ret, "hs300_level": hs300_lvl,
            "hs300_return_pct": hs300_ret, "alpha_pct": round(tot_ret - hs300_ret, 2)
        })

    res = {
        "account_name": f"V3.1 模型 (分层保送漏斗 · 单笔{int(single_stock_cap/10000)}万进取版)",
        "initial_capital": initial_capital, "final_value": history[-1]["portfolio_value"],
        "final_cash": round(cash, 2), "final_holdings": holdings, "regime_counts": regime_counts,
        "history": history, "trades": trades, "closed_trades": closed_trades
    }
    res["metrics"] = compute_metrics(res)
    return res
