import bisect
import json
import pickle
from pathlib import Path
from pit_universe_20y import get_20y_candidate_pools, CACHE_20Y_DIR
from pit_universe import select_pit_universe_at_date
from chanlun_engine import evaluate_asset, get_stage1_candidates
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_pit import with_held_metas
from model_v3_2_mainline import (
    HS300_DEFENSIVE_T1, HS300_DEFENSIVE_T2, get_sector_of,
    compute_sector_mainline_boosts, get_bear_severity_level
)
from test_strict_limits_all import get_day_bar_at, step_subaccount_strict
from limit_Execution_filter import is_limit_up_close, is_limit_down_close
import market_regime_v2

CYB_ACTIVE_DATE = "2010-03-01"  # 创业板首批28家于2009-10-30上市，满4个月具备完整日线/60m/15m缠论结构后正式激活
KCB_ACTIVE_DATE = "2019-08-26"  # 科创板首批25家于2019-07-22上市，满1个月后正式激活
EVAL_20Y_CACHE_FILE = Path(__file__).resolve().parent / "cache_20y_eval.pkl"

def precompute_20y_pit_rosters(hs_pool, cyb_pool, kcb_pool, indexed_data, trade_dates):
    full_meta_map = {m[0]: m for m in (hs_pool + cyb_pool + kcb_pool)}
    rosters_by_date = {}
    cur_hs = select_pit_universe_at_date(hs_pool, indexed_data, trade_dates[0], top_n=100, min_listed_days=60)
    cur_cyb = []
    cur_kcb = []

    for i, dt in enumerate(trade_dates):
        if dt >= CYB_ACTIVE_DATE and not cur_cyb:
            cur_cyb = select_pit_universe_at_date(cyb_pool, indexed_data, dt, top_n=50, min_listed_days=30)
        if dt >= KCB_ACTIVE_DATE and not cur_kcb:
            cur_kcb = select_pit_universe_at_date(kcb_pool, indexed_data, dt, top_n=35, min_listed_days=20)
        rosters_by_date[dt] = (cur_hs, cur_cyb, cur_kcb)

        if i + 1 < len(trade_dates):
            cur_ym = dt[:7]
            nxt_ym = trade_dates[i + 1][:7]
            if cur_ym != nxt_ym and (cur_ym.endswith("-06") or cur_ym.endswith("-12")):
                cur_hs = select_pit_universe_at_date(hs_pool, indexed_data, dt, top_n=100, min_listed_days=60)
                if dt >= CYB_ACTIVE_DATE:
                    cur_cyb = select_pit_universe_at_date(cyb_pool, indexed_data, dt, top_n=50, min_listed_days=40)
                if dt >= KCB_ACTIVE_DATE:
                    cur_kcb = select_pit_universe_at_date(kcb_pool, indexed_data, dt, top_n=35, min_listed_days=20)

    return rosters_by_date, full_meta_map

def get_20y_tiered_candidates(rosters_tuple, holdings, full_meta_map, indexed_data, dt, eval_cache):
    hs_u, cyb_u, kcb_u = rosters_tuple
    cyb_active = (dt >= CYB_ACTIVE_DATE) and (len(cyb_u) >= 8)
    kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)

    active_list = hs_u + (cyb_u if cyb_active else []) + (kcb_u if kcb_active else [])
    all_active = with_held_metas(active_list, holdings, full_meta_map)
    day_snaps = {}
    for sym, _, _ in all_active:
        s = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
        if s:
            day_snaps[sym] = s

    # 三阶段自适应保送名额（总数恒定 46 只）
    if not cyb_active:
        # 第一时代 (2006.10~2010.02 纯主板时代)：全部 46 个名额由当期沪深300(含中小板龙头)提供
        sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=44)
        sel_cyb, sel_kcb = [], []
    elif not kcb_active:
        # 第二时代 (2010.03~2019.08 沪深300+创业板双轨时代)：沪深300保送 18 只 + 创业板保送 28 只
        sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=18)
        sel_cyb = get_stage1_candidates(cyb_u, day_snaps, {}, top_k=28)
        sel_kcb = []
    else:
        # 第三时代 (2019.08~2026.10 三足鼎立时代)：沪深300保送 16 只 + 创业板 18 只 + 科创板 12 只
        sel_hs = get_stage1_candidates(hs_u, day_snaps, {}, top_k=16)
        sel_cyb = get_stage1_candidates(cyb_u, day_snaps, {}, top_k=18)
        sel_kcb = get_stage1_candidates(kcb_u, day_snaps, {}, top_k=12)

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

def simulate_20y_v3(name, rosters_by_date, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
                    initial_capital=1000000.0, base_stock_cap=250000.0, dyn_sizing_pct=None,
                    max_single_cap_cny=2500000.0, overext_mode="day_38", use_ratchet_stop=True, enforce_limits=True):
    market_regime_v2.REGIME_CONFIGS["BULL"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["RANGE"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["WEAK"].update({"max_exposure_pct": 0.70, "daily_buy_limit": 1, "min_score": 60, "single_pos_scale": 0.65})
    market_regime_v2.REGIME_CONFIGS["CRASH"].update({"max_exposure_pct": 0.0, "daily_buy_limit": 0, "min_score": 999, "single_pos_scale": 0.0})

    holdings = {}
    cash = initial_capital
    trades = []
    closed_trades = []
    regime_counts = {}
    mainline_wins = {"HS300": 0, "CYB": 0, "KCB": 0}
    limit_stats = {"blocked_limit_up_buys": 0, "blocked_limit_down_sells": 0}
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
        day_snaps, evaluated = get_20y_tiered_candidates(rosters_by_date[dt], holdings, full_meta_map, indexed_data, dt, eval_cache)
        eval_map = {e["symbol"]: e for e in evaluated}

        regime, r_cfg, _ = market_regime_v2.classify_market_regime(bench_day_slice, day_snaps, evaluated, mode="dual_factor")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # 1. SELL check
        for sym in list(holdings.keys()):
            h = holdings[sym]
            ev = eval_map.get(sym)
            snap = day_snaps.get(sym)
            if not ev and not h.get("pending_forced_sell"):
                continue

            sell_reason = h.get("pending_sell_reason")
            if not sell_reason and ev:
                p_m15 = ev["periods"].get("15m")
                if p_m15:
                    zs15 = p_m15.get("lastZs")
                    bi15 = p_m15.get("lastBi")
                    div15 = p_m15.get("divRatio")
                    if use_ratchet_stop and regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
                        sell_reason = f"[熊市锁死止损] {h['name']}({sym})跌破初始止损位({h['init_stop']})"
                    elif zs15 and ev["price"] < zs15["dd"]:
                        sell_reason = f"[结构破位止损] {h['name']}({sym})跌破15m中枢DD({zs15['dd']})"
                    elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                        sell_reason = f"[15m顶背驰一卖] {h['name']}({sym})面积比={div15}"

            if sell_reason:
                dbar = get_day_bar_at(indexed_data, sym, dt)
                if enforce_limits and is_limit_down_close(sym, dt, snap, dbar):
                    h["pending_forced_sell"] = True
                    h["pending_sell_reason"] = sell_reason + f" ({dt}跌停顺延)"
                    limit_stats["blocked_limit_down_sells"] += 1
                    continue

                price = h["last_price"]
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

        # 2. BUY check with Dynamic Mainline + Bear Shield + Overext Filter
        sec_boosts, main_winner = compute_sector_mainline_boosts(evaluated)
        mainline_wins[main_winner] += 1
        bear_sev = get_bear_severity_level(bench_day_slice) if regime == "WEAK" else "NONE"

        def route_priority(cand):
            sym = cand["symbol"]
            sec = get_sector_of(sym)
            base_s = cand["score"]
            p_day = cand["periods"].get("day")
            zs_day = p_day.get("lastZs") if p_day else None
            above_day_zg = bool(zs_day and p_day["close"] > zs_day["zg"])

            overext_penalty = 0
            if overext_mode == "day_38" and zs_day and zs_day["zg"] > 0:
                dist_zg = (p_day["close"] - zs_day["zg"]) / zs_day["zg"]
                if dist_zg > 0.38: overext_penalty = 25
                elif dist_zg > 0.28: overext_penalty = 10
            elif overext_mode == "m60_25":
                p_m60 = cand["periods"].get("60m")
                zs_60 = p_m60.get("lastZs") if p_m60 else None
                if zs_60 and zs_60["zg"] > 0 and (cand["price"] - zs_60["zg"]) / zs_60["zg"] > 0.25:
                    overext_penalty = 20

            if regime in ("BULL", "RANGE"):
                m_boost = sec_boosts.get(sec, 0)
                if not above_day_zg:
                    m_boost = m_boost // 2
                elastic_bonus = 4 if (regime == "BULL" and sec in ("CYB", "KCB") and m_boost >= 8 and above_day_zg) else 0
                adj_s = base_s + m_boost + elastic_bonus - overext_penalty
            else:
                if bear_sev == "WEAK_DEEP":
                    if sym in HS300_DEFENSIVE_T1: adj_s = base_s + 26
                    elif sym in HS300_DEFENSIVE_T2: adj_s = base_s + 14
                    elif sec == "HS300": adj_s = base_s + 2
                    else: adj_s = base_s - (4 if (sec == main_winner and above_day_zg) else 16)
                    if cand["price"] > 120.0: adj_s -= 30
                else:
                    if sym in HS300_DEFENSIVE_T1: adj_s = base_s + 18
                    elif sym in HS300_DEFENSIVE_T2: adj_s = base_s + 12
                    elif sec == "HS300": adj_s = base_s + 6
                    else: adj_s = base_s - (2 if (sec == main_winner and above_day_zg) else 8)
                    if cand["price"] > 180.0: adj_s -= 25
                adj_s -= overext_penalty

            return (
                adj_s,
                cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0,
                cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0
            )

        ranked_cands = sorted(evaluated, key=route_priority, reverse=True)
        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_equity = cash + mkt_val_now

        eff_max_exp = 0.55 if bear_sev == "WEAK_DEEP" else r_cfg["max_exposure_pct"]
        remaining_room = max(0.0, cur_equity * eff_max_exp - mkt_val_now)

        if dyn_sizing_pct is not None:
            eff_stock_cap = min(max_single_cap_cny, max(base_stock_cap, cur_equity * dyn_sizing_pct))
        else:
            eff_stock_cap = base_stock_cap
        eff_etf_cap = eff_stock_cap * 1.2
        eff_base_min = min(eff_stock_cap * 0.55, 400000.0)

        daily_limit = r_cfg["daily_buy_limit"]
        min_score = 65 if bear_sev == "WEAK_DEEP" else r_cfg["min_score"]
        pos_scale = 0.55 if bear_sev == "WEAK_DEEP" else r_cfg["single_pos_scale"]
        eff_min_cash = eff_base_min * max(pos_scale, 0.45)
        budget = min(cash, remaining_room)
        buy_count = 0

        if daily_limit > 0 and remaining_room >= eff_min_cash:
            for cand in ranked_cands:
                if buy_count >= daily_limit:
                    break
                sym = cand["symbol"]
                if sym in holdings or cand["score"] < min_score or budget < eff_min_cash or cand["price"] <= 0:
                    continue
                if regime == "WEAK" and cand["price"] > (150.0 if bear_sev == "WEAK_DEEP" else 220.0):
                    continue
                snap = day_snaps.get(sym)
                dbar = get_day_bar_at(indexed_data, sym, dt)
                if enforce_limits and is_limit_up_close(sym, dt, snap, dbar):
                    limit_stats["blocked_limit_up_buys"] += 1
                    continue

                # 单票流动性容量约束：单笔不超过当日真实成交额的 1.0% (零冲击成本)
                liq_cap = (snap["amountYi"] * 1e8 * 0.01) if (snap and snap["amountYi"] > 0) else max_single_cap_cny
                target_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_scale, max(250000.0, liq_cap))
                alloc = min(budget * 0.95, target_amt, remaining_room)
                lots = int(alloc // (cand["price"] * 100.0))
                shares = lots * 100
                if shares < 100:
                    continue
                p_m15 = cand["periods"].get("15m")
                zs15 = p_m15.get("lastZs") if p_m15 else None
                raw_stop = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)
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
                    "stop_loss": raw_stop, "init_stop": raw_stop, "buy_regime": regime
                }
                trades.append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": cand["price"], "amount": round(gross, 2), "fee": round(fee, 2), "stop_loss": raw_stop, "score": cand["score"], "regime": regime, "reason": f"[{r_cfg['label']}] 买入{cand['name']}({sym})"})

        hs_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if get_sector_of(s) == "HS300")
        cyb_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if get_sector_of(s) == "CYB")
        kcb_val = sum(h["shares"] * h["last_price"] for s, h in holdings.items() if get_sector_of(s) == "KCB")
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
    m["peak_value"] = pk["portfolio_value"]
    m["peak_return_pct"] = pk["portfolio_return_pct"]
    m["peak_date"] = pk["date"]
    m["trough_value"] = tr["portfolio_value"]
    m["trough_return_pct"] = tr["portfolio_return_pct"]
    m["trough_date"] = tr["date"]
    res["metrics"] = m
    return res

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    hs_pool, cyb_pool, kcb_pool = get_20y_candidate_pools()
    all_syms = list({m[0] for m in (hs_pool + cyb_pool + kcb_pool)} | {"sh000300"})

    print(f"Building 20-Year HFQ dataset (2005-01-01 to 2026-10-29) for {len(all_syms)} symbols...")
    raw_20y = {s: pickle.load(open(CACHE_20Y_DIR / f"{s}.pkl", "rb")) for s in all_syms if (CACHE_20Y_DIR / f"{s}.pkl").exists()}
    ds_20y = build_period_dataset(raw_20y, "2005-01-01", "2026-10-29", "2026-10-29")
    idx_20y = prepare_indexed_data(ds_20y)
    bench_20y = idx_20y["sh000300"]["day"]
    bmap_20y = {b["time"]: b["close"] for b in bench_20y}
    td_20y = [b["time"] for b in bench_20y if "2006-10-30" <= b["time"] <= "2026-10-29"]
    print(f"Running 20-Year PIT Backtest over {len(td_20y)} trading days ({td_20y[0]} -> {td_20y[-1]})...")

    rosters_20y, fmap = precompute_20y_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_20y, td_20y)
    eval_20y = pickle.load(open(EVAL_20Y_CACHE_FILE, "rb")) if EVAL_20Y_CACHE_FILE.exists() else {}

    print("1. Running V3.3 Ultimate [15% Dynamic Sizing + Day Overext Filter] (20Y PIT)...")
    market_regime_v2._BENCH_CACHE.clear()
    res_v33_day = simulate_20y_v3(
        "V3.3 终极版 [15%动态扩容 + 日线透支过滤]", rosters_20y, fmap, idx_20y, td_20y, bmap_20y, eval_20y,
        initial_capital=1000000.0, base_stock_cap=250000.0, dyn_sizing_pct=0.15, max_single_cap_cny=2500000.0,
        overext_mode="day_38", use_ratchet_stop=True, enforce_limits=True
    )

    print("2. Running V3.3 Ultimate [15% Dynamic Sizing + 60m Overext Filter] (20Y PIT)...")
    market_regime_v2._BENCH_CACHE.clear()
    res_v33_m60 = simulate_20y_v3(
        "V3.3 终极版 [15%动态扩容 + 60m脉冲过滤]", rosters_20y, fmap, idx_20y, td_20y, bmap_20y, eval_20y,
        initial_capital=1000000.0, base_stock_cap=250000.0, dyn_sizing_pct=0.15, max_single_cap_cny=2500000.0,
        overext_mode="m60_25", use_ratchet_stop=True, enforce_limits=True
    )

    print("3. Running V3.2 Mainline [Fixed 25w] (20Y PIT)...")
    market_regime_v2._BENCH_CACHE.clear()
    res_v32_fix = simulate_20y_v3(
        "V3.2 主线版 [固定单笔25万]", rosters_20y, fmap, idx_20y, td_20y, bmap_20y, eval_20y,
        initial_capital=1000000.0, base_stock_cap=250000.0, dyn_sizing_pct=None, max_single_cap_cny=250000.0,
        overext_mode="none", use_ratchet_stop=True, enforce_limits=True
    )

    print("4. Running V3.0 Steady [Fixed 12.5w] (20Y PIT)...")
    market_regime_v2._BENCH_CACHE.clear()
    res_v30_fix = simulate_20y_v3(
        "V3.0 稳健版 [固定单笔12.5万·8仓]", rosters_20y, fmap, idx_20y, td_20y, bmap_20y, eval_20y,
        initial_capital=1000000.0, base_stock_cap=125000.0, dyn_sizing_pct=None, max_single_cap_cny=125000.0,
        overext_mode="day_38", use_ratchet_stop=True, enforce_limits=True
    )

    with open(EVAL_20Y_CACHE_FILE, "wb") as f:
        pickle.dump(eval_20y, f)

    with open(base / "backtest_results_20y_pit.json", "w", encoding="utf-8") as f:
        json.dump({
            "v3_3_day": res_v33_day, "v3_3_m60": res_v33_m60,
            "v3_2_fixed": res_v32_fix, "v3_0_steady": res_v30_fix
        }, f, ensure_ascii=False, indent=2)

    for r in [res_v33_day, res_v33_m60, res_v32_fix, res_v30_fix]:
        m = r["metrics"]
        print(f"=== {r['account_name']} (2006.10~2026.10) ===")
        print(f"  期末总净值: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | 同期沪深300: {m['hs300_return_pct']:+.2f}% | Alpha: {m['alpha_pct']:+.2f}%")
        print(f"  最高净值  : ¥{m['peak_value']:,.2f} ({m['peak_return_pct']:+.2f}%, {m['peak_date']}) | 最低谷底: ¥{m['trough_value']:,.2f} ({m['trough_date']})")
        print(f"  最大回撤  : -{m['max_drawdown_pct']:.2f}% ({m['max_drawdown_date']}) | 夏普: {m['sharpe_ratio']:.2f} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f}")
        print(f"  板块盈亏  : {r['sector_pnl_attribution']} | 涨跌停拦截: {r['limit_stats']}")
