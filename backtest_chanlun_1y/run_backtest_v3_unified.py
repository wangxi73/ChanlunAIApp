import bisect
import json
import pickle
from pathlib import Path
from pit_universe import get_candidate_pools
from chanlun_engine import evaluate_asset, get_stage1_candidates
from data_loader import load_all_data_2y
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR
from run_backtest_10y_pit import precompute_pit_rosters, with_held_metas, KCB_ACTIVE_DATE
import market_regime_v2

EVAL_CACHE_FILE = Path(__file__).resolve().parent / "cache_10y_eval.pkl"

def build_unified_pit_rosters(rosters_by_date):
    unified = {}
    for dt, (hs_u, cyb_u, kcb_u) in rosters_by_date.items():
        kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
        seen = set()
        pool = []
        for m in (hs_u + cyb_u + (kcb_u if kcb_active else [])):
            if m[0] not in seen:
                seen.add(m[0])
                pool.append(m)
        unified[dt] = pool
    return unified

def simulate_v3_unified(name, unified_rosters, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
                        initial_capital=1000000.0, single_stock_cap=250000.0, single_etf_cap=300000.0, min_cash=150000.0,
                        use_ratchet_stop=True):
    market_regime_v2.REGIME_CONFIGS["BULL"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["RANGE"].update({"max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0})
    market_regime_v2.REGIME_CONFIGS["WEAK"].update({"max_exposure_pct": 0.70, "daily_buy_limit": 1, "min_score": 60, "single_pos_scale": 0.60})
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
        u_pool = with_held_metas(unified_rosters[dt], holdings, full_meta_map)

        day_snaps = {}
        for sym, _, _ in u_pool:
            s = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
            if s:
                day_snaps[sym] = s

        # Unified Top-52 Stage-1 Funnel across (HS300 + ChiNext + STAR50)
        stage1_selected = get_stage1_candidates(u_pool, day_snaps, holdings, top_k=52)
        evaluated = []
        for meta in stage1_selected:
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

        eval_map = {e["symbol"]: e for e in evaluated}
        regime, r_cfg, r_meta = market_regime_v2.classify_market_regime(bench_day_slice, day_snaps, evaluated, mode="dual_factor")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # 1. SELL check
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
                sell_reason = f"[V3熊市锁死止损] {h['name']}({sym})在{r_cfg['label']}下跌破初始止损位({h['init_stop']})离场"
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

        # 2. Regime-Adaptive Sector Routing in Unified Pool
        def route_priority(cand):
            sym = cand["symbol"]
            is_growth_20cm = sym.startswith("sz30") or sym.startswith("sh688") or sym in ("sz159915", "sh588000")
            base_s = cand["score"]
            if regime == "BULL":
                # 强牛主升期：赋予通过全市场成交额漏斗的20cm双创龙头 +8分弹性优先权
                adj_s = base_s + (8 if is_growth_20cm else 0)
            elif regime == "RANGE":
                adj_s = base_s
            else:
                # 弱势/急跌防守期：赋予沪深300低波蓝筹红利 +15分避险优先权
                adj_s = base_s + (15 if not is_growth_20cm else 0)
            return (
                adj_s,
                cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0,
                cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0
            )

        ranked_cands = sorted(evaluated, key=route_priority, reverse=True)

        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_equity = cash + mkt_val_now
        max_allowed_mkt_val = cur_equity * r_cfg["max_exposure_pct"]
        remaining_room = max(0.0, max_allowed_mkt_val - mkt_val_now)

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
                if sym in holdings:
                    continue
                if cand["score"] < min_score or budget < eff_min_cash or cand["price"] <= 0:
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
                trades.append({
                    "date": dt, "action": "BUY", "symbol": sym, "name": cand["name"],
                    "shares": shares, "price": cand["price"], "amount": round(gross, 2),
                    "fee": round(fee, 2), "stop_loss": stop_loss, "score": cand["score"],
                    "regime": regime, "reason": f"[V3·{r_cfg['label']}] 买入{cand['name']}({sym}) (分={cand['score']})"
                })

        hs_val = cyb_val = kcb_val = 0.0
        for sym, h in holdings.items():
            mv = h["shares"] * h["last_price"]
            if sym.startswith("sh688") or sym == "sh588000":
                kcb_val += mv
            elif sym.startswith("sz30") or sym == "sz159915":
                cyb_val += mv
            else:
                hs_val += mv

        tot_val = round(cash + hs_val + cyb_val + kcb_val, 2)
        tot_ret = round((tot_val - initial_capital) / initial_capital * 100.0, 2)
        hs300_lvl = bench_map[dt]
        hs300_ret = round((hs300_lvl - init_bench) / init_bench * 100.0, 2)
        history.append({
            "date": dt, "portfolio_value": tot_val,
            "cash": round(cash, 2),
            "hs300_mkt_val": round(hs_val, 2),
            "cyb_mkt_val": round(cyb_val, 2),
            "kcb_mkt_val": round(kcb_val, 2),
            "holdings_count": len(holdings),
            "regime": regime,
            "portfolio_return_pct": tot_ret,
            "hs300_level": hs300_lvl,
            "hs300_return_pct": hs300_ret,
            "alpha_pct": round(tot_ret - hs300_ret, 2)
        })

    sec_stats = {"沪深300": [0, 0.0], "创业板": [0, 0.0], "科创板": [0, 0.0]}
    for c in closed_trades:
        sym = c["symbol"]
        sec = "科创板" if (sym.startswith("sh688") or sym == "sh588000") else ("创业板" if (sym.startswith("sz30") or sym == "sz159915") else "沪深300")
        sec_stats[sec][0] += 1
        sec_stats[sec][1] = round(sec_stats[sec][1] + c["pnl_amt"], 2)

    res = {
        "account_name": name, "initial_capital": initial_capital,
        "final_value": history[-1]["portfolio_value"],
        "final_cash": round(cash, 2),
        "final_holdings": holdings,
        "regime_counts": regime_counts,
        "sector_pnl_attribution": sec_stats,
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
    hs_pool, cyb_pool, kcb_pool = get_candidate_pools()
    all_syms = list({m[0] for m in (hs_pool + cyb_pool + kcb_pool)} | {"sh000300"})

    raw_2y = load_all_data_2y(all_syms, max_workers=12)
    cache_1y_dir = base / "cache"
    for s in all_syms:
        cf1 = cache_1y_dir / f"{s}.pkl"
        if cf1.exists():
            try:
                d1 = pickle.load(open(cf1, "rb"))
                if d1.get("m15"):
                    raw_2y[s] = d1
            except Exception:
                pass
    idx_2y = prepare_indexed_data(raw_2y)
    bench_2y = idx_2y["sh000300"]["day"]
    bmap_2y = {b["time"]: b["close"] for b in bench_2y}

    # --- Period A: 2026-06-15 to 2026-10-29 ---
    td_615 = [b["time"] for b in bench_2y if "2026-06-15" <= b["time"] <= "2026-10-29"]
    rosters_615, fmap = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_2y, td_615)
    u_rosters_615 = build_unified_pit_rosters(rosters_615)
    market_regime_v2._BENCH_CACHE.clear()
    eval_615 = {}
    res_615_12w = simulate_v3_unified(
        "V3.0 统一三大核心指数模型 [单笔12.5万·分散8仓] (2026.06.15~10.29)",
        u_rosters_615, fmap, idx_2y, td_615, bmap_2y, eval_615,
        initial_capital=1000000.0, single_stock_cap=125000.0, single_etf_cap=150000.0, min_cash=75000.0, use_ratchet_stop=True
    )
    res_615_25w = simulate_v3_unified(
        "V3.0 统一三大核心指数模型 [单笔25万·集中4仓] (2026.06.15~10.29)",
        u_rosters_615, fmap, idx_2y, td_615, bmap_2y, eval_615,
        initial_capital=1000000.0, single_stock_cap=250000.0, single_etf_cap=300000.0, min_cash=150000.0, use_ratchet_stop=True
    )

    # --- Period B: 1-Year (2025-10-29 to 2026-10-29) ---
    td_1y = [b["time"] for b in bench_2y if "2025-10-29" <= b["time"] <= "2026-10-29"]
    rosters_1y, _ = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_2y, td_1y)
    u_rosters_1y = build_unified_pit_rosters(rosters_1y)
    market_regime_v2._BENCH_CACHE.clear()
    eval_1y = {}
    res_1y_12w = simulate_v3_unified(
        "V3.0 统一三大核心指数模型 [单笔12.5万·分散8仓] (最近1年)",
        u_rosters_1y, fmap, idx_2y, td_1y, bmap_2y, eval_1y,
        initial_capital=1000000.0, single_stock_cap=125000.0, single_etf_cap=150000.0, min_cash=75000.0, use_ratchet_stop=True
    )
    res_1y_25w = simulate_v3_unified(
        "V3.0 统一三大核心指数模型 [单笔25万·集中4仓] (最近1年)",
        u_rosters_1y, fmap, idx_2y, td_1y, bmap_2y, eval_1y,
        initial_capital=1000000.0, single_stock_cap=250000.0, single_etf_cap=300000.0, min_cash=150000.0, use_ratchet_stop=True
    )

    # --- Period C: 10-Year (2016-10-28 to 2026-10-29) ---
    raw_10y = {s: pickle.load(open(CACHE_10Y_DIR / f"{s}.pkl", "rb")) for s in all_syms if (CACHE_10Y_DIR / f"{s}.pkl").exists()}
    ds_10y = build_period_dataset(raw_10y, "2015-10-01", "2026-10-29", "2026-10-29")
    idx_10y = prepare_indexed_data(ds_10y)
    bench_10y = idx_10y["sh000300"]["day"]
    bmap_10y = {b["time"]: b["close"] for b in bench_10y}
    td_10y = [b["time"] for b in bench_10y if "2016-10-28" <= b["time"] <= "2026-10-29"]
    rosters_10y, _ = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_10y, td_10y)
    u_rosters_10y = build_unified_pit_rosters(rosters_10y)
    market_regime_v2._BENCH_CACHE.clear()
    eval_10y = pickle.load(open(EVAL_CACHE_FILE, "rb")) if EVAL_CACHE_FILE.exists() else {}
    res_10y_12w = simulate_v3_unified(
        "V3.0 统一三大核心指数模型 [单笔12.5万·分散8仓] (近10年PIT)",
        u_rosters_10y, fmap, idx_10y, td_10y, bmap_10y, eval_10y,
        initial_capital=1000000.0, single_stock_cap=125000.0, single_etf_cap=150000.0, min_cash=75000.0, use_ratchet_stop=False
    )
    res_10y_25w = simulate_v3_unified(
        "V3.0 统一三大核心指数模型 [单笔25万·集中4仓] (近10年PIT)",
        u_rosters_10y, fmap, idx_10y, td_10y, bmap_10y, eval_10y,
        initial_capital=1000000.0, single_stock_cap=250000.0, single_etf_cap=300000.0, min_cash=150000.0, use_ratchet_stop=False
    )

    with open(base / "backtest_results_v3_unified.json", "w", encoding="utf-8") as f:
        json.dump({
            "p_20260615": {"cap_12_5w": res_615_12w, "cap_25w": res_615_25w},
            "p_1y": {"cap_12_5w": res_1y_12w, "cap_25w": res_1y_25w},
            "p_10y": {"cap_12_5w": res_10y_12w, "cap_25w": res_10y_25w},
        }, f, ensure_ascii=False, indent=2)

    for r in [res_615_12w, res_615_25w, res_1y_12w, res_1y_25w, res_10y_12w, res_10y_25w]:
        m = r["metrics"]
        print(f"=== {r['account_name']} ===")
        print(f"  期末: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | 沪深300: {m['hs300_return_pct']:+.2f}% | Alpha: {m['alpha_pct']:+.2f}%")
        print(f"  最高: ¥{m['peak_value']:,.2f} ({m['peak_return_pct']:+.2f}%, {m['peak_date']}) | 最低: ¥{m['trough_value']:,.2f} ({m['trough_return_pct']:+.2f}%, {m['trough_date']})")
        print(f"  回撤: -{m['max_drawdown_pct']:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f}")
        print(f"  三大板块已平仓盈亏贡献: {r['sector_pnl_attribution']}")
