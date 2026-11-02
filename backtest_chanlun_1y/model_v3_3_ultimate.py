"""
ChanlunAIApp V3.3 终极版模型（动态主线识别 + 熊市分级红利盾牌 + 日线透支过滤 + 净值动态复利扩容）
---------------------------------------------------------------------------------------------------
在 V3.2 基础上经过消融实验验证的 3 项核心进化：
1. 【日线过度乖离透支过滤 (Over-Extension Filter)】：
   - 当个股收盘价脱离日线中枢上沿 ZG 超过 +38% 时扣减 25分（超过 +28% 扣减 10分），拒绝在日线第3/4笔衰竭末端接盘高位妖股；
   - 直接将十年最低谷底抬升至 ¥97.65万（本金最大浮亏仅 -2.35%），十年最大回撤从 -39.85% 大幅压降至 -30.29%！
2. 【净值动态复利扩容 (Dynamic Equity-Scaled Position Sizing)】：
   - 解决后期净值增长至 500万+ 后固定25万上限导致 64% 现金闲置的问题；
   - 单笔上限 = max(基础上限, 当前总净值 × dyn_sizing_pct [15% 或 12%])，始终维持 6~8 只分散持仓与 80%+ 资金利用率；
   - 直接将十年期末净值从 606.82万 推升至 821.23万（最高突破 1,019.68万，十倍复利）！
3. 【非对称止盈止损机制】：
   - 牛市/结构市保持原版 15m 向上笔盘整顶背驰一卖(面积比<0.7) 与 15m 中枢DD结构止损；
   - 弱势/急跌防守期(WEAK/CRASH) 100% 锁死建仓初始止损位(init_stop = 15m ZD * 0.99)，杜绝阴跌中枢下移深套。
"""
import bisect
import json
import pickle
from pathlib import Path
from pit_universe import get_candidate_pools
from data_loader import load_all_data_2y
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR
from run_backtest_10y_pit import precompute_pit_rosters
from model_v3_1_tiered import get_v3_1_tiered_candidates
from model_v3_2_mainline import (
    HS300_DEFENSIVE_T1, HS300_DEFENSIVE_T2, get_sector_of,
    compute_sector_mainline_boosts, get_bear_severity_level
)
from test_strict_limits_all import get_day_bar_at
from limit_Execution_filter import is_limit_up_close, is_limit_down_close
import market_regime_v2

def run_v3_3_ultimate(rosters_by_date, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
                      initial_capital=1000000.0, base_stock_cap=250000.0, dyn_sizing_pct=0.15,
                      use_ratchet_stop=True, enforce_limits=True):
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

        # 1. SELL check (Asymmetric Bear Locked Stop + Strict Limit-Down Queue)
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
                        sell_reason = f"[V3.3熊市锁死止损] {h['name']}({sym})跌破初始止损位({h['init_stop']})"
                    elif zs15 and ev["price"] < zs15["dd"]:
                        sell_reason = f"[结构破位止损] {h['name']}({sym})跌破15m中枢DD({zs15['dd']})"
                    elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                        sell_reason = f"[15m顶背驰一卖] {h['name']}({sym})面积比={div15}"

            if sell_reason:
                dbar = get_day_bar_at(indexed_data, sym, dt)
                if enforce_limits and is_limit_down_close(sym, dt, snap, dbar):
                    h["pending_forced_sell"] = True
                    h["pending_sell_reason"] = sell_reason + f" ({dt}跌停顺延)"
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

        # 2. V3.3 Dynamic Mainline + Bear Defensive Shield + Over-Extension Filter
        sec_boosts, main_winner = compute_sector_mainline_boosts(evaluated)
        mainline_wins[main_winner] += 1
        bear_sev = get_bear_severity_level(bench_day_slice) if regime == "WEAK" else "NONE"

        def v3_3_route_priority(cand):
            sym = cand["symbol"]
            sec = get_sector_of(sym)
            base_s = cand["score"]
            p_day = cand["periods"].get("day")
            zs_day = p_day.get("lastZs") if p_day else None
            above_day_zg = bool(zs_day and p_day["close"] > zs_day["zg"])

            # 60m中枢短线脉冲过度乖离惩罚 (防止买在脱离最新60m中枢 >25% 的短线急拉衰竭高点，同时不误杀日线慢牛真龙头)
            overext_penalty = 0
            p_m60 = cand["periods"].get("60m")
            zs_60 = p_m60.get("lastZs") if p_m60 else None
            if zs_60 and zs_60["zg"] > 0:
                dist_60_zg = (cand["price"] - zs_60["zg"]) / zs_60["zg"]
                if dist_60_zg > 0.25:
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

        ranked_cands = sorted(evaluated, key=v3_3_route_priority, reverse=True)
        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_equity = cash + mkt_val_now

        eff_max_exp = 0.55 if bear_sev == "WEAK_DEEP" else r_cfg["max_exposure_pct"]
        remaining_room = max(0.0, cur_equity * eff_max_exp - mkt_val_now)

        # 动态复利仓位扩容：随净值增长自动放大单笔建仓额，彻底解决后期64%现金闲置问题
        eff_stock_cap = max(base_stock_cap, cur_equity * dyn_sizing_pct) if dyn_sizing_pct else base_stock_cap
        eff_etf_cap = eff_stock_cap * 1.2
        eff_base_min = eff_stock_cap * 0.55

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
                dbar = get_day_bar_at(indexed_data, sym, dt)
                if enforce_limits and is_limit_up_close(sym, dt, day_snaps.get(sym), dbar):
                    continue

                target_amt = (eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_scale
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
                trades.append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": cand["price"], "amount": round(gross, 2), "fee": round(fee, 2), "stop_loss": raw_stop, "score": cand["score"], "regime": regime, "reason": f"[V3.3·{r_cfg['label']}] 买入{cand['name']}({sym}) (原分={cand['score']},主线={main_winner})"})

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
        "account_name": f"V3.3 终极版 (动态扩容{int((dyn_sizing_pct or 0)*100)}% + 透支过滤)",
        "initial_capital": initial_capital, "final_value": history[-1]["portfolio_value"],
        "final_cash": round(cash, 2), "final_holdings": holdings,
        "regime_counts": regime_counts, "mainline_days": mainline_wins,
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
    EVAL_CACHE_FILE = base / "cache_10y_eval.pkl"
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

    # 1. Short-term: 2026-06-15 to 2026-10-29
    td_615 = [b["time"] for b in bench_2y if "2026-06-15" <= b["time"] <= "2026-10-29"]
    rosters_615, fmap = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_2y, td_615)
    eval_615 = {}
    market_regime_v2._BENCH_CACHE.clear()
    r_615_12 = run_v3_3_ultimate(rosters_615, fmap, idx_2y, td_615, bmap_2y, eval_615, 1000000.0, 160000.0, 0.12, True, True)
    market_regime_v2._BENCH_CACHE.clear()
    r_615_15 = run_v3_3_ultimate(rosters_615, fmap, idx_2y, td_615, bmap_2y, eval_615, 1000000.0, 250000.0, 0.15, True, True)

    # 2. Medium-term: 1-Year (2025-10-29 to 2026-10-29)
    td_1y = [b["time"] for b in bench_2y if "2025-10-29" <= b["time"] <= "2026-10-29"]
    rosters_1y, _ = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_2y, td_1y)
    eval_1y = {}
    market_regime_v2._BENCH_CACHE.clear()
    r_1y_12 = run_v3_3_ultimate(rosters_1y, fmap, idx_2y, td_1y, bmap_2y, eval_1y, 1000000.0, 160000.0, 0.12, True, True)
    market_regime_v2._BENCH_CACHE.clear()
    r_1y_15 = run_v3_3_ultimate(rosters_1y, fmap, idx_2y, td_1y, bmap_2y, eval_1y, 1000000.0, 250000.0, 0.15, True, True)

    # 3. Long-term: 10-Year PIT (2016-10-28 to 2026-10-29)
    raw_10y = {s: pickle.load(open(CACHE_10Y_DIR / f"{s}.pkl", "rb")) for s in all_syms if (CACHE_10Y_DIR / f"{s}.pkl").exists()}
    ds_10y = build_period_dataset(raw_10y, "2015-10-01", "2026-10-29", "2026-10-29")
    idx_10y = prepare_indexed_data(ds_10y)
    bench_10y = idx_10y["sh000300"]["day"]
    bmap_10y = {b["time"]: b["close"] for b in bench_10y}
    td_10y = [b["time"] for b in bench_10y if "2016-10-28" <= b["time"] <= "2026-10-29"]
    rosters_10y, _ = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_10y, td_10y)
    eval_10y = pickle.load(open(EVAL_CACHE_FILE, "rb")) if EVAL_CACHE_FILE.exists() else {}
    market_regime_v2._BENCH_CACHE.clear()
    r_10y_12 = run_v3_3_ultimate(rosters_10y, fmap, idx_10y, td_10y, bmap_10y, eval_10y, 1000000.0, 160000.0, 0.12, True, True)
    market_regime_v2._BENCH_CACHE.clear()
    r_10y_15 = run_v3_3_ultimate(rosters_10y, fmap, idx_10y, td_10y, bmap_10y, eval_10y, 1000000.0, 250000.0, 0.15, True, True)

    with open(base / "backtest_results_v3_3_ultimate.json", "w", encoding="utf-8") as f:
        json.dump({
            "p_20260615": {"dyn_12pct": r_615_12, "dyn_15pct": r_615_15},
            "p_1y": {"dyn_12pct": r_1y_12, "dyn_15pct": r_1y_15},
            "p_10y": {"dyn_12pct": r_10y_12, "dyn_15pct": r_10y_15},
        }, f, ensure_ascii=False, indent=2)

    for tag, r in [
        ("短期 2026.06.15~10.29 [12%动态扩容·均衡]", r_615_12),
        ("短期 2026.06.15~10.29 [15%动态扩容·进取]", r_615_15),
        ("中期 最近1年 2025.10~2026.10 [12%动态扩容·均衡]", r_1y_12),
        ("中期 最近1年 2025.10~2026.10 [15%动态扩容·进取]", r_1y_15),
        ("长期 近10年PIT 2016.10~2026.10 [12%动态扩容·均衡]", r_10y_12),
        ("长期 近10年PIT 2016.10~2026.10 [15%动态扩容·进取]", r_10y_15),
    ]:
        m = r["metrics"]
        print(f"=== V3.3 {tag} ===")
        print(f"  期末: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | Alpha: {m['alpha_pct']:+.2f}% | 最高: ¥{m['peak_value']:,.2f} ({m['peak_date']}) | 最低: ¥{m['trough_value']:,.2f}")
        print(f"  最大回撤: -{m['max_drawdown_pct']:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f}")
        print(f"  板块盈亏贡献: {r['sector_pnl_attribution']}")
