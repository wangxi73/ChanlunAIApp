"""
ChanlunAIApp V3.2 实验版模型（牛市动态主线识别 + 熊市分级红利避险路由）
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
from test_strict_limits_all import get_day_bar_at
from limit_Execution_filter import is_limit_up_close, is_limit_down_close
import market_regime_v2

# T1：沪深300中避险属性最强的「高股息红利/公用事业/国有大行/能源资源」终极盾牌
HS300_DEFENSIVE_T1 = {
    "sh601398", "sh601288", "sh601988", "sh601939", "sh600036",
    "sh601166", "sh600000", "sz000001", "sz002142",
    "sh600900", "sh601985", "sh600905", "sh600011", "sh600795", "sh601991", "sh600406",
    "sh601728", "sh600941", "sh600050",
    "sh601088", "sh601225", "sh600028", "sh601857", "sh600938", "sh601816",
    "sh510300",
}

# T2：沪深300中低波稳健型「必选消费/医药龙头/央企重工保险」次级防御白马
HS300_DEFENSIVE_T2 = {
    "sh601318", "sh601601", "sh601628", "sh601336",
    "sh600519", "sz000858", "sz000333", "sh600887", "sz000651", "sh600690",
    "sh603288", "sh600600", "sz002032", "sz002508",
    "sh600276", "sh600436", "sz000538", "sh600332", "sh601607", "sh600535",
    "sh601766", "sh600150", "sh600019", "sh600585", "sh600066", "sz002352",
}

def get_sector_of(sym: str) -> str:
    if sym.startswith("sh688") or sym == "sh588000":
        return "KCB"
    if sym.startswith("sz30") or sym == "sz159915":
        return "CYB"
    return "HS300"

def compute_sector_mainline_boosts(evaluated: list) -> tuple:
    by_sec = {"HS300": [], "CYB": [], "KCB": []}
    for ev in evaluated:
        by_sec[get_sector_of(ev["symbol"])].append(ev)

    sec_power = {}
    for sec, items in by_sec.items():
        if not items:
            sec_power[sec] = -999.0
            continue
        scores = sorted([x["score"] for x in items], reverse=True)
        top_k = scores[:min(5, len(scores))]
        avg_top_score = sum(top_k) / len(top_k)
        strong_cnt = sum(1 for s in scores if s >= 75)
        strong_ratio = strong_cnt / len(items)
        doms = [x["snapshot"]["buyerDom"] for x in items if x.get("snapshot")]
        avg_dom = (sum(doms) / len(doms)) if doms else 0.0
        sec_power[sec] = avg_top_score * 0.65 + strong_ratio * 35.0 + avg_dom * 0.45

    ranked_secs = sorted(sec_power.items(), key=lambda x: x[1], reverse=True)
    boosts = {"HS300": 0, "CYB": 0, "KCB": 0}
    winner = ranked_secs[0][0] if ranked_secs else "HS300"
    if ranked_secs:
        top1_sec, top1_val = ranked_secs[0]
        if top1_val >= 52.0:
            boosts[top1_sec] = 14
        if len(ranked_secs) >= 2:
            top2_sec, top2_val = ranked_secs[1]
            if top2_val >= 50.0 and (top1_val - top2_val) <= 6.0:
                boosts[top2_sec] = 8
            elif top2_val >= 54.0:
                boosts[top2_sec] = 5
    return boosts, winner

def get_bear_severity_level(bench_day_bars: list) -> str:
    if not bench_day_bars or len(bench_day_bars) < 60:
        return "WEAK_MILD"
    closes = [b["close"] for b in bench_day_bars]
    c = closes[-1]
    ma20 = sum(closes[-20:]) / 20.0
    ma60 = sum(closes[-60:]) / 60.0
    ma120 = sum(closes[-120:]) / 120.0 if len(closes) >= 120 else ma60
    peak20 = max(closes[-20:])
    dd20 = (c - peak20) / peak20 * 100.0
    if (c < ma60 and ma20 < ma60) or (c < ma60 and ma60 < ma120) or (dd20 <= -5.0 and c < ma20):
        return "WEAK_DEEP"
    return "WEAK_MILD"

def run_v3_2_mainline(rosters_by_date, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
                      initial_capital=1000000.0, single_stock_cap=250000.0, single_etf_cap=300000.0, min_cash=150000.0,
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

        # 1. SELL check (with strict limit-down queue)
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
                        sell_reason = f"[V3.2熊市锁死止损] {h['name']}({sym})跌破初始止损位({h['init_stop']})"
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

        # 2. V3.2 Dynamic Mainline & Tiered Bear Defensive Routing
        sec_boosts, main_winner = compute_sector_mainline_boosts(evaluated)
        mainline_wins[main_winner] += 1
        bear_sev = get_bear_severity_level(bench_day_slice) if regime == "WEAK" else "NONE"

        def v3_2_route_priority(cand):
            sym = cand["symbol"]
            sec = get_sector_of(sym)
            base_s = cand["score"]
            p_day = cand["periods"].get("day")
            zs_day = p_day.get("lastZs") if p_day else None
            above_day_zg = bool(zs_day and p_day["close"] > zs_day["zg"])

            if regime in ("BULL", "RANGE"):
                # 牛市/结构市：按当日实时主线板块(sec_boosts)精准提权，且要求个股自身站上日线中枢ZG才享受全额主线提权！
                m_boost = sec_boosts.get(sec, 0)
                if not above_day_zg:
                    m_boost = m_boost // 2
                elastic_bonus = 4 if (regime == "BULL" and sec in ("CYB", "KCB") and m_boost >= 8 and above_day_zg) else 0
                adj_s = base_s + m_boost + elastic_bonus
            else:
                # 熊市/弱势防守期：随熊市加深程度(WEAK_MILD vs WEAK_DEEP)分级导向 T1 终极红利避险盾牌
                if bear_sev == "WEAK_DEEP":
                    if sym in HS300_DEFENSIVE_T1:
                        adj_s = base_s + 26
                    elif sym in HS300_DEFENSIVE_T2:
                        adj_s = base_s + 14
                    elif sec == "HS300":
                        adj_s = base_s + 2
                    else:
                        adj_s = base_s - (4 if (sec == main_winner and above_day_zg) else 16)
                    if cand["price"] > 120.0:
                        adj_s -= 30
                else:
                    if sym in HS300_DEFENSIVE_T1:
                        adj_s = base_s + 18
                    elif sym in HS300_DEFENSIVE_T2:
                        adj_s = base_s + 12
                    elif sec == "HS300":
                        adj_s = base_s + 6
                    else:
                        adj_s = base_s - (2 if (sec == main_winner and above_day_zg) else 8)
                    if cand["price"] > 180.0:
                        adj_s -= 25

            return (
                adj_s,
                cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0,
                cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0
            )

        ranked_cands = sorted(evaluated, key=v3_2_route_priority, reverse=True)
        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_equity = cash + mkt_val_now

        eff_max_exp = 0.55 if bear_sev == "WEAK_DEEP" else r_cfg["max_exposure_pct"]
        remaining_room = max(0.0, cur_equity * eff_max_exp - mkt_val_now)

        daily_limit = r_cfg["daily_buy_limit"]
        min_score = 65 if bear_sev == "WEAK_DEEP" else r_cfg["min_score"]
        pos_scale = 0.55 if bear_sev == "WEAK_DEEP" else r_cfg["single_pos_scale"]
        eff_min_cash = min_cash * max(pos_scale, 0.45)
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
                trades.append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": cand["price"], "amount": round(gross, 2), "fee": round(fee, 2), "stop_loss": stop_loss, "score": cand["score"], "regime": regime, "reason": f"[V3.2·{r_cfg['label']}] 买入{cand['name']}({sym}) (原分={cand['score']},主线={main_winner})"})

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
        "account_name": f"V3.2 实验版 (动态主线+分级红利避险 · 单笔{int(single_stock_cap/10000)}万)",
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

    td_615 = [b["time"] for b in bench_2y if "2026-06-15" <= b["time"] <= "2026-10-29"]
    rosters_615, fmap = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_2y, td_615)
    eval_615 = {}
    market_regime_v2._BENCH_CACHE.clear()
    r_615_16w = run_v3_2_mainline(rosters_615, fmap, idx_2y, td_615, bmap_2y, eval_615, 1000000.0, 160000.0, 200000.0, 80000.0, True, True)
    market_regime_v2._BENCH_CACHE.clear()
    r_615_25w = run_v3_2_mainline(rosters_615, fmap, idx_2y, td_615, bmap_2y, eval_615, 1000000.0, 250000.0, 300000.0, 150000.0, True, True)

    td_1y = [b["time"] for b in bench_2y if "2025-10-29" <= b["time"] <= "2026-10-29"]
    rosters_1y, _ = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_2y, td_1y)
    eval_1y = {}
    market_regime_v2._BENCH_CACHE.clear()
    r_1y_16w = run_v3_2_mainline(rosters_1y, fmap, idx_2y, td_1y, bmap_2y, eval_1y, 1000000.0, 160000.0, 200000.0, 80000.0, True, True)
    market_regime_v2._BENCH_CACHE.clear()
    r_1y_25w = run_v3_2_mainline(rosters_1y, fmap, idx_2y, td_1y, bmap_2y, eval_1y, 1000000.0, 250000.0, 300000.0, 150000.0, True, True)

    raw_10y = {s: pickle.load(open(CACHE_10Y_DIR / f"{s}.pkl", "rb")) for s in all_syms if (CACHE_10Y_DIR / f"{s}.pkl").exists()}
    ds_10y = build_period_dataset(raw_10y, "2015-10-01", "2026-10-29", "2026-10-29")
    idx_10y = prepare_indexed_data(ds_10y)
    bench_10y = idx_10y["sh000300"]["day"]
    bmap_10y = {b["time"]: b["close"] for b in bench_10y}
    td_10y = [b["time"] for b in bench_10y if "2016-10-28" <= b["time"] <= "2026-10-29"]
    rosters_10y, _ = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_10y, td_10y)
    eval_10y = pickle.load(open(EVAL_CACHE_FILE, "rb")) if EVAL_CACHE_FILE.exists() else {}
    market_regime_v2._BENCH_CACHE.clear()
    r_10y_16w = run_v3_2_mainline(rosters_10y, fmap, idx_10y, td_10y, bmap_10y, eval_10y, 1000000.0, 160000.0, 200000.0, 80000.0, False, True)
    market_regime_v2._BENCH_CACHE.clear()
    r_10y_25w = run_v3_2_mainline(rosters_10y, fmap, idx_10y, td_10y, bmap_10y, eval_10y, 1000000.0, 250000.0, 300000.0, 150000.0, False, True)

    with open(base / "backtest_results_v3_2_mainline.json", "w", encoding="utf-8") as f:
        json.dump({
            "p_20260615": {"cap_16w": r_615_16w, "cap_25w": r_615_25w},
            "p_1y": {"cap_16w": r_1y_16w, "cap_25w": r_1y_25w},
            "p_10y": {"cap_16w": r_10y_16w, "cap_25w": r_10y_25w},
        }, f, ensure_ascii=False, indent=2)

    for tag, r in [
        ("短期 2026.06.15~10.29 [单笔16w]", r_615_16w),
        ("短期 2026.06.15~10.29 [单笔25w]", r_615_25w),
        ("中期 最近1年 2025.10~2026.10 [单笔16w]", r_1y_16w),
        ("中期 最近1年 2025.10~2026.10 [单笔25w]", r_1y_25w),
        ("长期 近10年PIT 2016.10~2026.10 [单笔16w]", r_10y_16w),
        ("长期 近10年PIT 2016.10~2026.10 [单笔25w]", r_10y_25w),
    ]:
        m = r["metrics"]
        print(f"=== V3.2 {tag} ===")
        print(f"  期末: ¥{m['final_value']:,.2f} ({m['total_return_pct']:+.2f}%) | Alpha: {m['alpha_pct']:+.2f}% | 最高: ¥{m['peak_value']:,.2f} ({m['peak_date']}) | 最低: ¥{m['trough_value']:,.2f}")
        print(f"  最大回撤: -{m['max_drawdown_pct']:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | 胜率: {m['win_rate_pct']:.2f}% | 盈亏比: {m['profit_loss_ratio']:.2f}")
        print(f"  主线胜出天数: {r['mainline_days']} | 板块盈亏: {r['sector_pnl_attribution']}")
