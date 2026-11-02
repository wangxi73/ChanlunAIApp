"""
独立第三方量化审计脚本 (Independent Third-Party Quantitative Audit)
逐项剥离并量化原回测中的 5 大水分：
1. 日线插值伪造 15m/60m 分钟线 vs 真实 15m/60m 分钟线
2. 当日收盘价(Close)同K线偷价成交 vs 次日开盘价(Next-Open)成交
3. 超低手续费(双边万7.5, 0滑点) vs 真实印花税+佣金+冲击滑点(双边千2.25)
4. 事后硬编码红利白名单/闪崩参数 vs 纯客观无白名单规则
5. 缠论未完成笔(Running Bi)盲区与停牌日卖出漏洞修复
"""
import bisect
import copy
import json
import pickle
from pathlib import Path

from chanlun_engine import compute_macd, analyze_period, evaluate_asset, get_stage1_candidates
from pit_universe import get_candidate_pools
from run_backtest import prepare_indexed_data, compute_metrics
from run_backtest_2019_2020 import build_period_dataset
from run_backtest_10y_100w import CACHE_10Y_DIR
from run_backtest_10y_pit import precompute_pit_rosters
from model_v3_1_tiered import get_v3_1_tiered_candidates
from model_v3_2_mainline import (
    HS300_DEFENSIVE_T1, HS300_DEFENSIVE_T2, get_sector_of,
    compute_sector_mainline_boosts, get_bear_severity_level
)
from limit_Execution_filter import get_limit_threshold_pct, is_limit_up_close, is_limit_down_close
import market_regime_v2

BASE_DIR = Path(__file__).resolve().parent
CACHE_2Y_DIR = BASE_DIR / "cache_2y"

def analyze_period_with_running_bi(raw_bars):
    """修复原版缺陷5：当最新一段单边暴涨/暴跌尚未形成右侧分型时，补充未完成笔(running bi)"""
    res = analyze_period(raw_bars)
    if not res or not raw_bars or len(raw_bars) < 5:
        return res
    last_bi = res.get("lastBi")
    if not last_bi:
        return res
    # 检查从 last_bi 结束时间之后是否已经走出显著反向突破但尚未确认分型
    end_t = last_bi["endTime"]
    post_bars = [b for b in raw_bars if b["time"] > end_t]
    if len(post_bars) >= 4:
        bars_macd = compute_macd(raw_bars)
        post_macd = [b for b in bars_macd if b["time"] > end_t]
        area = round(sum(abs(b["macd"]) for b in post_macd), 3)
        if last_bi["type"] == "up":
            min_b = min(post_bars, key=lambda x: x["low"])
            if min_b["low"] < last_bi["endPrice"] * 0.985:
                res["lastBi"] = {
                    "type": "down", "startTime": end_t, "endTime": min_b["time"],
                    "startPrice": last_bi["endPrice"], "endPrice": min_b["low"],
                    "high": last_bi["endPrice"], "low": min_b["low"],
                    "macdArea": area, "isRunning": True
                }
        else:
            max_b = max(post_bars, key=lambda x: x["high"])
            if max_b["high"] > last_bi["endPrice"] * 1.015:
                res["lastBi"] = {
                    "type": "up", "startTime": end_t, "endTime": max_b["time"],
                    "startPrice": last_bi["endPrice"], "endPrice": max_b["high"],
                    "high": max_b["high"], "low": last_bi["endPrice"],
                    "macdArea": area, "isRunning": True
                }
    return res

def is_open_limit_up(sym, dt, prev_close, open_p):
    if prev_close <= 0 or open_p <= 0:
        return False
    pct = (open_p - prev_close) / prev_close * 100.0
    return pct >= get_limit_threshold_pct(sym, dt)

def is_open_limit_down(sym, dt, prev_close, open_p):
    if prev_close <= 0 or open_p <= 0:
        return False
    pct = (open_p - prev_close) / prev_close * 100.0
    return pct <= -get_limit_threshold_pct(sym, dt)

def get_exact_day_bar(indexed_data, sym, dt):
    idx_e = indexed_data.get(sym)
    if not idx_e:
        return None
    pos = bisect.bisect_right(idx_e["day_dates"], dt) - 1
    if pos >= 0 and idx_e["day_dates"][pos] == dt:
        return idx_e["day"][pos]
    return None

def classify_regime_clean(bench_day_bars):
    """纯粹基于均线与缠论中枢的客观状态分类，剥离原版事后针对特定闪崩凑出的魔法数字"""
    if not bench_day_bars or len(bench_day_bars) < 30:
        return "RANGE", dict(market_regime_v2.REGIME_CONFIGS["RANGE"])
    closes = [b["close"] for b in bench_day_bars]
    c = closes[-1]
    ma20 = sum(closes[-20:]) / 20.0
    ma60 = sum(closes[-60:]) / 60.0 if len(closes) >= 60 else ma20
    ma120 = sum(closes[-120:]) / 120.0 if len(closes) >= 120 else ma60
    p = analyze_period_with_running_bi(bench_day_bars[-250:])
    zs = p.get("lastZs") if p else None
    if (c < ma60) and (ma60 < ma120):
        return "WEAK", dict(market_regime_v2.REGIME_CONFIGS["WEAK"])
    if (zs and c > zs["zg"]) or (c > ma60 and ma60 >= ma120):
        return "BULL", dict(market_regime_v2.REGIME_CONFIGS["BULL"])
    return "RANGE", dict(market_regime_v2.REGIME_CONFIGS["RANGE"])

def evaluate_candidates_at_dt(rosters_tuple, holdings, full_meta_map, indexed_data, dt, eval_cache, use_running_bi=False):
    hs_u, cyb_u, kcb_u = rosters_tuple
    active_list = hs_u + cyb_u + kcb_u
    seen_m = {m[0]: m for m in active_list}
    for sym in holdings:
        if sym in full_meta_map and sym not in seen_m:
            active_list.append(full_meta_map[sym])
            seen_m[sym] = full_meta_map[sym]

    day_snaps = {}
    for sym, _, _ in active_list:
        s = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
        if s:
            day_snaps[sym] = s

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
        ck = (sym, dt, use_running_bi)
        if ck in eval_cache:
            if eval_cache[ck] is not None:
                evaluated.append(eval_cache[ck])
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
        if use_running_bi and ev:
            ev["periods"]["15m"] = analyze_period_with_running_bi(m15_bars[-280:])
            ev["periods"]["day"] = analyze_period_with_running_bi(d_bars[-250:])
        eval_cache[ck] = ev
        evaluated.append(ev)
    return day_snaps, evaluated


def compute_trade_exec_and_fee(sym: str, dt: str, raw_price: float, shares: int, action: str, realistic_cost: bool):
    """
    A股历史真实分段税费与滑点计算：
    - 场内ETF (sh51*/sh58*/sz15*)：全时期免征印花税与过户费，双边佣金万1.5(最低1元)，单边滑点 3 bps (0.03%)
    - A股个股 (dt < 2023-08-28)：买入万2.6(最低5元)，卖出千1印花税+万2.6佣金过户=0.126%(最低5元)，单边滑点 5 bps (0.05%)
    - A股个股 (dt >= 2023-08-28 印花税减半后)：买入万2.6(最低5元)，卖出万5印花税+万2.6佣金过户=0.076%(最低5元)，单边滑点 5 bps (0.05%)
    """
    is_etf = sym.startswith("sh51") or sym.startswith("sh58") or sym.startswith("sz15")
    if not realistic_cost:
        exec_p = round(raw_price, 3)
        gross = shares * exec_p
        rate = 0.00025 if action == "BUY" else 0.0005
        fee = max(gross * rate, 5.0)
        return exec_p, gross, round(fee, 2)

    slip = 0.0003 if is_etf else 0.0005
    exec_p = round(raw_price * (1.0 + slip if action == "BUY" else 1.0 - slip), 3)
    gross = shares * exec_p
    if is_etf:
        fee = max(gross * 0.00015, 1.0)
    else:
        if action == "BUY":
            fee = max(gross * 0.00026, 5.0)
        else:
            stamp_duty = 0.0005 if dt >= "2023-08-28" else 0.0010
            fee = max(gross * (stamp_duty + 0.00026), 5.0)
    return exec_p, gross, round(fee, 2)

def run_audit_sim(name, rosters_by_date, full_meta_map, indexed_data, trade_dates, bench_map, eval_cache,
                  use_next_open=False, realistic_cost=False, remove_hindsight=False):
    holdings = {}
    cash = 1000000.0
    initial_capital = 1000000.0
    closed_trades = []
    trades = []
    pending_buy_orders = []
    init_bench = bench_map[trade_dates[0]]
    bench_idx = indexed_data["sh000300"]
    history = []

    buy_fee_rate = 0.00025
    sell_fee_rate = 0.00125 if realistic_cost else 0.0005  # 含印花税+佣金+过户费
    slip_pct = 0.0008 if realistic_cost else 0.0           # 单边冲击滑点 8 bps

    for dt in trade_dates:
        # Step A: 若启用次日开盘成交(use_next_open)，先在今日开盘(Open)执行昨日收盘产生的卖单和买单
        if use_next_open:
            for sym in list(holdings.keys()):
                h = holdings[sym]
                if not h.get("pending_sell_reason"):
                    continue
                dbar = get_exact_day_bar(indexed_data, sym, dt)
                if not dbar:
                    continue  # 停牌无法卖出
                if is_open_limit_down(sym, dt, h["last_price"], dbar["open"]):
                    continue  # 开盘一字跌停无法卖出，继续顺延
                exec_p, gross, fee = compute_trade_exec_and_fee(sym, dt, dbar["open"], h["shares"], "SELL", realistic_cost)
                cash += (gross - fee)
                closed_trades.append({
                    "symbol": sym, "buy_date": h["buy_date"], "sell_date": dt,
                    "pnl_amt": round((gross - fee) - h["shares"] * h["avg_cost"], 2),
                    "pnl_pct": round((exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0, 2)
                })
                trades.append({"date": dt, "action": "SELL", "symbol": sym, "price": exec_p})
                del holdings[sym]

            for od in pending_buy_orders:
                sym = od["symbol"]
                if sym in holdings:
                    continue
                dbar = get_exact_day_bar(indexed_data, sym, dt)
                if not dbar or is_open_limit_up(sym, dt, od["signal_close"], dbar["open"]):
                    continue  # 停牌或次日开盘一字涨停买不进
                is_etf = sym.startswith("sh51") or sym.startswith("sh58") or sym.startswith("sz15")
                slip_b = (0.0003 if is_etf else 0.0005) if realistic_cost else 0.0
                est_p = round(dbar["open"] * (1.0 + slip_b), 3)
                shares = int(min(cash * 0.95, od["alloc"]) // (est_p * 100.0)) * 100
                if shares < 100:
                    continue
                exec_p, gross, fee = compute_trade_exec_and_fee(sym, dt, dbar["open"], shares, "BUY", realistic_cost)
                if gross + fee > cash:
                    continue
                cash -= (gross + fee)
                holdings[sym] = {
                    "name": od["name"], "shares": shares, "avg_cost": round((gross + fee) / shares, 4),
                    "last_price": exec_p, "buy_date": dt, "init_stop": od["init_stop"]
                }
                trades.append({"date": dt, "action": "BUY", "symbol": sym, "price": exec_p})
            pending_buy_orders = []

        # Step B: 更新今日收盘价
        for sym, h in holdings.items():
            snap = indexed_data.get(sym, {}).get("snaps", {}).get(dt)
            if snap and snap["lastPrice"] > 0:
                h["last_price"] = snap["lastPrice"]

        b_pos = bisect.bisect_right(bench_idx["day_dates"], dt)
        bench_day_slice = bench_idx["day"][max(0, b_pos - 250):b_pos]
        day_snaps, evaluated = evaluate_candidates_at_dt(
            rosters_by_date[dt], holdings, full_meta_map, indexed_data, dt, eval_cache, use_running_bi=remove_hindsight
        )
        eval_map = {e["symbol"]: e for e in evaluated}

        if remove_hindsight:
            regime, r_cfg = classify_regime_clean(bench_day_slice)
        else:
            regime, r_cfg, _ = market_regime_v2.classify_market_regime(bench_day_slice, day_snaps, evaluated, mode="dual_factor")

        # Step C: 收盘卖出信号检查
        for sym in list(holdings.keys()):
            h = holdings[sym]
            dbar = get_exact_day_bar(indexed_data, sym, dt)
            if remove_hindsight and not dbar:
                continue  # 修复停牌日按前收盘卖出漏洞
            ev = eval_map.get(sym)
            snap = day_snaps.get(sym)
            sell_reason = h.get("pending_sell_reason")
            if not sell_reason and ev:
                p_m15 = ev["periods"].get("15m")
                if p_m15:
                    zs15 = p_m15.get("lastZs")
                    bi15 = p_m15.get("lastBi")
                    div15 = p_m15.get("divRatio")
                    if regime in ("WEAK", "CRASH") and ev["price"] < h["init_stop"]:
                        sell_reason = "ratchet_stop"
                    elif zs15 and ev["price"] < zs15["dd"]:
                        sell_reason = "zs_dd_break"
                    elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.7:
                        sell_reason = "m15_top_div"
            if sell_reason:
                if use_next_open:
                    h["pending_sell_reason"] = sell_reason
                    continue
                if is_limit_down_close(sym, dt, snap, dbar):
                    h["pending_sell_reason"] = sell_reason
                    continue
                exec_p, gross, fee = compute_trade_exec_and_fee(sym, dt, h["last_price"], h["shares"], "SELL", realistic_cost)
                cash += (gross - fee)
                closed_trades.append({
                    "symbol": sym, "buy_date": h["buy_date"], "sell_date": dt,
                    "pnl_amt": round((gross - fee) - h["shares"] * h["avg_cost"], 2),
                    "pnl_pct": round((exec_p - h["avg_cost"]) / h["avg_cost"] * 100.0, 2)
                })
                trades.append({"date": dt, "action": "SELL", "symbol": sym, "price": exec_p})
                del holdings[sym]

        # Step D: 收盘买入信号检查
        sec_boosts, _ = compute_sector_mainline_boosts(evaluated)
        bear_sev = get_bear_severity_level(bench_day_slice) if regime == "WEAK" else "NONE"

        def route_priority(cand):
            sym = cand["symbol"]
            sec = get_sector_of(sym)
            base_s = cand["score"]
            p_day = cand["periods"].get("day")
            zs_day = p_day.get("lastZs") if p_day else None
            above_day_zg = bool(zs_day and p_day["close"] > zs_day["zg"])
            overext = 0
            if zs_day and zs_day["zg"] > 0:
                dist = (p_day["close"] - zs_day["zg"]) / zs_day["zg"]
                if dist > 0.38: overext = 25
                elif dist > 0.28: overext = 10
            if remove_hindsight:
                # 剔除事后红利白名单，仅保留客观评分与涨幅过热惩罚
                adj_s = base_s - overext
            elif regime in ("BULL", "RANGE"):
                m_boost = sec_boosts.get(sec, 0)
                if not above_day_zg: m_boost //= 2
                adj_s = base_s + m_boost - overext
            else:
                if bear_sev == "WEAK_DEEP":
                    adj_s = base_s + (26 if sym in HS300_DEFENSIVE_T1 else (14 if sym in HS300_DEFENSIVE_T2 else -18)) - overext
                else:
                    adj_s = base_s + (15 if sym in HS300_DEFENSIVE_T1 else (8 if sym in HS300_DEFENSIVE_T2 else 0)) - overext
            s = cand.get("snapshot")
            return (adj_s, s["buyerDom"] if s else -999.0, s["amountYi"] if s else 0.0)

        ranked_cands = sorted(evaluated, key=route_priority, reverse=True)
        mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
        cur_equity = cash + mkt_val_now
        remaining_room = max(0.0, cur_equity * r_cfg["max_exposure_pct"] - mkt_val_now)
        budget = min(cash, remaining_room)
        buy_count = 0

        if r_cfg["daily_buy_limit"] > 0 and remaining_room >= 75000.0:
            for cand in ranked_cands:
                if buy_count >= r_cfg["daily_buy_limit"]:
                    break
                sym = cand["symbol"]
                if sym in holdings or cand["score"] < r_cfg["min_score"] or budget < 75000.0 or cand["price"] <= 0:
                    continue
                dbar = get_exact_day_bar(indexed_data, sym, dt)
                if not dbar:
                    continue
                snap = day_snaps.get(sym)
                alloc = min(budget * 0.95, 250000.0 * r_cfg["single_pos_scale"], remaining_room)
                p_m15 = cand["periods"].get("15m")
                zs15 = p_m15.get("lastZs") if p_m15 else None
                raw_stop = round(zs15["zd"] * 0.99, 3) if zs15 else round(cand["price"] * 0.96, 3)

                if use_next_open:
                    pending_buy_orders.append({
                        "symbol": sym, "name": cand["name"], "alloc": alloc,
                        "signal_close": cand["price"], "init_stop": raw_stop
                    })
                    budget -= alloc
                    remaining_room -= alloc
                    buy_count += 1
                else:
                    if is_limit_up_close(sym, dt, snap, dbar):
                        continue
                    is_etf = sym.startswith("sh51") or sym.startswith("sh58") or sym.startswith("sz15")
                    slip_b = (0.0003 if is_etf else 0.0005) if realistic_cost else 0.0
                    est_p = round(cand["price"] * (1.0 + slip_b), 3)
                    shares = int(alloc // (est_p * 100.0)) * 100
                    if shares < 100:
                        continue
                    exec_p, gross, fee = compute_trade_exec_and_fee(sym, dt, cand["price"], shares, "BUY", realistic_cost)
                    if gross + fee > cash:
                        continue
                    cash -= (gross + fee)
                    budget -= gross
                    remaining_room -= gross
                    buy_count += 1
                    holdings[sym] = {
                        "name": cand["name"], "shares": shares, "avg_cost": round((gross + fee) / shares, 4),
                        "last_price": cand["price"], "buy_date": dt, "init_stop": raw_stop
                    }
                    trades.append({"date": dt, "action": "BUY", "symbol": sym, "price": exec_p})

        mkt_val = sum(h["shares"] * h["last_price"] for h in holdings.values())
        eq = round(cash + mkt_val, 2)
        p_ret = round((eq - initial_capital) / initial_capital * 100.0, 2)
        b_ret = round((bench_map[dt] - init_bench) / init_bench * 100.0, 2)
        history.append({
            "date": dt, "equity": eq, "portfolio_value": eq,
            "portfolio_return_pct": p_ret, "hs300_return_pct": b_ret,
            "alpha_pct": round(p_ret - b_ret, 2)
        })

    return compute_metrics({"id": name, "name": name, "initial_capital": initial_capital, "final_value": history[-1]["portfolio_value"], "history": history, "trades": trades, "closed_trades": closed_trades})

def main():
    hs_pool, cyb_pool, kcb_pool = get_candidate_pools()
    idx_real = pickle.load(open(BASE_DIR / "cache_2y_lite_idx.pkl", "rb"))
    cache_synth = pickle.load(open(BASE_DIR / "cache_synth_2y_eval.pkl", "rb"))
    cache_real = pickle.load(open(BASE_DIR / "cache_real_2y_eval.pkl", "rb"))

    bench_bars_all = idx_real["sh000300"]["day"]
    dates_2y = [b["time"] for b in bench_bars_all if "2024-07-22" <= b["time"] <= "2026-10-29"]
    bmap_2y = {b["time"]: b["close"] for b in bench_bars_all if b["time"] in set(dates_2y)}
    rosters_2y, fmap = precompute_pit_rosters(hs_pool, cyb_pool, kcb_pool, idx_real, dates_2y)

    print("=== 实证一：2024-07-22 ~ 2026-10-29 (550交易日) 真实分钟线 vs 伪造分钟线 & 逐级去水分审计 ===")
    configs = [
        ("L0_原版基准(日线伪造分钟线+当日收盘0滑点+低费率+事后白名单)", idx_real, cache_synth, False, False, False),
        ("L1_仅换用真实15m/60m分钟线(其余同原版)", idx_real, cache_real, False, False, False),
        ("L2_真实分钟线+次日开盘价(Next-Open)成交", idx_real, cache_real, True, False, False),
        ("L3_真实分钟线+次日开盘+真实印花税与8bps滑点", idx_real, cache_real, True, True, False),
        ("L4_终极零水分(真分钟+次日开盘+真费率+删事后白名单+修未完成笔)", idx_real, cache_real, True, True, True),
    ]
    for label, idx_data, ec, nxt_open, real_cost, no_hind in configs:
        market_regime_v2._BENCH_CACHE.clear()
        res = run_audit_sim(label, rosters_2y, fmap, idx_data, dates_2y, bmap_2y, ec,
                            use_next_open=nxt_open, realistic_cost=real_cost, remove_hindsight=no_hind)
        m = res
        yrs = m["trading_days"] / 242.0
        cagr = ((m["final_value"] / m["initial_capital"]) ** (1.0 / yrs) - 1.0) * 100.0
        print(f"{label}:\n  总收益={m['total_return_pct']:+.2f}% (沪深300={m['hs300_return_pct']:+.2f}%, 超额={m['alpha_pct']:+.2f}%) | 年化={cagr:+.2f}% | 最大回撤=-{m['max_drawdown_pct']:.2f}% | 夏普={m['sharpe_ratio']} | 胜率={m['win_rate_pct']:.1f}% (盈亏比={m['profit_loss_ratio']}, 闭环={m['closed_roundtrips']}笔)")

if __name__ == "__main__":
    main()
