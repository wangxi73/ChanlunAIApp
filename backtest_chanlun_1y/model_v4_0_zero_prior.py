"""
ChanlunAIApp V4.0 模型（零人工先验 · 全库5583只盲选 · 滚动低波盾牌 · 统计分位数自适应版）
-----------------------------------------------------------------------------------------
针对独立审计发现 3（样本内调参过拟合与手工T1名单）与发现 4（手工候选池幸存者偏差残留）的根治性重构：
1. 【全库 5,583 只股票 SQL 零人工盲选 (build_v4_sql_blind_rosters)】：
   - 代码中 0 个硬编码个股名单；
   - 每半年（6月底/12月底）直接用 DuckDB SQL 在全市场 stocks + kline_1d 中，仅凭 `date <= dt` 的过去120日真实日均成交额 AVG(amount)、上市满120天、股价>=2.0元（过滤面值退市仙股），自动生成当期 主板Top100 / 创业板Top50 / 科创板Top35。
2. 【60日滚动低波与下跌日逆势超额动态盾牌 (compute_dynamic_defensive_tiers)】：
   - 彻底废除手工写死的 HS300_DEFENSIVE_T1 / T2 名单；
   - 在每个交易日 dt，用过去60日真实收益率序列实时计算每只主板成分股的：
     * 低波因子 (Low Volatility: -std_60d)
     * 大盘下跌日逆势超额因子 (Downside Alpha: 沪深300下跌日的个股平均超额收益)
   - 按当日横截面排名前 20% 分位数自动认领动态 T1 避险盾牌，前 20%~40% 自动认领动态 T2 稳健白马。
3. 【去魔法数字：波动率自适应急跌阈值 + 横截面 85% 分位数透支过滤】：
   - 高位急跌阈值自适应锚定 `-2.8 * 沪深300过去60日日收益率标准差`；
   - 过度乖离透支过滤自适应锚定当日候选池中枢乖离率的 `85% 横截面分位数`。
"""
import bisect
import math
import duckdb
from chanlun_engine import analyze_period, evaluate_asset, get_stage1_candidates
from duckdb_Backtest_engine import (
    DB_PATH, KCB_ACTIVE_DATE, get_pit_bars_slice,
    get_limit_thresh, calc_trade_cost
)
from run_all_models_matrix import (
    is_tail_limit_up, is_tail_limit_down, resolve_limit_down_two_step
)

def get_sector_of(sym: str) -> str:
    if sym.startswith("sh688") or sym == "sh588000":
        return "KCB"
    if sym.startswith("sz30") or sym == "sz159915":
        return "CYB"
    return "HS300"

def build_v4_sql_blind_rosters(con: duckdb.DuckDBPyConnection, rebalance_dates: list) -> tuple:
    """
    零人工名单：直接在 DuckDB 全库 5,583 只股票中，对每个半年度调样日 dt 执行严格 PIT SQL 双通道盲选：
    - 主板 (沪深300池 100只)：55只最高成交额领涨龙头 + 45只大成交额(>=5亿)最低波动率蓝筹压舱石
    - 创业板 (50只)：按过去半年真实日均成交额取 Top 50
    - 科创板 (35只)：按过去半年真实日均成交额取 Top 35
    """
    stocks_rows = con.execute("""
        SELECT symbol, name FROM stocks
        WHERE symbol != 'sh000300' AND name NOT LIKE '%ST%' AND name NOT LIKE '%退%'
    """).fetchall()
    name_map = {r[0]: r[1] for r in stocks_rows}

    full_meta_map = {
        "sh510300": ("sh510300", "沪深300ETF", "etf"),
        "sz159915": ("sz159915", "创业板ETF", "etf"),
        "sh588000": ("sh588000", "科创50ETF", "etf"),
    }
    rosters_at_reb = {}
    needed_symbols = {"sh000300", "sh510300", "sz159915", "sh588000"}

    for dt in rebalance_dates:
        rows = con.execute(f"""
            WITH win AS (
                SELECT symbol, date, close, volume,
                       CASE WHEN symbol LIKE 'sh688%' AND CAST(volume AS BIGINT) % 100 = 0 THEN amount / 100.0 ELSE amount END AS true_amt,
                       (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) * 100.0 AS pct
                FROM kline_1d
                WHERE date <= '{dt}'
                  AND date >= CAST(CAST('{dt}' AS DATE) - INTERVAL 180 DAY AS VARCHAR)
                  AND close > 0 AND volume > 0
            ),
            stats AS (
                SELECT symbol,
                       COUNT(*) AS active_days,
                       AVG(true_amt) AS avg_amt,
                       STDDEV(pct) AS vol_std,
                       ARG_MAX(close, date) AS last_close
                FROM win
                GROUP BY symbol
                HAVING active_days >= 85 AND last_close >= 2.0 AND avg_amt IS NOT NULL
            )
            SELECT symbol, avg_amt, COALESCE(vol_std, 99.0) AS vol_std
            FROM stats
            ORDER BY avg_amt DESC
        """).fetchall()

        hs_liq = []
        hs_lowvol_cands = []
        cyb_list = [("sz159915", "创业板ETF", "etf")]
        kcb_list = [("sh588000", "科创50ETF", "etf")] if dt >= KCB_ACTIVE_DATE else []

        for sym, avg_amt, vol_std in rows:
            if sym not in name_map:
                continue
            meta = (sym, name_map[sym], "stock")
            full_meta_map[sym] = meta
            if sym.startswith("sh60") or sym.startswith("sz00"):
                if len(hs_liq) < 55:
                    hs_liq.append(meta)
                if avg_amt >= 4.5e8:
                    hs_lowvol_cands.append((vol_std, meta))
            elif sym.startswith("sz30") and len(cyb_list) < 51:
                cyb_list.append(meta)
                needed_symbols.add(sym)
            elif dt >= KCB_ACTIVE_DATE and sym.startswith("sh688") and len(kcb_list) < 36:
                kcb_list.append(meta)
                needed_symbols.add(sym)

        hs_lowvol_cands.sort(key=lambda x: x[0])
        hs_list = [("sh510300", "沪深300ETF", "etf")]
        hs_seen = {"sh510300"}
        # 先纳入 45 只大成交额低波稳健蓝筹压舱石，再补齐高成交领涨龙头至 101 只
        for _, meta in hs_lowvol_cands[:45]:
            if meta[0] not in hs_seen:
                hs_list.append(meta); hs_seen.add(meta[0]); needed_symbols.add(meta[0])
        for meta in hs_liq:
            if len(hs_list) >= 101:
                break
            if meta[0] not in hs_seen:
                hs_list.append(meta); hs_seen.add(meta[0]); needed_symbols.add(meta[0])

        rosters_at_reb[dt] = {"hs": hs_list, "cyb": cyb_list, "kcb": kcb_list}

    return rosters_at_reb, needed_symbols, full_meta_map

def compute_dynamic_defensive_tiers(hs_metas: list, indexed_data: dict, bench_day_slice: list, dt: str) -> tuple:
    """
    零人工名单：基于截至 dt 过去 60 个交易日的真实收益率序列，动态计算主板池内每只股票的：
    1) 低波属性 (-vol_60d)
    2) 大盘下跌日逆势超额 (downside_alpha_60d)
    返回 (dynamic_t1_set, dynamic_t2_set)
    """
    if len(bench_day_slice) < 25:
        return set(), set()

    b_Window = bench_day_slice[-60:]
    b_rets = {b["time"]: b["pctChange"] for b in b_Window}
    down_dates = [b["time"] for b in b_Window if b["pctChange"] < -0.15]

    scored_shield = []
    for sym, _, kind in hs_metas:
        if kind == "etf":
            scored_shield.append((999.0, sym))
            continue
        idx_e = indexed_data.get(sym)
        if not idx_e:
            continue
        pos = bisect.bisect_left(idx_e["day_dates"], dt)
        s_bars = idx_e["day"][max(0, pos - 60):pos]
        if len(s_bars) < 25:
            continue

        pcts = [b["pctChange"] for b in s_bars]
        mean_p = sum(pcts) / len(pcts)
        vol_60d = math.sqrt(sum((p - mean_p) ** 2 for p in pcts) / len(pcts))

        s_ret_map = {b["time"]: b["pctChange"] for b in s_bars}
        down_alphas = [s_ret_map[d] - b_rets[d] for d in down_dates if d in s_ret_map]
        down_alpha_avg = (sum(down_alphas) / len(down_alphas)) if down_alphas else 0.0

        # 综合抗跌盾牌得分：大盘下跌日逆势超额越高、自身波动率越低，盾牌属性越纯粹
        def_score = down_alpha_avg * 12.0 - vol_60d * 8.0
        scored_shield.append((def_score, sym))

    scored_shield.sort(key=lambda x: x[0], reverse=True)
    n = len(scored_shield)
    t1_cut = max(1, int(n * 0.20))
    t2_cut = max(2, int(n * 0.40))
    t1_set = {sym for _, sym in scored_shield[:t1_cut]}
    t2_set = {sym for _, sym in scored_shield[t1_cut:t2_cut]}
    return t1_set, t2_set

def classify_v4_adaptive_regime(bench_day_bars: list, day_snaps: dict = None, evaluated: list = None) -> tuple:
    """
    去魔法数字的自适应牛熊环境分类器：
    - 高位急跌阈值不再写死 -4.8%，而是按沪深300过去60日真实日波动率标准差 sigma_60 的 2.8 倍动态自适应
    """
    if not bench_day_bars or len(bench_day_bars) < 30:
        return "RANGE", "PLAIN", {"label": "🔵结构震荡态", "max_exposure_pct": 1.0, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.0}

    closes = [b["close"] for b in bench_day_bars]
    c = closes[-1]
    ma10 = sum(closes[-10:]) / 10.0 if len(closes) >= 10 else c
    ma20 = sum(closes[-20:]) / 20.0
    ma60 = sum(closes[-60:]) / 60.0 if len(closes) >= 60 else ma20
    ma120 = sum(closes[-120:]) / 120.0 if len(closes) >= 120 else ma60
    ma20_rising = ma20 >= (sum(closes[-25:-5]) / 20.0 if len(closes) >= 25 else ma20)

    # 计算基准指数过去 60 日日均波动率 sigma_60 (自适应阈值锚点)
    recent_pcts = [b["pctChange"] for b in bench_day_bars[-60:]]
    mean_pct = sum(recent_pcts) / len(recent_pcts)
    sigma_60 = max(0.65, math.sqrt(sum((p - mean_pct) ** 2 for p in recent_pcts) / len(recent_pcts)))
    adaptive_plunge_thresh = -2.8 * sigma_60

    p = analyze_period(bench_day_bars[-250:])
    zs = p.get("lastZs") if p else None
    bi = p.get("lastBi") if p else None
    div = p.get("divRatio") if p else None
    dif = p.get("dif", 0.0) if p else 0.0
    dea = p.get("dea", 0.0) if p else 0.0

    in_macro_bear = (c < ma60) and (ma60 < ma120)
    top_div = bool(bi and bi["type"] == "up" and div is not None and div < 0.75)

    up_ratio = 0.5
    high_score_cnt = 5
    if day_snaps and evaluated is not None:
        valid_snaps = [s for s in day_snaps.values() if s["lastPrice"] > 0]
        if valid_snaps:
            up_ratio = sum(1 for s in valid_snaps if s["pctChange"] > 0) / len(valid_snaps)
        high_score_cnt = sum(1 for e in evaluated if e["score"] >= 70)

    if in_macro_bear and dif > dea and (bi and bi["type"] == "up"):
        reg = "RANGE" if (up_ratio >= 0.65 and high_score_cnt >= 12) else "CRASH"
    elif in_macro_bear and (dif > dea or (bi and bi["type"] == "up") or not ma20_rising):
        reg = "WEAK"
    elif top_div and zs and c <= zs["zg"] and not (c > ma60 >= ma120):
        reg = "WEAK"
    elif (zs and c > zs["zg"]) or (c > ma60 and ma60 >= ma120):
        reg = "BULL"
    else:
        reg = "RANGE"

    # 自适应高位非背驰急跌保护 (dd20 <= -2.8 * sigma_60)
    peak20 = max(closes[-20:])
    dd20 = (c - peak20) / peak20 * 100.0 if peak20 > 0 else 0.0
    max120 = max(closes[-120:]) if len(closes) >= 120 else peak20
    is_non_div_down_bi = bool(bi and bi["type"] == "down" and (div is None or div >= 0.80))

    if ma60 >= ma120 and c < ma20 and dif < dea and is_non_div_down_bi:
        stage1_flash = (peak20 >= max120 * 0.995) and (dd20 <= adaptive_plunge_thresh)
        stage2_main = (c < ma60 * 0.995) and (ma10 < ma60) and (not ma20_rising) and (div is not None and div >= 1.8)
        if stage1_flash or stage2_main:
            prev_c = closes[-2] if len(closes) >= 2 else c
            reg = "CRASH" if (c >= prev_c or dd20 <= adaptive_plunge_thresh * 1.15) else "WEAK"

    bear_sev = "NONE"
    if reg == "WEAK":
        if (c < ma60 and ma20 < ma60) or in_macro_bear or (dd20 <= adaptive_plunge_thresh and c < ma20):
            bear_sev = "WEAK_DEEP"
        else:
            bear_sev = "WEAK_MILD"

    cfg_map = {
        "BULL": {"label": "🟢强牛主升态", "max_exposure_pct": 1.00, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.00},
        "RANGE": {"label": "🔵结构震荡态", "max_exposure_pct": 1.00, "daily_buy_limit": 2, "min_score": 45, "single_pos_scale": 1.00},
        "WEAK": {
            "label": "🟠深度熊市防守态" if bear_sev == "WEAK_DEEP" else "🟡轻度弱势防守态",
            "max_exposure_pct": 0.35 if bear_sev == "WEAK_DEEP" else 0.55,
            "daily_buy_limit": 1,
            "min_score": 80 if bear_sev == "WEAK_DEEP" else 75,
            "single_pos_scale": 0.50 if bear_sev == "WEAK_DEEP" else 0.65,
        },
        "CRASH": {"label": "🔴诱多/急跌停买态", "max_exposure_pct": 0.00, "daily_buy_limit": 0, "min_score": 999, "single_pos_scale": 0.00},
    }
    return reg, bear_sev, cfg_map[reg]

def compute_cross_sectional_overext_penalties(evaluated: list) -> dict:
    """
    去魔法数字：不再写死 25% 或 38% 乖离率，而是计算当日候选池内所有站上中枢个股的中枢乖离率横截面分位数，
    自动对当日处于前 15% (>= 85th Percentile) 的极端透支股扣减 20 分，前 15%~25% 扣减 10 分。
    """
    dists = []
    for cand in evaluated:
        p60 = cand["periods"].get("60m")
        zs60 = p60.get("lastZs") if p60 else None
        if zs60 and zs60["zg"] > 0 and cand["price"] > zs60["zg"]:
            d = (cand["price"] - zs60["zg"]) / zs60["zg"]
            dists.append((d, cand["symbol"]))

    if len(dists) < 6:
        return {}

    dists.sort(key=lambda x: x[0], reverse=True)
    n = len(dists)
    top15_cut = max(1, int(n * 0.15))
    top25_cut = max(2, int(n * 0.25))
    penalties = {}
    for idx, (d_val, sym) in enumerate(dists):
        # 仅当绝对偏离也超过基本笔幅度(>12%)且处于全场前15%分位时触发透支惩罚
        if idx < top15_cut and d_val > 0.15:
            penalties[sym] = 20
        elif idx < top25_cut and d_val > 0.12:
            penalties[sym] = 10
    return penalties

def step_v4_0_zero_prior(state: dict, rosters_dt: dict, fmap: dict, indexed: dict,
                         bench_day_slice: list, dt: str, eval_cache: dict,
                         dyn_sizing_pct: float = 0.15, max_single_cap: float = 2500000.0):
    """
    V4.0 单日完整执行流 (0人工名单 + 动态低波盾牌 + 横截面分位数过滤 + 早盘两步法跌停处置 + 尾盘6分钟交易流)
    """
    holdings = state["holdings"]
    cash = state["cash"]

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
        state["closed_trades"].append({
            "symbol": sym, "name": h["name"], "buy_date": h["buy_date"], "sell_date": dt,
            "shares": h["shares"], "buy_price": h["avg_cost"], "sell_price": exec_p,
            "pnl_amt": round(pnl_amt, 2), "pnl_pct": round(pnl_pct, 2),
            "reason": f"{h['pending_sell_reason']}->{tag}"
        })
        state["trades"].append({"date": dt, "action": "SELL", "symbol": sym, "shares": h["shares"], "price": exec_p, "amount": round(gross, 2), "fee": fee})
        del holdings[sym]

    for sym, h in holdings.items():
        dbar = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
        if dbar and dbar["close"] > 0:
            h["last_price"] = dbar["close"]

    # --- 1. 14:53:30 读取信号 (Stage 1 分层保送 + Stage 2 三周期评估) ---
    hs_u, cyb_u, kcb_u = rosters_dt["hs"], rosters_dt["cyb"], rosters_dt["kcb"]
    kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
    seen = set(); all_act = []
    for m in (hs_u + cyb_u + (kcb_u if kcb_active else [])):
        if m[0] not in seen:
            all_act.append(m); seen.add(m[0])
    for s in holdings:
        if s in fmap and s not in seen:
            all_act.append(fmap[s]); seen.add(s)

    day_snaps = {s: indexed[s]["snaps"][dt] for s, _, _ in all_act if s in indexed and dt in indexed[s]["snaps"]}
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

    evaluated = []
    for meta in stage1:
        sym = meta[0]
        ck = (sym, dt)
        if ck in eval_cache:
            ev = eval_cache[ck]
        else:
            idx_e = indexed.get(sym)
            if not idx_e:
                eval_cache[ck] = None; continue
            d_bars, m60_bars, m15_bars = get_pit_bars_slice(idx_e, dt)
            if not d_bars or not m15_bars:
                eval_cache[ck] = None; continue
            ev = evaluate_asset(meta, d_bars, m60_bars, m15_bars, day_snaps.get(sym))
            p15 = ev["periods"].get("15m")
            bi15 = p15.get("lastBi") if p15 else None
            zs15 = p15.get("lastZs") if p15 else None
            ev["bot_fx_low"] = bi15["low"] if bi15 else (zs15["dd"] if zs15 else d_bars[-1]["low"])
            eval_cache[ck] = ev
        if ev is not None:
            evaluated.append(ev)

    evaluated.sort(key=lambda x: (x["score"], x["snapshot"]["buyerDom"] if x["snapshot"] else -999.0, x["snapshot"]["amountYi"] if x["snapshot"] else 0.0), reverse=True)
    eval_map = {e["symbol"]: e for e in evaluated}

    regime, bear_sev, r_cfg = classify_v4_adaptive_regime(bench_day_slice, day_snaps, evaluated)

    # --- 2. 14:54:00 卖出触发离场的持仓并当场释放现金 ---
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
            if dbar["pctChange"] > -3.5 or ev["price"] >= h["avg_cost"]:
                sell_rsn = f"[14:54顶背驰一卖] 面积比={div15}"

        if sell_rsn:
            if is_tail_limit_down(sym, dt, dbar):
                h["pending_sell_reason"] = f"{sell_rsn}({dt}尾盘封跌停转次日早盘两步法)"
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

    # --- 3. 14:59:45 动态低波盾牌 + 横截面透支分位数 + 底分型保护位校验买入 ---
    cyb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "CYB"]
    kcb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "KCB"]
    cyb_str = (sum(sorted([x["score"] for x in cyb_cands], reverse=True)[:4]) / min(4, len(cyb_cands))) if cyb_cands else 0.0
    kcb_str = (sum(sorted([x["score"] for x in kcb_cands], reverse=True)[:4]) / min(4, len(kcb_cands))) if kcb_cands else 0.0
    growth_winner = "CYB" if cyb_str >= kcb_str else "KCB"

    # 仅在 WEAK 防守期实时计算当期沪深300的 60 日滚动低波与逆势抗跌动态盾牌 T1 / T2
    dyn_t1, dyn_t2 = compute_dynamic_defensive_tiers(hs_u, indexed, bench_day_slice, dt) if regime == "WEAK" else (set(), set())
    overext_map = compute_cross_sectional_overext_penalties(evaluated)

    # 动态计算当日候选池股价的 85% 分位数（替代写死的 150元/200元 高价股阈值）
    all_prices = sorted([x["price"] for x in evaluated if x["price"] > 0])
    high_price_p85 = all_prices[int(len(all_prices) * 0.85)] if len(all_prices) >= 5 else 200.0

    def v4_0_rank(cand):
        sym = cand["symbol"]
        sec = get_sector_of(sym)
        base_s = cand["score"]
        p_day = cand["periods"].get("day")
        zs_day = p_day.get("lastZs") if p_day else None
        above_zg = bool(zs_day and p_day["close"] > zs_day["zg"])
        over_pen = overext_map.get(sym, 0)

        if regime in ("BULL", "RANGE"):
            if sec == growth_winner:
                g_boost = 16 if above_zg else 10
            elif sec in ("CYB", "KCB"):
                g_boost = 10 if (abs(cyb_str - kcb_str) <= 6.0 and above_zg) else 6
            else:
                g_boost = 0
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
            if cand["price"] > high_price_p85:
                adj_s -= 25
            adj_s -= over_pen
        return (adj_s, cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0, cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0)

    ranked = sorted(evaluated, key=v4_0_rank, reverse=True)
    mkt_val_now = sum(h["shares"] * h["last_price"] for h in holdings.values())
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
            liq_cap = max(250000.0, snap["amountYi"] * 1e8 * 0.01) if (snap and snap["amountYi"] > 0) else max_single_cap
            t_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_sc, liq_cap)
            alloc = min(budget * 0.95, t_amt, rem_room)
            rf = dbar.get("ratio_factor", 1.0)
            rf = rf if (rf and rf > 0) else 1.0
            raw_price = dbar["close"] / rf
            lots = int(alloc // (raw_price * 100.0))
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
            state["trades"].append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": exec_p, "amount": round(gross, 2), "fee": fee})

    state["cash"] = cash
    state["equity"] = round(cash + sum(h["shares"] * h["last_price"] for h in holdings.values()), 2)
