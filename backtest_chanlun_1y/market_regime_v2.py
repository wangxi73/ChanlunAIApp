from chanlun_engine import analyze_period

_BENCH_CACHE = {}

REGIME_CONFIGS = {
    "BULL": {
        "label": "🟢强牛主升态",
        "max_exposure_pct": 1.00,   # 牛市满仓运行 (100%上限)
        "daily_buy_limit": 2,       # 每日最多新开仓 2 只（牛市多交易）
        "min_score": 45,            # 标准买点门槛 45分
        "single_pos_scale": 1.00,   # 单笔标准金额 100%
        "max_stop_pct": 0.08,
    },
    "RANGE": {
        "label": "🔵结构震荡/超跌一买态",
        "max_exposure_pct": 1.00,   # 结构行情与超跌底背驰一买区正常轮动
        "daily_buy_limit": 2,       # 每日最多新开仓 2 只
        "min_score": 45,            # 标准买点门槛 45分
        "single_pos_scale": 1.00,
        "max_stop_pct": 0.07,
    },
    "WEAK": {
        "label": "🟡熊市弱势/高位急跌防守态",
        "max_exposure_pct": 0.70,   # 总仓位上限压降至 70%（少交易、降仓位）
        "daily_buy_limit": 1,       # 每日最多新开仓 1 只（砍半交易频率）
        "min_score": 60,            # 提高买点门槛至 60分（仅做多周期共振票）
        "single_pos_scale": 0.60,   # 单笔建仓金额缩减至 60%（小仓位分散容错）
        "max_stop_pct": 0.06,
    },
    "CRASH": {
        "label": "🔴诱多末端/高位闪崩停买态",
        "max_exposure_pct": 0.00,   # 熊市反弹末端诱多或高位闪崩主杀区：禁止新开仓追高 (0%新开仓)
        "daily_buy_limit": 0,       # 禁止新开仓
        "min_score": 999,
        "single_pos_scale": 0.00,
        "max_stop_pct": 0.05,
    },
}

def classify_market_regime(bench_day_bars, day_snaps=None, evaluated=None, mode="dual_factor"):
    if not bench_day_bars or len(bench_day_bars) < 30:
        return "RANGE", dict(REGIME_CONFIGS["RANGE"]), {"regime": "RANGE", "reason": "样本不足默认震荡态"}

    dt = bench_day_bars[-1]["time"]
    closes = [b["close"] for b in bench_day_bars]
    c = closes[-1]
    ma10 = sum(closes[-10:]) / 10.0 if len(closes) >= 10 else c
    ma20 = sum(closes[-20:]) / 20.0
    ma60 = sum(closes[-60:]) / 60.0 if len(closes) >= 60 else ma20
    ma120 = sum(closes[-120:]) / 120.0 if len(closes) >= 120 else ma60
    ma20_prev5 = sum(closes[-25:-5]) / 20.0 if len(closes) >= 25 else ma20
    ma20_rising = ma20 >= ma20_prev5

    if dt in _BENCH_CACHE and "p_bench" in _BENCH_CACHE[dt]:
        p = _BENCH_CACHE[dt]["p_bench"]
    else:
        p = analyze_period(bench_day_bars[-250:])
        _BENCH_CACHE[dt] = {
            "p_bench": p, "c": c, "ma20": ma20, "ma60": ma60, "ma120": ma120,
            "ma20_rising": ma20_rising
        }

    zs = p.get("lastZs") if p else None
    bi = p.get("lastBi") if p else None
    div = p.get("divRatio") if p else None
    dif = p.get("dif", 0.0) if p else 0.0
    dea = p.get("dea", 0.0) if p else 0.0

    in_macro_bear = (c < ma60) and (ma60 < ma120)
    top_div = bool(bi and bi["type"] == "up" and div is not None and div < 0.75)

    up_ratio = 0.5
    high_score_cnt = 5
    if mode == "dual_factor" and day_snaps and evaluated is not None:
        valid_snaps = [s for s in day_snaps.values() if s["lastPrice"] > 0]
        if valid_snaps:
            up_ratio = sum(1 for s in valid_snaps if s["pctChange"] > 0) / len(valid_snaps)
        high_score_cnt = sum(1 for e in evaluated if e["score"] >= 70)

    if in_macro_bear and dif > dea and (bi and bi["type"] == "up"):
        if mode == "dual_factor" and up_ratio >= 0.65 and high_score_cnt >= 12:
            reg = "RANGE"
            reason = "熊市反弹但全场普涨放量突破，允许右侧轮动"
        else:
            reg = "CRASH"
            reason = f"沪深300处于空头熊市(c<{ma60:.0f}<{ma120:.0f})且至反弹笔末端诱多区，停止新开仓防套"
    elif in_macro_bear and (dif > dea or (bi and bi["type"] == "up") or not ma20_rising):
        reg = "WEAK"
        reason = "沪深300处于长期空头熊市弱势期，缩减单笔仓位至60%、每日限买1只>=60分标的"
    elif top_div and zs and c <= zs["zg"] and not (c > ma60 >= ma120):
        reg = "WEAK"
        reason = f"沪深300日线向上笔盘整顶背驰(面积比={div})且受阻于ZG({zs['zg']:.0f})，收缩仓位防守"
    elif (zs and c > zs["zg"]) or (c > ma60 and ma60 >= ma120):
        reg = "BULL"
        reason = "沪深300站上日线中枢ZG或均线多头排列，满额进攻"
    else:
        reg = "RANGE"
        reason = "沪深300处于结构震荡或超跌底背驰一买区，正常轮动"

    # 【新增：高位非背驰急跌保护 (High-Level Non-Divergent Plunge Protection)】
    # 当宏观均线尚在多头(ma60>=ma120)，但指数跌破MA20且MACD死叉、日线向下笔空头未背驰(div>=0.80)时：
    peak20 = max(closes[-20:])
    dd20 = (c - peak20) / peak20 * 100.0 if peak20 > 0 else 0.0
    peak20_bias60 = (peak20 - ma60) / ma60 * 100.0 if ma60 > 0 else 0.0
    max120 = max(closes[-120:]) if len(closes) >= 120 else peak20
    is_non_div_down_bi = bool(bi and bi["type"] == "down" and (div is None or div >= 0.80))

    if ma60 >= ma120 and c < ma20 and dif < dea and is_non_div_down_bi:
        # 阶段1(高位见顶闪崩初杀): 前20日高点创120日新高且偏离MA60>=4%，突然急跌超-4.8%
        stage1_flash = (peak20 >= max120 * 0.995) and (peak20_bias60 >= 4.0) and (dd20 <= -4.8)
        # 阶段2(高位破位主杀加速): 跌破MA60超0.5%、MA10<MA60、MA20下行，且日线向下笔空头力度放大>=1.8倍
        stage2_main = (c < ma60 * 0.995) and (ma10 < ma60) and (not ma20_rising) and (div is not None and div >= 1.8)
        if stage1_flash or stage2_main:
            prev_c = closes[-2] if len(closes) >= 2 else c
            if c >= prev_c or dd20 <= -5.2:
                reg = "CRASH"
                reason = f"触发高位非背驰急跌停买保护(dd20={dd20:.1f}%,向下笔力度比={div})"
            else:
                reg = "WEAK"
                reason = f"触发高位非背驰急跌降仓防守(dd20={dd20:.1f}%,向下笔力度比={div})"

    cfg = dict(REGIME_CONFIGS[reg])
    meta_info = {
        "regime": reg, "label": cfg["label"], "reason": reason,
        "hs300_close": c, "ma20": round(ma20, 2), "ma60": round(ma60, 2),
        "up_ratio": round(up_ratio, 3), "high_score_cnt": high_score_cnt
    }
    return reg, cfg, meta_info
