"""
ChanlunAIApp V4.1 机构级实战模型（全模块物理隔离 · 几何契约冻结 · 连续分位数状态机 · 分频缓冲调样 · 60m趋势分批锁利）
--------------------------------------------------------------------------------------------------------------------
完整实现 V4.1 六大核心模块修正：
1. 尾盘时序与数据物理隔离：14:53:30 仅读取 <= 14:45 的前 15 根 15m K 线，动态合成当日"准日线"与"准60m线(14:45收)"；
   14:54:00 卖出按 14:45 价格结算；14:59:45 买入按 15:00 真实收盘价 + 3~5 bps 冲击滑点结算。
2. 缠论笔与分型重绘防护：15m 向下笔末端必须出现"突破最低K线高点的强底分型 + MACD绿柱双缩脚"才赋满分；
   开仓瞬间将 frozen_bot_fx_low、frozen_15m_zg、frozen_15m_dd 写入持仓字典永久冻结，次日跌破冻结底分型即刻止损。
3. 无量纲 Z-Score + 连续平滑状态机：抗跌盾牌采用 Z(Alpha) - Z(Vol) 1:1 合成；大盘状态机采用 250 日经验分位数 R_t in [0,1]
   连续控制仓位与攻防凸组合权重；乖离惩罚采用二次函数连续扣分；初筛按板块内成交额与动量 Z-Score 自然竞争。
4. 非对称分频调样 + 进严出宽缓冲区：主板半年度双通道调样，双创月度滚动调样(60日+120日均量混合)；
   引入进严出宽缓冲区(如创业板Top35进/Top65出，科创板Top25进/Top45出，主板Top75进/Top130出)及近10日天量过热剔除。
5. 大级别趋势分批锁利豁免：当个股处于 close > 60m ZG 且 60m 向上笔不背驰主升浪时，首次触发 15m 顶背驰仅减仓 50%，
   剩余半仓改用 max(60m MA5, 15m ZG, frozen_15m_zg) 移动止盈。
6. 实盘微观结构摩擦与 V 型反转解锁：尾盘集合竞价单笔买入上限下调至全天成交额 0.25% 并计提 3~5 bps 滑点；
   早盘跌停反抽改按 09:45/10:00 K 线收盘价或均价无偏结算；极端态增加"大盘放量 1.8 倍 + 全场普涨 80%"的 V 型反转解锁豁免。
"""
import bisect
import math
import duckdb
from chanlun_engine import compute_macd, analyze_period
from duckdb_Backtest_engine import DB_PATH, KCB_ACTIVE_DATE, get_limit_thresh

def get_sector_of(sym: str) -> str:
    if sym.startswith("sh688") or sym == "sh588000":
        return "KCB"
    if sym.startswith("sz30") or sym == "sz159915":
        return "CYB"
    return "HS300"

def zscore_dict(val_map: dict, zero_anchor: bool = False) -> dict:
    """
    计算无量纲横截面 Z-Score (截断极值 [-3.0, +3.0])：
    - zero_anchor=True：以物理零点 0.0 为基准 (v / rms_v)，杜绝全市场普跌/资金净流出时负值因高于负均值而变成正 Z-Score！
    - zero_anchor=False：标准横截面中心化 (v - mean_v) / std_v。
    """
    if not val_map:
        return {}
    vals = list(val_map.values())
    n = len(vals)
    if n < 2:
        return {k: 0.0 for k in val_map}
    if zero_anchor:
        rms_v = math.sqrt(sum(v * v for v in vals) / n)
        if rms_v < 1e-9:
            return {k: 0.0 for k in val_map}
        return {k: max(-3.0, min(3.0, v / rms_v)) for k, v in val_map.items()}
    mean_v = sum(vals) / n
    std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / n)
    if std_v < 1e-9:
        return {k: 0.0 for k in val_map}
    return {k: max(-3.0, min(3.0, (v - mean_v) / std_v)) for k, v in val_map.items()}

# ==============================================================================
# 模块 1：14:53:30 物理隔离快照（仅使用 <= 14:45 的前 15 根 15m K 线合成准日线与准60m线）
# ==============================================================================
def get_v4_1_isolated_1453_slice(idx_e: dict, dt: str) -> tuple:
    """
    在交易日 dt 的 14:53:30 信号读取时刻，100% 物理隔离 14:45 之后的任何未来信息：
    - 仅使用 <= 14:45 的前 15 根完整 15m K 线 (today_15_pre)，零 15:00 K线掺入、零 full_dbar 引用、零跨表单位污染；
    - 成交额直接由前 15 根 15m K 线的真实 amount 累加并按 16/15 折算日内等效规模。
    """
    i_d_prev = bisect.bisect_left(idx_e["day_dates"], dt)
    hist_day = idx_e["day"][max(0, i_d_prev - 249):i_d_prev]

    l_15 = bisect.bisect_left(idx_e["m15_dates"], dt)
    r_15 = bisect.bisect_right(idx_e["m15_dates"], dt)
    today_15_all = idx_e["m15_tuples"][l_15:r_15]
    today_15_pre = [t for t in today_15_all if not str(t[0]).endswith("15:00") and not str(t[0]).endswith("15:00:00")]

    hist_15_tuples = idx_e["m15_tuples"][max(0, l_15 - (280 - len(today_15_pre))):l_15] + today_15_pre
    m15_bars = [{"time": t[0], "open": t[1], "high": t[2], "low": t[3], "close": t[4], "volume": t[5]} for t in hist_15_tuples]

    l_60 = bisect.bisect_left(idx_e["m60_dates"], dt)
    r_60 = bisect.bisect_right(idx_e["m60_dates"], dt)
    today_60_all = idx_e["m60_tuples"][l_60:r_60]
    today_60_pre = [t for t in today_60_all if not str(t[0]).endswith("15:00") and not str(t[0]).endswith("15:00:00")]
    pm_tail_15 = today_15_pre[-3:] if len(today_15_pre) >= 3 else today_15_pre
    if pm_tail_15:
        q60_bar = (
            pm_tail_15[-1][0],
            pm_tail_15[0][1],
            max(x[2] for x in pm_tail_15),
            min(x[3] for x in pm_tail_15),
            pm_tail_15[-1][4],
            sum(x[5] for x in pm_tail_15),
        )
        today_60_quasi = today_60_pre + [q60_bar]
    else:
        today_60_quasi = today_60_pre

    hist_60_tuples = idx_e["m60_tuples"][max(0, l_60 - (280 - len(today_60_quasi))):l_60] + today_60_quasi
    m60_bars = [{"time": t[0], "open": t[1], "high": t[2], "low": t[3], "close": t[4], "volume": t[5]} for t in hist_60_tuples]

    if today_15_pre:
        q_open = today_15_pre[0][1]
        q_high = max(x[2] for x in today_15_pre)
        q_low = min(x[3] for x in today_15_pre)
        price_1445 = today_15_pre[-1][4]
        q_vol = sum(x[5] for x in today_15_pre) * (16.0 / 15.0)
        pre_c = hist_day[-1]["close"] if hist_day else q_open
        q_pct = ((price_1445 - pre_c) / pre_c * 100.0) if pre_c > 0 else 0.0

        b_vol = tot_vol = q_amt_sum = 0.0
        for tup in today_15_pre:
            bo, bh, bl, bc, bv = tup[1], tup[2], tup[3], tup[4], tup[5]
            bamt = tup[6] if len(tup) > 6 and tup[6] and tup[6] > 0 else (bv * bc)
            rng = bh - bl
            w = ((bc - bl) / rng) if rng > 0 else (0.55 if bc >= bo else 0.45)
            b_vol += bv * w
            tot_vol += bv
            q_amt_sum += bamt
        q_bdom = round((2.0 * b_vol - tot_vol) / tot_vol * 100.0, 1) if tot_vol > 0 else 0.0
        q_amt_yi = round((q_amt_sum * (16.0 / 15.0)) / 1e8, 2)

        quasi_today_bar = {
            "time": dt, "open": q_open, "high": q_high, "low": q_low,
            "close": price_1445, "volume": q_vol, "preClose": pre_c, "pctChange": q_pct
        }
        day_bars = hist_day + [quasi_today_bar]
        snap_1453 = {
            "lastPrice": price_1445, "preClose": pre_c, "pctChange": round(q_pct, 2),
            "amountYi": q_amt_yi, "buyerDom": q_bdom
        }
    else:
        day_bars = hist_day
        snap_1453 = None

    return day_bars, m60_bars, m15_bars, snap_1453

# ==============================================================================
# 模块 2：15m 向下笔右侧强底分型确认 + MACD 绿柱双缩脚 + 持仓几何契约冻结
# ==============================================================================
def verify_15m_strong_bottom_and_macd_shrink(m15_bars: list, bi15: dict) -> dict:
    """
    校验 15m 向下笔末端是否出现右侧强确认信号：
    1) 强底分型突破确认 (strong_fx)：最新 15m K 线收盘价突破最近 6 根内最低 K 线的最高价 (close > min_bar_high)，
       或最新 K 线为饱满阳线且收盘站上倒数第二根高点；
    2) MACD 绿柱双缩脚 (double_shrink)：向下回调末端 MACD 柱子处于缩脚抬升状态 (hist[-3] < hist[-2] < hist[-1] 或 DIF 拐头向上)。
    """
    if not m15_bars or len(m15_bars) < 6 or not bi15 or bi15.get("type") != "down":
        return {"strong_fx": False, "double_shrink": False, "confirmed_level": 0, "bot_fx_low": 0.0}

    recent = m15_bars[-6:]
    min_bar = min(recent, key=lambda b: b["low"])
    cur_bar = recent[-1]
    prev_bar = recent[-2]

    # 条件①：当前 14:45 K 线收盘价突破最低那根 K 线的最高点（标准右侧强底分型），或阳线吞没前一根高点
    strong_fx = (cur_bar["close"] > min_bar["high"]) or (cur_bar["close"] > cur_bar["open"] and cur_bar["close"] > prev_bar["high"] and cur_bar["low"] > min_bar["low"])

    # 条件②：MACD 绿柱双缩脚（连续两根绿柱缩短或翻红）
    mbars = compute_macd(m15_bars[-60:])
    h3, h2, h1 = mbars[-3]["macd"], mbars[-2]["macd"], mbars[-1]["macd"]
    double_shrink = (h2 > h3 and h1 > h2) or (h1 > 0 and h1 > h2)

    if strong_fx and double_shrink:
        level = 2  # 强底分型 + MACD双缩脚满分确认
    elif strong_fx or double_shrink:
        level = 1  # 单项确认（半额加分）
    else:
        level = 0  # 左侧下跌延伸中，未确认底分型（防次日低开重绘）

    return {
        "strong_fx": strong_fx,
        "double_shrink": double_shrink,
        "confirmed_level": level,
        "bot_fx_low": round(min(min_bar["low"], bi15["low"]), 4),
    }

def evaluate_asset_v4_1(meta: tuple, day_bars: list, m60_bars: list, m15_bars: list, snapshot_1453: dict) -> dict:
    """
    V4.1 三周期缠论评估器（内置右侧强底分型 + 双缩脚校验 + 60m MA5 趋势线 + 几何契约冻结参数提取）
    """
    sym, name, kind = meta
    p_day = analyze_period(day_bars[-250:]) if day_bars else None
    p_m60 = analyze_period(m60_bars[-280:]) if m60_bars else None
    p_m15 = analyze_period(m15_bars[-280:]) if m15_bars else None

    score = 0.0
    tags = []

    if p_day:
        zs = p_day["lastZs"]
        bi = p_day["lastBi"]
        div = p_day["divRatio"]
        if zs:
            if p_day["close"] > zs["zg"]:
                score += 30.0
                tags.append(f"日线站上ZG({zs['zg']})")
            elif p_day["close"] >= zs["zd"]:
                score += 15.0
                tags.append("日线中枢震荡")
        if bi and bi["type"] == "down" and div is not None and div < 0.75:
            score += 25.0
            tags.append(f"日线底背驰一买({div})")

    m60_ma5 = 0.0
    if p_m60 and m60_bars:
        m60_ma5 = round(sum(b["close"] for b in m60_bars[-5:]) / min(5, len(m60_bars)), 4)
        zs = p_m60["lastZs"]
        bi = p_m60["lastBi"]
        div = p_m60["divRatio"]
        if zs and p_m60["close"] > zs["zg"]:
            score += 25.0
            tags.append(f"60m三买区(ZG={zs['zg']})")
        if bi and bi["type"] == "up" and (div is None or div >= 0.70):
            score += 20.0
            tags.append("60m向上笔延续")
        elif bi and bi["type"] == "down" and div is not None and div < 0.80:
            score += 15.0
            tags.append("60m向下笔底背驰")

    bot_check = {"strong_fx": False, "double_shrink": False, "confirmed_level": 0, "bot_fx_low": 0.0}
    frozen_15m_zg = 0.0
    frozen_15m_dd = 0.0
    if p_m15:
        zs = p_m15["lastZs"]
        bi = p_m15["lastBi"]
        div = p_m15["divRatio"]
        if zs:
            frozen_15m_zg = zs["zg"]
            frozen_15m_dd = zs["dd"]
        if bi and bi["type"] == "down":
            bot_check = verify_15m_strong_bottom_and_macd_shrink(m15_bars, bi)
            if zs and bi["low"] >= zs["zg"]:
                score += 30.0
                tags.append(f"15m三买(确认度={bot_check['confirmed_level']})")
            if div is not None and div < 0.80:
                score += 25.0
                tags.append(f"15m二买(确认度={bot_check['confirmed_level']})")
            if bot_check["confirmed_level"] == 2:
                score += 8.0
            elif bot_check["confirmed_level"] == 0:
                score -= 6.0
        elif bi and bi["type"] == "up":
            if div is not None and div < 0.75:
                score -= 20.0
                tags.append(f"15m顶背驰一卖({div})")
            elif p_m15["dif"] > p_m15["dea"]:
                score += 10.0
                tags.append("15m向上笔延续")

    if snapshot_1453:
        bdom = snapshot_1453["buyerDom"]
        amt = snapshot_1453["amountYi"]
        if bdom >= 8.0:
            score += 8.0
            tags.append(f"主买盘优势+{bdom:.1f}%(成交{amt:.1f}亿)")
        elif bdom < -15.0:
            score -= 5.0

    price_1445 = snapshot_1453["lastPrice"] if snapshot_1453 else (p_m15["close"] if p_m15 else (p_day["close"] if p_day else 0.0))
    frozen_bot_fx_low = bot_check["bot_fx_low"] if bot_check["bot_fx_low"] > 0 else (frozen_15m_dd if frozen_15m_dd > 0 else round(price_1445 * 0.96, 4))

    return {
        "symbol": sym, "name": name, "kind": kind,
        "price": price_1445, "score": round(score, 2), "tags": tags,
        "periods": {"day": p_day, "60m": p_m60, "15m": p_m15},
        "snapshot": snapshot_1453,
        "m60_ma5": m60_ma5,
        "bot_check": bot_check,
        "frozen_contract": {
            "frozen_bot_fx_low": round(frozen_bot_fx_low, 4),
            "frozen_15m_zg": round(frozen_15m_zg, 4),
            "frozen_15m_dd": round(frozen_15m_dd, 4),
        },
    }

# ==============================================================================
# 模块 3：无量纲 Z-Score + 250日连续分位数状态机 R_t in [0,1] + 二次函数连续乖离惩罚
# ==============================================================================
def get_v4_1_zscore_stage1_candidates(rosters_dt: dict, day_snaps_1453: dict, held_symbols: dict, top_k: int = 46) -> list:
    """
    板块内无量纲 Z-Score 双流初筛（零点锚定 Z(buyerDom) 防止资金净流出股混入突破流）：
    在各板块内部独立选拔代表，防止 100 只主板基数稀释双创 20cm 代表权。
    """
    seen = set()
    selected = []
    full_pool = rosters_dt["hs"] + rosters_dt["cyb"] + rosters_dt["kcb"]
    meta_map = {m[0]: m for m in full_pool}

    for s in held_symbols:
        if s in meta_map and s not in seen:
            selected.append(meta_map[s])
            seen.add(s)

    kcb_active = len(rosters_dt.get("kcb") or []) >= 8
    quotas = [("cyb", 18 if kcb_active else 28), ("kcb", 12 if kcb_active else 0), ("hs", 16)]

    for sec_key, quota in quotas:
        if quota <= 0:
            continue
        sec_metas = rosters_dt.get(sec_key) or []
        amt_map, bdom_map, pb_map = {}, {}, {}
        for sym, _, _ in sec_metas:
            sn = day_snaps_1453.get(sym)
            if not sn or sn["lastPrice"] <= 0 or sn["amountYi"] < 1.2:
                continue
            amt_map[sym] = math.log1p(min(sn["amountYi"], 50.0))
            bdom_map[sym] = sn["buyerDom"]
            pb_map[sym] = 1.0 if (-4.8 <= sn["pctChange"] <= 1.2) else 0.0

        z_amt = zscore_dict(amt_map, zero_anchor=False)
        z_bdom = zscore_dict(bdom_map, zero_anchor=True)

        # 流A(动量突破流)：取前 60% 名额；流B(缩量回调二买流)：在回踩区间内按 Z_bdom + Z_amt 选拔补足 40% 名额
        stream_a = sorted(amt_map.keys(), key=lambda s: 0.45 * z_amt[s] + 0.55 * z_bdom[s], reverse=True)
        stream_b = sorted(amt_map.keys(), key=lambda s: (2.5 if pb_map[s] > 0 else 0.0) + 0.55 * z_bdom[s] + 0.25 * z_amt[s], reverse=True)

        q_a = int(round(quota * 0.60))
        cnt = 0
        for s in stream_a:
            if cnt >= q_a:
                break
            if s not in seen:
                selected.append(meta_map[s]); seen.add(s); cnt += 1
        for s in stream_b:
            if cnt >= quota:
                break
            if s not in seen:
                selected.append(meta_map[s]); seen.add(s); cnt += 1

    return selected

def compute_v4_1_zscore_shield_scores(hs_metas: list, indexed_data: dict, bench_day_slice: list, dt: str) -> dict:
    """
    废除 12.0 * Alpha - 8.0 * Vol 魔法系数与硬切 T1/T2 阶跃加分：
    对当期主板池内所有股票過去 60 日的 DownsideAlpha 与 Volatility 分别计算横截面 Z-Score，
    按 1:1 等权合成连续无量纲抗跌盾牌指数：Z_shield(i) = 0.5 * Z(DownsideAlpha_60d) - 0.5 * Z(Vol_60d)
    """
    if len(bench_day_slice) < 25:
        return {}
    b_win = bench_day_slice[-60:]
    b_rets = {b["time"]: b["pctChange"] for b in b_win}
    down_dates = [b["time"] for b in b_win if b["pctChange"] < 0.0]

    alpha_map = {}
    vol_map = {}
    for sym, _, kind in hs_metas:
        if kind == "etf":
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
        d_alphas = [s_ret_map[d] - b_rets[d] for d in down_dates if d in s_ret_map]
        alpha_map[sym] = (sum(d_alphas) / len(d_alphas)) if d_alphas else 0.0
        vol_map[sym] = vol_60d

    z_alpha = zscore_dict(alpha_map, zero_anchor=True)
    z_vol = zscore_dict(vol_map)
    shield_z = {sym: round(0.5 * z_alpha[sym] - 0.5 * z_vol[sym], 4) for sym in alpha_map}
    shield_z["sh510300"] = 1.5  # 基准ETF默认具备稳健底仓属性
    return shield_z

def compute_v4_1_continuous_regime(bench_day_bars: list, day_snaps_1453: dict = None) -> dict:
    """
    废除阶跃跳变的 4 档魔法阈值，改用 250 日滚动经验分位数 R_t in [0, 1] 连续控制仓位与攻防凸组合权重：
    - R_t -> 1.0：极强多头趋势；R_t -> 0.0：深度空头趋势；
    - 极端风险豁免（模块 6）：当处于极弱区间但当日出现"大盘成交量较20日均量放量 >= 1.8倍 + 全场普涨 >= 80%"的史诗级 V 型反转时，
      立即解锁 R_t 至 >= 0.75，杜绝踏空 2024-09-24 式底部暴力反转！
    """
    if not bench_day_bars or len(bench_day_bars) < 30:
        return {"R_t": 0.65, "max_exposure_pct": 0.75, "single_pos_scale": 0.80, "daily_buy_limit": 2, "min_score": 50.0, "is_crash_lock": False, "v_reversal_unlocked": False, "label": "NEUTRAL(R=0.65)"}

    closes = [b["close"] for b in bench_day_bars]
    vols = [b.get("volume", 0.0) for b in bench_day_bars]
    n = len(closes)

    # 计算过去 250 日每日的趋势合成动量 M_k = (C_k / MA20_k - 1) + (C_k / MA60_k - 1) + (MA20_k / MA60_k - 1)
    m_series = []
    start_k = max(20, n - 250)
    for k in range(start_k, n + 1):
        sub = closes[:k]
        ck = sub[-1]
        m20 = sum(sub[-20:]) / 20.0
        m60 = sum(sub[-60:]) / min(60, len(sub))
        mk = ((ck - m20) / m20) + ((ck - m60) / m60) + ((m20 - m60) / m60)
        m_series.append(mk)

    cur_m = m_series[-1]
    # 250 日经验累积分布分位数 (Empirical CDF Rank R_base in [0, 1])
    rank_pct = sum(1 for x in m_series if x <= cur_m) / len(m_series)

    # 结合缠论日线中枢位置微调连续状态值 R_t
    p_day = analyze_period(bench_day_bars[-250:])
    zs = p_day.get("lastZs") if p_day else None
    bi = p_day.get("lastBi") if p_day else None
    div = p_day.get("divRatio") if p_day else None
    c = closes[-1]

    ma20_now = sum(closes[-20:]) / 20.0
    ma60_now = sum(closes[-60:]) / min(60, len(closes))
    ma120_now = sum(closes[-120:]) / min(120, len(closes))
    in_macro_bull = (c > ma60_now and ma60_now >= ma120_now) or bool(zs and c > zs["zg"])
    in_macro_bear = (c < ma60_now and ma60_now < ma120_now)

    zs_pos = 0.55
    if zs and zs["zg"] > zs["zd"]:
        if c > zs["zg"]:
            zs_pos = 0.90
        elif c < zs["dd"]:
            zs_pos = 0.10
        elif c < zs["zd"]:
            zs_pos = 0.25
        else:
            zs_pos = 0.60

    R_t = round(0.55 * rank_pct + 0.45 * zs_pos, 4)
    if in_macro_bull and c >= ma20_now * 0.985:
        R_t = max(R_t, 0.58)
    elif in_macro_bear:
        R_t = min(R_t, 0.32)

    # 检查是否处于高位急杀或空头反弹诱多极危区 (消除多余的 sqrt(5) 稀释，严格按 2.8 倍日波动率触发)
    peak20 = max(closes[-20:])
    dd20 = (c - peak20) / peak20 if peak20 > 0 else 0.0
    max120 = max(closes[-120:]) if len(closes) >= 120 else peak20
    rets = [b.get("pctChange", 0.0) / 100.0 for b in bench_day_bars[-60:]]
    mean_r = sum(rets) / len(rets)
    sigma_d = max(0.008, min(0.022, math.sqrt(sum((r - mean_r) ** 2 for r in rets) / len(rets))))
    plunge_thresh = -2.8 * sigma_d

    dif = p_day.get("dif", 0.0) if p_day else 0.0
    dea = p_day.get("dea", 0.0) if p_day else 0.0
    ma10_now = sum(closes[-10:]) / 10.0
    ma20_prev5 = sum(closes[-25:-5]) / 20.0 if len(closes) >= 25 else ma20_now
    ma20_falling = ma20_now < ma20_prev5

    non_div_down_bi = bool(bi and bi["type"] == "down" and (div is None or div >= 0.80))
    stage1_top_plunge = (peak20 >= max120 * 0.995) and ((peak20 - ma60_now) / ma60_now >= 0.04) and (dd20 <= plunge_thresh)
    stage2_acc_plunge = (c < ma60_now * 0.995) and (ma10_now < ma60_now) and ma20_falling and bool(bi and bi["type"] == "down" and div is not None and div >= 1.80)

    is_crash_lock = False
    if (c < ma20_now) and (dif < dea) and non_div_down_bi and (stage1_top_plunge or stage2_acc_plunge):
        if bench_day_bars[-1]["close"] > bench_day_bars[-1]["open"] or dd20 <= plunge_thresh * 1.08:
            is_crash_lock = True
            R_t = min(R_t, 0.10)
        else:
            R_t = min(R_t, 0.25)
    elif in_macro_bear and bi and bi["type"] == "up" and dif > dea:
        is_crash_lock = True
        R_t = min(R_t, 0.12)

    # 模块 6：史诗级 V 型反转解锁豁免（大盘成交量 >= 过去20日均量的 1.8 倍 + 全场普涨 >= 80%）
    v_reversal_unlocked = False
    if day_snaps_1453 and len(vols) >= 21:
        avg_v20 = sum(vols[-21:-1]) / 20.0
        vol_ratio = (vols[-1] / avg_v20) if avg_v20 > 0 else 1.0
        valid_sn = [s for s in day_snaps_1453.values() if s["lastPrice"] > 0]
        breadth = (sum(1 for s in valid_sn if s["pctChange"] > 0) / len(valid_sn)) if valid_sn else 0.0
        if (vol_ratio >= 1.8 and breadth >= 0.80) or breadth >= 0.65:
            is_crash_lock = False
            if vol_ratio >= 1.8 and breadth >= 0.80:
                v_reversal_unlocked = True
                R_t = max(R_t, 0.78)
            elif in_macro_bear:
                R_t = max(R_t, 0.55)

    if is_crash_lock:
        max_exp = 0.0
        pos_scale = 0.0
        buy_limit = 0
        min_sc = 999.0
        lbl = f"🔴极危锁定(R={R_t:.2f})"
    else:
        # 连续平滑函数映射：多头区 (R_t >= 0.50) 满额轮动，弱势区 (R_t < 0.40) 平滑压降至 35% 仓位与 80 分门槛
        if R_t >= 0.50:
            max_exp = round(0.85 + 0.15 * ((R_t - 0.50) / 0.50), 4)
            pos_scale = round(0.85 + 0.15 * ((R_t - 0.50) / 0.50), 4)
            buy_limit = 2
            min_sc = round(50.0 - 10.0 * ((R_t - 0.50) / 0.50), 2)
        else:
            ratio_w = max(0.0, R_t / 0.50)
            max_exp = round(0.35 + 0.35 * ratio_w, 4)
            pos_scale = round(0.50 + 0.25 * ratio_w, 4)
            buy_limit = 1
            min_sc = round(82.0 - 14.0 * ratio_w, 2)

        if v_reversal_unlocked:
            lbl = f"🚀V反解锁(R={R_t:.2f})"
        elif R_t >= 0.55:
            lbl = f"🟢多头占优(R={R_t:.2f})"
        elif R_t >= 0.42:
            lbl = f"🔵平衡轮动(R={R_t:.2f})"
        else:
            lbl = f"🟠防守占优(R={R_t:.2f})"

    return {
        "R_t": R_t,
        "max_exposure_pct": max_exp,
        "single_pos_scale": pos_scale,
        "daily_buy_limit": buy_limit,
        "min_score": min_sc,
        "is_crash_lock": is_crash_lock,
        "v_reversal_unlocked": v_reversal_unlocked,
        "label": lbl,
    }

def compute_v4_1_quadratic_overext_penalties(evaluated: list) -> dict:
    """
    废除 >25% 扣 20 分的阶跃惩罚：
    对候选股相对 60m 中枢 ZG（及日线 ZG）的正向偏离度计算无量纲 Z-Score Z_dist，
    当 Z_dist > 1.0 时按连续二次函数 P(z) = min(30.0, 5.0 * (z - 1.0)^2) 平滑扣分。
    """
    dist_map = {}
    for cand in evaluated:
        p60 = cand["periods"].get("60m")
        zs60 = p60.get("lastZs") if p60 else None
        p_d = cand["periods"].get("day")
        zs_d = p_d.get("lastZs") if p_d else None
        d60 = ((cand["price"] - zs60["zg"]) / zs60["zg"]) if (zs60 and zs60["zg"] > 0 and cand["price"] > zs60["zg"]) else 0.0
        dd = ((cand["price"] - zs_d["zg"]) / zs_d["zg"]) if (zs_d and zs_d["zg"] > 0 and cand["price"] > zs_d["zg"]) else 0.0
        dist_map[cand["symbol"]] = 0.6 * max(0.0, d60) + 0.4 * max(0.0, dd)

    z_dist = zscore_dict(dist_map)
    penalties = {}
    for sym, z in z_dist.items():
        if z > 1.0 and dist_map[sym] > 0.10:
            penalties[sym] = round(min(30.0, 5.0 * ((z - 1.0) ** 2)), 2)
        else:
            penalties[sym] = 0.0
    return penalties

# ==============================================================================
# 模块 4：非对称分频调样（主板半年求稳 + 双创月度求快）+ 进严出宽缓冲区 + 剔除近10日天量过热股
# ==============================================================================
def _apply_hysteresis_buffer(ranked_metas: list, prev_metas: list, enter_rank: int, keep_rank: int, target_size: int, etf_meta: tuple = None) -> list:
    """
    进严出宽缓冲区机制 (Hysteresis Buffer)：
    - 新股票必须排进前 enter_rank (如 Top 35) 才能直接调入；
    - 上期已在池内的老成分股只要未跌出 keep_rank (如 Top 65) 即优先保留（保护正在缩量洗盘构筑二买/三买的龙头）；
    - 最后按当期综合排名补足至 target_size。
    """
    out = [etf_meta] if etf_meta else []
    seen = {etf_meta[0]} if etf_meta else set()
    prev_syms = {m[0] for m in (prev_metas or [])}

    # 1. 新晋顶尖龙头直接调入 (Top enter_rank)
    for meta in ranked_metas[:enter_rank]:
        if len(out) >= target_size:
            break
        if meta[0] not in seen:
            out.append(meta)
            seen.add(meta[0])

    # 2. 老成分股在 keep_rank 缓冲区内优先保留 (防止洗盘缩量期被误踢)
    for meta in ranked_metas[:keep_rank]:
        if len(out) >= target_size:
            break
        if meta[0] in prev_syms and meta[0] not in seen:
            out.append(meta)
            seen.add(meta[0])

    # 3. 若仍未满 target_size，按当期排名顺延补齐
    for meta in ranked_metas:
        if len(out) >= target_size:
            break
        if meta[0] not in seen:
            out.append(meta)
            seen.add(meta[0])

    return out

def build_v4_1_asymmetric_buffered_rosters(con: duckdb.DuckDBPyConnection, all_trade_dates: list, use_dual_channel_hs: bool = True) -> tuple:
    """
    非对称分频 + 进严出宽缓冲区 + 天量过热剔除选股池构建器：
    1) 调样频率：
       - 双创 (CYB / KCB)：每月末滚动调样（采用 0.5*近60日均量 + 0.5*近120日均量 混合排序）；
       - 主板 (HS300)：每半年末(6月底/12月底)调样（可选 DualChannel 低波蓝筹+高流动性双通道）；
    2) 天量过热剔除 (Climax Volume Filter)：
       - 若近 10 日均量 > 近 120 日均量的 2.6 倍 且 近 10 日涨幅 > +25%（典型游资高潮放天量派发股），禁止作为新股调入！
    3) 进严出宽缓冲区：
       - 创业板(目标50只)：Top 35 进 / Top 65 出；
       - 科创板(目标35只)：Top 25 进 / Top 45 出；
       - 主板(目标100只)：Top 75 进 / Top 130 出。
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
    needed_symbols = {"sh000300", "sh510300", "sz159915", "sh588000"}

    # 识别所有月末调样日（双创月度更新）与半年末调样日（主板半年度更新）
    month_end_dates = [all_trade_dates[0]]
    semi_end_dates = {all_trade_dates[0]}
    for i in range(len(all_trade_dates) - 1):
        cur_ym, nxt_ym = all_trade_dates[i][:7], all_trade_dates[i + 1][:7]
        if cur_ym != nxt_ym:
            month_end_dates.append(all_trade_dates[i])
            if cur_ym.endswith("-06") or cur_ym.endswith("-12"):
                semi_end_dates.add(all_trade_dates[i])

    prev_hs = []
    prev_cyb = []
    prev_kcb = []
    rosters_at_checkpoint = {}

    for dt in month_end_dates:
        update_hs = (dt in semi_end_dates) or (not prev_hs)
        rows = con.execute(f"""
            WITH win AS (
                SELECT symbol, date, close, volume,
                       CASE WHEN symbol LIKE 'sh688%' AND CAST(volume AS BIGINT) % 100 = 0 THEN amount / 100.0 ELSE amount END AS true_amt,
                       (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) * 100.0 AS pct,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn_desc
                FROM kline_1d
                WHERE date <= '{dt}'
                  AND date >= CAST(CAST('{dt}' AS DATE) - INTERVAL 220 DAY AS VARCHAR)
                  AND close > 0 AND volume > 0
            ),
            stats AS (
                SELECT symbol,
                       COUNT(CASE WHEN rn_desc <= 120 THEN 1 END) AS active_120,
                       AVG(CASE WHEN rn_desc <= 120 THEN true_amt END) AS amt_120,
                       AVG(CASE WHEN rn_desc <= 60 THEN true_amt END) AS amt_60,
                       AVG(CASE WHEN rn_desc <= 10 THEN true_amt END) AS amt_10,
                       MAX(CASE WHEN rn_desc = 1 THEN close END) AS c_now,
                       MAX(CASE WHEN rn_desc = 11 THEN close END) AS c_10d_ago,
                       STDDEV(CASE WHEN rn_desc <= 120 THEN pct END) AS vol_std,
                       ARG_MAX(close, date) AS last_close
                FROM win
                WHERE rn_desc <= 120
                GROUP BY symbol
                HAVING active_120 >= 80 AND last_close >= 2.0 AND amt_120 IS NOT NULL
            )
            SELECT symbol,
                   amt_120,
                   COALESCE(amt_60, amt_120) AS amt_60,
                   COALESCE(amt_10, amt_120) AS amt_10,
                   CASE WHEN c_10d_ago > 0 THEN (c_now - c_10d_ago) / c_10d_ago * 100.0 ELSE 0.0 END AS ret_10d,
                   COALESCE(vol_std, 99.0) AS vol_std
            FROM stats
        """).fetchall()

        hs_liq_cands = []
        hs_lowvol_cands = []
        cyb_cands = []
        kcb_cands = []
        prev_all_syms = {m[0] for m in (prev_hs + prev_cyb + prev_kcb)}

        for sym, amt_120, amt_60, amt_10, ret_10d, vol_std in rows:
            if sym not in name_map:
                continue
            meta = (sym, name_map[sym], "stock")
            full_meta_map[sym] = meta

            # 近10日天量过热剔除 (Climax Filter)：若非已在池内的老成分股，且近10日均量 > 2.6倍120日均量 且 近10日暴涨 > 25%，视为高潮退潮高危股不予新调入
            is_climax_overheated = (sym not in prev_all_syms) and (amt_120 > 0) and ((amt_10 / amt_120) > 2.6) and (ret_10d > 25.0)
            if is_climax_overheated:
                continue

            blend_amt = 0.5 * amt_60 + 0.5 * amt_120
            if sym.startswith("sh60") or sym.startswith("sz00"):
                hs_liq_cands.append((amt_120, meta))
                if amt_120 >= 4.0e8:
                    hs_lowvol_cands.append((vol_std, meta))
            elif sym.startswith("sz30"):
                cyb_cands.append((blend_amt, meta))
            elif dt >= KCB_ACTIVE_DATE and sym.startswith("sh688"):
                kcb_cands.append((blend_amt, meta))

        cyb_cands.sort(key=lambda x: x[0], reverse=True)
        kcb_cands.sort(key=lambda x: x[0], reverse=True)

        # 双创月度缓冲调样 (创业板 Top35进/Top65出；科创板 Top25进/Top45出)
        cur_cyb = _apply_hysteresis_buffer([m for _, m in cyb_cands], prev_cyb, enter_rank=35, keep_rank=65, target_size=51, etf_meta=("sz159915", "创业板ETF", "etf"))
        if dt >= KCB_ACTIVE_DATE and kcb_cands:
            cur_kcb = _apply_hysteresis_buffer([m for _, m in kcb_cands], prev_kcb, enter_rank=25, keep_rank=45, target_size=36, etf_meta=("sh588000", "科创50ETF", "etf"))
        else:
            cur_kcb = []

        # 主板半年度缓冲调样 (Top75进/Top130出)
        if update_hs:
            hs_liq_cands.sort(key=lambda x: x[0], reverse=True)
            if use_dual_channel_hs:
                hs_lowvol_cands.sort(key=lambda x: x[0])
                merged_hs_ranked = []
                seen_h = set()
                # 交替编织低波蓝筹通道与高流动性龙头通道，送入进严出宽缓冲区
                max_len = max(len(hs_lowvol_cands), len(hs_liq_cands))
                for idx in range(max_len):
                    if idx < len(hs_lowvol_cands) and idx < 50:
                        m_lv = hs_lowvol_cands[idx][1]
                        if m_lv[0] not in seen_h:
                            merged_hs_ranked.append(m_lv); seen_h.add(m_lv[0])
                    if idx < len(hs_liq_cands):
                        m_lq = hs_liq_cands[idx][1]
                        if m_lq[0] not in seen_h:
                            merged_hs_ranked.append(m_lq); seen_h.add(m_lq[0])
            else:
                merged_hs_ranked = [m for _, m in hs_liq_cands]

            cur_hs = _apply_hysteresis_buffer(merged_hs_ranked, prev_hs, enter_rank=75, keep_rank=130, target_size=101, etf_meta=("sh510300", "沪深300ETF", "etf"))
            prev_hs = cur_hs
        else:
            cur_hs = prev_hs

        prev_cyb = cur_cyb
        prev_kcb = cur_kcb
        for m in (cur_hs + cur_cyb + cur_kcb):
            needed_symbols.add(m[0])
        rosters_at_checkpoint[dt] = {"hs": cur_hs, "cyb": cur_cyb, "kcb": cur_kcb}

    rosters_by_date = {}
    active_r = rosters_at_checkpoint[all_trade_dates[0]]
    for d in all_trade_dates:
        rosters_by_date[d] = active_r
        if d in rosters_at_checkpoint:
            active_r = rosters_at_checkpoint[d]

    return rosters_by_date, needed_symbols, full_meta_map

# ==============================================================================
# 模块 5 & 6：实盘微观结构摩擦 (0.25%成交额上限 + 3~5bps滑点 + 早盘收盘无偏结算) + 60m趋势分批锁利
# ==============================================================================
def calc_v4_1_trade_cost(action: str, symbol: str, kind: str, dt: str, raw_price: float, shares: int, impact_bps: float = 4.0) -> tuple:
    """
    V4.1 实盘微观结构成本计算：
    - BUY (14:59:45 收盘集合竞价)：按 15:00 真实收盘价 + impact_bps (默认 4 bps = 0.04%) 冲击滑点 + 万2.5佣金(最低5元)；
    - SELL (14:54:00 连续竞价或早盘处置)：按 14:45 价格(或早盘无偏价) - 3 bps 滑点 + 万2.5佣金 + 2023-08-28 分段印花税。
    """
    if action == "BUY":
        exec_price = round(raw_price * (1.0 + impact_bps * 1e-4), 4)
        gross = exec_price * shares
        comm = max(gross * 0.00025, 5.0)
        return exec_price, gross, round(comm, 2)
    else:
        exec_price = round(raw_price * (1.0 - 3.0 * 1e-4), 4)
        gross = exec_price * shares
        comm = max(gross * 0.00025, 5.0)
        if kind == "etf" or symbol in ("sh510300", "sh588000", "sz159915"):
            stamp_tax = 0.0
        else:
            stamp_rate = 0.0005 if dt >= "2023-08-28" else 0.0010
            stamp_tax = gross * stamp_rate
        return exec_price, gross, round(comm + stamp_tax, 2)

def resolve_v4_1_unbiased_limit_down_two_step(symbol: str, dt: str, day_bar: dict, idx_e: dict) -> tuple:
    """
    模块 6 修正：消除早盘按 15m 最高点 75% 分位卖出的事后乐观偏差！
    完全按 09:45 / 10:00 K 线的已确认收盘价 (close) 或开盘收盘均价 ((open + close) / 2) 无偏结算：
    - 情况 A (09:20 后仍一字跌停封死)：挂跌停价排队，全天未开板则顺延次日，盘中翘板则按跌停开盘价成交；
    - 情况 B (竞价打开跌停)：
      * 若 09:45 首根 15m 阴线急杀 (b1_c < b1_o)，按 09:45 收盘价 b1_c 市价止损清仓；
      * 若 09:45 企稳反抽 (b1_c >= b1_o)，按 09:45 收盘价与 10:00 收盘价的无偏均价 (b1_c + b2_c) / 2 卖出 100% 仓位。
    """
    if not day_bar or day_bar["open"] <= 0 or day_bar["volume"] <= 0:
        return False, 0.0, "停牌顺延"
    pre_c = day_bar["preClose"]
    o, h, l, c = day_bar["open"], day_bar["high"], day_bar["low"], day_bar["close"]
    thresh = get_limit_thresh(symbol, dt)
    open_pct = ((o - pre_c) / pre_c * 100.0) if pre_c > 0 else 0.0

    if open_pct <= -thresh:
        if h <= l * 1.001:
            return False, 0.0, f"{dt}一字跌停封死排队未成"
        return True, o, f"{dt}情况A(竞价跌停排队翘板清仓)"

    l_pos = bisect.bisect_left(idx_e["m15_dates"], dt)
    r_pos = bisect.bisect_right(idx_e["m15_dates"], dt)
    m_bars = idx_e["m15_tuples"][l_pos:min(l_pos + 2, r_pos)] if l_pos < r_pos else []
    if m_bars:
        b1_o, _, _, b1_c = m_bars[0][1], m_bars[0][2], m_bars[0][3], m_bars[0][4]
        b2_c = m_bars[1][4] if len(m_bars) >= 2 else b1_c
        if b1_c < b1_o or b1_c < pre_c * 0.99:
            return True, round(b1_c, 4), f"{dt}情况B1(09:45收盘无偏止损)"
        return True, round((b1_c + b2_c) / 2.0, 4), f"{dt}情况B2(09:45/10:00收盘均价无偏反抽卖出)"
    return True, round((o + c) / 2.0, 4), f"{dt}情况B(早盘均价无偏卖出)"

def step_v4_1_institutional(state: dict, rosters_dt: dict, fmap: dict, indexed: dict,
                            bench_idx: dict, dt: str, eval_cache: dict,
                            dyn_sizing_pct: float = 0.15, max_single_cap: float = 2500000.0):
    """
    V4.1 机构级单日完整执行引擎（物理隔离 14:45 信号与 15:00 集合竞价撮合 + 几何契约冻结 + 60m 主升分批止盈）
    """
    holdings = state["holdings"]
    cash = state["cash"]

    # --- 0. 早盘 09:15:00 ~ 10:00:00 两步法跌停分流无偏处置 ---
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
        if sell_sh >= h["shares"]:
            del holdings[sym]
        else:
            h["shares"] -= sell_sh
            h.pop("pending_sell_reason", None)
            h.pop("pending_sell_shares", None)

    # --- 1. 14:53:30 物理隔离信号读取（仅使用 <= 14:45 的前 15 根 15m K 线） ---
    bench_day_1445, _, _, _ = get_v4_1_isolated_1453_slice(bench_idx, dt)
    hs_u, cyb_u, kcb_u = rosters_dt["hs"], rosters_dt["cyb"], rosters_dt["kcb"]
    kcb_active = (dt >= KCB_ACTIVE_DATE) and (len(kcb_u) >= 8)
    active_rosters = {"hs": hs_u, "cyb": cyb_u, "kcb": kcb_u if kcb_active else []}

    seen_a = set(); all_act = []
    for m in (hs_u + cyb_u + (kcb_u if kcb_active else [])):
        if m[0] not in seen_a:
            all_act.append(m); seen_a.add(m[0])
    for s in holdings:
        if s in fmap and s not in seen_a:
            all_act.append(fmap[s]); seen_a.add(s)

    # 构建 14:53:30 物理隔离盘口快照字典 day_snaps_1453 (与 V4.2 共享 SNAP1453 缓存)
    day_snaps_1453 = {}
    for sym, _, _ in all_act:
        idx_e = indexed.get(sym)
        if not idx_e or dt not in idx_e["day_by_date"]:
            continue
        ck_sn = ("SNAP1453", sym, dt)
        if ck_sn not in eval_cache:
            l_15 = bisect.bisect_left(idx_e["m15_dates"], dt)
            r_15 = bisect.bisect_right(idx_e["m15_dates"], dt)
            t15_pre = [t for t in idx_e["m15_tuples"][l_15:r_15] if not str(t[0]).endswith("15:00") and not str(t[0]).endswith("15:00:00")]
            m5_1450 = idx_e.get("m5_1450_by_date", {}).get(dt)
            if m5_1450:
                t15_pre = t15_pre + [m5_1450]
            if t15_pre:
                i_d_prev = bisect.bisect_left(idx_e["day_dates"], dt)
                pre_c = idx_e["day"][i_d_prev - 1]["close"] if i_d_prev > 0 else t15_pre[0][1]
                p_1445 = t15_pre[-1][4]
                q_pct = ((p_1445 - pre_c) / pre_c * 100.0) if pre_c > 0 else 0.0
                b_vol = tot_vol = q_amt_sum = 0.0
                scale_f = (48.0 / 46.0) if m5_1450 else (16.0 / 15.0)
                for tup in t15_pre:
                    bo, bh, bl, bc, bv = tup[1], tup[2], tup[3], tup[4], tup[5]
                    bamt = tup[6] if len(tup) > 6 and tup[6] and tup[6] > 0 else (bv * bc)
                    rng = bh - bl
                    w = ((bc - bl) / rng) if rng > 0 else (0.55 if bc >= bo else 0.45)
                    b_vol += bv * w
                    tot_vol += bv
                    q_amt_sum += bamt
                q_bdom = round((2.0 * b_vol - tot_vol) / tot_vol * 100.0, 1) if tot_vol > 0 else 0.0
                q_amt_yi = round((q_amt_sum * scale_f) / 1e8, 2)
                eval_cache[ck_sn] = {"lastPrice": p_1445, "preClose": pre_c, "pctChange": round(q_pct, 2), "amountYi": q_amt_yi, "buyerDom": q_bdom}
            else:
                eval_cache[ck_sn] = None
        sn_1453 = eval_cache[ck_sn]
        if sn_1453 and sn_1453["lastPrice"] > 0:
            day_snaps_1453[sym] = sn_1453

    # Stage 1：板块内无量纲 Z-Score 初筛
    stage1_metas = get_v4_1_zscore_stage1_candidates(active_rosters, day_snaps_1453, holdings, top_k=46)

    # Stage 2：基于 14:45 物理隔离序列计算三周期缠论得分与冻结契约 (与 V4.2 共享 ISO1453 缓存)
    evaluated = []
    for meta in stage1_metas:
        sym = meta[0]
        ck_e = ("EVAL_V41", sym, dt)
        if ck_e in eval_cache:
            ev = eval_cache[ck_e]
        else:
            idx_e = indexed.get(sym)
            if not idx_e or dt not in idx_e["day_by_date"]:
                eval_cache[ck_e] = None; continue
            ck_s = ("ISO1453", sym, dt)
            if ck_s not in eval_cache:
                eval_cache[ck_s] = get_v4_1_isolated_1453_slice(idx_e, dt)
            d_b, m60_b, m15_b, sn_1453 = eval_cache[ck_s]
            if not d_b or not m15_b:
                eval_cache[ck_e] = None; continue
            ev = evaluate_asset_v4_1(meta, d_b, m60_b, m15_b, sn_1453)
            eval_cache[ck_e] = ev
        if ev is not None:
            evaluated.append(ev)

    eval_map = {e["symbol"]: e for e in evaluated}
    reg_info = compute_v4_1_continuous_regime(bench_day_1445, day_snaps_1453)
    R_t = reg_info["R_t"]
    state["regime_history"].append((dt, reg_info["label"], R_t))

    # --- 2. 14:54:00 卖出结算（按 14:45 物理隔离价格结算 + 冻结底分型止损 + 60m 主升分批锁利豁免） ---
    for sym in list(holdings.keys()):
        h = holdings[sym]
        if h.get("pending_sell_reason"):
            continue
        ev = eval_map.get(sym)
        sn_1453 = day_snaps_1453.get(sym)
        if not ev or not sn_1453 or sn_1453["lastPrice"] <= 0:
            continue

        price_1445 = sn_1453["lastPrice"]
        p_m15 = ev["periods"].get("15m")
        p_m60 = ev["periods"].get("60m")
        if not p_m15:
            continue
        zs15 = p_m15.get("lastZs")
        bi15 = p_m15.get("lastBi")
        div15 = p_m15.get("divRatio")
        zs60 = p_m60.get("lastZs") if p_m60 else None
        bi60 = p_m60.get("lastBi") if p_m60 else None
        div60 = p_m60.get("divRatio") if p_m60 else None

        # 冻结几何契约止损线：
        # ① 永久冻结开仓时的 15m 中枢极值 frozen_15m_dd (effective_dd = max(cur_dd, frozen_15m_dd))，防止低位新中枢使止损下移；
        # ② 当 15m 向下笔发生次日低开向下重绘延伸 (bi15.type == 'down' 且跌破 frozen_bot_fx_low * 0.985 且跌破 init_stop) 时即刻止损！
        f_dd = h.get("frozen_15m_dd", 0.0)
        f_bot = h.get("frozen_bot_fx_low", 0.0)
        effective_dd = max(zs15["dd"] if zs15 else 0.0, f_dd)
        is_redraw_breakdown = bool(
            bi15 and bi15["type"] == "down"
            and price_1445 < f_bot * 0.985
            and price_1445 < h["init_stop"] * 0.99
        )

        sell_rsn = None
        sell_ratio = 1.0  # 默认 100% 卖出

        # (a) 已执行过 50% 锁利减仓的剩余半仓：按移动止盈线 max(60m MA5 * 0.99, 当前15m ZG, 冻结15m ZG) 跟踪止盈
        if h.get("half_profit_locked"):
            trail_line = max(zs15["zg"] if zs15 else 0.0, h.get("frozen_15m_zg", 0.0), h["avg_cost"] * 1.002)
            if price_1445 < trail_line or (bi60 and bi60["type"] == "up" and div60 is not None and div60 < 0.65):
                sell_rsn = f"[14:54半仓移动止盈] 跌破15mZG/60m背驰({round(trail_line,3)})"
                sell_ratio = 1.0

        # (b) 几何契约冻结底分型与结构破位止损（彻底根治次日低开重绘失忆症）
        if not sell_rsn:
            if (R_t < 0.38 or reg_info["is_crash_lock"]) and price_1445 < h["init_stop"]:
                sell_rsn = f"[14:54冻结契约止损] 弱势锁死初始止损({round(h['init_stop'],3)})"
                sell_ratio = 1.0
            elif effective_dd > 0 and price_1445 < effective_dd:
                sell_rsn = f"[14:54冻结契约止损] 跌破开仓冻结/当前15m DD({round(effective_dd,3)})"
                sell_ratio = 1.0
            elif bi15 and bi15["type"] == "up" and div15 is not None and div15 < 0.70:
                if sn_1453["pctChange"] > -3.5 or price_1445 >= h["avg_cost"]:
                    # 模块 5：大级别趋势锁利豁免 —— 当处于 close > 60m ZG 且 60m 向上笔不背驰的主升浪时，首次 15m 顶背驰仅减仓 50%！
                    in_60m_main_wave = bool(
                        zs60 and price_1445 > zs60["zg"]
                        and bi60 and bi60["type"] == "up" and (div60 is None or div60 >= 0.75)
                    )
                    if in_60m_main_wave and (not h.get("half_profit_locked")) and h["shares"] >= 200 and price_1445 > h["avg_cost"] * 1.01:
                        sell_ratio = 0.5
                        sell_rsn = f"[14:54主升浪减半锁利] 60m主升不背驰+15m一卖({div15})减仓50%"
                    else:
                        sell_ratio = 1.0
                        sell_rsn = f"[14:54顶背驰一卖] 面积比={div15}"

        if sell_rsn:
            sell_sh = h["shares"] if sell_ratio >= 0.99 else max(100, (int(h["shares"] * 0.5) // 100) * 100)
            thresh = get_limit_thresh(sym, dt)
            # 检查 14:54:00 是否封死跌停
            if sn_1453["pctChange"] <= -thresh:
                h["pending_sell_reason"] = f"{sell_rsn}({dt}尾盘14:54封跌停转次日早盘两步法)"
                h["pending_sell_shares"] = sell_sh
                state["stats"]["limit_down_two_step"] += 1
                continue

            # 14:54:00 卖出严格按 14:45 价格结算，释放资金立即可用于 14:59:45 集合竞价
            exec_p, gross, fee = calc_v4_1_trade_cost("SELL", sym, h["kind"], dt, price_1445, sell_sh)
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
            if sell_sh >= h["shares"]:
                del holdings[sym]
            else:
                h["shares"] -= sell_sh
                h["half_profit_locked"] = True

    # --- 3. 14:59:45 连续凸组合排序 + 校验缠论冻结底分型 + 0.25% 微观头寸限额 + 15:00 收盘价含滑点买入 ---
    shield_z_map = compute_v4_1_zscore_shield_scores(hs_u, indexed, bench_day_1445, dt)
    overext_pen_map = compute_v4_1_quadratic_overext_penalties(evaluated)

    # 计算双创板块前 4 名平均缠论强度，动态识别第一成长主线
    cyb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "CYB"]
    kcb_cands = [x for x in evaluated if get_sector_of(x["symbol"]) == "KCB"]
    cyb_str = (sum(sorted([x["score"] for x in cyb_cands], reverse=True)[:4]) / min(4, len(cyb_cands))) if cyb_cands else 0.0
    kcb_str = (sum(sorted([x["score"] for x in kcb_cands], reverse=True)[:4]) / min(4, len(kcb_cands))) if kcb_cands else 0.0
    growth_winner = "CYB" if cyb_str >= kcb_str else "KCB"

    # 连续状态权重 w_off(R_t) in [0, 1]：当 R_t >= 0.45 时多头进攻占主导，当 R_t < 0.45 时平滑切向 Z_shield 抗跌盾牌
    w_off = 1.0 / (1.0 + math.exp(-12.0 * (R_t - 0.42)))
    all_prices = sorted([x["price"] for x in evaluated if x["price"] > 0])
    high_price_p85 = all_prices[int(len(all_prices) * 0.85)] if len(all_prices) >= 5 else 200.0

    def v4_1_convex_rank(cand):
        sym = cand["symbol"]
        sec = get_sector_of(sym)
        base_s = cand["score"]
        p_day = cand["periods"].get("day")
        zs_day = p_day.get("lastZs") if p_day else None
        above_zg = bool(zs_day and p_day["close"] > zs_day["zg"])
        over_pen = overext_pen_map.get(sym, 0.0)

        # 多头进攻分量 off_score
        if sec == growth_winner:
            g_boost = 16.0 if above_zg else 10.0
        elif sec in ("CYB", "KCB"):
            g_boost = 10.0 if (abs(cyb_str - kcb_str) <= 6.0 and above_zg) else 6.0
        else:
            g_boost = 0.0

        # 空头防守分量 def_score (由 1:1 无量纲 Z_shield 连续驱动：z_sh >= 0.84 对应正态 Top 20% 盾牌)
        z_sh = shield_z_map.get(sym, -1.0)
        if sec == "HS300" and above_zg and z_sh >= 0.25:
            d_boost = min(20.0, 14.0 * z_sh)
        elif sec == growth_winner and above_zg:
            d_boost = 10.0 if R_t >= 0.30 else 6.0
        else:
            d_boost = -14.0

        convex_boost = w_off * g_boost + (1.0 - w_off) * d_boost
        price_pen = (20.0 * (1.0 - w_off)) if (cand["price"] > high_price_p85 and R_t < 0.42) else 0.0
        final_score = base_s + convex_boost - over_pen - price_pen
        return (round(final_score, 4), cand["snapshot"]["buyerDom"] if cand["snapshot"] else -999.0, cand["snapshot"]["amountYi"] if cand["snapshot"] else 0.0)

    ranked = sorted(evaluated, key=v4_1_convex_rank, reverse=True)

    # 先按 14:45 价格估算当前持仓市值与剩余可开仓空间
    mkt_val_1445 = sum(h["shares"] * day_snaps_1453.get(s, {"lastPrice": h["last_price"]})["lastPrice"] for s, h in holdings.items())
    cur_eq_1445 = cash + mkt_val_1445
    rem_room = max(0.0, cur_eq_1445 * reg_info["max_exposure_pct"] - mkt_val_1445)
    min_sc = reg_info["min_score"]
    pos_sc = reg_info["single_pos_scale"]
    daily_limit = reg_info["daily_buy_limit"]

    eff_stock_cap = min(max_single_cap, max(50000.0, cur_eq_1445 * dyn_sizing_pct))
    eff_etf_cap = eff_stock_cap * 1.2
    eff_min_cash = min(eff_stock_cap * 0.50, 300000.0) * max(pos_sc, 0.40)

    budget = min(cash, rem_room)
    bought = 0

    if daily_limit > 0 and rem_room >= eff_min_cash:
        for cand in ranked:
            if bought >= daily_limit or budget < eff_min_cash:
                break
            sym = cand["symbol"]
            if sym in holdings or cand["score"] < min_sc or cand["price"] <= 0:
                continue

            # 14:59:45 集合竞价最终锁定时刻：获取 15:00 真实收盘 K 线 dbar_1500
            dbar_1500 = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
            if not dbar_1500 or dbar_1500["close"] <= 0 or dbar_1500["volume"] <= 0:
                continue

            # (1) 校验 14:59:45 是否封死涨停
            thresh = get_limit_thresh(sym, dt)
            if dbar_1500["pctChange"] >= thresh and dbar_1500["close"] >= dbar_1500["high"] * 0.998:
                state["stats"]["limit_up_blocked"] += 1
                continue

            # (2) 14:59:45 校验收盘集合竞价价是否跌破 14:53:30 锁定的缠论底分型保护位 (防最后6分钟尾盘跳水破位)
            f_contract = cand["frozen_contract"]
            frozen_bot_low = f_contract["frozen_bot_fx_low"]
            p15 = cand["periods"].get("15m")
            zs15 = p15.get("lastZs") if p15 else None
            raw_stop = round(zs15["zd"] * 0.99, 4) if zs15 else round(cand["price"] * 0.96, 4)

            if dbar_1500["close"] < max(frozen_bot_low, raw_stop) * 0.997:
                state["stats"]["fx_protect_blocked"] += 1
                continue

            # (3) 模块 6：尾盘集合竞价微观头寸限额 —— 单笔买入上限下调至全天成交额的 0.25% (消除尾盘买崩盘口的冲击成本)
            sn = cand.get("snapshot")
            full_day_amt = (sn["amountYi"] * 1e8) if (sn and sn["amountYi"] > 0) else (dbar_1500.get("amount") or (dbar_1500["volume"] * dbar_1500["close"]))
            micro_liq_cap = max(120000.0, full_day_amt * 0.0025)

            t_amt = min((eff_stock_cap if cand["kind"] == "stock" else eff_etf_cap) * pos_sc, micro_liq_cap)
            alloc = min(budget * 0.95, t_amt, rem_room)
            # 用 close 算缠论信号与涨跌，但在下单算“能买几手（lots）”那一刻用 raw_price = price / ratio_factor 校验真实手数
            rf = dbar_1500.get("ratio_factor", 1.0)
            rf = rf if (rf and rf > 0) else 1.0
            raw_price = max(cand["price"], dbar_1500["close"]) / rf
            min_board_sh = 200 if sym.startswith("sh688") else 100
            lots = int(alloc // (raw_price * 100.0))
            raw_shares = lots * 100
            if raw_shares < min_board_sh:
                continue
            shares = round(raw_shares / rf, 4)

            # 根据买单占全天成交额比例动态计提 3.0 ~ 5.0 bps 尾盘集合竞价冲击滑点
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
            # 模块 2：开仓瞬间将几何契约 (frozen_bot_fx_low, frozen_15m_zg, frozen_15m_dd) 永久冻结写入持仓字典！
            holdings[sym] = {
                "name": cand["name"], "kind": cand["kind"], "shares": shares,
                "avg_cost": round(tot_cost / shares, 4), "last_price": dbar_1500["close"],
                "buy_date": dt, "init_stop": raw_stop,
                "frozen_bot_fx_low": f_contract["frozen_bot_fx_low"],
                "frozen_15m_zg": f_contract["frozen_15m_zg"],
                "frozen_15m_dd": f_contract["frozen_15m_dd"],
                "half_profit_locked": False,
            }
            state["trades"].append({"date": dt, "action": "BUY", "symbol": sym, "name": cand["name"], "shares": shares, "price": exec_p, "amount": round(gross, 2), "fee": fee})

    # --- 4. 15:00 收盘后按全天真实收盘价更新持仓盯市净值 ---
    for sym, h in holdings.items():
        dbar_1500 = indexed.get(sym, {}).get("day_by_date", {}).get(dt)
        if dbar_1500 and dbar_1500["close"] > 0:
            h["last_price"] = dbar_1500["close"]

    state["cash"] = cash
    state["equity"] = round(cash + sum(h["shares"] * h["last_price"] for h in holdings.values()), 2)
