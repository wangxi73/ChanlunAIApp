def compute_macd(bars):
    ema12 = ema26 = dea = 0.0
    out = []
    for idx, b in enumerate(bars):
        c = b["close"]
        if idx == 0:
            ema12 = ema26 = c
        else:
            ema12 = ema12 * (11.0 / 13.0) + c * (2.0 / 13.0)
            ema26 = ema26 * (25.0 / 27.0) + c * (2.0 / 27.0)
        dif = ema12 - ema26
        dea = dea * 0.8 + dif * 0.2
        macd = (dif - dea) * 2.0
        out.append({
            "time": b["time"], "open": b["open"], "high": b["high"],
            "low": b["low"], "close": c, "volume": b.get("volume", 0.0),
            "dif": dif, "dea": dea, "macd": macd
        })
    return out

def analyze_period(raw_bars):
    if not raw_bars:
        return None
    bars = compute_macd(raw_bars)
    merged = [{
        "time": bars[0]["time"], "open": bars[0]["open"],
        "high": bars[0]["high"], "low": bars[0]["low"],
        "close": bars[0]["close"], "macdAbs": abs(bars[0]["macd"])
    }]
    up = bars[1]["high"] >= bars[0]["high"] if len(bars) > 1 else True
    for b in bars[1:]:
        last = merged[-1]
        if (b["high"] <= last["high"] and b["low"] >= last["low"]) or (b["high"] >= last["high"] and b["low"] <= last["low"]):
            if up:
                last["high"] = max(last["high"], b["high"])
                last["low"] = max(last["low"], b["low"])
            else:
                last["high"] = min(last["high"], b["high"])
                last["low"] = min(last["low"], b["low"])
            last["close"] = b["close"]
            last["macdAbs"] += abs(b["macd"])
        else:
            up = b["high"] > last["high"]
            merged.append({
                "time": b["time"], "open": b["open"],
                "high": b["high"], "low": b["low"],
                "close": b["close"], "macdAbs": abs(b["macd"])
            })

    fx = []
    for i in range(1, len(merged) - 1):
        p, c, n = merged[i - 1], merged[i], merged[i + 1]
        is_top = c["high"] > p["high"] and c["high"] > n["high"] and c["low"] > min(p["low"], n["low"])
        is_bot = c["low"] < p["low"] and c["low"] < n["low"] and c["high"] < max(p["high"], n["high"])
        if not (is_top or is_bot):
            continue
        kind = "top" if is_top else "bottom"
        val = c["high"] if is_top else c["low"]
        if not fx:
            fx.append((i, kind, val, c["time"]))
        else:
            li, lkind, lval, ltime = fx[-1]
            if lkind == kind:
                if (kind == "top" and val >= lval) or (kind == "bottom" and val <= lval):
                    fx[-1] = (i, kind, val, c["time"])
            else:
                if i - li >= 3 and ((lkind == "bottom" and val > lval) or (lkind == "top" and val < lval)):
                    fx.append((i, kind, val, c["time"]))
    strokes = []
    for i in range(len(fx) - 1):
        a, b = fx[i], fx[i + 1]
        area = sum(merged[k]["macdAbs"] for k in range(a[0], b[0] + 1))
        strokes.append({
            "type": "up" if b[1] == "top" else "down",
            "startTime": a[3], "endTime": b[3],
            "startPrice": a[2], "endPrice": b[2],
            "high": max(a[2], b[2]), "low": min(a[2], b[2]),
            "macdArea": round(area * 1000.0) / 1000.0
        })

    zs_list = []
    i = 0
    while i <= len(strokes) - 3:
        b1, b2, b3 = strokes[i], strokes[i + 1], strokes[i + 2]
        zg = min(b1["high"], b2["high"], b3["high"])
        zd = max(b1["low"], b2["low"], b3["low"])
        if zg > zd:
            gg = max(b1["high"], b2["high"], b3["high"])
            dd = min(b1["low"], b2["low"], b3["low"])
            cnt = 3
            j = i + 3
            while j < len(strokes):
                if strokes[j]["low"] <= zg and strokes[j]["high"] >= zd:
                    gg = max(gg, strokes[j]["high"])
                    dd = min(dd, strokes[j]["low"])
                    cnt += 1
                    j += 1
                else:
                    break
            zs_list.append({
                "zg": round(zg * 1000.0) / 1000.0,
                "zd": round(zd * 1000.0) / 1000.0,
                "gg": round(gg * 1000.0) / 1000.0,
                "dd": round(dd * 1000.0) / 1000.0,
                "biCount": cnt
            })
            i = j - 1 if j > i + 3 else i + 1
        else:
            i += 1

    last_bi = strokes[-1] if strokes else None
    last_zs = zs_list[-1] if zs_list else None
    div_ratio = None
    if last_bi:
        same_type = [s for s in strokes[:-1] if s["type"] == last_bi["type"]]
        if same_type and same_type[-1]["macdArea"] > 0:
            div_ratio = round((last_bi["macdArea"] / same_type[-1]["macdArea"]) * 100.0) / 100.0

    last_bar = bars[-1]
    return {
        "time": last_bar["time"], "close": last_bar["close"],
        "dif": round(last_bar["dif"] * 1000.0) / 1000.0,
        "dea": round(last_bar["dea"] * 1000.0) / 1000.0,
        "macd": round(last_bar["macd"] * 1000.0) / 1000.0,
        "lastBi": last_bi, "lastZs": last_zs, "divRatio": div_ratio
    }

def evaluate_asset(meta, day_bars, m60_bars, m15_bars, snapshot):
    sym, name, kind = meta
    p_day = analyze_period(day_bars[-250:]) if day_bars else None
    p_m60 = analyze_period(m60_bars[-280:]) if m60_bars else None
    p_m15 = analyze_period(m15_bars[-280:]) if m15_bars else None

    score = 0
    tags = []

    if p_day and p_day["lastZs"]:
        zs = p_day["lastZs"]
        if p_day["close"] > zs["zg"]:
            score += 30
            tags.append(f"日线站上中枢ZG({zs['zg']})")
        elif p_day["close"] >= zs["zd"]:
            score += 15
            tags.append(f"日线中枢[{zs['zd']}-{zs['zg']}]震荡")
    if p_day and p_day["lastBi"] and p_day["lastBi"]["type"] == "down":
        if p_day["divRatio"] is not None and p_day["divRatio"] < 0.75:
            score += 25
            tags.append(f"日线向下离开笔盘整底背驰一买(面积比={p_day['divRatio']})")
        else:
            tags.append("日线向下离开笔未背驰")

    if p_m60 and p_m60["lastZs"]:
        zs = p_m60["lastZs"]
        if p_m60["close"] > zs["zg"]:
            score += 25
            tags.append(f"60m中枢上方三买区(ZG={zs['zg']})")
    if p_m60 and p_m60["lastBi"]:
        bi = p_m60["lastBi"]
        if bi["type"] == "up" and (p_m60["divRatio"] is None or p_m60["divRatio"] >= 0.7):
            score += 20
            tags.append(f"60m一买向上笔反弹延续(力度比={p_m60['divRatio'] if p_m60['divRatio'] is not None else 1.0})")
        elif bi["type"] == "down" and p_m60["divRatio"] is not None and p_m60["divRatio"] < 0.8:
            score += 15
            tags.append(f"60m向下笔底背驰(面积比={p_m60['divRatio']})")

    if p_m15:
        zs = p_m15["lastZs"]
        bi = p_m15["lastBi"]
        if zs and bi and bi["type"] == "down" and bi["low"] >= zs["zg"]:
            score += 30
            tags.append(f"15m回调不破中枢上沿ZG({zs['zg']})构成标准三买")
        elif bi and bi["type"] == "down" and p_m15["divRatio"] is not None and p_m15["divRatio"] < 0.8:
            score += 25
            tags.append(f"15m向下笔MACD面积收敛背驰二买(面积比={p_m15['divRatio']})")
        elif bi and bi["type"] == "up" and p_m15["divRatio"] is not None and p_m15["divRatio"] < 0.75:
            score -= 20
            tags.append(f"15m向上笔盘整顶背驰一卖(面积比={p_m15['divRatio']})")
        elif bi and bi["type"] == "up" and p_m15["dif"] > p_m15["dea"]:
            score += 10
            tags.append("15m向上笔延续")

    if snapshot:
        if snapshot["buyerDom"] >= 8.0:
            score += 8
            tags.append(f"主买盘优势{snapshot['buyerDom']:+.1f}%(成交{snapshot['amountYi']:.1f}亿)")
        elif snapshot["buyerDom"] < -15.0:
            score -= 5

    price = snapshot["lastPrice"] if snapshot and snapshot["lastPrice"] > 0 else (p_m15["close"] if p_m15 else (p_day["close"] if p_day else 0.0))
    return {
        "symbol": sym, "name": name, "kind": kind,
        "price": round(price * 1000.0) / 1000.0,
        "score": score, "tags": tags,
        "periods": {"day": p_day, "60m": p_m60, "15m": p_m15},
        "snapshot": snapshot
    }

def get_stage1_candidates(universe, snapshots, held_symbols, top_k=44):
    meta_map = {m[0]: m for m in universe}
    selected = []
    seen = set()

    for sym in held_symbols:
        if sym in meta_map and sym not in seen:
            selected.append(meta_map[sym])
            seen.add(sym)

    def score_a(sym):
        s = snapshots.get(sym)
        if not s or s["lastPrice"] <= 0 or s["amountYi"] < 1.2:
            return -1e6
        return min(s["amountYi"], 50.0) * 0.45 + s["buyerDom"] * 0.9

    def score_b(sym):
        s = snapshots.get(sym)
        if not s or s["lastPrice"] <= 0 or s["amountYi"] < 1.2:
            return -1e6
        pullback = 25.0 if (-4.8 <= s["pctChange"] <= 1.2) else 0.0
        return pullback + s["buyerDom"] * 0.8 + min(s["amountYi"], 25.0) * 0.3

    pool_codes = [m[0] for m in universe if m[0] not in seen]
    bucket_a_limit = max((top_k * 3) // 5, 15)
    sorted_a = sorted(pool_codes, key=score_a, reverse=True)
    for sym in sorted_a[:bucket_a_limit]:
        if len(selected) >= top_k:
            break
        if score_a(sym) > -1e5:
            selected.append(meta_map[sym])
            seen.add(sym)

    rem_codes = [c for c in pool_codes if c not in seen]
    sorted_b = sorted(rem_codes, key=score_b, reverse=True)
    for sym in sorted_b:
        if len(selected) >= top_k:
            break
        if score_b(sym) > -1e5:
            selected.append(meta_map[sym])
            seen.add(sym)

    return selected
