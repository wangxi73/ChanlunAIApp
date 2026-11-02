import json
import pickle
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_2Y_DIR = Path(__file__).resolve().parent / "cache_2y"
CACHE_2Y_DIR.mkdir(parents=True, exist_ok=True)

def fetch_single_symbol_2y(symbol):
    cache_2y = CACHE_2Y_DIR / f"{symbol}.pkl"
    if cache_2y.exists():
        try:
            with open(cache_2y, "rb") as f:
                return symbol, pickle.load(f)
        except Exception:
            pass

    old_cache = CACHE_DIR / f"{symbol}.pkl"
    old_data = None
    if old_cache.exists():
        try:
            with open(old_cache, "rb") as f:
                old_data = pickle.load(f)
        except Exception:
            pass

    day_bars = []
    try:
        url_d = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,850,qfq"
        req_d = urllib.request.Request(url_d, headers={"User-Agent": "Mozilla/5.0"})
        raw_d = json.loads(urllib.request.urlopen(req_d, timeout=12).read().decode())
        stk = raw_d.get("data", {}).get(symbol, {})
        arr_d = stk.get("qfqday") or stk.get("day") or []
        for r in arr_d:
            if len(r) >= 6:
                day_bars.append({
                    "time": str(r[0]), "open": float(r[1]), "close": float(r[2]),
                    "high": float(r[3]), "low": float(r[4]), "volume": float(r[5])
                })
    except Exception:
        if old_data:
            day_bars = old_data["day"]

    m30_raw = []
    if symbol != "sh000300":
        try:
            url_30 = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={symbol}&scale=30&ma=no&datalen=5200"
            req_30 = urllib.request.Request(url_30, headers={"User-Agent": "Mozilla/5.0"})
            arr_30 = json.loads(urllib.request.urlopen(req_30, timeout=15).read().decode())
            if isinstance(arr_30, list):
                for r in arr_30:
                    m30_raw.append({
                        "time": str(r["day"]), "date": str(r["day"])[:10],
                        "open": float(r["open"]), "high": float(r["high"]),
                        "low": float(r["low"]), "close": float(r["close"]),
                        "volume": float(r["volume"])
                    })
        except Exception:
            pass
    by_date_15 = {}
    if old_data and "m15" in old_data:
        for b in old_data["m15"]:
            by_date_15.setdefault(b["date"], []).append(b)

    by_date_30 = {}
    for b in m30_raw:
        by_date_30.setdefault(b["date"], []).append(b)

    m15_bars = []
    m60_bars = []
    daily_snaps = {}

    for d_idx, db in enumerate(day_bars):
        dt = db["time"]
        pre_close = day_bars[d_idx - 1]["close"] if d_idx > 0 else db["open"]
        pct_chg = ((db["close"] - pre_close) / pre_close * 100.0) if pre_close > 0 else 0.0

        if dt in by_date_15 and len(by_date_15[dt]) >= 8:
            adj_15 = by_date_15[dt]
            m15_bars.extend(adj_15)
            for k in range(0, len(adj_15), 4):
                chunk = adj_15[k:k + 4]
                m60_bars.append({
                    "time": chunk[-1]["time"], "date": dt,
                    "open": chunk[0]["open"],
                    "high": max(x["high"] for x in chunk),
                    "low": min(x["low"] for x in chunk),
                    "close": chunk[-1]["close"],
                    "volume": sum(x["volume"] for x in chunk)
                })
            if old_data and dt in old_data.get("snaps", {}):
                daily_snaps[dt] = old_data["snaps"][dt]
                continue

        day_30 = by_date_30.get(dt, [])
        if day_30:
            raw_close = day_30[-1]["close"]
            factor = (db["close"] / raw_close) if raw_close > 0 else 1.0
            if abs(factor - 1.0) < 1e-4:
                factor = 1.0
            adj_15 = []
            buy_vol = sell_vol = tot_amt = 0.0
            for b in day_30:
                o = round(b["open"] * factor, 3)
                h = round(b["high"] * factor, 3)
                l = round(b["low"] * factor, 3)
                c = round(b["close"] * factor, 3)
                v = b["volume"]
                m = round((o + c) / 2.0, 3)
                if c >= o:
                    adj_15.append({"time": b["time"] + "_1", "date": dt, "open": o, "high": max(o, m), "low": l, "close": m, "volume": v * 0.5})
                    adj_15.append({"time": b["time"], "date": dt, "open": m, "high": h, "low": min(m, c), "close": c, "volume": v * 0.5})
                else:
                    adj_15.append({"time": b["time"] + "_1", "date": dt, "open": o, "high": h, "low": min(o, m), "close": m, "volume": v * 0.5})
                    adj_15.append({"time": b["time"], "date": dt, "open": m, "high": max(m, c), "low": l, "close": c, "volume": v * 0.5})
                tot_amt += v * b["close"]
                rng = b["high"] - b["low"]
                w = ((b["close"] - b["low"]) / rng) if rng > 0 else (0.55 if b["close"] >= b["open"] else 0.45)
                buy_vol += v * w
                sell_vol += v * (1.0 - w)
            m15_bars.extend(adj_15)
            for k in range(0, len(adj_15), 4):
                chunk = adj_15[k:k + 4]
                m60_bars.append({
                    "time": chunk[-1]["time"], "date": dt,
                    "open": chunk[0]["open"],
                    "high": max(x["high"] for x in chunk),
                    "low": min(x["low"] for x in chunk),
                    "close": chunk[-1]["close"],
                    "volume": sum(x["volume"] for x in chunk)
                })
            buyer_dom = ((buy_vol - sell_vol) / (buy_vol + sell_vol) * 100.0) if (buy_vol + sell_vol) > 0 else 0.0
            amount_yi = tot_amt / 1e8
        else:
            amount_yi = (db["volume"] * 100.0 * db["close"]) / 1e8
            buyer_dom = 0.0

        daily_snaps[dt] = {
            "symbol": symbol, "lastPrice": db["close"], "preClose": pre_close,
            "pctChange": round(pct_chg, 2), "amountYi": round(amount_yi, 2),
            "buyerDom": round(buyer_dom, 1)
        }

    payload = {"day": day_bars, "m60": m60_bars, "m15": m15_bars, "snaps": daily_snaps}
    if day_bars:
        with open(cache_2y, "wb") as f:
            pickle.dump(payload, f)
    return symbol, payload

def load_all_data_2y(symbols, max_workers=12):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_single_symbol_2y, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, data = fut.result()
            results[sym] = data
            if i % 25 == 0 or i == len(symbols):
                print(f"Loaded 2Y {i}/{len(symbols)} symbols...")
    return results
load_all_data = load_all_data_2y
