import json
import pickle
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from universe import get_universes
from run_backtest import prepare_indexed_data, simulate_account, compute_metrics

CACHE_2018_DIR = Path(__file__).resolve().parent / "cache_2018"
CACHE_2018_DIR.mkdir(parents=True, exist_ok=True)

def build_intraday_from_daily(day_bars):
    m60_bars = []
    m15_bars = []
    daily_snaps = {}

    for d_idx, db in enumerate(day_bars):
        dt = db["time"]
        o, h, l, c, vol = db["open"], db["high"], db["low"], db["close"], db["volume"]
        pre_close = day_bars[d_idx - 1]["close"] if d_idx > 0 else o
        pct_chg = ((c - pre_close) / pre_close * 100.0) if pre_close > 0 else 0.0

        # Construct 4 realistic 60m bars per day preserving exact daily O, H, L, C
        m1 = round(o + (c - o) * 0.35, 3)
        m2 = round(o + (c - o) * 0.65, 3)
        if c >= o:
            b60_raw = [
                ("10:30:00", o, max(o, l + (h - l) * 0.35), l, max(o, m1), vol * 0.30),
                ("11:30:00", max(o, m1), max(o, m1, l + (h - l) * 0.65), min(o, m1), m1, vol * 0.22),
                ("14:00:00", m1, h, min(m1, m2), m2, vol * 0.23),
                ("15:00:00", m2, max(m2, c), min(m2, c, l + (h - l) * 0.45), c, vol * 0.25),
            ]
        else:
            b60_raw = [
                ("10:30:00", o, h, min(o, m1), m1, vol * 0.30),
                ("11:30:00", m1, max(m1, l + (h - l) * 0.65), min(m1, m2), m2, vol * 0.22),
                ("14:00:00", m2, max(m2, c), l, min(m2, c), vol * 0.23),
                ("15:00:00", min(m2, c), max(m2, c, l + (h - l) * 0.35), min(m2, c), c, vol * 0.25),
            ]

        buy_vol = sell_vol = 0.0
        for t_sfx, bo, bh, bl, bc, bv in b60_raw:
            bh = round(max(bo, bh, bl, bc), 3)
            bl = round(min(bo, bh, bl, bc), 3)
            bo = round(bo, 3)
            bc = round(bc, 3)
            m60_bars.append({
                "time": f"{dt} {t_sfx}", "date": dt,
                "open": bo, "high": bh, "low": bl, "close": bc, "volume": bv
            })
            # Split each 60m bar into four 15m bars preserving exact 60m O, H, L, C
            p1 = round(bo + (bc - bo) * 0.3, 3)
            p2 = round(bo + (bc - bo) * 0.6, 3)
            p3 = round(bo + (bc - bo) * 0.85, 3)
            if bc >= bo:
                sub15 = [
                    (f"{dt} {t_sfx}_1", bo, max(bo, p1), bl, p1, bv * 0.28),
                    (f"{dt} {t_sfx}_2", p1, max(p1, p2), min(bo, p1), p2, bv * 0.22),
                    (f"{dt} {t_sfx}_3", p2, bh, min(p2, p3), p3, bv * 0.24),
                    (f"{dt} {t_sfx}_4", p3, max(p3, bc), min(p3, bc), bc, bv * 0.26),
                ]
            else:
                sub15 = [
                    (f"{dt} {t_sfx}_1", bo, bh, min(bo, p1), p1, bv * 0.28),
                    (f"{dt} {t_sfx}_2", p1, max(p1, p2), min(p1, p2), p2, bv * 0.22),
                    (f"{dt} {t_sfx}_3", p2, max(p2, p3), bl, p3, bv * 0.24),
                    (f"{dt} {t_sfx}_4", p3, max(p3, bc), min(p3, bc), bc, bv * 0.26),
                ]
            for st, so, sh, sl, sc, sv in sub15:
                sh = round(max(so, sh, sl, sc), 3)
                sl = round(min(so, sh, sl, sc), 3)
                m15_bars.append({
                    "time": st, "date": dt,
                    "open": so, "high": sh, "low": sl, "close": sc, "volume": sv
                })
                rng = sh - sl
                w = ((sc - sl) / rng) if rng > 0 else (0.55 if sc >= so else 0.45)
                buy_vol += sv * w
                sell_vol += sv * (1.0 - w)

        buyer_dom = ((buy_vol - sell_vol) / (buy_vol + sell_vol) * 100.0) if (buy_vol + sell_vol) > 0 else 0.0
        # Tencent daily volume is in lots (100 shares)
        amount_yi = (vol * 100.0 * ((o + h + l + c) / 4.0)) / 1e8
        daily_snaps[dt] = {
            "symbol": db.get("symbol", ""), "lastPrice": c, "preClose": pre_close,
            "pctChange": round(pct_chg, 2), "amountYi": round(amount_yi, 2),
            "buyerDom": round(buyer_dom, 1)
        }
    return m60_bars, m15_bars, daily_snaps

def fetch_2018_symbol(symbol):
    cf = CACHE_2018_DIR / f"{symbol}.pkl"
    if cf.exists():
        try:
            with open(cf, "rb") as f:
                return symbol, pickle.load(f)
        except Exception:
            pass

    day_bars = []
    try:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,2016-12-01,2019-04-05,650,"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = json.loads(urllib.request.urlopen(req, timeout=12).read().decode())
        stk = raw.get("data", {}).get(symbol, {})
        arr = stk.get("day") or stk.get("qfqday") or []
        for r in arr:
            if len(r) >= 6:
                c = float(r[2])
                if c > 0:
                    day_bars.append({
                        "symbol": symbol, "time": str(r[0]),
                        "open": float(r[1]), "close": c,
                        "high": float(r[3]), "low": float(r[4]), "volume": float(r[5])
                    })
    except Exception:
        pass

    if symbol != "sh000300" and day_bars:
        m60_bars, m15_bars, daily_snaps = build_intraday_from_daily(day_bars)
    else:
        m60_bars, m15_bars, daily_snaps = [], [], {}

    payload = {"day": day_bars, "m60": m60_bars, "m15": m15_bars, "snaps": daily_snaps}
    if day_bars:
        with open(cf, "wb") as f:
            pickle.dump(payload, f)
    return symbol, payload

if __name__ == "__main__":
    hs300_u, all_a_u = get_universes()
    cache_dir = Path(__file__).resolve().parent / "cache"
    cached_syms = {p.stem for p in cache_dir.glob("*.pkl")}
    if cached_syms:
        all_a_u = [m for m in all_a_u if m[0] in cached_syms]
    all_syms = ["sh000300"] + [m[0] for m in all_a_u]
    print(f"Fetching 2017-2019 K-line data for {len(all_syms)} symbols...")

    raw_data = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch_2018_symbol, s): s for s in all_syms}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, d = fut.result()
            raw_data[sym] = d
            if i % 50 == 0 or i == len(all_syms):
                print(f"Loaded 2018-2019 {i}/{len(all_syms)} symbols...")

    indexed = prepare_indexed_data(raw_data)
    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2018-01-29" <= b["time"] <= "2019-04-01"]
    print(f"Running backtest over {len(trade_dates)} trading days ({trade_dates[0]} -> {trade_dates[-1]})...")

    res_hs300 = simulate_account(
        "quant_hs300_2018", "账户①：无AI介入 · 仅限沪深300 (¥100,000)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map
    )
    res_all_a = simulate_account(
        "quant_all_a_2018", "账户②：无AI介入 · 全市场沪深A股 (¥100,000)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map
    )

    m_hs300 = compute_metrics(res_hs300)
    m_all_a = compute_metrics(res_all_a)
    res_hs300["metrics"] = m_hs300
    res_all_a["metrics"] = m_all_a

    out_path = Path(__file__).resolve().parent / "backtest_results_2018_2019.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"quant_hs300": res_hs300, "quant_all_a": res_all_a}, f, ensure_ascii=False, indent=2)

    print("2018-2019 HS300 Metrics:", json.dumps(m_hs300, ensure_ascii=False))
    print("2018-2019 All-A Metrics:", json.dumps(m_all_a, ensure_ascii=False))
