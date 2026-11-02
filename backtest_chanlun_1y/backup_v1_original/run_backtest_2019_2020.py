import bisect
import json
import pickle
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from universe import get_universes
from run_backtest import prepare_indexed_data, simulate_account, compute_metrics
from run_backtest_2018 import build_intraday_from_daily

CACHE_FULL_DIR = Path(__file__).resolve().parent / "cache_full_hfq"
CACHE_FULL_DIR.mkdir(parents=True, exist_ok=True)

def fetch_hfq_factors(symbol):
    if symbol == "sh000300" or symbol.startswith("sh51") or symbol.startswith("sh58") or symbol.startswith("sz15"):
        return [], []
    try:
        url = f"https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=10).read().decode("gbk", errors="ignore")
        m = re.search(r"\"data\":(\[.*?\])", txt)
        if not m:
            return [], []
        arr = json.loads(m.group(1))
        arr.sort(key=lambda x: x["d"])
        return [x["d"] for x in arr], [float(x["f"]) for x in arr]
    except Exception:
        return [], []

def fetch_full_symbol_with_hfq(symbol):
    cf = CACHE_FULL_DIR / f"{symbol}.pkl"
    if cf.exists():
        try:
            with open(cf, "rb") as f:
                d = pickle.load(f)
                if d.get("day"):
                    return symbol, d
        except Exception:
            pass

    f_dates, f_vals = fetch_hfq_factors(symbol)
    raw_bars = []
    try:
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=2300"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        arr = json.loads(urllib.request.urlopen(req, timeout=12).read().decode())
        if isinstance(arr, list):
            for r in arr:
                dt = str(r["day"])[:10]
                c = float(r["close"])
                if c > 0:
                    raw_bars.append({
                        "symbol": symbol, "time": dt,
                        "open": float(r["open"]), "close": c,
                        "high": float(r["high"]), "low": float(r["low"]),
                        "volume": float(r["volume"]) / 100.0
                    })
    except Exception:
        pass

    payload = {"raw_day": raw_bars, "f_dates": f_dates, "f_vals": f_vals}
    if raw_bars:
        with open(cf, "wb") as f:
            pickle.dump(payload, f)
    return symbol, payload

def build_period_dataset(raw_cache, start_warmup, end_date, anchor_date):
    dataset = {}
    for sym, item in raw_cache.items():
        raw_day = [b for b in item.get("raw_day", []) if start_warmup <= b["time"] <= end_date]
        if not raw_day:
            dataset[sym] = {"day": [], "m60": [], "m15": [], "snaps": {}}
            continue
        f_dates = item.get("f_dates", [])
        f_vals = item.get("f_vals", [])

        def get_f(dt):
            if not f_dates:
                return 1.0
            idx = bisect.bisect_right(f_dates, dt) - 1
            return f_vals[idx] if idx >= 0 else 1.0

        base_f = get_f(anchor_date)
        if base_f <= 0:
            base_f = 1.0

        adj_day = []
        for b in raw_day:
            dt = b["time"]
            ratio = (get_f(dt) / base_f) if sym != "sh000300" else 1.0
            adj_day.append({
                "symbol": sym, "time": dt,
                "open": round(b["open"] * ratio, 3),
                "high": round(b["high"] * ratio, 3),
                "low": round(b["low"] * ratio, 3),
                "close": round(b["close"] * ratio, 3),
                "volume": b["volume"]
            })

        if sym != "sh000300":
            m60, m15, snaps = build_intraday_from_daily(adj_day)
            # Keep true unadjusted daily turnover (amountYi) so liquidity filter isn't distorted
            for b in raw_day:
                dt = b["time"]
                if dt in snaps:
                    snaps[dt]["amountYi"] = round((b["volume"] * 100.0 * ((b["open"]+b["high"]+b["low"]+b["close"])/4.0)) / 1e8, 2)
        else:
            m60, m15, snaps = [], [], {}

        dataset[sym] = {"day": adj_day, "m60": m60, "m15": m15, "snaps": snaps}
    return dataset

if __name__ == "__main__":
    hs300_u, all_a_u = get_universes()
    cache_dir = Path(__file__).resolve().parent / "cache"
    cached_syms = {p.stem for p in cache_dir.glob("*.pkl")}
    if cached_syms:
        all_a_u = [m for m in all_a_u if m[0] in cached_syms]
    all_syms = ["sh000300"] + [m[0] for m in all_a_u]
    print(f"Fetching full K-line + HFQ factors for {len(all_syms)} symbols...")

    raw_cache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_full_symbol_with_hfq, s): s for s in all_syms}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, d = fut.result()
            raw_cache[sym] = d
            if i % 50 == 0 or i == len(all_syms):
                print(f"Loaded {i}/{len(all_syms)} symbols...")

    dataset = build_period_dataset(raw_cache, "2018-01-01", "2020-04-05", "2019-04-01")
    indexed = prepare_indexed_data(dataset)
    bench_bars = indexed["sh000300"]["day"]
    bench_map = {b["time"]: b["close"] for b in bench_bars}
    trade_dates = [b["time"] for b in bench_bars if "2019-04-01" <= b["time"] <= "2020-04-01"]
    print(f"Running backtest over {len(trade_dates)} trading days ({trade_dates[0]} -> {trade_dates[-1]})...")

    res_hs300 = simulate_account(
        "quant_hs300_2019_2020", "账户①：无AI介入 · 仅限沪深300 (¥100,000)",
        "仅限沪深300成分股", hs300_u, indexed, trade_dates, bench_map
    )
    res_all_a = simulate_account(
        "quant_all_a_2019_2020", "账户②：无AI介入 · 全市场沪深A股 (¥100,000)",
        "全市场沪深A股(两级漏斗)", all_a_u, indexed, trade_dates, bench_map
    )

    m_hs300 = compute_metrics(res_hs300)
    m_all_a = compute_metrics(res_all_a)
    res_hs300["metrics"] = m_hs300
    res_all_a["metrics"] = m_all_a

    out_path = Path(__file__).resolve().parent / "backtest_results_2019_2020.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"quant_hs300": res_hs300, "quant_all_a": res_all_a}, f, ensure_ascii=False, indent=2)

    print("2019-2020 HS300 Metrics (HFQ-adjusted):", json.dumps(m_hs300, ensure_ascii=False))
    print("2019-2020 All-A Metrics (HFQ-adjusted):", json.dumps(m_all_a, ensure_ascii=False))
