import json
import pickle
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pit_universe import get_candidate_pools, select_pit_universe_at_date
from run_backtest_2019_2020 import fetch_hfq_factors
from run_backtest_10y_100w import CACHE_10Y_DIR

CACHE_20Y_DIR = Path(__file__).resolve().parent / "cache_20y_hfq"
CACHE_20Y_DIR.mkdir(parents=True, exist_ok=True)

# 补充 2006-2015 年真实统治 A 股的老五朵金花（老地产基建、煤飞色舞、老金融、老商贸工业）
HISTORICAL_2006_2015_GIANTS = [
    # 老地产与大基建
    ("sz000002", "万科A", "stock"), ("sh600383", "金地集团", "stock"), ("sz000069", "华侨城A", "stock"),
    ("sh600649", "城投控股", "stock"), ("sh601618", "中国中冶", "stock"), ("sh601186", "中国铁建", "stock"),
    ("sh601390", "中国中铁", "stock"), ("sh601668", "中国建筑", "stock"), ("sh600820", "隧道股份", "stock"),
    ("sz000402", "金融街", "stock"), ("sh600266", "城建发展", "stock"), ("sh600663", "陆家嘴", "stock"),
    # 老银行与老券商金融
    ("sh601328", "交通银行", "stock"), ("sh600016", "民生银行", "stock"), ("sh600015", "华夏银行", "stock"),
    ("sh601169", "北京银行", "stock"), ("sh601009", "南京银行", "stock"), ("sz000166", "申万宏源", "stock"),
    ("sh601099", "太平洋", "stock"), ("sh600369", "西南证券", "stock"), ("sz000686", "东北证券", "stock"),
    ("sz000750", "国海证券", "stock"), ("sh600816", "建元信托", "stock"),
    # 2007/2010 煤飞色舞主力（有色、煤炭、钢铁、航运、化工）
    ("sh600362", "江西铜业", "stock"), ("sz000878", "云南铜业", "stock"), ("sz000060", "中金岭南", "stock"),
    ("sh600497", "驰宏锌锗", "stock"), ("sz000630", "铜陵有色", "stock"), ("sh601168", "西部矿业", "stock"),
    ("sh601958", "金钼股份", "stock"), ("sh600549", "厦门钨业", "stock"), ("sz000792", "盐湖股份", "stock"),
    ("sh600010", "包钢股份", "stock"), ("sz000898", "鞍钢股份", "stock"), ("sh600808", "马钢股份", "stock"),
    ("sz000709", "河钢股份", "stock"), ("sz000932", "华菱钢铁", "stock"), ("sh600005", "武钢股份", "stock"),
    ("sh600188", "兖矿能源", "stock"), ("sh601699", "潞安环能", "stock"), ("sh600123", "兰花科创", "stock"),
    ("sz000983", "山西焦煤", "stock"), ("sh601001", "晋控煤业", "stock"), ("sh601898", "中煤能源", "stock"),
    ("sh600026", "中远海能", "stock"), ("sh601872", "招商轮船", "stock"), ("sh601866", "中远海发", "stock"),
    # 老商贸、消费、医药、汽车、家电、电力设备
    ("sz002024", "苏宁易购", "stock"), ("sh600859", "王府井", "stock"), ("sh600694", "大商股份", "stock"),
    ("sh600655", "豫园股份", "stock"), ("sz000800", "一汽解放", "stock"), ("sh600741", "华域汽车", "stock"),
    ("sh600085", "同仁堂", "stock"), ("sz000423", "东阿阿胶", "stock"), ("sz000869", "张裕A", "stock"),
    ("sh600839", "四川长虹", "stock"), ("sz000100", "TCL科技", "stock"), ("sh600089", "特变电工", "stock"),
    ("sh600588", "用友网络", "stock"), ("sh600100", "同方股份", "stock"), ("sz000066", "中国长城", "stock"),
]

def get_20y_candidate_pools():
    hs_pool, cyb_pool, kcb_pool = get_candidate_pools()
    hs_map = {m[0]: m for m in (hs_pool + HISTORICAL_2006_2015_GIANTS)}
    return list(hs_map.values()), cyb_pool, kcb_pool

def fetch_20y_symbol_with_hfq(symbol):
    cf = CACHE_20Y_DIR / f"{symbol}.pkl"
    if cf.exists():
        try:
            with open(cf, "rb") as f:
                d = pickle.load(f)
                if d.get("raw_day"):
                    return symbol, d
        except Exception:
            pass

    f_dates, f_vals = [], []
    old_cf = CACHE_10Y_DIR / f"{symbol}.pkl"
    if old_cf.exists():
        try:
            with open(old_cf, "rb") as f:
                old_d = pickle.load(f)
                f_dates = old_d.get("f_dates", [])
                f_vals = old_d.get("f_vals", [])
        except Exception:
            pass
    if not f_dates and symbol != "sh000300":
        f_dates, f_vals = fetch_hfq_factors(symbol)

    raw_bars = []
    try:
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=6000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        arr = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
        if isinstance(arr, list):
            for r in arr:
                dt = str(r["day"])[:10]
                if dt >= "2004-06-01":
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

if __name__ == "__main__":
    hs_pool, cyb_pool, kcb_pool = get_20y_candidate_pools()
    all_syms = list({m[0] for m in (hs_pool + cyb_pool + kcb_pool)} | {"sh000300"})
    need = [s for s in all_syms if not (CACHE_20Y_DIR / f"{s}.pkl").exists()]
    print(f"20Y Total Pool: {len(all_syms)} symbols (HS300:{len(hs_pool)}, CYB:{len(cyb_pool)}, KCB:{len(kcb_pool)}) | Need fetch: {len(need)}")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_20y_symbol_with_hfq, s): s for s in need}
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 50 == 0 or i == len(need):
                print(f"Fetched 20Y {i}/{len(need)} symbols...")
    print("All 20Y symbols cached!")
