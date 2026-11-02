import bisect
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sector_universes import get_three_sector_universes
from run_backtest_10y_100w import fetch_10y_symbol_with_hfq, CACHE_10Y_DIR

# 2016-2020年真实在沪深300中权重较高、但后来大幅衰落/被剔除的历史真实成分股 (消除幸存者偏差)
HISTORICAL_HS300_FALLEN = [
    ("sz002456", "欧菲光", "stock"), ("sh600340", "华夏幸福", "stock"), ("sh600606", "绿地控股", "stock"),
    ("sh600703", "三安光电", "stock"), ("sz002008", "大族激光", "stock"), ("sz002236", "大华股份", "stock"),
    ("sh600535", "天士力", "stock"), ("sh601155", "新城控股", "stock"), ("sh600332", "白云山", "stock"),
    ("sh600066", "宇通客车", "stock"), ("sz002146", "荣盛发展", "stock"), ("sh600029", "南方航空", "stock"),
    ("sh601111", "中国国航", "stock"), ("sh600115", "中国东航", "stock"), ("sh601336", "新华保险", "stock"),
    ("sh601901", "方正证券", "stock"), ("sh600109", "国金证券", "stock"), ("sz000783", "长江证券", "stock"),
    ("sh601555", "东吴证券", "stock"), ("sh600795", "国电电力", "stock"), ("sh601991", "大唐发电", "stock"),
    ("sz000157", "中联重科", "stock"), ("sh601607", "上海医药", "stock"), ("sh600061", "国投资本", "stock"),
    ("sz000728", "国元证券", "stock"), ("sh600837", "海通证券", "stock"), ("sz002736", "国信证券", "stock"),
    ("sh601788", "光大证券", "stock"), ("sh601377", "兴业证券", "stock"), ("sz002508", "老板电器", "stock"),
    ("sz002027", "分众传媒", "stock"), ("sz002602", "世纪华通", "stock"), ("sz002555", "三七互娱", "stock"),
    ("sh600637", "东方明珠", "stock"), ("sh601098", "中南传媒", "stock"), ("sz002032", "苏泊尔", "stock"),
]

# 2016-2020年真实在创业板指中权重极高、但后来腰斩80%-95%被剔除的历史真实成分股 (消除幸存者偏差)
HISTORICAL_CYB_FALLEN = [
    ("sz300315", "掌趣科技", "stock"), ("sz300182", "捷成股份", "stock"), ("sz300085", "银之杰", "stock"),
    ("sz300010", "豆神教育", "stock"), ("sz300197", "节能铁汉", "stock"), ("sz300055", "万邦达", "stock"),
    ("sz300266", "兴源环境", "stock"), ("sz300159", "新研股份", "stock"), ("sz300133", "华策影视", "stock"),
    ("sz300244", "迪安诊断", "stock"), ("sz300147", "香雪制药", "stock"), ("sz300078", "思创医惠", "stock"),
    ("sz300166", "东方国信", "stock"), ("sz300212", "易华录", "stock"), ("sz300271", "华宇软件", "stock"),
    ("sz300300", "海峡创新", "stock"), ("sz300290", "荣科科技", "stock"), ("sz300045", "华力创通", "stock"),
    ("sz300036", "超图软件", "stock"), ("sz300047", "天源迪科", "stock"), ("sz300052", "中青宝", "stock"),
    ("sz300077", "国民技术", "stock"), ("sz300113", "顺网科技", "stock"), ("sz300257", "开山股份", "stock"),
    ("sz300324", "旋极信息", "stock"), ("sz300369", "绿盟科技", "stock"), ("sz300377", "赢时胜", "stock"),
]

def get_candidate_pools():
    hs300_base, cyb_base, kcb_base = get_three_sector_universes()
    hs_map = {m[0]: m for m in (hs300_base + HISTORICAL_HS300_FALLEN)}
    cyb_map = {m[0]: m for m in (cyb_base + HISTORICAL_CYB_FALLEN)}
    kcb_map = {m[0]: m for m in kcb_base}
    return list(hs_map.values()), list(cyb_map.values()), list(kcb_map.values())

if __name__ == "__main__":
    hs_pool, cyb_pool, kcb_pool = get_candidate_pools()
    all_meta = hs_pool + cyb_pool + kcb_pool
    need = [m[0] for m in all_meta if not (CACHE_10Y_DIR / f"{m[0]}.pkl").exists()]
    print(f"Total Candidate Pool -> HS300:{len(hs_pool)}, CYB:{len(cyb_pool)}, KCB:{len(kcb_pool)} | Need fetch:{len(need)}")
    if need:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_10y_symbol_with_hfq, s): s for s in need}
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if i % 15 == 0 or i == len(need):
                    print(f"Fetched {i}/{len(need)} historical symbols...")
    print("Historical candidate pools ready!")

def select_pit_universe_at_date(candidate_pool, indexed_data, as_of_date, top_n, min_listed_days=60):
    """
    在调样日 as_of_date，仅使用 <= as_of_date 的历史数据：
    1. 标的必须在 as_of_date 前已上市满 min_listed_days 个交易日；
    2. 计算截至 as_of_date 过去 120 个交易日的日均真实成交额 (amountYi)；
    3. 按历史日均成交额从高到低取前 top_n 只作为当期真实指数成分股（ETF若已上市则保留）。
    """
    scored = []
    etfs = []
    for meta in candidate_pool:
        sym, name, kind = meta
        idx_e = indexed_data.get(sym)
        if not idx_e or not idx_e["day_dates"]:
            continue
        pos = bisect.bisect_right(idx_e["day_dates"], as_of_date)
        if pos < min_listed_days:
            continue
        if kind == "etf":
            etfs.append(meta)
            continue
        # 取截至 as_of_date 的过去最多 120 个交易日
        start_pos = max(0, pos - 120)
        recent_dates = idx_e["day_dates"][start_pos:pos]
        snaps = idx_e["snaps"]
        amts = [snaps[d]["amountYi"] for d in recent_dates if d in snaps and snaps[d]["lastPrice"] > 0]
        if len(amts) < 20:
            continue
        avg_amt = sum(amts) / len(amts)
        scored.append((avg_amt, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = etfs + [m for _, m in scored[:top_n]]
    return selected

if __name__ == "__main__":
    from run_backtest import prepare_indexed_data
    from run_backtest_2019_2020 import build_period_dataset
    hs_pool, cyb_pool, kcb_pool = get_candidate_pools()
    all_syms = list({m[0] for m in (hs_pool + cyb_pool + kcb_pool)} | {"sh000300"})
    raw_cache = {s: pickle.load(open(CACHE_10Y_DIR / f"{s}.pkl", "rb")) for s in all_syms if (CACHE_10Y_DIR / f"{s}.pkl").exists()}
    dataset = build_period_dataset(raw_cache, "2015-10-01", "2026-10-29", "2026-10-29")
    indexed = prepare_indexed_data(dataset)

    for dt in ["2016-10-28", "2018-06-29", "2020-06-30", "2022-06-30", "2024-06-28", "2026-06-30"]:
        cyb_pit = select_pit_universe_at_date(cyb_pool, indexed, dt, top_n=50, min_listed_days=60)
        kcb_pit = select_pit_universe_at_date(kcb_pool, indexed, dt, top_n=35, min_listed_days=20)
        names_cyb = [m[1] for m in cyb_pit if m[2]=="stock"][:12]
        names_kcb = [m[1] for m in kcb_pit if m[2]=="stock"][:8]
        print(f"[{dt}] 当期真实创业板池({len(cyb_pit)}只)前12: {names_cyb}")
        print(f"[{dt}] 当期真实科创板池({len(kcb_pit)}只)前8 : {names_kcb}")
