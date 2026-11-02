from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from universe import CSI300_UNIVERSE, get_universes
from run_backtest_10y_100w import fetch_10y_symbol_with_hfq, CACHE_10Y_DIR

# 补充2016年前已上市的老牌创业板指核心成分股
EXTRA_CYB_STOCKS = [
    ("sz300001", "特锐德", "stock"), ("sz300002", "神州泰岳", "stock"), ("sz300003", "乐普医疗", "stock"),
    ("sz300009", "安科生物", "stock"), ("sz300012", "华测检测", "stock"), ("sz300017", "网宿科技", "stock"),
    ("sz300024", "机器人", "stock"), ("sz300026", "红日药业", "stock"), ("sz300027", "华谊兄弟", "stock"),
    ("sz300058", "蓝色光标", "stock"), ("sz300070", "碧水源", "stock"), ("sz300072", "三聚环保", "stock"),
    ("sz300073", "当升科技", "stock"), ("sz300088", "长信科技", "stock"), ("sz300115", "长盈精密", "stock"),
    ("sz300136", "信维通信", "stock"), ("sz300142", "沃森生物", "stock"), ("sz300144", "宋城演艺", "stock"),
    ("sz300146", "汤臣倍健", "stock"), ("sz300168", "万达信息", "stock"), ("sz300207", "欣旺达", "stock"),
    ("sz300223", "北京君正", "stock"), ("sz300251", "光线传媒", "stock"), ("sz300253", "卫宁健康", "stock"),
    ("sz300285", "国瓷材料", "stock"), ("sz300296", "利亚德", "stock"), ("sz300316", "晶盛机电", "stock"),
    ("sz300347", "泰格医药", "stock"), ("sz300357", "我武生物", "stock"), ("sz300383", "光环新网", "stock"),
    ("sz300408", "三环集团", "stock"), ("sz300413", "芒果超媒", "stock"), ("sz300450", "先导智能", "stock"),
    ("sz300454", "深信服", "stock"), ("sz300496", "中科创达", "stock"),
]

# 补充2019.07-2020上市的首批科创50核心成分股
EXTRA_KCB_STOCKS = [
    ("sh688001", "华兴源创", "stock"), ("sh688002", "睿创微纳", "stock"), ("sh688005", "容百科技", "stock"),
    ("sh688006", "杭可科技", "stock"), ("sh688009", "中国通号", "stock"), ("sh688016", "心脉医疗", "stock"),
    ("sh688019", "安集科技", "stock"), ("sh688029", "南微医学", "stock"), ("sh688033", "天宜上佳", "stock"),
    ("sh688065", "凯赛生物", "stock"), ("sh688088", "虹软科技", "stock"), ("sh688099", "晶晨股份", "stock"),
    ("sh688122", "西部超导", "stock"), ("sh688169", "石头科技", "stock"), ("sh688180", "君实生物", "stock"),
    ("sh688185", "康希诺", "stock"), ("sh688202", "美迪西", "stock"), ("sh688208", "道通科技", "stock"),
    ("sh688363", "华熙生物", "stock"), ("sh688390", "固德威", "stock"), ("sh688396", "华润微", "stock"),
    ("sh688561", "奇安信", "stock"), ("sh688599", "天合光能", "stock"), ("sh688777", "中控技术", "stock"),
]

def get_three_sector_universes():
    hs300_u, all_a_u = get_universes()
    cached_10y = {p.stem for p in CACHE_10Y_DIR.glob("*.pkl")}

    cyb_map = {"sz159915": ("sz159915", "创业板ETF", "etf")}
    kcb_map = {"sh588000": ("sh588000", "科创50ETF", "etf")}

    for sym, name, kind in all_a_u + EXTRA_CYB_STOCKS + EXTRA_KCB_STOCKS:
        if sym.startswith("sz30") or sym == "sz159915":
            cyb_map[sym] = (sym, name, kind)
        elif sym.startswith("sh688") or sym == "sh588000":
            kcb_map[sym] = (sym, name, kind)

    return list(CSI300_UNIVERSE), list(cyb_map.values()), list(kcb_map.values())

if __name__ == "__main__":
    hs300_u, cyb_u, kcb_u = get_three_sector_universes()
    need_fetch = [m[0] for m in (hs300_u + cyb_u + kcb_u) if not (CACHE_10Y_DIR / f"{m[0]}.pkl").exists()]
    print(f"Sector sizes -> HS300: {len(hs300_u)}, CYB: {len(cyb_u)}, KCB: {len(kcb_u)} | Need fetch: {len(need_fetch)}")
    if need_fetch:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_10y_symbol_with_hfq, s): s for s in need_fetch}
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if i % 15 == 0 or i == len(need_fetch):
                    print(f"Fetched {i}/{len(need_fetch)} new sector symbols...")
    print("All sector symbols cached!")
