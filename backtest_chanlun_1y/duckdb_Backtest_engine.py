"""
唯一数据源与实盘级回测引擎 (DuckDB Direct + Strict PIT + 14:53:30->14:54:00->14:59:45 + 2023-08-28分段税费)
"""
import bisect
import time
import duckdb
from pit_universe_20y import get_20y_candidate_pools

DB_PATH = "/Volumes/Lexar-2T/A_Share_10Y_Klines/a_share_10y.duckdb"
KCB_ACTIVE_DATE = "2019-08-26"

def get_limit_thresh(symbol: str, dt: str) -> float:
    if symbol.startswith("sh688") or symbol == "sh588000":
        return 19.6
    if (symbol.startswith("sz30") or symbol == "sz159915") and dt >= "2020-08-24":
        return 19.6
    return 9.75

def calc_trade_cost(action: str, symbol: str, kind: str, dt: str, raw_price: float, shares: int) -> tuple:
    """
    实盘级尾盘黄金6分钟撮合与分段税费 (以 2023-08-28 印花税减半为界):
    - 14:59:45 收盘集合竞价买入：集合竞价单一价格撮合(exec_price = close)，佣金万2.5 (最低5元)
    - 14:54:00 连续竞价卖出 / 早盘反抽卖出：含 2 bps (0.02%) 盘口价差滑点 + 佣金万2.5 + 分段印花税(2023-08-28前千1，之后万5，ETF免征)
    """
    if action == "BUY":
        exec_price = round(raw_price, 4)
        gross = exec_price * shares
        comm = max(gross * 0.00025, 5.0)
        return exec_price, gross, round(comm, 2)
    else:
        exec_price = round(raw_price * (1.0 - 0.0002), 4)
        gross = exec_price * shares
        comm = max(gross * 0.00025, 5.0)
        if kind == "etf" or symbol in ("sh510300", "sh588000", "sz159915"):
            stamp_tax = 0.0
        else:
            stamp_rate = 0.0005 if dt >= "2023-08-28" else 0.0010
            stamp_tax = gross * stamp_rate
        return exec_price, gross, round(comm + stamp_tax, 2)

def load_duckdb_universe_data():
    t0 = time.time()
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA disable_progress_bar;")

    hs_pool, cyb_pool, kcb_pool = get_20y_candidate_pools()
    core_rows = con.execute("SELECT symbol, name FROM stocks WHERE in_core = TRUE").fetchall()
    db_name_map = {r[0]: r[1] for r in core_rows}

    hs_pool = [m for m in hs_pool if m[0] in db_name_map]
    cyb_pool = [m for m in cyb_pool if m[0] in db_name_map]
    kcb_pool = [m for m in kcb_pool if m[0] in db_name_map]

    full_meta_map = {m[0]: m for m in (hs_pool + cyb_pool + kcb_pool)}
    all_a_pool = []
    for sym, name in core_rows:
        if sym == "sh000300":
            continue
        kind = "etf" if sym in ("sh510300", "sh588000", "sz159915") else "stock"
        meta = full_meta_map.get(sym, (sym, name, kind))
        full_meta_map[sym] = meta
        all_a_pool.append(meta)

    target_syms = sorted(list(full_meta_map.keys()) + ["sh000300"])
    sym_list_sql = ",".join(f"'{s}'" for s in target_syms)

    from pathlib import Path
    db_root = Path(DB_PATH).parent
    def query_kline_table(tbl_name, subdir, cols):
        try:
            return con.execute(f"SELECT {cols} FROM {tbl_name} WHERE symbol IN ({sym_list_sql}) AND close > 0 ORDER BY symbol, {cols.split(',')[1].strip()}").fetchall()
        except Exception:
            files = [str(db_root / subdir / f"{sym}.parquet") for sym in target_syms if (db_root / subdir / f"{sym}.parquet").exists() and (db_root / subdir / f"{sym}.parquet").stat().st_size > 100]
            files_sql = ",".join(f"'{p}'" for p in files)
            return con.execute(f"SELECT {cols} FROM read_parquet([{files_sql}]) WHERE close > 0 ORDER BY symbol, {cols.split(',')[1].strip()}").fetchall()

    print(f"1. Querying kline_1d, kline_60m, kline_15m via DuckDB for {len(target_syms)} symbols...")
    rows_1d = query_kline_table("kline_1d", "1d", "symbol, date, open, high, low, close, volume, amount")
    rows_60m = query_kline_table("kline_60m", "60m", "symbol, date, datetime, open, high, low, close, volume")
    rows_15m = query_kline_table("kline_15m", "15m", "symbol, date, datetime, open, high, low, close, volume")
    con.close()

    indexed = {s: {
        "day": [], "day_dates": [], "day_by_date": {},
        "m60_tuples": [], "m60_dates": [],
        "m15_tuples": [], "m15_dates": [],
        "snaps": {}
    } for s in target_syms}

    for sym, dt, o, h, l, c, v, amt in rows_1d:
        e = indexed[sym]
        prev_c = e["day"][-1]["close"] if e["day"] else o
        pct = ((c - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
        bar = {"symbol": sym, "time": dt, "open": o, "high": h, "low": l, "close": c, "volume": v, "amount": amt, "preClose": prev_c, "pctChange": pct}
        e["day"].append(bar)
        e["day_dates"].append(dt)
        e["day_by_date"][dt] = bar
        if sym != "sh000300":
            # 直接使用 kline_1d 表自带的真实 amount 成交额 (元 -> 亿)
            amt_yi = (amt / 1e8) if (amt and amt > 0) else ((v * c) / 1e8)
            # 基于当日真实日K线多空位置与实体力度计算主买盘优势 buyerDom
            rng = h - l
            if rng > 0:
                pos_w = (c - l) / rng
                body_w = (c - o) / rng
                bdom = (pos_w * 2.0 - 1.0) * 32.0 + body_w * 28.0 + max(-8.0, min(8.0, pct)) * 2.2
            else:
                bdom = 25.0 if pct > 0 else (-25.0 if pct < 0 else 0.0)
            e["snaps"][dt] = {
                "symbol": sym, "lastPrice": c, "preClose": prev_c,
                "pctChange": round(pct, 2), "amountYi": round(amt_yi, 2),
                "buyerDom": round(bdom, 1)
            }

    for sym, dt, dtm, o, h, l, c, v in rows_60m:
        e = indexed[sym]
        e["m60_tuples"].append((dtm, o, h, l, c, v))
        e["m60_dates"].append(dt)

    for sym, dt, dtm, o, h, l, c, v in rows_15m:
        e = indexed[sym]
        e["m15_tuples"].append((dtm, o, h, l, c, v))
        e["m15_dates"].append(dt)

    print(f"DuckDB full dataset indexed in {time.time() - t0:.2f}s!")
    return indexed, full_meta_map, hs_pool, cyb_pool, kcb_pool, all_a_pool

def get_pit_bars_slice(idx_e: dict, dt: str):
    """
    严格 PIT 窗口截断 (WHERE date <= dt):
    获取截至交易日 dt 的前 250 根日线、前 280 根 60m K线、前 280 根 15m K线；
    若某标的(如sh600519) 15m/60m 与 1d 复权源不同，按当日收盘价动态对齐比例，防止跨表复权错位误触发止损。
    """
    i_d = bisect.bisect_right(idx_e["day_dates"], dt)
    i_60 = bisect.bisect_right(idx_e["m60_dates"], dt)
    i_15 = bisect.bisect_right(idx_e["m15_dates"], dt)
    d_bars = idx_e["day"][max(0, i_d - 250):i_d]
    raw_60 = idx_e["m60_tuples"][max(0, i_60 - 280):i_60]
    raw_15 = idx_e["m15_tuples"][max(0, i_15 - 280):i_15]
    if not d_bars or not raw_15:
        return d_bars, [], []

    d_close = d_bars[-1]["close"]
    m15_close = raw_15[-1][4]
    ratio = (d_close / m15_close) if (m15_close > 0 and abs(d_close - m15_close) / d_close > 0.005) else 1.0

    if ratio == 1.0:
        m60_bars = [{"time": t[0], "open": t[1], "high": t[2], "low": t[3], "close": t[4], "volume": t[5]} for t in raw_60]
        m15_bars = [{"time": t[0], "open": t[1], "high": t[2], "low": t[3], "close": t[4], "volume": t[5]} for t in raw_15]
    else:
        m60_bars = [{"time": t[0], "open": t[1]*ratio, "high": t[2]*ratio, "low": t[3]*ratio, "close": t[4]*ratio, "volume": t[5]} for t in raw_60]
        m15_bars = [{"time": t[0], "open": t[1]*ratio, "high": t[2]*ratio, "low": t[3]*ratio, "close": t[4]*ratio, "volume": t[5]} for t in raw_15]

    return d_bars, m60_bars, m15_bars
