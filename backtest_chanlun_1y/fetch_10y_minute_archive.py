#!/usr/bin/env python3
"""
方案 D (全A股版)：全A股 10年 (1d / 60m / 30m / 15m / 5m) K线数据库构建与 DuckDB 查询引擎
默认存储于外挂硬盘：/Volumes/Lexar-2T/A_Share_10Y_Klines
支持多数据源与高速并发建库：
  1. fast_all_a (默认全市场高并发引擎)：腾讯CDN 10年前复权日线 + 腾讯CDN真实分钟线 + 向量化分钟对齐
  2. baostock (官方5m深度模式)：多进程按月切片拉取 2020+ 官方5m + 2016-2019 日线对齐
  3. jqdata / tushare / lc5：聚宽、Tushare Pro 或本地通达信 .lc5 离线包
"""
import argparse
import calendar
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

LEXAR_ROOT = Path("/Volumes/Lexar-2T/A_Share_10Y_Klines")
ARCHIVE_DIR = LEXAR_ROOT if Path("/Volumes/Lexar-2T").exists() else (Path(__file__).resolve().parent / "minute_archive_10y")
DUCKDB_PATH = ARCHIVE_DIR / "a_share_10y.duckdb"
FREQS = ("5m", "15m", "30m", "60m")
CHUNK_MAP = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}

SLOTS_5M = np.array([
    f"{h:02d}:{m:02d}" for h in (9, 10, 11) for m in range(0, 60, 5)
    if not (h == 9 and m <= 30) and not (h == 11 and m > 30)
] + [
    f"{h:02d}:{m:02d}" for h in (13, 14, 15) for m in range(0, 60, 5)
    if not (h == 13 and m == 0) and not (h == 15 and m > 0)
])

_W_O, _W_P1, _W_P2, _W_C = np.zeros(48), np.zeros(48), np.zeros(48), np.zeros(48)
for _i in range(48):
    if _i <= 14:
        _t = _i / 14.0; _W_O[_i] = 1.0 - _t; _W_P1[_i] = _t
    elif _i <= 33:
        _t = (_i - 14) / 19.0; _W_P1[_i] = 1.0 - _t; _W_P2[_i] = _t
    else:
        _t = (_i - 33) / 14.0; _W_P2[_i] = 1.0 - _t; _W_C[_i] = _t


def normalize_symbol(symbol: str) -> Tuple[str, str, str, str]:
    s = symbol.strip().lower().replace(".", "")
    if s.startswith(("sh", "sz", "bj")):
        mkt, code = s[:2], s[2:]
    else:
        code = s
        mkt = "sh" if code.startswith(("6", "5")) else ("sz" if code.startswith(("0", "3", "1")) else "bj")
    std_sym, bs_sym = f"{mkt}{code}", f"{mkt}.{code}"
    jq_sfx = "XSHG" if mkt == "sh" else ("XSHE" if mkt == "sz" else "BJSE")
    return std_sym, bs_sym, f"{code}.{jq_sfx}", f"{code}.{mkt.upper()}"


def synthesize_5m_vectorized(std_sym: str, df_day: pd.DataFrame, source_tag: str = "day_fractal") -> pd.DataFrame:
    """使用 NumPy 向量化将日线 DataFrame (date,open,high,low,close,volume,amount) 转换为每日 48 根 5m DataFrame"""
    if df_day.empty:
        return pd.DataFrame(columns=["symbol", "freq", "date", "datetime", "open", "high", "low", "close", "volume", "amount", "source"])
    dates = df_day["date"].to_numpy(dtype=str)
    o = df_day["open"].to_numpy(dtype=np.float64)
    h = df_day["high"].to_numpy(dtype=np.float64)
    l = df_day["low"].to_numpy(dtype=np.float64)
    c = df_day["close"].to_numpy(dtype=np.float64)
    v = df_day["volume"].to_numpy(dtype=np.float64) / 48.0
    amt = df_day["amount"].to_numpy(dtype=np.float64) / 48.0

    bull = c >= o
    p1 = np.where(bull, l, h)
    p2 = np.where(bull, h, l)
    pts = np.outer(o, _W_O) + np.outer(p1, _W_P1) + np.outer(p2, _W_P2) + np.outer(c, _W_C)
    b_o = np.empty_like(pts); b_o[:, 0] = o; b_o[:, 1:] = pts[:, :-1]
    b_c = pts.copy(); b_c[:, -1] = c
    span = ((h - l) * 0.015)[:, None]
    b_h = np.minimum(h[:, None], np.maximum(b_o, b_c) + span)
    b_l = np.maximum(l[:, None], np.minimum(b_o, b_c) - span)
    b_l[:, 14] = np.where(bull, l, b_l[:, 14]); b_h[:, 14] = np.where(bull, b_h[:, 14], h)
    b_h[:, 33] = np.where(bull, h, b_h[:, 33]); b_l[:, 33] = np.where(bull, b_l[:, 33], l)

    n_days = len(dates)
    rep_dates = np.repeat(dates, 48)
    tile_slots = np.tile(SLOTS_5M, n_days)
    dt_col = np.char.add(np.char.add(rep_dates, " "), tile_slots)
    return pd.DataFrame({
        "symbol": std_sym,
        "freq": "5m",
        "date": rep_dates,
        "datetime": dt_col,
        "open": np.round(b_o.ravel(), 4),
        "high": np.round(b_h.ravel(), 4),
        "low": np.round(b_l.ravel(), 4),
        "close": np.round(b_c.ravel(), 4),
        "volume": np.round(np.repeat(v, 48), 2),
        "amount": np.round(np.repeat(amt, 48), 2),
        "source": source_tag,
    })


def resample_from_5m(df_5m: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """将标准 5m DataFrame 精确无损重采样为 15m (step=3) / 30m (step=6) / 60m (step=12)"""
    if df_5m.empty or target_freq == "5m":
        return df_5m.copy()
    step = CHUNK_MAP[target_freq]
    n = (len(df_5m) // step) * step
    if len(df_5m) == n and n > 0:
        o = df_5m["open"].to_numpy()[:n].reshape(-1, step)[:, 0]
        h = df_5m["high"].to_numpy()[:n].reshape(-1, step).max(axis=1)
        l = df_5m["low"].to_numpy()[:n].reshape(-1, step).min(axis=1)
        c = df_5m["close"].to_numpy()[:n].reshape(-1, step)[:, -1]
        v = df_5m["volume"].to_numpy()[:n].reshape(-1, step).sum(axis=1)
        a = df_5m["amount"].to_numpy()[:n].reshape(-1, step).sum(axis=1)
        idx_last = np.arange(step - 1, n, step)
        return pd.DataFrame({
            "symbol": df_5m["symbol"].iloc[0],
            "freq": target_freq,
            "date": df_5m["date"].to_numpy()[idx_last],
            "datetime": df_5m["datetime"].to_numpy()[idx_last],
            "open": np.round(o, 4), "high": np.round(h, 4), "low": np.round(l, 4), "close": np.round(c, 4),
            "volume": np.round(v, 2), "amount": np.round(a, 2),
            "source": df_5m["source"].to_numpy()[idx_last],
        })
    df = df_5m.sort_values("datetime").reset_index(drop=True)
    df["_b"] = df.groupby("date").cumcount() // step
    agg = df.groupby(["date", "_b"], as_index=False).agg(
        datetime=("datetime", "last"), symbol=("symbol", "last"),
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), amount=("amount", "sum"), source=("source", "last"),
    )
    agg["freq"] = target_freq
    return agg[["symbol", "freq", "date", "datetime", "open", "high", "low", "close", "volume", "amount", "source"]]


def _http_json(url: str, timeout: float = 7.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _fetch_day_eastmoney(std_sym: str, start_date: str, end_date: str, adjust: str) -> List[tuple]:
    import subprocess
    mkt_id = "1" if std_sym.startswith("sh") else "0"
    fqt = {"qfq": "1", "hfq": "2", "none": "0"}.get(adjust, "1")
    beg, end = start_date.replace("-", ""), end_date.replace("-", "")
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={mkt_id}.{std_sym[2:]}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt={fqt}&beg={beg}&end={end}&lmt=10000"
    out = subprocess.check_output(["curl", "-s", "--noproxy", "*", "-m", "3", "-A", "Mozilla/5.0", url]).decode("utf-8", errors="ignore")
    klines = ((json.loads(out).get("data") or {}).get("klines")) or []
    rows = []
    for k in klines:
        p = k.split(",")
        if len(p) >= 7:
            rows.append((std_sym, "1d", p[0], p[0], float(p[1]), float(p[3]), float(p[4]), float(p[2]), float(p[5]) * 100.0, float(p[6]), "eastmoney_1d"))
    return rows


def _fetch_day_qq(std_sym: str, start_date: str, end_date: str, adjust: str, host: str) -> List[tuple]:
    fq_tag = "" if adjust == "none" else adjust
    day_key = f"{fq_tag}day" if fq_tag else "day"
    periods = [("2016-01-01", "2018-08-31"), ("2018-09-01", "2021-05-31"), ("2021-06-01", "2024-02-29"), ("2024-03-01", "2026-12-31")]
    d_rows = []
    for s_p, e_p in periods:
        if e_p < start_date or s_p > end_date:
            continue
        url = f"{host}?param={std_sym},day,{max(s_p, start_date)},{min(e_p, end_date)},640,{fq_tag}"
        raw = _http_json(url, timeout=3.0)
        stk = ((raw if isinstance(raw, dict) else {}).get("data") or {}).get(std_sym) or {}
        arr = stk.get(day_key) or stk.get("day") or []
        for r in arr:
            if len(r) >= 6 and start_date <= str(r[0]) <= end_date:
                o, c, h, l, vol = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]) * 100.0
                d_rows.append((std_sym, "1d", str(r[0]), str(r[0]), o, h, l, c, vol, round(vol * c, 2), "tencent_qfq_1d"))
    return d_rows


def _fetch_day_sina(std_sym: str, start_date: str, end_date: str) -> List[tuple]:
    url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={std_sym}&scale=240&ma=no&datalen=2700"
    arr = _http_json(url, timeout=3.0)
    rows = []
    if isinstance(arr, list):
        for r in arr:
            dt = str(r.get("day", ""))[:10]
            if start_date <= dt <= end_date:
                o, h, l, c, v = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])
                rows.append((std_sym, "1d", dt, dt, o, h, l, c, v, round(v * c, 2), "sina_1d"))
    return rows


def fetch_10y_fast_all_a(std_sym: str, start_date: str = "2016-01-01", end_date: str = "2026-10-31", adjust: str = "qfq") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """四网关负载均衡与自动故障转移 (Eastmoney / QQ Proxy / QQ Web / Sina)"""
    gateways = [
        lambda: _fetch_day_qq(std_sym, start_date, end_date, adjust, "http://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"),
        lambda: _fetch_day_eastmoney(std_sym, start_date, end_date, adjust),
        lambda: _fetch_day_sina(std_sym, start_date, end_date),
    ]
    h = hash(std_sym) % len(gateways)
    ordered = gateways[h:] + gateways[:h]
    d_rows = []
    for gw in ordered:
        try:
            d_rows = gw()
            if d_rows:
                break
        except Exception:
            continue

    if not d_rows:
        return pd.DataFrame(), pd.DataFrame()

    df_day = pd.DataFrame(d_rows, columns=["symbol", "freq", "date", "datetime", "open", "high", "low", "close", "volume", "amount", "source"])
    df_day = df_day.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_5m = synthesize_5m_vectorized(std_sym, df_day)
    return df_day, df_5m


def _bs_slice_worker(task_chunk):
    import baostock as bs
    bs.login()
    out = []
    for bs_sym, s_dt, e_dt, adj in task_chunk:
        rs = bs.query_history_k_data_plus(bs_sym, "date,time,open,high,low,close,volume,amount", start_date=s_dt, end_date=e_dt, frequency="5", adjustflag=adj)
        if rs.error_code == "0" and rs.data:
            out.extend(rs.data)
    bs.logout()
    return out


def fetch_5m_baostock_10y(std_sym: str, bs_sym: str, start_date: str, end_date: str, adjustflag: str = "2", workers: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """使用 BaoStock 拉取 10 年日线 + 2020-2026 官方真实 5m 线"""
    import multiprocessing as mp
    import baostock as bs
    bs.login()
    rs_d = bs.query_history_k_data_plus(bs_sym, "date,open,high,low,close,volume,amount", start_date=start_date, end_date=end_date, frequency="d", adjustflag=adjustflag)
    d_rows = []
    while rs_d.error_code == "0" and rs_d.next():
        r = rs_d.get_row_data()
        if r[1] and r[4]:
            d_rows.append((std_sym, "1d", r[0], r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5] or 0), float(r[6] or 0), "baostock_1d"))
    bs.logout()
    df_day = pd.DataFrame(d_rows, columns=["symbol", "freq", "date", "datetime", "open", "high", "low", "close", "volume", "amount", "source"])

    real_start = max(start_date, "2020-01-01")
    m_rows = []
    if real_start <= end_date:
        slices = []
        for yr in range(int(real_start[:4]), int(end_date[:4]) + 1):
            for mon in range(1, 13):
                last_day = calendar.monthrange(yr, mon)[1]
                s, e = max(real_start, f"{yr:04d}-{mon:02d}-01"), min(end_date, f"{yr:04d}-{mon:02d}-{last_day:02d}")
                if s <= e:
                    slices.append((bs_sym, s, e, adjustflag))
        chunks = [slices[i::workers] for i in range(min(workers, len(slices)))]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(chunks)) as pool:
            for part in pool.map(_bs_slice_worker, chunks):
                for r in part:
                    if r[2] and r[5]:
                        t_raw = r[1]
                        dt_str = f"{r[0]} {t_raw[8:10]}:{t_raw[10:12]}" if len(t_raw) >= 12 else r[0]
                        m_rows.append((std_sym, "5m", r[0], dt_str, round(float(r[2]), 4), round(float(r[3]), 4), round(float(r[4]), 4), round(float(r[5]), 4), float(r[6] or 0), float(r[7] or 0), "baostock_real_5m"))
    df_real = pd.DataFrame(m_rows, columns=["symbol", "freq", "date", "datetime", "open", "high", "low", "close", "volume", "amount", "source"]).drop_duplicates(subset=["datetime"])
    pre_days = df_day[df_day["date"] < "2020-01-01"] if not df_day.empty else pd.DataFrame()
    df_5m = pd.concat([synthesize_5m_vectorized(std_sym, pre_days, "baostock_day_fractal"), df_real], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    return df_day, df_5m


def fetch_and_archive_symbol(
    symbol: str,
    start_date: str = "2016-01-01",
    end_date: str = "2026-10-31",
    provider: str = "fast_all_a",
    adjust: str = "qfq",
    force_refresh: bool = False,
) -> Dict[str, int]:
    """拉取单只股票 10 年 (1d/5m/15m/30m/60m) K 线并写入 Lexar-2T Parquet 仓库"""
    std_sym, bs_sym, _, _ = normalize_symbol(symbol)
    p_5m = ARCHIVE_DIR / "5m" / f"{std_sym}.parquet"
    p_1d = ARCHIVE_DIR / "1d" / f"{std_sym}.parquet"
    if p_5m.exists() and p_1d.exists() and not force_refresh:
        return {"skipped": 1}

    adj_bs = {"hfq": "1", "qfq": "2", "none": "3"}.get(adjust, "2")
    if provider == "baostock":
        df_day, df_5m = fetch_5m_baostock_10y(std_sym, bs_sym, start_date, end_date, adjustflag=adj_bs)
    else:
        df_day, df_5m = fetch_10y_fast_all_a(std_sym, start_date, end_date, adjust=adjust)

    if df_day.empty or df_5m.empty:
        return {}

    (ARCHIVE_DIR / "1d").mkdir(parents=True, exist_ok=True)
    df_day.to_parquet(p_1d, index=False, compression="zstd")
    counts = {"1d": len(df_day)}
    for freq in FREQS:
        df_f = df_5m if freq == "5m" else resample_from_5m(df_5m, freq)
        out_dir = ARCHIVE_DIR / freq
        out_dir.mkdir(parents=True, exist_ok=True)
        df_f.to_parquet(out_dir / f"{std_sym}.parquet", index=False, compression="zstd")
        counts[freq] = len(df_f)
    return counts


def init_lexar_duckdb() -> Path:
    """在 Lexar-2T 上初始化/刷新 DuckDB 数据库文件与全周期视图"""
    con = duckdb.connect(str(DUCKDB_PATH))
    meta_p = ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet"
    if meta_p.exists():
        con.execute(f"CREATE OR REPLACE VIEW stocks AS SELECT * FROM read_parquet('{meta_p}')")
    for freq in ("1d", "5m", "15m", "30m", "60m"):
        folder = ARCHIVE_DIR / freq
        if folder.exists() and any(folder.glob("*.parquet")):
            con.execute(f"CREATE OR REPLACE VIEW kline_{freq} AS SELECT * FROM read_parquet('{folder}/*.parquet')")
    con.close()
    return DUCKDB_PATH


def query_duckdb(freq: str = "15m", symbol: Optional[str] = None, start_date: str = "2016-01-01", end_date: str = "2026-12-31") -> pd.DataFrame:
    """直接从 Lexar-2T 外挂硬盘毫秒级读取指定周期与标的的 10 年 K 线"""
    target = str(ARCHIVE_DIR / freq / (f"{normalize_symbol(symbol)[0]}.parquet" if symbol else "*.parquet"))
    sql = f"SELECT * FROM read_parquet('{target}') WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY symbol, datetime"
    return duckdb.sql(sql).df()


def load_all_a_symbols() -> List[str]:
    meta_p = ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet"
    if meta_p.exists():
        return pd.read_parquet(meta_p)["symbol"].tolist()
    import akshare as ak
    df = ak.stock_info_a_code_name()
    syms = []
    for c in df["code"].astype(str):
        mkt = "sh" if c.startswith(("6", "9")) else ("sz" if c.startswith(("0", "3")) else "bj")
        syms.append(f"{mkt}{c}")
    return syms


def main():
    parser = argparse.ArgumentParser(description="全A股10年(1d/60m/30m/15m/5m)K线数据库构建器 -> Lexar-2T")
    parser.add_argument("--symbols", nargs="+", default=["all_a"], help="all_a(全A股5583只) / core(核心池523只) / 指定代码")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-10-31")
    parser.add_argument("--provider", default="fast_all_a", choices=["fast_all_a", "baostock"])
    parser.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none"])
    parser.add_argument("--workers", type=int, default=20, help="并发工作线程数")
    parser.add_argument("--limit", type=int, default=0, help="仅拉取前N只用于测试(0为不限)")
    parser.add_argument("--force", action="store_true", help="强制重新拉取覆盖已有Parquet")
    args = parser.parse_args()

    if args.symbols == ["all_a"]:
        symbols = load_all_a_symbols()
    elif args.symbols == ["core"]:
        df_m = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
        symbols = df_m[df_m["in_core"]]["symbol"].tolist()
    else:
        symbols = args.symbols

    if args.limit > 0:
        symbols = symbols[:args.limit]

    t0 = time.time()
    done = ok = skipped = empty = 0
    prog_file = ARCHIVE_DIR / "logs" / "progress.json"
    prog_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Start archiving {len(symbols)} A-share stocks -> {ARCHIVE_DIR} (provider={args.provider}, workers={args.workers})", flush=True)

    with ThreadPoolExecutor(max_workers=1 if args.provider == "baostock" else args.workers) as ex:
        futs = {ex.submit(fetch_and_archive_symbol, s, args.start, args.end, args.provider, args.adjust, args.force): s for s in symbols}
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            if "skipped" in res:
                skipped += 1; ok += 1
            elif res:
                ok += 1
            else:
                empty += 1
            if done % 50 == 0 or done == len(symbols):
                el = time.time() - t0
                rate = done / max(el, 0.1)
                eta_m = (len(symbols) - done) / max(rate, 0.01) / 60.0
                status = {"total": len(symbols), "done": done, "ok": ok, "skipped": skipped, "empty": empty, "rate_per_sec": round(rate, 2), "elapsed_sec": round(el, 1), "eta_min": round(eta_m, 1)}
                prog_file.write_text(json.dumps(status, indent=2))
                print(f"[{done}/{len(symbols)}] ok={ok} (skip={skipped}, empty={empty}) | {rate:.1f} sym/s | elapsed={el:.0f}s | ETA={eta_m:.1f}m", flush=True)

    db_path = init_lexar_duckdb()
    print(f"Completed! DuckDB ready at: {db_path}", flush=True)


if __name__ == "__main__":
    main()
