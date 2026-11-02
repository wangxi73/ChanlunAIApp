#!/usr/bin/env python3
"""
全A股 10年 K线数据库 (Lexar-2T) 口径100%统一化与等比前复权/后复权双因子升级脚本
统一标准：
  1. 原始底表：100% 采用交易所不复权真实日线 (Raw OHLCV) + 新浪16位高精度等比后复权因子表 (hfq.js)
  2. 默认价格列 (open/high/low/close)：统一为【等比前复权 Ratio-QFQ】
     - 恒正 (>0)，彻底消灭茅台等高分红白马股算术前复权变负数的问题
     - 任意两日价格比值 100% 等于真实含权复权收益率
     - 最新一日价格 100% 等于当前实盘真实股价
  3. 附带因子列：同时保存 `hfq_factor` (累计后复权因子) 与 `ratio_factor` (等比前复权因子)
     - 真实不复权价 = close / ratio_factor
     - 标准后复权价(HFQ) = close / ratio_factor * hfq_factor
  4. 来源标签 (source)：全市场 5583 只股票、5 个周期 (1d/60m/30m/15m/5m) 100% 统一为 `unified_ratio_qfq`
"""
import json
import pickle
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

from fetch_10y_minute_archive import ARCHIVE_DIR, FREQS, SLOTS_5M, _W_C, _W_O, _W_P1, _W_P2, init_lexar_duckdb, resample_from_5m

LOCAL_CACHE_10Y = Path(__file__).resolve().parent / "cache_10y_hfq"


def fetch_sina_hfq_factors(std_sym: str) -> Tuple[List[str], List[float]]:
    """获取单只股票的新浪 16 位高精度等比后复权因子 (date_asc, factor_asc)"""
    if std_sym.startswith(("sh000", "sh51", "sh58", "sz15", "bj")):
        return [], []
    cf = LOCAL_CACHE_10Y / f"{std_sym}.pkl"
    if cf.exists():
        try:
            d = pickle.load(open(cf, "rb"))
            if d.get("f_dates") and d.get("f_vals"):
                return d["f_dates"], d["f_vals"]
        except Exception:
            pass
    try:
        url = f"https://finance.sina.com.cn/realstock/company/{std_sym}/hfq.js"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=4.0).read().decode("gbk", errors="ignore")
        m = re.search(r"\"data\":(\[.*?\])", txt)
        if not m:
            return [], []
        arr = json.loads(m.group(1))
        arr = [x for x in arr if float(x.get("f", 0)) > 0]
        arr.sort(key=lambda x: x["d"])
        return [str(x["d"])[:10] for x in arr], [float(x["f"]) for x in arr]
    except Exception:
        return [], []


def _fetch_raw_day_network(std_sym: str, start_date: str = "2016-01-01", end_date: str = "2026-10-31") -> pd.DataFrame:
    """从腾讯不复权网关 / 东财不复权网关拉取 10 年真实不复权日线"""
    rows = []
    if True:
        periods = [("2016-01-01", "2018-08-31"), ("2018-09-01", "2021-05-31"), ("2021-06-01", "2024-02-29"), ("2024-03-01", "2026-12-31")]
        for s_p, e_p in periods:
            url = f"http://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?param={std_sym},day,{max(s_p, start_date)},{min(e_p, end_date)},640,"
            try:
                raw = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=4.0).read().decode("utf-8", errors="ignore"))
                arr = (((raw.get("data") or {}).get(std_sym)) or {}).get("day") or []
                for r in arr:
                    if len(r) >= 6 and float(r[2]) > 0:
                        o, c, h, l, v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]) * 100.0
                        rows.append((str(r[0]), o, h, l, c, v, round(v * c, 2)))
            except Exception:
                pass
    if not rows:
        mkt_id = "1" if std_sym.startswith("sh") else "0"
        url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={mkt_id}.{std_sym[2:]}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&beg=20160101&end=20261231&lmt=10000"
        try:
            out = subprocess.check_output(["curl", "-s", "--noproxy", "*", "-m", "4", "-A", "Mozilla/5.0", url]).decode("utf-8", errors="ignore")
            for k in ((json.loads(out).get("data") or {}).get("klines")) or []:
                p = k.split(",")
                if len(p) >= 7 and float(p[2]) > 0:
                    rows.append((p[0], float(p[1]), float(p[3]), float(p[4]), float(p[2]), float(p[5]) * 100.0, float(p[6])))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return df[(df["date"] >= start_date) & (df["date"] <= end_date) & (df["close"] > 0) & (df["low"] > 0)].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def get_raw_daily_df(std_sym: str) -> pd.DataFrame:
    """获取单只股票的不复权原始日线：优先复用 Lexar 中已存的 sina_1d (不复权) 或本地缓存"""
    p_1d = ARCHIVE_DIR / "1d" / f"{std_sym}.parquet"
    if p_1d.exists():
        try:
            df_ex = pd.read_parquet(p_1d)
            if not df_ex.empty and df_ex["source"].iloc[0] == "sina_1d":
                df_clean = df_ex[(df_ex["close"] > 0) & (df_ex["low"] > 0) & (df_ex["open"] > 0)].copy()
                if not df_clean.empty:
                    return df_clean[["date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
            if not df_ex.empty and df_ex["source"].iloc[0] == "unified_ratio_qfq" and "ratio_factor" in df_ex.columns:
                rf = df_ex["ratio_factor"].to_numpy()
                df_raw = df_ex[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                for col in ("open", "high", "low", "close"):
                    df_raw[col] = np.round(df_raw[col] / rf, 4)
                return df_raw
        except Exception:
            pass
    cf = LOCAL_CACHE_10Y / f"{std_sym}.pkl"
    if cf.exists():
        try:
            d = pickle.load(open(cf, "rb"))
            rd = d.get("raw_day") or []
            if rd and str(rd[-1]["time"]) >= "2026-10-28":
                df_c = pd.DataFrame(rd).rename(columns={"time": "date"})
                df_c["volume"] = df_c["volume"] * 100.0
                df_c["amount"] = np.round(df_c["volume"] * df_c["close"], 2)
                df_c = df_c[(df_c["date"] >= "2016-01-01") & (df_c["close"] > 0) & (df_c["low"] > 0)]
                if not df_c.empty:
                    return df_c[["date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        except Exception:
            pass
    return _fetch_raw_day_network(std_sym)


def unify_single_symbol(std_sym: str, force: bool = False) -> str:
    """将单只股票的 1d/5m/15m/30m/60m 统一转换为等比前复权 + 双复权因子表 (unified_ratio_qfq)"""
    p_1d = ARCHIVE_DIR / "1d" / f"{std_sym}.parquet"
    if p_1d.exists() and not force:
        try:
            df_chk = pd.read_parquet(p_1d, columns=["source", "low"])
            if not df_chk.empty and df_chk["source"].iloc[0] == "unified_ratio_qfq" and df_chk["low"].min() > 0:
                return "already_unified"
        except Exception:
            pass

    df_raw = get_raw_daily_df(std_sym)
    if df_raw.empty:
        return "empty"

    f_dates, f_vals = fetch_sina_hfq_factors(std_sym)
    dates = df_raw["date"].to_numpy(dtype=str)
    if f_dates and f_vals:
        idx = np.clip(np.searchsorted(np.array(f_dates, dtype=str), dates, side="right") - 1, 0, len(f_vals) - 1)
        hfq_f = np.array(f_vals, dtype=np.float64)[idx]
    else:
        hfq_f = np.ones(len(dates), dtype=np.float64)

    latest_f = hfq_f[-1] if hfq_f[-1] > 0 else 1.0
    ratio_f = hfq_f / latest_f

    o = np.round(df_raw["open"].to_numpy(dtype=np.float64) * ratio_f, 4)
    h = np.round(df_raw["high"].to_numpy(dtype=np.float64) * ratio_f, 4)
    l = np.round(df_raw["low"].to_numpy(dtype=np.float64) * ratio_f, 4)
    c = np.round(df_raw["close"].to_numpy(dtype=np.float64) * ratio_f, 4)
    v = df_raw["volume"].to_numpy(dtype=np.float64)
    amt = df_raw["amount"].to_numpy(dtype=np.float64)

    df_1d = pd.DataFrame({
        "symbol": std_sym, "freq": "1d", "date": dates, "datetime": dates,
        "open": o, "high": h, "low": l, "close": c,
        "volume": v, "amount": amt,
        "hfq_factor": np.round(hfq_f, 6), "ratio_factor": np.round(ratio_f, 6),
        "source": "unified_ratio_qfq",
    })
    df_1d.to_parquet(p_1d, index=False, compression="zstd")

    bull = c >= o
    p1, p2 = np.where(bull, l, h), np.where(bull, h, l)
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
    dt_col = np.char.add(np.char.add(rep_dates, " "), np.tile(SLOTS_5M, n_days))
    df_5m = pd.DataFrame({
        "symbol": std_sym, "freq": "5m", "date": rep_dates, "datetime": dt_col,
        "open": np.round(b_o.ravel(), 4), "high": np.round(b_h.ravel(), 4),
        "low": np.round(b_l.ravel(), 4), "close": np.round(b_c.ravel(), 4),
        "volume": np.round(np.repeat(v / 48.0, 48), 2), "amount": np.round(np.repeat(amt / 48.0, 48), 2),
        "hfq_factor": np.repeat(np.round(hfq_f, 6), 48), "ratio_factor": np.repeat(np.round(ratio_f, 6), 48),
        "source": "unified_ratio_qfq",
    })
    df_5m.to_parquet(ARCHIVE_DIR / "5m" / f"{std_sym}.parquet", index=False, compression="zstd")

    for freq, step in [("15m", 3), ("30m", 6), ("60m", 12)]:
        n = (len(df_5m) // step) * step
        idx_last = np.arange(step - 1, n, step)
        df_f = pd.DataFrame({
            "symbol": std_sym, "freq": freq,
            "date": df_5m["date"].to_numpy()[idx_last], "datetime": df_5m["datetime"].to_numpy()[idx_last],
            "open": df_5m["open"].to_numpy()[:n].reshape(-1, step)[:, 0],
            "high": df_5m["high"].to_numpy()[:n].reshape(-1, step).max(axis=1),
            "low": df_5m["low"].to_numpy()[:n].reshape(-1, step).min(axis=1),
            "close": df_5m["close"].to_numpy()[:n].reshape(-1, step)[:, -1],
            "volume": np.round(df_5m["volume"].to_numpy()[:n].reshape(-1, step).sum(axis=1), 2),
            "amount": np.round(df_5m["amount"].to_numpy()[:n].reshape(-1, step).sum(axis=1), 2),
            "hfq_factor": df_5m["hfq_factor"].to_numpy()[idx_last],
            "ratio_factor": df_5m["ratio_factor"].to_numpy()[idx_last],
            "source": "unified_ratio_qfq",
        })
        df_f.to_parquet(ARCHIVE_DIR / freq / f"{std_sym}.parquet", index=False, compression="zstd")
    return "unified"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="全A股 Lexar-2T 数据库统一化为等比前复权 + 双复权因子")
    parser.add_argument("--symbols", nargs="+", default=["all_a"])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.symbols == ["all_a"]:
        df_m = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
        symbols = df_m["symbol"].tolist()
    else:
        symbols = args.symbols

    t0 = time.time()
    stats = {"unified": 0, "already_unified": 0, "empty": 0}
    prog_p = ARCHIVE_DIR / "logs" / "unify_progress.json"
    print(f"Start unifying {len(symbols)} symbols -> unified_ratio_qfq (workers={args.workers})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(unify_single_symbol, s, args.force): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), 1):
            st = fut.result()
            stats[st] = stats.get(st, 0) + 1
            if i % 100 == 0 or i == len(symbols):
                el = time.time() - t0
                rate = i / max(el, 0.1)
                eta = (len(symbols) - i) / max(rate, 0.01) / 60.0
                prog_p.write_text(json.dumps({"total": len(symbols), "done": i, "stats": stats, "rate": round(rate, 2), "eta_min": round(eta, 1)}, indent=2))
                print(f"[{i}/{len(symbols)}] {stats} | {rate:.1f} sym/s | elapsed={el:.0f}s | ETA={eta:.1f}m", flush=True)

    init_lexar_duckdb()
    print(f"Unification complete in {time.time()-t0:.1f}s -> {stats}", flush=True)


if __name__ == "__main__":
    main()
