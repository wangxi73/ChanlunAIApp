#!/usr/bin/env python3
"""
全A股 K线数据库 (Lexar-2T) 每日盘后通达信 (TDX) 增量更新脚本
数据源：17 节点国泰君安通达信官方行情主站集群 (shtdx.gtjas.com / sztdx.gtjas.com)
更新内容：
  1. 日线 (1d)   : 从通达信 (cat=9) 拉取最新日线 (vol*100 统一为股, amount 为元)
  2. 分钟线 (5m) : 从通达信 (cat=0) 直接拉取每日 48 根交易所真实 5m K线 (tdx_real_5m)
  3. 15m/30m/60m : 由通达信真实 5m 按 3/6/12 根无损聚合 (tdx_real_15m / tdx_real_30m / tdx_real_60m)
  4. 复权与除权保护：严格保持 unified_ratio_qfq 标准；若发生除权除息，原地缩放历史价格与 ratio_factor，100% 保留历史真实分钟线
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pytdx.hq import TdxHq_API

from fetch_10y_minute_archive import ARCHIVE_DIR, init_lexar_duckdb, normalize_symbol
from unify_lexar_kline_db import fetch_sina_hfq_factors
from upgrade_real_minutes_tdx import TDX_IPS


def get_tdx_latest_trading_date() -> str:
    """从通达信主站探测最新已收盘或进行中的交易日 YYYY-MM-DD"""
    for ip in TDX_IPS[:5]:
        api = TdxHq_API()
        try:
            if api.connect(ip, 7709, time_out=2.0):
                bars = api.get_index_bars(9, 1, "000001", 0, 2)
                api.disconnect()
                if bars:
                    return str(bars[-1]["datetime"])[:10]
        except Exception:
            pass
    return time.strftime("%Y-%m-%d")


def _agg_from_5m_tdx(df_5m: pd.DataFrame, step: int, freq_label: str, src_label: str) -> pd.DataFrame:
    n = (len(df_5m) // step) * step
    if n == 0:
        return pd.DataFrame()
    idx_last = np.arange(step - 1, n, step)
    return pd.DataFrame({
        "symbol": df_5m["symbol"].iloc[0], "freq": freq_label,
        "date": df_5m["date"].to_numpy()[idx_last], "datetime": df_5m["datetime"].to_numpy()[idx_last],
        "open": df_5m["open"].to_numpy()[:n].reshape(-1, step)[:, 0],
        "high": df_5m["high"].to_numpy()[:n].reshape(-1, step).max(axis=1),
        "low": df_5m["low"].to_numpy()[:n].reshape(-1, step).min(axis=1),
        "close": df_5m["close"].to_numpy()[:n].reshape(-1, step)[:, -1],
        "volume": np.round(df_5m["volume"].to_numpy()[:n].reshape(-1, step).sum(axis=1), 2),
        "amount": np.round(df_5m["amount"].to_numpy()[:n].reshape(-1, step).sum(axis=1), 2),
        "hfq_factor": df_5m["hfq_factor"].to_numpy()[idx_last],
        "ratio_factor": df_5m["ratio_factor"].to_numpy()[idx_last],
        "source": src_label,
    })


def _rescale_history_in_place(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    if abs(scale - 1.0) < 1e-6 or df.empty:
        return df
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        df[col] = np.round(df[col] * scale, 4)
    df["ratio_factor"] = np.round(df["ratio_factor"] * scale, 6)
    return df


def update_symbol_from_tdx(api: TdxHq_API, symbol: str, target_date: str, check_dividend: bool = False, force: bool = False) -> str:
    std_sym = normalize_symbol(symbol)[0]
    if std_sym.startswith("bj"):
        return "skipped_bj"
    p_1d = ARCHIVE_DIR / "1d" / f"{std_sym}.parquet"
    if not p_1d.exists():
        return "missing_base"

    df_1d = pd.read_parquet(p_1d)
    if df_1d.empty:
        return "missing_base"
    last_dt = str(df_1d["date"].max())
    if last_dt >= target_date and not force:
        return "up_to_date"

    mkt = 1 if std_sym.startswith("sh") else 0
    code = std_sym[2:]
    is_idx = std_sym.startswith(("sh000", "sz399"))
    fn = api.get_index_bars if is_idx else api.get_security_bars

    d_bars = fn(9, mkt, code, 0, 15)
    if not d_bars:
        return "no_tdx_data"
    new_d = [b for b in d_bars if str(b["datetime"])[:10] > last_dt or (force and str(b["datetime"])[:10] == last_dt)]
    if not new_d:
        return "up_to_date"

    n_new_days = len(new_d)
    m5_count = min(800, max(96, (n_new_days + 1) * 48))
    m5_bars = fn(0, mkt, code, 0, m5_count) or []

    old_last_hfq = float(df_1d["hfq_factor"].iloc[-1])
    new_Dates = [str(b["datetime"])[:10] for b in new_d]
    hfq_map = {dt: old_last_hfq for dt in new_Dates}
    latest_hfq = old_last_hfq

    if check_dividend and not is_idx:
        f_dates, f_vals = fetch_sina_hfq_factors(std_sym)
        if f_dates and f_vals and f_dates[-1] > last_dt:
            for dt in new_Dates:
                idx = max(0, int(np.searchsorted(np.array(f_dates, dtype=str), dt, side="right")) - 1)
                hfq_map[dt] = float(f_vals[idx])
            latest_hfq = hfq_map[new_Dates[-1]]

    hist_scale = (old_last_hfq / latest_hfq) if latest_hfq > 0 else 1.0
    cutoff_dt = new_Dates[0]

    d_recs = []
    for b in new_d:
        dt = str(b["datetime"])[:10]
        o, h, l, c = float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"])
        if c <= 0 or l <= 0:
            continue
        hfq = hfq_map.get(dt, latest_hfq)
        rf = round(hfq / latest_hfq, 6)
        vol = round(float(b.get("vol", 0)) * 100.0, 2)
        amt = round(float(b.get("amount", 0)), 2)
        d_recs.append((std_sym, "1d", dt, dt, round(o * rf, 4), round(h * rf, 4), round(l * rf, 4), round(c * rf, 4), vol, amt, round(hfq, 6), rf, "unified_ratio_qfq"))

    if not d_recs:
        return "up_to_date"

    df_new_1d = pd.DataFrame(d_recs, columns=df_1d.columns)
    base_1d = _rescale_history_in_place(df_1d[df_1d["date"] < cutoff_dt], hist_scale)
    merged_1d = pd.concat([base_1d, df_new_1d], ignore_index=True).sort_values("date").reset_index(drop=True)
    tmp_1d = p_1d.with_suffix(".parquet.tmp")
    merged_1d.to_parquet(tmp_1d, index=False, compression="zstd")
    tmp_1d.replace(p_1d)

    m5_recs = []
    new_dt_set = {r[2] for r in d_recs}
    for b in m5_bars:
        dt_m = str(b["datetime"])[:16]
        dt_d = dt_m[:10]
        if dt_d not in new_dt_set:
            continue
        o, h, l, c = float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"])
        if c <= 0 or l <= 0:
            continue
        hfq = hfq_map.get(dt_d, latest_hfq)
        rf = round(hfq / latest_hfq, 6)
        vol = round(float(b.get("vol", 0)), 2)
        amt = round(float(b.get("amount", 0)), 2)
        m5_recs.append((std_sym, "5m", dt_d, dt_m, round(o * rf, 4), round(h * rf, 4), round(l * rf, 4), round(c * rf, 4), vol, amt, round(hfq, 6), rf, "tdx_real_5m"))

    df_new_5m = pd.DataFrame(m5_recs, columns=df_1d.columns).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    df_new_5m = df_new_5m[df_new_5m.groupby("date")["datetime"].transform("count") == 48].reset_index(drop=True)
    if df_new_5m.empty:
        return "updated_1d_only"

    for freq, df_add in [
        ("5m", df_new_5m),
        ("15m", _agg_from_5m_tdx(df_new_5m, 3, "15m", "tdx_real_15m")),
        ("30m", _agg_from_5m_tdx(df_new_5m, 6, "30m", "tdx_real_30m")),
        ("60m", _agg_from_5m_tdx(df_new_5m, 12, "60m", "tdx_real_60m")),
    ]:
        pf = ARCHIVE_DIR / freq / f"{std_sym}.parquet"
        if not pf.exists():
            continue
        old_f = pd.read_parquet(pf)
        base_f = _rescale_history_in_place(old_f[old_f["date"] < cutoff_dt], hist_scale)
        merged_f = pd.concat([base_f, df_add], ignore_index=True).sort_values("datetime").reset_index(drop=True)
        tmp_f = pf.with_suffix(".parquet.tmp")
        merged_f.to_parquet(tmp_f, index=False, compression="zstd")
        tmp_f.replace(pf)
    return "updated"


def _tdx_daily_worker(ip: str, sym_chunk: List[str], target_date: str, check_div: bool, force: bool) -> Dict[str, int]:
    api = TdxHq_API(auto_retry=True)
    if not api.connect(ip, 7709, time_out=2.5):
        for alt_ip in TDX_IPS:
            if api.connect(alt_ip, 7709, time_out=2.5):
                break
    stats: Dict[str, int] = {}
    for s in sym_chunk:
        try:
            st = update_symbol_from_tdx(api, s, target_date, check_dividend=check_div, force=force)
        except Exception:
            try:
                api.disconnect()
            except Exception:
                pass
            api.connect(ip, 7709, time_out=2.5)
            st = "err"
        stats[st] = stats.get(st, 0) + 1
    try:
        api.disconnect()
    except Exception:
        pass
    return stats


def main():
    parser = argparse.ArgumentParser(description="每日从通达信(TDX)更新全A股最新日线与 5m/15m/30m/60m 分钟线")
    parser.add_argument("--symbols", nargs="+", default=["all_a"], help="all_a(全市场) / core(核心池) / 指定代码")
    parser.add_argument("--target-date", default="auto")
    parser.add_argument("--check-dividend", action="store_true", help="同步检查除权除息因子并缩放历史")
    parser.add_argument("--force", action="store_true", help="强制重刷目标日期的日线与分钟线")
    args = parser.parse_args()

    target_date = get_tdx_latest_trading_date() if args.target_date == "auto" else args.target_date
    meta_df = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
    if args.symbols == ["all_a"]:
        symbols = meta_df["symbol"].tolist()
    elif args.symbols == ["core"]:
        symbols = meta_df[meta_df["in_core"]]["symbol"].tolist()
    else:
        symbols = args.symbols

    t0 = time.time()
    n_w = min(len(TDX_IPS), max(1, len(symbols)))
    chunks = [symbols[i::n_w] for i in range(n_w)]
    print(f"Start TDX daily & minute update -> target_date={target_date}, symbols={len(symbols)}, servers={n_w}", flush=True)

    tot_stats: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=n_w) as ex:
        futs = [ex.submit(_tdx_daily_worker, TDX_IPS[i], chunks[i], target_date, args.check_dividend, args.force) for i in range(n_w)]
        for fut in as_completed(futs):
            for k, v in fut.result().items():
                tot_stats[k] = tot_stats.get(k, 0) + v

    init_lexar_duckdb()
    el = time.time() - t0
    print(f"TDX daily & minute update completed in {el:.1f}s ({len(symbols)/max(el,0.1):.1f} sym/s) -> {tot_stats}", flush=True)


if __name__ == "__main__":
    main()
