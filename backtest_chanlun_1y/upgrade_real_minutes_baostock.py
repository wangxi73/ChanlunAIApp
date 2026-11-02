#!/usr/bin/env python3
"""
从 BaoStock 单线程稳态拉取 2020-01 ~ 2024-10 (TDX真实分钟起点前) 的交易所真实分钟 K 线 (15m/30m/60m/5m)
并写入 /Volumes/Lexar-2T/A_Share_10Y_Klines
特性：
  1. 自动检测每只股票在 2020-01-01 之后仍为 unified_ratio_qfq 的精确日期范围 [need_min, need_max]
  2. 已是 baostock_real_* 或 2024-10 后上市的次新股自动 0 请求毫秒级跳过
  3. 严格单线程 + 切片间微延时 (0.12s)，防触发 BaoStock 10001011 黑名单
  4. 自动应用 ratio_factor 等比前复权 + 过滤停牌 0 价行 + .tmp 原子落盘
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from fetch_10y_minute_archive import ARCHIVE_DIR, SLOTS_5M, init_lexar_duckdb, normalize_symbol

SLICES_2020_2024 = [
    ("2020-01-01", "2020-06-30"), ("2020-07-01", "2020-12-31"),
    ("2021-01-01", "2021-06-30"), ("2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-10-31"),
]


def _agg_from_15m(df_15m: pd.DataFrame, step: int, freq_label: str, src_label: str) -> pd.DataFrame:
    n = (len(df_15m) // step) * step
    idx_last = np.arange(step - 1, n, step)
    return pd.DataFrame({
        "symbol": df_15m["symbol"].iloc[0], "freq": freq_label,
        "date": df_15m["date"].to_numpy()[idx_last], "datetime": df_15m["datetime"].to_numpy()[idx_last],
        "open": df_15m["open"].to_numpy()[:n].reshape(-1, step)[:, 0],
        "high": df_15m["high"].to_numpy()[:n].reshape(-1, step).max(axis=1),
        "low": df_15m["low"].to_numpy()[:n].reshape(-1, step).min(axis=1),
        "close": df_15m["close"].to_numpy()[:n].reshape(-1, step)[:, -1],
        "volume": np.round(df_15m["volume"].to_numpy()[:n].reshape(-1, step).sum(axis=1), 2),
        "amount": np.round(df_15m["amount"].to_numpy()[:n].reshape(-1, step).sum(axis=1), 2),
        "hfq_factor": df_15m["hfq_factor"].to_numpy()[idx_last],
        "ratio_factor": df_15m["ratio_factor"].to_numpy()[idx_last],
        "source": src_label,
    })


def _expand_15m_to_5m(df_15m: pd.DataFrame, std_sym: str) -> pd.DataFrame:
    n_days = len(df_15m) // 16
    o, h, l, c = df_15m["open"].to_numpy(), df_15m["high"].to_numpy(), df_15m["low"].to_numpy(), df_15m["close"].to_numpy()
    bull = c >= o
    m1, m2 = np.round(o + (c - o) / 3.0, 4), np.round(o + 2.0 * (c - o) / 3.0, 4)
    b_o = np.column_stack([o, m1, m2]).ravel()
    b_c = np.column_stack([m1, m2, c]).ravel()
    b_h = np.column_stack([np.where(bull, np.maximum(o, m1), h), np.where(bull, h, np.maximum(m1, m2)), np.maximum(m2, c)]).ravel()
    b_l = np.column_stack([np.where(bull, l, np.minimum(o, m1)), np.where(bull, np.minimum(m1, m2), l), np.minimum(m2, c)]).ravel()
    rep_dates = np.repeat(df_15m["date"].to_numpy(dtype=str), 3)
    dt_col = np.char.add(np.char.add(rep_dates, " "), np.tile(SLOTS_5M, n_days))
    return pd.DataFrame({
        "symbol": std_sym, "freq": "5m", "date": rep_dates, "datetime": dt_col,
        "open": b_o, "high": b_h, "low": b_l, "close": b_c,
        "volume": np.round(np.repeat(df_15m["volume"].to_numpy() / 3.0, 3), 2),
        "amount": np.round(np.repeat(df_15m["amount"].to_numpy() / 3.0, 3), 2),
        "hfq_factor": np.repeat(df_15m["hfq_factor"].to_numpy(), 3),
        "ratio_factor": np.repeat(df_15m["ratio_factor"].to_numpy(), 3),
        "source": "baostock_real15m_5m",
    })


def _safe_relogin() -> bool:
    import baostock as bs
    for wait_s in (2.0, 5.0, 10.0):
        try:
            bs.logout()
        except Exception:
            pass
        time.sleep(wait_s)
        try:
            lg = bs.login()
            if getattr(lg, "error_code", "") == "0":
                return True
        except Exception:
            pass
    return False


def upgrade_one_symbol_2020_2024(symbol: str, force: bool = False) -> Tuple[str, int]:
    import baostock as bs
    std_sym, bs_sym, _, _ = normalize_symbol(symbol)
    if std_sym.startswith(("bj", "sh000", "sh51", "sh58", "sz15", "sz399")):
        return "skipped_non_stock", 0
    p_1d = ARCHIVE_DIR / "1d" / f"{std_sym}.parquet"
    p_15m = ARCHIVE_DIR / "15m" / f"{std_sym}.parquet"
    if not p_1d.exists() or not p_15m.exists():
        return "missing_base", 0

    chk = pd.read_parquet(p_15m, columns=["date", "source"])
    need_synth = chk[(chk["date"] >= "2020-01-01") & (chk["date"] <= "2024-10-31") & (chk["source"] == "unified_ratio_qfq")]
    if need_synth.empty and not force:
        return "already_real", 0

    need_min = str(need_synth["date"].min()) if not need_synth.empty else "2020-01-01"
    need_max = str(need_synth["date"].max()) if not need_synth.empty else "2024-10-14"

    df_1d = pd.read_parquet(p_1d, columns=["date", "hfq_factor", "ratio_factor"])
    raw_rows = []
    for s_dt, e_dt in SLICES_2020_2024:
        if e_dt < need_min or s_dt > need_max:
            continue
        q_start, q_end = max(s_dt, need_min), min(e_dt, need_max)
        slice_ok = False
        for _ in range(4):
            try:
                rs = bs.query_history_k_data_plus(
                    bs_sym, "date,time,open,high,low,close,volume,amount",
                    start_date=q_start, end_date=q_end, frequency="15", adjustflag="3",
                )
                if rs.error_code == "0" and rs.data is not None:
                    raw_rows.extend(rs.data)
                    slice_ok = True
                    time.sleep(0.12)
                    break
            except Exception:
                pass
            if not _safe_relogin():
                return "bs_blocked", 0
        if not slice_ok:
            return "slice_incomplete", 0

    if not raw_rows:
        return "no_bs_data", 0

    f_map_hfq = dict(zip(df_1d["date"], df_1d["hfq_factor"]))
    f_map_rf = dict(zip(df_1d["date"], df_1d["ratio_factor"]))
    last_hfq, last_rf = float(df_1d["hfq_factor"].iloc[-1]), float(df_1d["ratio_factor"].iloc[-1])

    recs = []
    for r in raw_rows:
        if not r[2] or not r[4] or not r[5]:
            continue
        o_r, h_r, l_r, c_r = float(r[2]), float(r[3]), float(r[4]), float(r[5])
        if o_r <= 0 or l_r <= 0 or c_r <= 0:
            continue
        dt_d, t_raw = r[0], r[1]
        dt_m = f"{dt_d} {t_raw[8:10]}:{t_raw[10:12]}" if len(t_raw) >= 12 else dt_d
        rf, hfq = f_map_rf.get(dt_d, last_rf), f_map_hfq.get(dt_d, last_hfq)
        recs.append((std_sym, "15m", dt_d, dt_m, round(o_r * rf, 4), round(h_r * rf, 4), round(l_r * rf, 4), round(c_r * rf, 4), float(r[6] or 0), float(r[7] or 0), hfq, rf, "baostock_real_15m"))

    df_real15 = pd.DataFrame(recs, columns=["symbol", "freq", "date", "datetime", "open", "high", "low", "close", "volume", "amount", "hfq_factor", "ratio_factor", "source"]).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    df_real15 = df_real15[df_real15.groupby("date")["datetime"].transform("count") == 16].reset_index(drop=True)
    if df_real15.empty:
        return "no_bs_data", 0

    bs_dates = set(df_real15["date"].unique())
    for freq, df_new in [
        ("15m", df_real15),
        ("30m", _agg_from_15m(df_real15, 2, "30m", "baostock_real_30m")),
        ("60m", _agg_from_15m(df_real15, 4, "60m", "baostock_real_60m")),
        ("5m", _expand_15m_to_5m(df_real15, std_sym)),
    ]:
        p_f = ARCHIVE_DIR / freq / f"{std_sym}.parquet"
        old_f = pd.read_parquet(p_f)
        keep_f = old_f[~old_f["date"].isin(bs_dates)]
        merged = pd.concat([keep_f, df_new], ignore_index=True).sort_values("datetime").reset_index(drop=True)
        tmp_f = p_f.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp_f, index=False, compression="zstd")
        tmp_f.replace(p_f)
    return "upgraded", len(df_real15)


def main():
    import baostock as bs
    parser = argparse.ArgumentParser(description="从 BaoStock 单线程拉取 2020-01 ~ 2024-08 真实分钟K线到 Lexar-2T")
    parser.add_argument("--symbols", nargs="+", default=["all_a"], help="core(核心523只优先) / all_a(全市场，核心池排最前)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta_df = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
    if args.symbols == ["core"]:
        symbols = meta_df[meta_df["in_core"]]["symbol"].tolist()
    elif args.symbols == ["all_a"]:
        symbols = meta_df["symbol"].tolist()
    else:
        symbols = args.symbols

    prog_file = ARCHIVE_DIR / "logs" / "baostock_2020_2024_progress.json"
    prog_file.parent.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    print(f"BaoStock single-thread login: {lg.error_code} {lg.error_msg} | Total symbols: {len(symbols)}", flush=True)
    t0 = time.time()
    stats = {"upgraded": 0, "already_real": 0, "skipped_non_stock": 0, "no_bs_data": 0}
    net_fetched = 0
    net_time = 0.0

    for idx, sym in enumerate(symbols, 1):
        ts = time.time()
        st, n_bars = upgrade_one_symbol_2020_2024(sym, force=args.force)
        dt_sym = time.time() - ts
        stats[st] = stats.get(st, 0) + 1
        if st == "upgraded":
            net_fetched += 1
            net_time += dt_sym
        avg_s = (net_time / net_fetched) if net_fetched > 0 else 12.0
        rem = len(symbols) - idx
        eta_min = round(rem * avg_s / 60.0, 1)
        payload = {
            "done": idx, "total": len(symbols), "pct": round(idx * 100.0 / len(symbols), 2),
            "current_symbol": sym, "last_status": st, "last_15m_bars": n_bars,
            "last_sec": round(dt_sym, 1), "avg_sec_per_fetched": round(avg_s, 1),
            "stats": stats, "elapsed_min": round((time.time() - t0) / 60.0, 1),
        }
        prog_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if st == "upgraded" or idx % 20 == 0 or idx == len(symbols):
            print(f"[{idx}/{len(symbols)} ({payload['pct']:.1f}%)] {sym} -> {st} ({n_bars} bars in {dt_sym:.1f}s) | stats={stats} | avg={avg_s:.1f}s/sym", flush=True)
        if st == "bs_blocked":
            print("BaoStock rate limit detected, sleeping 60s before resuming...", flush=True)
            time.sleep(60)
            _safe_relogin()

    try:
        bs.logout()
    except Exception:
        pass
    init_lexar_duckdb()
    print(f"Completed all {len(symbols)} symbols -> {stats}", flush=True)


if __name__ == "__main__":
    main()


def main():
    import baostock as bs
    parser = argparse.ArgumentParser(description="从 BaoStock 单线程拉取 2020-01 ~ 2024-08 真实分钟K线到 Lexar-2T")
    parser.add_argument("--symbols", nargs="+", default=["all_a"], help="core(核心523只优先) / all_a(全市场，核心池排最前)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta_df = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
    if args.symbols == ["core"]:
        symbols = meta_df[meta_df["in_core"]]["symbol"].tolist()
    elif args.symbols == ["all_a"]:
        symbols = meta_df["symbol"].tolist()
    else:
        symbols = args.symbols

    prog_file = ARCHIVE_DIR / "logs" / "baostock_2020_2024_progress.json"
    prog_file.parent.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    print(f"BaoStock single-thread login: {lg.error_code} {lg.error_msg} | Total symbols: {len(symbols)}", flush=True)
    t0 = time.time()
    stats = {"upgraded": 0, "already_real": 0, "skipped_non_stock": 0, "no_bs_data": 0}
    net_fetched = 0
    net_time = 0.0

    for idx, sym in enumerate(symbols, 1):
        ts = time.time()
        st, n_bars = upgrade_one_symbol_2020_2024(sym, force=args.force)
        dt_sym = time.time() - ts
        stats[st] = stats.get(st, 0) + 1
        if st == "upgraded":
            net_fetched += 1
            net_time += dt_sym
        avg_s = (net_time / net_fetched) if net_fetched > 0 else 12.0
        rem = len(symbols) - idx
        eta_min = round(rem * avg_s / 60.0, 1)
        payload = {
            "done": idx, "total": len(symbols), "pct": round(idx * 100.0 / len(symbols), 2),
            "current_symbol": sym, "last_status": st, "last_15m_bars": n_bars,
            "last_sec": round(dt_sym, 1), "avg_sec_per_fetched": round(avg_s, 1),
            "stats": stats, "elapsed_min": round((time.time() - t0) / 60.0, 1),
        }
        prog_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if st == "upgraded" or idx % 20 == 0 or idx == len(symbols):
            print(f"[{idx}/{len(symbols)} ({payload['pct']:.1f}%)] {sym} -> {st} ({n_bars} bars in {dt_sym:.1f}s) | stats={stats} | avg={avg_s:.1f}s/sym", flush=True)
        if st == "bs_blocked":
            print("BaoStock rate limit detected, sleeping 60s before resuming...", flush=True)
            time.sleep(60)
            _safe_relogin()

    try:
        bs.logout()
    except Exception:
        pass
    init_lexar_duckdb()
    print(f"Completed all {len(symbols)} symbols -> {stats}", flush=True)


if __name__ == "__main__":
    main()


def main():
    import baostock as bs
    parser = argparse.ArgumentParser(description="从 BaoStock 单线程拉取 2020-01 ~ 2024-08 真实分钟K线到 Lexar-2T")
    parser.add_argument("--symbols", nargs="+", default=["all_a"], help="core(核心523只优先) / all_a(全市场，核心池排最前)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta_df = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
    if args.symbols == ["core"]:
        symbols = meta_df[meta_df["in_core"]]["symbol"].tolist()
    elif args.symbols == ["all_a"]:
        symbols = meta_df["symbol"].tolist()
    else:
        symbols = args.symbols

    prog_file = ARCHIVE_DIR / "logs" / "baostock_2020_2024_progress.json"
    prog_file.parent.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    print(f"BaoStock single-thread login: {lg.error_code} {lg.error_msg} | Total symbols: {len(symbols)}", flush=True)
    t0 = time.time()
    stats = {"upgraded": 0, "already_real": 0, "skipped_non_stock": 0, "no_bs_data": 0}
    net_fetched = 0
    net_time = 0.0

    for idx, sym in enumerate(symbols, 1):
        ts = time.time()
        st, n_bars = upgrade_one_symbol_2020_2024(sym, force=args.force)
        dt_sym = time.time() - ts
        stats[st] = stats.get(st, 0) + 1
        if st == "upgraded":
            net_fetched += 1
            net_time += dt_sym
        avg_s = (net_time / net_fetched) if net_fetched > 0 else 12.0
        rem = len(symbols) - idx
        eta_min = round(rem * avg_s / 60.0, 1)
        payload = {
            "done": idx, "total": len(symbols), "pct": round(idx * 100.0 / len(symbols), 2),
            "current_symbol": sym, "last_status": st, "last_15m_bars": n_bars,
            "last_sec": round(dt_sym, 1), "avg_sec_per_fetched": round(avg_s, 1),
            "stats": stats, "elapsed_min": round((time.time() - t0) / 60.0, 1),
        }
        prog_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if st == "upgraded" or idx % 20 == 0 or idx == len(symbols):
            print(f"[{idx}/{len(symbols)} ({payload['pct']:.1f}%)] {sym} -> {st} ({n_bars} bars in {dt_sym:.1f}s) | stats={stats} | avg={avg_s:.1f}s/sym", flush=True)
        if st == "bs_blocked":
            print("BaoStock rate limit detected, sleeping 60s before resuming...", flush=True)
            time.sleep(60)
            _safe_relogin()

    try:
        bs.logout()
    except Exception:
        pass
    init_lexar_duckdb()
    print(f"Completed all {len(symbols)} symbols -> {stats}", flush=True)


if __name__ == "__main__":
    main()


def main():
    import baostock as bs
    parser = argparse.ArgumentParser(description="从 BaoStock 单线程拉取 2020-01 ~ 2024-08 真实分钟K线到 Lexar-2T")
    parser.add_argument("--symbols", nargs="+", default=["all_a"], help="core(核心523只优先) / all_a(全市场，核心池排最前)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta_df = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
    if args.symbols == ["core"]:
        symbols = meta_df[meta_df["in_core"]]["symbol"].tolist()
    elif args.symbols == ["all_a"]:
        symbols = meta_df["symbol"].tolist()
    else:
        symbols = args.symbols

    prog_file = ARCHIVE_DIR / "logs" / "baostock_2020_2024_progress.json"
    prog_file.parent.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    print(f"BaoStock single-thread login: {lg.error_code} {lg.error_msg} | Total symbols: {len(symbols)}", flush=True)
    t0 = time.time()
    stats = {"upgraded": 0, "already_real": 0, "skipped_non_stock": 0, "no_bs_data": 0}
    net_fetched = 0
    net_time = 0.0

    for idx, sym in enumerate(symbols, 1):
        ts = time.time()
        st, n_bars = upgrade_one_symbol_2020_2024(sym, force=args.force)
        dt_sym = time.time() - ts
        stats[st] = stats.get(st, 0) + 1
        if st == "upgraded":
            net_fetched += 1
            net_time += dt_sym
        avg_s = (net_time / net_fetched) if net_fetched > 0 else 12.0
        rem = len(symbols) - idx
        eta_min = round(rem * avg_s / 60.0, 1)
        payload = {
            "done": idx, "total": len(symbols), "pct": round(idx * 100.0 / len(symbols), 2),
            "current_symbol": sym, "last_status": st, "last_15m_bars": n_bars,
            "last_sec": round(dt_sym, 1), "avg_sec_per_fetched": round(avg_s, 1),
            "stats": stats, "elapsed_min": round((time.time() - t0) / 60.0, 1),
        }
        prog_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if st == "upgraded" or idx % 20 == 0 or idx == len(symbols):
            print(f"[{idx}/{len(symbols)} ({payload['pct']:.1f}%)] {sym} -> {st} ({n_bars} bars in {dt_sym:.1f}s) | stats={stats} | avg={avg_s:.1f}s/sym", flush=True)
        if st == "bs_blocked":
            print("BaoStock rate limit detected, sleeping 60s before resuming...", flush=True)
            time.sleep(60)
            _safe_relogin()

    try:
        bs.logout()
    except Exception:
        pass
    init_lexar_duckdb()
    print(f"Completed all {len(symbols)} symbols -> {stats}", flush=True)


if __name__ == "__main__":
    main()


def main():
    import baostock as bs
    parser = argparse.ArgumentParser(description="从 BaoStock 单线程拉取 2020-01 ~ 2024-08 真实分钟K线到 Lexar-2T")
    parser.add_argument("--symbols", nargs="+", default=["all_a"], help="core(核心523只优先) / all_a(全市场，核心池排最前)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta_df = pd.read_parquet(ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet")
    if args.symbols == ["core"]:
        symbols = meta_df[meta_df["in_core"]]["symbol"].tolist()
    elif args.symbols == ["all_a"]:
        symbols = meta_df["symbol"].tolist()
    else:
        symbols = args.symbols

    prog_file = ARCHIVE_DIR / "logs" / "baostock_2020_2024_progress.json"
    prog_file.parent.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    print(f"BaoStock single-thread login: {lg.error_code} {lg.error_msg} | Total symbols: {len(symbols)}", flush=True)
    t0 = time.time()
    stats = {"upgraded": 0, "already_real": 0, "skipped_non_stock": 0, "no_bs_data": 0}
    net_fetched = 0
    net_time = 0.0

    for idx, sym in enumerate(symbols, 1):
        ts = time.time()
        st, n_bars = upgrade_one_symbol_2020_2024(sym, force=args.force)
        dt_sym = time.time() - ts
        stats[st] = stats.get(st, 0) + 1
        if st == "upgraded":
            net_fetched += 1
            net_time += dt_sym
        avg_s = (net_time / net_fetched) if net_fetched > 0 else 12.0
        rem = len(symbols) - idx
        eta_min = round(rem * avg_s / 60.0, 1)
        payload = {
            "done": idx, "total": len(symbols), "pct": round(idx * 100.0 / len(symbols), 2),
            "current_symbol": sym, "last_status": st, "last_15m_bars": n_bars,
            "last_sec": round(dt_sym, 1), "avg_sec_per_fetched": round(avg_s, 1),
            "stats": stats, "elapsed_min": round((time.time() - t0) / 60.0, 1),
        }
        prog_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if st == "upgraded" or idx % 20 == 0 or idx == len(symbols):
            print(f"[{idx}/{len(symbols)} ({payload['pct']:.1f}%)] {sym} -> {st} ({n_bars} bars in {dt_sym:.1f}s) | stats={stats} | avg={avg_s:.1f}s/sym", flush=True)
        if st == "bs_blocked":
            print("BaoStock rate limit detected, sleeping 60s before resuming...", flush=True)
            time.sleep(60)
            _safe_relogin()

    try:
        bs.logout()
    except Exception:
        pass
    init_lexar_duckdb()
    print(f"Completed all {len(symbols)} symbols -> {stats}", flush=True)


if __name__ == "__main__":
    main()
