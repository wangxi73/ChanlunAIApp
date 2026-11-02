#!/usr/bin/env python3
"""
全A股 K线数据库 (Lexar-2T) 每日盘后增量更新脚本 (unified_ratio_qfq 标准版)
功能特性：
  1. 自动探测最新交易日（对比行情接口最新日线日期）
  2. 自动同步全A股最新股票池（若有新上市IPO股票自动全量建档）
  3. 仅检查每只股票是否缺失最新交易日；若缺失或发生除权除息，调用 unify_single_symbol 更新等比前复权与双复权因子 (hfq_factor / ratio_factor)
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import pandas as pd

from fetch_10y_minute_archive import ARCHIVE_DIR, _http_json, init_lexar_duckdb, normalize_symbol
from unify_lexar_kline_db import fetch_sina_hfq_factors, unify_single_symbol


def get_latest_trading_date() -> str:
    """探测当前A股最新交易日 YYYY-MM-DD"""
    for url in [
        "http://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?param=sh600519,day,,,5,qfq",
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh600519&scale=240&ma=no&datalen=5",
    ]:
        try:
            raw = _http_json(url, timeout=5.0)
            if isinstance(raw, dict):
                arr = ((raw.get("data") or {}).get("sh600519") or {}).get("qfqday") or []
                if arr:
                    return str(arr[-1][0])[:10]
            elif isinstance(raw, list) and raw:
                return str(raw[-1]["day"])[:10]
        except Exception:
            continue
    return datetime.now().strftime("%Y-%m-%d")


def sync_stock_metadata(refresh_remote: bool = False) -> List[str]:
    meta_p = ARCHIVE_DIR / "metadata" / "all_a_stocks.parquet"
    df_old = pd.read_parquet(meta_p) if meta_p.exists() else pd.DataFrame()
    if not refresh_remote and not df_old.empty:
        return df_old["symbol"].tolist()
    try:
        import akshare as ak
        df_ak = ak.stock_info_a_code_name()
        existing = {r["symbol"]: r for r in df_old.to_dict("records")} if not df_old.empty else {}
        for _, r in df_ak.iterrows():
            c = str(r["code"]).strip()
            mkt = "sh" if c.startswith(("6", "5")) else ("sz" if c.startswith(("0", "3")) else "bj")
            sym = f"{mkt}{c}"
            if sym not in existing:
                existing[sym] = {"symbol": sym, "code": c, "market": mkt, "name": str(r["name"]).strip(), "in_core": False}
        df_new = pd.DataFrame(sorted(existing.values(), key=lambda x: (not x.get("in_core", False), x["symbol"])))
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_parquet(meta_p, index=False)
        return df_new["symbol"].tolist()
    except Exception:
        return df_old["symbol"].tolist() if not df_old.empty else []


def update_symbol_daily(symbol: str, target_date: str, force: bool = False) -> str:
    std_sym = normalize_symbol(symbol)[0]
    p_1d = ARCHIVE_DIR / "1d" / f"{std_sym}.parquet"
    if not p_1d.exists() or force:
        return unify_single_symbol(std_sym, force=True)
    try:
        df_chk = pd.read_parquet(p_1d, columns=["date", "source", "hfq_factor"])
        if df_chk.empty or df_chk["source"].iloc[0] != "unified_ratio_qfq":
            return unify_single_symbol(std_sym, force=True)
        if str(df_chk["date"].max()) < target_date:
            p_1d.unlink(missing_ok=True)
            return unify_single_symbol(std_sym, force=True)
        return "up_to_date"
    except Exception:
        return unify_single_symbol(std_sym, force=True)


def main():
    parser = argparse.ArgumentParser(description="全A股 Lexar-2T K线数据库每日更新器 (unified_ratio_qfq)")
    parser.add_argument("--symbols", nargs="+", default=["all_a"])
    parser.add_argument("--target-date", default="auto")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--sync-new-ipo", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_date = get_latest_trading_date() if args.target_date == "auto" else args.target_date
    symbols = sync_stock_metadata(refresh_remote=args.sync_new_ipo) if args.symbols == ["all_a"] else args.symbols
    t0 = time.time()
    stats = {"up_to_date": 0, "unified": 0, "already_unified": 0, "empty": 0}
    print(f"Daily update started -> target_date={target_date}, symbols={len(symbols)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(update_symbol_daily, s, target_date, args.force): s for s in symbols}
        for i, fut in enumerate(as_completed(futs), 1):
            st = fut.result()
            stats[st] = stats.get(st, 0) + 1
            if i % 500 == 0 or i == len(symbols):
                el = time.time() - t0
                print(f"[{i}/{len(symbols)}] {stats} | {i/max(el,0.1):.1f} sym/s | elapsed={el:.1f}s", flush=True)
    init_lexar_duckdb()
    print(f"Complete in {time.time()-t0:.1f}s -> {stats}")


if __name__ == "__main__":
    main()
