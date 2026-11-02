#!/usr/bin/env python3
"""
全库 (Lexar-2T) 成交量(股)与成交额(元)物理恒等式统一校准器
校准规则：
  1. 针对 tdx_real_* 分钟线：pytdx 分钟接口返回的 vol 单位已是“股”，若检测出 VWAP = amount/volume 与真实价相差 100 倍，自动将 volume 除以 100
  2. 针对 1d 与 unified_ratio_qfq：科创板(sh688)在腾讯日线接口返回的 volume 已是“股”，若与真实分钟成交量/额相差 100 倍，自动将 volume 与 amount 除以 100
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/Lexar-2T/A_Share_10Y_Klines")


def calibrate_one(sym: str) -> str:
    changed_any = False
    for freq in ("15m", "30m", "60m"):
        pf = ROOT / freq / f"{sym}.parquet"
        if not pf.exists():
            continue
        df = pd.read_parquet(pf)
        m_tdx = df["source"].str.startswith("tdx_real")
        if m_tdx.any():
            sub = df.loc[m_tdx]
            raw_c = float((sub["close"] / sub["ratio_factor"]).median())
            vwap = float(sub["amount"].sum() / max(sub["volume"].sum(), 1.0))
            if raw_c > 0 and (raw_c / max(vwap, 1e-9)) > 30.0:
                df.loc[m_tdx, "volume"] = np.round(df.loc[m_tdx, "volume"] * 0.01, 2)
                tmp = pf.with_suffix(".parquet.tmp")
                df.to_parquet(tmp, index=False, compression="zstd")
                tmp.replace(pf)
                changed_any = True

    p_1d = ROOT / "1d" / f"{sym}.parquet"
    p_60m = ROOT / "60m" / f"{sym}.parquet"
    if not p_1d.exists() or not p_60m.exists():
        return "skip"
    df_1d = pd.read_parquet(p_1d)
    df_60m = pd.read_parquet(p_60m)
    real_60 = df_60m[df_60m["source"].str.startswith(("tdx_real", "baostock_real"))]
    if not real_60.empty:
        sample_dt = str(real_60["date"].iloc[-1])
        r_vol = float(real_60[real_60["date"] == sample_dt]["volume"].sum())
        d_row = df_1d[df_1d["date"] == sample_dt]
        if not d_row.empty and r_vol > 0:
            if float(d_row["volume"].iloc[0]) / r_vol > 30.0:
                df_1d["volume"] = np.round(df_1d["volume"] * 0.01, 2)
                df_1d["amount"] = np.round(df_1d["amount"] * 0.01, 2)
                tmp = p_1d.with_suffix(".parquet.tmp")
                df_1d.to_parquet(tmp, index=False, compression="zstd")
                tmp.replace(p_1d)
                for freq in ("5m", "15m", "30m", "60m"):
                    pf = ROOT / freq / f"{sym}.parquet"
                    df_f = pd.read_parquet(pf)
                    m_uf = df_f["source"] == "unified_ratio_qfq"
                    if m_uf.any():
                        df_f.loc[m_uf, "volume"] = np.round(df_f.loc[m_uf, "volume"] * 0.01, 2)
                        df_f.loc[m_uf, "amount"] = np.round(df_f.loc[m_uf, "amount"] * 0.01, 2)
                        tmp = pf.with_suffix(".parquet.tmp")
                        df_f.to_parquet(tmp, index=False, compression="zstd")
                        tmp.replace(pf)
                changed_any = True
    return "calibrated" if changed_any else "ok"


if __name__ == "__main__":
    syms = sorted(p.stem for p in (ROOT / "1d").glob("*.parquet"))
    with ThreadPoolExecutor(20) as ex:
        res = list(ex.map(calibrate_one, syms))
    print("Calibration summary:", pd.Series(res).value_counts().to_dict(), flush=True)
