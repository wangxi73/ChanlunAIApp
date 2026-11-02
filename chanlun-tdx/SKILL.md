---
name: chanlun-tdx
description: Use this skill when the user asks to analyze A-share stocks using Chanlun (缠论/缠中说禅) and wants to fetch the latest K-line data directly from TongDaXin (通达信). It provides a Python script to fetch market data and perform basic Chanlun analysis, such as identifying Bi (笔), Duan (线段), and ZhongShu (中枢).
---

# Chanlun A-Share Analysis with TDX Data

This skill provides a complete workflow for fetching A-share K-line data directly from TongDaXin (通达信) servers using `pytdx` and analyzing it using Chanlun (缠中说禅/缠论) via the `czsc` library.

To keep the system environment clean, this skill uses a dedicated Python virtual environment (`.venv`).

## Setup and Dependencies

The environment will be automatically created the first time you use the wrapper script. If you want to set it up manually, you can run:

```bash
cd chanlun-tdx
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## How to use

1. Run the wrapper script `scripts/run.sh`. You can provide a stock symbol (e.g., `000001` for Ping An Bank) and the market (`0` for Shenzhen, `1` for Shanghai).
2. The script will automatically use the virtual environment, connect to a TDX server, fetch the latest K-line data, and output the latest Chanlun elements (like `Bi` 笔, `Duan` 线段, `ZhongShu` 中枢).

## Scripts

### `scripts/run.sh`
This is the recommended entry point. It wraps `analyze.py` and ensures it runs inside the isolated `.venv` environment.

**Usage:**
```bash
./chanlun-tdx/scripts/run.sh --code 000001 --market 0
```

### `scripts/analyze.py`
The core Python script. If you have activated the virtual environment yourself (`source .venv/bin/activate`), you can run it directly:

```bash
python3 chanlun-tdx/scripts/analyze.py --code 000001 --market 0
```

- `--code`: The 6-digit stock code (e.g. `000001`, `300750`)
- `--market`: `0` for Shenzhen (SZ), `1` for Shanghai (SH)
