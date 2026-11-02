import argparse
import pandas as pd
from pytdx.hq import TdxHq_API

try:
    from czsc import CZSC, RawBar
except ImportError:
    print("Please install czsc and its dependencies: python3 -m pip install czsc")
    exit(1)

def fetch_tdx_data(code, market, count=800):
    api = TdxHq_API()
    
    servers = [
        ('119.147.212.81', 7709),
        ('119.147.171.206', 7709),
        ('119.147.86.171', 7709),
        ('114.113.227.185', 7709),
        ('106.120.74.86', 7709),
        ('113.105.142.136', 7709),
        ('121.14.104.70', 7709),
        ('114.80.63.12', 7709)
    ]
    
    connected = False
    for ip, port in servers:
        print(f"Trying to connect to TDX server {ip}:{port}...")
        try:
            if api.connect(ip, port, time_out=1.5):
                connected = True
                print(f"Successfully connected to {ip}:{port}")
                break
        except Exception:
            pass

    if connected:
        # category 9 corresponds to daily K-line
        data = api.get_security_bars(9, market, code, 0, count)
        api.disconnect()
        if not data:
            print("Failed to fetch data or no data available.")
            return pd.DataFrame()
        df = api.to_df(data)
        
        # pytdx df has: datetime, open, close, high, low, vol, amount
        df['dt'] = pd.to_datetime(df['datetime'])
        df['symbol'] = code
        return df
    else:
        print("Failed to connect to any TDX server.")
        return pd.DataFrame()

def analyze_chanlun(df):
    bars = []
    for i, row in df.iterrows():
        bar = RawBar(
            symbol=row['symbol'],
            id=i,
            dt=row['dt'],
            freq="D",
            open=row['open'],
            close=row['close'],
            high=row['high'],
            low=row['low'],
            vol=row['vol'],
            amount=row['amount']
        )
        bars.append(bar)
    
    c = CZSC(bars)
    
    print(f"\n[{bars[-1].dt.strftime('%Y-%m-%d')}] Chanlun Analysis for {bars[0].symbol}")
    print("=" * 50)
    print(f"Total K-lines: {len(c.bars_raw)}")
    print(f"Latest FX (分型): {c.fx_list[-1] if c.fx_list else 'None'}")
    print(f"Latest Bi (笔): {c.bi_list[-1] if c.bi_list else 'None'}")
    print(f"Latest Duan (线段): {c.duan_list[-1] if c.duan_list else 'None'}")
    
    print("\nRecent 5 Bi (笔):")
    for bi in c.bi_list[-5:]:
        print(bi)
        
    print("\nRecent 3 ZhongShu (中枢):")
    for zs in c.zs_list[-3:]:
        print(zs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TDX data and run Chanlun analysis")
    parser.add_argument("--code", type=str, default="000001", help="Stock code (e.g., 000001)")
    parser.add_argument("--market", type=int, default=0, help="Market: 0 for SZ, 1 for SH")
    parser.add_argument("--count", type=int, default=800, help="Number of K-lines to fetch")
    args = parser.parse_args()
    
    print(f"Fetching data for {args.code} (Market {args.market})...")
    df = fetch_tdx_data(args.code, args.market, args.count)
    if not df.empty:
        analyze_chanlun(df)
