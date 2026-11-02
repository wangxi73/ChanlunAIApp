"""
A股实盘涨跌停严苛撮合过滤器 (Strict Limit-Up / Limit-Down Execution Filter)
"""

def get_limit_threshold_pct(symbol: str, dt: str) -> float:
    if symbol.startswith("sh51") or symbol.startswith("sh58") or symbol.startswith("sz15"):
        if symbol == "sh588000" or (symbol == "sz159915" and dt >= "2020-08-24"):
            return 19.6
        return 9.8
    if symbol.startswith("sh688"):
        return 19.6
    if symbol.startswith("sz30") and dt >= "2020-08-24":
        return 19.6
    return 9.75

def is_limit_up_close(symbol: str, dt: str, snap: dict, last_day_bar: dict = None) -> bool:
    if not snap or snap.get("lastPrice", 0) <= 0:
        return False
    pct = snap.get("pctChange", 0.0)
    thresh = get_limit_threshold_pct(symbol, dt)
    if pct < thresh:
        return False
    if last_day_bar and last_day_bar.get("high", 0) > 0:
        # 收盘价等于或极接近当日最高价(封死涨停)
        return last_day_bar["close"] >= last_day_bar["high"] * 0.998
    return True

def is_limit_down_close(symbol: str, dt: str, snap: dict, last_day_bar: dict = None) -> bool:
    if not snap or snap.get("lastPrice", 0) <= 0:
        return False
    pct = snap.get("pctChange", 0.0)
    thresh = get_limit_threshold_pct(symbol, dt)
    if pct > -thresh:
        return False
    if last_day_bar and last_day_bar.get("low", 0) > 0:
        # 收盘价等于或极接近当日最低价(封死跌停)
        return last_day_bar["close"] <= last_day_bar["low"] * 1.002
    return True
