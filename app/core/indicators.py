from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out.get("Volume", pd.Series(index=out.index, dtype=float)).fillna(0)

    out["Return_1D"] = close.pct_change()
    out["Return_5D"] = close.pct_change(5)
    out["Return_20D"] = close.pct_change(20)
    for n in [5, 10, 20, 60, 120, 240]:
        out[f"MA{n}"] = close.rolling(n).mean()
    out["EMA10"] = close.ewm(span=10, adjust=False).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["High_20D"] = close.rolling(20).max()
    out["High_55D"] = close.rolling(55).max()
    out["Low_10D"] = close.rolling(10).min()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD_DIF"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD_DIF"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD_DIF"] - out["MACD_SIGNAL"]

    out["BB_MID"] = out["MA20"]
    out["BB_STD"] = close.rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + 2 * out["BB_STD"]
    out["BB_LOWER"] = out["BB_MID"] - 2 * out["BB_STD"]
    out["BB_WIDTH"] = (out["BB_UPPER"] - out["BB_LOWER"]) / out["BB_MID"]

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["Volatility_20D"] = out["Return_1D"].rolling(20).std() * np.sqrt(252)

    out["Volume_MA20"] = volume.rolling(20).mean()
    out["Volume_Ratio"] = volume / out["Volume_MA20"].replace(0, np.nan)
    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume).cumsum()

    out["High_52W"] = close.rolling(240).max()
    out["Low_52W"] = close.rolling(240).min()
    out["Drawdown"] = close / close.cummax() - 1
    return out
