from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_tw_ticker(code: str) -> str:
    """Convert user input to the Taiwan stock code used by FinMind."""
    code = str(code).strip().upper()
    if not code:
        return "2330"
    if code in ["TAIEX", "TWII"]:
        return "TAIEX"
    return code.replace(".TW", "").replace(".TWO", "")


def display_code(ticker: str) -> str:
    ticker = str(ticker).upper()
    return ticker.replace(".TW", "").replace(".TWO", "")


def safe_float(value, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def is_valid_number(value) -> bool:
    return value is not None and not pd.isna(value) and np.isfinite(float(value))


def dataframe_records(df: pd.DataFrame) -> list[dict]:
    """Serialize a DataFrame into JSON-safe records."""
    if df is None or df.empty:
        return []
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index(names="date")
    else:
        out = out.reset_index()
    out = out.replace({np.nan: None, np.inf: None, -np.inf: None})
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict("records")


def series_dict(row: pd.Series) -> dict:
    data = row.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in data.items()}
