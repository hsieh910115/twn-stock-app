from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data import load_price_data
from .utils import safe_float


def dividend_yield_pct(info: Dict, last_close: float = np.nan) -> float:
    value = safe_float(info.get("finmindDividendYield"))
    if not pd.isna(value) and value > 0:
        return value
    for key in ["dividendYield", "trailingAnnualDividendYield"]:
        value = safe_float(info.get(key))
        if not pd.isna(value) and value > 0:
            return value * 100 if value <= 1 else value
    value = safe_float(info.get("fiveYearAvgDividendYield"))
    if not pd.isna(value) and value > 0:
        return value
    close = safe_float(last_close)
    if not pd.isna(close) and close > 0:
        for key in ["dividendRate", "trailingAnnualDividendRate", "lastDividendValue"]:
            cash_dividend = safe_float(info.get(key))
            if not pd.isna(cash_dividend) and cash_dividend > 0:
                return cash_dividend / close * 100
    return np.nan


def derive_fundamental_metrics(info: Dict, fast_info: Dict, last_close: float, fin_table: Optional[pd.DataFrame] = None) -> Dict:
    last_close = safe_float(last_close)
    pe = safe_float(info.get("trailingPE"))
    if pd.isna(pe) or pe <= 0:
        pe = safe_float(fast_info.get("trailingPE"))
    eps = safe_float(info.get("trailingEps"))
    if pd.isna(eps) or eps == 0:
        eps = safe_float(fast_info.get("trailingEps"))
    shares = safe_float(info.get("sharesOutstanding"))
    if pd.isna(shares):
        shares = safe_float(fast_info.get("shares"))
    if (pd.isna(eps) or eps == 0) and fin_table is not None and not fin_table.empty and "淨利(億)" in fin_table.columns and not pd.isna(shares) and shares > 0:
        recent_net_income = fin_table["淨利(億)"].dropna().tail(4).sum() * 1e8
        if recent_net_income != 0:
            eps = recent_net_income / shares
    if (pd.isna(pe) or pe <= 0) and not pd.isna(last_close) and not pd.isna(eps) and eps > 0:
        pe = last_close / eps
    dividend_yield = dividend_yield_pct(info, last_close)
    if pd.isna(dividend_yield):
        dividend_yield = dividend_yield_pct(fast_info, last_close)
    market_cap = safe_float(info.get("marketCap"))
    if pd.isna(market_cap):
        market_cap = safe_float(fast_info.get("marketCap"))
    if pd.isna(market_cap):
        market_cap = safe_float(fast_info.get("market_cap"))
    if pd.isna(market_cap) and not pd.isna(last_close) and not pd.isna(shares) and shares > 0:
        market_cap = last_close * shares
    beta = safe_float(info.get("beta"))
    if pd.isna(beta):
        beta = safe_float(fast_info.get("beta"))
    return {"pe": pe, "eps": eps, "dividend_yield_pct": dividend_yield, "market_cap": market_cap, "beta": beta}


def estimate_beta_vs_twii(stock_df: pd.DataFrame) -> float:
    try:
        market_raw, _ = load_price_data("0050")
        aligned = pd.concat([stock_df["Close"].pct_change(), market_raw["Close"].pct_change()], axis=1).dropna()
        aligned.columns = ["stock", "market"]
        aligned = aligned.tail(252 * 2)
        aligned = aligned[(aligned["stock"].abs() < 0.3) & (aligned["market"].abs() < 0.15)]
        if len(aligned) < 60:
            return np.nan
        market_var = aligned["market"].var()
        if market_var == 0 or pd.isna(market_var):
            return np.nan
        return float(aligned["stock"].cov(aligned["market"]) / market_var)
    except Exception:
        return np.nan


def normalize_financial_type(x: str) -> Optional[str]:
    text = str(x).lower()
    mapping = [
        (["revenue", "營業收入", "營收"], "營收(億)"),
        (["grossprofit", "gross_profit", "毛利"], "毛利(億)"),
        (["operatingincome", "operating_income", "營業利益", "營業收入淨額"], "營業利益(億)"),
        (["netincome", "net_income", "本期淨利", "淨利"], "淨利(億)"),
    ]
    for keys, col in mapping:
        if any(key in text for key in keys):
            return col
    return None


def build_financial_table(stmt: pd.DataFrame, ticker_symbol: Optional[str] = None) -> pd.DataFrame:
    if stmt is None or stmt.empty or not {"date", "type", "value"}.issubset(stmt.columns):
        return pd.DataFrame()
    data = stmt.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["quarter"] = data["date"].apply(lambda d: f"{d.year}Q{((d.month - 1) // 3) + 1}" if pd.notna(d) else np.nan)
    data["項目"] = data["type"].apply(normalize_financial_type)
    data["value"] = pd.to_numeric(data["value"], errors="coerce") / 1e8
    data = data.dropna(subset=["quarter", "項目", "value"])
    if data.empty:
        return pd.DataFrame()
    out = data.pivot_table(index="quarter", columns="項目", values="value", aggfunc="last").sort_index().tail(8)
    if "營收(億)" in out.columns:
        out["營收QoQ%"] = out["營收(億)"].pct_change() * 100
        out["營收YoY%"] = out["營收(億)"].pct_change(4) * 100
    if "淨利(億)" in out.columns:
        out["淨利YoY%"] = out["淨利(億)"].pct_change(4) * 100
    order = ["營收(億)", "毛利(億)", "營業利益(億)", "淨利(億)", "營收QoQ%", "營收YoY%", "淨利YoY%"]
    return out[[c for c in order if c in out.columns]].round(2)
