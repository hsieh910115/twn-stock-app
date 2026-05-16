from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .utils import normalize_tw_ticker, safe_float


def get_finmind_token() -> str:
    return os.getenv("FINMIND_TOKEN", "")


def finmind_request(dataset: str, **params) -> pd.DataFrame:
    url = "https://api.finmindtrade.com/api/v4/data"
    token = get_finmind_token()
    query = {"dataset": dataset, **{k: v for k, v in params.items() if v is not None and v != ""}}
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = requests.get(url, params=query, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("status", 200)) != 200:
        raise RuntimeError(payload.get("msg", f"FinMind API error: {dataset}"))
    return pd.DataFrame(payload.get("data", []))


@lru_cache(maxsize=256)
def load_price_data(ticker_symbol: str) -> Tuple[pd.DataFrame, str]:
    stock_id = normalize_tw_ticker(ticker_symbol)
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 5 + 10)).strftime("%Y-%m-%d")

    if stock_id == "TAIEX":
        idx = finmind_request(
            "TaiwanStockTotalReturnIndex",
            index_id="TAIEX",
            start_date=start_date,
            end_date=end_date,
        )
        if idx.empty:
            raise RuntimeError("無法取得加權指數 TAIEX 資料。")
        idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
        value_col = next((c for c in ["price", "value", "close", "index"] if c in idx.columns), None)
        if value_col is None:
            numeric_cols = [c for c in idx.columns if c != "date" and pd.api.types.is_numeric_dtype(idx[c])]
            value_col = numeric_cols[0] if numeric_cols else None
        if value_col is None:
            raise RuntimeError("TAIEX 回傳欄位無法辨識。")
        close = pd.to_numeric(idx[value_col], errors="coerce")
        out = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0},
            index=idx["date"],
        )
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out.dropna(subset=["Close"]).sort_index(), "TAIEX"

    df = finmind_request(
        "TaiwanStockPrice",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )
    if df.empty:
        raise RuntimeError(f"無法取得 {stock_id} 股價資料，請確認代碼是否正確。")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.rename(columns={"open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
    keep = ["date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan if col != "Volume" else 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).set_index("date").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["Close"]), stock_id


@lru_cache(maxsize=1)
def load_stock_name_map() -> Dict[str, Dict]:
    try:
        info_df = finmind_request("TaiwanStockInfo")
        if info_df.empty:
            return {}
        out = {}
        for _, row in info_df.iterrows():
            sid = str(row.get("stock_id", "")).strip()
            if sid:
                out[sid] = {
                    "shortName": row.get("stock_name"),
                    "longName": row.get("stock_name"),
                    "industry": row.get("industry_category"),
                    "market": row.get("type"),
                }
        return out
    except Exception:
        return {}


def resolve_stock_input(user_input: str) -> str:
    text = str(user_input).strip()
    if not text:
        return "2330"
    normalized = normalize_tw_ticker(text)
    if normalized.isdigit() or normalized == "TAIEX":
        return normalized
    stock_map = load_stock_name_map()
    for code, info in stock_map.items():
        if text == str(info.get("shortName", "")):
            return code
    for code, info in stock_map.items():
        if text in str(info.get("shortName", "")):
            return code
    return text


@lru_cache(maxsize=256)
def load_ticker_info(ticker_symbol: str) -> Dict:
    stock_id = normalize_tw_ticker(ticker_symbol)
    info = load_stock_name_map().get(stock_id, {}).copy()
    if not info and stock_id == "TAIEX":
        info = {"shortName": "加權指數", "longName": "TAIEX 加權指數"}
    try:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        per_df = finmind_request("TaiwanStockPER", data_id=stock_id, start_date=start_date, end_date=end_date)
        if not per_df.empty:
            per_df["date"] = pd.to_datetime(per_df["date"], errors="coerce")
            latest = per_df.sort_values("date").iloc[-1]
            info["trailingPE"] = safe_float(latest.get("PER"))
            info["priceToBook"] = safe_float(latest.get("PBR"))
            info["finmindDividendYield"] = safe_float(latest.get("dividend_yield"))
    except Exception:
        pass
    return info


@lru_cache(maxsize=256)
def load_fast_info(ticker_symbol: str) -> Dict:
    stock_id = normalize_tw_ticker(ticker_symbol)
    if stock_id == "TAIEX":
        return {}
    out = {}
    try:
        ticker = yf.Ticker(f"{stock_id}.TW")
        try:
            info = ticker.get_info() or {}
        except Exception:
            info = ticker.info or {}
        try:
            fast_info = ticker.fast_info or {}
        except Exception:
            fast_info = {}
        out.update({
            "trailingPE": safe_float(info.get("trailingPE")),
            "trailingEps": safe_float(info.get("trailingEps")),
            "dividendYield": safe_float(info.get("dividendYield")),
            "marketCap": safe_float(info.get("marketCap")),
            "shares": safe_float(info.get("sharesOutstanding")),
            "beta": safe_float(info.get("beta")),
            "market_cap": safe_float(fast_info.get("market_cap")),
        })
    except Exception:
        pass
    return out


@lru_cache(maxsize=256)
def load_financials(ticker_symbol: str) -> pd.DataFrame:
    stock_id = normalize_tw_ticker(ticker_symbol)
    if stock_id == "TAIEX":
        return pd.DataFrame()
    try:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
        return finmind_request(
            "TaiwanStockFinancialStatements",
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        return pd.DataFrame()
