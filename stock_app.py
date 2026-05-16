"""
stock_app.py
台股投資分析
注意：本工具僅供研究、紀律化分析與交易前檢核，不構成投資建議。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import re
import urllib3

# =========================
# 基本設定
# =========================
APP_TITLE = "台股投資分析"
DEFAULT_WATCHLIST = "2330"

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {background: #ffffff; padding: 1rem; border-radius: 1rem; border: 1px solid #eeeeee;}
    .small-note {font-size: 0.85rem; color: #666666;}
    .risk-box {padding: 1rem; border-radius: 1rem; border: 1px solid #eeeeee; background: #fafafa;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 工具函數
# =========================
def normalize_tw_ticker(code: str) -> str:
    """將使用者輸入轉成 FinMind 使用的台股代碼。支援 2330、2330.TW、8069.TWO、TAIEX。"""
    code = str(code).strip().upper()
    if not code:
        return "2330"
    if code in ["TAIEX", "TWII", "TAIEX"]:
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


def format_number(value, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except Exception:
        return "—"


def format_large_twd(value) -> str:
    value = safe_float(value)
    if pd.isna(value):
        return "—"
    if abs(value) >= 1e12:
        return f"{value / 1e12:.2f} 兆"
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f} 億"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.2f} 萬"
    return f"{value:.0f}"


def get_finmind_token() -> str:
    """可選：在 .streamlit/secrets.toml 放 FINMIND_TOKEN，或設定環境變數 FINMIND_TOKEN。"""
    try:
        token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        token = ""
    return token or os.getenv("FINMIND_TOKEN", "")


def finmind_request(dataset: str, **params) -> pd.DataFrame:
    """統一呼叫 FinMind REST API。"""
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


@st.cache_data(ttl=60 * 10, show_spinner=False)
def load_price_data(ticker_symbol: str) -> Tuple[pd.DataFrame, str]:
    """使用 FinMind 抓取台股日 OHLCV。一律抓較長區間，切換分析期間時最新日固定。"""
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
        out = pd.DataFrame({
            "Open": close, "High": close, "Low": close, "Close": close, "Volume": 0,
        }, index=idx["date"])
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
    rename_map = {
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    keep = ["date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan if col != "Volume" else 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).set_index("date").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["Close"]), stock_id


def period_to_days(years: int, months: int) -> int:
    """把側欄的幾年幾個月轉成總天數。"""
    years = max(int(years), 0)
    months = max(int(months), 0)
    return max(years * 365 + months * 30, 30)


def trim_to_user_period(df: pd.DataFrame, target_start: pd.Timestamp) -> pd.DataFrame:
    """指標計算完後，切回使用者指定的分析期間。若 target_start 非交易日，自動取最近交易日。"""
    past_dates = df.index[df.index <= target_start]
    if past_dates.empty:
        return df.copy()
    actual_start = past_dates.max()
    return df[df.index >= actual_start].copy()


def is_valid_number(value) -> bool:
    """判斷數值是否可用，避免短期間缺 MA120/MA240 時被誤判為跌破。"""
    return value is not None and not pd.isna(value) and np.isfinite(float(value))


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_stock_name_map() -> Dict[str, Dict]:
    """從 FinMind 取得股票名稱對照。"""
    try:
        info_df = finmind_request("TaiwanStockInfo")
        if info_df.empty:
            return {}
        out = {}
        for _, row in info_df.iterrows():
            sid = str(row.get("stock_id", "")).strip()
            if not sid:
                continue
            out[sid] = {
                "shortName": row.get("stock_name"),
                "longName": row.get("stock_name"),
                "industry": row.get("industry_category"),
                "market": row.get("type"),
            }
        return out
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_ticker_info(ticker_symbol: str) -> Dict:
    """讀取 FinMind 股票名稱、PER/PBR/殖利率等資料。"""
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
            # FinMind 欄位常見：PER、PBR、dividend_yield
            info["trailingPE"] = safe_float(latest.get("PER"))
            info["priceToBook"] = safe_float(latest.get("PBR"))
            info["finmindDividendYield"] = safe_float(latest.get("dividend_yield"))
    except Exception:
        pass
    return info


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_fast_info(ticker_symbol: str) -> Dict:
    stock_id = normalize_tw_ticker(ticker_symbol)
    if stock_id == "TAIEX":
        return {}

    yf_symbol = f"{stock_id}.TW"
    out = {}

    try:
        t = yf.Ticker(yf_symbol)

        try:
            info = t.get_info() or {}
        except Exception:
            info = t.info or {}

        try:
            fi = t.fast_info or {}
        except Exception:
            fi = {}

        out.update({
            "trailingPE": safe_float(info.get("trailingPE")),
            "trailingEps": safe_float(info.get("trailingEps")),
            "dividendYield": safe_float(info.get("dividendYield")),
            "marketCap": safe_float(info.get("marketCap")),
            "shares": safe_float(info.get("sharesOutstanding")),
            "beta": safe_float(info.get("beta")),
            "market_cap": safe_float(fi.get("market_cap")),
        })

    except Exception:
        pass

    return out


def dividend_yield_pct(info: Dict, last_close: float = np.nan) -> float:
    # FinMind 的 dividend_yield 已經是百分比，不要再 *100
    v = safe_float(info.get("finmindDividendYield"))
    if not pd.isna(v) and v > 0:
        return v

    # yfinance 常是小數，例如 0.0098 = 0.98%
    for key in ["dividendYield", "trailingAnnualDividendYield"]:
        v = safe_float(info.get(key))
        if not pd.isna(v) and v > 0:
            return v * 100 if v <= 1 else v

    v = safe_float(info.get("fiveYearAvgDividendYield"))
    if not pd.isna(v) and v > 0:
        return v

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

    return {
        "pe": pe,
        "eps": eps,
        "dividend_yield_pct": dividend_yield,
        "market_cap": market_cap,
        "beta": beta,
    }


def estimate_beta_vs_twii(stock_df: pd.DataFrame) -> float:
    try:
        # 用 0050 當市場代理，比 TAIEX API 穩
        market_raw, _ = load_price_data("0050")

        s_ret = stock_df["Close"].pct_change()
        m_ret = market_raw["Close"].pct_change()

        aligned = pd.concat([s_ret, m_ret], axis=1).dropna()
        aligned.columns = ["stock", "market"]

        # 固定取最近兩年
        aligned = aligned.tail(252 * 2)

        # 避免極端值
        aligned = aligned[
            (aligned["stock"].abs() < 0.3) &
            (aligned["market"].abs() < 0.15)
        ]

        if len(aligned) < 60:
            return np.nan

        market_var = aligned["market"].var()

        if market_var == 0 or pd.isna(market_var):
            return np.nan

        beta = aligned["stock"].cov(aligned["market"]) / market_var

        return float(beta)

    except Exception as e:
        print("Beta error:", e)
        return np.nan


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_financials(ticker_symbol: str) -> pd.DataFrame:
    """使用 FinMind 取得綜合損益表原始資料。"""
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


def normalize_financial_type(x: str) -> Optional[str]:
    """把 FinMind 財報 type 欄位盡量對應到 App 使用的四個欄位。"""
    t = str(x).lower()
    mapping = [
        (["revenue", "營業收入", "營收"], "營收(億)"),
        (["grossprofit", "gross_profit", "毛利"], "毛利(億)"),
        (["operatingincome", "operating_income", "營業利益", "營業收入淨額"], "營業利益(億)"),
        (["netincome", "net_income", "本期淨利", "淨利"], "淨利(億)"),
    ]
    for keys, col in mapping:
        if any(k in t for k in keys):
            return col
    return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """建立偏交易員使用的技術指標：趨勢、動能、波動、量能、風險。"""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out.get("Volume", pd.Series(index=out.index, dtype=float)).fillna(0)

    # 報酬與均線
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

    # RSI（Wilder smoothing，比單純 rolling 平均更接近常見技術分析版本）
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD_DIF"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD_DIF"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD_DIF"] - out["MACD_SIGNAL"]

    # 布林通道
    out["BB_MID"] = out["MA20"]
    out["BB_STD"] = close.rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + 2 * out["BB_STD"]
    out["BB_LOWER"] = out["BB_MID"] - 2 * out["BB_STD"]
    out["BB_WIDTH"] = (out["BB_UPPER"] - out["BB_LOWER"]) / out["BB_MID"]

    # ATR 與波動
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["Volatility_20D"] = out["Return_1D"].rolling(20).std() * np.sqrt(252)

    # 量能
    out["Volume_MA20"] = volume.rolling(20).mean()
    out["Volume_Ratio"] = volume / out["Volume_MA20"].replace(0, np.nan)
    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume).cumsum()

    # 52 週位置與回撤
    out["High_52W"] = close.rolling(240).max()
    out["Low_52W"] = close.rolling(240).min()
    out["Drawdown"] = close / close.cummax() - 1

    return out


def score_stock(df: pd.DataFrame, info: Dict, mode: str) -> Dict:
    """統一 -10~+10 評分架構。"""

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0

    reasons: List[str] = []
    warnings: List[str] = []

    close = safe_float(last["Close"])
    rsi = safe_float(last["RSI14"])

    pe = safe_float(info.get("trailingPE"))
    eps = safe_float(info.get("trailingEps"))

    dividend_yield = dividend_yield_pct(info)
    beta = safe_float(info.get("beta"))

    # =========================================================
    # 短線／波段
    # =========================================================
    if "短線" in mode:

        # ===== 趨勢結構（±3）
        if (
            is_valid_number(last.get("MA5"))
            and is_valid_number(last.get("MA20"))
        ):

            if close > last["MA5"] > last["MA20"]:
                score += 3
                reasons.append("短均線呈多頭排列，短線趨勢強勢。")

            elif close < last["MA5"] < last["MA20"]:
                score -= 3
                warnings.append("短均線呈空頭排列，短線結構偏弱。")

        # ===== MACD 動能（±2）
        if (
            is_valid_number(last.get("MACD_HIST"))
            and is_valid_number(prev.get("MACD_HIST"))
        ):

            if (
                last["MACD_HIST"] > 0
                and last["MACD_HIST"] > prev["MACD_HIST"]
            ):
                score += 2
                reasons.append("MACD 動能持續擴大。")

            elif last["MACD_HIST"] < 0:
                score -= 2
                warnings.append("MACD 動能仍偏弱。")

        # ===== RSI（±2）
        if 50 <= rsi <= 68:
            score += 2
            reasons.append("RSI 位於強勢區且未過熱。")

        elif 68 < rsi <= 80:
            score += 1
            warnings.append("RSI 偏高，需留意短線過熱。")

        elif rsi > 80:
            score -= 2
            warnings.append("RSI 過熱，容易拉回。")

        elif rsi < 30:
            score -= 1
            warnings.append("RSI 過低，代表短線賣壓仍重。")

        # ===== 量能（±1）
        if (
            is_valid_number(last.get("Volume_Ratio"))
            and last["Volume_Ratio"] > 1.5
            and close > prev["Close"]
        ):
            score += 1
            reasons.append("量能放大且收漲，市場買盤積極。")

        elif (
            is_valid_number(last.get("Volume_Ratio"))
            and last["Volume_Ratio"] > 1.8
            and close < prev["Close"]
        ):
            score -= 1
            warnings.append("放量下跌，短線賣壓偏重。")

        # ===== 20日動能（±1）
        if is_valid_number(last.get("Return_20D")):

            if last["Return_20D"] > 0.15:
                score += 1
                reasons.append("近 20 日動能強勢。")

            elif last["Return_20D"] < -0.15:
                score -= 1
                warnings.append("近 20 日動能偏弱。")

        # ===== 布林位置（±1）
        if (
            is_valid_number(last.get("BB_UPPER"))
            and is_valid_number(last.get("BB_LOWER"))
        ):

            if close > last["BB_UPPER"]:
                score -= 1
                warnings.append("股價偏離布林上軌，追高風險較高。")

            elif close < last["BB_LOWER"]:
                score += 1
                reasons.append("股價接近布林下軌，可能有反彈機會。")

    # =========================================================
    # 長線／存股
    # =========================================================
    else:

        # ===== 中長期趨勢（±3）
        if (
            is_valid_number(last.get("MA60"))
            and is_valid_number(last.get("MA120"))
        ):

            if close > last["MA60"] > last["MA120"]:
                score += 3
                reasons.append("季線與半年線呈多頭排列。")

            elif close < last["MA60"] < last["MA120"]:
                score -= 3
                warnings.append("中長期均線偏空。")

        # ===== 年線結構（±2）
        if is_valid_number(last.get("MA240")):

            if close > last["MA240"]:
                score += 2
                reasons.append("股價位於年線之上。")

            else:
                score -= 2
                warnings.append("股價仍低於年線。")

        # ===== PE 估值（±2）
        if not pd.isna(pe):

            if 0 < pe <= 15:
                score += 2
                reasons.append("本益比偏低，估值相對合理。")

            elif 15 < pe <= 25:
                score += 1
                reasons.append("本益比尚屬合理區間。")

            elif pe > 40:
                score -= 2
                warnings.append("本益比偏高，估值壓力較大。")

        # ===== EPS（±1）
        if not pd.isna(eps):

            if eps > 5:
                score += 1
                reasons.append("EPS 表現良好。")

            elif eps <= 0:
                score -= 1
                warnings.append("EPS 非正值。")

        # ===== 殖利率（±1）
        if not pd.isna(dividend_yield):

            if dividend_yield >= 5:
                score += 1
                reasons.append("殖利率具吸引力。")

            elif dividend_yield < 1:
                score -= 1
                warnings.append("殖利率偏低。")

        # ===== Beta（±1）
        if not pd.isna(beta):

            if beta <= 1:
                score += 1
                reasons.append("Beta 較低，波動相對穩定。")

            elif beta > 1.6:
                score -= 1
                warnings.append("Beta 偏高，波動風險較大。")

    # =========================================================
    # 額外風險提醒（不計分）
    # =========================================================

    if (
        is_valid_number(last.get("Drawdown"))
        and last["Drawdown"] < -0.35
    ):
        warnings.append("距歷史高點回撤超過 35%。")

    # =========================================================
    # -10~10 → 0~10
    # =========================================================

    score = max(-10, min(10, score))

    score_10 = (score + 10) / 20 * 10
    score_10 = round(score_10, 1)

    # =========================================================
    # 評語
    # =========================================================

    if score_10 >= 8.5:
        label = "條件優良／可優先觀察"
        level = "success"

    elif score_10 >= 7:
        label = "偏多格局／可列入候選"
        level = "info"

    elif score_10 >= 5.5:
        label = "中性偏多／等待確認"
        level = "info"

    elif score_10 >= 4:
        label = "中性偏弱／保守觀望"
        level = "warning"

    elif score_10 >= 2.5:
        label = "偏弱格局／不宜積極"
        level = "warning"

    else:
        label = "高風險／暫不建議介入"
        level = "error"

    return {
        "score": score,
        "score_10": score_10,
        "label": label,
        "level": level,
        "reasons": reasons[:6],
        "warnings": warnings[:6],
    }

def compute_backtest_stats(bt: pd.DataFrame) -> Dict:
    """根據 bt 內的 Return_1D、Signal、Strategy_Return 計算績效。"""
    if bt.empty or len(bt) < 30:
        return {}
    bt = bt.copy()
    bt["BuyHold_Equity"] = (1 + bt["Return_1D"]).cumprod()
    bt["Strategy_Equity"] = (1 + bt["Strategy_Return"]).cumprod()
    days = max((bt.index[-1] - bt.index[0]).days, 1)
    years = days / 365.25
    total_return = bt["Strategy_Equity"].iloc[-1] - 1
    buyhold_return = bt["BuyHold_Equity"].iloc[-1] - 1
    cagr = bt["Strategy_Equity"].iloc[-1] ** (1 / years) - 1 if bt["Strategy_Equity"].iloc[-1] > 0 else np.nan
    vol = bt["Strategy_Return"].std() * np.sqrt(252)
    sharpe = cagr / vol if vol and not pd.isna(vol) and vol != 0 else np.nan
    max_dd = (bt["Strategy_Equity"] / bt["Strategy_Equity"].cummax() - 1).min()
    active_returns = bt.loc[bt["Signal"].shift(1).fillna(0) == 1, "Strategy_Return"]
    win_rate = (active_returns > 0).mean() if len(active_returns) else np.nan
    exposure = bt["Signal"].mean()
    trades = int(((bt["Signal"].diff() == 1).sum()))
    return {
        "df": bt,
        "total_return": total_return,
        "buyhold_return": buyhold_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "exposure": exposure,
        "trades": trades,
    }


def run_backtest(df: pd.DataFrame, fast_ma: int = 20, slow_ma: int = 60) -> Dict:
    """簡單均線回測：fast_ma > slow_ma 時持有。"""
    df = df.copy()
    
    # 計算均線
    df["MA_fast"] = df["Close"].rolling(fast_ma).mean()
    df["MA_slow"] = df["Close"].rolling(slow_ma).mean()
    
    # 生成訊號（1 = 持有，0 = 空手）
    df["signal"] = 0
    df.loc[df["MA_fast"] > df["MA_slow"], "signal"] = 1
    
    # 前一天的持倉位置
    df["position"] = df["signal"].shift(1).fillna(0)
    
    # 計算日報酬
    df["return"] = df["Close"].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["return"]
    
    # 累積淨值
    df["equity"] = (1 + df["strategy_return"]).cumprod()
    
    # 買賣訊號：1=買進，-1=賣出，0=無動作
    df["trade"] = df["position"].diff()
    
    # 計算總報酬
    total_return = df["equity"].iloc[-1] - 1
    
    return {
        "result": df,
        "total_return": total_return,
    }


def plot_equity_curve_with_signals(df: pd.DataFrame) -> None:
    """繪製淨值曲線並標注買賣點。"""
    fig = go.Figure()
    
    # 淨值曲線
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["equity"],
        mode="lines",
        name="策略淨值",
        line=dict(width=2, color="#4472C4"),
        hovertemplate="淨值 %{y:.3f}<extra></extra>"
    ))
    
    # 買進點
    buy_points = df[df["trade"] == 1]
    if not buy_points.empty:
        fig.add_trace(go.Scatter(
            x=buy_points.index,
            y=buy_points["equity"],
            mode="markers",
            name="買進",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="#00B050"
            ),
            hovertemplate="買進<br>淨值 %{y:.3f}<extra></extra>"
        ))
    
    # 賣出點
    sell_points = df[df["trade"] == -1]
    if not sell_points.empty:
        fig.add_trace(go.Scatter(
            x=sell_points.index,
            y=sell_points["equity"],
            mode="markers",
            name="賣出",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="#FF0000"
            ),
            hovertemplate="賣出<br>淨值 %{y:.3f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="策略淨值 vs 日期",
        xaxis_title="日期",
        yaxis_title="淨值",
        hovermode="x unified",
        height=450,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def apply_backtest_execution(
    data: pd.DataFrame,
    signal: pd.Series,
    mode: str = "理想",
    cost_rate: float = 0.0,
) -> pd.DataFrame:
    """套用回測成交模式與交易成本。"""
    bt = data.copy()
    bt["Signal"] = signal.reindex(bt.index).fillna(0)

    # 進場或出場都算一次換手
    trade = bt["Signal"].diff().abs().fillna(0)

    if mode == "真實":
        # 今日收盤產生訊號，隔日開盤進場，吃隔日 Open -> Close 報酬
        bt["NextOpen_Return"] = bt["Close"] / bt["Open"] - 1
        bt["Strategy_Return"] = bt["Signal"].shift(1).fillna(0) * bt["NextOpen_Return"]
    else:
        # 原本邏輯：今日收盤產生訊號，隔日吃 Close -> Close 報酬
        bt["Strategy_Return"] = bt["Signal"].shift(1).fillna(0) * bt["Return_1D"]

    # 扣交易成本
    bt["Strategy_Return"] = bt["Strategy_Return"] - trade * cost_rate

    return bt


def position_from_entry_exit(entry: pd.Series, exit_: pd.Series, index: pd.Index) -> pd.Series:
    """用進場/出場條件轉成持有狀態。entry=True 進場，exit_=True 出場。"""
    holding = False
    signal = []
    entry = entry.reindex(index).fillna(False)
    exit_ = exit_.reindex(index).fillna(False)
    for e, x in zip(entry, exit_):
        if holding and x:
            holding = False
        if (not holding) and e:
            holding = True
        signal.append(1 if holding else 0)
    return pd.Series(signal, index=index, dtype=float)


STRATEGY_PRESETS: Dict[str, Dict[str, str]] = {
    "保守均線趨勢｜少交易": {
        "style": "保守／低頻／防守型",
        "rule": "Close > MA20 且 MA20 > MA60 才持有；否則空手。",
        "fit": "適合震盪或偏空環境，用來避開大跌；缺點是容易錯過飆股主升段。",
    },
    "長線大波段｜不太操作": {
        "style": "長期／低頻／偏存股",
        "rule": "Close > MA60 且 MA60 > MA120 才持有；否則空手。",
        "fit": "適合不想頻繁交易、重視大方向的人；反應慢，但雜訊較少。",
    },
    "EMA動能｜短線波段": {
        "style": "短中線／中頻／動能型",
        "rule": "EMA10 > EMA20 且 RSI > 50 持有；跌回 EMA10 < EMA20 或 RSI < 45 出場。",
        "fit": "適合高波動成長股，反應比 MA 快；缺點是盤整時容易被洗。",
    },
    "突破追價｜不要錯過飆股": {
        "style": "進攻／中高頻／飆股型",
        "rule": "收盤創近 20 日新高且量比 > 1.2 進場；跌破 MA20 或近 10 日低點出場。",
        "fit": "適合 AI、強勢題材、主升段股票；缺點是追高風險較高，需要嚴格停損。",
    },
    "RSI反轉｜頻繁操作搶反彈": {
        "style": "短線／高頻／反彈型",
        "rule": "RSI < 30 進場；RSI > 55 或 Close > MA20 出場。",
        "fit": "適合震盪盤搶反彈；不適合單邊下跌，容易越接越低。",
    },
    "布林下軌反彈｜有賺就好": {
        "style": "短線／中頻／均值回歸",
        "rule": "Close < BB 下軌進場；Close 回到 BB 中線出場。",
        "fit": "適合箱型震盪股票；遇到趨勢崩跌時要搭配停損。",
    },
}


def backtest_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    params: Optional[Dict] = None,
    execution_mode: str = "理想",
    cost_rate: float = 0.0,
) -> Dict:
    """多策略回測。所有策略都用昨日訊號決定今日持有，避免偷看未來。"""
    params = params or {}
    data = df.copy()
    min_cols = ["Return_1D", "MA20", "MA60", "RSI14", "MACD_HIST", "EMA10", "EMA20", "BB_LOWER", "BB_MID", "Volume_Ratio"]
    data = data.dropna(subset=[c for c in min_cols if c in data.columns]).copy()
    if len(data) < 60:
        return {}

    if strategy_name == "保守均線趨勢｜少交易":
        short = int(params.get("short_ma", 20))
        long = int(params.get("long_ma", 60))
        if f"MA{short}" not in data.columns:
            data[f"MA{short}"] = data["Close"].rolling(short).mean()
        if f"MA{long}" not in data.columns:
            data[f"MA{long}"] = data["Close"].rolling(long).mean()
        data = data.dropna(subset=[f"MA{short}", f"MA{long}"])
        signal = ((data["Close"] > data[f"MA{short}"]) & (data[f"MA{short}"] > data[f"MA{long}"])).astype(float)

    elif strategy_name == "長線大波段｜不太操作":
        short = int(params.get("short_ma", 60))
        long = int(params.get("long_ma", 120))
        if f"MA{short}" not in data.columns:
            data[f"MA{short}"] = data["Close"].rolling(short).mean()
        if f"MA{long}" not in data.columns:
            data[f"MA{long}"] = data["Close"].rolling(long).mean()
        data = data.dropna(subset=[f"MA{short}", f"MA{long}"])
        signal = ((data["Close"] > data[f"MA{short}"]) & (data[f"MA{short}"] > data[f"MA{long}"])).astype(float)

    elif strategy_name == "EMA動能｜短線波段":
        fast = int(params.get("fast_ema", 10))
        slow = int(params.get("slow_ema", 20))
        rsi_enter = float(params.get("rsi_enter", 50))
        rsi_exit = float(params.get("rsi_exit", 45))
        data[f"EMA{fast}"] = data["Close"].ewm(span=fast, adjust=False).mean()
        data[f"EMA{slow}"] = data["Close"].ewm(span=slow, adjust=False).mean()
        data = data.dropna(subset=[f"EMA{fast}", f"EMA{slow}", "RSI14"])
        entry = (data[f"EMA{fast}"] > data[f"EMA{slow}"]) & (data["RSI14"] > rsi_enter)
        exit_ = (data[f"EMA{fast}"] < data[f"EMA{slow}"]) | (data["RSI14"] < rsi_exit)
        signal = position_from_entry_exit(entry, exit_, data.index)

    elif strategy_name == "突破追價｜不要錯過飆股":
        lookback = int(params.get("breakout_n", 20))
        exit_ma = int(params.get("exit_ma", 20))
        volume_min = float(params.get("volume_min", 1.2))
        data["Breakout_High"] = data["Close"].rolling(lookback).max().shift(1)
        if f"MA{exit_ma}" not in data.columns:
            data[f"MA{exit_ma}"] = data["Close"].rolling(exit_ma).mean()
        data["Exit_Low"] = data["Close"].rolling(10).min().shift(1)
        data = data.dropna(subset=["Breakout_High", f"MA{exit_ma}", "Exit_Low", "Volume_Ratio"])
        entry = (data["Close"] > data["Breakout_High"]) & (data["Volume_Ratio"] >= volume_min)
        exit_ = (data["Close"] < data[f"MA{exit_ma}"]) | (data["Close"] < data["Exit_Low"])
        signal = position_from_entry_exit(entry, exit_, data.index)

    elif strategy_name == "RSI反轉｜頻繁操作搶反彈":
        enter = float(params.get("rsi_low", 30))
        exit_rsi = float(params.get("rsi_high", 55))
        entry = data["RSI14"] < enter
        exit_ = (data["RSI14"] > exit_rsi) | (data["Close"] > data["MA20"])
        signal = position_from_entry_exit(entry, exit_, data.index)

    elif strategy_name == "布林下軌反彈｜有賺就好":
        entry = data["Close"] < data["BB_LOWER"]
        exit_ = data["Close"] >= data["BB_MID"]
        signal = position_from_entry_exit(entry, exit_, data.index)

    else:
        return {}

    bt = apply_backtest_execution(
        data=data,
        signal=signal,
        mode=execution_mode,
        cost_rate=cost_rate,
    )

    return compute_backtest_stats(bt)


def backtest_all_strategies(
    df: pd.DataFrame,
    execution_mode: str = "理想",
    cost_rate: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for name, meta in STRATEGY_PRESETS.items():
        bt = backtest_strategy(
            df,
            name,
            execution_mode=execution_mode,
            cost_rate=cost_rate,
        )
        if not bt:
            continue
        rows.append({
            "策略": name,
            "定位": meta["style"],
            "策略總報酬%": round(bt["total_return"] * 100, 2),
            "買進持有%": round(bt["buyhold_return"] * 100, 2),
            "年化報酬%": round(bt["cagr"] * 100, 2),
            "最大回撤%": round(bt["max_dd"] * 100, 2),
            "Sharpe": round(bt["sharpe"], 2) if not pd.isna(bt["sharpe"]) else np.nan,
            "持股時間%": round(bt["exposure"] * 100, 2),
            "交易次數": bt["trades"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Sharpe", "策略總報酬%"], ascending=False, na_position="last")


def optimize_parameters(
    df: pd.DataFrame,
    strategy_family: str,
    execution_mode: str = "理想",
    cost_rate: float = 0.0,
    optimize_target: str = "穩健分數最高",
) -> pd.DataFrame:
    """依目前左側設定的資料期間做簡單參數最佳化。\n\n    注意：這是歷史區間最佳，不保證未來最佳；主要用來比較參數穩定性。\n    """
    rows = []
    if strategy_family == "均線趨勢 MA":
        for short in [5, 10, 20, 30, 60]:
            for long in [20, 60, 90, 120, 240]:
                if short >= long:
                    continue
                name = "保守均線趨勢｜少交易" if long <= 90 else "長線大波段｜不太操作"
                bt = backtest_strategy(
                    df,
                    name,
                    {"short_ma": short, "long_ma": long},
                    execution_mode=execution_mode,
                    cost_rate=cost_rate,
                )
                if not bt:
                    continue
                rows.append({"策略族": strategy_family, "參數": f"Close > MA{short} 且 MA{short} > MA{long}", "短均線": short, "長均線": long, **_opt_stats(bt)})

    elif strategy_family == "EMA動能":
        for fast in [5, 8, 10, 12, 15]:
            for slow in [20, 30, 50, 60]:
                if fast >= slow:
                    continue
                for rsi_enter in [45, 50, 55]:
                    bt = backtest_strategy(
                        df,
                        "EMA動能｜短線波段",
                        {"fast_ema": fast, "slow_ema": slow, "rsi_enter": rsi_enter, "rsi_exit": rsi_enter - 5},
                        execution_mode=execution_mode,
                        cost_rate=cost_rate,
                    )
                    if not bt:
                        continue
                    rows.append({"策略族": strategy_family, "參數": f"EMA{fast} > EMA{slow}, RSI>{rsi_enter}", "快EMA": fast, "慢EMA": slow, "RSI進場": rsi_enter, **_opt_stats(bt)})

    elif strategy_family == "突破追價":
        for n in [10, 20, 30, 55]:
            for exit_ma in [10, 20, 30]:
                for vol in [1.0, 1.2, 1.5]:
                    bt = backtest_strategy(
                        df,
                        "突破追價｜不要錯過飆股",
                        {"breakout_n": n, "exit_ma": exit_ma, "volume_min": vol},
                        execution_mode=execution_mode,
                        cost_rate=cost_rate,
                    )
                    if not bt:
                        continue
                    rows.append({"策略族": strategy_family, "參數": f"突破{n}日高, 量比>{vol}, 跌破MA{exit_ma}出場", "突破天數": n, "出場MA": exit_ma, "量比門檻": vol, **_opt_stats(bt)})

    elif strategy_family == "長線大波段 MA":
        for short in [40, 60, 90, 120]:
            for long in [120, 180, 240]:
                if short >= long:
                    continue
                bt = backtest_strategy(
                    df,
                    "長線大波段｜不太操作",
                    {"short_ma": short, "long_ma": long},
                    execution_mode=execution_mode,
                    cost_rate=cost_rate,
                )
                if not bt:
                    continue
                rows.append({"策略族": strategy_family, "參數": f"Close > MA{short} 且 MA{short} > MA{long}", "短均線": short, "長均線": long, **_opt_stats(bt)})

    elif strategy_family == "RSI反轉":
        for low in [20, 25, 30, 35]:
            for high in [50, 55, 60, 65]:
                if low >= high:
                    continue
                bt = backtest_strategy(
                    df,
                    "RSI反轉｜頻繁操作搶反彈",
                    {"rsi_low": low, "rsi_high": high},
                    execution_mode=execution_mode,
                    cost_rate=cost_rate,
                )
                if not bt:
                    continue
                rows.append({"策略族": strategy_family, "參數": f"RSI<{low}進場, RSI>{high}或站上MA20出場", "RSI低檔": low, "RSI出場": high, **_opt_stats(bt)})

    elif strategy_family == "布林反彈":
        # 目前布林策略的結構固定，這裡用不同布林倍數重算後測試。
        for k in [1.5, 2.0, 2.5, 3.0]:
            tmp = df.copy()
            tmp["BB_UPPER"] = tmp["BB_MID"] + k * tmp["BB_STD"]
            tmp["BB_LOWER"] = tmp["BB_MID"] - k * tmp["BB_STD"]
            bt = backtest_strategy(
                tmp,
                "布林下軌反彈｜有賺就好",
                execution_mode=execution_mode,
                cost_rate=cost_rate,
            )
            if not bt:
                continue
            rows.append({"策略族": strategy_family, "參數": f"Close < BB下軌({k}σ)進場, 回中線出場", "布林倍數": k, **_opt_stats(bt)})

    elif strategy_family == "全部策略":
        for fam in ["均線趨勢 MA", "長線大波段 MA", "EMA動能", "突破追價", "RSI反轉", "布林反彈"]:
            part = optimize_parameters(
                df,
                fam,
                execution_mode=execution_mode,
                cost_rate=cost_rate,
            )
            if not part.empty:
                rows.extend(part.to_dict("records"))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # 排序不是只看報酬，避免選到很會暴衝但回撤超大的參數。
    out["穩健分數"] = out["Sharpe"].fillna(0) * 2 + out["年化報酬%"].fillna(0) / 20 + out["最大回撤%"].fillna(-100) / 20
    
    if optimize_target == "報酬最高":
        out = out.sort_values("策略總報酬%", ascending=False)
    else:
        out = out.sort_values("穩健分數", ascending=False)
    
    return out.head(20)


def _opt_stats(bt: Dict) -> Dict:
    return {
        "策略總報酬%": round(bt["total_return"] * 100, 2),
        "年化報酬%": round(bt["cagr"] * 100, 2),
        "最大回撤%": round(bt["max_dd"] * 100, 2),
        "Sharpe": round(bt["sharpe"], 2) if not pd.isna(bt["sharpe"]) else np.nan,
        "持股時間%": round(bt["exposure"] * 100, 2),
        "交易次數": bt["trades"],
    }

def make_price_chart(df: pd.DataFrame, rows: int = 260, close_overlay: pd.Series = None):
    """專業 K 線圖：台股紅漲綠跌、疊加 MA、布林通道、成交量與 MACD 副圖。"""
    plot = df.tail(rows).copy().reset_index()
    date_col = plot.columns[0]
    plot = plot.rename(columns={date_col: "日期"})

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.62, 0.18, 0.20],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # =========================
    # 布林通道：上下軌灰色區域 + 紫色中線
    # =========================
    if {"BB_UPPER", "BB_LOWER", "BB_MID"}.issubset(plot.columns):
        fig.add_trace(
            go.Scatter(
                x=plot["日期"],
                y=plot["BB_UPPER"],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
                name="布林上軌",
                hovertemplate="布林上軌 %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot["日期"],
                y=plot["BB_LOWER"],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(160,160,160,0.18)",
                name="布林下軌",
                hovertemplate="布林下軌 %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot["日期"],
                y=plot["BB_MID"],
                mode="lines",
                line=dict(color="#B455FF", width=2, dash="dash"),
                name="布林中線",
                hovertemplate="布林中線 %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # =========================
    # K 線：台股慣例，漲紅跌綠
    # =========================
    fig.add_trace(
        go.Candlestick(
            x=plot["日期"],
            open=plot["Open"],
            high=plot["High"],
            low=plot["Low"],
            close=plot["Close"],
            name="K線",
            increasing=dict(line=dict(color="#D32F2F"), fillcolor="#D32F2F"),
            decreasing=dict(line=dict(color="#00A65A"), fillcolor="#00A65A"),
            hovertemplate=(
                "開盤 %{open:,.2f}<br>最高 %{high:,.2f}"
                "<br>最低 %{low:,.2f}<br>收盤 %{close:,.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # =========================
    # 均線
    # =========================
    ma_specs = [
        ("MA5", "週線 MA5", "#FFD400"),
        ("MA20", "月線 MA20", "#00A65A"),
        ("MA60", "季線 MA60", "#4FC3F7"),
        ("MA120", "半年線 MA120", "#0D47A1"),
        ("MA240", "年線 MA240", "#8B5A2B"),
    ]
    for col_name, display_name, color in ma_specs:
        if col_name in plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot["日期"],
                    y=plot[col_name],
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    name=display_name,
                    hovertemplate=f"{display_name} %{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # =========================
    # 成交量
    # =========================
    volume_colors = np.where(plot["Close"] >= plot["Open"], "#D32F2F", "#00A65A")
    fig.add_trace(
        go.Bar(
            x=plot["日期"],
            y=plot["Volume"],
            name="成交量",
            marker_color=volume_colors,
            opacity=0.45,
            hovertemplate="成交量 %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # =========================
    # MACD 副圖：DIF / MACD Signal / Histogram / 零軸
    # =========================
    if {"MACD_DIF", "MACD_SIGNAL", "MACD_HIST"}.issubset(plot.columns):
        hist_colors = np.where(plot["MACD_HIST"] >= 0, "#16A34A", "#EF4444")

        fig.add_trace(
            go.Bar(
                x=plot["日期"],
                y=plot["MACD_HIST"],
                name="Histogram",
                marker_color=hist_colors,
                opacity=0.8,
                hovertemplate="Histogram %{y:,.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot["日期"],
                y=plot["MACD_DIF"],
                mode="lines",
                line=dict(color="#1E90FF", width=2),
                name="DIF",
                hovertemplate="DIF %{y:,.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot["日期"],
                y=plot["MACD_SIGNAL"],
                mode="lines",
                line=dict(color="#FF8C00", width=2),
                name="MACD",
                hovertemplate="MACD %{y:,.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray",
            opacity=0.7,
            row=3,
            col=1,
        )

    # =========================
    # Layout：hover 不再用巨大 unified box，改為十字線對齊
    # =========================
    fig.update_layout(
        height=900,
        hovermode="x",
        spikedistance=-1,
        hoverdistance=100,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    
    fig.update_xaxes(
        matches="x",
        showspikes=True,
        spikecolor="black",
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )

    fig.update_yaxes(
        showspikes=True,
        spikecolor="black",
        spikemode="across",
        spikethickness=1,
    )

    fig.update_yaxes(title_text="股價", row=1, col=1, fixedrange=False)
    fig.update_yaxes(title_text="成交量", row=2, col=1, fixedrange=False)
    fig.update_yaxes(title_text="MACD", row=3, col=1, fixedrange=False)

    fig.update_xaxes(
        showspikes=True,
        spikecolor="black",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        showline=True,
        linewidth=1,
        linecolor="rgba(0,0,0,0.15)",
    )
    fig.update_yaxes(
        showspikes=True,
        spikecolor="black",
        spikemode="across",
        spikethickness=1,
        showline=True,
        linewidth=1,
        linecolor="rgba(0,0,0,0.15)",
    )

    return fig

def make_backtest_chart(bt_df: pd.DataFrame):
    plot = bt_df.reset_index()
    date_col = plot.columns[0]
    plot = plot.rename(columns={date_col: "Date"})
    long = plot.melt("Date", value_vars=["BuyHold_Equity", "Strategy_Equity"], var_name="策略", value_name="淨值")
    return (
        alt.Chart(long)
        .mark_line()
        .encode(x=alt.X("Date:T", title="日期"), y=alt.Y("淨值:Q", title="淨值"), color=alt.Color("策略:N", scale=alt.Scale(domain=["BuyHold_Equity", "Strategy_Equity"], range=["#0B63CE", "#7DBDFF"]), legend=alt.Legend(title="策略")))
        .properties(height=320)
    )


def build_financial_table(stmt: pd.DataFrame, ticker_symbol: Optional[str] = None) -> pd.DataFrame:
    """整理 FinMind 近 8 季財報。若資料表欄位不符，則回傳空表避免 App 中斷。"""
    if stmt is None or stmt.empty:
        return pd.DataFrame()
    data = stmt.copy()
    if not {"date", "type", "value"}.issubset(data.columns):
        return pd.DataFrame()

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
    out = out[[c for c in order if c in out.columns]]
    return out.round(2)


def risk_plan(last: pd.Series, capital: float, risk_pct: float) -> Dict:
    close = safe_float(last["Close"])
    atr = safe_float(last["ATR14"])
    if pd.isna(close) or pd.isna(atr) or atr <= 0:
        return {}
    stop_loss = max(close - 2 * atr, 0)
    take_profit = close + 3 * atr
    risk_per_share = close - stop_loss
    risk_budget = capital * risk_pct / 100
    shares = int(risk_budget // risk_per_share) if risk_per_share > 0 else 0
    # 台股通常以張為主，這裡同步給股與張
    lots = shares / 1000
    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_budget": risk_budget,
        "shares": shares,
        "lots": lots,
        "risk_per_share": risk_per_share,
    }

def strategy_execution_advice(df: pd.DataFrame, strategy_name: str, params: Optional[Dict] = None) -> Dict:
    params = params or {}
    data = df.copy()
    last = data.iloc[-1]
    prev = data.iloc[-2]
    close = safe_float(last["Close"])

    status = "等待進場"
    action = "目前尚未符合進場條件。"
    entry = "—"
    exit_rule = "—"
    key_level = np.nan
    distance_pct = np.nan
    reasons = []

    if strategy_name == "保守均線趨勢｜少交易":
        short = int(params.get("short_ma", 20))
        long = int(params.get("long_ma", 60))
        short_col = f"MA{short}"
        long_col = f"MA{long}"

        if short_col not in data.columns:
            data[short_col] = data["Close"].rolling(short).mean()
        if long_col not in data.columns:
            data[long_col] = data["Close"].rolling(long).mean()

        last = data.iloc[-1]
        ma_s = safe_float(last.get(short_col))
        ma_l = safe_float(last.get(long_col))

        entry = f"收盤價 > MA{short}，且 MA{short} > MA{long}"
        exit_rule = f"收盤價跌破 MA{short}，或 MA{short} 轉弱低於 MA{long}"
        key_level = ma_s

        if close > ma_s and ma_s > ma_l:
            status = "可持有"
            action = f"目前已符合策略條件。若尚未進場，可等下一交易日確認未跌破 MA{short} 再考慮進場。"
        else:
            action = f"目前尚未符合條件，可等待收盤價重新站上 MA{short}，且 MA{short} 高於 MA{long}。"

        reasons.append(f"現價 {close:.2f}，MA{short}={ma_s:.2f}，MA{long}={ma_l:.2f}")

    elif strategy_name == "長線大波段｜不太操作":
        short = int(params.get("short_ma", 60))
        long = int(params.get("long_ma", 120))
        short_col = f"MA{short}"
        long_col = f"MA{long}"

        if short_col not in data.columns:
            data[short_col] = data["Close"].rolling(short).mean()
        if long_col not in data.columns:
            data[long_col] = data["Close"].rolling(long).mean()

        last = data.iloc[-1]
        ma_s = safe_float(last.get(short_col))
        ma_l = safe_float(last.get(long_col))

        entry = f"收盤價 > MA{short}，且 MA{short} > MA{long}"
        exit_rule = f"收盤價跌破 MA{short}，或 MA{short} 跌破 MA{long}"
        key_level = ma_s

        if close > ma_s and ma_s > ma_l:
            status = "可持有"
            action = f"目前符合長線波段條件，可視為持有區；若尚未進場，建議等回測 MA{short} 不破或隔日續強再進。"
        else:
            action = f"目前尚未符合長線條件，可等待收盤價站回 MA{short} 且 MA{short} 高於 MA{long}。"

        reasons.append(f"現價 {close:.2f}，MA{short}={ma_s:.2f}，MA{long}={ma_l:.2f}")

    elif strategy_name == "EMA動能｜短線波段":
        fast = int(params.get("fast_ema", 10))
        slow = int(params.get("slow_ema", 20))
        rsi_enter = float(params.get("rsi_enter", 50))
        rsi_exit = float(params.get("rsi_exit", 45))

        data[f"EMA{fast}"] = data["Close"].ewm(span=fast, adjust=False).mean()
        data[f"EMA{slow}"] = data["Close"].ewm(span=slow, adjust=False).mean()
        last = data.iloc[-1]

        ema_f = safe_float(last[f"EMA{fast}"])
        ema_s = safe_float(last[f"EMA{slow}"])
        rsi = safe_float(last["RSI14"])

        entry = f"EMA{fast} > EMA{slow}，且 RSI > {rsi_enter}"
        exit_rule = f"EMA{fast} < EMA{slow}，或 RSI < {rsi_exit}"
        key_level = ema_s

        if ema_f > ema_s and rsi > rsi_enter:
            status = "可持有"
            action = "目前動能條件成立。若尚未進場，可等下一根K線仍維持 EMA 多頭排列再進。"
        else:
            action = f"目前動能尚未完整成立，可等待 EMA{fast} 上穿 EMA{slow} 且 RSI 站上 {rsi_enter}。"

        reasons.append(f"EMA{fast}={ema_f:.2f}，EMA{slow}={ema_s:.2f}，RSI={rsi:.1f}")

    elif strategy_name == "突破追價｜不要錯過飆股":
        lookback = int(params.get("breakout_n", 20))
        exit_ma = int(params.get("exit_ma", 20))
        volume_min = float(params.get("volume_min", 1.2))

        breakout_high = data["Close"].rolling(lookback).max().shift(1).iloc[-1]
        exit_low = data["Close"].rolling(10).min().shift(1).iloc[-1]
        ma_exit = data["Close"].rolling(exit_ma).mean().iloc[-1]
        vol_ratio = safe_float(last["Volume_Ratio"])

        entry = f"收盤價突破近 {lookback} 日高點，且量比 > {volume_min}"
        exit_rule = f"跌破 MA{exit_ma}，或跌破近 10 日低點"
        key_level = breakout_high

        if close > breakout_high and vol_ratio >= volume_min:
            status = "可進場"
            action = "目前已符合突破進場條件。若要執行，應特別設定停損，避免假突破。"
        else:
            action = f"目前尚未突破。可等待收盤價突破 {breakout_high:.2f}，且量比大於 {volume_min}。"

        reasons.append(f"現價 {close:.2f}，突破價 {breakout_high:.2f}，量比 {vol_ratio:.2f}")
        reasons.append(f"出場參考：MA{exit_ma}={ma_exit:.2f}，10日低點={exit_low:.2f}")

    elif strategy_name == "RSI反轉｜頻繁操作搶反彈":
        rsi_low = float(params.get("rsi_low", 30))
        rsi_high = float(params.get("rsi_high", 55))
        rsi = safe_float(last["RSI14"])
        ma20 = safe_float(last["MA20"])

        entry = f"RSI < {rsi_low}"
        exit_rule = f"RSI > {rsi_high}，或收盤價站上 MA20"
        key_level = ma20

        if rsi < rsi_low:
            status = "觀察反彈"
            action = "目前 RSI 進入低檔反彈區，但仍需注意是否為下跌趨勢中的弱反彈。"
        else:
            action = f"目前 RSI 尚未低於 {rsi_low}，不符合反彈策略進場條件。"

        reasons.append(f"RSI={rsi:.1f}，MA20={ma20:.2f}")

    elif strategy_name == "布林下軌反彈｜有賺就好":
        bb_lower = safe_float(last["BB_LOWER"])
        bb_mid = safe_float(last["BB_MID"])

        entry = "收盤價跌破布林下軌"
        exit_rule = "收盤價回到布林中線"
        key_level = bb_lower

        if close < bb_lower:
            status = "觀察反彈"
            action = "目前已跌破布林下軌，符合均值回歸進場條件，但需避免接到趨勢下跌。"
        else:
            action = f"目前尚未跌破布林下軌，可等待價格接近或跌破 {bb_lower:.2f}。"

        reasons.append(f"現價 {close:.2f}，布林下軌={bb_lower:.2f}，布林中線={bb_mid:.2f}")

    if not pd.isna(key_level) and key_level > 0:
        distance_pct = (close / key_level - 1) * 100

    return {
        "status": status,
        "action": action,
        "entry": entry,
        "exit_rule": exit_rule,
        "key_level": key_level,
        "distance_pct": distance_pct,
        "reasons": reasons,
    }


def mode_difference_table() -> pd.DataFrame:
    return pd.DataFrame({
        "面向": ["核心目的", "主要看什麼", "加分條件", "扣分條件", "風控邏輯"],
        "短線／波段": [
            "抓 1–8 週價差與動能延續",
            "MA5、MA20、MACD、RSI、量比、20日強弱",
            "短均線向上、MACD擴大、放量上漲、RSI偏強未過熱",
            "跌破短均線、MACD轉弱、放量下跌、RSI過熱",
            "停損較嚴格，重視 ATR 與隔日跳空風險",
        ],
        "長線／存股": [
            "看半年以上趨勢、估值與現金流",
            "MA60、MA120、MA240、PE、EPS、殖利率、Beta",
            "站上季線/年線、PE合理、EPS為正、殖利率佳",
            "跌破季線/年線、估值過高、EPS弱、Beta過高",
            "可分批布局，但仍需限制單筆最大損失",
        ],
    })

# ===== 更新公告文字 =====
CHANGELOG_TEXT = """
2026.05.15 新增 AI 潛力股選股
2026.05.15 資料區間改為自訂開始日期～結束日期，回測新增自訂參數模式與買賣點標注
2026.05.15 最佳化參數修改為穩健分數最高、報酬最高兩種模式
2026.05.15 回測功能修改為理想與真實兩種模式，並加入交易成本估算
2026.05.14 新增 AI 妖股選股
2026.05.14 技術圖表新增副圖MACD
2026.05.13 新增底部更新公告區塊、意見回饋表單
"""

def render_changelog(changelog_text):
    items = []

    for line in changelog_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"(\d{4})\.(\d{2})\.(\d{2})\s+(.+)", line)

        if match:
            year, month, day, content = match.groups()
            date = f"{year}-{month}-{day}"
            items.append((date, content))

    html = """
<div style="max-height:220px; overflow-y:auto; padding:12px 16px; border:1px solid #e5e7eb; border-radius:12px; background-color:#fafafa; line-height:1.7;">
"""

    for date, content in items:
        html += f"""
<div style="margin-bottom:14px;">
    <span style="font-weight:700; color:#111827;">
        📌 {content}
    </span>
    <span style="font-size:0.82rem; color:#6b7280; margin-left:8px;">
        {date}
    </span>
</div>
"""

    html += """
</div>
"""

    return html

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# @st.cache_data(ttl=60 * 60 * 24)
# def get_tw_stock_list():
#     stock_dict = {}
#     headers = {"User-Agent": "Mozilla/5.0"}

#     for m in [2, 4]:
#         url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
#         res = requests.get(url, headers=headers, verify=False, timeout=15)

#         df = pd.read_html(res.text)[0].iloc[1:]

#         for _, row in df.iterrows():
#             try:
#                 code_name = str(row[0]).split()
#                 if len(code_name) != 2:
#                     continue

#                 code, name = code_name
#                 cat = str(row[4])

#                 if len(code) == 4 or code.startswith("00"):
#                     if cat not in ["權證", "牛熊證", "認購(售)權證"]:
#                         suffix = ".TW" if m == 2 else ".TWO"
#                         stock_dict[f"{code}{suffix}"] = {
#                             "name": name,
#                             "industry": cat,
#                         }
#             except Exception:
#                 continue

#     return stock_dict
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_tw_stock_list():
    stock_dict = {}

    try:
        info_df = finmind_request("TaiwanStockInfo")

        if info_df.empty:
            return {
                "2330.TW": {"name": "台積電", "industry": "半導體業"},
                "2317.TW": {"name": "鴻海", "industry": "其他電子業"},
                "2454.TW": {"name": "聯發科", "industry": "半導體業"},
                "0050.TW": {"name": "元大台灣50", "industry": "ETF"},
            }

        for _, row in info_df.iterrows():
            code = str(row.get("stock_id", "")).strip()
            name = str(row.get("stock_name", "")).strip()
            industry = str(row.get("industry_category", "")).strip()
            market = str(row.get("type", "")).strip()

            if not code or not name:
                continue

            if len(code) == 4 or code.startswith("00"):
                suffix = ".TWO" if "上櫃" in market else ".TW"
                stock_dict[f"{code}{suffix}"] = {
                    "name": name,
                    "industry": industry if industry else market,
                }

        return stock_dict

    except Exception:
        return {
            "2330.TW": {"name": "台積電", "industry": "半導體業"},
            "2317.TW": {"name": "鴻海", "industry": "其他電子業"},
            "2454.TW": {"name": "聯發科", "industry": "半導體業"},
            "0050.TW": {"name": "元大台灣50", "industry": "ETF"},
        }
# =========================
# 介面
# =========================
st.title("📈 台股投資分析")
st.caption("免責聲明：本平台僅供學習與研究參考，請自行判斷並注意投資風險。")
with st.sidebar:
    st.header("股票設定")

    raw_code = st.text_input(
        "股票代碼",
        value="2330",
        help="可輸入 2330、8069、0050，也可輸入 2330.TW/8069.TWO"
    )

    st.subheader("資料區間設定")
    today = pd.Timestamp.today().date()
    
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "開始日期",
            value=today - pd.DateOffset(years=1),
            format="YYYY-MM-DD"
        )
    with col_end:
        end_date = st.date_input(
            "結束日期",
            value=today,
            format="YYYY-MM-DD"
        )
    
    if start_date >= end_date:
        st.error("開始日期必須早於結束日期")
        st.stop()
    
    st.caption(f"分析區間：{start_date} ～ {end_date}")
    mode = st.radio("操作模式", ["短線／波段", "長線／存股"], horizontal=False)
    capital = st.number_input("帳戶資金（元）", min_value=1, value=100000, step=10000)
    risk_pct = st.number_input("單筆最大風險 %", min_value=0.01, max_value=100.0, value=1.0, step=0.1, format="%.2f", help="代表這一筆交易最多願意虧掉帳戶資金的百分比，例如 1% 表示 10 萬帳戶最多虧 1000 元。")
    analyze = st.button("更新並分析", type="primary", use_container_width=True)
    st.divider()
    watchlist_text = st.text_area("觀察清單", value=DEFAULT_WATCHLIST, height=100)
    run_watchlist = st.button("掃描觀察清單", use_container_width=True)
    st.caption("可輸入多檔股票製作觀察清單")

# 預設初次也分析，避免使用者打開空白頁
if not analyze and not run_watchlist:
    analyze = True

if analyze:
    ticker_input = normalize_tw_ticker(raw_code)
    try:
        with st.spinner("正在取得最新股價與財務資料..."):
            # 1. 抓取原始資料 (使用用戶指定的日期範圍)
            raw_df, resolved_ticker = load_price_data(ticker_input)
            
            # 2. 按照用戶選擇的日期範圍裁切
            # 轉換 start_date 和 end_date 為 Timestamp
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            # 裁切到用戶指定的範圍
            raw_df = raw_df[(raw_df.index >= start_ts) & (raw_df.index <= end_ts)].copy()
            
            if raw_df.empty:
                st.error(f"在 {start_date} 到 {end_date} 期間內無法取得資料，請檢查日期範圍或股票代碼。")
                st.stop()

            info = load_ticker_info(resolved_ticker)
            fast_info = load_fast_info(resolved_ticker)
            stmt = load_financials(resolved_ticker)
            
            # 3. 計算指標
            full_df = calculate_indicators(raw_df)
            
            # 4. 使用完整計算後的資料
            df = full_df.dropna(
                subset=["Close", "RSI14", "MACD_HIST"]
            )
            last_raw = full_df.iloc[-1]   # 最新一天（含今天未收盤完整資料）
            prev_raw = full_df.iloc[-2]   # 前一天
            
        if df.empty or len(df) < 20:
            st.error("資料量不足，無法計算完整指標。請增加月份、改用更長期間，或確認股票代碼。")
            st.stop()
        if len(df) < 60:
            st.warning("目前分析期間較短，部分長週期指標與回測結果會比較不穩定；若要看中長期策略，建議至少 1 年以上。")

        last = last_raw
        prev = prev_raw
        latest_date = full_df.index[-1].strftime("%Y-%m-%d")
        stale_days = (datetime.now() - full_df.index[-1]).days
        company_name = info.get("longName") or info.get("shortName") or display_code(resolved_ticker)

        st.subheader(f"{company_name}（{resolved_ticker}）")
        if stale_days >= 5:
            st.warning(f"最新資料日期為 {latest_date}，距今已 {stale_days} 天；可能遇到假日、停牌或資料源延遲。")
        else:
            st.caption(f"最新交易日：{latest_date}")

        score = score_stock(df, info, mode)
        score_text = (
            f"綜合評分：{score['score_10']}/10"
            f"｜{score['label']}"
        )

        if score["level"] == "success":
            st.success(score_text)
        elif score["level"] == "error":
            st.error(score_text)
        elif score["level"] == "warning":
            st.warning(score_text)
        else:
            st.info(score_text)
            
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        price_change = safe_float(last["Close"] - prev["Close"])
        price_change_pct = safe_float(last["Return_1D"] * 100)
        if price_change >= 0:
            delta_color = "normal"   # 台股：漲=紅
        else:
            delta_color = "inverse"  # 台股：跌=綠

        c1.metric(
            "收盤價",
            format_number(last["Close"], 2),
            delta=f"{price_change:+,.2f} ({price_change_pct:+.2f}%)",
            delta_color=delta_color,
        )
        c2.metric("RSI14", format_number(last["RSI14"], 1))
        c3.metric("MACD柱", format_number(last["MACD_HIST"], 2))
        c4.metric("20日量比", format_number(last["Volume_Ratio"], 2))
        c5.metric("20日年化波動", format_number(last["Volatility_20D"] * 100, 1, "%"))
        c6.metric("目前回撤", format_number(last["Drawdown"] * 100, 1, "%"))

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
            ["總覽", "技術圖表", "基本面", "風險控管", "歷史回測", "策略執行", "模式差異", "AI妖股選股","AI潛力股選股"]
            )        
        with tab1:
            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### 主要支持理由")
                if score["reasons"]:
                    for item in score["reasons"]:
                        st.write(f"✅ {item}")
                else:
                    st.write("目前沒有明顯偏多理由。")
            with right:
                st.markdown("#### 主要風險提醒")
                if score["warnings"]:
                    for item in score["warnings"]:
                        st.write(f"⚠️ {item}")
                else:
                    st.write("目前沒有明顯技術警訊，但仍需控管部位。")

            st.markdown("#### 關鍵價位")
            levels = pd.DataFrame(
                {
                    "項目": ["MA20（月線）", "MA60（季線）", "MA120（半年線）", "52週高點", "52週低點", "ATR14"],
                    "數值": [last["MA20"], last["MA60"], last["MA120"], last["High_52W"], last["Low_52W"], last["ATR14"]],
                    "與現價距離%": [
                        (last["MA20"] / last["Close"] - 1) * 100,
                        (last["MA60"] / last["Close"] - 1) * 100,
                        (last["MA120"] / last["Close"] - 1) * 100,
                        (last["High_52W"] / last["Close"] - 1) * 100,
                        (last["Low_52W"] / last["Close"] - 1) * 100,
                        np.nan,
                    ],
                }
            )
            st.dataframe(levels, hide_index=True, use_container_width=True)

        with tab2:
            st.plotly_chart(make_price_chart(df, close_overlay=full_df["Close"]), use_container_width=True)
            st.markdown("#### 指標明細")
            indicator_cols = ["Close", "Volume", "MA5", "MA20", "MA60", "MA120", "MA240", "BB_UPPER", "BB_MID", "BB_LOWER", "RSI14", "MACD_DIF", "MACD_SIGNAL", "MACD_HIST", "ATR14", "Volume_Ratio", "Return_5D", "Return_20D"]
            st.dataframe(df[indicator_cols].tail(30).iloc[::-1], use_container_width=True)

        with tab3:
            fin_table = build_financial_table(stmt, resolved_ticker)
            fund = derive_fundamental_metrics(info, fast_info, last["Close"], fin_table)
            # ===== 基本面資料 =====
            pe = safe_float(info.get("trailingPE"))

            # FinMind 的 dividend_yield 已經是 %
            dividend_yield = dividend_yield_pct(info, last["Close"])

            # 用 股價 / PE 反推 EPS
            eps = np.nan
            if not pd.isna(pe) and pe > 0:
                eps = last["Close"] / pe

            # Beta 若沒有則自行估算
            beta = safe_float(info.get("beta"))
            if pd.isna(beta):
                beta = estimate_beta_vs_twii(full_df)

            # ===== 顯示 =====
            f1, f2, f3, f4 = st.columns(4)

            f1.metric("本益比 PE", format_number(pe, 2))
            f2.metric("EPS", format_number(eps, 2))
            f3.metric("殖利率", format_number(dividend_yield, 2, "%"))
            f4.metric("Beta", format_number(beta, 2))

            st.caption(
                "PE、殖利率優先使用 FinMind；EPS 以股價 ÷ PE 推估；"
                "Beta 若無資料則以相對加權指數 TAIEX 日報酬估算。"
            )
            if fin_table.empty:
                st.info("此股票目前無法從 FinMind 取得完整季財務資料。")
            else:
                st.markdown("#### 財務趨勢")
                st.caption("QoQ 是季增率，YoY 是與去年同季相比。")
                st.dataframe(fin_table.iloc[::-1], use_container_width=True)
                if "營收QoQ%" in fin_table.columns and not fin_table["營收QoQ%"].dropna().empty:
                    latest_growth = fin_table["營收QoQ%"].dropna().iloc[-1]
                    if latest_growth > 0:
                        st.success(f"最新一季營收季增率約 {latest_growth:.2f}%，基本面短期動能偏正向。")
                    else:
                        st.warning(f"最新一季營收季增率約 {latest_growth:.2f}%，需確認是淡季、循環下滑或公司特殊因素。")

        with tab4:
            st.markdown("#### 單筆交易風險規劃")
            st.info(f"目前設定單筆最大風險為 {risk_pct:.2f}% ; 帳戶資金為 {capital:,.0f} 元，這筆交易理論上最多承受約 {capital * risk_pct / 100:,.0f} 元損失。")
            plan = risk_plan(last, capital, risk_pct)
            if not plan:
                st.info("ATR 資料不足，暫無法建立風險規劃。")
            else:
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("建議停損", format_number(plan["stop_loss"], 2))
                r2.metric("參考停利", format_number(plan["take_profit"], 2))
                r3.metric("可承受損失", format_number(plan["risk_budget"], 0))
                r4.metric("估算張數", f"{plan['lots']:.2f} 張")
                r5.metric("估算股數", f"{plan['shares']:,} 股")
                st.caption("單筆最大風險＝這一筆交易最多願意虧掉帳戶資金的比例。停損以 2×ATR、停利以 3×ATR 粗估；實務上請再搭配支撐壓力、財報事件與大盤環境調整。")

            st.markdown("#### 交易前檢核")
            checklist = [
                "是否確認最新資料日期不是過舊？",
                "是否知道本次進場理由是趨勢、反彈、存股，還是事件交易？",
                "是否先設定停損價、停利價與最大損失？",
                "若明天開低，是否仍能接受這個部位大小？",
                "是否避開重大財報、法說、除權息或政策事件前後的非預期風險？",
            ]
            for x in checklist:
                st.checkbox(x, value=False)

        with tab5:
            st.markdown("#### 多策略回測中心")
            st.info("檢查：不同交易規則在過去同一段資料期間內表現如何。")

            backtest_mode = st.selectbox(
                "回測模式",
                ["理想", "真實"],
                index=0,
                help="理想模式：假設訊號出現後，能完整參與隔天漲跌 ; 真實模式：今日收盤看到訊號，隔天開盤才實際進場。"
            )

            cost_pct = st.number_input(
                "單次換手成本％",
                min_value=0.0,
                max_value=2.0,
                value=0.30,
                step=0.05,
                format="%.2f",
                help="簡化估算手續費、交易稅與滑價。0.30 代表每次買進或賣出扣 0.3%。"
            )

            cost_rate = cost_pct / 100

            compare_df = backtest_all_strategies(
                df,
                execution_mode=backtest_mode,
                cost_rate=cost_rate,
            )
            if compare_df.empty:
                st.info("資料不足，無法進行多策略回測。建議增加左側資料期間。")
            else:
                st.markdown("##### 策略比較表")
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

                strategy_name = st.selectbox("選擇要畫圖與細看的策略", list(STRATEGY_PRESETS.keys()), index=0)
                meta = STRATEGY_PRESETS[strategy_name]
                st.caption(f"策略定位：{meta['style']}｜規則：{meta['rule']}｜適用情境：{meta['fit']}")

                bt = backtest_strategy(
                    df,
                    strategy_name,
                    execution_mode=backtest_mode,
                    cost_rate=cost_rate,
                )
                if not bt:
                    st.info("此策略在目前資料期間內資料不足，無法回測。")
                else:
                    b1, b2, b3, b4, b5, b6, b7 = st.columns(7)
                    b1.metric("策略總報酬", format_number(bt["total_return"] * 100, 1, "%"))
                    b2.metric("買進持有", format_number(bt["buyhold_return"] * 100, 1, "%"))
                    b3.metric("年化報酬", format_number(bt["cagr"] * 100, 1, "%"))
                    b4.metric("最大回撤", format_number(bt["max_dd"] * 100, 1, "%"))
                    b5.metric("Sharpe", format_number(bt["sharpe"], 2))
                    b6.metric("持股時間", format_number(bt["exposure"] * 100, 1, "%"))
                    b7.metric("交易次數", f"{bt['trades']} 次")
                    st.altair_chart(make_backtest_chart(bt["df"]), use_container_width=True)

                    if bt["total_return"] > bt["buyhold_return"] and bt["max_dd"] > -0.25:
                        st.success("此期間策略相對買進持有有加值，且最大回撤尚可；可再用更長區間與其他股票驗證。")
                    elif bt["total_return"] < bt["buyhold_return"]:
                        st.warning("此期間策略低於買進持有，可能太保守、太慢進場，或不適合這類強勢股。")
                    else:
                        st.info("此期間策略有一定效果，但仍需觀察回撤、交易次數與是否過度依賴單一行情。")

                st.markdown("---")
                st.markdown("#### 參數最佳化")
                st.caption("依照左側輸入的資料期間尋找歷史表現較穩健的參數組合。這不是保證未來最佳，只是用來避免憑感覺設定 MA20/MA60。")
                optimize_target = st.selectbox(
                    "最佳化目標",
                    ["穩健分數最高", "報酬最高"],
                    help="""穩健分數公式：

Sharpe × 2 + 年化報酬％ / 20 + 最大回撤％ / 20

（最大回撤為負值，因此回撤越大分數越低）"""
                )
                opt_family = st.selectbox("選擇要最佳化的策略族", ["全部策略", "均線趨勢 MA", "長線大波段 MA", "EMA動能", "突破追價", "RSI反轉", "布林反彈"])
                run_opt = st.button("執行參數最佳化", type="secondary", use_container_width=True)
                if run_opt:
                    with st.spinner("正在測試不同參數組合..."):
                        opt_df = optimize_parameters(
                            df,
                            opt_family,
                            execution_mode=backtest_mode,
                            cost_rate=cost_rate,
                            optimize_target=optimize_target,
                        )
                    if opt_df.empty:
                        st.info("目前資料期間不足或無法產生有效參數組合。")
                    else:
                        if optimize_target == "穩健分數最高":
                            st.caption(
                                "穩健分數 = Sharpe × 2 + 年化報酬％ / 20 + 最大回撤％ / 20"
                            )
                        else:
                            st.caption(
                                "依照總報酬％由高到低排序"
                            )
                        st.dataframe(opt_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "下載最佳化結果 CSV",
                            data=opt_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"optimization_{display_code(resolved_ticker)}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                        st.warning("提醒：建議可找 Sharpe 不錯、最大回撤可接受、交易次數不要太少，而且鄰近參數也表現穩定的組合。")

                st.markdown("---")
                st.subheader("自訂參數回測")

                custom_strategy = st.selectbox(
                    "選擇策略",
                    [
                        "保守均線趨勢｜少交易",
                        "長線大波段｜不太操作",
                        "EMA動能｜短線波段",
                        "突破追價｜不要錯過飆股",
                        "RSI反轉｜頻繁操作搶反彈",
                        "布林下軌反彈｜有賺就好",
                    ],
                    key="custom_strategy"
                )

                params = {}

                # ===== 不同策略顯示不同參數 =====

                if custom_strategy in ["保守均線趨勢｜少交易", "長線大波段｜不太操作"]:

                    c1, c2 = st.columns(2)

                    with c1:
                        params["short_ma"] = st.number_input(
                            "短均線",
                            min_value=2,
                            max_value=300,
                            value=20,
                            step=1,
                            key="custom_short_ma"
                        )

                    with c2:
                        params["long_ma"] = st.number_input(
                            "長均線",
                            min_value=3,
                            max_value=500,
                            value=60,
                            step=1,
                            key="custom_long_ma"
                        )

                elif custom_strategy == "EMA動能｜短線波段":

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        params["fast_ema"] = st.number_input(
                            "快 EMA",
                            2, 100, 10,
                            key="custom_fast_ema"
                        )

                    with c2:
                        params["slow_ema"] = st.number_input(
                            "慢 EMA",
                            3, 200, 20,
                            key="custom_slow_ema"
                        )

                    with c3:
                        params["rsi_enter"] = st.number_input(
                            "RSI進場",
                            1, 99, 50,
                            key="custom_rsi_enter"
                        )

                    with c4:
                        params["rsi_exit"] = st.number_input(
                            "RSI出場",
                            1, 99, 45,
                            key="custom_rsi_exit"
                        )

                elif custom_strategy == "突破追價｜不要錯過飆股":

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        params["breakout_n"] = st.number_input(
                            "突破天數",
                            5, 120, 20,
                            key="custom_breakout"
                        )

                    with c2:
                        params["exit_ma"] = st.number_input(
                            "出場均線",
                            5, 120, 20,
                            key="custom_exit_ma"
                        )

                    with c3:
                        params["volume_min"] = st.number_input(
                            "量比門檻",
                            0.5, 5.0, 1.2,
                            step=0.1,
                            key="custom_volume"
                        )

                elif custom_strategy == "RSI反轉｜頻繁操作搶反彈":

                    c1, c2 = st.columns(2)

                    with c1:
                        params["rsi_low"] = st.number_input(
                            "RSI低檔",
                            1, 50, 30,
                            key="custom_rsi_low"
                        )

                    with c2:
                        params["rsi_high"] = st.number_input(
                            "RSI出場",
                            40, 99, 55,
                            key="custom_rsi_high"
                        )

                # ===== 執行回測 =====

                custom_bt = backtest_strategy(
                    df,
                    custom_strategy,
                    params=params,
                    execution_mode=backtest_mode,
                    cost_rate=cost_rate,
                )

                if custom_bt:

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "累積報酬",
                            f"{custom_bt['total_return'] * 100:.2f}%"
                        )

                    with col2:
                        st.metric(
                            "年化報酬",
                            f"{custom_bt['cagr'] * 100:.2f}%"
                        )

                    with col3:
                        st.metric(
                            "最大回撤",
                            f"{custom_bt['max_dd'] * 100:.2f}%"
                        )

                    with col4:
                        st.metric(
                            "Sharpe",
                            f"{custom_bt['sharpe']:.2f}"
                        )

                    # ===== 買賣點 =====
                    bt_df = custom_bt["df"].copy()

                    bt_df["trade"] = bt_df["Signal"].diff()

                    bt_df["equity"] = (
                        1 + bt_df["Strategy_Return"]
                    ).cumprod()

                    plot_equity_curve_with_signals(bt_df)

                with st.expander("各策略適合什麼情境？"):
                    st.markdown("""
                    - **保守均線趨勢**：適合想避開大跌，有賺就好的防守型操作。
                    - **長線大波段**：適合不想常看盤，以季線、半年線判斷大方向的人。
                    - **EMA動能**：比 MA 更靈敏，適合短線波段與高波動成長股。
                    - **突破追價**：適合不想錯過飆股主升段，但要接受追高與停損。
                    - **RSI反轉**：適合頻繁操作、震盪盤搶反彈；不適合一路破底的股票。
                    - **布林下軌反彈**：適合箱型整理股，跌到區間下緣搶短彈。
                    """)

        with tab6:
            st.markdown("#### 策略執行助手")
            st.info("將回測策略轉換成實際操作條件：現在是否可進場、若尚未進場應等待什麼條件、進場後何時出場。")

            exec_strategy = st.selectbox(
                "選擇要執行的策略",
                list(STRATEGY_PRESETS.keys()),
                index=0,
                key="exec_strategy",
            )

            st.caption(
                f"策略規則：{STRATEGY_PRESETS[exec_strategy]['rule']}｜"
                f"適用情境：{STRATEGY_PRESETS[exec_strategy]['fit']}"
            )

            params = {}

            with st.expander("進階：自訂策略參數"):
                if exec_strategy in ["保守均線趨勢｜少交易", "長線大波段｜不太操作"]:
                    p1, p2 = st.columns(2)
                    with p1:
                        params["short_ma"] = st.number_input("短均線", min_value=3, max_value=240, value=20 if exec_strategy == "保守均線趨勢｜少交易" else 60, step=1)
                    with p2:
                        params["long_ma"] = st.number_input("長均線", min_value=5, max_value=300, value=60 if exec_strategy == "保守均線趨勢｜少交易" else 120, step=1)

                elif exec_strategy == "EMA動能｜短線波段":
                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        params["fast_ema"] = st.number_input("快 EMA", min_value=3, max_value=60, value=10, step=1)
                    with p2:
                        params["slow_ema"] = st.number_input("慢 EMA", min_value=5, max_value=120, value=20, step=1)
                    with p3:
                        params["rsi_enter"] = st.number_input("RSI 進場門檻", min_value=1, max_value=99, value=50, step=1)
                    with p4:
                        params["rsi_exit"] = st.number_input("RSI 出場門檻", min_value=1, max_value=99, value=45, step=1)

                elif exec_strategy == "突破追價｜不要錯過飆股":
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        params["breakout_n"] = st.number_input("突破天數", min_value=5, max_value=120, value=20, step=1)
                    with p2:
                        params["exit_ma"] = st.number_input("出場 MA", min_value=5, max_value=120, value=20, step=1)
                    with p3:
                        params["volume_min"] = st.number_input("量比門檻", min_value=0.1, max_value=5.0, value=1.2, step=0.1)

                elif exec_strategy == "RSI反轉｜頻繁操作搶反彈":
                    p1, p2 = st.columns(2)
                    with p1:
                        params["rsi_low"] = st.number_input("RSI 低檔進場", min_value=1, max_value=50, value=30, step=1)
                    with p2:
                        params["rsi_high"] = st.number_input("RSI 出場", min_value=40, max_value=99, value=55, step=1)

            advice = strategy_execution_advice(full_df, exec_strategy, params)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("目前狀態", advice["status"])
            s2.metric("現價", format_number(last["Close"], 2))
            s3.metric("關鍵價位", format_number(advice["key_level"], 2))
            s4.metric("距離關鍵價位", format_number(advice["distance_pct"], 2, "%"))

            st.markdown("#### 現在該怎麼做")
            if "符合" in advice["status"] or "可" in advice["status"]:
                st.success(advice["action"])
            else:
                st.warning(advice["action"])
            
            st.markdown("#### 進出場規則")
            rule_df = pd.DataFrame({
                "項目": ["進場條件", "出場條件"],
                "規則": [advice["entry"], advice["exit_rule"]],
            })
            st.dataframe(rule_df, hide_index=True, use_container_width=True)

            st.markdown("#### 判斷依據")
            for r in advice["reasons"]:
                st.write(f"• {r}")

            st.caption("提醒：此頁只把策略轉成條件式操作規則，不代表預測未來價格，也不構成投資建議。")
              
        with tab7:
            st.markdown("## 短線與長線模式差異")

            st.info(
                "目前評分系統統一採用原始分數 -10～+10，"
                "再轉換為 0～10 分。分數越高，代表越符合該操作模式的條件。"
            )

            compare_df = pd.DataFrame({
                "面向": ["核心目的", "適合週期", "主要判斷", "加分重點", "扣分重點", "適合族群"],
                "短線／波段": [
                    "判斷短期趨勢與動能是否有延續機會",
                    "數天～數週",
                    "MA5、MA20、MACD、RSI、量比、20日動能、布林位置",
                    "短均線多頭排列、MACD轉強、RSI強勢未過熱、放量上漲",
                    "短均線空頭排列、MACD轉弱、RSI過熱、放量下跌",
                    "想抓波段價差、能接受較頻繁操作的人",
                ],
                "長線／存股": [
                    "判斷中長期結構、估值與持有風險是否合理",
                    "數月～數年",
                    "MA60、MA120、MA240、PE、EPS、殖利率、Beta",
                    "中長期均線多頭、站上年線、估值合理、獲利穩定、波動較低",
                    "跌破中長期均線、低於年線、估值過高、EPS不佳、Beta偏高",
                    "想長期持有、重視穩定性與風險控管的人",
                ],
            })

            st.dataframe(compare_df, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("## 評分轉換方式")

            score_rule_df = pd.DataFrame({
                "項目": ["原始最低分", "原始中性分", "原始最高分", "轉換方式"],
                "短線／波段": ["-10", "0", "+10", "(-10～+10) 轉換為 0～10 分"],
                "長線／存股": ["-10", "0", "+10", "(-10～+10) 轉換為 0～10 分"],
            })

            st.dataframe(score_rule_df, hide_index=True, use_container_width=True)

            st.caption(
                "轉換公式：顯示分數 = (原始分數 + 10) / 20 × 10。"
                "因此原始分數 0 會對應到 5 分，代表中性。"
            )

            st.markdown("---")
            st.markdown("## 短線／波段評分架構")

            short_score_df = pd.DataFrame({
                "評分面向": [
                    "趨勢結構",
                    "MACD 動能",
                    "RSI 位置",
                    "量能表現",
                    "20日動能",
                    "布林位置",
                ],
                "最高加分": ["+3", "+2", "+2", "+1", "+1", "+1"],
                "最低扣分": ["-3", "-2", "-2", "-1", "-1", "-1"],
                "判斷重點": [
                    "股價、MA5、MA20 是否形成短線多頭或空頭排列",
                    "MACD 柱狀體是否轉正且擴大",
                    "RSI 是否位於強勢但未過熱區間",
                    "是否放量上漲或放量下跌",
                    "近 20 日漲跌幅是否明顯強勢或弱勢",
                    "是否過度偏離布林上軌或接近下軌",
                ],
            })

            st.dataframe(short_score_df, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("## 長線／存股評分架構")

            long_score_df = pd.DataFrame({
                "評分面向": [
                    "中長期趨勢",
                    "年線結構",
                    "PE 估值",
                    "EPS 獲利能力",
                    "殖利率",
                    "Beta 風險",
                ],
                "最高加分": ["+3", "+2", "+2", "+1", "+1", "+1"],
                "最低扣分": ["-3", "-2", "-2", "-1", "-1", "-1"],
                "判斷重點": [
                    "股價、MA60、MA120 是否形成中長期多頭或空頭排列",
                    "股價是否站上年線 MA240",
                    "本益比是否處於合理或偏高區間",
                    "EPS 是否為正且具備基本獲利能力",
                    "殖利率是否具備長期現金流吸引力",
                    "Beta 是否過高，代表波動風險較大",
                ],
            })

            st.dataframe(long_score_df, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("## 0～10 分評語說明")

            rating_df = pd.DataFrame({
                "分數區間": ["8.5～10", "7.0～8.4", "5.5～6.9", "4.0～5.4", "2.5～3.9", "0～2.4"],
                "評語": [
                    "條件優良／可優先觀察",
                    "偏多格局／可列入候選",
                    "中性偏多／等待確認",
                    "中性偏弱／保守觀望",
                    "偏弱格局／不宜積極",
                    "高風險／暫不建議介入",
                ],
                "代表意義": [
                    "多數關鍵條件符合，目前結構相對完整",
                    "整體偏多，但仍需觀察追價風險",
                    "部分條件轉佳，但尚未全面確認",
                    "條件偏弱，進場需要更保守",
                    "多數條件不理想，不適合積極操作",
                    "風險明顯偏高，應優先避開",
                ],
            })

            st.dataframe(rating_df, hide_index=True, use_container_width=True)

            st.caption(
                "提醒：分數是用來輔助比較不同股票在同一模式下的相對狀態，"
                "不是未來漲跌預測，也不代表投資建議。"
            )


        with tab8:
            st.header(" AI 妖股選股")
            st.caption("每日 18:00 自動更新")

            st.warning(
                "此模型屬於「妖股 / 飆股 / 動能股」選股邏輯，目標是尋找短線活性高、趨勢強、價格具爆發力的股票，"
            )

            with st.expander("七大選股指標說明", expanded=False):
                st.markdown("""
                **1. 歷史波動率**  
                衡量近 20 日股價活性，波動越大代表市場關注度與妖股特性越強。

                **2. 布林通道寬度**  
                觀察布林通道是否擴張，通常代表行情開始進入強勢波動階段。

                **3. 均線乖離率**  
                判斷股價是否已脫離底部，進入中短期多頭趨勢。

                **4. 趨勢強度**  
                比較短期與長期均線距離，確認短線趨勢是否明顯強勢。

                **5. 布林上軌強度**  
                股價越貼近布林上軌，通常代表市場買盤越強。

                **6. 十日動能**  
                計算近 10 日漲幅，反映短線爆發力與動能強弱。

                **7. 五日線防守濾網**  
                收盤價需站穩 5 日線，避免動能轉弱或跌破短線趨勢。
                """)
                st.caption("以上七項指標透過歷史資料訓練後進行最佳化權重配置")
                
            csv_path = "data/ai_momentum_top.csv"

            if not os.path.exists(csv_path):
                st.warning("尚未產生 AI 選股資料，請先執行 GitHub Actions 或本地更新腳本。")
            else:
                df_ai = pd.read_csv(csv_path)
                if "Updated_At" in df_ai.columns:
                    updated_at = pd.to_datetime(df_ai["Updated_At"]).max()
                    updated_at = updated_at + pd.Timedelta(hours=8)
                    updated_at = updated_at.strftime("%Y-%m-%d %H:%M")
                else:
                    updated_at = "未知"

                st.info(f"資料更新時間：{updated_at}")

                top_n = st.slider("顯示前幾名", 5, 50, 20)

                show_cols = [
                    "Rank",
                    "ID",
                    "Name",
                    "Industry",
                    "Close",
                    "AI_Score",
                    "F_Hist_Vol",
                    "F_BB_Width",
                    "F_P_to_MA60",
                    "F_Trend_Strength",
                    "F_P_to_MA20",
                    "F_P_to_BBUpper",
                    "F_ROC_10",
                ]

                df_show = df_ai[show_cols].head(top_n).copy()

                st.subheader("妖股動能排名")

                st.dataframe(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("排名"),
                        "ID": st.column_config.TextColumn("代號"),
                        "Name": st.column_config.TextColumn("股名"),
                        "Industry": st.column_config.TextColumn("產業"),
                        "Close": st.column_config.NumberColumn("收盤價", format="%.2f"),
                        "AI_Score": st.column_config.ProgressColumn(
                            "AI 動能分數",
                            min_value=0,
                            max_value=100,
                            format="%.2f",
                        ),
                        "F_Hist_Vol": st.column_config.NumberColumn("歷史波動率", format="%.2f%%"),
                        "F_BB_Width": st.column_config.NumberColumn("布林寬度", format="%.2f%%"),
                        "F_P_to_MA60": st.column_config.NumberColumn("距 MA60", format="%.2f%%"),
                        "F_Trend_Strength": st.column_config.NumberColumn("趨勢強度", format="%.2f%%"),
                        "F_P_to_MA20": st.column_config.NumberColumn("距 MA20", format="%.2f%%"),
                        "F_P_to_BBUpper": st.column_config.NumberColumn("距布林上軌", format="%.2f%%"),
                        "F_ROC_10": st.column_config.NumberColumn("10日動能", format="%.2f%%"),
                    }
                )

                st.caption(
                    "AI 動能分數越高，代表該股票在全市場中具有較高的波動、趨勢、乖離與短線動能排名。"
                )

                csv = df_ai.to_csv(index=False, encoding="utf-8-sig")

                st.download_button(
                    "下載 AI 妖股動能選股結果 CSV",
                    data=csv,
                    file_name="ai_momentum_top.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                
                
        with tab9:
            st.header("AI 潛力股選股")
            st.caption("每日 18:00 自動更新，尋找尚未發動但具低估修復潛力的股票。")

            st.warning(
                "此模型屬於「左側布局 / 低估修復 / 潛伏型」選股邏輯，"
                "目標是尋找技術面尚未明顯轉強，但可能具備跌深修復、低檔整理或估值修復空間的股票。"
            )

            with st.expander("AI 潛力股選股邏輯說明", expanded=False):
                st.markdown("""
                **1. 跌深修復空間**  
                觀察股價距離 52 週高點的回落幅度，以及未來可能回補的空間。

                **2. 技術安全邊際**  
                不是找已經噴出的股票，而是確認股價沒有持續崩壞，並觀察 RSI 是否位於低檔整理區。

                **3. 均線糾結度**  
                當 MA10、MA20、MA60 靠得很近時，代表股價可能進入低檔整理或籌碼沉澱階段。

                **4. 防守能力**  
                觀察波動率是否過高，避免選到跌深但風險失控的股票。

                **5. 潛伏修復特徵**  
                若股價跌深、波動可控、RSI 未過熱、均線開始靠攏，代表可能具有左側布局價值。
                """)

                st.caption(
                    "提醒：AI 潛力股不是預測明天會上漲，而是找出可能適合分批觀察、等待修復的股票。"
                )

            csv_path = "data/ai_potential_top.csv"

            if not os.path.exists(csv_path):
                st.warning("尚未產生 AI 潛力股資料，請先執行 GitHub Actions 或本地更新腳本。")
            else:
                df_potential = pd.read_csv(csv_path)

                if "Updated_At" in df_potential.columns:
                    updated_at = pd.to_datetime(df_potential["Updated_At"]).max()
                    updated_at = updated_at + pd.Timedelta(hours=8)
                    updated_at = updated_at.strftime("%Y-%m-%d %H:%M")
                else:
                    updated_at = "未知"

                st.info(f"資料更新時間：{updated_at}")

                top_n = st.slider(
                    "顯示前幾名",
                    5,
                    50,
                    20,
                    key="potential_top_n"
                )

                show_cols = [
                    "Rank",
                    "ID",
                    "Name",
                    "Industry",
                    "Close",
                    "AI_Potential_Score",
                    "Tag",
                    "RSI14",
                    "Hist_Vol",
                    "MA_Compression",
                    "Drawdown_52W",
                    "Rebound_Space",
                    "P_to_MA60",
                    "P_to_MA120",
                    "Volume_Ratio",
                    "Reason",
                    "Risk",
                ]

                df_show = df_potential[show_cols].head(top_n).copy()

                st.subheader("潛力股排名")

                st.dataframe(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("排名"),
                        "ID": st.column_config.TextColumn("代號"),
                        "Name": st.column_config.TextColumn("股名"),
                        "Industry": st.column_config.TextColumn("產業"),
                        "Close": st.column_config.NumberColumn("收盤價", format="%.2f"),
                        "AI_Potential_Score": st.column_config.ProgressColumn(
                            "AI 潛力分數",
                            min_value=0,
                            max_value=100,
                            format="%.2f",
                        ),
                        "Tag": st.column_config.TextColumn("類型"),
                        "RSI14": st.column_config.NumberColumn("RSI14", format="%.2f"),
                        "Hist_Vol": st.column_config.NumberColumn("年化波動", format="%.2f%%"),
                        "MA_Compression": st.column_config.NumberColumn("均線糾結度", format="%.2f%%"),
                        "Drawdown_52W": st.column_config.NumberColumn("距52週高點", format="%.2f%%"),
                        "Rebound_Space": st.column_config.NumberColumn("修復空間", format="%.2f%%"),
                        "P_to_MA60": st.column_config.NumberColumn("距MA60", format="%.2f%%"),
                        "P_to_MA120": st.column_config.NumberColumn("距MA120", format="%.2f%%"),
                        "Volume_Ratio": st.column_config.NumberColumn("量比", format="%.2f"),
                        "Reason": st.column_config.TextColumn("主要理由"),
                        "Risk": st.column_config.TextColumn("主要風險"),
                    }
                )

                st.caption(
                    "AI 潛力分數越高，代表該股票越符合低檔整理、跌深修復、波動可控與左側潛伏條件。"
                )

                csv = df_potential.to_csv(index=False, encoding="utf-8-sig")

                st.download_button(
                    "下載 AI 潛力股選股結果 CSV",
                    data=csv,
                    file_name="ai_potential_top.csv",
                    mime="text/csv",
                    use_container_width=True,
                )     

    except Exception as exc:
        st.error(f"分析失敗：{exc}")
        st.info("請確認網路連線、股票代碼，或稍後再試。若是上櫃股票，直接輸入 8069 或 8069.TWO 都可以。")

if run_watchlist:
    codes = [x.strip() for x in watchlist_text.replace("\n", ",").split(",") if x.strip()]
    rows = []
    progress = st.progress(0)
    for i, code in enumerate(codes):
        try:
            raw_df, resolved = load_price_data(normalize_tw_ticker(code))
            
            # 按用戶指定的日期範圍裁切
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            raw_df = raw_df[(raw_df.index >= start_ts) & (raw_df.index <= end_ts)].copy()
            
            if raw_df.empty:
                continue
            
            info = load_ticker_info(resolved)
            full_df = calculate_indicators(raw_df)
            df = full_df.dropna(subset=["RSI14", "MACD_HIST"])
            if df.empty or len(df) < 20:
                continue
            last = df.iloc[-1]
            score = score_stock(df, info, mode)
            dy = dividend_yield_pct(info, last["Close"])
            rows.append(
                {
                    "代碼": resolved,
                    "名稱": info.get("shortName") or info.get("longName") or display_code(resolved),
                    "日期": df.index[-1].strftime("%Y-%m-%d"),
                    "收盤": round(last["Close"], 2),
                    "1日%": round(last["Return_1D"] * 100, 2),
                    "20日%": round(last["Return_20D"] * 100, 2),
                    "RSI": round(last["RSI14"], 1),
                    "量比": round(last["Volume_Ratio"], 2),
                    "PE": round(safe_float(info.get("trailingPE")), 2) if not pd.isna(safe_float(info.get("trailingPE"))) else np.nan,
                    "殖利率%": round(dy, 2) if not pd.isna(dy) else np.nan,
                    "分數": score["score_10"],
                    "結論": score["label"],
                }
            )
        except Exception:
            rows.append({"代碼": code, "名稱": "讀取失敗", "結論": "請確認代碼或資料源"})
        progress.progress((i + 1) / max(len(codes), 1))

    st.subheader("觀察清單掃描結果")
    if rows:
        watch_df = pd.DataFrame(rows).sort_values("分數", ascending=False, na_position="last") if "分數" in pd.DataFrame(rows).columns else pd.DataFrame(rows)
        st.dataframe(watch_df, use_container_width=True, hide_index=True)
        st.download_button(
            "下載觀察清單結果 CSV",
            data=watch_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"watchlist_scan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("沒有成功取得任何觀察清單資料。")

st.divider()

with st.expander("下載目前分析資料"):
    csv = df.to_csv(index=True).encode("utf-8-sig")
    st.download_button("下載技術指標 CSV", data=csv, file_name=f"{display_code(resolved_ticker)}_analysis.csv", mime="text/csv")

# ===== 底部更新公告 =====
with st.expander("📢 更新公告", expanded=False):
    st.markdown(
        render_changelog(CHANGELOG_TEXT),
        unsafe_allow_html=True
    )
# ===== 意見回饋 =====
FEEDBACK_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeUretKEOaprGWpM7Gp5SKFAOjoc8RUhRTnB2fOwdksg7pMcA/viewform?usp=dialog"
with st.expander("💬 意見回饋", expanded=False):
    st.caption("如果有任何建議、功能想法或發現問題，歡迎匿名留下回饋。")

    st.link_button(
        "前往匿名回饋表單",
        FEEDBACK_FORM_URL,
        use_container_width=True
    )