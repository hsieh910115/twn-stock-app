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
            info["dividendYield"] = safe_float(latest.get("dividend_yield"))
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
    """盡量從 FinMind 欄位推估殖利率，並統一轉成百分比。

    FinMind 對台股常出現 dividendYield 空白，因此依序嘗試：
    dividendYield、trailingAnnualDividendYield、fiveYearAvgDividendYield、
    dividendRate / 股價、trailingAnnualDividendRate / 股價、lastDividendValue / 股價。
    """
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
    """當 FinMind beta 空白時，用近期間股價報酬相對加權指數 TAIEX 粗估 Beta。"""
    try:
        market_raw, _ = load_price_data("TAIEX")
        market = calculate_indicators(market_raw) if "Return_1D" not in market_raw.columns else market_raw
        s_ret = stock_df["Close"].pct_change().rename("stock")
        m_ret = market["Close"].pct_change().rename("market")
        aligned = pd.concat([s_ret, m_ret], axis=1).dropna()
        aligned = aligned[(aligned["stock"].abs() < 0.3) & (aligned["market"].abs() < 0.15)]
        if len(aligned) < 30 or aligned["market"].var() == 0:
            return np.nan
        return float(aligned["stock"].cov(aligned["market"]) / aligned["market"].var())
    except Exception:
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
    """依操作模式給分。短線重視動能、量能與波動；長線重視趨勢品質、估值與股息。"""
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

    if "短線" in mode:
        # 短線：先看趨勢與動能有沒有同步，避免只因便宜就接刀。
        if is_valid_number(last.get("MA5")) and is_valid_number(last.get("MA20")) and close > last["MA5"] > last["MA20"]:
            score += 2
            reasons.append("短線價格站上 MA5 且 MA5 高於 MA20，短期攻擊型態較強。")
        elif is_valid_number(last.get("MA5")) and is_valid_number(last.get("MA20")) and close < last["MA5"] < last["MA20"]:
            score -= 2
            warnings.append("短線價格跌破 MA5 且 MA5 低於 MA20，短期動能偏弱。")

        if is_valid_number(last.get("MACD_HIST")) and is_valid_number(prev.get("MACD_HIST")) and last["MACD_HIST"] > 0 and last["MACD_HIST"] > prev["MACD_HIST"]:
            score += 2
            reasons.append("MACD 柱狀體為正且擴大，波段動能改善。")
        elif is_valid_number(last.get("MACD_HIST")) and last["MACD_HIST"] < 0:
            score -= 1
            warnings.append("MACD 柱狀體仍為負，短線追價需保守。")

        if 45 <= rsi <= 68:
            score += 1
            reasons.append("RSI 位於偏強但未過熱區，短線仍有延續空間。")
        elif rsi > 75:
            score -= 2
            warnings.append("RSI 高於 75，短線過熱，容易出現拉回。")
        elif rsi < 30:
            score += 1
            warnings.append("RSI 低於 30，可能有反彈機會，但需確認不是趨勢轉空。")

        if is_valid_number(last.get("Volume_Ratio")) and last["Volume_Ratio"] > 1.5 and close > prev["Close"]:
            score += 1
            reasons.append("成交量明顯放大且收漲，短線買盤較積極。")
        elif is_valid_number(last.get("Volume_Ratio")) and last["Volume_Ratio"] > 1.8 and close < prev["Close"]:
            score -= 1
            warnings.append("放量下跌，短線賣壓偏重。")

        if is_valid_number(last.get("Return_20D")) and last["Return_20D"] > 0.08:
            score += 1
            reasons.append("近 20 日漲幅較強，屬相對強勢股候選。")
        elif is_valid_number(last.get("Return_20D")) and last["Return_20D"] < -0.08:
            score -= 1
            warnings.append("近 20 日跌幅偏大，短線結構仍需觀察。")

    else:
        # 長線：先看大結構、估值、股息與獲利品質，不過度被短線波動干擾。
        if is_valid_number(last.get("MA60")) and is_valid_number(last.get("MA120")) and close > last["MA60"] > last["MA120"]:
            score += 2
            reasons.append("股價站上季線，且季線高於半年線，中長期趨勢偏多。")
        elif is_valid_number(last.get("MA60")) and close < last["MA60"]:
            score -= 2
            warnings.append("股價低於季線，長線加碼宜放慢。")

        if is_valid_number(last.get("MA240")):
            if close > last["MA240"]:
                score += 1
                reasons.append("股價位於年線之上，長期結構尚佳。")
            else:
                score -= 1
                warnings.append("股價低於年線，長期趨勢仍需修復。")
        else:
            warnings.append("目前期間較短，年線資料不足，長線判斷需搭配更長資料。")

        if not pd.isna(pe):
            if 0 < pe <= 20:
                score += 2
                reasons.append("本益比不高，估值相對保守。")
            elif pe > 35:
                score -= 1
                warnings.append("本益比偏高，需確認未來成長足以支撐估值。")

        if not pd.isna(eps):
            if eps > 0:
                score += 1
                reasons.append("EPS 為正，具備基本獲利能力。")
            else:
                score -= 1
                warnings.append("EPS 非正值，長線持有需特別檢查獲利品質。")

        if not pd.isna(dividend_yield):
            if dividend_yield >= 4:
                score += 2
                reasons.append("殖利率高於 4%，對長線現金流較有吸引力。")
            elif dividend_yield < 1:
                warnings.append("殖利率偏低，長線報酬主要仰賴資本利得。")

        if not pd.isna(beta):
            if beta <= 1:
                score += 1
                reasons.append("Beta 不高，波動相對溫和。")
            elif beta > 1.4:
                warnings.append("Beta 偏高，長期持有仍可能承受較大波動。")

    # 共同風險：不論短線長線，都提醒估值與過度偏離。
    if is_valid_number(last.get("BB_UPPER")) and close > last["BB_UPPER"]:
        warnings.append("股價高於布林上軌，短期偏離均值，追高風險較高。")
    if is_valid_number(last.get("Drawdown")) and last["Drawdown"] < -0.25:
        warnings.append("目前距歷史高點回撤超過 25%，需確認是否為基本面轉弱。")

    if score >= 6:
        label = "偏多觀察／可列入候選"
        level = "success"
    elif score >= 3:
        label = "中性偏多／等待更佳切入點"
        level = "info"
    elif score <= -3:
        label = "偏弱／先避開或嚴控風險"
        level = "error"
    else:
        label = "中性整理／不急著出手"
        level = "warning"

    return {"score": score, "label": label, "level": level, "reasons": reasons[:6], "warnings": warnings[:6]}

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


def backtest_strategy(df: pd.DataFrame, strategy_name: str, params: Optional[Dict] = None) -> Dict:
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

    bt = data.copy()
    bt["Signal"] = signal.reindex(bt.index).fillna(0)
    bt["Strategy_Return"] = bt["Signal"].shift(1).fillna(0) * bt["Return_1D"]
    return compute_backtest_stats(bt)


def backtest_all_strategies(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, meta in STRATEGY_PRESETS.items():
        bt = backtest_strategy(df, name)
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


def optimize_parameters(df: pd.DataFrame, strategy_family: str) -> pd.DataFrame:
    """依目前左側設定的資料期間做簡單參數最佳化。\n\n    注意：這是歷史區間最佳，不保證未來最佳；主要用來比較參數穩定性。\n    """
    rows = []
    if strategy_family == "均線趨勢 MA":
        for short in [5, 10, 20, 30, 60]:
            for long in [20, 60, 90, 120, 240]:
                if short >= long:
                    continue
                name = "保守均線趨勢｜少交易" if long <= 90 else "長線大波段｜不太操作"
                bt = backtest_strategy(df, name, {"short_ma": short, "long_ma": long})
                if not bt:
                    continue
                rows.append({"策略族": strategy_family, "參數": f"Close > MA{short} 且 MA{short} > MA{long}", "短均線": short, "長均線": long, **_opt_stats(bt)})

    elif strategy_family == "EMA動能":
        for fast in [5, 8, 10, 12, 15]:
            for slow in [20, 30, 50, 60]:
                if fast >= slow:
                    continue
                for rsi_enter in [45, 50, 55]:
                    bt = backtest_strategy(df, "EMA動能｜短線波段", {"fast_ema": fast, "slow_ema": slow, "rsi_enter": rsi_enter, "rsi_exit": rsi_enter - 5})
                    if not bt:
                        continue
                    rows.append({"策略族": strategy_family, "參數": f"EMA{fast} > EMA{slow}, RSI>{rsi_enter}", "快EMA": fast, "慢EMA": slow, "RSI進場": rsi_enter, **_opt_stats(bt)})

    elif strategy_family == "突破追價":
        for n in [10, 20, 30, 55]:
            for exit_ma in [10, 20, 30]:
                for vol in [1.0, 1.2, 1.5]:
                    bt = backtest_strategy(df, "突破追價｜不要錯過飆股", {"breakout_n": n, "exit_ma": exit_ma, "volume_min": vol})
                    if not bt:
                        continue
                    rows.append({"策略族": strategy_family, "參數": f"突破{n}日高, 量比>{vol}, 跌破MA{exit_ma}出場", "突破天數": n, "出場MA": exit_ma, "量比門檻": vol, **_opt_stats(bt)})

    elif strategy_family == "長線大波段 MA":
        for short in [40, 60, 90, 120]:
            for long in [120, 180, 240]:
                if short >= long:
                    continue
                bt = backtest_strategy(df, "長線大波段｜不太操作", {"short_ma": short, "long_ma": long})
                if not bt:
                    continue
                rows.append({"策略族": strategy_family, "參數": f"Close > MA{short} 且 MA{short} > MA{long}", "短均線": short, "長均線": long, **_opt_stats(bt)})

    elif strategy_family == "RSI反轉":
        for low in [20, 25, 30, 35]:
            for high in [50, 55, 60, 65]:
                if low >= high:
                    continue
                bt = backtest_strategy(df, "RSI反轉｜頻繁操作搶反彈", {"rsi_low": low, "rsi_high": high})
                if not bt:
                    continue
                rows.append({"策略族": strategy_family, "參數": f"RSI<{low}進場, RSI>{high}或站上MA20出場", "RSI低檔": low, "RSI出場": high, **_opt_stats(bt)})

    elif strategy_family == "布林反彈":
        # 目前布林策略的結構固定，這裡用不同布林倍數重算後測試。
        for k in [1.5, 2.0, 2.5, 3.0]:
            tmp = df.copy()
            tmp["BB_UPPER"] = tmp["BB_MID"] + k * tmp["BB_STD"]
            tmp["BB_LOWER"] = tmp["BB_MID"] - k * tmp["BB_STD"]
            bt = backtest_strategy(tmp, "布林下軌反彈｜有賺就好")
            if not bt:
                continue
            rows.append({"策略族": strategy_family, "參數": f"Close < BB下軌({k}σ)進場, 回中線出場", "布林倍數": k, **_opt_stats(bt)})

    elif strategy_family == "全部策略":
        for fam in ["均線趨勢 MA", "長線大波段 MA", "EMA動能", "突破追價", "RSI反轉", "布林反彈"]:
            part = optimize_parameters(df, fam)
            if not part.empty:
                rows.extend(part.to_dict("records"))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # 排序不是只看報酬，避免選到很會暴衝但回撤超大的參數。
    out["穩健分數"] = out["Sharpe"].fillna(0) * 2 + out["年化報酬%"].fillna(0) / 20 + out["最大回撤%"].fillna(-100) / 20
    return out.sort_values("穩健分數", ascending=False).head(20)


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
    """專業 K 線圖：台股紅漲綠跌、疊加 MA 與布林灰色區域，並附成交量。"""
    plot = df.tail(rows).copy().reset_index()
    date_col = plot.columns[0]
    plot = plot.rename(columns={date_col: "日期"})

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # 布林通道灰色區域：先畫上軌，再用下軌填滿到上軌
    if {"BB_UPPER", "BB_LOWER", "BB_MID"}.issubset(plot.columns):
        fig.add_trace(
            go.Scatter(
                x=plot["日期"], y=plot["BB_UPPER"], mode="lines",
                line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
                name="布林上軌", hovertemplate="%{x}<br>布林上軌 %{y:,.2f}<extra></extra>",
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot["日期"], y=plot["BB_LOWER"], mode="lines",
                line=dict(color="rgba(120,120,120,0.45)", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(160,160,160,0.18)",
                name="布林下軌", hovertemplate="%{x}<br>布林下軌 %{y:,.2f}<extra></extra>",
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot["日期"], y=plot["BB_MID"], mode="lines",
                line=dict(color="rgba(120,120,120,0.8)", width=1.2, dash="dash"),
                name="布林中線", hovertemplate="%{x}<br>布林中線 %{y:,.2f}<extra></extra>",
            ), row=1, col=1
        )

    # K 線：台股慣例，漲紅跌綠
    fig.add_trace(
        go.Candlestick(
            x=plot["日期"],
            open=plot["Open"], high=plot["High"], low=plot["Low"], close=plot["Close"],
            name="K線",
            increasing=dict(line=dict(color="#D32F2F"), fillcolor="#D32F2F"),
            decreasing=dict(line=dict(color="#00A65A"), fillcolor="#00A65A"),
            hovertemplate=(
                "%{x}<br>開盤 %{open:,.2f}<br>最高 %{high:,.2f}"
                "<br>最低 %{low:,.2f}<br>收盤 %{close:,.2f}<extra></extra>"
            ),
        ), row=1, col=1
    )

    ma_specs = [
        ("MA5", "週線 MA5", "#FFD400"),
        ("MA20", "月線 MA20", "#00A65A"),
        ("MA60", "季線 MA60", "#4FC3F7"),
        ("MA120", "半年線 MA120", "#0D47A1"),
        ("MA240", "年線 MA240", "#8B5A2B"),
    ]
    for col, name, color in ma_specs:
        if col in plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot["日期"], y=plot[col], mode="lines",
                    line=dict(color=color, width=1.6),
                    name=name, hovertemplate=f"%{{x}}<br>{name} %{{y:,.2f}}<extra></extra>",
                ), row=1, col=1
            )

    # 成交量：漲紅跌綠
    volume_colors = np.where(plot["Close"] >= plot["Open"], "#D32F2F", "#00A65A")
    fig.add_trace(
        go.Bar(
            x=plot["日期"], y=plot["Volume"], name="成交量",
            marker_color=volume_colors, opacity=0.45,
            hovertemplate="%{x}<br>成交量 %{y:,.0f}<extra></extra>",
        ), row=2, col=1
    )

    # if close_overlay is not None:
    #     overlay = close_overlay.tail(rows + 5).reset_index()
    #     overlay.columns = ["日期", "收盤價"]
    #     fig.add_trace(
    #         go.Scatter(
    #             x=overlay["日期"],
    #             y=overlay["收盤價"],
    #             mode="lines",
    #             line=dict(color="#000000", width=1.8),
    #             name="收盤價",
    #             hovertemplate="%{x}<br>收盤 %{y:,.2f}<extra></extra>",
    #         ), row=1, col=1
    #     )    


    fig.update_layout(
        height=650,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="股價", row=1, col=1, fixedrange=False)
    fig.update_yaxes(title_text="成交量", row=2, col=1, fixedrange=False)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
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

# =========================
# 介面
# =========================
st.title("📈 台股投資分析")
st.caption("分析工具：即時更新股價、技術面、基本面、風險控管、簡易回測與觀察清單。")

with st.sidebar:
    st.header("股票設定")
    raw_code = st.text_input("股票代碼", value="2330", help="可輸入 2330、8069、0050，也可輸入 2330.TW/8069.TWO")
    st.markdown("資料期間")
    pc1, pc2 = st.columns(2)
    with pc1:
        period_years = st.number_input("年", min_value=0, max_value=30, value=1, step=1)
    with pc2:
        period_months = st.number_input("月", min_value=0, max_value=11, value=0, step=1)
    if period_years == 0 and period_months == 0:
        period_months = 6
    total_days = period_to_days(period_years, period_months)
    st.caption(f"目前分析區間：{period_years} 年 {period_months} 個月")
    # 這裡的 start/end 僅供顯示參考，實際會以資料最新日往回推
    ref_end = datetime.now().date()
    ref_start = ref_end - timedelta(days=total_days)
    st.caption(f"回測參考區間：{ref_start} ～ {ref_end}")
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
            # 1. 抓取原始資料 (固定抓 5 年，確保不同區間底層資料一致)
            raw_df, resolved_ticker = load_price_data(ticker_input)
            
            # 2. 找出資料中的最新日期，並以此往回計算分析起始日
            actual_end = raw_df.index[-1]
            actual_start = actual_end - timedelta(days=total_days)

            info = load_ticker_info(resolved_ticker)
            fast_info = load_fast_info(resolved_ticker)
            stmt = load_financials(resolved_ticker)
            
            # 3. 計算指標
            full_df = calculate_indicators(raw_df)
            
            # 4. 裁切回使用者要求的區間
            df = trim_to_user_period(full_df, actual_start).dropna(
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
        if score["level"] == "success":
            st.success(f"綜合評分：{score['score']}｜{score['label']}")
        elif score["level"] == "error":
            st.error(f"綜合評分：{score['score']}｜{score['label']}")
        elif score["level"] == "warning":
            st.warning(f"綜合評分：{score['score']}｜{score['label']}")
        else:
            st.info(f"綜合評分：{score['score']}｜{score['label']}")

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

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["總覽", "技術圖表", "基本面", "風險控管", "簡易回測", "模式差異"])

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
            st.caption("K線顏色採台股習慣：紅K＝收漲、綠K＝收跌；黃色＝週線MA5、綠色＝月線MA20、淺藍＝季線MA60、深藍＝半年線MA120、棕色＝年線MA240；灰色區域＝布林上下通道，灰線＝布林上中下軌。")
            st.plotly_chart(make_price_chart(df, close_overlay=full_df["Close"]), use_container_width=True)
            st.markdown("#### 指標明細")
            indicator_cols = ["Close", "Volume", "MA5", "MA20", "MA60", "MA120", "MA240", "BB_UPPER", "BB_MID", "BB_LOWER", "RSI14", "MACD_DIF", "MACD_SIGNAL", "MACD_HIST", "ATR14", "Volume_Ratio", "Return_5D", "Return_20D"]
            st.dataframe(df[indicator_cols].tail(30).iloc[::-1], use_container_width=True)

        with tab3:
            fin_table = build_financial_table(stmt, resolved_ticker)
            fund = derive_fundamental_metrics(info, fast_info, last["Close"], fin_table)
            if pd.isna(fund.get("beta")):
                fund["beta"] = estimate_beta_vs_twii(df)
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("本益比 PE", format_number(fund["pe"], 2))
            f2.metric("EPS", format_number(fund["eps"], 2))
            f3.metric("殖利率", format_number(fund["dividend_yield_pct"], 2, "%"))
            f4.metric("市值", format_large_twd(fund["market_cap"]))
            f5.metric("Beta", format_number(fund["beta"], 2))
            st.caption("PE、殖利率優先使用 FinMind；若 FinMind 無資料，則以 yfinance 補 PE、EPS、殖利率、市值與 Beta。Beta 若仍空白，會用本分析期間相對加權指數 TAIEX 的日報酬粗估。")
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

            compare_df = backtest_all_strategies(df)
            if compare_df.empty:
                st.info("資料不足，無法進行多策略回測。建議增加左側資料期間。")
            else:
                st.markdown("##### 策略比較表")
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

                strategy_name = st.selectbox("選擇要畫圖與細看的策略", list(STRATEGY_PRESETS.keys()), index=0)
                meta = STRATEGY_PRESETS[strategy_name]
                st.caption(f"策略定位：{meta['style']}｜規則：{meta['rule']}｜適用情境：{meta['fit']}")

                bt = backtest_strategy(df, strategy_name)
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
                opt_family = st.selectbox("選擇要最佳化的策略族", ["全部策略", "均線趨勢 MA", "長線大波段 MA", "EMA動能", "突破追價", "RSI反轉", "布林反彈"])
                run_opt = st.button("執行參數最佳化", type="secondary", use_container_width=True)
                if run_opt:
                    with st.spinner("正在測試不同參數組合..."):
                        opt_df = optimize_parameters(df, opt_family)
                    if opt_df.empty:
                        st.info("目前資料期間不足或無法產生有效參數組合。")
                    else:
                        st.dataframe(opt_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "下載最佳化結果 CSV",
                            data=opt_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"optimization_{display_code(resolved_ticker)}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                        st.warning("提醒：建議可找 Sharpe 不錯、最大回撤可接受、交易次數不要太少，而且鄰近參數也表現穩定的組合。")

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
            st.markdown("#### 短線與長線分析差異")
            st.dataframe(mode_difference_table(), hide_index=True, use_container_width=True)
            if "短線" in mode:
                st.success("目前使用短線／波段模式：分數會更重視短均線、MACD、RSI、量能與近 20 日強弱。")
            else:
                st.success("目前使用長線／存股模式：分數會更重視季線、半年線、年線、PE、EPS、殖利率與 Beta。")
            st.caption("同一檔股票在兩種模式下可能得到不同結論，因為短線問的是『近期有沒有動能』，長線問的是『估值、趨勢與現金流是否適合長期持有』。")

        with st.expander("下載目前分析資料"):
            csv = df.to_csv(index=True).encode("utf-8-sig")
            st.download_button("下載技術指標 CSV", data=csv, file_name=f"{display_code(resolved_ticker)}_analysis.csv", mime="text/csv")

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
            actual_end = raw_df.index[-1]
            actual_start = actual_end - timedelta(days=total_days)
            
            info = load_ticker_info(resolved)
            full_df = calculate_indicators(raw_df)
            df = trim_to_user_period(full_df, actual_start).dropna(subset=["RSI14", "MACD_HIST"])
            if df.empty or len(df) < 20:
                continue
            last = df.iloc[-1]
            score = score_stock(df, info, mode)
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
                    "殖利率%": round(dividend_yield_pct(info, last["Close"]), 2) if not pd.isna(dividend_yield_pct(info, last["Close"])) else np.nan,
                    "分數": score["score"],
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
st.caption("Developed by hsieh910115")
