from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import altair as alt
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
    """將使用者輸入轉成 yfinance 台股代碼。支援 2330、2330.TW、8069.TWO、^TWII。"""
    code = str(code).strip().upper()
    if not code:
        return "2330.TW"
    if code.startswith("^") or code.endswith(".TW") or code.endswith(".TWO"):
        return code
    # 台灣上市預設 .TW；若查不到，下載函式會自動嘗試 .TWO
    return f"{code}.TW"


def display_code(ticker: str) -> str:
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


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_price_data(ticker_symbol: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str]:
    """抓取股價資料。上市用 .TW，若無資料自動嘗試 .TWO。

    重要：即使使用者只分析 0 年 6 個月，也會往前多抓一段 warm-up 資料，
    讓 MA60、MA120、MA240、52週高低點等長週期指標可以正常計算。
    後續畫圖與回測仍會再切回使用者指定的起始日期。
    """
    candidates = [ticker_symbol]
    if ticker_symbol.endswith(".TW"):
        candidates.append(ticker_symbol.replace(".TW", ".TWO"))

    last_error = ""
    warmup_start = (pd.to_datetime(start_date) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    end_plus_one = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for symbol in candidates:
        try:
            df = yf.Ticker(symbol).history(start=warmup_start, end=end_plus_one, auto_adjust=False)
            if not df.empty:
                df = df.copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.rename(columns=str.title)
                return df, symbol
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"無法取得 {ticker_symbol} 股價資料。{last_error}")


def period_to_dates(years: int, months: int) -> Tuple[str, str]:
    """把側欄的幾年幾個月轉成 yfinance 可用的起訖日期。"""
    years = max(int(years), 0)
    months = max(int(months), 0)
    total_days = max(years * 365 + months * 30, 30)
    end = datetime.now().date()
    start = end - timedelta(days=total_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def trim_to_user_period(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """指標計算完後，切回使用者指定的分析期間。"""
    start = pd.to_datetime(start_date)
    trimmed = df[df.index >= start].copy()
    return trimmed


def is_valid_number(value) -> bool:
    """判斷數值是否可用，避免短期間缺 MA120/MA240 時被誤判為跌破。"""
    return value is not None and not pd.isna(value) and np.isfinite(float(value))


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_ticker_info(ticker_symbol: str) -> Dict:
    try:
        info = yf.Ticker(ticker_symbol).get_info()
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_financials(ticker_symbol: str) -> pd.DataFrame:
    try:
        stmt = yf.Ticker(ticker_symbol).quarterly_financials
        if stmt is None or stmt.empty:
            return pd.DataFrame()
        return stmt.copy()
    except Exception:
        return pd.DataFrame()


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
    dividend_yield = safe_float(info.get("dividendYield")) * 100
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
    "保守均線趨勢｜少交易、不要死太慘": {
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

    if strategy_name == "保守均線趨勢｜少交易、不要死太慘":
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
                name = "保守均線趨勢｜少交易、不要死太慘" if long <= 90 else "長線大波段｜不太操作"
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

def make_price_chart(df: pd.DataFrame, rows: int = 240):
    """固定顏色與圖例，讓 Close、均線、布林通道容易辨識。"""
    plot = df.tail(rows).reset_index()
    date_col = plot.columns[0]
    plot = plot.rename(columns={date_col: "日期"})
    rename_map = {
        "Close": "Close 收盤價",
        "MA20": "MA20 月線",
        "MA60": "MA60 季線",
        "MA120": "MA120 半年線",
        "BB_UPPER": "BB 上軌",
        "BB_LOWER": "BB 下軌",
    }
    available = [c for c in rename_map if c in plot.columns]
    long = plot[["日期"] + available].rename(columns=rename_map).melt(
        id_vars="日期", var_name="線種", value_name="數值"
    ).dropna()

    color_scale = alt.Scale(
        domain=["Close 收盤價", "MA20 月線", "MA60 季線", "MA120 半年線", "BB 上軌", "BB 下軌"],
        range=["#0B63CE", "#F58518", "#2CA02C", "#9467BD", "#9AA0A6", "#9AA0A6"],
    )
    dash_scale = alt.Scale(
        domain=["Close 收盤價", "MA20 月線", "MA60 季線", "MA120 半年線", "BB 上軌", "BB 下軌"],
        range=[[1, 0], [4, 2], [6, 3], [2, 2], [8, 4], [8, 4]],
    )
    size_scale = alt.Scale(
        domain=["Close 收盤價", "MA20 月線", "MA60 季線", "MA120 半年線", "BB 上軌", "BB 下軌"],
        range=[3.2, 2.0, 2.0, 1.8, 1.4, 1.4],
    )

    return (
        alt.Chart(long)
        .mark_line(interpolate="monotone")
        .encode(
            x=alt.X("日期:T", title="日期"),
            y=alt.Y("數值:Q", title="股價", scale=alt.Scale(zero=False)),
            color=alt.Color("線種:N", scale=color_scale, legend=alt.Legend(title="線種")),
            strokeDash=alt.StrokeDash("線種:N", scale=dash_scale, legend=None),
            size=alt.Size("線種:N", scale=size_scale, legend=None),
            tooltip=[
                alt.Tooltip("日期:T", title="日期"),
                alt.Tooltip("線種:N", title="線種"),
                alt.Tooltip("數值:Q", title="數值", format=",.2f"),
            ],
        )
        .properties(height=430)
    )

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


def build_financial_table(stmt: pd.DataFrame) -> pd.DataFrame:
    """整理至少 8 季財報；若資料源不足則顯示可取得的完整季度，並加入 QoQ 與 YoY。"""
    if stmt.empty:
        return pd.DataFrame()
    data = stmt.iloc[:, :8].T.sort_index()
    wanted = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
    existing = [x for x in wanted if x in data.columns]
    if not existing:
        return pd.DataFrame()

    out = data[existing].copy()
    # 移除完全空白的季度，避免 2026Q1 這類尚未更新的欄位造成誤判。
    out = out.dropna(how="all")
    out.index = [f"{d.year}Q{((d.month - 1) // 3) + 1}" for d in out.index]

    for col in out.columns:
        out[col] = out[col] / 1e8
    out = out.rename(columns={
        "Total Revenue": "營收(億)",
        "Gross Profit": "毛利(億)",
        "Operating Income": "營業利益(億)",
        "Net Income": "淨利(億)",
    })
    if "營收(億)" in out.columns:
        out["營收QoQ%"] = out["營收(億)"].pct_change() * 100
        out["營收YoY%"] = out["營收(億)"].pct_change(4) * 100
    if "淨利(億)" in out.columns:
        out["淨利YoY%"] = out["淨利(億)"].pct_change(4) * 100
    return out.tail(8).round(2)

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
st.title("📈 台股專業投資分析終端")
st.caption("每日可用的分析工具：即時更新股價、技術面、基本面、風險控管、簡易回測與觀察清單。")

with st.sidebar:
    st.header("股票設定")
    raw_code = st.text_input("股票代碼", value="2330", help="可輸入 2330、2330.TW、8069.TWO、0050")
    st.markdown("資料期間")
    pc1, pc2 = st.columns(2)
    with pc1:
        period_years = st.number_input("年", min_value=0, max_value=30, value=5, step=1)
    with pc2:
        period_months = st.number_input("月", min_value=0, max_value=11, value=0, step=1)
    if period_years == 0 and period_months == 0:
        period_months = 6
    start_date, end_date = period_to_dates(period_years, period_months)
    st.caption(f"目前分析區間：約 {period_years} 年 {period_months} 個月；支援 0 年 1～11 個月的短線分析。")
    st.caption(f"回測區間：約 {start_date} ～ {end_date}")
    mode = st.radio("操作模式", ["短線／波段", "長線／存股"], horizontal=False)
    capital = st.number_input("帳戶資金（元）", min_value=10000, value=300000, step=10000)
    risk_pct = st.number_input("單筆最大風險 %", min_value=0.01, max_value=100.0, value=1.0, step=0.1, format="%.2f", help="代表這一筆交易最多願意虧掉帳戶資金的百分比，例如 1% 表示 30 萬帳戶最多虧 3000 元。")
    analyze = st.button("更新並分析", type="primary", use_container_width=True)
    st.divider()
    watchlist_text = st.text_area("觀察清單", value=DEFAULT_WATCHLIST, height=100)
    run_watchlist = st.button("掃描觀察清單", use_container_width=True)
    st.caption("資料由 yfinance 取得；台股最新交易日可能因資料源延遲而非今日。")

# 預設初次也分析，避免使用者打開空白頁
if not analyze and not run_watchlist:
    analyze = True

if analyze:
    ticker_input = normalize_tw_ticker(raw_code)
    try:
        with st.spinner("正在取得最新股價與財務資料..."):
            raw_df, resolved_ticker = load_price_data(ticker_input, start_date=start_date, end_date=end_date)
            info = load_ticker_info(resolved_ticker)
            stmt = load_financials(resolved_ticker)
            full_df = calculate_indicators(raw_df)
            df = trim_to_user_period(full_df, start_date).dropna(subset=["RSI14", "MACD_HIST"])

        if df.empty or len(df) < 20:
            st.error("資料量不足，無法計算完整指標。請增加月份、改用更長期間，或確認股票代碼。")
            st.stop()
        if len(df) < 60:
            st.warning("目前分析期間較短，部分長週期指標與回測結果會比較不穩定；若要看中長期策略，建議至少 1 年以上。")

        last = df.iloc[-1]
        prev = df.iloc[-2]
        latest_date = df.index[-1].strftime("%Y-%m-%d")
        stale_days = (datetime.now() - df.index[-1]).days
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
        c1.metric("收盤價", format_number(last["Close"], 2), delta=f"{last['Return_1D'] * 100:.2f}%")
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
            st.caption("圖例已固定：深藍＝收盤價、橘色＝MA20、綠色＝MA60、紫色＝MA120、灰色虛線＝布林上下軌。")
            st.altair_chart(make_price_chart(df), use_container_width=True)
            st.markdown("#### 指標明細")
            indicator_cols = ["Close", "Volume", "MA5", "MA20", "MA60", "MA120", "RSI14", "MACD_DIF", "MACD_SIGNAL", "MACD_HIST", "ATR14", "Volume_Ratio", "Return_5D", "Return_20D"]
            st.dataframe(df[indicator_cols].tail(30).iloc[::-1], use_container_width=True)

        with tab3:
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("本益比 PE", format_number(info.get("trailingPE"), 2))
            f2.metric("EPS", format_number(info.get("trailingEps"), 2))
            f3.metric("殖利率", format_number(safe_float(info.get("dividendYield")) * 100, 2, "%"))
            f4.metric("市值", format_large_twd(info.get("marketCap")))
            f5.metric("Beta", format_number(info.get("beta"), 2))

            fin_table = build_financial_table(stmt)
            if fin_table.empty:
                st.info("此股票目前無法從 yfinance 取得完整季財務資料。")
            else:
                st.markdown("#### 近八季財務趨勢")
                st.caption("由最新完整季度往前列出最多 8 季；若資料源尚未更新，會自動略過整季空白資料。QoQ 是季增率，YoY 是與去年同季相比。")
                st.dataframe(fin_table.iloc[::-1], use_container_width=True)
                if "營收QoQ%" in fin_table.columns and not fin_table["營收QoQ%"].dropna().empty:
                    latest_growth = fin_table["營收QoQ%"].dropna().iloc[-1]
                    if latest_growth > 0:
                        st.success(f"最新一季營收季增率約 {latest_growth:.2f}%，基本面短期動能偏正向。")
                    else:
                        st.warning(f"最新一季營收季增率約 {latest_growth:.2f}%，需確認是淡季、循環下滑或公司特殊因素。")

        with tab4:
            st.markdown("#### 單筆交易風險規劃")
            st.info(f"目前設定單筆最大風險為 {risk_pct:.2f}%：若帳戶資金為 {capital:,.0f} 元，這筆交易理論上最多承受約 {capital * risk_pct / 100:,.0f} 元損失。")
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
            st.info("這裡不是預測未來，而是檢查：同一段資料期間內，不同交易規則過去表現如何。請優先看最大回撤、Sharpe、持股時間，再看報酬。")

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
                opt_family = st.selectbox("選擇要最佳化的策略族", ["均線趨勢 MA", "EMA動能", "突破追價"])
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
                        st.warning("提醒：不要只選第一名。請找 Sharpe 不錯、最大回撤可接受、交易次數不要太少，而且鄰近參數也表現穩定的組合。")

                with st.expander("各策略適合什麼情境？"):
                    st.markdown("""
                    - **保守均線趨勢**：適合想避開大跌、不要死太慘，有賺就好的防守型操作。
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
        st.info("請確認網路連線、股票代碼，或稍後再試。若是上櫃股票，可直接輸入 8069.TWO 這類格式。")

if run_watchlist:
    codes = [x.strip() for x in watchlist_text.replace("\n", ",").split(",") if x.strip()]
    rows = []
    progress = st.progress(0)
    for i, code in enumerate(codes):
        try:
            raw_df, resolved = load_price_data(normalize_tw_ticker(code), start_date=start_date, end_date=end_date)
            info = load_ticker_info(resolved)
            full_df = calculate_indicators(raw_df)
            df = trim_to_user_period(full_df, start_date).dropna(subset=["RSI14", "MACD_HIST"])
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
                    "殖利率%": round(safe_float(info.get("dividendYield")) * 100, 2) if not pd.isna(safe_float(info.get("dividendYield"))) else np.nan,
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
