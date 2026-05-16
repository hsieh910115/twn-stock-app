from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .financials import dividend_yield_pct
from .utils import is_valid_number, safe_float


def score_stock(df: pd.DataFrame, info: Dict, mode: str) -> Dict:
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
        if is_valid_number(last.get("MA5")) and is_valid_number(last.get("MA20")):
            if close > last["MA5"] > last["MA20"]:
                score += 3
                reasons.append("短均線呈多頭排列，短線趨勢強勢。")
            elif close < last["MA5"] < last["MA20"]:
                score -= 3
                warnings.append("短均線呈空頭排列，短線結構偏弱。")
        if is_valid_number(last.get("MACD_HIST")) and is_valid_number(prev.get("MACD_HIST")):
            if last["MACD_HIST"] > 0 and last["MACD_HIST"] > prev["MACD_HIST"]:
                score += 2
                reasons.append("MACD 動能持續擴大。")
            elif last["MACD_HIST"] < 0:
                score -= 2
                warnings.append("MACD 動能仍偏弱。")
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
        if is_valid_number(last.get("Volume_Ratio")) and last["Volume_Ratio"] > 1.5 and close > prev["Close"]:
            score += 1
            reasons.append("量能放大且收漲，市場買盤積極。")
        elif is_valid_number(last.get("Volume_Ratio")) and last["Volume_Ratio"] > 1.8 and close < prev["Close"]:
            score -= 1
            warnings.append("放量下跌，短線賣壓偏重。")
        if is_valid_number(last.get("Return_20D")):
            if last["Return_20D"] > 0.15:
                score += 1
                reasons.append("近 20 日動能強勢。")
            elif last["Return_20D"] < -0.15:
                score -= 1
                warnings.append("近 20 日動能偏弱。")
        if is_valid_number(last.get("BB_UPPER")) and is_valid_number(last.get("BB_LOWER")):
            if close > last["BB_UPPER"]:
                score -= 1
                warnings.append("股價偏離布林上軌，追高風險較高。")
            elif close < last["BB_LOWER"]:
                score += 1
                reasons.append("股價接近布林下軌，可能有反彈機會。")
    else:
        if is_valid_number(last.get("MA60")) and is_valid_number(last.get("MA120")):
            if close > last["MA60"] > last["MA120"]:
                score += 3
                reasons.append("季線與半年線呈多頭排列。")
            elif close < last["MA60"] < last["MA120"]:
                score -= 3
                warnings.append("中長期均線偏空。")
        if is_valid_number(last.get("MA240")):
            if close > last["MA240"]:
                score += 2
                reasons.append("股價位於年線之上。")
            else:
                score -= 2
                warnings.append("股價仍低於年線。")
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
        if not pd.isna(eps):
            if eps > 5:
                score += 1
                reasons.append("EPS 表現良好。")
            elif eps <= 0:
                score -= 1
                warnings.append("EPS 非正值。")
        if not pd.isna(dividend_yield):
            if dividend_yield >= 5:
                score += 1
                reasons.append("殖利率具吸引力。")
            elif dividend_yield < 1:
                score -= 1
                warnings.append("殖利率偏低。")
        if not pd.isna(beta):
            if beta <= 1:
                score += 1
                reasons.append("Beta 較低，波動相對穩定。")
            elif beta > 1.6:
                score -= 1
                warnings.append("Beta 偏高，波動風險較大。")

    if is_valid_number(last.get("Drawdown")) and last["Drawdown"] < -0.35:
        warnings.append("距歷史高點回撤超過 35%。")

    score = max(-10, min(10, score))
    score_10 = round((score + 10) / 20 * 10, 1)
    if score_10 >= 8.5:
        label, level = "條件優良／可優先觀察", "success"
    elif score_10 >= 7:
        label, level = "偏多格局／可列入候選", "info"
    elif score_10 >= 5.5:
        label, level = "中性偏多／等待確認", "info"
    elif score_10 >= 4:
        label, level = "中性偏弱／保守觀望", "warning"
    elif score_10 >= 2.5:
        label, level = "偏弱格局／不宜積極", "warning"
    else:
        label, level = "高風險／暫不建議介入", "error"
    return {"score": score, "score_10": score_10, "label": label, "level": level, "reasons": reasons[:6], "warnings": warnings[:6]}


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
    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_budget": risk_budget,
        "shares": shares,
        "lots": shares / 1000,
        "risk_per_share": risk_per_share,
    }
