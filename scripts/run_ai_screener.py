# scripts/run_ai_screener.py
import os
import requests
import urllib3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from io import StringIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_tw_stock_list():
    stock_dict = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for m in [2, 4]:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
        res = requests.get(url, headers=headers, verify=False, timeout=20)
        df = pd.read_html(StringIO(res.text))[0].iloc[1:]

        for _, row in df.iterrows():
            try:
                code_name = str(row[0]).split()
                if len(code_name) != 2:
                    continue

                code, name = code_name
                cat = str(row[4])

                if len(code) == 4 or code.startswith("00"):
                    if cat not in ["權證", "牛熊證", "認購(售)權證"]:
                        suffix = ".TW" if m == 2 else ".TWO"
                        stock_dict[f"{code}{suffix}"] = {
                            "name": name,
                            "industry": cat,
                        }
            except Exception:
                continue

    return stock_dict


def run_ai_momentum_scan(batch_size=50):
    stock_dict = get_tw_stock_list()
    all_tickers = list(stock_dict.keys())

    records = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"Downloading {min(i + batch_size, len(all_tickers))}/{len(all_tickers)}")

        try:
            data = yf.download(
                batch,
                period="100d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            for ticker in batch:
                try:
                    df = data[ticker] if len(batch) > 1 else data
                    df = df.dropna()

                    if df.empty or len(df) < 60:
                        continue

                    close = df["Close"]

                    ma5 = close.rolling(5).mean().iloc[-1]
                    ma20 = close.rolling(20).mean().iloc[-1]
                    ma60 = close.rolling(60).mean().iloc[-1]
                    current_close = close.iloc[-1]

                    daily_ret = close.pct_change()
                    hist_vol = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

                    std20 = close.rolling(20).std().iloc[-1]
                    bb_upper = ma20 + 2 * std20
                    bb_lower = ma20 - 2 * std20
                    bb_width = (bb_upper - bb_lower) / ma20 * 100

                    p_to_ma60 = (current_close / ma60 - 1) * 100
                    trend_str = (ma5 / ma60 - 1) * 100
                    p_to_ma20 = (current_close / ma20 - 1) * 100
                    p_to_bbupper = (current_close / bb_upper - 1) * 100
                    roc_10 = (current_close - close.iloc[-11]) / close.iloc[-11] * 100

                    values = [
                        hist_vol,
                        bb_width,
                        p_to_ma60,
                        trend_str,
                        p_to_ma20,
                        p_to_bbupper,
                        roc_10,
                    ]

                    if any(pd.isna(v) or np.isinf(v) for v in values):
                        continue

                    records.append({
                        "ID": ticker.replace(".TW", "").replace(".TWO", ""),
                        "Ticker": ticker,
                        "Name": stock_dict[ticker]["name"],
                        "Industry": stock_dict[ticker]["industry"],
                        "Close": current_close,
                        "MA5": ma5,
                        "F_Hist_Vol": hist_vol,
                        "F_BB_Width": bb_width,
                        "F_P_to_MA60": p_to_ma60,
                        "F_Trend_Strength": trend_str,
                        "F_P_to_MA20": p_to_ma20,
                        "F_P_to_BBUpper": p_to_bbupper,
                        "F_ROC_10": roc_10,
                    })

                except Exception:
                    continue

        except Exception as e:
            print(f"Batch failed: {e}")
            continue

    df_res = pd.DataFrame(records)

    if df_res.empty:
        return df_res

    features = [
        "F_Hist_Vol",
        "F_BB_Width",
        "F_P_to_MA60",
        "F_Trend_Strength",
        "F_P_to_MA20",
        "F_P_to_BBUpper",
        "F_ROC_10",
    ]

    weights = [29.08, 19.33, 10.39, 7.67, 7.26, 5.09, 4.25]

    for f in features:
        df_res[f"{f}_Rank"] = df_res[f].rank(pct=True)

    df_res["AI_Score"] = 0.0

    for f, w in zip(features, weights):
        df_res["AI_Score"] += df_res[f"{f}_Rank"] * w

    df_res["AI_Score"] = df_res["AI_Score"] / sum(weights) * 100

    df_res = df_res[df_res["Close"] >= df_res["MA5"]].copy()
    df_res = df_res.sort_values("AI_Score", ascending=False).reset_index(drop=True)
    df_res.insert(0, "Rank", df_res.index + 1)

    df_res["Updated_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df_res


def run_ai_potential_scan(batch_size=50):
    stock_dict = get_tw_stock_list()
    all_tickers = list(stock_dict.keys())

    records = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"Potential scan {min(i + batch_size, len(all_tickers))}/{len(all_tickers)}")

        try:
            data = yf.download(
                batch,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            for ticker in batch:
                try:
                    df = data[ticker] if len(batch) > 1 else data
                    df = df.dropna()

                    if df.empty or len(df) < 120:
                        continue

                    close = df["Close"]
                    volume = df["Volume"]

                    current_close = close.iloc[-1]
                    ma10 = close.rolling(10).mean().iloc[-1]
                    ma20 = close.rolling(20).mean().iloc[-1]
                    ma60 = close.rolling(60).mean().iloc[-1]
                    ma120 = close.rolling(120).mean().iloc[-1]

                    high_52w = close.rolling(240).max().iloc[-1] if len(close) >= 240 else close.max()
                    low_52w = close.rolling(240).min().iloc[-1] if len(close) >= 240 else close.min()

                    drawdown_52w = (current_close / high_52w - 1) * 100
                    rebound_space = (high_52w / current_close - 1) * 100

                    daily_ret = close.pct_change()
                    hist_vol = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

                    delta = close.diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
                    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    rsi14 = (100 - (100 / (1 + rs))).iloc[-1]

                    ma_values = [ma10, ma20, ma60]
                    ma_compression = (max(ma_values) - min(ma_values)) / current_close * 100

                    volume_ma20 = volume.rolling(20).mean().iloc[-1]
                    volume_ratio = volume.iloc[-1] / volume_ma20 if volume_ma20 > 0 else np.nan

                    p_to_ma60 = (current_close / ma60 - 1) * 100
                    p_to_ma120 = (current_close / ma120 - 1) * 100

                    # ===== 潛力股分數：0~100 =====
                    score = 0
                    reasons = []
                    risks = []

                    # 1. 跌深修復空間 25
                    if -45 <= drawdown_52w <= -15:
                        score += 18
                        reasons.append("股價距離52週高點有修復空間")
                    elif drawdown_52w < -45:
                        score += 8
                        risks.append("跌幅過深，需留意基本面是否惡化")

                    if rebound_space >= 25:
                        score += 7
                        reasons.append("潛在反彈空間較大")

                    # 2. 技術安全邊際 25
                    if 35 <= rsi14 <= 55:
                        score += 10
                        reasons.append("RSI位於低檔整理區")
                    elif rsi14 < 30:
                        score += 5
                        risks.append("RSI過低，短線賣壓仍重")
                    elif rsi14 > 70:
                        score -= 8
                        risks.append("RSI偏高，不符合低檔潛伏")

                    if ma_compression <= 8:
                        score += 10
                        reasons.append("均線糾結，可能處於整理蓄勢")
                    elif ma_compression <= 15:
                        score += 5

                    if current_close >= low_52w * 1.08:
                        score += 5
                        reasons.append("股價已脫離52週低點")

                    # 3. 防守能力 20
                    if hist_vol <= 35:
                        score += 10
                        reasons.append("波動相對可控")
                    elif hist_vol > 60:
                        score -= 8
                        risks.append("波動過高，風險較大")

                    if volume_ratio >= 0.8:
                        score += 5
                        reasons.append("成交量尚未明顯枯竭")

                    if current_close > ma120:
                        score += 5
                        reasons.append("股價仍在中長期均線之上")
                    elif current_close < ma120:
                        risks.append("仍低於中長期均線，需耐心等待")

                    # 4. 估值替代訊號 20
                    if p_to_ma60 < -5:
                        score += 8
                        reasons.append("股價低於季線，具修復空間")
                    elif p_to_ma60 > 15:
                        score -= 5
                        risks.append("股價已明顯高於季線，低估性下降")

                    if p_to_ma120 < -5:
                        score += 7
                        reasons.append("股價低於半年線，具左側布局特徵")

                    if -30 <= drawdown_52w <= -10 and hist_vol <= 45:
                        score += 5
                        reasons.append("跌深但波動未失控")

                    # 5. 左側潛伏特徵 10
                    if ma_compression <= 10 and 35 <= rsi14 <= 60 and volume_ratio >= 0.7:
                        score += 10
                        reasons.append("具低檔潛伏與整理跡象")

                    score = max(0, min(100, score))

                    if score >= 80:
                        tag = "護城河價值型"
                    elif score >= 70 and drawdown_52w <= -25:
                        tag = "轉折曙光型"
                    elif score >= 65 and volume_ratio < 1:
                        tag = "孤兒股翻身型"
                    elif hist_vol <= 30:
                        tag = "防守修復型"
                    else:
                        tag = "低檔觀察型"

                    if not reasons:
                        reasons.append("低估條件尚不明顯")
                    if not risks:
                        risks.append("仍需觀察技術面是否轉強")

                    records.append({
                        "ID": ticker.replace(".TW", "").replace(".TWO", ""),
                        "Ticker": ticker,
                        "Name": stock_dict[ticker]["name"],
                        "Industry": stock_dict[ticker]["industry"],
                        "Close": current_close,
                        "AI_Potential_Score": round(score, 2),
                        "Tag": tag,
                        "RSI14": round(rsi14, 2),
                        "Hist_Vol": round(hist_vol, 2),
                        "MA_Compression": round(ma_compression, 2),
                        "Drawdown_52W": round(drawdown_52w, 2),
                        "Rebound_Space": round(rebound_space, 2),
                        "P_to_MA60": round(p_to_ma60, 2),
                        "P_to_MA120": round(p_to_ma120, 2),
                        "Volume_Ratio": round(volume_ratio, 2),
                        "Reason": "、".join(reasons[:3]),
                        "Risk": "、".join(risks[:2]),
                    })

                except Exception:
                    continue

        except Exception as e:
            print(f"Potential batch failed: {e}")
            continue

    df_res = pd.DataFrame(records)

    if df_res.empty:
        return df_res

    df_res = df_res.sort_values("AI_Potential_Score", ascending=False).reset_index(drop=True)
    df_res.insert(0, "Rank", df_res.index + 1)
    df_res["Updated_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df_res


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df_momentum = run_ai_momentum_scan(batch_size=50)
    momentum_path = "data/ai_momentum_top.csv"
    df_momentum.to_csv(momentum_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {momentum_path}")

    df_potential = run_ai_potential_scan(batch_size=50)
    potential_path = "data/ai_potential_top.csv"
    df_potential.to_csv(potential_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {potential_path}")