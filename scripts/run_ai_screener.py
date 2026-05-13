# scripts/run_ai_screener.py
import os
import requests
import urllib3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_tw_stock_list():
    stock_dict = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for m in [2, 4]:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
        res = requests.get(url, headers=headers, verify=False, timeout=20)
        df = pd.read_html(res.text)[0].iloc[1:]

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


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = run_ai_momentum_scan(batch_size=50)

    output_path = "data/ai_momentum_top.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")