# 📈 台股分析 APP

一個以 Python + Streamlit 建立的台股分析與視覺化平台，整合技術分析、基本面資訊、風險評估與互動式圖表，提供接近專業看盤軟體的操作體驗。

---

# ✨ 功能特色

## 📊 技術分析圖表

* K 線圖（台股紅漲綠跌）
* MA 均線系統

  * MA5
  * MA20
  * MA60
  * MA120
  * MA240
* 布林通道（Bollinger Bands）
* 成交量副圖
* MACD 指標副圖

  * DIF
  * MACD Signal
  * Histogram
  * 零軸
* 十字線同步對齊
* TradingView 風格互動操作

---

## 📈 股票資訊

* 即時股價
* 漲跌幅
* 成交量
* 市值
* 本益比（PE）
* EPS
* Beta
* 殖利率

---

## ⭐ 觀察清單

* 自訂股票追蹤
* 自動計算分數
* 顯示技術面與基本面資訊
* 快速比較多檔股票

---

## 🔍 回測與策略分析

* 多種投資模式

  * 長線 / 存股
  * 短線 / 波段
* 技術條件評分
* 風險控制設定
* 策略參數最佳化

---

# 🖥️ 使用技術

| 技術        | 說明         |
| --------- | ---------- |
| Python    | 核心開發語言     |
| Streamlit | Web APP 框架 |
| Plotly    | 互動式圖表      |
| Pandas    | 資料分析       |
| NumPy     | 數值運算       |
| yfinance  | 股票資料來源     |
| FinMind   | 台股基本面資料    |

---

# 🚀 安裝方式

## 1️⃣ Clone 專案

```bash
git clone https://github.com/yourname/your-repo.git
cd your-repo
```

---

## 2️⃣ 安裝套件

```bash
pip install -r requirements.txt
```

---

## 3️⃣ 啟動 APP

```bash
streamlit run app.py
```

---

# 🔑 API 設定（FinMind）

建議使用 Streamlit Secrets 儲存 Token。

建立：

```text
.streamlit/secrets.toml
```

內容：

```toml
FINMIND_TOKEN = "YOUR_TOKEN"
```

程式中讀取：

```python
st.secrets["FINMIND_TOKEN"]
```

---

# 📷 畫面預覽

## 主畫面

* 股票資訊總覽
* 技術分析圖
* 觀察清單
* 更新公告

---

# ⚠️ 免責聲明

本平台僅供學習與研究參考，資料與分析結果可能存在誤差或延遲，不保證完全正確，亦不構成任何投資建議，請自行判斷並注意投資風險。

---

# 📌 更新公告

APP 內建 Changelog 更新公告系統，可於畫面底部查看版本更新內容。

---

# 📬 開發者

Developed by hsieh910115
