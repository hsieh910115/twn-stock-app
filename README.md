## 📈 Taiwan Stock Analysis Dashboard

An interactive stock analysis platform built with Python and Streamlit, designed for Taiwan stock market investors.
The dashboard integrates technical analysis, risk evaluation, basic financial metrics, and visual trading indicators into a clean and modern interface.

---

### ✨ Features

* Interactive candlestick (K-line) charts
* Moving averages (MA5 / MA20 / MA60 / MA120 / MA240)
* Bollinger Bands
* MACD & volume subplots
* Technical trend analysis
* Risk and support/resistance evaluation
* Watchlist system
* Backtesting and strategy analysis
* Responsive TradingView-style chart interaction



### ⚠ Disclaimer

This platform is intended for educational and research purposes only.
Market data and analysis results may contain delays or inaccuracies and should not be considered financial advice or investment recommendations.

---

## Preview
<img width="1904" height="1006" alt="截圖 2026-05-14 01 03 51" src="https://github.com/user-attachments/assets/b54dee98-b39a-4339-a13c-a6a6f0bbcb22" />
## 非 Streamlit API 架構

## 新架構規劃

原本的 `stock_app.py` 會先保留作為 Streamlit 版本。新的非 Streamlit 架構從 `app/` 開始：

```text
app/
  core/       # 資料抓取、技術指標、評分、財務資料、回測
  api/        # FastAPI endpoints
  web/        # 未來 React / Next.js 前端
```

## 啟動 FastAPI

```bash
conda activate stock
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

或是不切換 shell 環境，直接用：

```bash
conda run -n stock uvicorn app.api.main:app --reload
```

啟動後可測試：

- `http://127.0.0.1:8000/`：第一版網頁 App
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/api/analyze?ticker=2330`
- `http://127.0.0.1:8000/api/watchlist?symbols=2330,2454`
- `http://127.0.0.1:8000/api/backtest?ticker=2330`
- `http://127.0.0.1:8000/api/optimize?ticker=2330`
- `http://127.0.0.1:8000/api/strategy-advice?ticker=2330`
- `http://127.0.0.1:8000/api/strategies`
- `http://127.0.0.1:8000/api/ai/momentum`
- `http://127.0.0.1:8000/api/ai/potential`

## 目前非 Streamlit 網頁已支援

- 單股總覽：評分、收盤價、RSI、MACD、量比、最新交易日
- 技術分析：價格走勢、K 線、MA20/MA60、布林通道、成交量、MACD
- 基本面：PE、EPS、殖利率、Beta、市值、近 8 季財務趨勢
- 風險控管：ATR 停損/停利、可承受損失、估算股數與張數、交易前檢核
- 歷史回測：六種預設策略、策略比較、淨值曲線、參數最佳化
- 策略執行助手：目前狀態、進出場規則、關鍵價位與距離
- 觀察清單：多檔股票批次掃描
- AI 選股：妖股動能與潛力股 CSV 排名
- 模式差異：短線/長線評分邏輯與 0-10 分評語

## 雲端部署

本機開發時使用：

```bash
uvicorn app.api.main:app --reload
```

雲端部署時不要使用 `--reload`，而且必須綁定平台提供的 port：

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
```

此專案已提供：

- `render.yaml`：Render Blueprint，可連 GitHub repository 自動部署
- `Procfile`：支援 Render / Railway / Heroku 類型平台
- `Dockerfile`：支援 Fly.io、VPS、Cloud Run 或其他 Docker 平台
- `.env.example`：環境變數範例

建議部署環境變數：

```text
FINMIND_TOKEN=你的 FinMind token，可留空但請求額度較受限
APP_CACHE_TTL_SECONDS=900
```

部署到雲端後，你的筆電不需要開著。其他人只要打開雲端網址就能使用 App。
