from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.analysis import analyze_stock, backtest_summary, load_ai_csv, optimize_summary, scan_watchlist, strategy_advice_summary
from app.core.backtest import STRATEGY_PRESETS

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(
    title="台股投資分析 API",
    description="Streamlit 之外的新架構：提供股票分析、觀察清單、回測與 AI CSV 資料。",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/analyze")
def analyze(
    ticker: str = Query("台積電", description="股票名稱或代碼，例如 台積電、2330、8069.TWO"),
    mode: str = Query("短線／波段", description="短線／波段 或 長線／存股"),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    capital: float = 100000,
    risk_pct: float = 10,
) -> dict:
    try:
        return analyze_stock(ticker, mode, start_date, end_date, capital, risk_pct)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/watchlist")
def watchlist(
    symbols: str = Query(..., description="逗號分隔，例如 2330,2454,台積電"),
    mode: str = "短線／波段",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    items = [item.strip() for item in symbols.replace("\n", ",").split(",") if item.strip()]
    return {"rows": scan_watchlist(items, mode, start_date, end_date)}


@app.get("/api/backtest")
def backtest(
    ticker: str = "台積電",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    execution_mode: str = "理想",
    cost_rate: float = 0.003,
    strategy_name: Optional[str] = None,
) -> dict:
    try:
        return backtest_summary(ticker, start_date, end_date, execution_mode, cost_rate, strategy_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/optimize")
def optimize(
    ticker: str = "台積電",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strategy_family: str = "全部策略",
    execution_mode: str = "理想",
    cost_rate: float = 0.003,
    optimize_target: str = "穩健分數最高",
) -> dict:
    try:
        return optimize_summary(ticker, start_date, end_date, strategy_family, execution_mode, cost_rate, optimize_target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/strategy-advice")
def strategy_advice(
    ticker: str = "台積電",
    strategy_name: str = "保守均線趨勢｜少交易",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    try:
        return strategy_advice_summary(ticker, strategy_name, start_date, end_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/strategies")
def strategies() -> dict:
    return {"strategies": [{"name": name, **meta} for name, meta in STRATEGY_PRESETS.items()]}


@app.get("/api/ai/{kind}")
def ai_csv(kind: str, top_n: int = 20) -> dict:
    try:
        return load_ai_csv(kind, top_n)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
