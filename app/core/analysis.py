from __future__ import annotations

import os
import time
from datetime import date, datetime
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import STRATEGY_PRESETS, backtest_all_strategies, backtest_strategy, optimize_parameters, strategy_execution_advice
from .data import load_fast_info, load_financials, load_price_data, load_ticker_info, resolve_stock_input
from .financials import build_financial_table, derive_fundamental_metrics, estimate_beta_vs_twii
from .indicators import calculate_indicators
from .scoring import risk_plan, score_stock
from .utils import dataframe_records, display_code, safe_float, series_dict


CACHE_TTL_SECONDS = int(os.getenv("APP_CACHE_TTL_SECONDS", "900"))


def _cache_bucket() -> int:
    return int(time.time() // max(CACHE_TTL_SECONDS, 1))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def prepare_analysis_frame(ticker: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw_df, resolved_ticker = load_price_data(resolve_stock_input(ticker))
    if start_date is not None:
        raw_df = raw_df[raw_df.index >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        raw_df = raw_df[raw_df.index <= pd.Timestamp(end_date)].copy()
    if raw_df.empty:
        raise ValueError("指定期間內無法取得資料，請檢查日期範圍或股票代碼。")
    full_df = calculate_indicators(raw_df)
    usable_df = full_df.dropna(subset=["Close", "RSI14", "MACD_HIST"])
    if usable_df.empty or len(usable_df) < 20:
        raise ValueError("資料量不足，無法計算完整指標。")
    return raw_df, usable_df, resolved_ticker


def analyze_stock(
    ticker: str,
    mode: str = "短線／波段",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    capital: float = 100000,
    risk_pct: float = 10,
) -> dict:
    return _analyze_stock_cached(ticker, mode, start_date, end_date, float(capital), float(risk_pct), _cache_bucket())


@lru_cache(maxsize=256)
def _analyze_stock_cached(
    ticker: str,
    mode: str,
    start_date: Optional[date],
    end_date: Optional[date],
    capital: float,
    risk_pct: float,
    cache_bucket: int,
) -> dict:
    _, df, resolved_ticker = prepare_analysis_frame(ticker, start_date, end_date)
    info = load_ticker_info(resolved_ticker)
    fast_info = load_fast_info(resolved_ticker)
    stmt = load_financials(resolved_ticker)
    fin_table = build_financial_table(stmt, resolved_ticker)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    fundamentals = derive_fundamental_metrics(info, fast_info, last["Close"], fin_table)
    if fundamentals.get("beta") is None or pd.isna(fundamentals.get("beta")):
        fundamentals["beta"] = estimate_beta_vs_twii(df)

    levels = pd.DataFrame({
        "項目": ["MA20（月線）", "MA60（季線）", "MA120（半年線）", "52週高點", "52週低點", "ATR14"],
        "數值": [last.get("MA20"), last.get("MA60"), last.get("MA120"), last.get("High_52W"), last.get("Low_52W"), last.get("ATR14")],
        "與現價距離%": [
            (last.get("MA20") / last["Close"] - 1) * 100,
            (last.get("MA60") / last["Close"] - 1) * 100,
            (last.get("MA120") / last["Close"] - 1) * 100,
            (last.get("High_52W") / last["Close"] - 1) * 100,
            (last.get("Low_52W") / last["Close"] - 1) * 100,
            np.nan,
        ],
    })

    return _json_safe({
        "ticker": resolved_ticker,
        "display_code": display_code(resolved_ticker),
        "name": info.get("longName") or info.get("shortName") or display_code(resolved_ticker),
        "mode": mode,
        "latest_date": df.index[-1],
        "stale_days": (datetime.now() - df.index[-1]).days,
        "latest": series_dict(last),
        "previous": series_dict(prev),
        "price_change": safe_float(last["Close"] - prev["Close"]),
        "price_change_pct": safe_float(last["Return_1D"] * 100),
        "score": score_stock(df, {**fast_info, **info}, mode),
        "fundamentals": fundamentals,
        "risk_plan": risk_plan(last, capital, risk_pct),
        "levels": dataframe_records(levels),
        "financials": dataframe_records(fin_table),
        "price_history": dataframe_records(df.tail(260)),
        "recent_indicators": dataframe_records(df.tail(30).iloc[::-1]),
    })


def scan_watchlist(
    symbols: list[str],
    mode: str = "短線／波段",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[dict]:
    return _scan_watchlist_cached(tuple(symbols), mode, start_date, end_date, _cache_bucket())


@lru_cache(maxsize=128)
def _scan_watchlist_cached(
    symbols: tuple[str, ...],
    mode: str,
    start_date: Optional[date],
    end_date: Optional[date],
    cache_bucket: int,
) -> list[dict]:
    rows = []
    for symbol in symbols:
        try:
            _, df, resolved = prepare_analysis_frame(symbol, start_date, end_date)
            info = load_ticker_info(resolved)
            last = df.iloc[-1]
            score = score_stock(df, info, mode)
            rows.append(_json_safe({
                "代碼": resolved,
                "名稱": info.get("shortName") or info.get("longName") or display_code(resolved),
                "日期": df.index[-1],
                "收盤": round(safe_float(last["Close"]), 2),
                "1日%": round(safe_float(last["Return_1D"] * 100), 2),
                "20日%": round(safe_float(last["Return_20D"] * 100), 2),
                "RSI": round(safe_float(last["RSI14"]), 1),
                "量比": round(safe_float(last["Volume_Ratio"]), 2),
                "分數": score["score_10"],
                "結論": score["label"],
            }))
        except Exception as exc:
            rows.append({"代碼": symbol, "名稱": "讀取失敗", "結論": str(exc)})
    return sorted(rows, key=lambda row: row.get("分數") or -1, reverse=True)


def backtest_summary(
    ticker: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    execution_mode: str = "理想",
    cost_rate: float = 0.003,
    strategy_name: Optional[str] = None,
) -> dict:
    return _backtest_summary_cached(ticker, start_date, end_date, execution_mode, float(cost_rate), strategy_name, _cache_bucket())


@lru_cache(maxsize=256)
def _backtest_summary_cached(
    ticker: str,
    start_date: Optional[date],
    end_date: Optional[date],
    execution_mode: str,
    cost_rate: float,
    strategy_name: Optional[str],
    cache_bucket: int,
) -> dict:
    _, df, resolved = prepare_analysis_frame(ticker, start_date, end_date)
    comparison = backtest_all_strategies(df, execution_mode=execution_mode, cost_rate=cost_rate)
    strategy_name = strategy_name or next(iter(STRATEGY_PRESETS))
    detail = backtest_strategy(df, strategy_name, execution_mode=execution_mode, cost_rate=cost_rate)
    detail_payload = {}
    if detail:
        detail_payload = {
            key: value for key, value in detail.items()
            if key != "df"
        }
        detail_payload["equity_curve"] = dataframe_records(detail["df"][["BuyHold_Equity", "Strategy_Equity", "Signal"]].tail(260))
    return _json_safe({
        "ticker": resolved,
        "comparison": dataframe_records(comparison),
        "strategy": strategy_name,
        "strategy_meta": STRATEGY_PRESETS.get(strategy_name),
        "detail": detail_payload,
    })


def optimize_summary(
    ticker: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strategy_family: str = "全部策略",
    execution_mode: str = "理想",
    cost_rate: float = 0.003,
    optimize_target: str = "穩健分數最高",
) -> dict:
    return _optimize_summary_cached(ticker, start_date, end_date, strategy_family, execution_mode, float(cost_rate), optimize_target, _cache_bucket())


@lru_cache(maxsize=128)
def _optimize_summary_cached(
    ticker: str,
    start_date: Optional[date],
    end_date: Optional[date],
    strategy_family: str,
    execution_mode: str,
    cost_rate: float,
    optimize_target: str,
    cache_bucket: int,
) -> dict:
    _, df, resolved = prepare_analysis_frame(ticker, start_date, end_date)
    result = optimize_parameters(df, strategy_family, execution_mode, cost_rate, optimize_target)
    return _json_safe({"ticker": resolved, "rows": dataframe_records(result)})


def strategy_advice_summary(
    ticker: str,
    strategy_name: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict:
    return _strategy_advice_summary_cached(ticker, strategy_name, start_date, end_date, _cache_bucket())


@lru_cache(maxsize=256)
def _strategy_advice_summary_cached(
    ticker: str,
    strategy_name: str,
    start_date: Optional[date],
    end_date: Optional[date],
    cache_bucket: int,
) -> dict:
    _, df, resolved = prepare_analysis_frame(ticker, start_date, end_date)
    advice = strategy_execution_advice(df, strategy_name)
    return _json_safe({
        "ticker": resolved,
        "strategy": strategy_name,
        "meta": STRATEGY_PRESETS.get(strategy_name),
        "advice": advice,
    })


def load_ai_csv(kind: str, top_n: int = 20) -> dict:
    mapping = {
        "momentum": "data/ai_momentum_top.csv",
        "potential": "data/ai_potential_top.csv",
    }
    path = mapping.get(kind)
    if not path:
        raise ValueError("kind 必須是 momentum 或 potential。")
    if not os.path.exists(path):
        return {"kind": kind, "updated_at": None, "rows": []}
    df = pd.read_csv(path)
    updated_at = None
    if "Updated_At" in df.columns and not df["Updated_At"].dropna().empty:
        updated_at = pd.to_datetime(df["Updated_At"]).max() + pd.Timedelta(hours=8)
    return _json_safe({"kind": kind, "updated_at": updated_at, "rows": dataframe_records(df.head(top_n))})
