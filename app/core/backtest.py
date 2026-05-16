from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


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


def compute_backtest_stats(bt: pd.DataFrame) -> Dict:
    if bt.empty or len(bt) < 30:
        return {}
    bt = bt.copy()
    bt["BuyHold_Equity"] = (1 + bt["Return_1D"]).cumprod()
    bt["Strategy_Equity"] = (1 + bt["Strategy_Return"]).cumprod()
    days = max((bt.index[-1] - bt.index[0]).days, 1)
    years = days / 365.25
    total_return = bt["Strategy_Equity"].iloc[-1] - 1
    cagr = bt["Strategy_Equity"].iloc[-1] ** (1 / years) - 1 if bt["Strategy_Equity"].iloc[-1] > 0 else np.nan
    vol = bt["Strategy_Return"].std() * np.sqrt(252)
    sharpe = cagr / vol if vol and not pd.isna(vol) and vol != 0 else np.nan
    return {
        "df": bt,
        "total_return": total_return,
        "buyhold_return": bt["BuyHold_Equity"].iloc[-1] - 1,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": (bt["Strategy_Equity"] / bt["Strategy_Equity"].cummax() - 1).min(),
        "win_rate": (bt.loc[bt["Signal"].shift(1).fillna(0) == 1, "Strategy_Return"] > 0).mean(),
        "exposure": bt["Signal"].mean(),
        "trades": int((bt["Signal"].diff() == 1).sum()),
    }


def apply_backtest_execution(data: pd.DataFrame, signal: pd.Series, mode: str = "理想", cost_rate: float = 0.0) -> pd.DataFrame:
    bt = data.copy()
    bt["Signal"] = signal.reindex(bt.index).fillna(0)
    trade = bt["Signal"].diff().abs().fillna(0)
    if mode == "真實":
        bt["NextOpen_Return"] = bt["Close"] / bt["Open"] - 1
        bt["Strategy_Return"] = bt["Signal"].shift(1).fillna(0) * bt["NextOpen_Return"]
    else:
        bt["Strategy_Return"] = bt["Signal"].shift(1).fillna(0) * bt["Return_1D"]
    bt["Strategy_Return"] = bt["Strategy_Return"] - trade * cost_rate
    return bt


def position_from_entry_exit(entry: pd.Series, exit_: pd.Series, index: pd.Index) -> pd.Series:
    holding = False
    signal = []
    entry = entry.reindex(index).fillna(False)
    exit_ = exit_.reindex(index).fillna(False)
    for enter, leave in zip(entry, exit_):
        if holding and leave:
            holding = False
        if (not holding) and enter:
            holding = True
        signal.append(1 if holding else 0)
    return pd.Series(signal, index=index, dtype=float)


def backtest_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    params: Optional[Dict] = None,
    execution_mode: str = "理想",
    cost_rate: float = 0.0,
) -> Dict:
    params = params or {}
    data = df.copy()
    min_cols = ["Return_1D", "MA20", "MA60", "RSI14", "EMA10", "EMA20", "BB_LOWER", "BB_MID", "Volume_Ratio"]
    data = data.dropna(subset=[c for c in min_cols if c in data.columns]).copy()
    if len(data) < 60:
        return {}
    if strategy_name == "保守均線趨勢｜少交易":
        short, long = int(params.get("short_ma", 20)), int(params.get("long_ma", 60))
        for n in [short, long]:
            if f"MA{n}" not in data.columns:
                data[f"MA{n}"] = data["Close"].rolling(n).mean()
        data = data.dropna(subset=[f"MA{short}", f"MA{long}"])
        signal = ((data["Close"] > data[f"MA{short}"]) & (data[f"MA{short}"] > data[f"MA{long}"])).astype(float)
    elif strategy_name == "長線大波段｜不太操作":
        short, long = int(params.get("short_ma", 60)), int(params.get("long_ma", 120))
        for n in [short, long]:
            if f"MA{n}" not in data.columns:
                data[f"MA{n}"] = data["Close"].rolling(n).mean()
        data = data.dropna(subset=[f"MA{short}", f"MA{long}"])
        signal = ((data["Close"] > data[f"MA{short}"]) & (data[f"MA{short}"] > data[f"MA{long}"])).astype(float)
    elif strategy_name == "EMA動能｜短線波段":
        fast, slow = int(params.get("fast_ema", 10)), int(params.get("slow_ema", 20))
        rsi_enter, rsi_exit = float(params.get("rsi_enter", 50)), float(params.get("rsi_exit", 45))
        data[f"EMA{fast}"] = data["Close"].ewm(span=fast, adjust=False).mean()
        data[f"EMA{slow}"] = data["Close"].ewm(span=slow, adjust=False).mean()
        entry = (data[f"EMA{fast}"] > data[f"EMA{slow}"]) & (data["RSI14"] > rsi_enter)
        exit_ = (data[f"EMA{fast}"] < data[f"EMA{slow}"]) | (data["RSI14"] < rsi_exit)
        signal = position_from_entry_exit(entry, exit_, data.index)
    elif strategy_name == "突破追價｜不要錯過飆股":
        lookback, exit_ma = int(params.get("breakout_n", 20)), int(params.get("exit_ma", 20))
        volume_min = float(params.get("volume_min", 1.2))
        data["Breakout_High"] = data["Close"].rolling(lookback).max().shift(1)
        data[f"MA{exit_ma}"] = data["Close"].rolling(exit_ma).mean()
        data["Exit_Low"] = data["Close"].rolling(10).min().shift(1)
        data = data.dropna(subset=["Breakout_High", f"MA{exit_ma}", "Exit_Low", "Volume_Ratio"])
        entry = (data["Close"] > data["Breakout_High"]) & (data["Volume_Ratio"] >= volume_min)
        exit_ = (data["Close"] < data[f"MA{exit_ma}"]) | (data["Close"] < data["Exit_Low"])
        signal = position_from_entry_exit(entry, exit_, data.index)
    elif strategy_name == "RSI反轉｜頻繁操作搶反彈":
        entry = data["RSI14"] < float(params.get("rsi_low", 30))
        exit_ = (data["RSI14"] > float(params.get("rsi_high", 55))) | (data["Close"] > data["MA20"])
        signal = position_from_entry_exit(entry, exit_, data.index)
    elif strategy_name == "布林下軌反彈｜有賺就好":
        signal = position_from_entry_exit(data["Close"] < data["BB_LOWER"], data["Close"] >= data["BB_MID"], data.index)
    else:
        return {}
    return compute_backtest_stats(apply_backtest_execution(data, signal, mode=execution_mode, cost_rate=cost_rate))


def backtest_all_strategies(df: pd.DataFrame, execution_mode: str = "理想", cost_rate: float = 0.0) -> pd.DataFrame:
    rows = []
    for name, meta in STRATEGY_PRESETS.items():
        bt = backtest_strategy(df, name, execution_mode=execution_mode, cost_rate=cost_rate)
        if bt:
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


def _opt_stats(bt: Dict) -> Dict:
    return {
        "策略總報酬%": round(bt["total_return"] * 100, 2),
        "年化報酬%": round(bt["cagr"] * 100, 2),
        "最大回撤%": round(bt["max_dd"] * 100, 2),
        "Sharpe": round(bt["sharpe"], 2) if not pd.isna(bt["sharpe"]) else np.nan,
        "持股時間%": round(bt["exposure"] * 100, 2),
        "交易次數": bt["trades"],
    }


def optimize_parameters(
    df: pd.DataFrame,
    strategy_family: str = "全部策略",
    execution_mode: str = "理想",
    cost_rate: float = 0.0,
    optimize_target: str = "穩健分數最高",
) -> pd.DataFrame:
    rows = []
    families = ["均線趨勢 MA", "長線大波段 MA", "EMA動能", "突破追價", "RSI反轉", "布林反彈"]
    if strategy_family == "全部策略":
        for family in families:
            part = optimize_parameters(df, family, execution_mode, cost_rate, optimize_target)
            if not part.empty:
                rows.extend(part.to_dict("records"))
    elif strategy_family == "均線趨勢 MA":
        for short in [5, 10, 20, 30, 60]:
            for long in [20, 60, 90, 120, 240]:
                if short >= long:
                    continue
                name = "保守均線趨勢｜少交易" if long <= 90 else "長線大波段｜不太操作"
                bt = backtest_strategy(df, name, {"short_ma": short, "long_ma": long}, execution_mode, cost_rate)
                if bt:
                    rows.append({"策略族": strategy_family, "參數": f"Close > MA{short} 且 MA{short} > MA{long}", "短均線": short, "長均線": long, **_opt_stats(bt)})
    elif strategy_family == "長線大波段 MA":
        for short in [40, 60, 90, 120]:
            for long in [120, 180, 240]:
                if short >= long:
                    continue
                bt = backtest_strategy(df, "長線大波段｜不太操作", {"short_ma": short, "long_ma": long}, execution_mode, cost_rate)
                if bt:
                    rows.append({"策略族": strategy_family, "參數": f"Close > MA{short} 且 MA{short} > MA{long}", "短均線": short, "長均線": long, **_opt_stats(bt)})
    elif strategy_family == "EMA動能":
        for fast in [5, 8, 10, 12, 15]:
            for slow in [20, 30, 50, 60]:
                if fast >= slow:
                    continue
                for rsi_enter in [45, 50, 55]:
                    bt = backtest_strategy(df, "EMA動能｜短線波段", {"fast_ema": fast, "slow_ema": slow, "rsi_enter": rsi_enter, "rsi_exit": rsi_enter - 5}, execution_mode, cost_rate)
                    if bt:
                        rows.append({"策略族": strategy_family, "參數": f"EMA{fast} > EMA{slow}, RSI>{rsi_enter}", "快EMA": fast, "慢EMA": slow, "RSI進場": rsi_enter, **_opt_stats(bt)})
    elif strategy_family == "突破追價":
        for n in [10, 20, 30, 55]:
            for exit_ma in [10, 20, 30]:
                for volume_min in [1.0, 1.2, 1.5]:
                    bt = backtest_strategy(df, "突破追價｜不要錯過飆股", {"breakout_n": n, "exit_ma": exit_ma, "volume_min": volume_min}, execution_mode, cost_rate)
                    if bt:
                        rows.append({"策略族": strategy_family, "參數": f"突破{n}日高, 量比>{volume_min}, 跌破MA{exit_ma}出場", "突破天數": n, "出場MA": exit_ma, "量比門檻": volume_min, **_opt_stats(bt)})
    elif strategy_family == "RSI反轉":
        for low in [20, 25, 30, 35]:
            for high in [50, 55, 60, 65]:
                if low >= high:
                    continue
                bt = backtest_strategy(df, "RSI反轉｜頻繁操作搶反彈", {"rsi_low": low, "rsi_high": high}, execution_mode, cost_rate)
                if bt:
                    rows.append({"策略族": strategy_family, "參數": f"RSI<{low}進場, RSI>{high}或站上MA20出場", "RSI低檔": low, "RSI出場": high, **_opt_stats(bt)})
    elif strategy_family == "布林反彈":
        for k in [1.5, 2.0, 2.5, 3.0]:
            tmp = df.copy()
            tmp["BB_UPPER"] = tmp["BB_MID"] + k * tmp["BB_STD"]
            tmp["BB_LOWER"] = tmp["BB_MID"] - k * tmp["BB_STD"]
            bt = backtest_strategy(tmp, "布林下軌反彈｜有賺就好", execution_mode=execution_mode, cost_rate=cost_rate)
            if bt:
                rows.append({"策略族": strategy_family, "參數": f"Close < BB下軌({k}σ)進場, 回中線出場", "布林倍數": k, **_opt_stats(bt)})

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["穩健分數"] = out["Sharpe"].fillna(0) * 2 + out["年化報酬%"].fillna(0) / 20 + out["最大回撤%"].fillna(-100) / 20
    sort_col = "策略總報酬%" if optimize_target == "報酬最高" else "穩健分數"
    return out.sort_values(sort_col, ascending=False).head(20)


def strategy_execution_advice(df: pd.DataFrame, strategy_name: str, params: Optional[Dict] = None) -> Dict:
    params = params or {}
    data = df.copy()
    last = data.iloc[-1]
    close = float(last["Close"])
    status = "等待進場"
    action = "目前尚未符合進場條件。"
    entry = "—"
    exit_rule = "—"
    key_level = np.nan
    reasons = []

    if strategy_name == "保守均線趨勢｜少交易":
        short, long = int(params.get("short_ma", 20)), int(params.get("long_ma", 60))
        for n in [short, long]:
            if f"MA{n}" not in data.columns:
                data[f"MA{n}"] = data["Close"].rolling(n).mean()
        last = data.iloc[-1]
        ma_s, ma_l = float(last[f"MA{short}"]), float(last[f"MA{long}"])
        entry = f"收盤價 > MA{short}，且 MA{short} > MA{long}"
        exit_rule = f"收盤價跌破 MA{short}，或 MA{short} 轉弱低於 MA{long}"
        key_level = ma_s
        if close > ma_s > ma_l:
            status = "可持有"
            action = f"目前已符合策略條件。若尚未進場，可等下一交易日確認未跌破 MA{short} 再考慮進場。"
        else:
            action = f"目前尚未符合條件，可等待收盤價重新站上 MA{short}，且 MA{short} 高於 MA{long}。"
        reasons.append(f"現價 {close:.2f}，MA{short}={ma_s:.2f}，MA{long}={ma_l:.2f}")
    elif strategy_name == "長線大波段｜不太操作":
        short, long = int(params.get("short_ma", 60)), int(params.get("long_ma", 120))
        for n in [short, long]:
            if f"MA{n}" not in data.columns:
                data[f"MA{n}"] = data["Close"].rolling(n).mean()
        last = data.iloc[-1]
        ma_s, ma_l = float(last[f"MA{short}"]), float(last[f"MA{long}"])
        entry = f"收盤價 > MA{short}，且 MA{short} > MA{long}"
        exit_rule = f"收盤價跌破 MA{short}，或 MA{short} 跌破 MA{long}"
        key_level = ma_s
        if close > ma_s > ma_l:
            status = "可持有"
            action = f"目前符合長線波段條件，可視為持有區；若尚未進場，建議等回測 MA{short} 不破或隔日續強再進。"
        else:
            action = f"目前尚未符合長線條件，可等待收盤價站回 MA{short} 且 MA{short} 高於 MA{long}。"
        reasons.append(f"現價 {close:.2f}，MA{short}={ma_s:.2f}，MA{long}={ma_l:.2f}")
    elif strategy_name == "EMA動能｜短線波段":
        fast, slow = int(params.get("fast_ema", 10)), int(params.get("slow_ema", 20))
        rsi_enter, rsi_exit = float(params.get("rsi_enter", 50)), float(params.get("rsi_exit", 45))
        data[f"EMA{fast}"] = data["Close"].ewm(span=fast, adjust=False).mean()
        data[f"EMA{slow}"] = data["Close"].ewm(span=slow, adjust=False).mean()
        last = data.iloc[-1]
        ema_f, ema_s, rsi = float(last[f"EMA{fast}"]), float(last[f"EMA{slow}"]), float(last["RSI14"])
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
        lookback, exit_ma = int(params.get("breakout_n", 20)), int(params.get("exit_ma", 20))
        volume_min = float(params.get("volume_min", 1.2))
        breakout_high = data["Close"].rolling(lookback).max().shift(1).iloc[-1]
        exit_low = data["Close"].rolling(10).min().shift(1).iloc[-1]
        ma_exit = data["Close"].rolling(exit_ma).mean().iloc[-1]
        vol_ratio = float(last["Volume_Ratio"])
        entry = f"收盤價突破近 {lookback} 日高點，且量比 > {volume_min}"
        exit_rule = f"跌破 MA{exit_ma}，或跌破近 10 日低點"
        key_level = breakout_high
        if close > breakout_high and vol_ratio >= volume_min:
            status = "可進場"
            action = "目前已符合突破進場條件。若要執行，應特別設定停損，避免假突破。"
        else:
            action = f"目前尚未突破。可等待收盤價突破 {breakout_high:.2f}，且量比大於 {volume_min}。"
        reasons.extend([f"現價 {close:.2f}，突破價 {breakout_high:.2f}，量比 {vol_ratio:.2f}", f"出場參考：MA{exit_ma}={ma_exit:.2f}，10日低點={exit_low:.2f}"])
    elif strategy_name == "RSI反轉｜頻繁操作搶反彈":
        rsi_low, rsi_high = float(params.get("rsi_low", 30)), float(params.get("rsi_high", 55))
        rsi, ma20 = float(last["RSI14"]), float(last["MA20"])
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
        bb_lower, bb_mid = float(last["BB_LOWER"]), float(last["BB_MID"])
        entry = "收盤價跌破布林下軌"
        exit_rule = "收盤價回到布林中線"
        key_level = bb_lower
        if close < bb_lower:
            status = "觀察反彈"
            action = "目前已跌破布林下軌，符合均值回歸進場條件，但需避免接到趨勢下跌。"
        else:
            action = f"目前尚未跌破布林下軌，可等待價格接近或跌破 {bb_lower:.2f}。"
        reasons.append(f"現價 {close:.2f}，布林下軌={bb_lower:.2f}，布林中線={bb_mid:.2f}")

    distance_pct = (close / key_level - 1) * 100 if not pd.isna(key_level) and key_level > 0 else np.nan
    return {"status": status, "action": action, "entry": entry, "exit_rule": exit_rule, "key_level": key_level, "distance_pct": distance_pct, "reasons": reasons}
