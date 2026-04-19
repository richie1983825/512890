from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def resolve_data_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    primary = root / "data" / "512890.SH.parquet"
    fallback = root / "512890.SH.parquet"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    candidates = sorted(root.rglob("512890.SH.parquet"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("未找到 512890.SH.parquet，请放在 data/ 目录")


def load_and_forward_adjust(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    required_cols = {"trade_date", "open", "high", "low", "close", "volume", "adj_factor"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {sorted(missing)}")

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("trade_date").set_index("trade_date")

    latest_adj = df["adj_factor"].iloc[-1]
    if latest_adj == 0:
        raise ValueError("最新 adj_factor 为 0，无法进行前向复权")

    scale = df["adj_factor"] / latest_adj
    out = pd.DataFrame(index=df.index)
    out["Open"] = df["open"] * scale
    out["High"] = df["high"] * scale
    out["Low"] = df["low"] * scale
    out["Close"] = df["close"] * scale
    out["Volume"] = df["volume"]
    return out.dropna()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_strategy_features(
    df: pd.DataFrame,
    fit_window_days: int,
    trend_window_days: int,
    vol_window_days: int,
) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"].to_numpy(dtype=float)
    n = len(data)
    if n < 5:
        return data.iloc[0:0].copy()

    trend_min_periods = max(3, int(trend_window_days) // 2)
    max_fit_window = max(5, n - trend_min_periods - 1)
    effective_fit_window = min(max(int(fit_window_days), 5), max_fit_window)

    pred = np.full(n, np.nan, dtype=float)
    slope = np.full(n, np.nan, dtype=float)

    x = np.arange(effective_fit_window, dtype=float)
    x_mean = float(x.mean())
    x_center = x - x_mean
    x_var = float((x_center**2).sum())

    for i in range(effective_fit_window - 1, n):
        y = close[i - effective_fit_window + 1 : i + 1]
        if np.isnan(y).any():
            continue
        y_mean = float(y.mean())
        beta = float(np.dot(x_center, y - y_mean) / max(x_var, 1e-12))
        alpha = y_mean - beta * x_mean
        pred[i] = alpha + beta * (effective_fit_window - 1)
        slope[i] = beta

    data["PolyBasePred"] = pred
    data["PolySlope"] = slope
    data["PolyDevPct"] = data["Close"] / data["PolyBasePred"] - 1.0
    data["PolyDevTrend"] = data["PolyDevPct"].diff().ewm(
        span=max(3, int(trend_window_days)),
        adjust=False,
        min_periods=max(3, int(trend_window_days) // 2),
    ).mean()
    data["RollingVolPct"] = data["Close"].pct_change().rolling(
        window=max(2, int(vol_window_days)),
        min_periods=max(2, int(vol_window_days)),
    ).std(ddof=0)

    feature_cols = ["PolyBasePred", "PolySlope", "PolyDevPct", "PolyDevTrend", "RollingVolPct"]
    return data.dropna(subset=feature_cols)


def add_ma_strategy_features(
    df: pd.DataFrame,
    ma_window_days: int,
    trend_window_days: int,
    vol_window_days: int,
) -> pd.DataFrame:
    data = df.copy()
    ma_window = max(2, int(ma_window_days))

    data["MABase"] = data["Close"].rolling(window=ma_window, min_periods=ma_window).mean()
    data["MADevPct"] = data["Close"] / data["MABase"] - 1.0
    data["MADevTrend"] = data["MADevPct"].diff().ewm(
        span=max(3, int(trend_window_days)),
        adjust=False,
        min_periods=max(3, int(trend_window_days) // 2),
    ).mean()
    data["RollingVolPct"] = data["Close"].pct_change().rolling(
        window=max(2, int(vol_window_days)),
        min_periods=max(2, int(vol_window_days)),
    ).std(ddof=0)

    feature_cols = ["MABase", "MADevPct", "MADevTrend", "RollingVolPct"]
    return data.dropna(subset=feature_cols)


def combine_train_val_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_df, val_df]).sort_index()
    return combined.loc[~combined.index.duplicated(keep="last")]