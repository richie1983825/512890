from __future__ import annotations

from itertools import product
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from backtesting import Backtest

from strategies.breakout_retest_momentum_strategy import BreakoutRetestMomentumStrategy
from strategies.event_driven_launch_strategy import EventDrivenLaunchStrategy
from strategies.launch_breakout_momentum_strategy import LaunchBreakoutMomentumStrategy
from strategies.moving_average_dynamic_grid_strategy import MovingAverageDynamicGridStrategy
from strategies.polyfit_dynamic_grid_strategy import PolyfitDynamicGridStrategy


POLYFIT_SCAN_PARAM_NAMES = [
    "fit_window_days",
    "trend_window_days",
    "vol_window_days",
    "base_grid_pct",
    "volatility_scale",
    "trend_sensitivity",
    "max_grid_levels",
    "take_profit_grid",
    "stop_loss_grid",
    "max_holding_days",
    "cooldown_days",
    "min_signal_strength",
    "position_size",
    "position_sizing_coef",
]

MA_SCAN_PARAM_NAMES = [
    "ma_window_days",
    "trend_window_days",
    "vol_window_days",
    "base_grid_pct",
    "volatility_scale",
    "trend_sensitivity",
    "max_grid_levels",
    "take_profit_grid",
    "stop_loss_grid",
    "max_holding_days",
    "cooldown_days",
    "min_signal_strength",
    "position_size",
    "position_sizing_coef",
]

LAUNCH_SCAN_PARAM_NAMES = [
    "fast_ema_window",
    "slow_ema_window",
    "trend_slope_window",
    "breakout_window",
    "exit_window",
    "atr_window",
    "volume_window",
    "short_vol_window",
    "long_vol_window",
    "breakout_buffer_atr",
    "min_trend_slope",
    "min_breakout_return",
    "min_close_strength",
    "min_volume_ratio",
    "max_compression_ratio",
    "min_range_to_atr",
    "min_continuation_return",
    "min_continuation_volume_ratio",
    "initial_stop_atr_mult",
    "trailing_atr_mult",
    "max_holding_days",
    "position_size",
]

RETEST_SCAN_PARAM_NAMES = [
    "fast_ema_window",
    "slow_ema_window",
    "trend_slope_window",
    "breakout_window",
    "exit_window",
    "atr_window",
    "volume_window",
    "short_vol_window",
    "long_vol_window",
    "breakout_buffer_atr",
    "retest_tolerance_atr",
    "retest_window_days",
    "min_trend_slope",
    "min_breakout_volume_ratio",
    "min_retest_volume_ratio",
    "max_setup_compression_ratio",
    "min_breakout_range_to_atr",
    "max_extension_pct",
    "initial_stop_atr_mult",
    "trailing_atr_mult",
    "max_holding_days",
    "position_size",
]

EVENT_SCAN_PARAM_NAMES = [
    "fast_ema_window",
    "slow_ema_window",
    "trend_slope_window",
    "breakout_window",
    "exit_window",
    "atr_window",
    "volume_window",
    "short_vol_window",
    "long_vol_window",
    "min_gap_return",
    "min_event_return",
    "min_close_strength",
    "min_volume_ratio",
    "max_compression_ratio",
    "min_range_to_atr",
    "min_trend_slope",
    "max_extension_pct",
    "initial_stop_atr_mult",
    "trailing_atr_mult",
    "max_holding_days",
    "position_size",
]

LAUNCH_TARGET_PERIODS: list[tuple[str, str]] = [
    ("2024-03-01", "2024-03-31"),
    ("2024-09-01", "2024-09-30"),
    ("2025-04-01", "2025-04-30"),
    ("2025-10-01", "2025-10-31"),
]


def build_param_space() -> dict[str, list]:
    return {
        "fit_window_days": [630, 756, 882],
        "trend_window_days": [10, 15, 20],
        "vol_window_days": [5, 10, 15, 20, 30],
        "base_grid_pct": [0.008, 0.010, 0.012, 0.015],
        "volatility_scale": [0.0, 0.5, 1.0, 1.5, 2.0],
        "trend_sensitivity": [4.0, 6.0, 8.0, 10.0],
        "max_grid_levels": [2, 3, 4],
        "take_profit_grid": [0.6, 0.8, 1.0],
        "stop_loss_grid": [1.2, 1.6, 2.0],
        "max_holding_days": [15, 25, 35],
        "cooldown_days": [0, 1, 2],
        "min_signal_strength": [0.30, 0.45, 0.60],
        "position_size": [i / 100 for i in range(0, 101)],
        "position_sizing_coef": [10.0, 20.0, 30.0, 40.0, 60.0],
    }


def build_ma_param_space() -> dict[str, list]:
    return {
        "ma_window_days": [5, 10, 30, 60],
        "trend_window_days": [5, 10, 15, 20],
        "vol_window_days": [5, 10, 15, 20, 30],
        "base_grid_pct": [0.008, 0.010, 0.012, 0.015],
        "volatility_scale": [0.0, 0.5, 1.0, 1.5, 2.0],
        "trend_sensitivity": [4.0, 6.0, 8.0, 10.0],
        "max_grid_levels": [2, 3, 4],
        "take_profit_grid": [0.6, 0.8, 1.0],
        "stop_loss_grid": [1.2, 1.6, 2.0],
        "max_holding_days": [15, 25, 35],
        "cooldown_days": [0, 1, 2],
        "min_signal_strength": [0.30, 0.45, 0.60],
        "position_size": [i / 100 for i in range(0, 101)],
        "position_sizing_coef": [10.0, 20.0, 30.0, 40.0, 60.0],
    }


def build_launch_param_space() -> dict[str, list]:
    return {
        "fast_ema_window": [6, 10, 15, 20],
        "slow_ema_window": [20, 30, 45, 60],
        "trend_slope_window": [5, 10, 20, 30],
        "breakout_window": [8, 12, 20, 30, 40],
        "exit_window": [5, 10, 15],
        "atr_window": [10, 14, 20],
        "volume_window": [5, 10, 20],
        "short_vol_window": [5, 10],
        "long_vol_window": [20, 40],
        "breakout_buffer_atr": [-0.10, 0.0, 0.10, 0.20],
        "min_trend_slope": [-0.0010, -0.0005, 0.0, 0.0005],
        "min_breakout_return": [0.0, 0.003, 0.006, 0.010],
        "min_close_strength": [0.35, 0.50, 0.60, 0.70],
        "min_volume_ratio": [0.5, 0.7, 0.9, 1.1],
        "max_compression_ratio": [0.75, 0.9, 1.05, 1.2],
        "min_range_to_atr": [0.4, 0.6, 0.8, 1.0],
        "min_continuation_return": [0.0, 0.001, 0.002, 0.004],
        "min_continuation_volume_ratio": [0.5, 0.7, 0.9, 1.1],
        "initial_stop_atr_mult": [1.2, 1.5, 1.8, 2.2],
        "trailing_atr_mult": [1.8, 2.2, 2.6, 3.0],
        "max_holding_days": [15, 25, 35],
        "position_size": [0.3, 0.5, 0.7, 0.9],
    }


def build_retest_param_space() -> dict[str, list]:
    return {
        "fast_ema_window": [10, 15, 20],
        "slow_ema_window": [30, 45, 60],
        "trend_slope_window": [10, 20, 30],
        "breakout_window": [20, 30, 40, 60],
        "exit_window": [5, 10, 15],
        "atr_window": [10, 14, 20],
        "volume_window": [5, 10, 20],
        "short_vol_window": [5, 10],
        "long_vol_window": [20, 40],
        "breakout_buffer_atr": [0.0, 0.1, 0.15, 0.25],
        "retest_tolerance_atr": [0.3, 0.6, 0.9],
        "retest_window_days": [3, 5, 8, 12],
        "min_trend_slope": [0.0, 0.0005, 0.0010],
        "min_breakout_volume_ratio": [0.8, 1.0, 1.2, 1.5],
        "min_retest_volume_ratio": [0.6, 0.8, 1.0],
        "max_setup_compression_ratio": [0.75, 0.9, 1.05, 1.2],
        "min_breakout_range_to_atr": [0.6, 0.8, 1.0],
        "max_extension_pct": [0.03, 0.05, 0.08],
        "initial_stop_atr_mult": [1.2, 1.5, 1.8],
        "trailing_atr_mult": [1.8, 2.2, 2.6],
        "max_holding_days": [15, 25, 35],
        "position_size": [0.3, 0.5, 0.7, 0.9],
    }


def build_event_param_space() -> dict[str, list]:
    return {
        "fast_ema_window": [10, 15, 20],
        "slow_ema_window": [30, 45, 60],
        "trend_slope_window": [10, 20, 30],
        "breakout_window": [10, 20, 30, 40],
        "exit_window": [5, 10, 15],
        "atr_window": [10, 14, 20],
        "volume_window": [5, 10, 20],
        "short_vol_window": [5, 10],
        "long_vol_window": [20, 40],
        "min_gap_return": [0.0, 0.003, 0.007],
        "min_event_return": [0.005, 0.008, 0.012, 0.015],
        "min_close_strength": [0.45, 0.55, 0.65],
        "min_volume_ratio": [0.8, 1.0, 1.3, 1.6],
        "max_compression_ratio": [0.7, 0.85, 1.0, 1.2],
        "min_range_to_atr": [0.6, 0.8, 1.0, 1.2],
        "min_trend_slope": [-0.001, 0.0, 0.0005],
        "max_extension_pct": [0.05, 0.08, 0.12],
        "initial_stop_atr_mult": [1.0, 1.4, 1.8],
        "trailing_atr_mult": [1.8, 2.2, 2.6],
        "max_holding_days": [10, 15, 20, 25],
        "position_size": [0.3, 0.5, 0.7, 0.9],
    }


def build_optimized_params() -> dict[str, float | int]:
    return {
        "fit_window_days": 756,
        "trend_window_days": 15,
        "vol_window_days": 15,
        "base_grid_pct": 0.012,
        "volatility_scale": 1.0,
        "trend_sensitivity": 8.0,
        "max_grid_levels": 3,
        "take_profit_grid": 0.8,
        "stop_loss_grid": 1.6,
        "max_holding_days": 25,
        "cooldown_days": 1,
        "min_signal_strength": 0.45,
        "position_size": 0.5,
        "position_sizing_coef": 30.0,
    }


def build_ma_optimized_params() -> dict[str, float | int]:
    return {
        "ma_window_days": 30,
        "trend_window_days": 10,
        "vol_window_days": 15,
        "base_grid_pct": 0.012,
        "volatility_scale": 1.0,
        "trend_sensitivity": 8.0,
        "max_grid_levels": 3,
        "take_profit_grid": 0.8,
        "stop_loss_grid": 1.6,
        "max_holding_days": 25,
        "cooldown_days": 1,
        "min_signal_strength": 0.45,
        "position_size": 0.5,
        "position_sizing_coef": 30.0,
    }


def build_launch_optimized_params() -> dict[str, float | int]:
    return {
        "fast_ema_window": 15,
        "slow_ema_window": 45,
        "trend_slope_window": 20,
        "breakout_window": 30,
        "exit_window": 10,
        "atr_window": 14,
        "volume_window": 10,
        "short_vol_window": 5,
        "long_vol_window": 20,
        "breakout_buffer_atr": 0.15,
        "min_trend_slope": 0.0005,
        "min_breakout_return": 0.012,
        "min_close_strength": 0.6,
        "min_volume_ratio": 1.2,
        "max_compression_ratio": 0.85,
        "min_range_to_atr": 1.1,
        "min_continuation_return": 0.002,
        "min_continuation_volume_ratio": 0.8,
        "initial_stop_atr_mult": 1.5,
        "trailing_atr_mult": 2.2,
        "max_holding_days": 25,
        "position_size": 0.5,
    }


def build_retest_optimized_params() -> dict[str, float | int]:
    return {
        "fast_ema_window": 15,
        "slow_ema_window": 45,
        "trend_slope_window": 20,
        "breakout_window": 30,
        "exit_window": 10,
        "atr_window": 14,
        "volume_window": 10,
        "short_vol_window": 5,
        "long_vol_window": 20,
        "breakout_buffer_atr": 0.1,
        "retest_tolerance_atr": 0.6,
        "retest_window_days": 8,
        "min_trend_slope": 0.0005,
        "min_breakout_volume_ratio": 1.2,
        "min_retest_volume_ratio": 0.9,
        "max_setup_compression_ratio": 0.85,
        "min_breakout_range_to_atr": 1.0,
        "max_extension_pct": 0.05,
        "initial_stop_atr_mult": 1.5,
        "trailing_atr_mult": 2.2,
        "max_holding_days": 25,
        "position_size": 0.5,
    }


def build_event_optimized_params() -> dict[str, float | int]:
    return {
        "fast_ema_window": 15,
        "slow_ema_window": 45,
        "trend_slope_window": 20,
        "breakout_window": 20,
        "exit_window": 10,
        "atr_window": 14,
        "volume_window": 10,
        "short_vol_window": 5,
        "long_vol_window": 20,
        "min_gap_return": 0.01,
        "min_event_return": 0.015,
        "min_close_strength": 0.65,
        "min_volume_ratio": 1.5,
        "max_compression_ratio": 0.7,
        "min_range_to_atr": 1.2,
        "min_trend_slope": -0.001,
        "max_extension_pct": 0.08,
        "initial_stop_atr_mult": 1.4,
        "trailing_atr_mult": 2.0,
        "max_holding_days": 20,
        "position_size": 0.5,
    }


def _valid_param_set(params: dict) -> bool:
    return (
        params["fit_window_days"] >= 252
        and params["trend_window_days"] >= 5
        and params["vol_window_days"] >= 2
        and params["base_grid_pct"] > 0
        and params["volatility_scale"] >= 0
        and params["trend_sensitivity"] >= 0
        and params["max_grid_levels"] >= 1
        and params["take_profit_grid"] > 0
        and params["stop_loss_grid"] >= params["take_profit_grid"]
        and params["max_holding_days"] >= 5
        and params["min_signal_strength"] > 0
        and 0 <= params["position_size"] <= 1
        and params["position_sizing_coef"] > 0
    )


def _valid_ma_param_set(params: dict) -> bool:
    return (
        params["ma_window_days"] >= 2
        and params["trend_window_days"] >= 3
        and params["vol_window_days"] >= 2
        and params["base_grid_pct"] > 0
        and params["volatility_scale"] >= 0
        and params["trend_sensitivity"] >= 0
        and params["max_grid_levels"] >= 1
        and params["take_profit_grid"] > 0
        and params["stop_loss_grid"] > 0
        and params["max_holding_days"] >= 5
        and params["cooldown_days"] >= 0
        and params["min_signal_strength"] >= 0
        and 0 <= params["position_size"] <= 1
        and params["position_sizing_coef"] > 0
    )


def _valid_launch_param_set(params: dict) -> bool:
    return (
        params["slow_ema_window"] > params["fast_ema_window"]
        and params["long_vol_window"] > params["short_vol_window"]
        and params["breakout_window"] >= params["fast_ema_window"]
        and params["breakout_window"] > params["exit_window"]
        and params["exit_window"] >= 3
        and params["trend_slope_window"] >= 3
        and params["atr_window"] >= 2
        and params["volume_window"] >= 2
        and params["breakout_buffer_atr"] >= 0
        and params["min_trend_slope"] >= 0
        and params["min_breakout_return"] > 0
        and 0 < params["min_close_strength"] <= 1
        and params["min_volume_ratio"] > 0
        and params["max_compression_ratio"] > 0
        and params["min_range_to_atr"] > 0
        and params["min_continuation_return"] >= 0
        and params["min_continuation_volume_ratio"] > 0
        and params["initial_stop_atr_mult"] > 0
        and params["trailing_atr_mult"] > 0
        and params["max_holding_days"] >= 5
        and 0 <= params["position_size"] <= 1
    )


def _valid_retest_param_set(params: dict) -> bool:
    return (
        params["slow_ema_window"] > params["fast_ema_window"]
        and params["long_vol_window"] > params["short_vol_window"]
        and params["breakout_window"] >= params["fast_ema_window"]
        and params["breakout_window"] > params["exit_window"]
        and params["exit_window"] >= 3
        and params["trend_slope_window"] >= 3
        and params["atr_window"] >= 2
        and params["volume_window"] >= 2
        and params["breakout_buffer_atr"] >= 0
        and params["retest_tolerance_atr"] >= 0
        and params["retest_window_days"] >= 2
        and params["min_trend_slope"] >= 0
        and params["min_breakout_volume_ratio"] > 0
        and params["min_retest_volume_ratio"] > 0
        and params["max_setup_compression_ratio"] > 0
        and params["min_breakout_range_to_atr"] > 0
        and params["max_extension_pct"] > 0
        and params["initial_stop_atr_mult"] > 0
        and params["trailing_atr_mult"] > 0
        and params["max_holding_days"] >= 5
        and 0 <= params["position_size"] <= 1
    )


def _valid_event_param_set(params: dict) -> bool:
    return (
        params["slow_ema_window"] > params["fast_ema_window"]
        and params["long_vol_window"] > params["short_vol_window"]
        and params["breakout_window"] >= params["fast_ema_window"]
        and params["breakout_window"] > params["exit_window"]
        and params["exit_window"] >= 3
        and params["trend_slope_window"] >= 3
        and params["atr_window"] >= 2
        and params["volume_window"] >= 2
        and params["min_gap_return"] >= 0
        and params["min_event_return"] > 0
        and 0 < params["min_close_strength"] <= 1
        and params["min_volume_ratio"] > 0
        and params["max_compression_ratio"] > 0
        and params["min_range_to_atr"] > 0
        and params["max_extension_pct"] > 0
        and params["initial_stop_atr_mult"] > 0
        and params["trailing_atr_mult"] > 0
        and params["max_holding_days"] >= 5
        and 0 <= params["position_size"] <= 1
    )


def _generate_param_combinations(space: dict[str, list]) -> list[dict]:
    keys = list(space.keys())
    combos = []
    for values in product(*(space[k] for k in keys)):
        item = dict(zip(keys, values, strict=False))
        if _valid_param_set(item):
            combos.append(item)
    return combos


def _sample_param_combinations(space: dict[str, list], max_evals: int, random_seed: int) -> list[dict]:
    keys = list(space.keys())
    rng = np.random.default_rng(random_seed)
    seen: set[tuple] = set()
    combos: list[dict] = []
    max_unique = 1
    for key in keys:
        max_unique *= len(space[key])
    target = min(max_evals, max_unique)
    attempts = 0
    max_attempts = max(target * 50, 1000)

    while len(combos) < target and attempts < max_attempts:
        attempts += 1
        item = {key: rng.choice(space[key]).item() for key in keys}
        if not _valid_param_set(item):
            continue
        key_tuple = tuple(item[key] for key in keys)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        combos.append(item)

    if not combos:
        raise ValueError("未采样到有效参数组合")
    return combos


def _sample_param_combinations_with_validator(
    space: dict[str, list],
    max_evals: int,
    random_seed: int,
    validator,
) -> list[dict]:
    keys = list(space.keys())
    rng = np.random.default_rng(random_seed)
    seen: set[tuple] = set()
    combos: list[dict] = []
    max_unique = 1
    for key in keys:
        max_unique *= len(space[key])
    target = min(max_evals, max_unique)
    attempts = 0
    max_attempts = max(target * 50, 1000)

    while len(combos) < target and attempts < max_attempts:
        attempts += 1
        item = {key: rng.choice(space[key]).item() for key in keys}
        if not validator(item):
            continue
        key_tuple = tuple(item[key] for key in keys)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        combos.append(item)

    if not combos:
        raise ValueError("未采样到有效参数组合")
    return combos


def configure_chinese_font() -> None:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def resolve_data_path() -> Path:
    root = Path(__file__).resolve().parent
    p1 = root / "data" / "512890.SH.parquet"
    p2 = root / "512890.SH.parquet"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
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


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
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
    if n < 252:
        return data.iloc[0:0].copy()

    effective_fit_window = min(max(int(fit_window_days), 252), n)

    pred = np.full(n, np.nan, dtype=float)
    slope = np.full(n, np.nan, dtype=float)

    x = np.arange(effective_fit_window, dtype=float)
    x_mean = float(x.mean())
    x_center = x - x_mean
    x_var = float((x_center ** 2).sum())

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

    data["MABase"] = data["Close"].rolling(
        window=ma_window,
        min_periods=ma_window,
    ).mean()
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


def add_momentum_features(
    df: pd.DataFrame,
    fast_ema_window: int,
    slow_ema_window: int,
    trend_slope_window: int,
    breakout_window: int,
    exit_window: int,
    atr_window: int,
    volume_window: int,
    short_vol_window: int,
    long_vol_window: int,
) -> pd.DataFrame:
    data = df.copy()
    data["PrevClose"] = data["Close"].shift(1)
    data["DailyReturn"] = data["Close"].pct_change()
    data["GapReturn"] = data["Open"] / data["PrevClose"] - 1.0
    data["FastEMA"] = data["Close"].ewm(
        span=max(2, int(fast_ema_window)),
        adjust=False,
        min_periods=max(2, int(fast_ema_window)),
    ).mean()
    data["SlowEMA"] = data["Close"].ewm(
        span=max(2, int(slow_ema_window)),
        adjust=False,
        min_periods=max(2, int(slow_ema_window)),
    ).mean()
    data["TrendSlope"] = data["FastEMA"].pct_change(periods=max(2, int(trend_slope_window)))
    data["BreakoutHigh"] = data["High"].shift(1).rolling(
        window=max(2, int(breakout_window)),
        min_periods=max(2, int(breakout_window)),
    ).max()
    data["ExitLow"] = data["Low"].shift(1).rolling(
        window=max(2, int(exit_window)),
        min_periods=max(2, int(exit_window)),
    ).min()
    tr = pd.concat(
        [
            (data["High"] - data["Low"]).abs(),
            (data["High"] - data["PrevClose"]).abs(),
            (data["Low"] - data["PrevClose"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR"] = tr.rolling(window=max(2, int(atr_window)), min_periods=max(2, int(atr_window))).mean()
    data["VolumeMA"] = data["Volume"].rolling(window=max(2, int(volume_window)), min_periods=max(2, int(volume_window))).mean()
    data["VolumeRatio"] = data["Volume"] / data["VolumeMA"]
    short_vol = data["DailyReturn"].rolling(window=max(2, int(short_vol_window)), min_periods=max(2, int(short_vol_window))).std(ddof=0)
    long_vol = data["DailyReturn"].rolling(window=max(3, int(long_vol_window)), min_periods=max(3, int(long_vol_window))).std(ddof=0)
    data["CompressionRatio"] = short_vol / long_vol.replace(0, np.nan)
    data["RangeToATR"] = (data["High"] - data["Low"]) / data["ATR"].replace(0, np.nan)
    feature_cols = [
        "PrevClose",
        "DailyReturn",
        "GapReturn",
        "FastEMA",
        "SlowEMA",
        "TrendSlope",
        "BreakoutHigh",
        "ExitLow",
        "ATR",
        "VolumeRatio",
        "CompressionRatio",
        "RangeToATR",
    ]
    return data.dropna(subset=feature_cols)


def run_strategy_backtest(
    base_data: pd.DataFrame,
    params: dict,
    warmup_data: pd.DataFrame | None = None,
    initial_position: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    if warmup_data is not None and len(warmup_data) > 0:
        raw = pd.concat([warmup_data, base_data]).sort_index()
        raw = raw.loc[~raw.index.duplicated(keep="last")]
    else:
        raw = base_data

    featured = add_strategy_features(
        raw,
        int(params["fit_window_days"]),
        int(params["trend_window_days"]),
        int(params["vol_window_days"]),
    )
    bt_data = featured.loc[(featured.index >= base_data.index[0]) & (featured.index <= base_data.index[-1])].copy()
    if bt_data.empty:
        raise ValueError("验证区间在特征计算后为空，请检查预热样本长度")

    bt = Backtest(
        bt_data,
        PolyfitDynamicGridStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=False,
    )

    kwargs = {
        "base_grid_pct": float(params["base_grid_pct"]),
        "volatility_scale": float(params["volatility_scale"]),
        "trend_sensitivity": float(params["trend_sensitivity"]),
        "max_grid_levels": int(params["max_grid_levels"]),
        "take_profit_grid": float(params["take_profit_grid"]),
        "stop_loss_grid": float(params["stop_loss_grid"]),
        "max_holding_days": int(params["max_holding_days"]),
        "cooldown_days": int(params["cooldown_days"]),
        "min_signal_strength": float(params["min_signal_strength"]),
        "position_size": float(params["position_size"]),
        "position_sizing_coef": float(params["position_sizing_coef"]),
        "initial_position": float(np.clip(initial_position, 0.0, 1.0)),
    }
    stats = bt.run(**kwargs)
    return stats, bt_data


def run_ma_strategy_backtest(
    base_data: pd.DataFrame,
    params: dict,
    warmup_data: pd.DataFrame | None = None,
    initial_position: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    if warmup_data is not None and len(warmup_data) > 0:
        raw = pd.concat([warmup_data, base_data]).sort_index()
        raw = raw.loc[~raw.index.duplicated(keep="last")]
    else:
        raw = base_data

    featured = add_ma_strategy_features(
        raw,
        int(params["ma_window_days"]),
        int(params["trend_window_days"]),
        int(params["vol_window_days"]),
    )
    bt_data = featured.loc[(featured.index >= base_data.index[0]) & (featured.index <= base_data.index[-1])].copy()
    if bt_data.empty:
        raise ValueError("MA 基准策略验证区间在特征计算后为空，请检查预热样本长度")

    bt = Backtest(
        bt_data,
        MovingAverageDynamicGridStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=False,
    )

    kwargs = {
        "base_grid_pct": float(params["base_grid_pct"]),
        "volatility_scale": float(params["volatility_scale"]),
        "trend_sensitivity": float(params["trend_sensitivity"]),
        "max_grid_levels": int(params["max_grid_levels"]),
        "take_profit_grid": float(params["take_profit_grid"]),
        "stop_loss_grid": float(params["stop_loss_grid"]),
        "max_holding_days": int(params["max_holding_days"]),
        "cooldown_days": int(params["cooldown_days"]),
        "min_signal_strength": float(params["min_signal_strength"]),
        "position_size": float(params["position_size"]),
        "position_sizing_coef": float(params["position_sizing_coef"]),
        "initial_position": float(np.clip(initial_position, 0.0, 1.0)),
    }
    stats = bt.run(**kwargs)
    return stats, bt_data


def _initial_position_from_deviation(
    bt_data: pd.DataFrame,
    deviation_col: str,
    base_grid_pct: float,
    min_signal_strength: float,
    max_grid_levels: int,
    position_size: float,
) -> float:
    if bt_data.empty or deviation_col not in bt_data.columns:
        return 0.0

    deviation = float(bt_data[deviation_col].iloc[0])
    if not np.isfinite(deviation) or deviation >= 0:
        return 0.0

    base_grid = max(float(base_grid_pct), 1e-9)
    signal_strength = abs(deviation) / base_grid
    if signal_strength < float(min_signal_strength):
        return 0.0

    level_ratio = np.clip(signal_strength / max(int(max_grid_levels), 1), 0.0, 1.0)
    max_size = float(np.clip(position_size, 0.0, 1.0))
    return float(np.clip(max_size * level_ratio, 0.0, max_size))


def _extract_ending_position(stats: pd.Series) -> float:
    strategy_obj = stats.get("_strategy")
    ending = getattr(strategy_obj, "ending_position", 0.0)
    return float(np.clip(ending, 0.0, 1.0))


def run_launch_strategy_backtest(
    base_data: pd.DataFrame,
    params: dict,
    warmup_data: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    if warmup_data is not None and len(warmup_data) > 0:
        raw = pd.concat([warmup_data, base_data]).sort_index()
        raw = raw.loc[~raw.index.duplicated(keep="last")]
    else:
        raw = base_data

    featured = add_momentum_features(
        raw,
        int(params["fast_ema_window"]),
        int(params["slow_ema_window"]),
        int(params["trend_slope_window"]),
        int(params["breakout_window"]),
        int(params["exit_window"]),
        int(params["atr_window"]),
        int(params["volume_window"]),
        int(params["short_vol_window"]),
        int(params["long_vol_window"]),
    )
    bt_data = featured.loc[(featured.index >= base_data.index[0]) & (featured.index <= base_data.index[-1])].copy()
    if bt_data.empty:
        raise ValueError("启动突破策略验证区间在特征计算后为空，请检查预热样本长度")

    bt = Backtest(
        bt_data,
        LaunchBreakoutMomentumStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    kwargs = {
        "breakout_buffer_atr": float(params["breakout_buffer_atr"]),
        "min_trend_slope": float(params["min_trend_slope"]),
        "min_breakout_return": float(params["min_breakout_return"]),
        "min_close_strength": float(params["min_close_strength"]),
        "min_volume_ratio": float(params["min_volume_ratio"]),
        "max_compression_ratio": float(params["max_compression_ratio"]),
        "min_range_to_atr": float(params["min_range_to_atr"]),
        "min_continuation_return": float(params["min_continuation_return"]),
        "min_continuation_volume_ratio": float(params["min_continuation_volume_ratio"]),
        "initial_stop_atr_mult": float(params["initial_stop_atr_mult"]),
        "trailing_atr_mult": float(params["trailing_atr_mult"]),
        "max_holding_days": int(params["max_holding_days"]),
        "position_size": float(params["position_size"]),
    }
    stats = bt.run(**kwargs)
    return stats, bt_data


def run_retest_strategy_backtest(
    base_data: pd.DataFrame,
    params: dict,
    warmup_data: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    if warmup_data is not None and len(warmup_data) > 0:
        raw = pd.concat([warmup_data, base_data]).sort_index()
        raw = raw.loc[~raw.index.duplicated(keep="last")]
    else:
        raw = base_data

    featured = add_momentum_features(
        raw,
        int(params["fast_ema_window"]),
        int(params["slow_ema_window"]),
        int(params["trend_slope_window"]),
        int(params["breakout_window"]),
        int(params["exit_window"]),
        int(params["atr_window"]),
        int(params["volume_window"]),
        int(params["short_vol_window"]),
        int(params["long_vol_window"]),
    )
    bt_data = featured.loc[(featured.index >= base_data.index[0]) & (featured.index <= base_data.index[-1])].copy()
    if bt_data.empty:
        raise ValueError("突破回踩确认策略验证区间在特征计算后为空，请检查预热样本长度")

    bt = Backtest(
        bt_data,
        BreakoutRetestMomentumStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    kwargs = {
        "breakout_buffer_atr": float(params["breakout_buffer_atr"]),
        "retest_tolerance_atr": float(params["retest_tolerance_atr"]),
        "retest_window_days": int(params["retest_window_days"]),
        "min_trend_slope": float(params["min_trend_slope"]),
        "min_breakout_volume_ratio": float(params["min_breakout_volume_ratio"]),
        "min_retest_volume_ratio": float(params["min_retest_volume_ratio"]),
        "max_setup_compression_ratio": float(params["max_setup_compression_ratio"]),
        "min_breakout_range_to_atr": float(params["min_breakout_range_to_atr"]),
        "max_extension_pct": float(params["max_extension_pct"]),
        "initial_stop_atr_mult": float(params["initial_stop_atr_mult"]),
        "trailing_atr_mult": float(params["trailing_atr_mult"]),
        "max_holding_days": int(params["max_holding_days"]),
        "position_size": float(params["position_size"]),
    }
    stats = bt.run(**kwargs)
    return stats, bt_data


def run_event_strategy_backtest(
    base_data: pd.DataFrame,
    params: dict,
    warmup_data: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    if warmup_data is not None and len(warmup_data) > 0:
        raw = pd.concat([warmup_data, base_data]).sort_index()
        raw = raw.loc[~raw.index.duplicated(keep="last")]
    else:
        raw = base_data

    featured = add_momentum_features(
        raw,
        int(params["fast_ema_window"]),
        int(params["slow_ema_window"]),
        int(params["trend_slope_window"]),
        int(params["breakout_window"]),
        int(params["exit_window"]),
        int(params["atr_window"]),
        int(params["volume_window"]),
        int(params["short_vol_window"]),
        int(params["long_vol_window"]),
    )
    bt_data = featured.loc[(featured.index >= base_data.index[0]) & (featured.index <= base_data.index[-1])].copy()
    if bt_data.empty:
        raise ValueError("事件型启动策略验证区间在特征计算后为空，请检查预热样本长度")

    bt = Backtest(
        bt_data,
        EventDrivenLaunchStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    kwargs = {
        "min_gap_return": float(params["min_gap_return"]),
        "min_event_return": float(params["min_event_return"]),
        "min_close_strength": float(params["min_close_strength"]),
        "min_volume_ratio": float(params["min_volume_ratio"]),
        "max_compression_ratio": float(params["max_compression_ratio"]),
        "min_range_to_atr": float(params["min_range_to_atr"]),
        "min_trend_slope": float(params["min_trend_slope"]),
        "max_extension_pct": float(params["max_extension_pct"]),
        "initial_stop_atr_mult": float(params["initial_stop_atr_mult"]),
        "trailing_atr_mult": float(params["trailing_atr_mult"]),
        "max_holding_days": int(params["max_holding_days"]),
        "position_size": float(params["position_size"]),
    }
    stats = bt.run(**kwargs)
    return stats, bt_data


def scan_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    selected = _sample_param_combinations(param_space, max_evals=max_evals, random_seed=random_seed)

    feature_cache: dict[tuple[int, int, int], pd.DataFrame] = {}
    results = []
    start = time.time()

    for i, params in enumerate(selected, start=1):
        key = (
            int(params["fit_window_days"]),
            int(params["trend_window_days"]),
            int(params["vol_window_days"]),
        )
        if key not in feature_cache:
            feature_cache[key] = add_strategy_features(base_data, key[0], key[1], key[2])

        bt_data = feature_cache[key]
        if bt_data.empty:
            continue

        bt = Backtest(
            bt_data,
            PolyfitDynamicGridStrategy,
            cash=100000,
            commission=0.0001,
            exclusive_orders=True,
            finalize_trades=True,
        )
        strategy_kwargs = {
            k: v
            for k, v in params.items()
            if k not in {"fit_window_days", "trend_window_days", "vol_window_days"}
        }
        stats = bt.run(**strategy_kwargs)
        results.append(
            {
                **params,
                "Return [%]": float(stats["Return [%]"]),
                "Return (Ann.) [%]": float(stats["Return (Ann.) [%]"]),
                "Max. Drawdown [%]": float(stats["Max. Drawdown [%]"]),
                "# Trades": int(stats["# Trades"]),
            }
        )

        if i % 200 == 0 or i == len(selected):
            elapsed = time.time() - start
            print(f"训练集扫描进度: {i}/{len(selected)}，耗时 {elapsed:.1f}s")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError("训练参数扫描结果为空，请增大样本长度或缩小拟合窗口")

    buy_hold_total = float(base_data["Close"].iloc[-1] / base_data["Close"].iloc[0] - 1)
    years = max(len(base_data) / 252.0, 1e-9)
    buy_hold_ann = float((1 + buy_hold_total) ** (1 / years) - 1) * 100
    result_df["TradeFreq"] = result_df["# Trades"] / years
    result_df["ExcessAnn"] = result_df["Return (Ann.) [%]"] - buy_hold_ann
    trade_bonus = np.minimum(result_df["TradeFreq"], 12.0)
    overtrade_penalty = np.maximum(result_df["TradeFreq"] - 18.0, 0.0)
    result_df["Score"] = (
        result_df["Return (Ann.) [%]"]
        + 0.45 * result_df["ExcessAnn"]
        - 0.55 * result_df["Max. Drawdown [%]"].abs()
        + 0.20 * trade_bonus
        - 0.55 * overtrade_penalty
    )
    result_df.loc[result_df["# Trades"] < 4, "Score"] -= 8.0
    result_df.loc[result_df["Return (Ann.) [%]"] < 10.0, "Score"] -= 4.0
    result_df.loc[result_df["Max. Drawdown [%]"].abs() > 15.0, "Score"] -= 4.0
    result_df = result_df.sort_values(["Return [%]", "Score"], ascending=[False, False]).reset_index(drop=True)
    best = result_df.iloc[0].to_dict()
    return best, result_df


def scan_ma_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    selected = _sample_param_combinations_with_validator(
        param_space,
        max_evals=max_evals,
        random_seed=random_seed,
        validator=_valid_ma_param_set,
    )

    feature_cache: dict[tuple[int, int, int], pd.DataFrame] = {}
    results = []
    start = time.time()

    for i, params in enumerate(selected, start=1):
        key = (
            int(params["ma_window_days"]),
            int(params["trend_window_days"]),
            int(params["vol_window_days"]),
        )
        if key not in feature_cache:
            feature_cache[key] = add_ma_strategy_features(base_data, key[0], key[1], key[2])

        bt_data = feature_cache[key]
        if bt_data.empty:
            continue

        bt = Backtest(
            bt_data,
            MovingAverageDynamicGridStrategy,
            cash=100000,
            commission=0.0001,
            exclusive_orders=True,
            finalize_trades=True,
        )
        strategy_kwargs = {
            k: v
            for k, v in params.items()
            if k not in {"ma_window_days", "trend_window_days", "vol_window_days"}
        }
        stats = bt.run(**strategy_kwargs)
        results.append(
            {
                **params,
                "Return [%]": float(stats["Return [%]"]),
                "Return (Ann.) [%]": float(stats["Return (Ann.) [%]"]),
                "Max. Drawdown [%]": float(stats["Max. Drawdown [%]"]),
                "# Trades": int(stats["# Trades"]),
            }
        )

        if i % 200 == 0 or i == len(selected):
            elapsed = time.time() - start
            print(f"MA 基准策略训练集扫描进度: {i}/{len(selected)}，耗时 {elapsed:.1f}s")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError("MA 基准策略训练参数扫描结果为空，请检查 MA 窗口与样本长度")

    buy_hold_total = float(base_data["Close"].iloc[-1] / base_data["Close"].iloc[0] - 1)
    years = max(len(base_data) / 252.0, 1e-9)
    buy_hold_ann = float((1 + buy_hold_total) ** (1 / years) - 1) * 100
    result_df["TradeFreq"] = result_df["# Trades"] / years
    result_df["ExcessAnn"] = result_df["Return (Ann.) [%]"] - buy_hold_ann
    trade_bonus = np.minimum(result_df["TradeFreq"], 12.0)
    overtrade_penalty = np.maximum(result_df["TradeFreq"] - 18.0, 0.0)
    result_df["Score"] = (
        result_df["Return (Ann.) [%]"]
        + 0.45 * result_df["ExcessAnn"]
        - 0.55 * result_df["Max. Drawdown [%]"].abs()
        + 0.20 * trade_bonus
        - 0.55 * overtrade_penalty
    )
    result_df.loc[result_df["# Trades"] < 4, "Score"] -= 8.0
    result_df.loc[result_df["Return (Ann.) [%]"] < 10.0, "Score"] -= 4.0
    result_df.loc[result_df["Max. Drawdown [%]"].abs() > 15.0, "Score"] -= 4.0
    result_df = result_df.sort_values(["Return [%]", "Score"], ascending=[False, False]).reset_index(drop=True)
    best = result_df.iloc[0].to_dict()
    return best, result_df


def _scan_momentum_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    validator,
    strategy_cls,
    feature_label: str,
    feature_param_names: tuple[str, ...],
    strategy_only_param_names: tuple[str, ...],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    selected = _sample_param_combinations_with_validator(
        param_space,
        max_evals=max_evals,
        random_seed=random_seed,
        validator=validator,
    )

    feature_cache: dict[tuple[int, ...], pd.DataFrame] = {}
    results = []
    start = time.time()

    for i, params in enumerate(selected, start=1):
        key = tuple(int(params[name]) for name in feature_param_names)
        if key not in feature_cache:
            feature_cache[key] = add_momentum_features(base_data, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8])

        bt_data = feature_cache[key]
        if bt_data.empty:
            continue

        bt = Backtest(
            bt_data,
            strategy_cls,
            cash=100000,
            commission=0.0001,
            exclusive_orders=True,
            finalize_trades=True,
        )
        strategy_kwargs = {name: params[name] for name in strategy_only_param_names}
        stats = bt.run(**strategy_kwargs)
        results.append(
            {
                **params,
                "Return [%]": float(stats["Return [%]"]),
                "Return (Ann.) [%]": float(stats["Return (Ann.) [%]"]),
                "Max. Drawdown [%]": float(stats["Max. Drawdown [%]"]),
                "# Trades": int(stats["# Trades"]),
            }
        )

        if i % 200 == 0 or i == len(selected):
            elapsed = time.time() - start
            print(f"{feature_label}训练集扫描进度: {i}/{len(selected)}，耗时 {elapsed:.1f}s")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError(f"{feature_label}训练参数扫描结果为空")

    buy_hold_total = float(base_data["Close"].iloc[-1] / base_data["Close"].iloc[0] - 1)
    years = max(len(base_data) / 252.0, 1e-9)
    buy_hold_ann = float((1 + buy_hold_total) ** (1 / years) - 1) * 100
    min_trades_target = max(3, int(np.ceil(years * 2.0)))
    result_df["TradeFreq"] = result_df["# Trades"] / years
    result_df["ExcessAnn"] = result_df["Return (Ann.) [%]"] - buy_hold_ann
    size_denom = result_df["position_size"].clip(lower=0.3)
    result_df["PositionAdjAnn"] = result_df["Return (Ann.) [%]"] / size_denom
    result_df["PositionAdjExcessAnn"] = result_df["ExcessAnn"] / size_denom
    result_df["UndertradePenalty"] = np.maximum(min_trades_target - result_df["# Trades"], 0.0)
    result_df["OvertradePenalty"] = np.maximum(result_df["TradeFreq"] - 10.0, 0.0)
    result_df["Score"] = (
        result_df["PositionAdjAnn"]
        + 0.70 * result_df["PositionAdjExcessAnn"]
        - 0.55 * result_df["Max. Drawdown [%]"].abs()
        + 0.25 * np.minimum(result_df["TradeFreq"], 8.0)
        - 1.20 * result_df["UndertradePenalty"]
        - 0.60 * result_df["OvertradePenalty"]
    )
    result_df.loc[result_df["# Trades"] < min_trades_target, "Score"] -= 12.0
    result_df.loc[result_df["Return (Ann.) [%]"] < 6.0, "Score"] -= 4.0
    result_df.loc[result_df["Max. Drawdown [%]"].abs() > 18.0, "Score"] -= 5.0

    eligible = result_df[result_df["# Trades"] >= min_trades_target]
    if not eligible.empty:
        result_df = eligible.copy()

    result_df = result_df.sort_values(["Score", "Return [%]"], ascending=[False, False]).reset_index(drop=True)
    best = result_df.iloc[0].to_dict()
    return best, result_df


def scan_launch_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    return _scan_momentum_parameters(
        base_data=base_data,
        param_space=param_space,
        validator=_valid_launch_param_set,
        strategy_cls=LaunchBreakoutMomentumStrategy,
        feature_label="启动突破策略",
        feature_param_names=(
            "fast_ema_window",
            "slow_ema_window",
            "trend_slope_window",
            "breakout_window",
            "exit_window",
            "atr_window",
            "volume_window",
            "short_vol_window",
            "long_vol_window",
        ),
        strategy_only_param_names=(
            "breakout_buffer_atr",
            "min_trend_slope",
            "min_breakout_return",
            "min_close_strength",
            "min_volume_ratio",
            "max_compression_ratio",
            "min_range_to_atr",
            "min_continuation_return",
            "min_continuation_volume_ratio",
            "initial_stop_atr_mult",
            "trailing_atr_mult",
            "max_holding_days",
            "position_size",
        ),
        max_evals=max_evals,
        random_seed=random_seed,
    )


def _evaluate_launch_params_on_periods(
    base_data: pd.DataFrame,
    params: dict,
    periods: list[tuple[str, str]],
    warmup_days: int = 360,
) -> tuple[float, int, list[dict]]:
    detail_rows: list[dict] = []
    total_score = 0.0
    hit_count = 0

    for start_str, end_str in periods:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        period_df = base_data.loc[(base_data.index >= start) & (base_data.index <= end)]
        if period_df.empty:
            detail_rows.append(
                {
                    "period": f"{start_str}~{end_str}",
                    "trades": 0,
                    "strategy_ret": np.nan,
                    "benchmark_ret": np.nan,
                    "period_score": -5.0,
                    "hit": 0,
                }
            )
            total_score -= 5.0
            continue

        warmup_start = period_df.index[0] - pd.Timedelta(days=warmup_days)
        warmup_df = base_data.loc[(base_data.index >= warmup_start) & (base_data.index < period_df.index[0])]

        try:
            stats, _ = run_launch_strategy_backtest(period_df, params, warmup_data=warmup_df)
            strat_ret = float(stats["Return [%]"]) / 100.0
            trades = int(stats["# Trades"])
        except Exception:
            strat_ret = -1.0
            trades = 0

        bench = period_df["Close"].dropna()
        bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1) if len(bench) >= 2 else 0.0

        # "跟踪到"定义为：该窗口有交易（先确保参与），收益作为次级优化目标。
        hit = int(trades > 0)
        hit_count += hit

        period_score = 0.0
        if trades == 0:
            period_score -= 5.0
        else:
            period_score += 2.0
        if strat_ret >= 0:
            period_score += 2.0
        if bench_ret > 0:
            period_score += 4.0 * (strat_ret - 0.10 * bench_ret)
            if strat_ret >= 0.20 * bench_ret:
                period_score += 2.0
        else:
            period_score += 2.0 * strat_ret
        if hit:
            period_score += 5.0

        detail_rows.append(
            {
                "period": f"{start_str}~{end_str}",
                "trades": trades,
                "strategy_ret": strat_ret,
                "benchmark_ret": bench_ret,
                "period_score": period_score,
                "hit": hit,
            }
        )
        total_score += period_score

    # 强化命中数量优先级
    total_score += 6.0 * hit_count
    if hit_count < len(periods):
        total_score -= 3.0 * (len(periods) - hit_count)

    return total_score, hit_count, detail_rows


def optimize_launch_for_target_periods(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    periods: list[tuple[str, str]],
    max_evals: int = 600,
    random_seed: int = 2026,
) -> dict:
    selected = _sample_param_combinations_with_validator(
        param_space,
        max_evals=max_evals,
        random_seed=random_seed,
        validator=_valid_launch_param_set,
    )

    best_params: dict | None = None
    best_score = -np.inf
    best_hit_count = -1

    for i, params in enumerate(selected, start=1):
        score, hit_count, _ = _evaluate_launch_params_on_periods(base_data, params, periods)
        if (hit_count > best_hit_count) or (hit_count == best_hit_count and score > best_score):
            best_hit_count = hit_count
            best_score = score
            best_params = dict(params)

        if i % 200 == 0 or i == len(selected):
            print(
                f"启动突破定向优化进度: {i}/{len(selected)}，"
                f"当前最优命中 {best_hit_count}/{len(periods)}，分数 {best_score:.2f}"
            )

    if best_params is None:
        raise ValueError("未找到可用的启动突破定向参数")

    final_score, final_hit, detail = _evaluate_launch_params_on_periods(base_data, best_params, periods)
    print("\n===== 启动突破定向优化结果 =====")
    print(f"目标周期命中数: {final_hit}/{len(periods)}")
    print(f"目标评分: {final_score:.2f}")
    for row in detail:
        strat_ret = row["strategy_ret"]
        bench_ret = row["benchmark_ret"]
        s_txt = "nan" if pd.isna(strat_ret) else f"{strat_ret * 100:.2f}%"
        b_txt = "nan" if pd.isna(bench_ret) else f"{bench_ret * 100:.2f}%"
        print(
            f"{row['period']} | trades={row['trades']} | 策略={s_txt} | 基准={b_txt} | hit={row['hit']}"
        )

    return best_params


def scan_retest_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    return _scan_momentum_parameters(
        base_data=base_data,
        param_space=param_space,
        validator=_valid_retest_param_set,
        strategy_cls=BreakoutRetestMomentumStrategy,
        feature_label="突破回踩确认策略",
        feature_param_names=(
            "fast_ema_window",
            "slow_ema_window",
            "trend_slope_window",
            "breakout_window",
            "exit_window",
            "atr_window",
            "volume_window",
            "short_vol_window",
            "long_vol_window",
        ),
        strategy_only_param_names=(
            "breakout_buffer_atr",
            "retest_tolerance_atr",
            "retest_window_days",
            "min_trend_slope",
            "min_breakout_volume_ratio",
            "min_retest_volume_ratio",
            "max_setup_compression_ratio",
            "min_breakout_range_to_atr",
            "max_extension_pct",
            "initial_stop_atr_mult",
            "trailing_atr_mult",
            "max_holding_days",
            "position_size",
        ),
        max_evals=max_evals,
        random_seed=random_seed,
    )


def scan_event_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    return _scan_momentum_parameters(
        base_data=base_data,
        param_space=param_space,
        validator=_valid_event_param_set,
        strategy_cls=EventDrivenLaunchStrategy,
        feature_label="事件型启动策略",
        feature_param_names=(
            "fast_ema_window",
            "slow_ema_window",
            "trend_slope_window",
            "breakout_window",
            "exit_window",
            "atr_window",
            "volume_window",
            "short_vol_window",
            "long_vol_window",
        ),
        strategy_only_param_names=(
            "min_gap_return",
            "min_event_return",
            "min_close_strength",
            "min_volume_ratio",
            "max_compression_ratio",
            "min_range_to_atr",
            "min_trend_slope",
            "max_extension_pct",
            "initial_stop_atr_mult",
            "trailing_atr_mult",
            "max_holding_days",
            "position_size",
        ),
        max_evals=max_evals,
        random_seed=random_seed,
    )


def calc_independent_annual_returns(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    annual_ret = clean.resample("YE").apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    annual_ret.index = annual_ret.index.year
    return annual_ret


def plot_annual_return_comparison(
    strategy_equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
) -> pd.DataFrame:
    strategy_annual = calc_independent_annual_returns(strategy_equity_curve["Equity"])
    buy_hold_annual = calc_independent_annual_returns(benchmark_close)

    compare = pd.DataFrame({"策略收益": strategy_annual, "长期持有": buy_hold_annual}).dropna(how="all")
    if compare.empty:
        return compare

    x = np.arange(len(compare.index))
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, compare["策略收益"].values * 100, width=width, color="#2ca02c", label="策略收益")
    plt.bar(x + width / 2, compare["长期持有"].values * 100, width=width, color="#1f77b4", label="长期持有")
    plt.axhline(0, color="#333333", linewidth=1, linestyle="--", label="基准线(0%)")
    plt.title(title)
    plt.xlabel("年份")
    plt.ylabel("收益率 (%)")
    plt.xticks(x, compare.index.astype(str))
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return compare


def plot_daily_cumulative_return_comparison(
    strategy_equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
    trades: pd.DataFrame | None = None,
    baseline_series: pd.Series | None = None,
    baseline_label: str = "策略基准累计收益",
) -> pd.DataFrame:
    strategy_daily = strategy_equity_curve["Equity"].pct_change().fillna(0.0)
    buy_hold_daily = benchmark_close.pct_change().fillna(0.0)
    strategy_cum = (1 + strategy_daily).cumprod() - 1
    buy_hold_cum = (1 + buy_hold_daily).cumprod() - 1
    compare = pd.DataFrame({"策略累计收益": strategy_cum, "长期持有累计收益": buy_hold_cum})

    if baseline_series is not None:
        baseline = baseline_series.reindex(compare.index).ffill().bfill()
        baseline_daily = baseline.pct_change().fillna(0.0)
        baseline_cum = (1 + baseline_daily).cumprod() - 1
        compare[baseline_label] = baseline_cum

    plt.figure(figsize=(14, 6))
    plt.plot(compare.index, compare["策略累计收益"] * 100, label="策略累计收益", color="#2ca02c", linewidth=1.3)
    plt.plot(compare.index, compare["长期持有累计收益"] * 100, label="长期持有累计收益", color="#1f77b4", linewidth=1.1)
    if baseline_series is not None and baseline_label in compare.columns:
        plt.plot(compare.index, compare[baseline_label] * 100, label=baseline_label, color="#ff7f0e", linewidth=1.1, linestyle="--")

    if trades is not None and len(trades) > 0:
        trade_df = trades.copy()
        trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
        trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
        line = compare["策略累计收益"]
        entry_points = line.reindex(trade_df["EntryTime"], method="ffill").dropna()
        exit_points = line.reindex(trade_df["ExitTime"], method="ffill").dropna()
        if not entry_points.empty:
            plt.scatter(entry_points.index, entry_points.values * 100, marker="^", color="#d62728", s=28, label="买点", zorder=5)
        if not exit_points.empty:
            plt.scatter(exit_points.index, exit_points.values * 100, marker="v", color="#9467bd", s=28, label="卖点", zorder=5)

    plt.axhline(0, color="#333333", linewidth=1, linestyle="--", label="基准线(0%)")
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("累计收益率 (%)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return compare


def plot_multi_strategy_cumulative_comparison(
    strategy_curves: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
) -> pd.DataFrame:
    compare_data: dict[str, pd.Series] = {}
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#8c564b", "#17becf"]
    plt.figure(figsize=(14, 6))

    for idx, (label, curve) in enumerate(strategy_curves.items()):
        daily = curve["Equity"].pct_change().fillna(0.0)
        cum = (1 + daily).cumprod() - 1
        compare_data[label] = cum
        plt.plot(cum.index, cum.values * 100, label=label, color=colors[idx % len(colors)], linewidth=1.3)

    buy_hold_daily = benchmark_close.pct_change().fillna(0.0)
    compare_data["长期持有累计收益"] = (1 + buy_hold_daily).cumprod() - 1
    compare = pd.DataFrame(compare_data)
    plt.plot(compare.index, compare["长期持有累计收益"] * 100, label="长期持有累计收益", color="#1f77b4", linewidth=1.1)
    plt.axhline(0, color="#333333", linewidth=1, linestyle="--", label="基准线(0%)")
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("累计收益率 (%)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return compare


def print_daily_cumulative_returns_with_signals(
    compare: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    label: str | None = None,
    max_rows: int = 120,
) -> None:
    """打印每日累计收益与买卖点标记，便于逐次回测复核。"""
    if compare.empty:
        print("每日累计收益为空，跳过打印")
        return

    report = compare.copy()
    report["买点"] = ""
    report["卖点"] = ""

    if trades is not None and len(trades) > 0:
        trade_df = trades.copy()
        trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
        trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])

        entry_counts = trade_df.groupby("EntryTime").size()
        exit_counts = trade_df.groupby("ExitTime").size()

        for ts, n in entry_counts.items():
            pos = report.index.get_indexer([ts], method="nearest")[0]
            if pos >= 0:
                report.iloc[pos, report.columns.get_loc("买点")] = f"买入x{int(n)}"

        for ts, n in exit_counts.items():
            pos = report.index.get_indexer([ts], method="nearest")[0]
            if pos >= 0:
                report.iloc[pos, report.columns.get_loc("卖点")] = f"卖出x{int(n)}"

    out = report.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    title = label or "未命名窗口"
    print(f"\n===== {title} 每日累计收益与买卖点 =====")
    if len(out) <= max_rows:
        print(out.to_string())
    else:
        head_n = max_rows // 2
        tail_n = max_rows - head_n
        print(f"总行数 {len(out)}，仅展示前 {head_n} 行和后 {tail_n} 行。")
        print(out.head(head_n).to_string())
        print("... (中间省略) ...")
        print(out.tail(tail_n).to_string())


def summarize_backtest_metrics(stats: pd.Series, benchmark_close: pd.Series) -> dict[str, float]:
    strategy_total = float(stats["Return [%]"]) / 100.0
    max_dd = float(stats["Max. Drawdown [%]"]) / 100.0
    n_trades = int(stats["# Trades"])

    bench = benchmark_close.dropna()
    buy_hold_total = float(bench.iloc[-1] / bench.iloc[0] - 1)
    days = len(bench)
    years = max(days / 252.0, 1e-9)
    months = max(days / 21.0, 1e-9)

    strategy_ann = float(stats["Return (Ann.) [%]"]) / 100.0
    buy_hold_ann = float((1 + buy_hold_total) ** (1 / years) - 1)
    strategy_month = float((1 + strategy_total) ** (1 / months) - 1)
    buy_hold_month = float((1 + buy_hold_total) ** (1 / months) - 1)

    trades = stats["_trades"]
    holding_avg = np.nan
    holding_median = np.nan
    holding_total = np.nan
    if isinstance(trades, pd.DataFrame) and len(trades) > 0:
        entry_time = pd.to_datetime(trades["EntryTime"])
        exit_time = pd.to_datetime(trades["ExitTime"])
        holding_days = (exit_time - entry_time).dt.days.clip(lower=1)
        holding_avg = float(holding_days.mean())
        holding_median = float(holding_days.median())
        holding_total = float(holding_days.sum())

    return {
        "总收益率": strategy_total,
        "最大回撤": max_dd,
        "超额收益": strategy_total - buy_hold_total,
        "年化收益率": strategy_ann,
        "年化超额收益": strategy_ann - buy_hold_ann,
        "月化收益率": strategy_month,
        "月化超额收益": strategy_month - buy_hold_month,
        "交易次数": float(n_trades),
        "年均交易频率": float(n_trades / years),
        "平均持有天数": holding_avg,
        "持有天数中位数": holding_median,
        "持有天数合计": holding_total,
    }


def build_rolling_splits_3y1y(data: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(data.index.year.unique().tolist())
    if len(years) < 4:
        raise ValueError("至少需要 4 个自然年数据才能进行 3年训练+1年验证")

    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for i in range(len(years) - 3):
        train_years = years[i : i + 3]
        val_year = years[i + 3]

        train_start = pd.Timestamp(f"{train_years[0]}-01-01")
        train_end = pd.Timestamp(f"{train_years[-1]}-12-31")
        val_start = pd.Timestamp(f"{val_year}-01-01")
        val_end = pd.Timestamp(f"{val_year}-12-31")

        train_df = data.loc[(data.index >= train_start) & (data.index <= train_end)]
        val_df = data.loc[(data.index >= val_start) & (data.index <= val_end)]
        if len(train_df) < 252 * 2 or len(val_df) < 120:
            continue

        splits.append((train_df.index[0], train_df.index[-1], val_df.index[0], val_df.index[-1]))

    if not splits:
        raise ValueError("未生成有效 3年训练+1年验证窗口，请检查数据长度")
    return splits


def run_walk_forward_validation_3y1y(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    print_daily: bool = True,
    daily_max_rows: int = 60,
    reports_dir: Path | None = None,
    fixed_params: dict | None = None,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    reports = reports_dir or (root / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    splits = build_rolling_splits_3y1y(base_data)
    rows: list[dict] = []

    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== 3年训练1年验证 {i}/{len(splits)} =====\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        if fixed_params is None:
            best_params, scan_df = scan_parameters(
                train_df,
                param_space,
                max_evals=max_evals,
                random_seed=random_seed + i,
            )
        else:
            best_params = dict(fixed_params)
            scan_df = pd.DataFrame([best_params])

        val_stats, val_data = run_strategy_backtest(
            val_df,
            best_params,
            warmup_data=train_df,
        )

        if generate_artifacts:
            annual_png = reports / f"wf3y1y_{i:02d}_annual_return_comparison.png"
            daily_png = reports / f"wf3y1y_{i:02d}_daily_cumulative_return_comparison.png"
            annual_csv = reports / f"wf3y1y_{i:02d}_annual_return_comparison.csv"
            daily_csv = reports / f"wf3y1y_{i:02d}_daily_cumulative_return_comparison.csv"
            scan_csv = reports / f"wf3y1y_{i:02d}_train_scan_top50.csv"

            annual_df = plot_annual_return_comparison(
                strategy_equity_curve=val_stats["_equity_curve"],
                benchmark_close=val_data["Close"],
                title=f"3年训练1年验证窗口{i}: 策略 vs 长期持有（年度独立收益）",
                output_path=annual_png,
            )
            daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=val_stats["_equity_curve"],
                benchmark_close=val_data["Close"],
                title=f"3年训练1年验证窗口{i}: 策略 vs 长期持有（每日累计收益）",
                output_path=daily_png,
                trades=val_stats["_trades"],
            )
            if print_daily:
                print_daily_cumulative_returns_with_signals(
                    compare=daily_df,
                    trades=val_stats["_trades"],
                    label=f"3年训练1年验证窗口{i}",
                    max_rows=daily_max_rows,
                )
            annual_df.to_csv(annual_csv, index=True, encoding="utf-8-sig")
            daily_df.to_csv(daily_csv, index=True, encoding="utf-8-sig")
            scan_df.head(50).to_csv(scan_csv, index=False, encoding="utf-8-sig")

        metrics = summarize_backtest_metrics(val_stats, val_data["Close"])
        rows.append(
            {
                "窗口": i,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{name: best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **metrics,
            }
        )

        print(f"验证窗口{i}总收益率: {metrics['总收益率'] * 100:.2f}%")
        print(f"验证窗口{i}超额收益: {metrics['超额收益'] * 100:.2f}%")
        print(f"验证窗口{i}最大回撤: {metrics['最大回撤'] * 100:.2f}%")

    return pd.DataFrame(rows)


def run_breakout_strategy_comparison_3y1y(
    base_data: pd.DataFrame,
    polyfit_param_space: dict[str, list],
    ma_param_space: dict[str, list],
    launch_param_space: dict[str, list],
    retest_param_space: dict[str, list],
    event_param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    reports_dir: Path | None = None,
    polyfit_fixed_params: dict | None = None,
    ma_fixed_params: dict | None = None,
    launch_fixed_params: dict | None = None,
    retest_fixed_params: dict | None = None,
    event_fixed_params: dict | None = None,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    reports = reports_dir or (root / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    splits = build_rolling_splits_3y1y(base_data)
    rows: list[dict] = []

    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== 五策略 3年训练1年验证 {i}/{len(splits)} =====\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        if polyfit_fixed_params is None:
            polyfit_best_params, polyfit_scan_df = scan_parameters(
                train_df,
                polyfit_param_space,
                max_evals=max_evals,
                random_seed=random_seed + i,
            )
        else:
            polyfit_best_params = dict(polyfit_fixed_params)
            polyfit_scan_df = pd.DataFrame([polyfit_best_params])

        if ma_fixed_params is None:
            ma_best_params, ma_scan_df = scan_ma_parameters(
                train_df,
                ma_param_space,
                max_evals=max_evals,
                random_seed=random_seed + 5_000 + i,
            )
        else:
            ma_best_params = dict(ma_fixed_params)
            ma_scan_df = pd.DataFrame([ma_best_params])

        if launch_fixed_params is None:
            launch_best_params, launch_scan_df = scan_launch_parameters(
                train_df,
                launch_param_space,
                max_evals=max_evals,
                random_seed=random_seed + 10_000 + i,
            )
        else:
            launch_best_params = dict(launch_fixed_params)
            launch_scan_df = pd.DataFrame([launch_best_params])

        if retest_fixed_params is None:
            retest_best_params, retest_scan_df = scan_retest_parameters(
                train_df,
                retest_param_space,
                max_evals=max_evals,
                random_seed=random_seed + 20_000 + i,
            )
        else:
            retest_best_params = dict(retest_fixed_params)
            retest_scan_df = pd.DataFrame([retest_best_params])

        if event_fixed_params is None:
            event_best_params, event_scan_df = scan_event_parameters(
                train_df,
                event_param_space,
                max_evals=max_evals,
                random_seed=random_seed + 30_000 + i,
            )
        else:
            event_best_params = dict(event_fixed_params)
            event_scan_df = pd.DataFrame([event_best_params])

        polyfit_stats, polyfit_val_data = run_strategy_backtest(val_df, polyfit_best_params, warmup_data=train_df)
        ma_stats, ma_val_data = run_ma_strategy_backtest(val_df, ma_best_params, warmup_data=train_df)
        launch_stats, launch_val_data = run_launch_strategy_backtest(val_df, launch_best_params, warmup_data=train_df)
        retest_stats, retest_val_data = run_retest_strategy_backtest(val_df, retest_best_params, warmup_data=train_df)
        event_stats, event_val_data = run_event_strategy_backtest(val_df, event_best_params, warmup_data=train_df)

        if generate_artifacts:
            polyfit_daily_png = reports / f"wf3y1y_{i:02d}_polyfit_daily_cumulative_return_comparison.png"
            ma_daily_png = reports / f"wf3y1y_{i:02d}_ma_daily_cumulative_return_comparison.png"
            launch_daily_png = reports / f"wf3y1y_{i:02d}_launch_daily_cumulative_return_comparison.png"
            retest_daily_png = reports / f"wf3y1y_{i:02d}_retest_daily_cumulative_return_comparison.png"
            event_daily_png = reports / f"wf3y1y_{i:02d}_event_daily_cumulative_return_comparison.png"
            pair_daily_png = reports / f"wf3y1y_{i:02d}_strategy_quint_daily_comparison.png"
            polyfit_daily_csv = reports / f"wf3y1y_{i:02d}_polyfit_daily_cumulative_return_comparison.csv"
            ma_daily_csv = reports / f"wf3y1y_{i:02d}_ma_daily_cumulative_return_comparison.csv"
            launch_daily_csv = reports / f"wf3y1y_{i:02d}_launch_daily_cumulative_return_comparison.csv"
            retest_daily_csv = reports / f"wf3y1y_{i:02d}_retest_daily_cumulative_return_comparison.csv"
            event_daily_csv = reports / f"wf3y1y_{i:02d}_event_daily_cumulative_return_comparison.csv"
            pair_daily_csv = reports / f"wf3y1y_{i:02d}_strategy_quint_daily_comparison.csv"
            polyfit_scan_csv = reports / f"wf3y1y_{i:02d}_polyfit_train_scan_top50.csv"
            ma_scan_csv = reports / f"wf3y1y_{i:02d}_ma_train_scan_top50.csv"
            launch_scan_csv = reports / f"wf3y1y_{i:02d}_launch_train_scan_top50.csv"
            retest_scan_csv = reports / f"wf3y1y_{i:02d}_retest_train_scan_top50.csv"
            event_scan_csv = reports / f"wf3y1y_{i:02d}_event_train_scan_top50.csv"

            polyfit_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                benchmark_close=polyfit_val_data["Close"],
                title=f"窗口{i}: 回归策略 vs 长期持有（每日累计收益）",
                output_path=polyfit_daily_png,
                trades=polyfit_stats["_trades"],
                baseline_series=polyfit_val_data["PolyBasePred"],
                baseline_label="回归基准累计收益",
            )
            ma_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=ma_stats["_equity_curve"],
                benchmark_close=ma_val_data["Close"],
                title=f"窗口{i}: MA 基准策略 vs 长期持有（每日累计收益）",
                output_path=ma_daily_png,
                trades=ma_stats["_trades"],
                baseline_series=ma_val_data["MABase"],
                baseline_label="MA基准累计收益",
            )
            launch_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=launch_stats["_equity_curve"],
                benchmark_close=launch_val_data["Close"],
                title=f"窗口{i}: 启动突破策略 vs 长期持有（每日累计收益）",
                output_path=launch_daily_png,
                trades=launch_stats["_trades"],
            )
            retest_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=retest_stats["_equity_curve"],
                benchmark_close=retest_val_data["Close"],
                title=f"窗口{i}: 突破回踩确认策略 vs 长期持有（每日累计收益）",
                output_path=retest_daily_png,
                trades=retest_stats["_trades"],
            )
            event_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=event_stats["_equity_curve"],
                benchmark_close=event_val_data["Close"],
                title=f"窗口{i}: 事件型启动策略 vs 长期持有（每日累计收益）",
                output_path=event_daily_png,
                trades=event_stats["_trades"],
            )
            pair_daily_df = plot_multi_strategy_cumulative_comparison(
                strategy_curves={
                    "回归策略累计收益": polyfit_stats["_equity_curve"],
                    "MA基准策略累计收益": ma_stats["_equity_curve"],
                    "启动突破策略累计收益": launch_stats["_equity_curve"],
                    "突破回踩确认策略累计收益": retest_stats["_equity_curve"],
                    "事件型启动策略累计收益": event_stats["_equity_curve"],
                },
                benchmark_close=val_df["Close"],
                title=f"窗口{i}: 回归策略 vs MA基准策略 vs 启动突破 vs 突破回踩确认 vs 事件型启动 vs 长期持有",
                output_path=pair_daily_png,
            )
            polyfit_daily_df.to_csv(polyfit_daily_csv, index=True, encoding="utf-8-sig")
            ma_daily_df.to_csv(ma_daily_csv, index=True, encoding="utf-8-sig")
            launch_daily_df.to_csv(launch_daily_csv, index=True, encoding="utf-8-sig")
            retest_daily_df.to_csv(retest_daily_csv, index=True, encoding="utf-8-sig")
            event_daily_df.to_csv(event_daily_csv, index=True, encoding="utf-8-sig")
            pair_daily_df.to_csv(pair_daily_csv, index=True, encoding="utf-8-sig")
            polyfit_scan_df.head(50).to_csv(polyfit_scan_csv, index=False, encoding="utf-8-sig")
            ma_scan_df.head(50).to_csv(ma_scan_csv, index=False, encoding="utf-8-sig")
            launch_scan_df.head(50).to_csv(launch_scan_csv, index=False, encoding="utf-8-sig")
            retest_scan_df.head(50).to_csv(retest_scan_csv, index=False, encoding="utf-8-sig")
            event_scan_df.head(50).to_csv(event_scan_csv, index=False, encoding="utf-8-sig")

        polyfit_metrics = summarize_backtest_metrics(polyfit_stats, polyfit_val_data["Close"])
        ma_metrics = summarize_backtest_metrics(ma_stats, ma_val_data["Close"])
        launch_metrics = summarize_backtest_metrics(launch_stats, launch_val_data["Close"])
        retest_metrics = summarize_backtest_metrics(retest_stats, retest_val_data["Close"])
        event_metrics = summarize_backtest_metrics(event_stats, event_val_data["Close"])
        rows.append(
            {
                "窗口": i,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"polyfit_{name}": polyfit_best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **{f"ma_{name}": ma_best_params[name] for name in MA_SCAN_PARAM_NAMES},
                **{f"launch_{name}": launch_best_params[name] for name in LAUNCH_SCAN_PARAM_NAMES},
                **{f"retest_{name}": retest_best_params[name] for name in RETEST_SCAN_PARAM_NAMES},
                **{f"event_{name}": event_best_params[name] for name in EVENT_SCAN_PARAM_NAMES},
                **{f"polyfit_{k}": v for k, v in polyfit_metrics.items()},
                **{f"ma_{k}": v for k, v in ma_metrics.items()},
                **{f"launch_{k}": v for k, v in launch_metrics.items()},
                **{f"retest_{k}": v for k, v in retest_metrics.items()},
                **{f"event_{k}": v for k, v in event_metrics.items()},
                "MA优于回归_超额收益差": ma_metrics["超额收益"] - polyfit_metrics["超额收益"],
                "MA优于回归_总收益差": ma_metrics["总收益率"] - polyfit_metrics["总收益率"],
                "启动优于回归_超额收益差": launch_metrics["超额收益"] - polyfit_metrics["超额收益"],
                "启动优于回归_总收益差": launch_metrics["总收益率"] - polyfit_metrics["总收益率"],
                "回踩优于回归_超额收益差": retest_metrics["超额收益"] - polyfit_metrics["超额收益"],
                "回踩优于回归_总收益差": retest_metrics["总收益率"] - polyfit_metrics["总收益率"],
                "事件优于回归_超额收益差": event_metrics["超额收益"] - polyfit_metrics["超额收益"],
                "事件优于回归_总收益差": event_metrics["总收益率"] - polyfit_metrics["总收益率"],
                "回踩优于启动_总收益差": retest_metrics["总收益率"] - launch_metrics["总收益率"],
                "事件优于启动_总收益差": event_metrics["总收益率"] - launch_metrics["总收益率"],
                "事件优于回踩_总收益差": event_metrics["总收益率"] - retest_metrics["总收益率"],
            }
        )

        print(f"窗口{i} 回归策略总收益率: {polyfit_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} MA 基准策略总收益率: {ma_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} 启动突破策略总收益率: {launch_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} 突破回踩确认策略总收益率: {retest_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} 事件型启动策略总收益率: {event_metrics['总收益率'] * 100:.2f}%")

    return pd.DataFrame(rows)


def run_polyfit_ma_comparison_3y1y(
    base_data: pd.DataFrame,
    polyfit_param_space: dict[str, list],
    ma_param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    reports_dir: Path | None = None,
    polyfit_fixed_params: dict | None = None,
    ma_fixed_params: dict | None = None,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    reports = reports_dir or (root / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    splits = build_rolling_splits_3y1y(base_data)
    rows: list[dict] = []
    polyfit_prev_ending_position: float | None = None
    ma_prev_ending_position: float | None = None

    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== 双策略 3年训练1年验证 {i}/{len(splits)} =====\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        if polyfit_fixed_params is None:
            polyfit_best_params, polyfit_scan_df = scan_parameters(
                train_df,
                polyfit_param_space,
                max_evals=max_evals,
                random_seed=random_seed + i,
            )
        else:
            polyfit_best_params = dict(polyfit_fixed_params)
            polyfit_scan_df = pd.DataFrame([polyfit_best_params])

        if ma_fixed_params is None:
            ma_best_params, ma_scan_df = scan_ma_parameters(
                train_df,
                ma_param_space,
                max_evals=max_evals,
                random_seed=random_seed + 5_000 + i,
            )
        else:
            ma_best_params = dict(ma_fixed_params)
            ma_scan_df = pd.DataFrame([ma_best_params])

        if polyfit_prev_ending_position is None:
            polyfit_featured = add_strategy_features(
                pd.concat([train_df, val_df]).sort_index().loc[~pd.concat([train_df, val_df]).sort_index().index.duplicated(keep="last")],
                int(polyfit_best_params["fit_window_days"]),
                int(polyfit_best_params["trend_window_days"]),
                int(polyfit_best_params["vol_window_days"]),
            )
            polyfit_bt_probe = polyfit_featured.loc[(polyfit_featured.index >= val_df.index[0]) & (polyfit_featured.index <= val_df.index[-1])].copy()
            polyfit_initial_position = _initial_position_from_deviation(
                polyfit_bt_probe,
                deviation_col="PolyDevPct",
                base_grid_pct=float(polyfit_best_params["base_grid_pct"]),
                min_signal_strength=float(polyfit_best_params["min_signal_strength"]),
                max_grid_levels=int(polyfit_best_params["max_grid_levels"]),
                position_size=float(polyfit_best_params["position_size"]),
            )
        else:
            polyfit_initial_position = float(np.clip(polyfit_prev_ending_position, 0.0, 1.0))

        if ma_prev_ending_position is None:
            ma_featured = add_ma_strategy_features(
                pd.concat([train_df, val_df]).sort_index().loc[~pd.concat([train_df, val_df]).sort_index().index.duplicated(keep="last")],
                int(ma_best_params["ma_window_days"]),
                int(ma_best_params["trend_window_days"]),
                int(ma_best_params["vol_window_days"]),
            )
            ma_bt_probe = ma_featured.loc[(ma_featured.index >= val_df.index[0]) & (ma_featured.index <= val_df.index[-1])].copy()
            ma_initial_position = _initial_position_from_deviation(
                ma_bt_probe,
                deviation_col="MADevPct",
                base_grid_pct=float(ma_best_params["base_grid_pct"]),
                min_signal_strength=float(ma_best_params["min_signal_strength"]),
                max_grid_levels=int(ma_best_params["max_grid_levels"]),
                position_size=float(ma_best_params["position_size"]),
            )
        else:
            ma_initial_position = float(np.clip(ma_prev_ending_position, 0.0, 1.0))

        polyfit_stats, polyfit_val_data = run_strategy_backtest(
            val_df,
            polyfit_best_params,
            warmup_data=train_df,
            initial_position=polyfit_initial_position,
        )
        ma_stats, ma_val_data = run_ma_strategy_backtest(
            val_df,
            ma_best_params,
            warmup_data=train_df,
            initial_position=ma_initial_position,
        )

        polyfit_prev_ending_position = _extract_ending_position(polyfit_stats)
        ma_prev_ending_position = _extract_ending_position(ma_stats)

        if generate_artifacts:
            polyfit_daily_png = reports / f"wf3y1y_{i:02d}_polyfit_daily_cumulative_return_comparison.png"
            ma_daily_png = reports / f"wf3y1y_{i:02d}_ma_daily_cumulative_return_comparison.png"
            pair_daily_png = reports / f"wf3y1y_{i:02d}_strategy_pair_daily_comparison.png"
            polyfit_daily_csv = reports / f"wf3y1y_{i:02d}_polyfit_daily_cumulative_return_comparison.csv"
            ma_daily_csv = reports / f"wf3y1y_{i:02d}_ma_daily_cumulative_return_comparison.csv"
            pair_daily_csv = reports / f"wf3y1y_{i:02d}_strategy_pair_daily_comparison.csv"
            polyfit_scan_csv = reports / f"wf3y1y_{i:02d}_polyfit_train_scan_top50.csv"
            ma_scan_csv = reports / f"wf3y1y_{i:02d}_ma_train_scan_top50.csv"

            polyfit_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                benchmark_close=polyfit_val_data["Close"],
                title=f"窗口{i}: 回归策略 vs 长期持有（每日累计收益）",
                output_path=polyfit_daily_png,
                trades=polyfit_stats["_trades"],
                baseline_series=polyfit_val_data["PolyBasePred"],
                baseline_label="回归基准累计收益",
            )
            ma_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=ma_stats["_equity_curve"],
                benchmark_close=ma_val_data["Close"],
                title=f"窗口{i}: MA 基准策略 vs 长期持有（每日累计收益）",
                output_path=ma_daily_png,
                trades=ma_stats["_trades"],
                baseline_series=ma_val_data["MABase"],
                baseline_label="MA基准累计收益",
            )
            pair_daily_df = plot_multi_strategy_cumulative_comparison(
                strategy_curves={
                    "回归策略累计收益": polyfit_stats["_equity_curve"],
                    "MA基准策略累计收益": ma_stats["_equity_curve"],
                },
                benchmark_close=val_df["Close"],
                title=f"窗口{i}: 回归策略 vs MA基准策略 vs 长期持有",
                output_path=pair_daily_png,
            )
            polyfit_daily_df.to_csv(polyfit_daily_csv, index=True, encoding="utf-8-sig")
            ma_daily_df.to_csv(ma_daily_csv, index=True, encoding="utf-8-sig")
            pair_daily_df.to_csv(pair_daily_csv, index=True, encoding="utf-8-sig")
            polyfit_scan_df.head(50).to_csv(polyfit_scan_csv, index=False, encoding="utf-8-sig")
            ma_scan_df.head(50).to_csv(ma_scan_csv, index=False, encoding="utf-8-sig")

        polyfit_metrics = summarize_backtest_metrics(polyfit_stats, polyfit_val_data["Close"])
        ma_metrics = summarize_backtest_metrics(ma_stats, ma_val_data["Close"])
        rows.append(
            {
                "窗口": i,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"polyfit_{name}": polyfit_best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **{f"ma_{name}": ma_best_params[name] for name in MA_SCAN_PARAM_NAMES},
                **{f"polyfit_{k}": v for k, v in polyfit_metrics.items()},
                **{f"ma_{k}": v for k, v in ma_metrics.items()},
                "polyfit_初始仓位": polyfit_initial_position,
                "polyfit_结束仓位": polyfit_prev_ending_position,
                "ma_初始仓位": ma_initial_position,
                "ma_结束仓位": ma_prev_ending_position,
                "MA优于回归_超额收益差": ma_metrics["超额收益"] - polyfit_metrics["超额收益"],
                "MA优于回归_总收益差": ma_metrics["总收益率"] - polyfit_metrics["总收益率"],
            }
        )

        print(f"窗口{i} 回归策略总收益率: {polyfit_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} 回归策略初始仓位: {polyfit_initial_position:.2%}，结束仓位: {polyfit_prev_ending_position:.2%}")
        print(f"窗口{i} MA 基准策略总收益率: {ma_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} MA 基准策略初始仓位: {ma_initial_position:.2%}，结束仓位: {ma_prev_ending_position:.2%}")

    return pd.DataFrame(rows)


def main() -> None:
    configure_chinese_font()

    data_path = resolve_data_path()
    base_data = load_and_forward_adjust(data_path)

    polyfit_param_space = build_param_space()
    ma_param_space = build_ma_param_space()
    print("执行双策略对比: 回归策略 vs MA基准策略，并以长期持有作为共同基准。")
    summary_df = run_polyfit_ma_comparison_3y1y(
        base_data,
        polyfit_param_space=polyfit_param_space,
        ma_param_space=ma_param_space,
        max_evals=320,
        random_seed=42,
        generate_artifacts=True,
    )

    reports = Path(__file__).resolve().parent / "reports"
    summary_path = reports / "wf3y1y_polyfit_ma_strategy_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n===== 双策略 3年训练1年验证汇总 =====")
    print(f"窗口数: {len(summary_df)}")
    if not summary_df.empty:
        print(f"回归策略平均年化收益率: {summary_df['polyfit_年化收益率'].mean() * 100:.2f}%")
        print(f"MA基准策略平均年化收益率: {summary_df['ma_年化收益率'].mean() * 100:.2f}%")
        print(f"MA优于回归平均总收益差: {summary_df['MA优于回归_总收益差'].mean() * 100:.2f}%")
    print(f"已导出: {summary_path}")


if __name__ == "__main__":
    main()
