from __future__ import annotations

from itertools import product
from pathlib import Path
import time

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import numpy as np
import pandas as pd
from backtesting import Backtest

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

def build_param_space() -> dict[str, list]:
    return {
        "fit_window_days": [252],
        "trend_window_days": [10, 15, 20],
        "vol_window_days": [5, 10, 15, 20, 30],
        "base_grid_pct": [0.008, 0.010, 0.012, 0.015],
        "volatility_scale": [0.0, 0.5, 1.0, 1.5, 2.0],
        "trend_sensitivity": [4.0, 6.0, 8.0, 10.0],
        "max_grid_levels": [2, 3, 4],
        "take_profit_grid": [0.6, 0.8, 1.0],
        "stop_loss_grid": [1.2, 1.6, 2.0],
        "max_holding_days": [15, 25, 35],
        "cooldown_days": [1],
        "min_signal_strength": [0.30, 0.45, 0.60],
        "position_size": [i / 100 for i in range(0, 101)],
        "position_sizing_coef": [10.0, 20.0, 30.0, 40.0, 60.0],
    }


def build_fixed_polyfit_param_space(fit_window_days: int, base_param_space: dict[str, list] | None = None) -> dict[str, list]:
    param_space = dict((base_param_space or build_param_space()).items())
    param_space["fit_window_days"] = [int(fit_window_days)]
    return param_space


def build_ma_param_space() -> dict[str, list]:
    return {
        "ma_window_days": [60],
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


def build_fixed_ma_param_space(ma_window_days: int, base_param_space: dict[str, list] | None = None) -> dict[str, list]:
    param_space = dict((base_param_space or build_ma_param_space()).items())
    param_space["ma_window_days"] = [int(ma_window_days)]
    return param_space


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


def _valid_param_set(params: dict) -> bool:
    return (
        params["fit_window_days"] >= 5
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
        "Arial Unicode MS",
        "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    fallbacks = [name for name in candidates if name in available]

    if fallbacks:
        plt.rcParams["font.sans-serif"] = [*fallbacks, "DejaVu Sans"]
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
    configure_chinese_font()
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


def calc_daily_position_ratio(
    strategy_equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
    trades: pd.DataFrame | None = None,
) -> pd.Series:
    index = strategy_equity_curve.index
    position_ratio = pd.Series(0.0, index=index, dtype=float)
    if trades is None or len(trades) == 0:
        return position_ratio

    trade_df = trades.copy()
    trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
    trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
    close = benchmark_close.reindex(index).ffill().bfill()
    equity = strategy_equity_curve["Equity"].reindex(index).ffill().bfill()

    for _, trade in trade_df.iterrows():
        entry_time = pd.Timestamp(trade["EntryTime"])
        exit_time = pd.Timestamp(trade["ExitTime"])
        size = float(abs(trade.get("Size", 0.0)))
        if size <= 0:
            continue
        active_mask = (index >= entry_time) & (index < exit_time)
        if not active_mask.any():
            continue
        notional = size * close.loc[active_mask]
        position_ratio.loc[active_mask] += notional / equity.loc[active_mask].replace(0.0, np.nan)

    return position_ratio.fillna(0.0).clip(lower=0.0)


def _title_with_year(title: str, index: pd.Index) -> str:
    dt_index = pd.to_datetime(index)
    start_year = int(dt_index.min().year)
    end_year = int(dt_index.max().year)
    year_text = f"{start_year}年" if start_year == end_year else f"{start_year}-{end_year}年"
    return f"{year_text} {title}"


def _configure_dense_date_axis(ax: plt.Axes, index: pd.Index) -> None:
    dt_index = pd.to_datetime(index)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.set_xticks(mdates.date2num(dt_index.to_pydatetime()), minor=True)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="x", which="minor", length=2, color="#999999")
    ax.set_xlim(dt_index.min(), dt_index.max())
    ax.grid(alpha=0.2, axis="x", which="major")
    ax.grid(alpha=0.06, axis="x", which="minor")


def plot_daily_cumulative_return_comparison(
    strategy_equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
    trades: pd.DataFrame | None = None,
    baseline_series: pd.Series | None = None,
    baseline_label: str = "策略基准累计收益",
) -> pd.DataFrame:
    configure_chinese_font()
    strategy_daily = strategy_equity_curve["Equity"].pct_change().fillna(0.0)
    buy_hold_daily = benchmark_close.pct_change().fillna(0.0)
    strategy_cum = (1 + strategy_daily).cumprod() - 1
    buy_hold_cum = (1 + buy_hold_daily).cumprod() - 1
    compare = pd.DataFrame({"策略累计收益": strategy_cum, "长期持有累计收益": buy_hold_cum})
    compare["策略仓位"] = calc_daily_position_ratio(strategy_equity_curve, benchmark_close, trades=trades)

    if baseline_series is not None:
        baseline = baseline_series.reindex(compare.index).ffill().bfill()
        baseline_daily = baseline.pct_change().fillna(0.0)
        baseline_cum = (1 + baseline_daily).cumprod() - 1
        compare[baseline_label] = baseline_cum

    fig, (ax_main, ax_pos) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    ax_main.plot(compare.index, compare["策略累计收益"] * 100, label="策略累计收益", color="#2ca02c", linewidth=1.3)
    ax_main.plot(compare.index, compare["长期持有累计收益"] * 100, label="长期持有累计收益", color="#1f77b4", linewidth=1.1)
    if baseline_series is not None and baseline_label in compare.columns:
        ax_main.plot(compare.index, compare[baseline_label] * 100, label=baseline_label, color="#ff7f0e", linewidth=1.1, linestyle="--")

    if trades is not None and len(trades) > 0:
        trade_df = trades.copy()
        trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
        trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
        line = compare["长期持有累计收益"]
        entry_points = line.reindex(trade_df["EntryTime"], method="ffill").dropna()
        exit_points = line.reindex(trade_df["ExitTime"], method="ffill").dropna()
        if not entry_points.empty:
            ax_main.scatter(entry_points.index, entry_points.values * 100, marker="^", color="#d62728", s=28, label="买点", zorder=5)
        if not exit_points.empty:
            ax_main.scatter(exit_points.index, exit_points.values * 100, marker="v", color="#9467bd", s=28, label="卖点", zorder=5)

    ax_main.axhline(0, color="#333333", linewidth=1, linestyle="--", label="基准线(0%)")
    ax_main.set_title(_title_with_year(title, compare.index))
    ax_main.set_ylabel("累计收益率 (%)")
    ax_main.legend()
    ax_main.grid(alpha=0.2)

    ax_pos.plot(compare.index, compare["策略仓位"] * 100, color="#6a9f6a", linewidth=1.0)
    ax_pos.fill_between(compare.index, 0, compare["策略仓位"] * 100, color="#8fbf8f", alpha=0.35)
    ax_pos.set_ylabel("仓位 (%)")
    ax_pos.set_xlabel("日期")
    ax_pos.grid(alpha=0.2, axis="y")
    ax_pos.set_ylim(0, max(100.0, float(compare["策略仓位"].max() * 110.0) if not compare.empty else 100.0))
    _configure_dense_date_axis(ax_pos, compare.index)

    fig.subplots_adjust(hspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return compare


def plot_multi_strategy_cumulative_comparison(
    strategy_curves: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
) -> pd.DataFrame:
    configure_chinese_font()
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
    plt.title(_title_with_year(title, compare.index))
    plt.xlabel("日期")
    plt.ylabel("累计收益率 (%)")
    plt.legend()
    plt.grid(alpha=0.2)
    _configure_dense_date_axis(plt.gca(), compare.index)
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


def _nearest_bar_index(index: pd.Index, ts: pd.Timestamp) -> int:
    pos = index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
    return max(int(pos), 0)


def _join_reasons(reasons: list[str], fallback: str) -> str:
    uniq = []
    for reason in reasons:
        if reason and reason not in uniq:
            uniq.append(reason)
    return "; ".join(uniq) if uniq else fallback


def _infer_grid_entry_reason(
    row: pd.Series,
    params: dict,
    base_col: str,
    dev_col: str,
    trend_col: str,
) -> str:
    close = float(row["Close"])
    base_value = float(row[base_col])
    dev_pct = float(row[dev_col])
    dev_trend = float(row[trend_col])
    rolling_vol_pct = float(row["RollingVolPct"])
    vol_multiplier = 1.0 + float(params["volatility_scale"]) * max(rolling_vol_pct, 0.0)
    dynamic_grid_step = float(params["base_grid_pct"]) * (1.0 + float(params["trend_sensitivity"]) * abs(dev_trend)) * vol_multiplier
    dynamic_grid_step = max(dynamic_grid_step, float(params["base_grid_pct"]) * 0.3)
    signal_strength = abs(dev_pct) / max(dynamic_grid_step, 1e-9)
    entry_level = int(np.clip(np.floor(signal_strength), 1, int(params["max_grid_levels"])))
    return (
        f"mean_reversion_grid;base={base_value:.4f};dev={dev_pct:.4%};"
        f"grid_step={dynamic_grid_step:.4%};level={entry_level};close={close:.4f}"
    )


def _infer_grid_exit_reason(
    row: pd.Series,
    entry_row: pd.Series,
    params: dict,
    dev_col: str,
    trend_col: str,
    holding_days: int,
) -> str:
    dev_pct = float(row[dev_col])
    dev_trend = float(row[trend_col])
    rolling_vol_pct = float(row["RollingVolPct"])
    vol_multiplier = 1.0 + float(params["volatility_scale"]) * max(rolling_vol_pct, 0.0)
    dynamic_grid_step = float(params["base_grid_pct"]) * (1.0 + float(params["trend_sensitivity"]) * abs(dev_trend)) * vol_multiplier
    dynamic_grid_step = max(dynamic_grid_step, float(params["base_grid_pct"]) * 0.3)
    entry_signal = abs(float(entry_row[dev_col])) / max(dynamic_grid_step, 1e-9)
    entry_level = int(np.clip(np.floor(entry_signal), 1, int(params["max_grid_levels"])))
    ref_step = max(dynamic_grid_step, float(params["base_grid_pct"]))
    tp_threshold = entry_level * ref_step * float(params["take_profit_grid"])
    sl_threshold = entry_level * ref_step * float(params["stop_loss_grid"])
    reasons: list[str] = []
    if holding_days >= int(params["max_holding_days"]):
        reasons.append(f"max_holding_days({int(params['max_holding_days'])})")
    if dev_pct >= tp_threshold:
        reasons.append(f"take_profit_grid(dev={dev_pct:.4%}>=tp={tp_threshold:.4%})")
    if dev_pct <= -sl_threshold:
        reasons.append(f"stop_loss_grid(dev={dev_pct:.4%}<=-{sl_threshold:.4%})")
    return _join_reasons(reasons, "grid_exit_unclassified")


def infer_trade_record_reasons(
    trades: pd.DataFrame | None,
    bt_data: pd.DataFrame,
    strategy_name: str,
    params: dict,
) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["EntryReason", "ExitReason"])

    trade_df = trades.copy()
    entry_reasons: list[str] = []
    exit_reasons: list[str] = []
    index = bt_data.index

    for _, trade in trade_df.iterrows():
        entry_bar = int(trade["EntryBar"]) if "EntryBar" in trade else _nearest_bar_index(index, pd.Timestamp(trade["EntryTime"]))
        exit_bar = int(trade["ExitBar"]) if "ExitBar" in trade else _nearest_bar_index(index, pd.Timestamp(trade["ExitTime"]))
        entry_bar = int(np.clip(entry_bar, 0, len(bt_data) - 1))
        exit_bar = int(np.clip(exit_bar, 0, len(bt_data) - 1))
        entry_row = bt_data.iloc[entry_bar]
        exit_row = bt_data.iloc[exit_bar]
        holding_days = exit_bar - entry_bar

        if strategy_name == "polyfit":
            entry_reason = "initial_position_carry" if entry_bar == 0 else _infer_grid_entry_reason(entry_row, params, "PolyBasePred", "PolyDevPct", "PolyDevTrend")
            exit_reason = _infer_grid_exit_reason(exit_row, entry_row, params, "PolyDevPct", "PolyDevTrend", holding_days)
        elif strategy_name == "ma":
            entry_reason = "initial_position_carry" if entry_bar == 0 else _infer_grid_entry_reason(entry_row, params, "MABase", "MADevPct", "MADevTrend")
            exit_reason = _infer_grid_exit_reason(exit_row, entry_row, params, "MADevPct", "MADevTrend", holding_days)
        else:
            entry_reason = "entry_unclassified"
            exit_reason = "exit_unclassified"

        entry_reasons.append(entry_reason)
        exit_reasons.append(exit_reason)

    return pd.DataFrame({"EntryReason": entry_reasons, "ExitReason": exit_reasons}, index=trade_df.index)


def export_trade_records_csv(
    trades: pd.DataFrame | None,
    output_path: Path,
    bt_data: pd.DataFrame | None = None,
    equity_curve: pd.DataFrame | None = None,
    strategy_name: str | None = None,
    params: dict | None = None,
    native_reason_records: list[dict] | None = None,
) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        empty = pd.DataFrame(
            columns=[
                "EntryTime",
                "ExitTime",
                "Size",
                "EntryPositionPct",
                "EntryPrice",
                "ExitPrice",
                "PnL",
                "ReturnPct",
                "HoldingDays",
                "EntryReason",
                "ExitReason",
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_path, index=False, encoding="utf-8-sig")
        return empty

    trade_df = trades.copy()
    trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
    trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
    trade_df["HoldingDays"] = (trade_df["ExitTime"] - trade_df["EntryTime"]).dt.days.clip(lower=1)
    trade_df["EntryPositionPct"] = np.nan
    if equity_curve is not None and len(equity_curve) > 0 and "EntryBar" in trade_df.columns and "Equity" in equity_curve.columns:
        equity_values = equity_curve["Equity"].reset_index(drop=True)
        entry_bars = trade_df["EntryBar"].fillna(-1).astype(int)
        valid_mask = entry_bars.between(0, len(equity_values) - 1)
        if valid_mask.any():
            entry_equity = entry_bars.loc[valid_mask].map(equity_values)
            entry_position_value = trade_df.loc[valid_mask, "Size"].abs() * trade_df.loc[valid_mask, "EntryPrice"]
            trade_df.loc[valid_mask, "EntryPositionPct"] = (
                entry_position_value / entry_equity.replace(0, np.nan)
            )
    if native_reason_records is not None and len(native_reason_records) > 0:
        native_df = pd.DataFrame(native_reason_records).copy()
        for col in ["EntryTime", "ExitTime"]:
            if col in native_df.columns:
                native_df[col] = pd.to_datetime(native_df[col])
        native_keep = [col for col in ["EntryTime", "ExitTime", "EntryReason", "ExitReason"] if col in native_df.columns]
        if {"EntryTime", "ExitTime", "EntryReason", "ExitReason"}.issubset(native_keep):
            native_df = native_df[native_keep].drop_duplicates(subset=["EntryTime", "ExitTime"], keep="last")
            trade_df = trade_df.merge(native_df, on=["EntryTime", "ExitTime"], how="left", suffixes=("", "_native"))
            if "EntryReason_native" in trade_df.columns:
                trade_df["EntryReason"] = trade_df.get("EntryReason").combine_first(trade_df["EntryReason_native"])
                trade_df = trade_df.drop(columns=[col for col in ["EntryReason_native", "ExitReason_native"] if col in trade_df.columns])

    if bt_data is not None and strategy_name is not None and params is not None:
        reason_df = infer_trade_record_reasons(trade_df, bt_data, strategy_name, params)
        if "EntryReason" in trade_df.columns:
            trade_df["EntryReason"] = trade_df["EntryReason"].fillna(reason_df["EntryReason"])
            trade_df["ExitReason"] = trade_df["ExitReason"].fillna(reason_df["ExitReason"])
        else:
            trade_df = pd.concat([trade_df, reason_df], axis=1)
    output_cols = [
        col
        for col in [
            "EntryTime",
            "ExitTime",
            "Size",
            "EntryPositionPct",
            "EntryPrice",
            "ExitPrice",
            "PnL",
            "ReturnPct",
            "HoldingDays",
            "EntryReason",
            "ExitReason",
            "Tag",
            "Duration",
        ]
        if col in trade_df.columns
    ]
    export_df = trade_df[output_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return export_df


def write_window_comparison_summary_markdown(
    output_path: Path,
    title: str,
    sections: list[tuple[str, list[Path]]],
) -> None:
    lines = [f"# {title}", ""]
    for section_title, image_paths in sections:
        if not image_paths:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        for image_path in image_paths:
            lines.append(f"### {image_path.stem}")
            lines.append("")
            lines.append(f"![{image_path.stem}]({image_path.name})")
            lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


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
            trades_csv = reports / f"wf3y1y_{i:02d}_trade_records.csv"

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
            export_trade_records_csv(val_stats["_trades"], trades_csv, bt_data=val_data, equity_curve=val_stats["_equity_curve"], strategy_name="polyfit", params=best_params, native_reason_records=getattr(val_stats.get("_strategy"), "trade_reason_records", None))

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


def run_ma_walk_forward_validation_3y1y(
    base_data: pd.DataFrame,
    ma_param_space: dict[str, list],
    ma_window_days: int,
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    print_daily: bool = True,
    daily_max_rows: int = 60,
    reports_dir: Path | None = None,
    report_prefix: str | None = None,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    reports = reports_dir or (root / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    prefix = report_prefix or f"wf3y1y_ma{int(ma_window_days):02d}"
    splits = build_rolling_splits_3y1y(base_data)
    rows: list[dict] = []
    ma_prev_ending_position: float | None = None
    annual_images: list[Path] = []
    daily_images: list[Path] = []

    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== MA固定周期 3年训练1年验证 {i}/{len(splits)} =====\n"
            f"MA周期: {int(ma_window_days)}\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        ma_best_params, ma_scan_df = scan_ma_parameters(
            train_df,
            ma_param_space,
            max_evals=max_evals,
            random_seed=random_seed + i,
        )

        combined_train_val = pd.concat([train_df, val_df]).sort_index()
        combined_train_val = combined_train_val.loc[~combined_train_val.index.duplicated(keep="last")]
        if ma_prev_ending_position is None:
            ma_featured = add_ma_strategy_features(
                combined_train_val,
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

        ma_stats, ma_val_data = run_ma_strategy_backtest(
            val_df,
            ma_best_params,
            warmup_data=train_df,
            initial_position=ma_initial_position,
        )
        ma_prev_ending_position = _extract_ending_position(ma_stats)

        if generate_artifacts:
            annual_png = reports / f"{prefix}_{i:02d}_annual_return_comparison.png"
            daily_png = reports / f"{prefix}_{i:02d}_daily_cumulative_return_comparison.png"
            annual_csv = reports / f"{prefix}_{i:02d}_annual_return_comparison.csv"
            daily_csv = reports / f"{prefix}_{i:02d}_daily_cumulative_return_comparison.csv"
            scan_csv = reports / f"{prefix}_{i:02d}_train_scan_top50.csv"
            trades_csv = reports / f"{prefix}_{i:02d}_trade_records.csv"

            annual_df = plot_annual_return_comparison(
                strategy_equity_curve=ma_stats["_equity_curve"],
                benchmark_close=ma_val_data["Close"],
                title=f"窗口{i}: MA{int(ma_window_days)} 策略 vs 长期持有（年度独立收益）",
                output_path=annual_png,
            )
            daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=ma_stats["_equity_curve"],
                benchmark_close=ma_val_data["Close"],
                title=f"窗口{i}: MA{int(ma_window_days)} 策略 vs 长期持有（每日累计收益）",
                output_path=daily_png,
                trades=ma_stats["_trades"],
                baseline_series=ma_val_data["MABase"],
                baseline_label="MA基准累计收益",
            )
            if print_daily:
                print_daily_cumulative_returns_with_signals(
                    compare=daily_df,
                    trades=ma_stats["_trades"],
                    label=f"MA{int(ma_window_days)} 窗口{i}",
                    max_rows=daily_max_rows,
                )
            annual_df.to_csv(annual_csv, index=True, encoding="utf-8-sig")
            daily_df.to_csv(daily_csv, index=True, encoding="utf-8-sig")
            ma_scan_df.head(50).to_csv(scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                ma_stats["_trades"],
                trades_csv,
                bt_data=ma_val_data,
                equity_curve=ma_stats["_equity_curve"],
                strategy_name="ma",
                params=ma_best_params,
                native_reason_records=getattr(ma_stats.get("_strategy"), "trade_reason_records", None),
            )
            annual_images.append(annual_png)
            daily_images.append(daily_png)

        ma_metrics = summarize_backtest_metrics(ma_stats, ma_val_data["Close"])
        rows.append(
            {
                "MA周期": int(ma_window_days),
                "窗口": i,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"ma_{name}": ma_best_params[name] for name in MA_SCAN_PARAM_NAMES},
                **{f"ma_{k}": v for k, v in ma_metrics.items()},
                "ma_初始仓位": ma_initial_position,
                "ma_结束仓位": ma_prev_ending_position,
            }
        )

        print(f"窗口{i} MA{int(ma_window_days)} 策略总收益率: {ma_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} MA{int(ma_window_days)} 策略超额收益: {ma_metrics['超额收益'] * 100:.2f}%")
        print(f"窗口{i} MA{int(ma_window_days)} 策略最大回撤: {ma_metrics['最大回撤'] * 100:.2f}%")

    if generate_artifacts and len(daily_images) == 4:
        summary_md = reports / f"{prefix}_comparison_summary.md"
        write_window_comparison_summary_markdown(
            summary_md,
            title=f"MA{int(ma_window_days)} 固定周期 3年训练1年验证图表汇总",
            sections=[
                ("年度独立收益对比图", annual_images),
                ("每日累计收益对比图", daily_images),
            ],
        )

    return pd.DataFrame(rows)


def run_polyfit_walk_forward_validation_3y1y(
    base_data: pd.DataFrame,
    polyfit_param_space: dict[str, list],
    fit_window_days: int,
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    print_daily: bool = True,
    daily_max_rows: int = 60,
    reports_dir: Path | None = None,
    report_prefix: str | None = None,
) -> pd.DataFrame:
    root = Path(__file__).resolve().parent
    reports = reports_dir or (root / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    prefix = report_prefix or f"wf3y1y_polyfitfit{int(fit_window_days):03d}"
    splits = build_rolling_splits_3y1y(base_data)
    rows: list[dict] = []
    polyfit_prev_ending_position: float | None = None
    annual_images: list[Path] = []
    daily_images: list[Path] = []

    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== Polyfit固定拟合窗 3年训练1年验证 {i}/{len(splits)} =====\n"
            f"拟合窗口: {int(fit_window_days)}\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        polyfit_best_params, polyfit_scan_df = scan_parameters(
            train_df,
            polyfit_param_space,
            max_evals=max_evals,
            random_seed=random_seed + i,
        )

        combined_train_val = pd.concat([train_df, val_df]).sort_index()
        combined_train_val = combined_train_val.loc[~combined_train_val.index.duplicated(keep="last")]
        if polyfit_prev_ending_position is None:
            polyfit_featured = add_strategy_features(
                combined_train_val,
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

        polyfit_stats, polyfit_val_data = run_strategy_backtest(
            val_df,
            polyfit_best_params,
            warmup_data=train_df,
            initial_position=polyfit_initial_position,
        )
        polyfit_prev_ending_position = _extract_ending_position(polyfit_stats)

        if generate_artifacts:
            annual_png = reports / f"{prefix}_{i:02d}_annual_return_comparison.png"
            daily_png = reports / f"{prefix}_{i:02d}_daily_cumulative_return_comparison.png"
            annual_csv = reports / f"{prefix}_{i:02d}_annual_return_comparison.csv"
            daily_csv = reports / f"{prefix}_{i:02d}_daily_cumulative_return_comparison.csv"
            scan_csv = reports / f"{prefix}_{i:02d}_train_scan_top50.csv"
            trades_csv = reports / f"{prefix}_{i:02d}_trade_records.csv"

            annual_df = plot_annual_return_comparison(
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                benchmark_close=polyfit_val_data["Close"],
                title=f"窗口{i}: Polyfit拟合窗{int(fit_window_days)} 策略 vs 长期持有（年度独立收益）",
                output_path=annual_png,
            )
            daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                benchmark_close=polyfit_val_data["Close"],
                title=f"窗口{i}: Polyfit拟合窗{int(fit_window_days)} 策略 vs 长期持有（每日累计收益）",
                output_path=daily_png,
                trades=polyfit_stats["_trades"],
                baseline_series=polyfit_val_data["PolyBasePred"],
                baseline_label="Polyfit基准累计收益",
            )
            if print_daily:
                print_daily_cumulative_returns_with_signals(
                    compare=daily_df,
                    trades=polyfit_stats["_trades"],
                    label=f"Polyfit拟合窗{int(fit_window_days)} 窗口{i}",
                    max_rows=daily_max_rows,
                )
            annual_df.to_csv(annual_csv, index=True, encoding="utf-8-sig")
            daily_df.to_csv(daily_csv, index=True, encoding="utf-8-sig")
            polyfit_scan_df.head(50).to_csv(scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                polyfit_stats["_trades"],
                trades_csv,
                bt_data=polyfit_val_data,
                equity_curve=polyfit_stats["_equity_curve"],
                strategy_name="polyfit",
                params=polyfit_best_params,
                native_reason_records=getattr(polyfit_stats.get("_strategy"), "trade_reason_records", None),
            )
            annual_images.append(annual_png)
            daily_images.append(daily_png)

        polyfit_metrics = summarize_backtest_metrics(polyfit_stats, polyfit_val_data["Close"])
        rows.append(
            {
                "拟合窗口": int(fit_window_days),
                "窗口": i,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"polyfit_{name}": polyfit_best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **{f"polyfit_{k}": v for k, v in polyfit_metrics.items()},
                "polyfit_初始仓位": polyfit_initial_position,
                "polyfit_结束仓位": polyfit_prev_ending_position,
            }
        )

        print(f"窗口{i} Polyfit拟合窗{int(fit_window_days)} 总收益率: {polyfit_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{i} Polyfit拟合窗{int(fit_window_days)} 超额收益: {polyfit_metrics['超额收益'] * 100:.2f}%")
        print(f"窗口{i} Polyfit拟合窗{int(fit_window_days)} 最大回撤: {polyfit_metrics['最大回撤'] * 100:.2f}%")

    if generate_artifacts and len(daily_images) == 4:
        summary_md = reports / f"{prefix}_comparison_summary.md"
        write_window_comparison_summary_markdown(
            summary_md,
            title=f"Polyfit拟合窗{int(fit_window_days)} 固定周期 3年训练1年验证图表汇总",
            sections=[
                ("年度独立收益对比图", annual_images),
                ("每日累计收益对比图", daily_images),
            ],
        )

    return pd.DataFrame(rows)


def run_polyfit_fit_window_comparison_3y1y(
    base_data: pd.DataFrame,
    fit_windows: list[int],
    polyfit_param_space: dict[str, list] | None = None,
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    print_daily: bool = True,
    daily_max_rows: int = 60,
    reports_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parent
    reports = reports_dir or (root / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    detailed_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict] = []

    for idx, fit_window in enumerate(fit_windows):
        fixed_param_space = build_fixed_polyfit_param_space(int(fit_window), polyfit_param_space)
        detail_df = run_polyfit_walk_forward_validation_3y1y(
            base_data,
            polyfit_param_space=fixed_param_space,
            fit_window_days=int(fit_window),
            max_evals=max_evals,
            random_seed=random_seed + idx * 10_000,
            generate_artifacts=generate_artifacts,
            print_daily=print_daily,
            daily_max_rows=daily_max_rows,
            reports_dir=reports,
            report_prefix=f"wf3y1y_polyfitfit{int(fit_window):03d}",
        )
        detail_summary_path = reports / f"wf3y1y_polyfitfit{int(fit_window):03d}_strategy_summary.csv"
        detail_df.to_csv(detail_summary_path, index=False, encoding="utf-8-sig")
        detailed_frames.append(detail_df)

        aggregate_rows.append(
            {
                "拟合窗口": int(fit_window),
                "平均总收益率": float(detail_df["polyfit_总收益率"].mean()),
                "平均超额收益": float(detail_df["polyfit_超额收益"].mean()),
                "平均最大回撤": float(detail_df["polyfit_最大回撤"].mean()),
                "平均年化收益率": float(detail_df["polyfit_年化收益率"].mean()),
                "平均月化收益率": float(detail_df["polyfit_月化收益率"].mean()),
                "平均交易次数": float(detail_df["polyfit_交易次数"].mean()),
                "平均持有天数": float(detail_df["polyfit_平均持有天数"].mean()),
                "最佳窗口": int(detail_df.loc[detail_df["polyfit_总收益率"].idxmax(), "窗口"]),
                "最佳窗口总收益率": float(detail_df["polyfit_总收益率"].max()),
            }
        )

    detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("平均总收益率", ascending=False).reset_index(drop=True)

    if generate_artifacts:
        detailed_df.to_csv(reports / "wf3y1y_polyfit_fit_window_comparison_detailed.csv", index=False, encoding="utf-8-sig")
        aggregate_df.to_csv(reports / "wf3y1y_polyfit_fit_window_comparison_summary.csv", index=False, encoding="utf-8-sig")

    return detailed_df, aggregate_df


def run_ma_period_comparison_3y1y(
    base_data: pd.DataFrame,
    ma_windows: list[int],
    ma_param_space: dict[str, list] | None = None,
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    print_daily: bool = True,
    daily_max_rows: int = 60,
    reports_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parent
    reports = reports_dir or (root / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    detailed_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict] = []

    for idx, ma_window in enumerate(ma_windows):
        fixed_param_space = build_fixed_ma_param_space(int(ma_window), ma_param_space)
        detail_df = run_ma_walk_forward_validation_3y1y(
            base_data,
            ma_param_space=fixed_param_space,
            ma_window_days=int(ma_window),
            max_evals=max_evals,
            random_seed=random_seed + idx * 10_000,
            generate_artifacts=generate_artifacts,
            print_daily=print_daily,
            daily_max_rows=daily_max_rows,
            reports_dir=reports,
            report_prefix=f"wf3y1y_ma{int(ma_window):02d}",
        )
        detail_summary_path = reports / f"wf3y1y_ma{int(ma_window):02d}_strategy_summary.csv"
        detail_df.to_csv(detail_summary_path, index=False, encoding="utf-8-sig")
        detailed_frames.append(detail_df)

        aggregate_rows.append(
            {
                "MA周期": int(ma_window),
                "平均总收益率": float(detail_df["ma_总收益率"].mean()),
                "平均超额收益": float(detail_df["ma_超额收益"].mean()),
                "平均最大回撤": float(detail_df["ma_最大回撤"].mean()),
                "平均年化收益率": float(detail_df["ma_年化收益率"].mean()),
                "平均月化收益率": float(detail_df["ma_月化收益率"].mean()),
                "平均交易次数": float(detail_df["ma_交易次数"].mean()),
                "平均持有天数": float(detail_df["ma_平均持有天数"].mean()),
                "最佳窗口": int(detail_df.loc[detail_df["ma_总收益率"].idxmax(), "窗口"]),
                "最佳窗口总收益率": float(detail_df["ma_总收益率"].max()),
            }
        )

    detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("平均总收益率", ascending=False).reset_index(drop=True)

    if generate_artifacts:
        detailed_df.to_csv(reports / "wf3y1y_ma_period_comparison_detailed.csv", index=False, encoding="utf-8-sig")
        aggregate_df.to_csv(reports / "wf3y1y_ma_period_comparison_summary.csv", index=False, encoding="utf-8-sig")

    return detailed_df, aggregate_df


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
            polyfit_trades_csv = reports / f"wf3y1y_{i:02d}_polyfit_trade_records.csv"
            ma_trades_csv = reports / f"wf3y1y_{i:02d}_ma_trade_records.csv"

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
            export_trade_records_csv(polyfit_stats["_trades"], polyfit_trades_csv, bt_data=polyfit_val_data, equity_curve=polyfit_stats["_equity_curve"], strategy_name="polyfit", params=polyfit_best_params, native_reason_records=getattr(polyfit_stats.get("_strategy"), "trade_reason_records", None))
            export_trade_records_csv(ma_stats["_trades"], ma_trades_csv, bt_data=ma_val_data, equity_curve=ma_stats["_equity_curve"], strategy_name="ma", params=ma_best_params, native_reason_records=getattr(ma_stats.get("_strategy"), "trade_reason_records", None))

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
        max_evals=800,
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
