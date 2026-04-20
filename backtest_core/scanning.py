from __future__ import annotations

import time

import numpy as np
import pandas as pd
from backtesting import Backtest

from strategies.moving_average_dynamic_grid_strategy import MovingAverageDynamicGridStrategy
from strategies.polyfit_deviation_ma_switch_strategy import PolyfitDeviationMASwitchStrategy
from strategies.polyfit_dynamic_grid_strategy import PolyfitDynamicGridStrategy

from .data import add_ma_strategy_features, add_strategy_features
from .parameters import sample_param_combinations, valid_ma_param_set, valid_polyfit_ma_switch_param_set, valid_polyfit_param_set


def _score_scan_results(result_df: pd.DataFrame, base_data: pd.DataFrame) -> pd.DataFrame:
    buy_hold_total = float(base_data["Close"].iloc[-1] / base_data["Close"].iloc[0] - 1)
    years = max(len(base_data) / 252.0, 1e-9)
    buy_hold_ann = float((1 + buy_hold_total) ** (1 / years) - 1) * 100
    scored = result_df.copy()
    scored["TradeFreq"] = scored["# Trades"] / years
    scored["ExcessAnn"] = scored["Return (Ann.) [%]"] - buy_hold_ann
    trade_bonus = np.minimum(scored["TradeFreq"], 12.0)
    overtrade_penalty = np.maximum(scored["TradeFreq"] - 18.0, 0.0)
    scored["Score"] = (
        scored["Return (Ann.) [%]"]
        + 0.45 * scored["ExcessAnn"]
        - 0.55 * scored["Max. Drawdown [%]"].abs()
        + 0.20 * trade_bonus
        - 0.55 * overtrade_penalty
    )
    scored.loc[scored["# Trades"] < 4, "Score"] -= 8.0
    scored.loc[scored["Return (Ann.) [%]"] < 10.0, "Score"] -= 4.0
    scored.loc[scored["Max. Drawdown [%]"].abs() > 15.0, "Score"] -= 4.0
    return scored.sort_values(["Return [%]", "Score"], ascending=[False, False]).reset_index(drop=True)


def scan_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    selected = sample_param_combinations(
        param_space,
        max_evals=max_evals,
        random_seed=random_seed,
        validator=valid_polyfit_param_set,
    )

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
            key_name: value
            for key_name, value in params.items()
            if key_name not in {"fit_window_days", "trend_window_days", "vol_window_days"}
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

    scored = _score_scan_results(result_df, base_data)
    return scored.iloc[0].to_dict(), scored


def scan_ma_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    selected = sample_param_combinations(
        param_space,
        max_evals=max_evals,
        random_seed=random_seed,
        validator=valid_ma_param_set,
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
            key_name: value
            for key_name, value in params.items()
            if key_name not in {"ma_window_days", "trend_window_days", "vol_window_days"}
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

    scored = _score_scan_results(result_df, base_data)
    return scored.iloc[0].to_dict(), scored


def scan_polyfit_ma_switch_parameters(
    base_data: pd.DataFrame,
    param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    selected = sample_param_combinations(
        param_space,
        max_evals=max_evals,
        random_seed=random_seed,
        validator=valid_polyfit_ma_switch_param_set,
    )

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
            PolyfitDeviationMASwitchStrategy,
            cash=100000,
            commission=0.0001,
            exclusive_orders=True,
            finalize_trades=True,
        )
        strategy_kwargs = {
            key_name: value
            for key_name, value in params.items()
            if key_name not in {"fit_window_days", "trend_window_days", "vol_window_days"}
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
            print(f"Polyfit 偏离度+MA切换策略训练集扫描进度: {i}/{len(selected)}，耗时 {elapsed:.1f}s")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError("Polyfit 偏离度+MA切换策略训练参数扫描结果为空，请检查参数范围")

    scored = _score_scan_results(result_df, base_data)
    return scored.iloc[0].to_dict(), scored