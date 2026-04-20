from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Backtest

from strategies.moving_average_dynamic_grid_strategy import MovingAverageDynamicGridStrategy
from strategies.polyfit_deviation_ma_stoploss_nextday_guard_switch_strategy import PolyfitDeviationMAStoplossNextdayGuardSwitchStrategy
from strategies.polyfit_deviation_ma_switch_strategy import PolyfitDeviationMASwitchStrategy
from strategies.polyfit_dynamic_grid_strategy import PolyfitDynamicGridStrategy

from .data import add_ma_strategy_features, add_strategy_features


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
    stats = bt.run(
        base_grid_pct=float(params["base_grid_pct"]),
        volatility_scale=float(params["volatility_scale"]),
        trend_sensitivity=float(params["trend_sensitivity"]),
        max_grid_levels=int(params["max_grid_levels"]),
        take_profit_grid=float(params["take_profit_grid"]),
        stop_loss_grid=float(params["stop_loss_grid"]),
        max_holding_days=int(params["max_holding_days"]),
        cooldown_days=int(params["cooldown_days"]),
        min_signal_strength=float(params["min_signal_strength"]),
        position_size=float(params["position_size"]),
        position_sizing_coef=float(params["position_sizing_coef"]),
        initial_position=float(np.clip(initial_position, 0.0, 1.0)),
    )
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
    stats = bt.run(
        base_grid_pct=float(params["base_grid_pct"]),
        volatility_scale=float(params["volatility_scale"]),
        trend_sensitivity=float(params["trend_sensitivity"]),
        max_grid_levels=int(params["max_grid_levels"]),
        take_profit_grid=float(params["take_profit_grid"]),
        stop_loss_grid=float(params["stop_loss_grid"]),
        max_holding_days=int(params["max_holding_days"]),
        cooldown_days=int(params["cooldown_days"]),
        min_signal_strength=float(params["min_signal_strength"]),
        position_size=float(params["position_size"]),
        position_sizing_coef=float(params["position_sizing_coef"]),
        initial_position=float(np.clip(initial_position, 0.0, 1.0)),
    )
    return stats, bt_data


def run_polyfit_ma_switch_backtest(
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
        raise ValueError("Polyfit 偏离度+MA切换策略验证区间在特征计算后为空，请检查预热样本长度")

    bt = Backtest(
        bt_data,
        PolyfitDeviationMASwitchStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=False,
    )
    stats = bt.run(
        base_grid_pct=float(params["base_grid_pct"]),
        volatility_scale=float(params["volatility_scale"]),
        trend_sensitivity=float(params["trend_sensitivity"]),
        max_grid_levels=int(params["max_grid_levels"]),
        take_profit_grid=float(params["take_profit_grid"]),
        stop_loss_grid=float(params["stop_loss_grid"]),
        max_holding_days=int(params["max_holding_days"]),
        cooldown_days=int(params["cooldown_days"]),
        min_signal_strength=float(params["min_signal_strength"]),
        position_size=float(params["position_size"]),
        position_sizing_coef=float(params["position_sizing_coef"]),
        initial_position=float(np.clip(initial_position, 0.0, 1.0)),
        flat_wait_days=int(params["flat_wait_days"]),
        switch_deviation_m1=float(params["switch_deviation_m1"]),
        switch_deviation_m2=float(params["switch_deviation_m2"]),
        switch_fast_ma_window=int(params["switch_fast_ma_window"]),
        switch_slow_ma_window=int(params["switch_slow_ma_window"]),
    )
    return stats, bt_data


def run_polyfit_ma_stoploss_nextday_guard_backtest(
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
        raise ValueError("止损次日偏离度保护策略验证区间在特征计算后为空，请检查预热样本长度")

    bt = Backtest(
        bt_data,
        PolyfitDeviationMAStoplossNextdayGuardSwitchStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=False,
    )
    stats = bt.run(
        base_grid_pct=float(params["base_grid_pct"]),
        volatility_scale=float(params["volatility_scale"]),
        trend_sensitivity=float(params["trend_sensitivity"]),
        max_grid_levels=int(params["max_grid_levels"]),
        take_profit_grid=float(params["take_profit_grid"]),
        stop_loss_grid=float(params["stop_loss_grid"]),
        max_holding_days=int(params["max_holding_days"]),
        cooldown_days=int(params["cooldown_days"]),
        min_signal_strength=float(params["min_signal_strength"]),
        position_size=float(params["position_size"]),
        position_sizing_coef=float(params["position_sizing_coef"]),
        initial_position=float(np.clip(initial_position, 0.0, 1.0)),
        flat_wait_days=int(params["flat_wait_days"]),
        switch_deviation_m1=float(params["switch_deviation_m1"]),
        switch_deviation_m2=float(params["switch_deviation_m2"]),
        switch_fast_ma_window=int(params["switch_fast_ma_window"]),
        switch_slow_ma_window=int(params["switch_slow_ma_window"]),
    )
    return stats, bt_data


def initial_position_from_deviation(
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


def extract_ending_position(stats: pd.Series) -> float:
    strategy_obj = stats.get("_strategy")
    ending = getattr(strategy_obj, "ending_position", 0.0)
    return float(np.clip(ending, 0.0, 1.0))