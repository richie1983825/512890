from __future__ import annotations

import numpy as np


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

POLYFIT_MA_SWITCH_SCAN_PARAM_NAMES = [
    *POLYFIT_SCAN_PARAM_NAMES,
    "flat_wait_days",
    "switch_deviation_m1",
    "switch_deviation_m2",
    "switch_fast_ma_window",
    "switch_slow_ma_window",
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


def build_polyfit_ma_switch_param_space() -> dict[str, list]:
    param_space = build_param_space()
    param_space.update(
        {
            "flat_wait_days": [5, 8, 10, 15],
            "switch_deviation_m1": [0.02, 0.03, 0.04, 0.05],
            "switch_deviation_m2": [0.005, 0.01, 0.015, 0.02],
            "switch_fast_ma_window": [5, 10, 20],
            "switch_slow_ma_window": [10, 20, 60],
        }
    )
    return param_space


def build_fixed_polyfit_ma_switch_param_space(
    base_polyfit_params: dict,
    switch_param_space: dict[str, list] | None = None,
) -> dict[str, list]:
    param_space = dict((switch_param_space or build_polyfit_ma_switch_param_space()).items())
    for name in POLYFIT_SCAN_PARAM_NAMES:
        param_space[name] = [base_polyfit_params[name]]
    return param_space


def build_fixed_polyfit_param_space(
    fit_window_days: int,
    base_param_space: dict[str, list] | None = None,
) -> dict[str, list]:
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


def build_fixed_ma_param_space(
    ma_window_days: int,
    base_param_space: dict[str, list] | None = None,
) -> dict[str, list]:
    param_space = dict((base_param_space or build_ma_param_space()).items())
    param_space["ma_window_days"] = [int(ma_window_days)]
    return param_space


def valid_polyfit_param_set(params: dict) -> bool:
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


def valid_polyfit_ma_switch_param_set(params: dict) -> bool:
    return (
        valid_polyfit_param_set(params)
        and params["flat_wait_days"] >= 0
        and params["switch_deviation_m1"] > 0
        and params["switch_deviation_m2"] >= 0
        and params["switch_deviation_m2"] < params["switch_deviation_m1"]
        and int(params["switch_fast_ma_window"]) < int(params["switch_slow_ma_window"])
    )


def valid_ma_param_set(params: dict) -> bool:
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


def sample_param_combinations(
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