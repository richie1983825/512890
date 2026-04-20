from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from backtest_core.backtests import (
    extract_ending_position,
    initial_position_from_deviation,
    run_polyfit_ma_switch_backtest,
)
from backtest_core.data import add_strategy_features, combine_train_val_data, load_and_forward_adjust, resolve_data_path
from backtest_core.parameters import build_param_space, build_polyfit_ma_switch_param_space
from backtest_core.scanning import scan_parameters
from backtest_core.workflows import build_rolling_splits_3y1y


def _resolve_initial_position(
    previous_position: float | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict,
) -> float:
    if previous_position is not None:
        return float(previous_position)

    featured = add_strategy_features(
        combine_train_val_data(train_df, val_df),
        int(params["fit_window_days"]),
        int(params["trend_window_days"]),
        int(params["vol_window_days"]),
    )
    probe = featured.loc[(featured.index >= val_df.index[0]) & (featured.index <= val_df.index[-1])].copy()
    return initial_position_from_deviation(
        probe,
        deviation_col="PolyDevPct",
        base_grid_pct=float(params["base_grid_pct"]),
        min_signal_strength=float(params["min_signal_strength"]),
        max_grid_levels=int(params["max_grid_levels"]),
        position_size=float(params["position_size"]),
    )


def run_extra_analysis() -> pd.DataFrame:
    data_path = resolve_data_path()
    base_data = load_and_forward_adjust(data_path)
    splits = build_rolling_splits_3y1y(base_data)

    base_param_space = build_param_space()
    switch_space = build_polyfit_ma_switch_param_space()

    windows: list[dict] = []
    polyfit_returns: list[float] = []
    polyfit_prev_ending_position: float | None = None

    print(f"Total windows: {len(splits)}")
    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"Scanning polyfit params for window {window_idx}: "
            f"{train_start.date()} -> {train_end.date()}"
        )
        best_params, _ = scan_parameters(
            train_df,
            base_param_space,
            max_evals=800,
            random_seed=42 + window_idx,
        )

        polyfit_initial_position = _resolve_initial_position(
            polyfit_prev_ending_position,
            train_df,
            val_df,
            best_params,
        )
        polyfit_stats, _ = run_polyfit_ma_switch_backtest(
            val_df,
            {
                **best_params,
                "flat_wait_days": 9999,
                "switch_deviation_m1": 1.0,
                "switch_deviation_m2": 0.0,
                "switch_fast_ma_window": 5,
                "switch_slow_ma_window": 10,
            },
            warmup_data=train_df,
            initial_position=polyfit_initial_position,
        )
        polyfit_prev_ending_position = extract_ending_position(polyfit_stats)
        polyfit_returns.append(float(polyfit_stats["Return [%]"]) / 100.0)

        windows.append(
            {
                "window_idx": window_idx,
                "train_df": train_df,
                "val_df": val_df,
                "base_params": dict(best_params),
            }
        )

    avg_polyfit_total_return = float(sum(polyfit_returns) / len(polyfit_returns)) if polyfit_returns else 0.0

    combinations = list(
        product(
            switch_space["flat_wait_days"],
            switch_space["switch_deviation_m1"],
            switch_space["switch_deviation_m2"],
            switch_space["switch_fast_ma_window"],
            switch_space["switch_slow_ma_window"],
        )
    )
    total_combos = len(combinations)
    print(f"Total switch combinations to test: {total_combos}")

    results: list[dict] = []
    for combo_index, (flat_wait_days, switch_deviation_m1, switch_deviation_m2, switch_fast_ma_window, switch_slow_ma_window) in enumerate(combinations, start=1):
        if float(switch_deviation_m2) >= float(switch_deviation_m1) or int(switch_fast_ma_window) >= int(switch_slow_ma_window):
            continue

        switch_prev_ending_position: float | None = None
        switch_returns: list[float] = []

        for window in windows:
            params = {
                **window["base_params"],
                "flat_wait_days": int(flat_wait_days),
                "switch_deviation_m1": float(switch_deviation_m1),
                "switch_deviation_m2": float(switch_deviation_m2),
                "switch_fast_ma_window": int(switch_fast_ma_window),
                "switch_slow_ma_window": int(switch_slow_ma_window),
            }
            switch_initial_position = _resolve_initial_position(
                switch_prev_ending_position,
                window["train_df"],
                window["val_df"],
                params,
            )
            switch_stats, _ = run_polyfit_ma_switch_backtest(
                window["val_df"],
                params,
                warmup_data=window["train_df"],
                initial_position=switch_initial_position,
            )
            switch_prev_ending_position = extract_ending_position(switch_stats)
            switch_returns.append(float(switch_stats["Return [%]"]) / 100.0)

        avg_switch_total_return = float(sum(switch_returns) / len(switch_returns)) if switch_returns else 0.0
        results.append(
            {
                "flat_wait_days": int(flat_wait_days),
                "switch_deviation_m1": float(switch_deviation_m1),
                "switch_deviation_m2": float(switch_deviation_m2),
                "switch_fast_ma_window": int(switch_fast_ma_window),
                "switch_slow_ma_window": int(switch_slow_ma_window),
                "avg_switch_total_return": avg_switch_total_return,
                "avg_polyfit_total_return": avg_polyfit_total_return,
                "avg_return_diff_vs_polyfit": avg_switch_total_return - avg_polyfit_total_return,
                **{f"window_{idx + 1}_switch_return": value for idx, value in enumerate(switch_returns)},
            }
        )
        if combo_index % 10 == 0 or combo_index == total_combos:
            print(f"Processed {combo_index}/{total_combos}")

    results_df = pd.DataFrame(results).sort_values(
        by=["avg_switch_total_return", "avg_return_diff_vs_polyfit"],
        ascending=[False, False],
    ).reset_index(drop=True)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "wf3y1y_polyfit_switch_global_nm_scan_summary.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nTop 10 Global Param Combinations:")
    print(
        results_df.head(10)[
            [
                "flat_wait_days",
                "switch_deviation_m1",
                "switch_deviation_m2",
                "switch_fast_ma_window",
                "switch_slow_ma_window",
                "avg_switch_total_return",
                "avg_return_diff_vs_polyfit",
            ]
        ].to_string(index=False)
    )

    print("\nBest Overall Parameters:")
    print(results_df.iloc[0].to_string())
    return results_df


if __name__ == "__main__":
    run_extra_analysis()
