from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtests import (
    extract_ending_position,
    initial_position_from_deviation,
    run_ma_strategy_backtest,
    run_polyfit_ma_stoploss_nextday_guard_backtest,
    run_polyfit_ma_switch_backtest,
    run_strategy_backtest,
)
from .data import add_ma_strategy_features, add_strategy_features, combine_train_val_data
from .parameters import (
    MA_SCAN_PARAM_NAMES,
    POLYFIT_MA_SWITCH_SCAN_PARAM_NAMES,
    POLYFIT_SCAN_PARAM_NAMES,
    build_fixed_ma_param_space,
    build_fixed_polyfit_ma_switch_param_space,
    build_fixed_polyfit_param_space,
)
from .reporting import (
    export_trade_records_csv,
    generate_interactive_backtest_report_html,
    plot_annual_return_comparison,
    plot_daily_cumulative_return_comparison,
    plot_multi_strategy_cumulative_comparison,
    print_daily_cumulative_returns_with_signals,
    summarize_backtest_metrics,
    write_window_comparison_summary_markdown,
)
from .scanning import scan_ma_parameters, scan_parameters, scan_polyfit_ma_stoploss_nextday_guard_parameters, scan_polyfit_ma_switch_parameters


def build_rolling_splits_3y1y(data: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(data.index.year.unique().tolist())
    if len(years) < 4:
        raise ValueError("至少需要 4 个自然年数据才能进行 3年训练+1年验证")

    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for index in range(len(years) - 3):
        train_years = years[index : index + 3]
        val_year = years[index + 3]

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


def _resolve_initial_position(
    previous_position: float | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict,
    feature_builder,
    feature_args: tuple[int, int, int],
    deviation_col: str,
) -> float:
    if previous_position is not None:
        return float(np.clip(previous_position, 0.0, 1.0))

    featured = feature_builder(combine_train_val_data(train_df, val_df), *feature_args)
    probe = featured.loc[(featured.index >= val_df.index[0]) & (featured.index <= val_df.index[-1])].copy()
    return initial_position_from_deviation(
        probe,
        deviation_col=deviation_col,
        base_grid_pct=float(params["base_grid_pct"]),
        min_signal_strength=float(params["min_signal_strength"]),
        max_grid_levels=int(params["max_grid_levels"]),
        position_size=float(params["position_size"]),
    )


def _daily_plot_price_series(data: pd.DataFrame) -> pd.Series:
    if "Open" in data.columns:
        return data["Open"]
    return data["Close"]


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
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(build_rolling_splits_3y1y(base_data), start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== 3年训练1年验证 {window_idx} =====\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        if fixed_params is None:
            best_params, scan_df = scan_parameters(train_df, param_space, max_evals=max_evals, random_seed=random_seed + window_idx)
        else:
            best_params = dict(fixed_params)
            scan_df = pd.DataFrame([best_params])

        val_stats, val_data = run_strategy_backtest(val_df, best_params, warmup_data=train_df)

        if generate_artifacts:
            annual_png = reports / f"wf3y1y_{window_idx:02d}_annual_return_comparison.png"
            daily_png = reports / f"wf3y1y_{window_idx:02d}_daily_cumulative_return_comparison.png"
            annual_csv = reports / f"wf3y1y_{window_idx:02d}_annual_return_comparison.csv"
            daily_csv = reports / f"wf3y1y_{window_idx:02d}_daily_cumulative_return_comparison.csv"
            scan_csv = reports / f"wf3y1y_{window_idx:02d}_train_scan_top50.csv"
            trades_csv = reports / f"wf3y1y_{window_idx:02d}_trade_records.csv"

            annual_df = plot_annual_return_comparison(
                strategy_equity_curve=val_stats["_equity_curve"],
                benchmark_close=val_data["Close"],
                title=f"3年训练1年验证窗口{window_idx}: 策略 vs 长期持有（年度独立收益）",
                output_path=annual_png,
            )
            daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=val_stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(val_data),
                title=f"3年训练1年验证窗口{window_idx}: 策略 vs 长期持有（每日累计收益）",
                output_path=daily_png,
                trades=val_stats["_trades"],
                strategy_obj=val_stats.get("_strategy"),
            )
            if print_daily:
                print_daily_cumulative_returns_with_signals(
                    daily_df,
                    val_stats["_trades"],
                    strategy_obj=val_stats.get("_strategy"),
                    label=f"3年训练1年验证窗口{window_idx}",
                    max_rows=daily_max_rows,
                )
            annual_df.to_csv(annual_csv, index=True, encoding="utf-8-sig")
            daily_df.to_csv(daily_csv, index=True, encoding="utf-8-sig")
            scan_df.head(50).to_csv(scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                val_stats["_trades"],
                trades_csv,
                bt_data=val_data,
                equity_curve=val_stats["_equity_curve"],
                strategy_name="polyfit",
                params=best_params,
                native_reason_records=getattr(val_stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=val_stats.get("_strategy"),
            )
            generate_interactive_backtest_report_html(
                bt_data=val_data,
                strategy_equity_curve=val_stats["_equity_curve"],
                output_path=reports / f"wf3y1y_{window_idx:02d}_interactive_report.html",
                title=f"3年训练1年验证窗口{window_idx} 交互回测报告",
                trades=val_stats.get("_trades"),
                strategy_obj=val_stats.get("_strategy"),
                baseline_series=val_data["PolyBasePred"] if "PolyBasePred" in val_data.columns else None,
                baseline_label="Polyfit基准累计收益",
            )

        metrics = summarize_backtest_metrics(val_stats, val_data["Close"])
        rows.append(
            {
                "窗口": window_idx,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{name: best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **metrics,
            }
        )

        print(f"验证窗口{window_idx}总收益率: {metrics['总收益率'] * 100:.2f}%")
        print(f"验证窗口{window_idx}超额收益: {metrics['超额收益'] * 100:.2f}%")
        print(f"验证窗口{window_idx}最大回撤: {metrics['最大回撤'] * 100:.2f}%")

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
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    prefix = report_prefix or f"wf3y1y_ma{int(ma_window_days):02d}"
    rows: list[dict] = []
    prev_ending_position: float | None = None
    annual_images: list[Path] = []
    daily_images: list[Path] = []

    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(build_rolling_splits_3y1y(base_data), start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== MA固定周期 3年训练1年验证 {window_idx} =====\n"
            f"MA周期: {int(ma_window_days)}\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        best_params, scan_df = scan_ma_parameters(train_df, ma_param_space, max_evals=max_evals, random_seed=random_seed + window_idx)
        initial_position = _resolve_initial_position(
            prev_ending_position,
            train_df,
            val_df,
            best_params,
            add_ma_strategy_features,
            (int(best_params["ma_window_days"]), int(best_params["trend_window_days"]), int(best_params["vol_window_days"])),
            "MADevPct",
        )

        stats, val_data = run_ma_strategy_backtest(val_df, best_params, warmup_data=train_df, initial_position=initial_position)
        prev_ending_position = extract_ending_position(stats)

        if generate_artifacts:
            annual_png = reports / f"{prefix}_{window_idx:02d}_annual_return_comparison.png"
            daily_png = reports / f"{prefix}_{window_idx:02d}_daily_cumulative_return_comparison.png"
            annual_csv = reports / f"{prefix}_{window_idx:02d}_annual_return_comparison.csv"
            daily_csv = reports / f"{prefix}_{window_idx:02d}_daily_cumulative_return_comparison.csv"
            scan_csv = reports / f"{prefix}_{window_idx:02d}_train_scan_top50.csv"
            trades_csv = reports / f"{prefix}_{window_idx:02d}_trade_records.csv"

            annual_df = plot_annual_return_comparison(stats["_equity_curve"], val_data["Close"], f"窗口{window_idx}: MA{int(ma_window_days)} 策略 vs 长期持有（年度独立收益）", annual_png)
            daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(val_data),
                title=f"窗口{window_idx}: MA{int(ma_window_days)} 策略 vs 长期持有（每日累计收益）",
                output_path=daily_png,
                trades=stats["_trades"],
                strategy_obj=stats.get("_strategy"),
                baseline_series=val_data["MABase"],
                baseline_label="MA基准累计收益",
            )
            if print_daily:
                print_daily_cumulative_returns_with_signals(
                    daily_df,
                    stats["_trades"],
                    strategy_obj=stats.get("_strategy"),
                    label=f"MA{int(ma_window_days)} 窗口{window_idx}",
                    max_rows=daily_max_rows,
                )
            annual_df.to_csv(annual_csv, index=True, encoding="utf-8-sig")
            daily_df.to_csv(daily_csv, index=True, encoding="utf-8-sig")
            scan_df.head(50).to_csv(scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                stats["_trades"],
                trades_csv,
                bt_data=val_data,
                equity_curve=stats["_equity_curve"],
                strategy_name="ma",
                params=best_params,
                native_reason_records=getattr(stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=stats.get("_strategy"),
            )
            generate_interactive_backtest_report_html(
                bt_data=val_data,
                strategy_equity_curve=stats["_equity_curve"],
                output_path=reports / f"{prefix}_{window_idx:02d}_interactive_report.html",
                title=f"窗口{window_idx}: MA{int(ma_window_days)} 交互回测报告",
                trades=stats.get("_trades"),
                strategy_obj=stats.get("_strategy"),
                baseline_series=val_data["MABase"] if "MABase" in val_data.columns else None,
                baseline_label="MA基准累计收益",
            )
            annual_images.append(annual_png)
            daily_images.append(daily_png)

        metrics = summarize_backtest_metrics(stats, val_data["Close"])
        rows.append(
            {
                "MA周期": int(ma_window_days),
                "窗口": window_idx,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"ma_{name}": best_params[name] for name in MA_SCAN_PARAM_NAMES},
                **{f"ma_{key}": value for key, value in metrics.items()},
                "ma_初始仓位": initial_position,
                "ma_结束仓位": prev_ending_position,
            }
        )

        print(f"窗口{window_idx} MA{int(ma_window_days)} 策略总收益率: {metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{window_idx} MA{int(ma_window_days)} 策略超额收益: {metrics['超额收益'] * 100:.2f}%")
        print(f"窗口{window_idx} MA{int(ma_window_days)} 策略最大回撤: {metrics['最大回撤'] * 100:.2f}%")

    if generate_artifacts and len(daily_images) == 4:
        write_window_comparison_summary_markdown(
            reports / f"{prefix}_comparison_summary.md",
            title=f"MA{int(ma_window_days)} 固定周期 3年训练1年验证图表汇总",
            sections=[("年度独立收益对比图", annual_images), ("每日累计收益对比图", daily_images)],
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
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    prefix = report_prefix or f"wf3y1y_polyfitfit{int(fit_window_days):03d}"
    rows: list[dict] = []
    prev_ending_position: float | None = None
    annual_images: list[Path] = []
    daily_images: list[Path] = []

    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(build_rolling_splits_3y1y(base_data), start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== Polyfit固定拟合窗 3年训练1年验证 {window_idx} =====\n"
            f"拟合窗口: {int(fit_window_days)}\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        best_params, scan_df = scan_parameters(train_df, polyfit_param_space, max_evals=max_evals, random_seed=random_seed + window_idx)
        initial_position = _resolve_initial_position(
            prev_ending_position,
            train_df,
            val_df,
            best_params,
            add_strategy_features,
            (int(best_params["fit_window_days"]), int(best_params["trend_window_days"]), int(best_params["vol_window_days"])),
            "PolyDevPct",
        )

        stats, val_data = run_strategy_backtest(val_df, best_params, warmup_data=train_df, initial_position=initial_position)
        prev_ending_position = extract_ending_position(stats)

        if generate_artifacts:
            annual_png = reports / f"{prefix}_{window_idx:02d}_annual_return_comparison.png"
            daily_png = reports / f"{prefix}_{window_idx:02d}_daily_cumulative_return_comparison.png"
            annual_csv = reports / f"{prefix}_{window_idx:02d}_annual_return_comparison.csv"
            daily_csv = reports / f"{prefix}_{window_idx:02d}_daily_cumulative_return_comparison.csv"
            scan_csv = reports / f"{prefix}_{window_idx:02d}_train_scan_top50.csv"
            trades_csv = reports / f"{prefix}_{window_idx:02d}_trade_records.csv"

            annual_df = plot_annual_return_comparison(stats["_equity_curve"], val_data["Close"], f"窗口{window_idx}: Polyfit拟合窗{int(fit_window_days)} 策略 vs 长期持有（年度独立收益）", annual_png)
            daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(val_data),
                title=f"窗口{window_idx}: Polyfit拟合窗{int(fit_window_days)} 策略 vs 长期持有（每日累计收益）",
                output_path=daily_png,
                trades=stats["_trades"],
                strategy_obj=stats.get("_strategy"),
                baseline_series=val_data["PolyBasePred"],
                baseline_label="Polyfit基准累计收益",
            )
            if print_daily:
                print_daily_cumulative_returns_with_signals(
                    daily_df,
                    stats["_trades"],
                    strategy_obj=stats.get("_strategy"),
                    label=f"Polyfit拟合窗{int(fit_window_days)} 窗口{window_idx}",
                    max_rows=daily_max_rows,
                )
            annual_df.to_csv(annual_csv, index=True, encoding="utf-8-sig")
            daily_df.to_csv(daily_csv, index=True, encoding="utf-8-sig")
            scan_df.head(50).to_csv(scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                stats["_trades"],
                trades_csv,
                bt_data=val_data,
                equity_curve=stats["_equity_curve"],
                strategy_name="polyfit",
                params=best_params,
                native_reason_records=getattr(stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=stats.get("_strategy"),
            )
            generate_interactive_backtest_report_html(
                bt_data=val_data,
                strategy_equity_curve=stats["_equity_curve"],
                output_path=reports / f"{prefix}_{window_idx:02d}_interactive_report.html",
                title=f"窗口{window_idx}: Polyfit拟合窗{int(fit_window_days)} 交互回测报告",
                trades=stats.get("_trades"),
                strategy_obj=stats.get("_strategy"),
                baseline_series=val_data["PolyBasePred"] if "PolyBasePred" in val_data.columns else None,
                baseline_label="Polyfit基准累计收益",
            )
            annual_images.append(annual_png)
            daily_images.append(daily_png)

        metrics = summarize_backtest_metrics(stats, val_data["Close"])
        rows.append(
            {
                "拟合窗口": int(fit_window_days),
                "窗口": window_idx,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"polyfit_{name}": best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **{f"polyfit_{key}": value for key, value in metrics.items()},
                "polyfit_初始仓位": initial_position,
                "polyfit_结束仓位": prev_ending_position,
            }
        )

        print(f"窗口{window_idx} Polyfit拟合窗{int(fit_window_days)} 总收益率: {metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{window_idx} Polyfit拟合窗{int(fit_window_days)} 超额收益: {metrics['超额收益'] * 100:.2f}%")
        print(f"窗口{window_idx} Polyfit拟合窗{int(fit_window_days)} 最大回撤: {metrics['最大回撤'] * 100:.2f}%")

    if generate_artifacts and len(daily_images) == 4:
        write_window_comparison_summary_markdown(
            reports / f"{prefix}_comparison_summary.md",
            title=f"Polyfit拟合窗{int(fit_window_days)} 固定周期 3年训练1年验证图表汇总",
            sections=[("年度独立收益对比图", annual_images), ("每日累计收益对比图", daily_images)],
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
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    detailed_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict] = []

    for idx, fit_window in enumerate(fit_windows):
        detail_df = run_polyfit_walk_forward_validation_3y1y(
            base_data,
            polyfit_param_space=build_fixed_polyfit_param_space(int(fit_window), polyfit_param_space),
            fit_window_days=int(fit_window),
            max_evals=max_evals,
            random_seed=random_seed + idx * 10_000,
            generate_artifacts=generate_artifacts,
            print_daily=print_daily,
            daily_max_rows=daily_max_rows,
            reports_dir=reports,
            report_prefix=f"wf3y1y_polyfitfit{int(fit_window):03d}",
        )
        detail_df.to_csv(reports / f"wf3y1y_polyfitfit{int(fit_window):03d}_strategy_summary.csv", index=False, encoding="utf-8-sig")
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
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    detailed_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict] = []

    for idx, ma_window in enumerate(ma_windows):
        detail_df = run_ma_walk_forward_validation_3y1y(
            base_data,
            ma_param_space=build_fixed_ma_param_space(int(ma_window), ma_param_space),
            ma_window_days=int(ma_window),
            max_evals=max_evals,
            random_seed=random_seed + idx * 10_000,
            generate_artifacts=generate_artifacts,
            print_daily=print_daily,
            daily_max_rows=daily_max_rows,
            reports_dir=reports,
            report_prefix=f"wf3y1y_ma{int(ma_window):02d}",
        )
        detail_df.to_csv(reports / f"wf3y1y_ma{int(ma_window):02d}_strategy_summary.csv", index=False, encoding="utf-8-sig")
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
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    polyfit_prev_ending_position: float | None = None
    ma_prev_ending_position: float | None = None

    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(build_rolling_splits_3y1y(base_data), start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== 双策略 3年训练1年验证 {window_idx} =====\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        if polyfit_fixed_params is None:
            polyfit_best_params, polyfit_scan_df = scan_parameters(train_df, polyfit_param_space, max_evals=max_evals, random_seed=random_seed + window_idx)
        else:
            polyfit_best_params = dict(polyfit_fixed_params)
            polyfit_scan_df = pd.DataFrame([polyfit_best_params])

        if ma_fixed_params is None:
            ma_best_params, ma_scan_df = scan_ma_parameters(train_df, ma_param_space, max_evals=max_evals, random_seed=random_seed + 5_000 + window_idx)
        else:
            ma_best_params = dict(ma_fixed_params)
            ma_scan_df = pd.DataFrame([ma_best_params])

        polyfit_initial_position = _resolve_initial_position(
            polyfit_prev_ending_position,
            train_df,
            val_df,
            polyfit_best_params,
            add_strategy_features,
            (int(polyfit_best_params["fit_window_days"]), int(polyfit_best_params["trend_window_days"]), int(polyfit_best_params["vol_window_days"])),
            "PolyDevPct",
        )
        ma_initial_position = _resolve_initial_position(
            ma_prev_ending_position,
            train_df,
            val_df,
            ma_best_params,
            add_ma_strategy_features,
            (int(ma_best_params["ma_window_days"]), int(ma_best_params["trend_window_days"]), int(ma_best_params["vol_window_days"])),
            "MADevPct",
        )

        polyfit_stats, polyfit_val_data = run_strategy_backtest(val_df, polyfit_best_params, warmup_data=train_df, initial_position=polyfit_initial_position)
        ma_stats, ma_val_data = run_ma_strategy_backtest(val_df, ma_best_params, warmup_data=train_df, initial_position=ma_initial_position)
        polyfit_prev_ending_position = extract_ending_position(polyfit_stats)
        ma_prev_ending_position = extract_ending_position(ma_stats)

        if generate_artifacts:
            polyfit_daily_png = reports / f"wf3y1y_{window_idx:02d}_polyfit_daily_cumulative_return_comparison.png"
            ma_daily_png = reports / f"wf3y1y_{window_idx:02d}_ma_daily_cumulative_return_comparison.png"
            pair_daily_png = reports / f"wf3y1y_{window_idx:02d}_strategy_pair_daily_comparison.png"
            polyfit_daily_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_daily_cumulative_return_comparison.csv"
            ma_daily_csv = reports / f"wf3y1y_{window_idx:02d}_ma_daily_cumulative_return_comparison.csv"
            pair_daily_csv = reports / f"wf3y1y_{window_idx:02d}_strategy_pair_daily_comparison.csv"
            polyfit_scan_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_train_scan_top50.csv"
            ma_scan_csv = reports / f"wf3y1y_{window_idx:02d}_ma_train_scan_top50.csv"
            polyfit_trades_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_trade_records.csv"
            ma_trades_csv = reports / f"wf3y1y_{window_idx:02d}_ma_trade_records.csv"

            polyfit_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(polyfit_val_data),
                title=f"窗口{window_idx}: 回归策略 vs 长期持有（每日累计收益）",
                output_path=polyfit_daily_png,
                trades=polyfit_stats["_trades"],
                strategy_obj=polyfit_stats.get("_strategy"),
                baseline_series=polyfit_val_data["PolyBasePred"],
                baseline_label="回归基准累计收益",
            )
            ma_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=ma_stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(ma_val_data),
                title=f"窗口{window_idx}: MA 基准策略 vs 长期持有（每日累计收益）",
                output_path=ma_daily_png,
                trades=ma_stats["_trades"],
                strategy_obj=ma_stats.get("_strategy"),
                baseline_series=ma_val_data["MABase"],
                baseline_label="MA基准累计收益",
            )
            pair_daily_df = plot_multi_strategy_cumulative_comparison(
                strategy_curves={
                    "回归策略累计收益": polyfit_stats["_equity_curve"],
                    "MA基准策略累计收益": ma_stats["_equity_curve"],
                },
                benchmark_close=val_df["Close"],
                title=f"窗口{window_idx}: 回归策略 vs MA基准策略 vs 长期持有",
                output_path=pair_daily_png,
            )
            polyfit_daily_df.to_csv(polyfit_daily_csv, index=True, encoding="utf-8-sig")
            ma_daily_df.to_csv(ma_daily_csv, index=True, encoding="utf-8-sig")
            pair_daily_df.to_csv(pair_daily_csv, index=True, encoding="utf-8-sig")
            polyfit_scan_df.head(50).to_csv(polyfit_scan_csv, index=False, encoding="utf-8-sig")
            ma_scan_df.head(50).to_csv(ma_scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                polyfit_stats["_trades"],
                polyfit_trades_csv,
                bt_data=polyfit_val_data,
                equity_curve=polyfit_stats["_equity_curve"],
                strategy_name="polyfit",
                params=polyfit_best_params,
                native_reason_records=getattr(polyfit_stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=polyfit_stats.get("_strategy"),
            )
            export_trade_records_csv(
                ma_stats["_trades"],
                ma_trades_csv,
                bt_data=ma_val_data,
                equity_curve=ma_stats["_equity_curve"],
                strategy_name="ma",
                params=ma_best_params,
                native_reason_records=getattr(ma_stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=ma_stats.get("_strategy"),
            )
            generate_interactive_backtest_report_html(
                bt_data=polyfit_val_data,
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                output_path=reports / f"wf3y1y_{window_idx:02d}_polyfit_interactive_report.html",
                title=f"窗口{window_idx}: 回归策略 交互回测报告",
                trades=polyfit_stats.get("_trades"),
                strategy_obj=polyfit_stats.get("_strategy"),
                baseline_series=polyfit_val_data["PolyBasePred"] if "PolyBasePred" in polyfit_val_data.columns else None,
                baseline_label="Polyfit基准累计收益",
            )
            generate_interactive_backtest_report_html(
                bt_data=ma_val_data,
                strategy_equity_curve=ma_stats["_equity_curve"],
                output_path=reports / f"wf3y1y_{window_idx:02d}_ma_interactive_report.html",
                title=f"窗口{window_idx}: MA策略 交互回测报告",
                trades=ma_stats.get("_trades"),
                strategy_obj=ma_stats.get("_strategy"),
                baseline_series=ma_val_data["MABase"] if "MABase" in ma_val_data.columns else None,
                baseline_label="MA基准累计收益",
            )

        polyfit_metrics = summarize_backtest_metrics(polyfit_stats, polyfit_val_data["Close"])
        ma_metrics = summarize_backtest_metrics(ma_stats, ma_val_data["Close"])
        rows.append(
            {
                "窗口": window_idx,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"polyfit_{name}": polyfit_best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **{f"ma_{name}": ma_best_params[name] for name in MA_SCAN_PARAM_NAMES},
                **{f"polyfit_{key}": value for key, value in polyfit_metrics.items()},
                **{f"ma_{key}": value for key, value in ma_metrics.items()},
                "polyfit_初始仓位": polyfit_initial_position,
                "polyfit_结束仓位": polyfit_prev_ending_position,
                "ma_初始仓位": ma_initial_position,
                "ma_结束仓位": ma_prev_ending_position,
                "MA优于回归_超额收益差": ma_metrics["超额收益"] - polyfit_metrics["超额收益"],
                "MA优于回归_总收益差": ma_metrics["总收益率"] - polyfit_metrics["总收益率"],
            }
        )

        print(f"窗口{window_idx} 回归策略总收益率: {polyfit_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{window_idx} 回归策略初始仓位: {polyfit_initial_position:.2%}，结束仓位: {polyfit_prev_ending_position:.2%}")
        print(f"窗口{window_idx} MA 基准策略总收益率: {ma_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{window_idx} MA 基准策略初始仓位: {ma_initial_position:.2%}，结束仓位: {ma_prev_ending_position:.2%}")

    return pd.DataFrame(rows)


def run_polyfit_switch_comparison_3y1y(
    base_data: pd.DataFrame,
    polyfit_param_space: dict[str, list],
    switch_param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    reports_dir: Path | None = None,
    polyfit_fixed_params: dict | None = None,
) -> pd.DataFrame:
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    polyfit_prev_ending_position: float | None = None
    switch_prev_ending_position: float | None = None
    annual_images: list[Path] = []
    daily_images: list[Path] = []

    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(build_rolling_splits_3y1y(base_data), start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== Polyfit vs 偏离度+MA切换 3年训练1年验证 {window_idx} =====\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        if polyfit_fixed_params is None:
            polyfit_best_params, polyfit_scan_df = scan_parameters(train_df, polyfit_param_space, max_evals=max_evals, random_seed=random_seed + window_idx)
        else:
            polyfit_best_params = dict(polyfit_fixed_params)
            polyfit_scan_df = pd.DataFrame([polyfit_best_params])

        fixed_switch_param_space = build_fixed_polyfit_ma_switch_param_space(polyfit_best_params, switch_param_space)
        switch_best_params, switch_scan_df = scan_polyfit_ma_switch_parameters(
            train_df,
            fixed_switch_param_space,
            max_evals=max_evals,
            random_seed=random_seed + 5_000 + window_idx,
        )

        polyfit_initial_position = _resolve_initial_position(
            polyfit_prev_ending_position,
            train_df,
            val_df,
            polyfit_best_params,
            add_strategy_features,
            (int(polyfit_best_params["fit_window_days"]), int(polyfit_best_params["trend_window_days"]), int(polyfit_best_params["vol_window_days"])),
            "PolyDevPct",
        )
        switch_initial_position = _resolve_initial_position(
            switch_prev_ending_position,
            train_df,
            val_df,
            switch_best_params,
            add_strategy_features,
            (int(switch_best_params["fit_window_days"]), int(switch_best_params["trend_window_days"]), int(switch_best_params["vol_window_days"])),
            "PolyDevPct",
        )

        polyfit_stats, polyfit_val_data = run_strategy_backtest(val_df, polyfit_best_params, warmup_data=train_df, initial_position=polyfit_initial_position)
        switch_stats, switch_val_data = run_polyfit_ma_switch_backtest(val_df, switch_best_params, warmup_data=train_df, initial_position=switch_initial_position)
        polyfit_prev_ending_position = extract_ending_position(polyfit_stats)
        switch_prev_ending_position = extract_ending_position(switch_stats)

        if generate_artifacts:
            polyfit_annual_png = reports / f"wf3y1y_{window_idx:02d}_polyfit_annual_return_comparison.png"
            polyfit_daily_png = reports / f"wf3y1y_{window_idx:02d}_polyfit_daily_cumulative_return_comparison.png"
            switch_annual_png = reports / f"wf3y1y_{window_idx:02d}_switch_annual_return_comparison.png"
            switch_daily_png = reports / f"wf3y1y_{window_idx:02d}_switch_daily_cumulative_return_comparison.png"
            pair_daily_png = reports / f"wf3y1y_{window_idx:02d}_polyfit_switch_pair_daily_comparison.png"
            polyfit_annual_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_annual_return_comparison.csv"
            polyfit_daily_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_daily_cumulative_return_comparison.csv"
            switch_annual_csv = reports / f"wf3y1y_{window_idx:02d}_switch_annual_return_comparison.csv"
            switch_daily_csv = reports / f"wf3y1y_{window_idx:02d}_switch_daily_cumulative_return_comparison.csv"
            pair_daily_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_switch_pair_daily_comparison.csv"
            polyfit_scan_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_train_scan_top50.csv"
            switch_scan_csv = reports / f"wf3y1y_{window_idx:02d}_switch_train_scan_top50.csv"
            polyfit_trades_csv = reports / f"wf3y1y_{window_idx:02d}_polyfit_trade_records.csv"
            switch_trades_csv = reports / f"wf3y1y_{window_idx:02d}_switch_trade_records.csv"

            polyfit_annual_df = plot_annual_return_comparison(
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                benchmark_close=polyfit_val_data["Close"],
                title=f"窗口{window_idx}: Polyfit策略 vs 长期持有（年度独立收益）",
                output_path=polyfit_annual_png,
            )
            polyfit_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(polyfit_val_data),
                title=f"窗口{window_idx}: Polyfit策略 vs 长期持有（每日累计收益）",
                output_path=polyfit_daily_png,
                trades=polyfit_stats["_trades"],
                strategy_obj=polyfit_stats.get("_strategy"),
                baseline_series=polyfit_val_data["PolyBasePred"],
                baseline_label="Polyfit基准累计收益",
            )
            switch_annual_df = plot_annual_return_comparison(
                strategy_equity_curve=switch_stats["_equity_curve"],
                benchmark_close=switch_val_data["Close"],
                title=f"窗口{window_idx}: 偏离度+MA切换策略 vs 长期持有（年度独立收益）",
                output_path=switch_annual_png,
            )
            switch_daily_df = plot_daily_cumulative_return_comparison(
                strategy_equity_curve=switch_stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(switch_val_data),
                title=f"窗口{window_idx}: 偏离度+MA切换策略 vs 长期持有（每日累计收益）",
                output_path=switch_daily_png,
                trades=switch_stats["_trades"],
                strategy_obj=switch_stats.get("_strategy"),
                baseline_series=switch_val_data["PolyBasePred"],
                baseline_label="Polyfit基准累计收益",
            )
            pair_daily_df = plot_multi_strategy_cumulative_comparison(
                strategy_curves={
                    "Polyfit策略累计收益": polyfit_stats["_equity_curve"],
                    "偏离度+MA切换策略累计收益": switch_stats["_equity_curve"],
                },
                benchmark_close=val_df["Close"],
                title=f"窗口{window_idx}: Polyfit策略 vs 偏离度+MA切换策略 vs 长期持有",
                output_path=pair_daily_png,
            )
            polyfit_annual_df.to_csv(polyfit_annual_csv, index=True, encoding="utf-8-sig")
            polyfit_daily_df.to_csv(polyfit_daily_csv, index=True, encoding="utf-8-sig")
            switch_annual_df.to_csv(switch_annual_csv, index=True, encoding="utf-8-sig")
            switch_daily_df.to_csv(switch_daily_csv, index=True, encoding="utf-8-sig")
            pair_daily_df.to_csv(pair_daily_csv, index=True, encoding="utf-8-sig")
            polyfit_scan_df.head(50).to_csv(polyfit_scan_csv, index=False, encoding="utf-8-sig")
            switch_scan_df.head(50).to_csv(switch_scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                polyfit_stats["_trades"],
                polyfit_trades_csv,
                bt_data=polyfit_val_data,
                equity_curve=polyfit_stats["_equity_curve"],
                strategy_name="polyfit",
                params=polyfit_best_params,
                native_reason_records=getattr(polyfit_stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=polyfit_stats.get("_strategy"),
            )
            export_trade_records_csv(
                switch_stats["_trades"],
                switch_trades_csv,
                bt_data=switch_val_data,
                equity_curve=switch_stats["_equity_curve"],
                strategy_name="polyfit_switch",
                params=switch_best_params,
                native_reason_records=getattr(switch_stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=switch_stats.get("_strategy"),
            )
            generate_interactive_backtest_report_html(
                bt_data=polyfit_val_data,
                strategy_equity_curve=polyfit_stats["_equity_curve"],
                output_path=reports / f"wf3y1y_{window_idx:02d}_polyfit_interactive_report.html",
                title=f"窗口{window_idx}: Polyfit策略 交互回测报告",
                trades=polyfit_stats.get("_trades"),
                strategy_obj=polyfit_stats.get("_strategy"),
                baseline_series=polyfit_val_data["PolyBasePred"] if "PolyBasePred" in polyfit_val_data.columns else None,
                baseline_label="Polyfit基准累计收益",
            )
            generate_interactive_backtest_report_html(
                bt_data=switch_val_data,
                strategy_equity_curve=switch_stats["_equity_curve"],
                output_path=reports / f"wf3y1y_{window_idx:02d}_switch_interactive_report.html",
                title=f"窗口{window_idx}: 偏离度+MA切换策略 交互回测报告",
                trades=switch_stats.get("_trades"),
                strategy_obj=switch_stats.get("_strategy"),
                baseline_series=switch_val_data["PolyBasePred"] if "PolyBasePred" in switch_val_data.columns else None,
                baseline_label="Polyfit基准累计收益",
            )
            annual_images.append(pair_daily_png)
            daily_images.append(switch_daily_png)

        polyfit_metrics = summarize_backtest_metrics(polyfit_stats, polyfit_val_data["Close"])
        switch_metrics = summarize_backtest_metrics(switch_stats, switch_val_data["Close"])
        rows.append(
            {
                "窗口": window_idx,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"polyfit_{name}": polyfit_best_params[name] for name in POLYFIT_SCAN_PARAM_NAMES},
                **{f"switch_{name}": switch_best_params[name] for name in POLYFIT_MA_SWITCH_SCAN_PARAM_NAMES},
                **{f"polyfit_{key}": value for key, value in polyfit_metrics.items()},
                **{f"switch_{key}": value for key, value in switch_metrics.items()},
                "polyfit_初始仓位": polyfit_initial_position,
                "polyfit_结束仓位": polyfit_prev_ending_position,
                "switch_初始仓位": switch_initial_position,
                "switch_结束仓位": switch_prev_ending_position,
                "switch优于polyfit_超额收益差": switch_metrics["超额收益"] - polyfit_metrics["超额收益"],
                "switch优于polyfit_总收益差": switch_metrics["总收益率"] - polyfit_metrics["总收益率"],
            }
        )

        print(f"窗口{window_idx} Polyfit策略总收益率: {polyfit_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{window_idx} Polyfit策略初始仓位: {polyfit_initial_position:.2%}，结束仓位: {polyfit_prev_ending_position:.2%}")
        print(f"窗口{window_idx} 偏离度+MA切换策略总收益率: {switch_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{window_idx} 偏离度+MA切换策略初始仓位: {switch_initial_position:.2%}，结束仓位: {switch_prev_ending_position:.2%}")
        print(
            f"窗口{window_idx} 最优切换参数: N={int(switch_best_params['flat_wait_days'])}, "
            f"M1={float(switch_best_params['switch_deviation_m1']):.2%}, "
            f"M2={float(switch_best_params['switch_deviation_m2']):.2%}, "
            f"MA={int(switch_best_params['switch_fast_ma_window'])}/{int(switch_best_params['switch_slow_ma_window'])}"
        )

    if generate_artifacts and len(annual_images) == 4 and len(daily_images) == 4:
        write_window_comparison_summary_markdown(
            reports / "wf3y1y_polyfit_switch_comparison_summary.md",
            title="Polyfit vs 偏离度+MA切换策略 3年训练1年验证图表汇总",
            sections=[("窗口对比图", annual_images), ("切换策略每日累计收益图", daily_images)],
        )

    return pd.DataFrame(rows)


def run_polyfit_stoploss_nextday_guard_switch_comparison_3y1y(
    base_data: pd.DataFrame,
    switch_param_space: dict[str, list],
    guard_switch_param_space: dict[str, list],
    max_evals: int = 800,
    random_seed: int = 42,
    generate_artifacts: bool = True,
    reports_dir: Path | None = None,
    polyfit_fixed_params: dict | None = None,
) -> pd.DataFrame:
    reports = reports_dir or (Path(__file__).resolve().parent.parent / "reports")
    if generate_artifacts:
        reports.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    switch_prev_ending_position: float | None = None
    guard_prev_ending_position: float | None = None
    annual_images: list[Path] = []
    daily_images: list[Path] = []

    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(build_rolling_splits_3y1y(base_data), start=1):
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]

        print(
            f"\n===== Switch vs 止损次日偏离度保护策略 3年训练1年验证 {window_idx} =====\n"
            f"训练区间: {train_start.date()} -> {train_end.date()}\n"
            f"验证区间: {val_start.date()} -> {val_end.date()}"
        )

        if polyfit_fixed_params is None:
            switch_best_params, switch_scan_df = scan_polyfit_ma_switch_parameters(
                train_df,
                switch_param_space,
                max_evals=max_evals,
                random_seed=random_seed + window_idx,
            )
        else:
            switch_best_params = dict(polyfit_fixed_params)
            switch_scan_df = pd.DataFrame([switch_best_params])

        fixed_guard_param_space = build_fixed_polyfit_ma_switch_param_space(switch_best_params, guard_switch_param_space)
        guard_best_params, guard_scan_df = scan_polyfit_ma_stoploss_nextday_guard_parameters(
            train_df,
            fixed_guard_param_space,
            max_evals=max_evals,
            random_seed=random_seed + 9_000 + window_idx,
        )

        switch_initial_position = _resolve_initial_position(
            switch_prev_ending_position,
            train_df,
            val_df,
            switch_best_params,
            add_strategy_features,
            (int(switch_best_params["fit_window_days"]), int(switch_best_params["trend_window_days"]), int(switch_best_params["vol_window_days"])),
            "PolyDevPct",
        )
        guard_initial_position = _resolve_initial_position(
            guard_prev_ending_position,
            train_df,
            val_df,
            guard_best_params,
            add_strategy_features,
            (int(guard_best_params["fit_window_days"]), int(guard_best_params["trend_window_days"]), int(guard_best_params["vol_window_days"])),
            "PolyDevPct",
        )

        switch_stats, switch_val_data = run_polyfit_ma_switch_backtest(
            val_df,
            switch_best_params,
            warmup_data=train_df,
            initial_position=switch_initial_position,
        )
        guard_stats, guard_val_data = run_polyfit_ma_stoploss_nextday_guard_backtest(val_df, guard_best_params, warmup_data=train_df, initial_position=guard_initial_position)
        switch_prev_ending_position = extract_ending_position(switch_stats)
        guard_prev_ending_position = extract_ending_position(guard_stats)

        if generate_artifacts:
            switch_annual_png = reports / f"wf3y1y_{window_idx:02d}_switch_annual_return_comparison.png"
            switch_daily_png = reports / f"wf3y1y_{window_idx:02d}_switch_daily_cumulative_return_comparison.png"
            guard_annual_png = reports / f"wf3y1y_{window_idx:02d}_guard_switch_annual_return_comparison.png"
            guard_daily_png = reports / f"wf3y1y_{window_idx:02d}_guard_switch_daily_cumulative_return_comparison.png"
            pair_daily_png = reports / f"wf3y1y_{window_idx:02d}_switch_guard_switch_pair_daily_comparison.png"
            switch_scan_csv = reports / f"wf3y1y_{window_idx:02d}_switch_train_scan_top50.csv"
            guard_scan_csv = reports / f"wf3y1y_{window_idx:02d}_guard_switch_train_scan_top50.csv"
            switch_trades_csv = reports / f"wf3y1y_{window_idx:02d}_switch_trade_records.csv"
            guard_trades_csv = reports / f"wf3y1y_{window_idx:02d}_guard_switch_trade_records.csv"

            plot_annual_return_comparison(
                strategy_equity_curve=switch_stats["_equity_curve"],
                benchmark_close=switch_val_data["Close"],
                title=f"窗口{window_idx}: Switch策略 vs 长期持有（年度独立收益）",
                output_path=switch_annual_png,
            )
            plot_daily_cumulative_return_comparison(
                strategy_equity_curve=switch_stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(switch_val_data),
                title=f"窗口{window_idx}: Switch策略 vs 长期持有（每日累计收益）",
                output_path=switch_daily_png,
                trades=switch_stats["_trades"],
                strategy_obj=switch_stats.get("_strategy"),
                baseline_series=switch_val_data["PolyBasePred"],
                baseline_label="Polyfit基准累计收益",
            )
            plot_annual_return_comparison(
                strategy_equity_curve=guard_stats["_equity_curve"],
                benchmark_close=guard_val_data["Close"],
                title=f"窗口{window_idx}: 止损次日偏离度保护策略 vs 长期持有（年度独立收益）",
                output_path=guard_annual_png,
            )
            plot_daily_cumulative_return_comparison(
                strategy_equity_curve=guard_stats["_equity_curve"],
                benchmark_close=_daily_plot_price_series(guard_val_data),
                title=f"窗口{window_idx}: 止损次日偏离度保护策略 vs 长期持有（每日累计收益）",
                output_path=guard_daily_png,
                trades=guard_stats["_trades"],
                strategy_obj=guard_stats.get("_strategy"),
                baseline_series=guard_val_data["PolyBasePred"],
                baseline_label="Polyfit基准累计收益",
            )
            plot_multi_strategy_cumulative_comparison(
                strategy_curves={
                    "Switch策略累计收益": switch_stats["_equity_curve"],
                    "止损次日偏离度保护策略累计收益": guard_stats["_equity_curve"],
                },
                benchmark_close=val_df["Close"],
                title=f"窗口{window_idx}: Switch策略 vs 止损次日偏离度保护策略 vs 长期持有",
                output_path=pair_daily_png,
            )
            switch_scan_df.head(50).to_csv(switch_scan_csv, index=False, encoding="utf-8-sig")
            guard_scan_df.head(50).to_csv(guard_scan_csv, index=False, encoding="utf-8-sig")
            export_trade_records_csv(
                switch_stats["_trades"],
                switch_trades_csv,
                bt_data=switch_val_data,
                equity_curve=switch_stats["_equity_curve"],
                strategy_name="polyfit_ma_switch",
                params=switch_best_params,
                native_reason_records=getattr(switch_stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=switch_stats.get("_strategy"),
            )
            export_trade_records_csv(
                guard_stats["_trades"],
                guard_trades_csv,
                bt_data=guard_val_data,
                equity_curve=guard_stats["_equity_curve"],
                strategy_name="polyfit_stoploss_nextday_guard_switch",
                params=guard_best_params,
                native_reason_records=getattr(guard_stats.get("_strategy"), "trade_reason_records", None),
                strategy_obj=guard_stats.get("_strategy"),
            )
            generate_interactive_backtest_report_html(
                bt_data=switch_val_data,
                strategy_equity_curve=switch_stats["_equity_curve"],
                output_path=reports / f"wf3y1y_{window_idx:02d}_switch_interactive_report.html",
                title=f"窗口{window_idx}: Switch策略 交互回测报告",
                trades=switch_stats.get("_trades"),
                strategy_obj=switch_stats.get("_strategy"),
                baseline_series=switch_val_data["PolyBasePred"] if "PolyBasePred" in switch_val_data.columns else None,
                baseline_label="Polyfit基准累计收益",
            )
            generate_interactive_backtest_report_html(
                bt_data=guard_val_data,
                strategy_equity_curve=guard_stats["_equity_curve"],
                output_path=reports / f"wf3y1y_{window_idx:02d}_guard_switch_interactive_report.html",
                title=f"窗口{window_idx}: 止损次日偏离度保护策略 交互回测报告",
                trades=guard_stats.get("_trades"),
                strategy_obj=guard_stats.get("_strategy"),
                baseline_series=guard_val_data["PolyBasePred"] if "PolyBasePred" in guard_val_data.columns else None,
                baseline_label="Polyfit基准累计收益",
            )
            annual_images.append(pair_daily_png)
            daily_images.append(guard_daily_png)

        switch_metrics = summarize_backtest_metrics(switch_stats, switch_val_data["Close"])
        guard_metrics = summarize_backtest_metrics(guard_stats, guard_val_data["Close"])
        rows.append(
            {
                "窗口": window_idx,
                "训练开始": str(train_start.date()),
                "训练结束": str(train_end.date()),
                "验证开始": str(val_start.date()),
                "验证结束": str(val_end.date()),
                **{f"switch_{name}": switch_best_params[name] for name in POLYFIT_MA_SWITCH_SCAN_PARAM_NAMES},
                **{f"guard_switch_{name}": guard_best_params[name] for name in POLYFIT_MA_SWITCH_SCAN_PARAM_NAMES},
                **{f"switch_{key}": value for key, value in switch_metrics.items()},
                **{f"guard_switch_{key}": value for key, value in guard_metrics.items()},
                "switch_初始仓位": switch_initial_position,
                "switch_结束仓位": switch_prev_ending_position,
                "guard_switch_初始仓位": guard_initial_position,
                "guard_switch_结束仓位": guard_prev_ending_position,
                "guard_switch优于switch_超额收益差": guard_metrics["超额收益"] - switch_metrics["超额收益"],
                "guard_switch优于switch_总收益差": guard_metrics["总收益率"] - switch_metrics["总收益率"],
            }
        )

        print(f"窗口{window_idx} Switch策略总收益率: {switch_metrics['总收益率'] * 100:.2f}%")
        print(f"窗口{window_idx} 止损次日偏离度保护策略总收益率: {guard_metrics['总收益率'] * 100:.2f}%")

    if generate_artifacts and len(annual_images) == 4 and len(daily_images) == 4:
        write_window_comparison_summary_markdown(
            reports / "wf3y1y_polyfit_stoploss_nextday_guard_switch_comparison_summary.md",
            title="Switch vs 止损次日偏离度保护策略 3年训练1年验证图表汇总",
            sections=[("窗口对比图", annual_images), ("止损次日偏离度保护策略每日累计收益图", daily_images)],
        )

    return pd.DataFrame(rows)