import pandas as pd
from pathlib import Path
from copy import deepcopy

from backtest_core.data import load_and_forward_adjust, resolve_data_path, add_strategy_features, combine_train_val_data
from backtest_core.backtests import run_strategy_backtest, run_polyfit_ma_switch_backtest, extract_ending_position, initial_position_from_deviation
from backtest_core.parameters import build_param_space, POLYFIT_SCAN_PARAM_NAMES
from backtest_core.scanning import scan_parameters
from backtest_core.workflows import build_rolling_splits_3y1y
from backtest_core.reporting import (
    plot_annual_return_comparison, 
    plot_daily_cumulative_return_comparison, 
    plot_multi_strategy_cumulative_comparison, 
    export_trade_records_csv, 
    generate_interactive_backtest_report_html,
    summarize_backtest_metrics, 
    write_window_comparison_summary_markdown, 
    configure_chinese_font
)


def _daily_plot_price_series(data: pd.DataFrame) -> pd.Series:
    if "Open" in data.columns:
        return data["Open"]
    return data["Close"]

def run_task():
    configure_chinese_font()
    report_root = Path("reports/global_best_switch_ma20_60_hold45_seed342")
    if report_root.exists():
        import shutil
        shutil.rmtree(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    
    base_data = load_and_forward_adjust(resolve_data_path())
    if not isinstance(base_data.index, pd.DatetimeIndex):
        if 'trade_date' in base_data.columns:
            base_data = base_data.set_index('trade_date')
        elif 'date' in base_data.columns:
            base_data = base_data.set_index('date')
    base_data.index = pd.to_datetime(base_data.index)

    splits = build_rolling_splits_3y1y(base_data)

    constrained_space = build_param_space()
    constrained_space['max_holding_days'] = [45]
    
    switch_fixed_params = {
        'flat_wait_days': 8,
        'switch_deviation_m1': 0.03,
        'switch_deviation_m2': 0.02,
        'switch_fast_ma_window': 20,
        'switch_slow_ma_window': 60,
        'max_holding_days': 45
    }
    
    poly_prev_end_pos = None
    switch_prev_end_pos = None

    summary_data = []
    switch_daily_images = []
    annual_images = []
    daily_images = []
    
    for idx, split in enumerate(splits):
        train_start, train_end, val_start, val_end = split
        train_df = base_data.loc[(base_data.index >= train_start) & (base_data.index <= train_end)]
        val_df = base_data.loc[(base_data.index >= val_start) & (base_data.index <= val_end)]
        
        window_name = f"window_{idx}_{val_start.year}"
        window_dir = report_root / window_name
        window_dir.mkdir(exist_ok=True)
        
        print(f"Processing {window_name} ({val_start.date()} to {val_end.date()})...")

        poly_best_params, scan_results = scan_parameters(train_df, constrained_space, max_evals=800, random_seed=342 + idx)
        scan_results.head(50).to_csv(window_dir / "polyfit_train_scan_top50.csv", index=False)
        
        switch_params = deepcopy(poly_best_params)
        switch_params.update(switch_fixed_params)
        pd.DataFrame([switch_params]).to_csv(window_dir / "switch_train_scan_top50.csv", index=False)

        if poly_prev_end_pos is not None:
            poly_init_pos = poly_prev_end_pos
        else:
            merged = combine_train_val_data(train_df, val_df)
            feat = add_strategy_features(
                merged,
                int(poly_best_params['fit_window_days']),
                int(poly_best_params['trend_window_days']),
                int(poly_best_params['vol_window_days']),
            )
            probe = feat.loc[(feat.index >= val_start) & (feat.index <= val_end)]
            poly_init_pos = initial_position_from_deviation(
                probe,
                deviation_col='PolyDevPct',
                base_grid_pct=float(poly_best_params['base_grid_pct']),
                min_signal_strength=float(poly_best_params['min_signal_strength']),
                max_grid_levels=int(poly_best_params['max_grid_levels']),
                position_size=float(poly_best_params['position_size']),
            )
            
        if switch_prev_end_pos is not None:
            switch_init_pos = switch_prev_end_pos
        else:
            merged = combine_train_val_data(train_df, val_df)
            feat = add_strategy_features(
                merged,
                int(switch_params['fit_window_days']),
                int(switch_params['trend_window_days']),
                int(switch_params['vol_window_days']),
            )
            probe = feat.loc[(feat.index >= val_start) & (feat.index <= val_end)]
            switch_init_pos = initial_position_from_deviation(
                probe,
                deviation_col='PolyDevPct',
                base_grid_pct=float(switch_params['base_grid_pct']),
                min_signal_strength=float(switch_params['min_signal_strength']),
                max_grid_levels=int(switch_params['max_grid_levels']),
                position_size=float(switch_params['position_size']),
            )

        poly_stats, poly_val_data = run_strategy_backtest(
            val_df,
            poly_best_params,
            warmup_data=train_df,
            initial_position=poly_init_pos,
        )
        switch_stats, switch_val_data = run_polyfit_ma_switch_backtest(
            val_df,
            switch_params,
            warmup_data=train_df,
            initial_position=switch_init_pos,
        )

        poly_prev_end_pos = extract_ending_position(poly_stats)
        switch_prev_end_pos = extract_ending_position(switch_stats)

        poly_annual_png = window_dir / "polyfit_annual_return_comparison.png"
        switch_annual_png = window_dir / "switch_annual_return_comparison.png"
        switch_daily_png = window_dir / "switch_daily_cumulative_return_comparison.png"
        pair_daily_png = window_dir / "daily_return_comparison.png"

        plot_annual_return_comparison(
            strategy_equity_curve=poly_stats["_equity_curve"],
            benchmark_close=poly_val_data["Close"],
            title=f"{window_name} Polyfit 策略 vs 长期持有（年度独立收益）",
            output_path=poly_annual_png,
        )
        plot_annual_return_comparison(
            strategy_equity_curve=switch_stats["_equity_curve"],
            benchmark_close=switch_val_data["Close"],
            title=f"{window_name} Switch 策略 vs 长期持有（年度独立收益）",
            output_path=switch_annual_png,
        )
        plot_daily_cumulative_return_comparison(
            strategy_equity_curve=poly_stats["_equity_curve"],
            benchmark_close=_daily_plot_price_series(poly_val_data),
            title=f"{window_name} Polyfit 策略 vs 长期持有（每日累计收益）",
            output_path=window_dir / "polyfit_daily_cumulative_return_comparison.png",
            trades=poly_stats.get("_trades"),
            strategy_obj=poly_stats.get("_strategy"),
            baseline_series=poly_val_data["PolyBasePred"] if "PolyBasePred" in poly_val_data.columns else None,
            baseline_label="Polyfit基准累计收益",
            signal_on_benchmark_curve=True,
        )
        plot_daily_cumulative_return_comparison(
            strategy_equity_curve=switch_stats["_equity_curve"],
            benchmark_close=_daily_plot_price_series(switch_val_data),
            title=f"{window_name} Switch 策略 vs 长期持有（每日累计收益）",
            output_path=switch_daily_png,
            trades=switch_stats.get("_trades"),
            strategy_obj=switch_stats.get("_strategy"),
            baseline_series=switch_val_data["PolyBasePred"] if "PolyBasePred" in switch_val_data.columns else None,
            baseline_label="Polyfit基准累计收益",
            signal_on_benchmark_curve=True,
        )
        pair_daily_df = plot_multi_strategy_cumulative_comparison(
            strategy_curves={
                "Polyfit": poly_stats["_equity_curve"],
                "Switch": switch_stats["_equity_curve"],
            },
            benchmark_close=poly_val_data["Close"],
            title=f"{window_name} Polyfit vs Switch 每日累计收益对比",
            output_path=pair_daily_png,
        )

        poly_val_data.to_csv(window_dir / "polyfit_daily_returns.csv", index=False)
        switch_val_data.to_csv(window_dir / "switch_daily_returns.csv", index=False)

        export_trade_records_csv(
            trades=poly_stats.get("_trades"),
            output_path=window_dir / "polyfit_trade_records.csv",
            bt_data=poly_val_data,
            equity_curve=poly_stats.get("_equity_curve"),
            strategy_name="polyfit",
            params=poly_best_params,
            native_reason_records=getattr(poly_stats.get("_strategy"), "trade_reason_records", None),
            strategy_obj=poly_stats.get("_strategy"),
        )
        export_trade_records_csv(
            trades=switch_stats.get("_trades"),
            output_path=window_dir / "switch_trade_records.csv",
            bt_data=switch_val_data,
            equity_curve=switch_stats.get("_equity_curve"),
            strategy_name="polyfit_switch",
            params=switch_params,
            native_reason_records=getattr(switch_stats.get("_strategy"), "trade_reason_records", None),
            strategy_obj=switch_stats.get("_strategy"),
        )
        generate_interactive_backtest_report_html(
            bt_data=poly_val_data,
            strategy_equity_curve=poly_stats.get("_equity_curve"),
            output_path=window_dir / "polyfit_interactive_report.html",
            title=f"{window_name} Polyfit 交互回测报告",
            trades=poly_stats.get("_trades"),
            strategy_obj=poly_stats.get("_strategy"),
            baseline_series=poly_val_data["PolyBasePred"] if "PolyBasePred" in poly_val_data.columns else None,
            baseline_label="Polyfit基准累计收益",
        )
        generate_interactive_backtest_report_html(
            bt_data=switch_val_data,
            strategy_equity_curve=switch_stats.get("_equity_curve"),
            output_path=window_dir / "switch_interactive_report.html",
            title=f"{window_name} Switch 交互回测报告",
            trades=switch_stats.get("_trades"),
            strategy_obj=switch_stats.get("_strategy"),
            baseline_series=switch_val_data["PolyBasePred"] if "PolyBasePred" in switch_val_data.columns else None,
            baseline_label="Polyfit基准累计收益",
        )

        p_metrics = summarize_backtest_metrics(poly_stats, poly_val_data["Close"])
        s_metrics = summarize_backtest_metrics(switch_stats, switch_val_data["Close"])

        row = {
            'window': f"{val_start.date()} to {val_end.date()}",
            'poly_total_return': p_metrics.get('总收益率', 0.0),
            'switch_total_return': s_metrics.get('总收益率', 0.0),
            'poly_sharpe': float(poly_stats.get('Sharpe Ratio', 0.0)),
            'switch_sharpe': float(switch_stats.get('Sharpe Ratio', 0.0)),
            'poly_max_drawdown': p_metrics.get('最大回撤', 0.0),
            'switch_max_drawdown': s_metrics.get('最大回撤', 0.0),
            'poly_init_pos': poly_init_pos,
            'poly_end_pos': poly_prev_end_pos,
            'switch_init_pos': switch_init_pos,
            'switch_end_pos': switch_prev_end_pos
        }
        for p_name in POLYFIT_SCAN_PARAM_NAMES:
            row[f'poly_{p_name}'] = poly_best_params.get(p_name)
        for k, v in switch_fixed_params.items():
            row[f'switch_{k}'] = v
            
        summary_data.append(row)
        switch_daily_images.append(switch_daily_png)
        annual_images.extend([poly_annual_png, switch_annual_png])
        daily_images.append(pair_daily_png)

        pair_daily_df.to_csv(window_dir / "strategy_pair_daily_comparison.csv", index=True, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_data)
    summary_path = report_root / "wf3y1y_polyfit_switch_strategy_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    write_window_comparison_summary_markdown(
        output_path=report_root / "window_comparison_summary.md",
        title="Global Best Switch 窗口对比汇总",
        sections=[
            ("Switch 每日累计收益对比", switch_daily_images),
            ("年度收益对比", annual_images),
            ("每日累计收益对比", daily_images),
        ],
    )

    print("\nSummary Table:")
    print(summary_df[['window', 'poly_total_return', 'switch_total_return']])
    print(f"\nAverage Polyfit Total Return: {summary_df['poly_total_return'].mean():.4f}")
    print(f"Average Switch Total Return: {summary_df['switch_total_return'].mean():.4f}")
    print(f"\nReports generated in: {report_root}")
    print(f"Summary CSV: {summary_path}")

if __name__ == "__main__":
    run_task()
