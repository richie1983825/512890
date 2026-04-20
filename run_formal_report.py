import pandas as pd
import numpy as np
import os
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
    summarize_backtest_metrics, 
    write_window_comparison_summary_markdown, 
    configure_chinese_font
)

def run_task():
    configure_chinese_font()
    report_root = Path("reports/global_best_switch_ma20_60_hold45_seed342")
    if report_root.exists():
        import shutil
        shutil.rmtree(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    base_data = load_and_forward_adjust(resolve_data_path())
    # The workflow.py expects the raw data with trade_date index (DatetimeIndex)
    if not isinstance(base_data.index, pd.DatetimeIndex):
        if 'trade_date' in base_data.columns:
            base_data = base_data.set_index('trade_date')
        elif 'date' in base_data.columns:
            base_data = base_data.set_index('date')
    
    splits = build_rolling_splits_3y1y(base_data)
    
    # Pre-calculate column versions for range filtering
    base_data_with_col = base_data.copy()
    if base_data_with_col.index.name != 'date':
        base_data_with_col = base_data_with_col.reset_index().rename(columns={base_data_with_col.index.name: 'date'})
    else:
        base_data_with_col = base_data_with_col.reset_index()

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
    
    for idx, split in enumerate(splits):
        train_start, train_end, val_start, val_end = split
        train_df = base_data_with_col[(base_data_with_col['date'] >= train_start) & (base_data_with_col['date'] <= train_end)]
        val_df = base_data_with_col[(base_data_with_col['date'] >= val_start) & (base_data_with_col['date'] <= val_end)]
        
        window_name = f"window_{idx}_{val_start.year}"
        window_dir = report_root / window_name
        window_dir.mkdir(exist_ok=True)
        
        print(f"Processing {window_name} ({val_start.date()} to {val_end.date()})...")
        
        scan_results = scan_parameters(train_df, constrained_space, max_evals=800, random_seed=342 + idx)
        poly_best_params = scan_results.iloc[0].to_dict()
        scan_results.head(50).to_csv(window_dir / "polyfit_train_scan_top50.csv", index=False)
        
        switch_params = deepcopy(poly_best_params)
        switch_params.update(switch_fixed_params)
        pd.DataFrame([switch_params]).to_csv(window_dir / "switch_train_scan_top50.csv", index=False)
        
        # Prepare for backtest
        train_bt = train_df.set_index('date')
        train_bt.index.name = 'trade_date'
        val_bt = val_df.set_index('date')
        val_bt.index.name = 'trade_date'

        if poly_prev_end_pos is not None:
            poly_init_pos = poly_prev_end_pos
        else:
            merged = combine_train_val_data(train_bt, val_bt)
            feat = add_strategy_features(merged)
            poly_init_pos = initial_position_from_deviation(feat, val_start)
            
        if switch_prev_end_pos is not None:
            switch_init_pos = switch_prev_end_pos
        else:
            merged = combine_train_val_data(train_bt, val_bt)
            feat = add_strategy_features(merged)
            switch_init_pos = initial_position_from_deviation(feat, val_start)
            
        poly_res = run_strategy_backtest(val_bt, poly_best_params, warmup_data=train_bt, initial_position=poly_init_pos)
        poly_stats, poly_val_data, poly_obj = poly_res[0], poly_res[1], poly_res[2]
        
        switch_res = run_polyfit_ma_switch_backtest(val_bt, switch_params, warmup_data=train_bt, initial_position=switch_init_pos)
        switch_stats, switch_val_data, switch_obj = switch_res[0], switch_res[1], switch_res[2]
        
        poly_prev_end_pos = extract_ending_position(poly_obj)
        switch_prev_end_pos = extract_ending_position(switch_obj)
        
        if poly_val_data.index.name == 'trade_date':
            poly_val_data = poly_val_data.reset_index().rename(columns={'trade_date': 'date'})
        if switch_val_data.index.name == 'trade_date':
            switch_val_data = switch_val_data.reset_index().rename(columns={'trade_date': 'date'})

        plot_annual_return_comparison(poly_val_data, switch_val_data, "Polyfit", "Switch", window_dir / "annual_return_comparison.png")
        plot_daily_cumulative_return_comparison(poly_val_data, switch_val_data, "Polyfit", "Switch", window_dir / "daily_return_comparison.png")
        
        poly_val_data.to_csv(window_dir / "polyfit_daily_returns.csv", index=False)
        switch_val_data.to_csv(window_dir / "switch_daily_returns.csv", index=False)
        
        export_trade_records_csv(poly_obj, window_dir / "polyfit_trade_records.csv")
        export_trade_records_csv(switch_obj, window_dir / "switch_trade_records.csv")
        
        p_metrics = summarize_backtest_metrics(poly_val_data, poly_obj)
        s_metrics = summarize_backtest_metrics(switch_val_data, switch_obj)
        
        def get_val(m, key_prefix):
            for k in [f'{key_prefix} (%)', f'{key_prefix}', f'Annualized {key_prefix} (%)', f'Annualized {key_prefix}']:
                if k in m:
                    v = m[k]
                    if isinstance(v, str) and '%' in v: v = float(v.strip('%')) / 100.0
                    elif isinstance(v, (int, float)):
                         if ' (%)' in k: v = v / 100.0
                    return v
            return 0.0

        row = {
            'window': f"{val_start.date()} to {val_end.date()}",
            'poly_total_return': get_val(p_metrics, 'Total Return'),
            'switch_total_return': get_val(s_metrics, 'Total Return'),
            'poly_sharpe': p_metrics.get('Sharpe Ratio', 0),
            'switch_sharpe': s_metrics.get('Sharpe Ratio', 0),
            'poly_max_drawdown': get_val(p_metrics, 'Max Drawdown'),
            'switch_max_drawdown': get_val(s_metrics, 'Max Drawdown'),
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

    summary_df = pd.DataFrame(summary_data)
    summary_path = report_root / "wf3y1y_polyfit_switch_strategy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    write_window_comparison_summary_markdown(summary_df, report_root)

    print("\nSummary Table:")
    print(summary_df[['window', 'poly_total_return', 'switch_total_return']])
    print(f"\nAverage Polyfit Total Return: {summary_df['poly_total_return'].mean():.4f}")
    print(f"Average Switch Total Return: {summary_df['switch_total_return'].mean():.4f}")
    print(f"\nReports generated in: {report_root}")
    print(f"Summary CSV: {summary_path}")

if __name__ == "__main__":
    run_task()
