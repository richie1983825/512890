import pandas as pd
import random
import itertools
import os
import sys

# Add current directory to path so we can import from main
sys.path.append(os.getcwd())

from main import (
    load_and_forward_adjust,
    resolve_data_path,
    build_launch_param_space,
    run_launch_strategy_backtest
)
from datetime import datetime

def get_month_data(df, year, month):
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"
    
    warmup_start = (pd.to_datetime(start_date) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
    
    month_df = df.loc[start_date:pd.to_datetime(end_date) - pd.Timedelta(days=1)]
    warmup_df = df.loc[warmup_start:pd.to_datetime(start_date) - pd.Timedelta(days=1)]
    
    return month_df, warmup_df

def main():
    try:
        data_path = resolve_data_path()
        base_data = load_and_forward_adjust(data_path)
        
        param_space_dict = build_launch_param_space()
        
        keys, values = zip(*param_space_dict.items())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        sample_size = min(2500, len(all_combinations))
        random.seed(42)
        sampled_params = random.sample(all_combinations, sample_size)
        
        target_months = [(2024, 3), (2024, 9), (2025, 4), (2025, 10)]
        month_datasets = []
        for y, m in target_months:
            m_df, w_df = get_month_data(base_data, y, m)
            month_datasets.append((f"{y}-{m:02d}", m_df, w_df))
        
        best_hit_count = -1
        best_params = None
        best_results = []
        
        print(f"Sampling {sample_size} parameter sets...", flush=True)
        
        for i, params in enumerate(sampled_params):
            hit_count = 0
            current_results = []
            valid_set = True
            for name, m_df, w_df in month_datasets:
                if m_df.empty:
                    current_results.append((name, 0.0, 0))
                    continue
                
                try:
                    # Suppress output if possible, but backtesting usually prints
                    stats, _ = run_launch_strategy_backtest(m_df, params, warmup_data=w_df)
                    trades = stats.get("# Trades", 0)
                    ret = stats.get("Return [%]", 0.0)
                    
                    if trades > 0 and ret > 0:
                        hit_count += 1
                    current_results.append((name, ret, trades))
                except Exception:
                    current_results.append((name, 0.0, 0))
            
            if hit_count > best_hit_count:
                best_hit_count = hit_count
                best_params = params
                best_results = current_results
                print(f"New best hit count: {best_hit_count} at index {i}", flush=True)
                if best_hit_count == len(target_months):
                    break
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{sample_size}...", flush=True)

        print("\nBest Achievable Hit Count:", best_hit_count)
        print("Best Parameter Set:", best_params)
        print("Per-month Results:")
        for name, ret, trades in best_results:
            print(f"  {name}: Return = {ret:.2f}%, Trades = {trades}")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
