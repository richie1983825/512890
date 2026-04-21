import pandas as pd
import numpy as np
from pathlib import Path
from backtest_core.data import resolve_data_path, load_and_forward_adjust
from backtest_core.parameters import (
    build_param_space, 
    build_ma_param_space, 
    build_polyfit_ma_switch_param_space, 
    build_fixed_polyfit_ma_switch_param_space
)
from backtest_core.scanning import (
    scan_parameters, 
    scan_ma_parameters, 
    scan_polyfit_ma_switch_parameters, 
    scan_polyfit_ma_stoploss_nextday_guard_parameters
)
from backtest_core.workflows import (
    run_strategy_backtest, 
    run_ma_strategy_backtest, 
    run_polyfit_ma_switch_backtest, 
    run_polyfit_ma_stoploss_nextday_guard_backtest,
    summarize_backtest_metrics
)

# 1) Load data
data_path = resolve_data_path()
full_df = load_and_forward_adjust(data_path)

# 2) Split data
train_df = full_df[full_df.index.year == 2025]
val_df = full_df[full_df.index.year == 2026]

if len(train_df) == 0 or len(val_df) == 0:
    years = sorted(full_df.index.year.unique())
    if len(years) >= 2:
        train_year = years[-2]
        val_year = years[-1]
        train_df = full_df[full_df.index.year == train_year]
        val_df = full_df[full_df.index.year == val_year]

print(f"Train range: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
print(f"Val range: {val_df.index.min()} to {val_df.index.max()} ({len(val_df)} rows)")

# 3) Scan best params on train
# Polyfit
poly_best_params, _ = scan_parameters(train_df, build_param_space(), max_evals=800, random_seed=342)
# MA
ma_best_params, _ = scan_ma_parameters(train_df, build_ma_param_space(), max_evals=800, random_seed=342)
# Switch
switch_best_params, _ = scan_polyfit_ma_switch_parameters(train_df, build_polyfit_ma_switch_param_space(), max_evals=800, random_seed=342)
# Guard Switch
guard_space = build_polyfit_ma_switch_param_space() 
fixed_guard_space = build_fixed_polyfit_ma_switch_param_space(switch_best_params, guard_space)
guard_best_params, _ = scan_polyfit_ma_stoploss_nextday_guard_parameters(train_df, fixed_guard_space, max_evals=360, random_seed=9342)

# 4) Run validation backtests
init_pos = 0
poly_stats, poly_val = run_strategy_backtest(val_df, poly_best_params, warmup_data=train_df, initial_position=init_pos)
ma_stats, ma_val = run_ma_strategy_backtest(val_df, ma_best_params, warmup_data=train_df, initial_position=init_pos)
switch_stats, switch_val = run_polyfit_ma_switch_backtest(val_df, switch_best_params, warmup_data=train_df, initial_position=init_pos)
guard_stats, guard_val = run_polyfit_ma_stoploss_nextday_guard_backtest(val_df, guard_best_params, warmup_data=train_df, initial_position=init_pos)

# 5) Compute metrics
results = []
for name, stats, vdata, params in [
    ("Polyfit", poly_stats, poly_val, poly_best_params),
    ("MA", ma_stats, ma_val, ma_best_params),
    ("Switch", switch_stats, switch_val, switch_best_params),
    ("Guard Switch", guard_stats, guard_val, guard_best_params)
]:
    m = summarize_backtest_metrics(stats, vdata['Close'])
    results.append({
        "Strategy": name,
        "Train Dates": f"{train_df.index.min().date()} to {train_df.index.max().date()}",
        "Val Dates": f"{val_df.index.min().date()} to {val_df.index.max().date()}",
        "Total Return": m['总收益率'],
        "Max Drawdown": m['最大回撤'],
        "Excess Return": m['超额收益'],
        "Annual Return": m['年化收益率'],
        "Trade Count": m['交易次数'],
        "Best Params": params
    })

# 6) Print table
res_df = pd.DataFrame(results)
cols_to_print = ["Strategy", "Train Dates", "Val Dates", "Total Return", "Max Drawdown", "Excess Return", "Annual Return", "Trade Count"]
print("\nValidation Results:")
print(res_df[cols_to_print].to_string(index=False))

# 7) Print best params
print("\nBest Parameters:")
for r in results:
    print(f"{r['Strategy']}: {r['Best Params']}")

print("\nAssumption: initial_position=0 for all strategies.")

# 8) Save to CSV
output_file = "reports/train2025_validate2026_four_strategy_summary.csv"
Path("reports").mkdir(exist_ok=True)
res_df.to_csv(output_file, index=False)
print(f"\nSummary saved to {output_file}")
