import pandas as pd
import os

files = {
    "MA_Polyfit": "reports/wf3y1y_polyfit_ma_strategy_summary.csv",
    "Switch": "reports/global_best_switch_ma20_60_hold45_seed342/wf3y1y_polyfit_switch_strategy_summary.csv",
    "Stoploss": "reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv"
}

final_stats = {}

# File 1: Base Polyfit and MA
if os.path.exists(files["MA_Polyfit"]):
    df1 = pd.read_csv(files["MA_Polyfit"])
    # Columns are 'polyfit_总收益率' and 'ma_总收益率'
    final_stats["Polyfit"] = df1["polyfit_总收益率"].mean()
    final_stats["MA"] = df1["ma_总收益率"].mean()

# File 2: Switch
if os.path.exists(files["Switch"]):
    df2 = pd.read_csv(files["Switch"])
    # Column is 'switch_total_return'
    final_stats["Switch"] = df2["switch_total_return"].mean()

# File 3: Guard Switch
if os.path.exists(files["Stoploss"]):
    df3 = pd.read_csv(files["Stoploss"])
    # Column is 'guard_switch_总收益率'
    final_stats["Stoploss Nextday Guard Switch"] = df3["guard_switch_总收益率"].mean()

ranking = sorted(final_stats.items(), key=lambda x: x[1], reverse=True)
print("\n--- Summary Ranking (Average Total Return) ---")
for i, (strat, val) in enumerate(ranking, 1):
    print(f"{i}. {strat}: {val:.4f}")

