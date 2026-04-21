import pandas as pd
import os

paths = [
    "reports/wf3y1y_polyfit_ma_strategy_summary.csv",
    "reports/global_best_switch_ma20_60_hold45_seed342/wf3y1y_polyfit_switch_strategy_summary.csv",
    "reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv"
]

for p in paths:
    if os.path.exists(p):
        print(f"\nFile: {p}")
        df = pd.read_csv(p)
        print("Columns:", df.columns.tolist())
        print("First 3 rows:")
        pd.set_option('display.max_columns', None)
        print(df.head(3))

# Final table
df1 = pd.read_csv(paths[0])
df2 = pd.read_csv(paths[1])
df3 = pd.read_csv(paths[2])

stats = {
    "Polyfit": df1["polyfit_总收益率"].mean(),
    "MA": df1["ma_总收益率"].mean(),
    "Switch": df2["switch_total_return"].mean(),
    "Stoploss Nextday Guard Switch": df3["guard_switch_总收益率"].mean()
}

ranking = sorted(stats.items(), key=lambda x: x[1], reverse=True)
print("\n--- Final Strategy Comparison (Average Total Return) ---")
for i, (strat, val) in enumerate(ranking, 1):
    print(f"{i}. {strat}: {val:.4f}")

