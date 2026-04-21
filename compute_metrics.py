import pandas as pd
import numpy as np

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# Files
summary1_path = 'reports/wf3y1y_polyfit_ma_strategy_summary.csv'
summary2_path = 'reports/global_best_switch_ma20_60_hold45_seed342/wf3y1y_polyfit_switch_strategy_summary.csv'
summary3_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv'

df1 = load_csv(summary1_path)
df2 = load_csv(summary2_path)
df3 = load_csv(summary3_path)

results = {}

# 1. Polyfit (from df1)
if df1 is not None:
    results['Polyfit'] = {
        'returns': df1['polyfit_总收益率'].values,
        'mdds': df1['polyfit_最大回撤'].values
    }

# 2. MA (from df1)
if df1 is not None:
    results['MA'] = {
        'returns': df1['ma_总收益率'].values,
        'mdds': df1['ma_最大回撤'].values
    }

# 3. Switch (from df2)
if df2 is not None:
    results['Switch'] = {
        'returns': df2['switch_total_return'].values,
        'mdds': df2['switch_max_drawdown'].values
    }

# 4. Guard Switch (from df3)
if df3 is not None:
    results['Guard Switch'] = {
        'returns': df3['guard_switch_总收益率'].values,
        'mdds': df3['guard_switch_最大回撤'].values
    }

print("| Strategy | Avg Total Return | Avg Max Drawdown | Per-Window MDD |")
print("| :--- | :---: | :---: | :--- |")
for name, data in results.items():
    returns = data['returns']
    mdds = data['mdds']
    avg_return = np.mean(returns) * 100
    avg_mdd = np.mean(mdds) * 100
    mdd_list = ", ".join([f"{x*100:.2f}%" for x in mdds])
    print(f"| {name} | {avg_return:.2f}% | {avg_mdd:.2f}% | {mdd_list} |")

print("\nFallback assumptions: Metrics are derived directly from strategy summary CSVs. For 'Switch', used global_best_switch_ma20_60_hold45_seed342 report.")
