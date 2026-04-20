import pandas as pd
import numpy as np

before_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary_before_profit_hold_stoploss.csv'
after_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv'

def get_stats(path):
    df = pd.read_csv(path)
    # Extract year from '验证开始'
    df['Year'] = pd.to_datetime(df['验证开始']).dt.year
    returns = df.set_index('Year')['guard_switch_总收益率']
    
    arithmetic_mean = returns.mean()
    compounded_total = (1 + returns).prod() - 1
    
    return returns, arithmetic_mean, compounded_total

res_before = get_stats(before_path)
res_after = get_stats(after_path)

years = sorted(list(set(res_before[0].index) | set(res_after[0].index)))

print(f"{'Year':<6} | {'Before':<10} | {'After':<10} | {'Diff':<10}")
print("-" * 45)
for y in years:
    b = res_before[0].get(y, np.nan)
    a = res_after[0].get(y, np.nan)
    print(f"{y:<6} | {b:>10.4f} | {a:>10.4f} | {a-b:>10.4f}")

print("-" * 45)
print(f"{'Mean':<6} | {res_before[1]:>10.4f} | {res_after[1]:>10.4f} | {res_after[1]-res_before[1]:>10.4f}")
print(f"{'Total':<6} | {res_before[2]:>10.4f} | {res_after[2]:>10.4f} | {res_after[2]-res_before[2]:>10.4f}")
