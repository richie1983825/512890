import pandas as pd
df = pd.read_csv('reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv')
cols = [c for c in df.columns if 'guard_switch' in c and 'wait' in c]
print(df[cols + ['验证开始']].head())
