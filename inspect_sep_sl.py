import pandas as pd
df_g = pd.read_csv('reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv')
print(df_g[df_g['EntryTime'].astype(str).str.contains('2024-09')][['EntryTime','ExitTime','ExitReason']])
