import pandas as pd
import os

base_file = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv'
guard_file = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv'

def load_and_format(path):
    df = pd.read_csv(path)
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    return df.sort_values('EntryTime')

b_df = load_and_format(base_file)
g_df = load_and_format(guard_file)

print(f"Baseline 2024 Trade Count: {len(b_df)}")
print(f"Guard 2024 Trade Count: {len(g_df)}")

# Find a common stop-loss
sl_base = b_df[b_df['ExitReason'].str.contains('stop_loss', na=False)]
if not sl_base.empty:
    print("\n--- Baseline Stop-Loss and what followed ---")
    sl_idx = sl_base.index[0]
    print(b_df.loc[sl_idx:sl_idx+3, ['EntryTime', 'ExitTime', 'ExitReason', 'EntryReason', 'EntryPrice', 'ExitPrice']])

sl_guard = g_df[g_df['ExitReason'].str.contains('stop_loss', na=False)]
if not sl_guard.empty:
    print("\n--- Guard Stop-Loss and what followed ---")
    # Finding a similar time frame in guard for comparison
    sl_time = sl_base.iloc[0]['ExitTime']
    sl_guard_nearby = g_df[g_df['ExitTime'] >= sl_time].head(4)
    print(sl_guard_nearby[['EntryTime', 'ExitTime', 'ExitReason', 'EntryReason', 'EntryPrice', 'ExitPrice']])
