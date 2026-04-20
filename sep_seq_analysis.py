import pandas as pd
import numpy as np

def load_trades(path):
    df = pd.read_csv(path)
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    return df

baseline_path = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv'
guard_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv'

df_b = load_trades(baseline_path)
df_g = load_trades(guard_path)

sep_b = df_b[df_b['EntryTime'].dt.month == 9]
sep_g = df_g[df_g['EntryTime'].dt.month == 9]

print("--- September Baseline Sequence ---")
print(sep_b[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']])
print(f"Total Sep PnL Baseline: {sep_b['PnL'].sum():.2f}")

print("\n--- September Guard Sequence ---")
print(sep_g[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']])
print(f"Total Sep PnL Guard: {sep_g['PnL'].sum():.2f}")

# Look at the last trade of Sep (the big winner)
last_b = sep_b.iloc[-1]
last_g = sep_g.iloc[-1]

print(f"\nFinal Big Trade Baseline (Entry {last_b['EntryTime'].date()}): PnL {last_b['PnL']:.2f}")
print(f"Final Big Trade Guard    (Entry {last_g['EntryTime'].date()}): PnL {last_g['PnL']:.2f}")
print(f"Difference in major move: {last_g['PnL'] - last_b['PnL']:.2f}")

# Check Feb as well
feb_b = df_b[df_b['EntryTime'].dt.month == 2]
feb_g = df_g[df_g['EntryTime'].dt.month == 2]
print("\n--- February Baseline Sequence ---")
print(feb_b[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']])
print("\n--- February Guard Sequence ---")
print(feb_g[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'ExitReason']])

