import pandas as pd
import os

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# Baseline vs Guard 2024
base_2024 = load_csv('reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv')
guard_2024 = load_csv('reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv')

def count_sl(df):
    if 'ExitReason' in df.columns:
        return df['ExitReason'].str.contains('stop_loss', na=False).sum()
    return 0

print(f"2024 Trade Count: Base={len(base_2024)}, Guard={len(guard_2024)}")
print(f"2024 SL Exits: Base={count_sl(base_2024)}, Guard={count_sl(guard_2024)}")

# Let's see some exit reasons from Guard 2024
if 'ExitReason' in guard_2024.columns:
    print("\nGuard 2024 ExitReasons (top 10):")
    print(guard_2024['ExitReason'].value_counts().head(10))

# Sample comparison of 2022
base_2022 = load_csv('reports/global_best_switch_ma20_60_hold45_seed342/window_0_2022/switch_trade_records.csv')
guard_2022 = load_csv('reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_01_guard_switch_trade_records.csv')
print(f"\n2022 Trade Count: Base={len(base_2022)}, Guard={len(guard_2022)}")
print(f"2022 SL Exits: Base={count_sl(base_2022)}, Guard={count_sl(guard_2022)}")

# Check for 'flat_wait' in EntryReason (if that's where the guard is recorded)
if not guard_2022.empty and 'EntryReason' in guard_2022.columns:
    print("\nGuard 2022 EntryReason 'flat_wait' presence:")
    print(guard_2022['EntryReason'].str.contains('flat_wait', na=False).sum())

