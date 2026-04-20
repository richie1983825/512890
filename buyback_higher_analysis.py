import pandas as pd
import glob
import os

files = glob.glob('reports/**/switch_trade_records.csv', recursive=True)

all_higher_reasons = []
immediate_total = 0
higher_total = 0
files_used = 0

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    
    required = {'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'EntryReason'}
    if not required.issubset(df.columns):
        continue
    
    if df.empty or len(df) < 2:
        continue

    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    # Sort by ExitTime then EntryTime
    df = df.sort_values(['ExitTime', 'EntryTime']).reset_index(drop=True)
    
    # Define trading dates based on unique entry dates in THIS file
    trading_dates = sorted(df['EntryTime'].unique())
    
    files_used += 1
    
    for i in range(len(df) - 1):
        exit_date = df.loc[i, 'ExitTime']
        exit_price = df.loc[i, 'ExitPrice']
        
        next_entry_date = df.loc[i+1, 'EntryTime']
        next_entry_price = df.loc[i+1, 'EntryPrice']
        next_reason = str(df.loc[i+1, 'EntryReason'])
        
        # Find next available trading date after exit_date
        future_dates = [d for d in trading_dates if d > exit_date]
        if not future_dates:
            continue
        next_avail = future_dates[0]
        
        if next_entry_date == next_avail:
            immediate_total += 1
            if next_entry_price > exit_price:
                higher_total += 1
                all_higher_reasons.append(next_reason)

reason_counts = pd.Series(all_higher_reasons).value_counts()
prefix_counts = pd.Series([r.split(';')[0] for r in all_higher_reasons]).value_counts()

print(f"files_used: {files_used}")
print(f"immediate_total: {immediate_total}")
print(f"higher_total: {higher_total}")
print(f"reason_count_sum: {reason_counts.sum()}")
print(f"prefix_count_sum: {prefix_counts.sum()}")

if reason_counts.sum() != higher_total or prefix_counts.sum() != higher_total:
    print("ERROR: count mismatch")

print("\n--- Top Full EntryReasons ---")
for r, c in reason_counts.items():
    print(f"{c}\t{r}")

print("\n--- EntryReason Prefixes ---")
for p, c in prefix_counts.items():
    print(f"{c}\t{p}")
