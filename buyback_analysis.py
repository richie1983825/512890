import pandas as pd
import glob
import os

files = glob.glob('reports/**/switch_trade_records.csv', recursive=True)

all_results = []
immediate_total = 0
higher_total = 0

reason_counts = {}
prefix_counts = {}

def get_next_trading_date(current_exit_date, all_entry_dates):
    # This is a simplification. Usually, the next trading date is the day after.
    # But since we don't have the full calendar, we assume if the next trade starts 
    # at the earliest available EntryTime strictly after current_exit_date, it's a candidate for "immediate".
    # However, many times immediate means T+1 or same day if the file allows it.
    # Looking at the requirement: "next trade row i+1 entry date equals next available trading date in same file after row i exit date".
    # Since we don't have the market calendar, we can use the sorted unique EntryTimes from the file itself as a proxy for trading dates.
    pass

for f in files:
    df = pd.read_csv(f)
    if df.empty or len(df) < 2:
        continue
    
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    df = df.sort_values('EntryTime').reset_index(drop=True)
    
    # All potential trading dates in this file (entry dates are at least trading dates)
    # We might need a better way if there are gaps. 
    # But the prompt says "next available trading date in same file".
    # Let's collect all unique entry and exit dates to form a "calendar" for this file.
    trading_dates = sorted(pd.concat([df['EntryTime'], df['ExitTime']]).unique())
    
    for i in range(len(df) - 1):
        exit_date = df.loc[i, 'ExitTime']
        exit_price = df.loc[i, 'ExitPrice']
        
        next_entry_date = df.loc[i+1, 'EntryTime']
        next_entry_price = df.loc[i+1, 'EntryPrice']
        next_reason = str(df.loc[i+1, 'EntryReason'])
        
        # Find next available trading date after exit_date in our proxy calendar
        future_dates = [d for d in trading_dates if d > exit_date]
        if not future_dates:
            continue
        next_avail = future_dates[0]
        
        if next_entry_date == next_avail:
            immediate_total += 1
            if next_entry_price > exit_price:
                higher_total += 1
            
            # Count reasons
            reason_counts[next_reason] = reason_counts.get(next_reason, 0) + 1
            
            # Count prefixes
            prefix = next_reason.split(';')[0]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

higher_ratio = higher_total / immediate_total if immediate_total > 0 else 0

print(f"immediate_total: {immediate_total}")
print(f"higher_total: {higher_total}")
print(f"higher_ratio: {higher_ratio:.4f}")

print("\n--- Full EntryReason Frequencies ---")
sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
for r, c in sorted_reasons:
    print(f"{c}\t{r}")

print("\n--- EntryReason Prefix Frequencies ---")
sorted_prefixes = sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)
for p, c in sorted_prefixes:
    print(f"{c}\t{p}")
