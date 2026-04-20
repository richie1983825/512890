import pandas as pd
import glob
import re
import os

files = glob.glob('reports/global_best_switch_ma20_60_hold45_seed342/**/switch_trade_records.csv', recursive=True)

stats = {
    'total_sells_with_next_trade': 0,
    'immediate_buyback_total': 0,
    'below_baseline_immediate_count': 0,
    'below_baseline_entry_less': 0,
    'below_baseline_entry_greater': 0,
    'below_baseline_entry_equal': 0,
    'files_scanned': 0
}

def extract_base(reason):
    match = re.search(r'base=([0-9]*\.?[0-9]+)', str(reason))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

for f in sorted(files):
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
    df = df.sort_values(['ExitTime', 'EntryTime']).reset_index(drop=True)
    
    trading_dates = sorted(df['EntryTime'].unique())
    stats['files_scanned'] += 1
    
    for i in range(len(df) - 1):
        exit_date = df.loc[i, 'ExitTime']
        exit_price = df.loc[i, 'ExitPrice']
        
        next_entry_date = df.loc[i+1, 'EntryTime']
        next_entry_price = df.loc[i+1, 'EntryPrice']
        next_reason = df.loc[i+1, 'EntryReason']
        
        stats['total_sells_with_next_trade'] += 1
        
        future_dates = [d for d in trading_dates if d > exit_date]
        if not future_dates:
            continue
        next_avail = future_dates[0]
        
        if next_entry_date == next_avail:
            stats['immediate_buyback_total'] += 1
            
            base = extract_base(next_reason)
            if base is not None and next_entry_price < base:
                stats['below_baseline_immediate_count'] += 1
                
                if next_entry_price < exit_price:
                    stats['below_baseline_entry_less'] += 1
                elif next_entry_price > exit_price:
                    stats['below_baseline_entry_greater'] += 1
                else:
                    stats['below_baseline_entry_equal'] += 1

print(f"1) total_sells_with_next_trade: {stats['total_sells_with_next_trade']}")
print(f"2) immediate_buyback_total: {stats['immediate_buyback_total']}")
print(f"3) below_baseline_immediate_count: {stats['below_baseline_immediate_count']}")

ratio = 0
if stats['immediate_buyback_total'] > 0:
    ratio = stats['below_baseline_immediate_count'] / stats['immediate_buyback_total']
print(f"4) below_baseline_immediate_ratio_over_immediate: {ratio:.4f}")

count = stats['below_baseline_immediate_count']
if count > 0:
    print(f"5) Below-baseline subset: EntryPrice < previous ExitPrice: {stats['below_baseline_entry_less']} ({stats['below_baseline_entry_less']/count:.4f})")
    print(f"6) Below-baseline subset: EntryPrice > previous ExitPrice: {stats['below_baseline_entry_greater']} ({stats['below_baseline_entry_greater']/count:.4f})")
    print(f"7) Below-baseline subset: EntryPrice == previous ExitPrice: {stats['below_baseline_entry_equal']} ({stats['below_baseline_entry_equal']/count:.4f})")
else:
    print("5) Below-baseline subset: EntryPrice < previous ExitPrice: 0 (0.0000)")
    print("6) Below-baseline subset: EntryPrice > previous ExitPrice: 0 (0.0000)")
    print("7) Below-baseline subset: EntryPrice == previous ExitPrice: 0 (0.0000)")

print(f"8) files_scanned: {stats['files_scanned']}")
