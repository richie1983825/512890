import pandas as pd
import glob
import os
import re

def get_base_price(reason):
    if not isinstance(reason, str):
        return None
    match = re.search(r'base=([\d\.]+)', reason)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def process_files(files):
    stats = {
        'files_scanned': 0,
        'sells_with_next_trade': 0,
        'immediate_buyback_count': 0,
        'lower_count': 0,
        'higher_count': 0,
        'equal_count': 0,
        'baseline_above_count': 0,
        'baseline_below_count': 0,
        'baseline_equal_count': 0,
        'baseline_unknown_count': 0
    }

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        
        required = {'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'EntryReason'}
        if not required.issubset(df.columns):
            continue
        
        df['EntryTime'] = pd.to_datetime(df['EntryTime'])
        df['ExitTime'] = pd.to_datetime(df['ExitTime'])
        df = df.sort_values(['ExitTime', 'EntryTime']).reset_index(drop=True)
        
        trading_dates = sorted(df['EntryTime'].unique())
        if not trading_dates or len(df) < 2:
            continue
            
        stats['files_scanned'] += 1
        
        for i in range(len(df) - 1):
            stats['sells_with_next_trade'] += 1
            
            exit_date = df.loc[i, 'ExitTime']
            exit_price = df.loc[i, 'ExitPrice']
            
            next_entry_date = df.loc[i+1, 'EntryTime']
            next_entry_price = df.loc[i+1, 'EntryPrice']
            next_reason = df.loc[i+1, 'EntryReason']
            
            future_dates = [d for d in trading_dates if d > exit_date]
            if not future_dates:
                continue
            next_avail = future_dates[0]
            
            if next_entry_date == next_avail:
                stats['immediate_buyback_count'] += 1
                
                # Price comparison
                if next_entry_price < exit_price:
                    stats['lower_count'] += 1
                elif next_entry_price > exit_price:
                    stats['higher_count'] += 1
                else:
                    stats['equal_count'] += 1
                
                # Baseline comparison
                base = get_base_price(next_reason)
                if base is None:
                    stats['baseline_unknown_count'] += 1
                else:
                    if next_entry_price > base:
                        stats['baseline_above_count'] += 1
                    elif next_entry_price < base:
                        stats['baseline_below_count'] += 1
                    else:
                        stats['baseline_equal_count'] += 1
                        
    return stats

def print_stats(name, stats):
    print(f"--- {name} ---")
    print(f"files_scanned: {stats['files_scanned']}")
    print(f"sells_with_next_trade: {stats['sells_with_next_trade']}")
    
    imm_count = stats['immediate_buyback_count']
    swnt = stats['sells_with_next_trade']
    imm_prob = imm_count / swnt if swnt > 0 else 0
    print(f"immediate_buyback_count: {imm_count}, immediate_buyback_prob: {imm_prob:.4f}")
    
    if imm_count > 0:
        print(f"lower_count: {stats['lower_count']}, lower_prob: {stats['lower_count']/imm_count:.4f}")
        print(f"higher_count: {stats['higher_count']}, higher_prob: {stats['higher_count']/imm_count:.4f}")
        print(f"equal_count: {stats['equal_count']}, equal_prob: {stats['equal_count']/imm_count:.4f}")
        
        print(f"baseline_above_count: {stats['baseline_above_count']}, baseline_above_prob: {stats['baseline_above_count']/imm_count:.4f}")
        print(f"baseline_below_count: {stats['baseline_below_count']}, baseline_below_prob: {stats['baseline_below_count']/imm_count:.4f}")
        print(f"baseline_equal_count: {stats['baseline_equal_count']}, baseline_equal_prob: {stats['baseline_equal_count']/imm_count:.4f}")
        print(f"baseline_unknown_count: {stats['baseline_unknown_count']}, baseline_unknown_prob: {stats['baseline_unknown_count']/imm_count:.4f}")
    else:
        print("No immediate buybacks found.")
    print()

all_files = glob.glob('reports/**/switch_trade_records.csv', recursive=True)
formal_files = [f for f in all_files if 'reports/global_best_switch_ma20_60_hold45_seed342/' in f]

all_stats = process_files(all_files)
formal_stats = process_files(formal_files)

print_stats("Global Metrics (All Files)", all_stats)
print_stats("Formal Baseline Metrics (reports/global_best_switch_ma20_60_hold45_seed342/)", formal_stats)
