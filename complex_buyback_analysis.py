import pandas as pd
import glob
import re
import os

files = glob.glob('reports/global_best_switch_ma20_60_hold45_seed342/**/switch_trade_records.csv', recursive=True)

stats = {
    'files_scanned': 0,
    'sells_with_next_trade': 0,
    'immediate_total': 0,
}

# Data for A: below-baseline immediate subset
below_baseline_immediate_data = []

# Data for B: stop-loss immediate subset
stop_loss_immediate_data = []

def extract_base(reason):
    match = re.search(r'base=([0-9]*\.?[0-9]+)', str(reason))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def get_category(reason):
    reason = str(reason)
    if '(' in reason:
        return reason.split('(')[0].strip()
    return reason.strip()

for f in sorted(files):
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    
    required = {'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'EntryReason', 'ExitReason'}
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
        exit_reason = df.loc[i, 'ExitReason']
        
        next_entry_date = df.loc[i+1, 'EntryTime']
        next_entry_price = df.loc[i+1, 'EntryPrice']
        next_reason = df.loc[i+1, 'EntryReason']
        
        stats['sells_with_next_trade'] += 1
        
        future_dates = [d for d in trading_dates if d > exit_date]
        if not future_dates:
            continue
        next_avail = future_dates[0]
        
        if next_entry_date == next_avail:
            stats['immediate_total'] += 1
            
            # Category and price comparison
            cat = get_category(exit_reason)
            price_cmp = 'equal'
            if next_entry_price < exit_price:
                price_cmp = 'less'
            elif next_entry_price > exit_price:
                price_cmp = 'greater'
            
            # Baseline check
            base = extract_base(next_reason)
            is_below_baseline = (base is not None and next_entry_price < base)
            
            if is_below_baseline:
                below_baseline_immediate_data.append({
                    'cat': cat,
                    'price_cmp': price_cmp,
                    'is_sl': 'stop_loss' in str(exit_reason).lower()
                })
            
            if 'stop_loss' in str(exit_reason).lower():
                stop_loss_immediate_data.append({
                    'price_cmp': price_cmp,
                    'is_below_baseline': is_below_baseline
                })

print(f"Files Scanned: {stats['files_scanned']}")
print(f"Sells with Next Trade: {stats['sells_with_next_trade']}")
print(f"Immediate Total: {stats['immediate_total']}")

# A) Below-baseline immediate subset analysis
df_a = pd.DataFrame(below_baseline_immediate_data)
print(f"\nA) Below-Baseline Immediate Subset (Total Count: {len(df_a)})")
if not df_a.empty:
    summary_a = df_a.groupby(['cat', 'price_cmp']).size().unstack(fill_value=0)
    # Ensure columns exist
    for col in ['less', 'greater', 'equal']:
        if col not in summary_a.columns:
            summary_a[col] = 0
    summary_a['total'] = summary_a.sum(axis=1)
    summary_a['p_less'] = summary_a['less'] / summary_a['total']
    summary_a['p_greater'] = summary_a['greater'] / summary_a['total']
    summary_a['p_equal'] = summary_a['equal'] / summary_a['total']
    print(summary_a[['total', 'less', 'greater', 'equal', 'p_less', 'p_greater', 'p_equal']].to_string())

# B) Stop-loss specific under immediate buyback
df_b = pd.DataFrame(stop_loss_immediate_data)
print(f"\nB) Stop-Loss-Specific Immediate Subset (Total Count: {len(df_b)})")
if not df_b.empty:
    counts = df_b['price_cmp'].value_counts()
    total = len(df_b)
    for res in ['less', 'greater', 'equal']:
        c = counts.get(res, 0)
        print(f"  EntryPrice {res:7} than ExitPrice: {c:4} ({c/total:.4f})")

# C) Stop-loss-specific under below-baseline immediate subset
df_c = df_a[df_a['is_sl'] == True]
print(f"\nC) Stop-Loss-Specific Below-Baseline Immediate Subset (Total Count: {len(df_c)})")
if not df_c.empty:
    counts = df_c['price_cmp'].value_counts()
    total = len(df_c)
    for res in ['less', 'greater', 'equal']:
        c = counts.get(res, 0)
        print(f"  EntryPrice {res:7} than ExitPrice: {c:4} ({c/total:.4f})")
