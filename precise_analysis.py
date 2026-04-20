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

data_all = []

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
            
            cat = get_category(exit_reason)
            price_cmp = 'equal'
            if next_entry_price < exit_price:
                price_cmp = 'lower'
            elif next_entry_price > exit_price:
                price_cmp = 'higher'
            
            base = extract_base(next_reason)
            is_below_baseline = (base is not None and next_entry_price < base)
            is_sl = 'stop_loss' in str(exit_reason).lower()
            
            data_all.append({
                'cat': cat,
                'price_cmp': price_cmp,
                'is_sl': is_sl,
                'is_below_baseline': is_below_baseline
            })

df = pd.DataFrame(data_all)
below_baseline_df = df[df['is_below_baseline'] == True]

print(f"1) Context counts:")
print(f"files_scanned={stats['files_scanned']}")
print(f"sells_with_next_trade={stats['sells_with_next_trade']}")
print(f"immediate_total={stats['immediate_total']}")
print(f"below_baseline_immediate_total={len(below_baseline_df)}")

print(f"\n2) Exit reason category distribution within below-baseline-immediate:")
cat_counts = below_baseline_df['cat'].value_counts()
total_bb = len(below_baseline_df)
for cat, count in cat_counts.items():
    pct = count / total_bb
    print(f"{cat}: n={count}, pct={pct:.4f}")
    c_df = below_baseline_df[below_baseline_df['cat'] == cat]
    cmp_counts = c_df['price_cmp'].value_counts()
    c_total = len(c_df)
    l_c = cmp_counts.get('lower', 0)
    h_c = cmp_counts.get('higher', 0)
    e_c = cmp_counts.get('equal', 0)
    print(f"  lower={l_c}({l_c/c_total:.4f}) higher={h_c}({h_c/c_total:.4f}) equal={e_c}({e_c/c_total:.4f})")

print(f"\n3) Stop-loss immediate (all immediate, no baseline filter):")
sl_imm = df[df['is_sl'] == True]
sl_imm_n = len(sl_imm)
print(f"stoploss_immediate_n={sl_imm_n}")
if sl_imm_n > 0:
    cmp_counts = sl_imm['price_cmp'].value_counts()
    l_c = cmp_counts.get('lower', 0)
    h_c = cmp_counts.get('higher', 0)
    e_c = cmp_counts.get('equal', 0)
    print(f"lower={l_c}({l_c/sl_imm_n:.4f})")
    print(f"higher={h_c}({h_c/sl_imm_n:.4f})")
    print(f"equal={e_c}({e_c/sl_imm_n:.4f})")

print(f"\n4) Stop-loss immediate AND below-baseline:")
sl_bb = below_baseline_df[below_baseline_df['is_sl'] == True]
sl_bb_n = len(sl_bb)
print(f"stoploss_below_baseline_n={sl_bb_n}")
if sl_bb_n > 0:
    cmp_counts = sl_bb['price_cmp'].value_counts()
    l_c = cmp_counts.get('lower', 0)
    h_c = cmp_counts.get('higher', 0)
    e_c = cmp_counts.get('equal', 0)
    print(f"lower={l_c}({l_c/sl_bb_n:.4f})")
    print(f"higher={h_c}({h_c/sl_bb_n:.4f})")
    print(f"equal={e_c}({e_c/sl_bb_n:.4f})")
