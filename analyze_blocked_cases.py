import pandas as pd
import numpy as np

def load_and_prep(path):
    df = pd.read_csv(path)
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    return df.sort_values('EntryTime').reset_index(drop=True)

baseline_path = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv'
guard_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv'

df_b = load_and_prep(baseline_path)
df_g = load_and_prep(guard_path)

all_trading_dates = sorted(pd.to_datetime(df_b['EntryTime']).unique())

blocked_cases = []

for i in range(len(df_b) - 1):
    curr_exit_date = df_b.loc[i, 'ExitTime']
    curr_exit_reason = str(df_b.loc[i, 'ExitReason'])
    
    if 'stop_loss_grid' not in curr_exit_reason:
        continue
    
    future_dates = [d for d in all_trading_dates if d > curr_exit_date]
    if not future_dates:
        continue
    next_trading_day = future_dates[0]
    
    next_b_entry_date = df_b.loc[i+1, 'EntryTime']
    if next_b_entry_date == next_trading_day:
        # Check Guard
        guard_same_day = df_g[df_g['EntryTime'] == next_trading_day]
        if guard_same_day.empty:
            b_entry_price = df_b.loc[i+1, 'EntryPrice']
            later_g = df_g[df_g['EntryTime'] > next_trading_day].head(1)
            
            if not later_g.empty:
                g_entry_date = later_g.iloc[0]['EntryTime']
                g_entry_price = later_g.iloc[0]['EntryPrice']
                price_diff = g_entry_price - b_entry_price
                
                blocked_cases.append({
                    'stoploss_exit_date': curr_exit_date.strftime('%Y-%m-%d'),
                    'baseline_entry_date': next_b_entry_date.strftime('%Y-%m-%d'),
                    'baseline_entry_price': b_entry_price,
                    'guard_entry_date': g_entry_date.strftime('%Y-%m-%d'),
                    'guard_entry_price': g_entry_price,
                    'price_diff': price_diff
                })

df_blocked = pd.DataFrame(blocked_cases)

if not df_blocked.empty:
    higher = (df_blocked['price_diff'] > 1e-7).sum()
    lower = (df_blocked['price_diff'] < -1e-7).sum()
    equal = len(df_blocked) - higher - lower
    
    print("### Blocked Cases Summary (2024)")
    print(f"- Total Blocked Cases: {len(df_blocked)}")
    print(f"- Guard Entry Price Higher than Baseline: {higher}")
    print(f"- Guard Entry Price Lower than Baseline: {lower}")
    print(f"- Guard Entry Price Equal to Baseline: {equal}")
    print(f"- Average Price Difference: {df_blocked['price_diff'].mean():.4f}")
    print(f"- Median Price Difference: {df_blocked['price_diff'].median():.4f}")
    print(f"- Max Increase (Costlier for Guard): {df_blocked['price_diff'].max():.4f}")
    print(f"- Max Decrease (Cheaper for Guard): {df_blocked['price_diff'].min():.4f}")
    
    print("\n### Blocked Cases Sorted by Absolute Price Difference")
    df_blocked['abs_diff'] = df_blocked['price_diff'].abs()
    df_sorted = df_blocked.sort_values('abs_diff', ascending=False)
    
    print("| Stop-loss Exit | Baseline Entry | Guard Entry | Baseline Price | Guard Price | Diff |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, row in df_sorted.iterrows():
        print(f"| {row['stoploss_exit_date']} | {row['baseline_entry_date']} | {row['guard_entry_date']} | {row['baseline_entry_price']:.4f} | {row['guard_entry_price']:.4f} | {row['price_diff']:.4f} |")
else:
    print("No blocked cases found where guard entered later.")
