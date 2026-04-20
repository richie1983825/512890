import pandas as pd
import numpy as np
import datetime

def load_and_prep(path):
    df = pd.read_csv(path)
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    return df.sort_values('EntryTime').reset_index(drop=True)

baseline_path = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv'
guard_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv'

df_b = load_and_prep(baseline_path)
df_g = load_and_prep(guard_path)

# Trading dates in 2024 (from baseline)
all_trading_dates = sorted(pd.to_datetime(df_b['EntryTime']).unique())

results = []

for i in range(len(df_b)):
    curr_exit_date = df_b.loc[i, 'ExitTime']
    curr_exit_reason = str(df_b.loc[i, 'ExitReason'])
    
    if 'stop_loss_grid' not in curr_exit_reason:
        continue
    
    # Next trading day after curr_exit_date
    future_dates = [d for d in all_trading_dates if d > curr_exit_date]
    if not future_dates:
        continue
    next_trading_day = future_dates[0]
    
    # Check if baseline re-entered on next_trading_day
    if i + 1 < len(df_b):
        next_b_entry_date = df_b.loc[i+1, 'EntryTime']
    else:
        continue
        
    if next_b_entry_date == next_trading_day:
        # Baseline re-entered on next trading day
        b_entry_price = df_b.loc[i+1, 'EntryPrice']
        b_entry_reason = df_b.loc[i+1, 'EntryReason'].split(';')[0]
        
        # Check Guard
        guard_same_day = df_g[df_g['EntryTime'] == next_trading_day]
        guard_entered_same_day = "yes" if not guard_same_day.empty else "no"
        
        first_later_entry_date = ""
        first_later_entry_price = np.nan
        price_diff = np.nan
        
        if guard_entered_same_day == "no":
            later_g = df_g[df_g['EntryTime'] > next_trading_day].head(1)
            if not later_g.empty:
                first_later_entry_date = later_g.iloc[0]['EntryTime'].strftime('%Y-%m-%d')
                first_later_entry_price = later_g.iloc[0]['EntryPrice']
                price_diff = first_later_entry_price - b_entry_price
        
        results.append({
            'stoploss_exit_date': curr_exit_date.strftime('%Y-%m-%d'),
            'baseline_nextday_entry_date': next_b_entry_date.strftime('%Y-%m-%d'),
            'baseline_nextday_entry_price': b_entry_price,
            'baseline_nextday_entry_reason_prefix': b_entry_reason,
            'guard_entered_same_day': guard_entered_same_day,
            'guard_first_later_entry_date': first_later_entry_date,
            'guard_first_later_entry_price': first_later_entry_price,
            'price_diff_vs_baseline_entry': price_diff
        })

res_df = pd.DataFrame(results)

# Manual format for markdown table
if not res_df.empty:
    def format_row(row):
        p_diff = f"{row['price_diff_vs_baseline_entry']:.4f}" if not pd.isna(row['price_diff_vs_baseline_entry']) else ""
        g_price = f"{row['guard_first_later_entry_price']:.4f}" if not pd.isna(row['guard_first_later_entry_price']) else ""
        return f"| {row['stoploss_exit_date']} | {row['baseline_nextday_entry_date']} | {row['baseline_nextday_entry_price']:.4f} | {row['baseline_nextday_entry_reason_prefix']} | {row['guard_entered_same_day']} | {row['guard_first_later_entry_date']} | {g_price} | {p_diff} |"

    print("| stoploss_exit_date | baseline_nextday_entry_date | baseline_nextday_entry_price | baseline_nextday_entry_reason_prefix | guard_entered_same_day | guard_first_later_entry_date | guard_first_later_entry_price | price_diff_vs_baseline_entry |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, row in res_df.iterrows():
        print(format_row(row))
else:
    print("No matching baseline re-entries found.")

print("\n--- Summary ---")
if not res_df.empty:
    count_total = len(res_df)
    count_blocked = (res_df['guard_entered_same_day'] == "no").sum()
    avg_diff = res_df[res_df['guard_entered_same_day'] == "no"]['price_diff_vs_baseline_entry'].mean()

    print(f"Total baseline next-day reentries after stop-loss: {count_total}")
    print(f"Number of these blocked by guard on the same day: {count_blocked}")
    if count_blocked > 0:
        print(f"Average price difference for blocked cases: {avg_diff:.4f}")
    else:
        print("No cases blocked by the guard.")
