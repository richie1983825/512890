import pandas as pd
import numpy as np
import calendar

# Paths
old_daily_path = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/strategy_pair_daily_comparison.csv'
new_daily_path = 'reports/switch_stoploss_reentry_ma20_ma60_scan_seed342/wf3y1y_03_switch_daily_cumulative_return_comparison.csv'
old_trades_path = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv'
new_trades_path = 'reports/switch_stoploss_reentry_ma20_ma60_scan_seed342/wf3y1y_03_switch_trade_records.csv'

# Load Daily Data
old_daily = pd.read_csv(old_daily_path, parse_dates=['trade_date'])
new_daily = pd.read_csv(new_daily_path, parse_dates=['trade_date'])

# Normalize columns
old_daily = old_daily.rename(columns={'Switch': 'cum_ret'})
new_daily = new_daily.rename(columns={'策略累计收益': 'cum_ret'})

# Ensure they cover the same dates or at least 2024
old_daily = old_daily.set_index('trade_date').sort_index()
new_daily = new_daily.set_index('trade_date').sort_index()

# Final return
old_final = old_daily['cum_ret'].iloc[-1]
new_final = new_daily['cum_ret'].iloc[-1]

# Monthly Analysis
def get_monthly_returns(df):
    daily_val = df['cum_ret'] + 1
    # Resample to month end
    monthly_val = daily_val.resample('ME').last()
    # Shift to get previous month end
    prev_val = monthly_val.shift(1).fillna(1.0)
    # The first entry in monthly_val/prev_val corresponds to the first month
    # But for the very first month, we should use the first value of the index's start (which is 1.0)
    monthly_ret = monthly_val / prev_val - 1
    return monthly_ret

old_monthly = get_monthly_returns(old_daily)
new_monthly = get_monthly_returns(new_daily)

monthly_comp = pd.DataFrame({
    'Old': old_monthly,
    'New': new_monthly
}).dropna()
monthly_comp['Diff'] = monthly_comp['New'] - monthly_comp['Old']
monthly_comp['AbsDiff'] = monthly_comp['Diff'].abs()

# Top 3 gaps
top_gaps = monthly_comp.sort_values('AbsDiff', ascending=False).head(3)

# Trades Analysis
old_trades = pd.read_csv(old_trades_path, parse_dates=['EntryTime', 'ExitTime'])
new_trades = pd.read_csv(new_trades_path, parse_dates=['EntryTime', 'ExitTime'])

def get_monthly_stats(trades_df, months):
    stats = {}
    for month in months:
        m_start = month.replace(day=1)
        last_day = calendar.monthrange(month.year, month.month)[1]
        m_end = month.replace(day=last_day)
        
        # Count trades exiting in this month
        mask = (trades_df['ExitTime'] >= m_start) & (trades_df['ExitTime'] <= m_end)
        m_trades = trades_df[mask]
        
        sl_count = 0
        if 'ExitReason' in m_trades.columns:
            sl_count = m_trades['ExitReason'].str.contains('stop_loss_grid', na=False).sum()
        
        stats[month] = {'count': len(m_trades), 'sl': sl_count}
    return stats

top_months = top_gaps.index.tolist()
old_stats = get_monthly_stats(old_trades, top_months)
new_stats = get_monthly_stats(new_trades, top_months)

# Output
print("--- 2024 Performance Comparison (Old vs New) ---")
print(f"Final Cumulative Return: Old={old_final:.4f}, New={new_final:.4f}")
print("\nMonthly Returns Comparison:")
print(f"{'Month':<10} | {'Old':<8} | {'New':<8} | {'Diff':<8}")
print("-" * 45)
for idx, row in monthly_comp.iterrows():
    print(f"{idx.strftime('%Y-%m'):<10} | {row['Old']:>8.2%} | {row['New']:>8.2%} | {row['Diff']:>+8.2%}")

print("\nTop 3 Monthly Gaps & Trade Activity (by ExitTime):")
print(f"{'Month':<10} | {'Gap':<8} | {'Old T/SL':<10} | {'New T/SL':<10}")
print("-" * 45)
for month in sorted(top_months):
    gap = monthly_comp.loc[month, 'Diff']
    o = old_stats[month]
    n = new_stats[month]
    print(f"{month.strftime('%Y-%m'):<10} | {gap:>+8.2%} | {o['count']:>2}/{o['sl']:>2}     | {n['count']:>2}/{n['sl']:>2}")
