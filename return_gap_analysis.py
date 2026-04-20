import pandas as pd
import numpy as np

def load_trades(path):
    df = pd.read_csv(path)
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    return df

baseline_path = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv'
guard_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv'

df_b = load_trades(baseline_path)
df_g = load_trades(guard_path)

# 1. Compare total returns from the trade record PnL
# Note: PnL is absolute value, we need to compare cumulative impact.
# Actually, the user says return dropped by 16%. Let's look at the summary metrics too if possible.
# But let's work with the trade records first.

print(f"Baseline: Total PnL = {df_b['PnL'].sum():.2f}, Mean ReturnPct = {df_b['ReturnPct'].mean():.4f}, Count = {len(df_b)}")
print(f"Guard:    Total PnL = {df_g['PnL'].sum():.2f}, Mean ReturnPct = {df_g['ReturnPct'].mean():.4f}, Count = {len(df_g)}")

# 2. Sequence Analysis
# We want to see the path divergence.
# We'll merge by time to see where they overlap or differ.

df_b['Source'] = 'Baseline'
df_g['Source'] = 'Guard'

# Combine and sort to see the timeline
combined = pd.concat([df_b, df_g]).sort_values(['EntryTime', 'Source'])

# Identify gaps or major mismatches in ReturnPct
# Let's group by month to see where the divergence happens
df_b['Month'] = df_b['EntryTime'].dt.to_period('M')
df_g['Month'] = df_g['EntryTime'].dt.to_period('M')

monthly_b = df_b.groupby('Month')['PnL'].sum()
monthly_g = df_g.groupby('Month')['PnL'].sum()
monthly_diff = (monthly_g - monthly_b).dropna()

print("\n--- Monthly PnL Difference (Guard - Baseline) ---")
print(monthly_diff.sort_values())

# 3. Look at Sep 2024 specifically
print("\n--- Sep 2024 Trades Baseline ---")
print(df_b[df_b['Month'] == '2024-09'][['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'ExitReason']])

print("\n--- Sep 2024 Trades Guard ---")
print(df_g[df_g['Month'] == '2024-09'][['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'ExitReason']])

# 4. Impact of the "Next-Day Guard"
# The guard prevents entry the day after a stop-loss.
# Let's find those instances in baseline and see what happened in guard.
sl_baseline = df_b[df_b['ExitReason'].str.contains('stop_loss', na=False)]
for idx, row in sl_baseline.iterrows():
    # Next trade in baseline
    next_trades = df_b[df_b['EntryTime'] > row['ExitTime']].head(1)
    if not next_trades.empty:
        next_b = next_trades.iloc[0]
        # Check if guard has a trade starting at the same time
        guard_at_same_time = df_g[df_g['EntryTime'] == next_b['EntryTime']]
        if guard_at_same_time.empty:
            # Guard missed or delayed this entry
            guard_later = df_g[df_g['EntryTime'] > next_b['EntryTime']].head(1)
            print(f"\nSL Exit at {row['ExitTime'].date()}. Baseline re-entered {next_b['EntryTime'].date()} (PnL: {next_b['PnL']:.1f}).")
            if not guard_later.empty:
                print(f"  Guard delayed entry to {guard_later.iloc[0]['EntryTime'].date()} (Next PnL: {guard_later.iloc[0]['PnL']:.1f})")
            else:
                print("  Guard never re-entered in 2024.")

