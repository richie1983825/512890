import pandas as pd
import os

def load_csv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame()

# 2024 is high interest because of the increase in trade count and drop in return
base_2024 = load_csv('reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv')
guard_2024 = load_csv('reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv')

def analyze_returns(df, name):
    if df.empty:
        return f"{name}: No records"
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['TradeReturn'] = (df['ExitPrice'] - df['EntryPrice']) / df['EntryPrice']
    avg_ret = df['TradeReturn'].mean()
    win_rate = (df['TradeReturn'] > 0).mean()
    return f"{name}: AvgRet={avg_ret:.4f}, WinRate={win_rate:.2%}, Trades={len(df)}"

print(analyze_returns(base_2024, "Base 2024"))
print(analyze_returns(guard_2024, "Guard 2024"))

# Look for high-return trades in Base that might be missing or split in Guard
if not base_2024.empty:
    print("\nBase 2024 Top Trades:")
    print(base_2024[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'TradeReturn']].sort_values('TradeReturn', ascending=False).head(5))

if not guard_2024.empty:
    print("\nGuard 2024 Top Trades:")
    # We need to calculate TradeReturn for guard if not present
    guard_2024['TradeReturn'] = (guard_2024['ExitPrice'] - guard_2024['EntryPrice']) / guard_2024['EntryPrice']
    print(guard_2024[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'TradeReturn']].sort_values('TradeReturn', ascending=False).head(5))
