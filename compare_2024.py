import pandas as pd
import os

old_summary_path = 'reports/global_best_switch_ma20_60_hold45_seed342/wf3y1y_polyfit_switch_strategy_summary.csv'
new_summary_path = 'reports/switch_stoploss_reentry_ma20_ma60_scan_seed342/wf3y1y_polyfit_switch_strategy_summary.csv'
old_trades_path = 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv'
new_trades_path = 'reports/switch_stoploss_reentry_ma20_ma60_scan_seed342/wf3y1y_03_switch_trade_records.csv'

def get_summary_row(path, window_name):
    if os.path.exists(path):
        df = pd.read_csv(path)
        row = df[df['window'] == window_name]
        if not row.empty:
            return row.iloc[0]
    return None

old_sum = get_summary_row(old_summary_path, 'window_2_2024')
new_sum = get_summary_row(new_summary_path, 'window_2_2024')

def load_trades(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

old_trades = load_trades(old_trades_path)
new_trades = load_trades(new_trades_path)

print("--- 2024 Analysis Comparison ---")
if old_sum is not None and new_sum is not None:
    old_ret = old_sum['TotalReturn']
    new_ret = new_sum['TotalReturn']
    print(f"1) Total Return: Old={old_ret:.4f}, New={new_ret:.4f}, Diff={new_ret-old_ret:.4f}")
    print(f"2) Max Drawdown: Old={old_sum['MaxDrawdown']:.4f}, New={new_sum['MaxDrawdown']:.4f}")
else:
    print("1) Total Return: Summary missing")
    print("2) Max Drawdown: Summary missing")

print(f"3) Trade Count: Old={len(old_trades)}, New={len(new_trades)}")

def count_sl(df):
    if 'ExitReason' in df.columns:
        return df['ExitReason'].str.contains('stop_loss_grid', na=False).sum()
    return 0

print(f"4) Stop-loss Exits: Old={count_sl(old_trades)}, New={count_sl(new_trades)}")

def avg_holding(df):
    if 'HoldingDays' in df.columns:
        return df['HoldingDays'].mean()
    return float('nan')

print(f"5) Avg HoldingDays: Old={avg_holding(old_trades):.2f}, New={avg_holding(new_trades):.2f}")

def top_exits(df, name):
    if 'ExitReason' in df.columns:
        prefixes = df['ExitReason'].str.split('_').str[0]
        counts = prefixes.value_counts().head(5)
        print(f"6) Top 5 ExitReason Prefixes ({name}):")
        for k, v in counts.items():
            print(f"   {k}: {v}")
    else:
        print(f"6) Top 5 ExitReason Prefixes ({name}): No ExitReason column")

top_exits(old_trades, "Old")
top_exits(new_trades, "New")
