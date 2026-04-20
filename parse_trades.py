import pandas as pd
import sys
from datetime import datetime, timedelta

files = [
    "reports/global_best_switch_ma20_60_hold45_seed342/window_1_2023/switch_trade_records.csv",
    "reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv"
]

target_dates = ["2023-08-15", "2023-09-04", "2024-07-31"]

df_list = []
for f in files:
    try:
        df_list.append(pd.read_csv(f))
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not df_list:
    sys.exit(1)

df = pd.concat(df_list)
df['EntryTime'] = pd.to_datetime(df['EntryTime']).dt.date

def find_nearby_trades(target_date_str, df, window=3):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    # Unique entry dates in the dataframe
    available_dates = sorted(df['EntryTime'].unique())
    
    # Filter dates within window
    nearby_dates = [d for d in available_dates if abs((d - target_dt).days) <= window]
    
    if target_dt in nearby_dates:
        # If exact date exists, return only that
        return df[df['EntryTime'] == target_dt]
    elif nearby_dates:
        # Otherwise return the closest ones in the window
        closest_date = min(nearby_dates, key=lambda d: abs((d - target_dt).days))
        return df[df['EntryTime'] == closest_date]
    return pd.DataFrame()

results = []
for date_str in target_dates:
    trades = find_nearby_trades(date_str, df)
    if not trades.empty:
        for _, row in trades.iterrows():
            results.append({
                "TargetDate": date_str,
                "EntryTime": row['EntryTime'],
                "ExitTime": row['ExitTime'],
                "EntryPrice": row['EntryPrice'],
                "EntryReason": row['EntryReason'],
                "ExitReason": row['ExitReason']
            })
    else:
        results.append({
            "TargetDate": date_str,
            "EntryTime": "No trades found",
            "ExitTime": "",
            "EntryPrice": "",
            "EntryReason": "",
            "ExitReason": ""
        })

output_df = pd.DataFrame(results)
# Formatting output columns
print(output_df[['TargetDate', 'EntryTime', 'ExitTime', 'EntryPrice', 'EntryReason', 'ExitReason']].to_string(index=False))

