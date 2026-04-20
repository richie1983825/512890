import pandas as pd
import os
import glob

def analyze():
    file_paths = glob.glob('reports/**/switch_trade_records.csv', recursive=True)
    
    samples = []
    reason_counts = {}
    total_immediate_buybacks = 0
    total_higher_price_buybacks = 0
    
    files_scanned = 0

    for path in file_paths:
        try:
            df = pd.read_csv(path)
            if df.empty or 'EntryTime' not in df.columns or 'ExitTime' not in df.columns or 'EntryPrice' not in df.columns or 'ExitPrice' not in df.columns:
                continue
            
            files_scanned += 1
            df['EntryTime'] = pd.to_datetime(df['EntryTime'])
            df['ExitTime'] = pd.to_datetime(df['ExitTime'])
            # Sort chronologically
            df = df.sort_values(['ExitTime', 'EntryTime']).reset_index(drop=True)
            
            # Use unique entry dates as the sequence of available trading days in this backtest
            all_entry_dates = sorted(df['EntryTime'].dt.date.unique())
            
            for i in range(len(df) - 1):
                exit_row = df.iloc[i]
                next_row = df.iloc[i+1]
                
                exit_date = exit_row['ExitTime'].date()
                exit_price = exit_row['ExitPrice']
                
                next_entry_date = next_row['EntryTime'].date()
                next_entry_price = next_row['EntryPrice']
                
                # Next trading day: the first entry date that is strictly after the current exit date
                next_trading_day = next((d for d in all_entry_dates if d > exit_date), None)
                
                if next_trading_day and next_entry_date == next_trading_day:
                    total_immediate_buybacks += 1
                    
                    if next_entry_price > exit_price:
                        total_higher_price_buybacks += 1
                        
                        entry_reason = next_row.get('EntryReason', '__MISSING__')
                        if pd.isna(entry_reason): entry_reason = '__MISSING__'
                        
                        reason_counts[entry_reason] = reason_counts.get(entry_reason, 0) + 1
                        
                        if len(samples) < 20:
                            samples.append({
                                'file': path,
                                'prev_exit_time': exit_row['ExitTime'],
                                'prev_exit_price': exit_price,
                                'next_entry_time': next_row['EntryTime'],
                                'next_entry_price': next_entry_price,
                                'price_diff_pct': (next_entry_price / exit_price - 1) * 100,
                                'entry_reason': entry_reason,
                                'prev_exit_reason': exit_row.get('ExitReason', 'N/A')
                            })
        except Exception as e:
            # print(f"Error processing {path}: {e}")
            continue

    print(f"Files scanned: {files_scanned}")
    print(f"total_immediate_buybacks: {total_immediate_buybacks}")
    print(f"total_higher_price_buybacks: {total_higher_price_buybacks}")
    
    if total_immediate_buybacks > 0:
        pct = (total_higher_price_buybacks / total_immediate_buybacks) * 100
        print(f"percentage_higher_within_immediate: {pct:.2f}%")
    
    print("\n--- EntryReason Distribution for Higher Price Buybacks ---")
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons:
        print(f"{reason}: {count}")

    if samples:
        print("\n--- Samples (up to 20) ---")
        df_samples = pd.DataFrame(samples)
        print(df_samples.to_string())

if __name__ == '__main__':
    analyze()
