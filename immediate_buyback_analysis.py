import pandas as pd
import os
import glob
from datetime import timedelta

def analyze():
    file_paths = glob.glob('reports/**/switch_trade_records.csv', recursive=True)
    
    total_sells_with_next = 0
    imm_count = 0
    lower_c = 0
    higher_c = 0
    equal_c = 0
    
    files_scanned = 0

    for path in file_paths:
        try:
            df = pd.read_csv(path)
            if df.empty or 'EntryTime' not in df.columns or 'ExitTime' not in df.columns or 'EntryPrice' not in df.columns or 'ExitPrice' not in df.columns:
                continue
            
            files_scanned += 1
            df['EntryTime'] = pd.to_datetime(df['EntryTime'])
            df['ExitTime'] = pd.to_datetime(df['ExitTime'])
            # Sort by ExitTime to ensure we are looking at chronological sequence of trades ending
            df = df.sort_values('ExitTime').reset_index(drop=True)
            
            # Unique entry dates in this file represent the "available trading days" for this strategy instance
            all_entry_dates = sorted(df['EntryTime'].dt.date.unique())
            
            for i in range(len(df) - 1):
                total_sells_with_next += 1
                
                exit_date = df.loc[i, 'ExitTime'].date()
                exit_price = df.loc[i, 'ExitPrice']
                
                next_entry_date = df.loc[i+1, 'EntryTime'].date()
                next_entry_price = df.loc[i+1, 'EntryPrice']
                
                # Find the next available trading day (EntryDate) after exit_date
                next_trading_day = next((d for d in all_entry_dates if d > exit_date), None)
                
                if next_trading_day and next_entry_date == next_trading_day:
                    imm_count += 1
                    if next_entry_price < exit_price:
                        lower_c += 1
                    elif next_entry_price > exit_price:
                        higher_c += 1
                    else:
                        equal_c += 1
        except Exception:
            continue

    print(f"Files scanned: {files_scanned}")
    print(f"sells_with_next_trade: {total_sells_with_next}")
    print(f"immediate_buyback_count: {imm_count}")
    
    imm_prob = imm_count / total_sells_with_next if total_sells_with_next > 0 else 0
    print(f"immediate_buyback_prob: {imm_prob:.4f}")
    
    if imm_count > 0:
        print(f"lower_count: {lower_c}")
        print(f"lower_prob: {lower_c / imm_count:.4f}")
        print(f"higher_count: {higher_c}")
        print(f"higher_prob: {higher_c / imm_count:.4f}")
        print(f"equal_count: {equal_c}")
        print(f"equal_prob: {equal_c / imm_count:.4f}")
    else:
        print("No immediate buybacks found.")

if __name__ == '__main__':
    analyze()
