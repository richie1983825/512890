import pandas as pd
import os
import glob
import json
from datetime import timedelta

def analyze_buybacks_strict(file_pattern):
    file_paths = glob.glob(file_pattern, recursive=True)
    
    overall_stats = {
        "A": {"total_sells": 0, "immediate_count": 0, "lower": 0, "higher": 0, "equal": 0},
        "B": {"total_sells": 0, "immediate_count": 0, "lower": 0, "higher": 0, "equal": 0}
    }
    
    file_results = []

    for path in sorted(file_paths):
        df = pd.read_csv(path)
        if df.empty or 'EntryTime' not in df.columns or 'ExitTime' not in df.columns:
            continue
            
        df['EntryTime'] = pd.to_datetime(df['EntryTime'])
        df['ExitTime'] = pd.to_datetime(df['ExitTime'])
        df = df.sort_values(['ExitTime', 'EntryTime']).reset_index(drop=True)
        
        # Unique sorted entry dates for case B
        all_entry_dates = sorted(df['EntryTime'].dt.date.unique())
        
        file_stats = {
            "path": path,
            "sells_count": 0,
            "A_count": 0,
            "B_count": 0
        }

        for i in range(len(df) - 1):
            curr_exit_time = df.loc[i, 'ExitTime']
            curr_exit_price = df.loc[i, 'ExitPrice']
            next_entry_time = df.loc[i+1, 'EntryTime']
            next_entry_price = df.loc[i+1, 'EntryPrice']

            if pd.isna(curr_exit_time) or pd.isna(next_entry_time) or pd.isna(curr_exit_price) or pd.isna(next_entry_price):
                continue

            file_stats["sells_count"] += 1
            curr_exit_date = curr_exit_time.date()
            next_entry_date = next_entry_time.date()

            # Case A: Strict Calendar T+1
            is_A = (next_entry_date == curr_exit_date + timedelta(days=1))
            if is_A:
                file_stats["A_count"] += 1
                overall_stats["A"]["immediate_count"] += 1
                if next_entry_price < curr_exit_price: overall_stats["A"]["lower"] += 1
                elif next_entry_price > curr_exit_price: overall_stats["A"]["higher"] += 1
                else: overall_stats["A"]["equal"] += 1

            # Case B: Next trading day in file
            # Find the smallest date in all_entry_dates > curr_exit_date
            next_trading_day = next((d for d in all_entry_dates if d > curr_exit_date), None)
            is_B = (next_entry_date == next_trading_day)
            if is_B:
                file_stats["B_count"] += 1
                overall_stats["B"]["immediate_count"] += 1
                if next_entry_price < curr_exit_price: overall_stats["B"]["lower"] += 1
                elif next_entry_price > curr_exit_price: overall_stats["B"]["higher"] += 1
                else: overall_stats["B"]["equal"] += 1
            
            # Common denominator for both? 
            # The prompt asks for denominator "sells_with_next_trade"
            overall_stats["A"]["total_sells"] += 1
            overall_stats["B"]["total_sells"] += 1

        file_results.append(file_stats)

    # Output Console Table
    print(f"{'File Path':<70} | {'Sells':<6} | {'A':<4} | {'B':<4}")
    print("-" * 90)
    for f in file_results:
        print(f"{f['path'][-70:]:<70} | {f['sells_count']:<6} | {f['A_count']:<4} | {f['B_count']:<4}")

    # Output JSON Summary
    summary = {}
    for key in ["A", "B"]:
        s = overall_stats[key]
        denom = s["total_sells"]
        imm = s["immediate_count"]
        summary[key] = {
            "sells_with_next_trade": denom,
            "immediate_count": imm,
            "prob_immediate": imm / denom if denom > 0 else 0,
            "among_immediate": {
                "lower": s["lower"], "p_lower": s["lower"] / imm if imm > 0 else 0,
                "higher": s["higher"], "p_higher": s["higher"] / imm if imm > 0 else 0,
                "equal": s["equal"], "p_equal": s["equal"] / imm if imm > 0 else 0
            }
        }
    
    print("\nSUMMARY JSON:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    analyze_buybacks_strict('reports/**/switch_trade_records.csv')
