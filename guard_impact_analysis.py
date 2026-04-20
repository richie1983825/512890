import pandas as pd
import os

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# Baseline trade records
baseline_files = {
    '2022': 'reports/global_best_switch_ma20_60_hold45_seed342/window_0_2022/switch_trade_records.csv',
    '2023': 'reports/global_best_switch_ma20_60_hold45_seed342/window_1_2023/switch_trade_records.csv',
    '2024': 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv',
    '2025': 'reports/global_best_switch_ma20_60_hold45_seed342/window_3_2025/switch_trade_records.csv'
}

# Guard experiment trade records
guard_files = {
    '2022': 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_01_guard_switch_trade_records.csv',
    '2023': 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_02_guard_switch_trade_records.csv',
    '2024': 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv',
    '2025': 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_04_guard_switch_trade_records.csv'
}

print(f"{'Year':<6} | {'Base Trades':<12} | {'Guard Trades':<12} | {'Diff':<6}")
print("-" * 45)

for year in ['2022', '2023', '2024', '2025']:
    df_base = load_csv(baseline_files[year])
    df_guard = load_csv(guard_files[year])
    
    n_base = len(df_base)
    n_guard = len(df_guard)
    diff = n_guard - n_base
    
    print(f"{year:<6} | {n_base:<12} | {n_guard:<12} | {diff:<6}")

# Also check performance summaries
base_sum = load_csv('reports/global_best_switch_ma20_60_hold45_seed342/wf3y1y_polyfit_switch_strategy_summary.csv')
guard_sum = load_csv('reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv')

print("\n--- Summary Performance Comparison ---")
# Align indices or match by year/window
# Baseline window names are like '2022-01-04 to 2022-12-30'
# Guard '验证开始' names are like '2022-01-04'

results = []
for idx, row in guard_sum.iterrows():
    v_start = str(row['验证开始'])
    year = v_start[:4]
    
    base_row = base_sum[base_sum['window'].astype(str).str.contains(year)]
    if not base_row.empty:
        results.append({
            'Year': year,
            'BaseRet': base_row.iloc[0]['switch_total_return'],
            'GuardRet': row['switch_总收益率'],
            'BaseMDD': base_row.iloc[0]['switch_max_drawdown'],
            'GuardMDD': row['switch_最大回撤']
        })

pdf = pd.DataFrame(results)
print(pdf.to_string(index=False))

