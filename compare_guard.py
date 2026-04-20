import pandas as pd
import os

baseline_summary = 'reports/global_best_switch_ma20_60_hold45_seed342/wf3y1y_polyfit_switch_strategy_summary.csv'
guard_summary = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv'

def get_summary_data(path, is_guard=False):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if is_guard:
        df = df[['验证开始', 'guard_switch_总收益率', 'guard_switch_最大回撤']]
        df['year'] = pd.to_datetime(df['验证开始']).dt.year
        df = df.rename(columns={'guard_switch_总收益率': 'Return', 'guard_switch_最大回撤': 'MDD'})
    else:
        df['year'] = pd.to_datetime(df['window'].str.split(' to ').str[0]).dt.year
        df = df.rename(columns={'switch_total_return': 'Return', 'switch_max_drawdown': 'MDD'})
    return df[['year', 'Return', 'MDD']]

base_df = get_summary_data(baseline_summary)
guard_df = get_summary_data(guard_summary, is_guard=True)

merged = pd.merge(base_df, guard_df, on='year', suffixes=('_base', '_guard'))
print("Yearly Performance Comparison:")
print(merged.to_string(index=False))

guard_files = {
    2022: 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_01_guard_switch_trade_records.csv',
    2023: 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_02_guard_switch_trade_records.csv',
    2024: 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_03_guard_switch_trade_records.csv',
    2025: 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_04_guard_switch_trade_records.csv'
}

base_files = {
    2022: 'reports/global_best_switch_ma20_60_hold45_seed342/window_0_2022/switch_trade_records.csv',
    2023: 'reports/global_best_switch_ma20_60_hold45_seed342/window_1_2023/switch_trade_records.csv',
    2024: 'reports/global_best_switch_ma20_60_hold45_seed342/window_2_2024/switch_trade_records.csv',
    2025: 'reports/global_best_switch_ma20_60_hold45_seed342/window_3_2025/switch_trade_records.csv'
}

print("\nYearly Trade Count Comparison:")
for year in [2022, 2023, 2024, 2025]:
    b_count = 0
    g_count = 0
    if os.path.exists(base_files[year]):
        b_count = len(pd.read_csv(base_files[year]))
    if os.path.exists(guard_files[year]):
        g_count = len(pd.read_csv(guard_files[year]))
    print(f"{year}: Baseline={b_count}, Guard={g_count}")
