import pandas as pd
import numpy as np

before_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary_before_profit_hold_stoploss.csv'
after_path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv'

def get_stats(path):
    df = pd.read_csv(path)
    df['Year'] = pd.to_datetime(df['验证开始']).dt.year
    return df

df_before = get_stats(before_path)
df_after = get_stats(after_path)

cols_to_compare = [
    ('guard_switch_总收益率', 'Return'),
    ('guard_switch_最大回撤', 'MaxDD'),
    ('guard_switch_年化收益率', 'AnnualRet'),
    ('guard_switch_交易次数', 'Trades'),
    ('guard_switch_平均持有天数', 'AvgHold')
]

for col_raw, col_name in cols_to_compare:
    print(f"--- Comparison for {col_name} ---")
    data = []
    for y in sorted(df_before['Year'].unique()):
        before_val = df_before[df_before['Year'] == y][col_raw].values[0]
        after_val = df_after[df_after['Year'] == y][col_raw].values[0]
        data.append({'Year': y, 'Before': before_val, 'After': after_val, 'Diff': after_val - before_val})
    print(pd.DataFrame(data).to_string(index=False))
    print()

print("--- Cumulative Guard Return ---")
b_ret = df_before['guard_switch_总收益率']
a_ret = df_after['guard_switch_总收益率']
b_arith = b_ret.mean()
a_arith = a_ret.mean()
b_total = (1 + b_ret).prod() - 1
a_total = (1 + a_ret).prod() - 1

print(f"Arithmetic Mean: Before={b_arith:.4f}, After={a_arith:.4f}, Diff={a_arith-b_arith:.4f}")
print(f"Compounded Total: Before={b_total:.4f}, After={a_total:.4f}, Diff={a_total-b_total:.4f}")
