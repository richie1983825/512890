import pandas as pd

df = pd.read_csv('reports/switch_stoploss_reentry_ma20_ma60_scan_seed342/wf3y1y_polyfit_switch_strategy_summary.csv')

# Use correct columns based on the CSV header
# Switch Strategy: switch_总收益率, switch_最大回撤, switch_超额收益, switch_交易次数, switch_平均持有天数
# Polyfit Strategy: polyfit_总收益率, polyfit_最大回撤, polyfit_超额收益, polyfit_交易次数, polyfit_平均持有天数

date = "2026-04-20"

# Process windows 1 to 4
for i, row in df.iterrows():
    if i >= 4: break
    win_idx = i + 1
    win_str = f"{win_idx:02d}"
    round_id = f"wf3y1y-switch-stoploss-reentry-mawindow-{win_str}"
    
    # Switch Strategy row
    switch_ret = row['switch_总收益率'] * 100
    switch_mdd = row['switch_最大回撤'] * 100
    switch_excess = row['switch_超额收益'] * 100
    print(f"| {date} | {round_id} | switch | {switch_ret:.2f}% | {switch_mdd:.2f}% | {switch_excess:.2f}% | {int(row['switch_交易次数'])} | {row['switch_平均持有天数']:.2f} |")
    
    # Polyfit Strategy row
    poly_ret = row['polyfit_总收益率'] * 100
    poly_mdd = row['polyfit_最大回撤'] * 100
    poly_excess = row['polyfit_超额收益'] * 100
    print(f"| {date} | {round_id} | polyfit | {poly_ret:.2f}% | {poly_mdd:.2f}% | {poly_excess:.2f}% | {int(row['polyfit_交易次数'])} | {row['polyfit_平均持有天数']:.2f} |")
