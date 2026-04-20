import pandas as pd
import numpy as np

path = 'reports/switch_stoploss_nextday_guard_scan_seed342/wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv'
df = pd.read_csv(path)

# Columns based on the head output
# Verification start date is in '验证开始'
# switch_total_return is 'switch_总收益率'
# stoploss_nextday_total_return (guard) is 'guard_switch_总收益率'

results = []
for _, row in df.iterrows():
    year = str(row['验证开始'])[:4]
    sw_ret = row['switch_总收益率']
    gd_ret = row['guard_switch_总收益率']
    results.append({
        'Year': year,
        'Switch': sw_ret,
        'Stoploss_Nextday': gd_ret,
        'Diff': gd_ret - sw_ret
    })

res_df = pd.DataFrame(results)

# Arithmetic Mean
mean_sw = res_df['Switch'].mean()
mean_gd = res_df['Stoploss_Nextday'].mean()

# Compounded Total Return: Product(1+r) - 1
total_sw = np.prod(1 + res_df['Switch']) - 1
total_gd = np.prod(1 + res_df['Stoploss_Nextday']) - 1

print("| Year | Switch Return | Stoploss Nextday Return | Diff |")
print("|------|---------------|-------------------------|------|")
for _, row in res_df.iterrows():
    print(f"| {row['Year']} | {row['Switch']:.4%} | {row['Stoploss_Nextday']:.4%} | {row['Diff']:+.4%} |")

print(f"\nArithmetic Mean Yearly Return:")
print(f"- Switch: {mean_sw:.4%}")
print(f"- Stoploss Nextday (Guard): {mean_gd:.4%}")
print(f"- Difference: {mean_gd - mean_sw:+.4%}")

print(f"\nCompounded Total Return (4 Windows):")
print(f"- Switch: {total_sw:.4%}")
print(f"- Stoploss Nextday (Guard): {total_gd:.4%}")
print(f"- Difference: {total_gd - total_sw:+.4%}")
