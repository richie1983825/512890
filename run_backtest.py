import pandas as pd
from pathlib import Path
from backtest_core.data import load_and_forward_adjust, resolve_data_path
from backtest_core.parameters import build_param_space, build_polyfit_ma_switch_param_space
from backtest_core.reporting import configure_chinese_font
from backtest_core.workflows import run_polyfit_switch_comparison_3y1y

def main():
    # 2) 调用 configure_chinese_font
    configure_chinese_font()

    # 1) 从 backtest_core.data 加载数据
    data_path = resolve_data_path()
    base_data = load_and_forward_adjust(data_path)

    # 5) 调用 run_polyfit_switch_comparison_3y1y
    # 3) 使用 build_param_space() 作为 polyfit 参数空间
    # 4) 使用 build_polyfit_ma_switch_param_space() 作为切换策略参数空间
    summary_df = run_polyfit_switch_comparison_3y1y(
        base_data,
        polyfit_param_space=build_param_space(),
        switch_param_space=build_polyfit_ma_switch_param_space(),
        max_evals=800,
        random_seed=42,
        generate_artifacts=True
    )

    # 6) 将结果保存到 reports/wf3y1y_polyfit_switch_strategy_summary.csv
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    save_path = reports_dir / "wf3y1y_polyfit_switch_strategy_summary.csv"
    summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    # 7) 在标准输出中打印该 summary 的关键列
    cols_to_print = [
        '窗口', 
        'switch_flat_wait_days', 
        'switch_switch_deviation_m1', 
        'switch_switch_deviation_m2', 
        'polyfit_总收益率', 
        'switch_总收益率', 
        'switch优于polyfit_总收益差'
    ]
    
    # Check if columns exist before printing to avoid errors if names differ slightly
    available_cols = [c for c in cols_to_print if c in summary_df.columns]
    print("\n===== 回测结果摘要 =====")
    print(summary_df[available_cols].to_string(index=False))

    # 8) 再打印整体平均 polyfit_总收益率、整体平均 switch_总收益率、平均总收益差
    print("\n===== 整体平均指标 =====")
    if 'polyfit_总收益率' in summary_df.columns:
        print(f"整体平均 polyfit_总收益率: {summary_df['polyfit_总收益率'].mean():.4f}")
    if 'switch_总收益率' in summary_df.columns:
        print(f"整体平均 switch_总收益率: {summary_df['switch_总收益率'].mean():.4f}")
    if 'switch优于polyfit_总收益差' in summary_df.columns:
        print(f"平均总收益差: {summary_df['switch优于polyfit_总收益差'].mean():.4f}")

if __name__ == "__main__":
    main()
