from __future__ import annotations

from pathlib import Path

from backtest_core.data import load_and_forward_adjust, resolve_data_path
from backtest_core.parameters import build_ma_param_space, build_param_space
from backtest_core.reporting import configure_chinese_font
from backtest_core.workflows import run_polyfit_ma_comparison_3y1y


def main() -> None:
    configure_chinese_font()

    data_path = resolve_data_path()
    base_data = load_and_forward_adjust(data_path)

    print("执行双策略对比: 回归策略 vs MA基准策略，并以长期持有作为共同基准。")
    summary_df = run_polyfit_ma_comparison_3y1y(
        base_data,
        polyfit_param_space=build_param_space(),
        ma_param_space=build_ma_param_space(),
        max_evals=800,
        random_seed=42,
        generate_artifacts=True,
    )

    reports = Path(__file__).resolve().parent / "reports"
    summary_path = reports / "wf3y1y_polyfit_ma_strategy_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n===== 双策略 3年训练1年验证汇总 =====")
    print(f"窗口数: {len(summary_df)}")
    if not summary_df.empty:
        print(f"回归策略平均年化收益率: {summary_df['polyfit_年化收益率'].mean() * 100:.2f}%")
        print(f"MA基准策略平均年化收益率: {summary_df['ma_年化收益率'].mean() * 100:.2f}%")
        print(f"MA优于回归平均总收益差: {summary_df['MA优于回归_总收益差'].mean() * 100:.2f}%")
    print(f"已导出: {summary_path}")


if __name__ == "__main__":
    main()