from __future__ import annotations

from pathlib import Path

import pandas as pd

from main import main as run_main_workflow


SUMMARY_PATH = Path(__file__).resolve().parent / "reports" / "wf3y1y_polyfit_ma_strategy_summary.csv"


def load_or_build_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        run_main_workflow()
    return pd.read_csv(SUMMARY_PATH)


def print_best_windows(summary_df: pd.DataFrame) -> None:
    polyfit_cols = ["窗口", "验证开始", "验证结束", "polyfit_总收益率", "polyfit_最大回撤", "polyfit_交易次数"]
    ma_cols = ["窗口", "验证开始", "验证结束", "ma_总收益率", "ma_最大回撤", "ma_交易次数"]

    best_polyfit = summary_df.sort_values("polyfit_总收益率", ascending=False).head(5)
    best_ma = summary_df.sort_values("ma_总收益率", ascending=False).head(5)

    print("===== Polyfit Top Windows =====")
    print(best_polyfit[polyfit_cols].to_string(index=False))
    print("\n===== MA Top Windows =====")
    print(best_ma[ma_cols].to_string(index=False))


def main() -> None:
    summary_df = load_or_build_summary()
    print_best_windows(summary_df)


if __name__ == "__main__":
    main()