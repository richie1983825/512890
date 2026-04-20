from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_core.data import load_and_forward_adjust, resolve_data_path
from backtest_core.parameters import build_polyfit_ma_switch_param_space
from backtest_core.reporting import configure_chinese_font
from backtest_core.workflows import run_polyfit_stoploss_nextday_guard_switch_comparison_3y1y


PREVIOUS_SUMMARY_PATH = Path("reports/global_best_switch_ma20_60_hold45_seed342/wf3y1y_polyfit_switch_strategy_summary.csv")
REPORT_ROOT = Path("reports/switch_stoploss_nextday_guard_scan_seed342")
SUMMARY_FILENAME = "wf3y1y_polyfit_stoploss_nextday_guard_switch_strategy_summary.csv"


def _load_previous_yearly_returns(summary_path: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_path)
    out = pd.DataFrame()
    out["year"] = summary_df["window"].astype(str).str.slice(0, 4).astype(int)
    out["previous_best_return"] = pd.to_numeric(summary_df["switch_total_return"], errors="coerce")
    return out.dropna().sort_values("year").reset_index(drop=True)


def _load_new_yearly_returns(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["year"] = pd.to_datetime(summary_df["验证开始"]).dt.year.astype(int)
    out["new_best_return"] = pd.to_numeric(summary_df["guard_switch_总收益率"], errors="coerce")
    return out.dropna().sort_values("year").reset_index(drop=True)


def _plot_yearly_return_comparison(compare_df: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(compare_df))
    width = 0.36

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, compare_df["previous_best_return"] * 100, width=width, color="#4c78a8", label="原始切换策略最优收益")
    plt.bar(x + width / 2, compare_df["new_best_return"] * 100, width=width, color="#f58518", label="止损次日偏离度保护策略最优收益")
    plt.axhline(0, color="#333333", linewidth=1, linestyle="--")
    plt.xticks(x, compare_df["year"].astype(str))
    plt.xlabel("年份")
    plt.ylabel("收益率 (%)")
    plt.title("止损次日偏离度保护策略与原始 Switch 策略年度最优收益对比")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _format_markdown_table(compare_df: pd.DataFrame) -> str:
    headers = ["year", "previous_best_return", "new_best_return", "return_diff"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in compare_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["year"])),
                    f"{float(row['previous_best_return']):.6f}",
                    f"{float(row['new_best_return']):.6f}",
                    f"{float(row['return_diff']):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_markdown_summary(compare_df: pd.DataFrame, output_path: Path, chart_path: Path) -> None:
    avg_prev = compare_df["previous_best_return"].mean()
    avg_new = compare_df["new_best_return"].mean()
    avg_diff = compare_df["return_diff"].mean()

    lines = [
        "# 止损次日偏离度保护策略与原始 Switch 策略年度最优收益对比",
        "",
        f"- 原始切换策略平均收益: {avg_prev:.2%}",
        f"- 止损次日偏离度保护策略平均收益: {avg_new:.2%}",
        f"- 平均收益差: {avg_diff:.2%}",
        "",
        f"![yearly comparison]({chart_path.name})",
        "",
        _format_markdown_table(compare_df),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_chinese_font()

    if not PREVIOUS_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"未找到原始策略汇总: {PREVIOUS_SUMMARY_PATH}")

    data_path = resolve_data_path()
    base_data = load_and_forward_adjust(data_path)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_df = run_polyfit_stoploss_nextday_guard_switch_comparison_3y1y(
        base_data,
        switch_param_space=build_polyfit_ma_switch_param_space(),
        guard_switch_param_space=build_polyfit_ma_switch_param_space(),
        max_evals=800,
        random_seed=342,
        generate_artifacts=True,
        reports_dir=REPORT_ROOT,
    )
    summary_path = REPORT_ROOT / SUMMARY_FILENAME
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    previous_df = _load_previous_yearly_returns(PREVIOUS_SUMMARY_PATH)
    new_df = _load_new_yearly_returns(summary_df)
    compare_df = previous_df.merge(new_df, on="year", how="inner")
    compare_df["return_diff"] = compare_df["new_best_return"] - compare_df["previous_best_return"]

    compare_csv = REPORT_ROOT / "switch_stoploss_nextday_guard_yearly_comparison.csv"
    compare_png = REPORT_ROOT / "switch_stoploss_nextday_guard_yearly_comparison.png"
    compare_md = REPORT_ROOT / "switch_stoploss_nextday_guard_yearly_comparison.md"
    compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")
    _plot_yearly_return_comparison(compare_df, compare_png)
    _write_markdown_summary(compare_df, compare_md, compare_png)

    print("\n===== 止损次日偏离度保护策略 vs 原始 Switch 策略 =====")
    print(compare_df.to_string(index=False))
    print(f"\n新扫描汇总: {summary_path}")
    print(f"年度对比CSV: {compare_csv}")
    print(f"年度对比图: {compare_png}")
    print(f"年度对比Markdown: {compare_md}")


if __name__ == "__main__":
    main()