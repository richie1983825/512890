from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd
from backtesting import Backtest

import main

BASE_DATA = None
FEATURE_CACHE = {}

SPACE = {
    "ma_window_weeks": [8, 10],
    "atr_window_weeks": [6, 8, 10],
    "buy_atr_mult": [0.5, 0.75, 1.0],
    "tier1_tp_atr_mult": [0.5, 0.75, 1.0],
    "tier2_tp_atr_mult": [1.5, 2.0, 2.5],
    "stop_atr_mult": [1.2, 1.5, 2.0],
    "trailing_atr_mult": [1.5, 2.0, 2.5],
    "max_holding_days": [20, 30, 40],
    "reversal_rsi_max": [50, 55, 60],
    "panic_daily_drop": [-0.015, -0.02, -0.025],
    "panic_gap_drop": [-0.01, -0.015, -0.02],
    "panic_atr_mult": [1.0, 1.5, 2.0],
    "momentum_breakout_atr_mult": [0.2, 0.35, 0.5],
    "momentum_rsi_max": [68, 72, 76],
    "high_vol_ratio_threshold": [1.0, 1.2, 1.4],
    "high_vol_stop_atr_mult": [1.0, 1.2, 1.5],
    "high_vol_trailing_atr_mult": [1.5, 1.8, 2.2],
    "high_vol_max_holding_days": [15, 20, 25],
    "max_flat_days": [10, 20, 30],
}

PARAM_KEYS = list(SPACE.keys())
RESULT_KEYS = PARAM_KEYS + ["Return [%]", "Return (Ann.) [%]", "Max. Drawdown [%]", "# Trades"]


def init_worker(parquet_path: str, train_start: str | None = None, train_end: str | None = None) -> None:
    global BASE_DATA, FEATURE_CACHE
    BASE_DATA = main.load_and_forward_adjust(Path(parquet_path))
    if train_start is not None and train_end is not None:
        start_ts = pd.Timestamp(train_start)
        end_ts = pd.Timestamp(train_end)
        BASE_DATA = BASE_DATA.loc[(BASE_DATA.index >= start_ts) & (BASE_DATA.index <= end_ts)]
    FEATURE_CACHE = {}


def valid_combo(p: dict) -> bool:
    if p["tier2_tp_atr_mult"] < p["tier1_tp_atr_mult"]:
        return False
    if p["high_vol_max_holding_days"] > p["max_holding_days"]:
        return False
    return True


def param_iter():
    for vals in product(*(SPACE[k] for k in PARAM_KEYS)):
        p = dict(zip(PARAM_KEYS, vals, strict=False))
        if valid_combo(p):
            yield p


def eval_one(p: dict) -> dict:
    global BASE_DATA, FEATURE_CACHE
    cache_key = (int(p["ma_window_weeks"]), int(p["atr_window_weeks"]))
    if cache_key not in FEATURE_CACHE:
        FEATURE_CACHE[cache_key] = main.add_strategy_features(BASE_DATA, cache_key[0], cache_key[1])

    d = FEATURE_CACHE[cache_key]
    bt = Backtest(
        d,
        main.AdaptiveShockReversalStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    kwargs = {k: v for k, v in p.items() if k not in {"ma_window_weeks", "atr_window_weeks"}}
    st = bt.run(**kwargs)

    return {
        **p,
        "Return [%]": float(st["Return [%]"]),
        "Return (Ann.) [%]": float(st["Return (Ann.) [%]"]),
        "Max. Drawdown [%]": float(st["Max. Drawdown [%]"]),
        "# Trades": int(st["# Trades"]),
    }


def count_valid_total() -> int:
    total = 0
    for _ in param_iter():
        total += 1
    return total


def append_rows(csv_path: Path, rows: list[dict], write_header: bool = False) -> None:
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_KEYS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_full_scan(
    workers: int,
    train_start: str | None = None,
    train_end: str | None = None,
    out_csv: Path | None = None,
    best_txt: Path | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    parquet_path = str(Path("512890.SH.parquet").resolve())
    total = count_valid_total()
    print(f"进程数: {workers}")
    print(f"全量有效组合数: {total}")

    rows: list[dict] = []
    best_row = None
    processed = 0
    start = time.time()

    if out_csv is not None and out_csv.exists():
        out_csv.unlink()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(parquet_path, train_start, train_end),
    ) as ex:
        for row in ex.map(eval_one, param_iter(), chunksize=64):
            processed += 1
            rows.append(row)

            if best_row is None or row["Return [%]"] > best_row["Return [%]"]:
                best_row = row

            if out_csv is not None and len(rows) % 5000 == 0:
                append_rows(out_csv, rows[-5000:], write_header=not out_csv.exists())

            if processed % 50000 == 0 or processed == total:
                elapsed = time.time() - start
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / speed if speed > 0 else 0
                print(
                    f"全量并行进度: {processed}/{total}, "
                    f"耗时 {elapsed/3600:.2f}h, 速度 {speed:.1f}组/s, ETA {eta/3600:.2f}h"
                )
                if best_txt is not None and best_row is not None:
                    best_txt.write_text(
                        "\n".join([f"{k}: {best_row[k]}" for k in RESULT_KEYS]),
                        encoding="utf-8",
                    )

    df = pd.DataFrame(rows).sort_values("Return [%]", ascending=False).reset_index(drop=True)
    if out_csv is not None:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    print(f"全量扫描完成，总耗时 {elapsed/3600:.2f}h")
    return df, best_row


def run_adaptive_backtest(data: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.DataFrame]:
    bt_data = main.add_strategy_features(
        data,
        int(params["ma_window_weeks"]),
        int(params["atr_window_weeks"]),
    )
    bt = Backtest(
        bt_data,
        main.AdaptiveShockReversalStrategy,
        cash=100000,
        commission=0.0001,
        exclusive_orders=True,
        finalize_trades=True,
    )
    kwargs = {k: params[k] for k in SPACE.keys() if k not in {"ma_window_weeks", "atr_window_weeks"}}
    stats = bt.run(**kwargs)
    return stats, bt_data


def main_run() -> None:
    workers = os.cpu_count() or 4
    n_splits = 6
    train_ratio = 0.55

    base_data = main.load_and_forward_adjust(Path("512890.SH.parquet"))
    splits = main.build_walk_forward_splits(
        base_data,
        n_splits=n_splits,
        train_ratio=train_ratio,
        min_val_days=84,
    )

    rows: list[dict] = []
    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        train_df = base_data.iloc[train_start:train_end]
        val_df = base_data.iloc[val_start:val_end]

        print(
            f"\n===== Adaptive FullMP 滚动验证 {i}/{len(splits)} =====\n"
            f"训练区间: {train_df.index[0].date()} -> {train_df.index[-1].date()}\n"
            f"验证区间: {val_df.index[0].date()} -> {val_df.index[-1].date()}"
        )

        scan_csv = Path(f"adaptive_full_wf_{i:02d}_scan.csv")
        best_txt = Path(f"adaptive_full_wf_{i:02d}_best.txt")
        scan_df, best = run_full_scan(
            workers=workers,
            train_start=str(train_df.index[0].date()),
            train_end=str(train_df.index[-1].date()),
            out_csv=scan_csv,
            best_txt=best_txt,
        )
        if best is None:
            continue

        val_stats, val_data = run_adaptive_backtest(val_df, best)

        annual_plot = Path(f"adaptive_full_wf_{i:02d}_annual_return_comparison.png")
        daily_plot = Path(f"adaptive_full_wf_{i:02d}_daily_cumulative_return_comparison.png")
        annual_csv = Path(f"adaptive_full_wf_{i:02d}_annual_return_comparison.csv")
        daily_csv = Path(f"adaptive_full_wf_{i:02d}_daily_cumulative_return_comparison.csv")

        annual_df = main.plot_annual_return_comparison(
            strategy_equity_curve=val_stats["_equity_curve"],
            benchmark_close=val_data["Close"],
            title=f"增强策略全量扫描滚动验证{i}: 策略 vs 长期持有（年度独立收益）",
            output_path=annual_plot,
        )
        daily_df = main.plot_daily_cumulative_return_comparison(
            strategy_equity_curve=val_stats["_equity_curve"],
            benchmark_close=val_data["Close"],
            title=f"增强策略全量扫描滚动验证{i}: 策略 vs 长期持有（每日累计收益）",
            output_path=daily_plot,
            trades=val_stats["_trades"],
        )
        annual_df.to_csv(annual_csv, index=True, encoding="utf-8-sig")
        daily_df.to_csv(daily_csv, index=True, encoding="utf-8-sig")

        metrics = main.summarize_backtest_metrics(val_stats, val_data["Close"])
        row = {
            "窗口": i,
            "训练开始": str(train_df.index[0].date()),
            "训练结束": str(train_df.index[-1].date()),
            "验证开始": str(val_df.index[0].date()),
            "验证结束": str(val_df.index[-1].date()),
            **{k: best[k] for k in SPACE.keys()},
            **metrics,
            "训练扫描文件": str(scan_csv),
        }
        rows.append(row)

        print(f"验证窗口{i}超额收益: {metrics['超额收益'] * 100:.2f}%")
        print(f"验证窗口{i}最大回撤: {metrics['最大回撤'] * 100:.2f}%")

        scan_df.head(200).to_csv(f"adaptive_full_wf_{i:02d}_scan_top200.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("adaptive_walk_forward_summary_full_mp.csv", index=False, encoding="utf-8-sig")
    if not summary_df.empty:
        print("\n===== Adaptive FullMP 稳健性汇总 =====")
        print(f"验证窗口数: {len(summary_df)}")
        print(f"平均超额收益: {summary_df['超额收益'].mean() * 100:.2f}%")
        print(f"超额收益标准差: {summary_df['超额收益'].std(ddof=0) * 100:.2f}%")
        print(f"平均最大回撤: {summary_df['最大回撤'].mean() * 100:.2f}%")
    print("已导出: adaptive_walk_forward_summary_full_mp.csv")


if __name__ == "__main__":
    main_run()
