from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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


def init_worker(parquet_path: str, train_start: str | None = None, train_end: str | None = None) -> None:
    global BASE_DATA, FEATURE_CACHE
    BASE_DATA = main.load_and_forward_adjust(Path(parquet_path))
    if train_start is not None and train_end is not None:
        start_ts = pd.Timestamp(train_start)
        end_ts = pd.Timestamp(train_end)
        BASE_DATA = BASE_DATA.loc[(BASE_DATA.index >= start_ts) & (BASE_DATA.index <= end_ts)]
    FEATURE_CACHE = {}


def normalize_param(p: dict) -> dict:
    out = dict(p)
    if out["tier2_tp_atr_mult"] < out["tier1_tp_atr_mult"]:
        out["tier2_tp_atr_mult"] = out["tier1_tp_atr_mult"]
    if out["high_vol_max_holding_days"] > out["max_holding_days"]:
        out["high_vol_max_holding_days"] = out["max_holding_days"]
    return out


def eval_one(param: dict) -> dict:
    global BASE_DATA, FEATURE_CACHE
    p = normalize_param(param)

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


def generate_unique_params(n: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    keys = list(SPACE.keys())
    seen = set()
    out = []

    while len(out) < n:
        cand = {k: rng.choice(SPACE[k]).item() for k in keys}
        cand = normalize_param(cand)
        key = tuple(cand[k] for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def run_parallel_scan(
    n_evals: int = 10000,
    workers: int | None = None,
    seed: int = 42,
    train_start: str | None = None,
    train_end: str | None = None,
) -> pd.DataFrame:
    workers = workers or os.cpu_count() or 4
    params = generate_unique_params(n_evals, seed=seed)

    start = time.time()
    rows = []
    parquet_path = str(Path("512890.SH.parquet").resolve())

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(parquet_path, train_start, train_end),
    ) as ex:
        futures = [ex.submit(eval_one, p) for p in params]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if i % 500 == 0 or i == n_evals:
                print(f"增强策略并行扫描进度: {i}/{n_evals}, 耗时 {time.time()-start:.1f}s")

    return pd.DataFrame(rows).sort_values("Return [%]", ascending=False).reset_index(drop=True)


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
    main.configure_chinese_font()

    n_evals = 6000
    n_splits = 8
    train_ratio = 0.55
    workers = os.cpu_count() or 4
    print(f"使用进程数: {workers}")
    print(f"每窗口扫描组数: {n_evals}")
    print(f"滚动验证窗口数: {n_splits}")

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
            f"\n===== Adaptive MP 滚动验证 {i}/{len(splits)} =====\n"
            f"训练区间: {train_df.index[0].date()} -> {train_df.index[-1].date()}\n"
            f"验证区间: {val_df.index[0].date()} -> {val_df.index[-1].date()}"
        )

        scan_df = run_parallel_scan(
            n_evals=n_evals,
            workers=workers,
            seed=42 + i,
            train_start=str(train_df.index[0].date()),
            train_end=str(train_df.index[-1].date()),
        )
        best = scan_df.iloc[0].to_dict()

        scan_df.head(200).to_csv(f"adaptive_wf_{i:02d}_scan_top200_mp.csv", index=False, encoding="utf-8-sig")

        val_stats, val_data = run_adaptive_backtest(val_df, best)

        annual_plot = Path(f"adaptive_wf_{i:02d}_annual_return_comparison.png")
        daily_plot = Path(f"adaptive_wf_{i:02d}_daily_cumulative_return_comparison.png")
        annual_csv = Path(f"adaptive_wf_{i:02d}_annual_return_comparison.csv")
        daily_csv = Path(f"adaptive_wf_{i:02d}_daily_cumulative_return_comparison.csv")

        annual_df = main.plot_annual_return_comparison(
            strategy_equity_curve=val_stats["_equity_curve"],
            benchmark_close=val_data["Close"],
            title=f"增强策略滚动验证{i}: 策略 vs 长期持有（年度独立收益）",
            output_path=annual_plot,
        )
        daily_df = main.plot_daily_cumulative_return_comparison(
            strategy_equity_curve=val_stats["_equity_curve"],
            benchmark_close=val_data["Close"],
            title=f"增强策略滚动验证{i}: 策略 vs 长期持有（每日累计收益）",
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
        }
        rows.append(row)

        print(f"验证窗口{i}超额收益: {metrics['超额收益'] * 100:.2f}%")
        print(f"验证窗口{i}最大回撤: {metrics['最大回撤'] * 100:.2f}%")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("adaptive_walk_forward_summary_mp.csv", index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        print("\n===== Adaptive MP 稳健性汇总 =====")
        print(f"验证窗口数: {len(summary_df)}")
        print(f"平均超额收益: {summary_df['超额收益'].mean() * 100:.2f}%")
        print(f"超额收益标准差: {summary_df['超额收益'].std(ddof=0) * 100:.2f}%")
        print(f"平均最大回撤: {summary_df['最大回撤'].mean() * 100:.2f}%")
    print("已导出: adaptive_walk_forward_summary_mp.csv")


if __name__ == "__main__":
    main_run()
