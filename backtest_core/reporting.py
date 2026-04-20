from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def configure_chinese_font() -> None:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "WenQuanYi Zen Hei",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    fallbacks = [name for name in candidates if name in available]

    if fallbacks:
        plt.rcParams["font.sans-serif"] = [*fallbacks, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def calc_independent_annual_returns(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    annual_ret = clean.resample("YE").apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    annual_ret.index = annual_ret.index.year
    return annual_ret


def plot_annual_return_comparison(
    strategy_equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
) -> pd.DataFrame:
    configure_chinese_font()
    strategy_annual = calc_independent_annual_returns(strategy_equity_curve["Equity"])
    buy_hold_annual = calc_independent_annual_returns(benchmark_close)

    compare = pd.DataFrame({"策略收益": strategy_annual, "长期持有": buy_hold_annual}).dropna(how="all")
    if compare.empty:
        return compare

    x = np.arange(len(compare.index))
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, compare["策略收益"].values * 100, width=width, color="#2ca02c", label="策略收益")
    plt.bar(x + width / 2, compare["长期持有"].values * 100, width=width, color="#1f77b4", label="长期持有")
    plt.axhline(0, color="#333333", linewidth=1, linestyle="--", label="基准线(0%)")
    plt.title(title)
    plt.xlabel("年份")
    plt.ylabel("收益率 (%)")
    plt.xticks(x, compare.index.astype(str))
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return compare


def calc_daily_position_ratio(
    strategy_equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
    trades: pd.DataFrame | None = None,
    strategy_obj: object | None = None,
) -> pd.Series:
    index = strategy_equity_curve.index
    position_ratio = pd.Series(0.0, index=index, dtype=float)
    trade_frames: list[pd.DataFrame] = []
    if trades is not None and len(trades) > 0:
        trade_df = trades.copy()
        trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
        trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
        trade_df["IsOpen"] = False
        trade_frames.append(trade_df)

    open_trade_df = _build_open_trade_frame(strategy_obj, benchmark_close)
    if len(open_trade_df) > 0:
        trade_frames.append(open_trade_df)

    if not trade_frames:
        return position_ratio

    trade_df = pd.concat(trade_frames, ignore_index=True, sort=False)
    close = benchmark_close.reindex(index).ffill().bfill()
    equity = strategy_equity_curve["Equity"].reindex(index).ffill().bfill()

    for _, trade in trade_df.iterrows():
        entry_time = pd.Timestamp(trade["EntryTime"])
        exit_time = pd.Timestamp(trade["ExitTime"])
        size = float(abs(trade.get("Size", 0.0)))
        if size <= 0:
            continue
        if bool(trade.get("IsOpen", False)):
            active_mask = (index >= entry_time) & (index <= exit_time)
        else:
            active_mask = (index >= entry_time) & (index < exit_time)
        if not active_mask.any():
            continue
        notional = size * close.loc[active_mask]
        position_ratio.loc[active_mask] += notional / equity.loc[active_mask].replace(0.0, np.nan)

    return position_ratio.fillna(0.0).clip(lower=0.0)


def _build_open_trade_frame(
    strategy_obj: object | None,
    bt_data: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    if strategy_obj is None:
        return pd.DataFrame()

    open_trades = getattr(strategy_obj, "trades", None)
    if not open_trades:
        return pd.DataFrame()

    if isinstance(bt_data, pd.DataFrame):
        end_index = bt_data.index
        close_series = bt_data["Close"]
    else:
        end_index = bt_data.index
        close_series = bt_data

    if len(end_index) == 0:
        return pd.DataFrame()

    end_time = pd.Timestamp(end_index[-1])
    end_close = float(close_series.iloc[-1])
    current_entry_reason = getattr(strategy_obj, "current_entry_reason", "") or "entry_runtime_unknown"

    rows: list[dict[str, object]] = []
    for trade in open_trades:
        size = float(abs(getattr(trade, "size", 0.0) or 0.0))
        if size <= 0:
            continue
        rows.append(
            {
                "EntryTime": pd.Timestamp(getattr(trade, "entry_time", end_time)),
                "ExitTime": end_time,
                "Size": size,
                "EntryPrice": float(getattr(trade, "entry_price", end_close) or end_close),
                "ExitPrice": end_close,
                "PnL": float(getattr(trade, "pl", 0.0) or 0.0),
                "ReturnPct": float(getattr(trade, "pl_pct", 0.0) or 0.0),
                "EntryBar": getattr(trade, "entry_bar", np.nan),
                "ExitBar": len(end_index) - 1,
                "Tag": getattr(trade, "tag", None),
                "EntryReason": current_entry_reason,
                "ExitReason": "open_at_period_end",
                "TradeStatus": "open_at_period_end",
                "IsOpen": True,
            }
        )

    return pd.DataFrame(rows)


def _title_with_year(title: str, index: pd.Index) -> str:
    dt_index = pd.to_datetime(index)
    start_year = int(dt_index.min().year)
    end_year = int(dt_index.max().year)
    year_text = f"{start_year}年" if start_year == end_year else f"{start_year}-{end_year}年"
    return f"{year_text} {title}"


def _configure_trading_day_axis(ax: plt.Axes, index: pd.Index) -> None:
    dt_index = pd.to_datetime(index)
    if len(dt_index) == 0:
        return

    x_count = len(dt_index)
    tick_count = min(12, x_count)
    tick_positions = np.linspace(0, x_count - 1, num=tick_count, dtype=int)
    tick_positions = np.unique(tick_positions)
    tick_labels = [pd.Timestamp(dt_index[pos]).strftime("%Y-%m-%d") for pos in tick_positions]

    ax.set_xlim(0, x_count - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.grid(alpha=0.2, axis="x")


def _time_to_trading_day_positions(index: pd.Index, ts_series: pd.Series) -> np.ndarray:
    dt_index = pd.to_datetime(index)
    dt_series = pd.to_datetime(ts_series)
    positions = dt_index.get_indexer(dt_series, method="nearest")
    return positions[positions >= 0]


def plot_daily_cumulative_return_comparison(
    strategy_equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
    trades: pd.DataFrame | None = None,
    strategy_obj: object | None = None,
    baseline_series: pd.Series | None = None,
    baseline_label: str = "策略基准累计收益",
    signal_on_benchmark_curve: bool = True,
) -> pd.DataFrame:
    configure_chinese_font()
    strategy_daily = strategy_equity_curve["Equity"].pct_change().fillna(0.0)
    buy_hold_daily = benchmark_close.pct_change().fillna(0.0)
    strategy_cum = (1 + strategy_daily).cumprod() - 1
    buy_hold_cum = (1 + buy_hold_daily).cumprod() - 1
    compare = pd.DataFrame({"策略累计收益": strategy_cum, "长期持有累计收益": buy_hold_cum})
    compare["策略仓位"] = calc_daily_position_ratio(
        strategy_equity_curve,
        benchmark_close,
        trades=trades,
        strategy_obj=strategy_obj,
    )

    if baseline_series is not None:
        baseline = baseline_series.reindex(compare.index).ffill().bfill()
        baseline_daily = baseline.pct_change().fillna(0.0)
        baseline_cum = (1 + baseline_daily).cumprod() - 1
        compare[baseline_label] = baseline_cum

    fig, (ax_main, ax_pos) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    x = np.arange(len(compare))
    ax_main.plot(x, compare["策略累计收益"].values * 100, label="策略累计收益", color="#2ca02c", linewidth=1.3)
    ax_main.plot(x, compare["长期持有累计收益"].values * 100, label="长期持有累计收益", color="#1f77b4", linewidth=1.1)
    if baseline_series is not None and baseline_label in compare.columns:
        ax_main.plot(x, compare[baseline_label].values * 100, label=baseline_label, color="#ff7f0e", linewidth=1.1, linestyle="--")

    trade_frames: list[pd.DataFrame] = []
    if trades is not None and len(trades) > 0:
        trade_df = trades.copy()
        trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
        trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
        trade_df["IsOpen"] = False
        trade_frames.append(trade_df)

    open_trade_df = _build_open_trade_frame(strategy_obj, benchmark_close)
    if len(open_trade_df) > 0:
        trade_frames.append(open_trade_df)

    if trade_frames:
        trade_df = pd.concat(trade_frames, ignore_index=True, sort=False)
        marker_line = compare["长期持有累计收益"] if signal_on_benchmark_curve else compare["策略累计收益"]

        entry_pos = _time_to_trading_day_positions(compare.index, trade_df["EntryTime"])
        closed_trade_df = trade_df.loc[~trade_df.get("IsOpen", False).fillna(False)]
        exit_pos = _time_to_trading_day_positions(compare.index, closed_trade_df["ExitTime"])
        open_pos = _time_to_trading_day_positions(
            compare.index,
            trade_df.loc[trade_df.get("IsOpen", False).fillna(False), "ExitTime"],
        )

        if len(entry_pos) > 0:
            ax_main.scatter(entry_pos, marker_line.iloc[entry_pos].values * 100, marker="^", color="#d62728", s=28, label="买点", zorder=5)
        if len(exit_pos) > 0:
            ax_main.scatter(exit_pos, marker_line.iloc[exit_pos].values * 100, marker="v", color="#9467bd", s=28, label="卖点", zorder=5)
        if len(open_pos) > 0:
            ax_main.scatter(open_pos, marker_line.iloc[open_pos].values * 100, marker="s", color="#8c564b", s=32, label="期末持有", zorder=6)

    ax_main.axhline(0, color="#333333", linewidth=1, linestyle="--", label="基准线(0%)")
    ax_main.set_title(_title_with_year(title, compare.index))
    ax_main.set_ylabel("累计收益率 (%)")
    ax_main.legend()
    ax_main.grid(alpha=0.2)

    ax_pos.plot(x, compare["策略仓位"].values * 100, color="#6a9f6a", linewidth=1.0)
    ax_pos.fill_between(x, 0, compare["策略仓位"].values * 100, color="#8fbf8f", alpha=0.35)
    ax_pos.set_ylabel("仓位 (%)")
    ax_pos.set_xlabel("交易日")
    ax_pos.grid(alpha=0.2, axis="y")
    ax_pos.set_ylim(0, max(100.0, float(compare["策略仓位"].max() * 110.0) if not compare.empty else 100.0))
    _configure_trading_day_axis(ax_pos, compare.index)

    fig.subplots_adjust(hspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return compare


def plot_multi_strategy_cumulative_comparison(
    strategy_curves: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    title: str,
    output_path: Path,
) -> pd.DataFrame:
    configure_chinese_font()
    compare_data: dict[str, pd.Series] = {}
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#8c564b", "#17becf"]
    plt.figure(figsize=(14, 6))

    for label, curve in strategy_curves.items():
        daily = curve["Equity"].pct_change().fillna(0.0)
        cum = (1 + daily).cumprod() - 1
        compare_data[label] = cum

    buy_hold_daily = benchmark_close.pct_change().fillna(0.0)
    compare_data["长期持有累计收益"] = (1 + buy_hold_daily).cumprod() - 1
    compare = pd.DataFrame(compare_data)
    x = np.arange(len(compare))

    for idx, label in enumerate(strategy_curves.keys()):
        plt.plot(x, compare[label].values * 100, label=label, color=colors[idx % len(colors)], linewidth=1.3)

    plt.plot(x, compare["长期持有累计收益"].values * 100, label="长期持有累计收益", color="#1f77b4", linewidth=1.1)
    plt.axhline(0, color="#333333", linewidth=1, linestyle="--", label="基准线(0%)")
    plt.title(_title_with_year(title, compare.index))
    plt.xlabel("交易日")
    plt.ylabel("累计收益率 (%)")
    plt.legend()
    plt.grid(alpha=0.2)
    _configure_trading_day_axis(plt.gca(), compare.index)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return compare


def print_daily_cumulative_returns_with_signals(
    compare: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    strategy_obj: object | None = None,
    label: str | None = None,
    max_rows: int = 120,
) -> None:
    if compare.empty:
        print("每日累计收益为空，跳过打印")
        return

    report = compare.copy()
    report["买点"] = ""
    report["卖点"] = ""
    report["持有状态"] = ""

    trade_frames: list[pd.DataFrame] = []
    if trades is not None and len(trades) > 0:
        trade_df = trades.copy()
        trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
        trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
        trade_df["IsOpen"] = False
        trade_frames.append(trade_df)

    open_trade_df = _build_open_trade_frame(strategy_obj, compare["策略累计收益"])
    if len(open_trade_df) > 0:
        trade_frames.append(open_trade_df)

    if trade_frames:
        trade_df = pd.concat(trade_frames, ignore_index=True, sort=False)

        for ts, count in trade_df.groupby("EntryTime").size().items():
            pos = report.index.get_indexer([ts], method="nearest")[0]
            if pos >= 0:
                report.iloc[pos, report.columns.get_loc("买点")] = f"买入x{int(count)}"

        closed_trade_df = trade_df.loc[~trade_df.get("IsOpen", False).fillna(False)]
        for ts, count in closed_trade_df.groupby("ExitTime").size().items():
            pos = report.index.get_indexer([ts], method="nearest")[0]
            if pos >= 0:
                report.iloc[pos, report.columns.get_loc("卖点")] = f"卖出x{int(count)}"

        open_trade_count = int(trade_df.get("IsOpen", pd.Series(dtype=bool)).fillna(False).sum())
        if open_trade_count > 0 and len(report.index) > 0:
            report.iloc[-1, report.columns.get_loc("持有状态")] = f"期末持有x{open_trade_count}"

    out = report.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    title = label or "未命名窗口"
    print(f"\n===== {title} 每日累计收益与买卖点 =====")
    if len(out) <= max_rows:
        print(out.to_string())
        return

    head_n = max_rows // 2
    tail_n = max_rows - head_n
    print(f"总行数 {len(out)}，仅展示前 {head_n} 行和后 {tail_n} 行。")
    print(out.head(head_n).to_string())
    print("... (中间省略) ...")
    print(out.tail(tail_n).to_string())


def _nearest_bar_index(index: pd.Index, ts: pd.Timestamp) -> int:
    pos = index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
    return max(int(pos), 0)


def _next_bar_timestamp(index: pd.Index, ts: pd.Timestamp) -> pd.Timestamp:
    dt_index = pd.DatetimeIndex(index)
    loc = dt_index.get_indexer([pd.Timestamp(ts)])[0]
    if loc == -1:
        loc = dt_index.get_indexer([pd.Timestamp(ts)], method="ffill")[0]
    if loc == -1:
        loc = _nearest_bar_index(dt_index, pd.Timestamp(ts))
    if loc < len(dt_index) - 1:
        return pd.Timestamp(dt_index[loc + 1])
    return pd.Timestamp(dt_index[loc])


def _normalize_native_reason_records(native_df: pd.DataFrame, bt_index: pd.Index) -> pd.DataFrame:
    normalized = native_df.copy()
    for col in ["EntryTime", "ExitTime"]:
        if col in normalized.columns:
            normalized[col] = pd.to_datetime(normalized[col]).map(lambda ts: _next_bar_timestamp(bt_index, ts))
    return normalized


def _join_reasons(reasons: list[str], fallback: str) -> str:
    unique_reasons = []
    for reason in reasons:
        if reason and reason not in unique_reasons:
            unique_reasons.append(reason)
    return "; ".join(unique_reasons) if unique_reasons else fallback


def _infer_grid_entry_reason(
    row: pd.Series,
    params: dict,
    base_col: str,
    dev_col: str,
    trend_col: str,
) -> str:
    close = float(row["Close"])
    base_value = float(row[base_col])
    dev_pct = float(row[dev_col])
    dev_trend = float(row[trend_col])
    rolling_vol_pct = float(row["RollingVolPct"])
    vol_multiplier = 1.0 + float(params["volatility_scale"]) * max(rolling_vol_pct, 0.0)
    dynamic_grid_step = float(params["base_grid_pct"]) * (1.0 + float(params["trend_sensitivity"]) * abs(dev_trend)) * vol_multiplier
    dynamic_grid_step = max(dynamic_grid_step, float(params["base_grid_pct"]) * 0.3)
    signal_strength = abs(dev_pct) / max(dynamic_grid_step, 1e-9)
    entry_level = int(np.clip(np.floor(signal_strength), 1, int(params["max_grid_levels"])))
    return (
        f"mean_reversion_grid;base={base_value:.4f};dev={dev_pct:.4%};"
        f"grid_step={dynamic_grid_step:.4%};level={entry_level};close={close:.4f}"
    )


def _infer_grid_exit_reason(
    row: pd.Series,
    entry_row: pd.Series,
    params: dict,
    dev_col: str,
    trend_col: str,
    holding_days: int,
) -> str:
    dev_pct = float(row[dev_col])
    dev_trend = float(row[trend_col])
    rolling_vol_pct = float(row["RollingVolPct"])
    vol_multiplier = 1.0 + float(params["volatility_scale"]) * max(rolling_vol_pct, 0.0)
    dynamic_grid_step = float(params["base_grid_pct"]) * (1.0 + float(params["trend_sensitivity"]) * abs(dev_trend)) * vol_multiplier
    dynamic_grid_step = max(dynamic_grid_step, float(params["base_grid_pct"]) * 0.3)
    entry_signal = abs(float(entry_row[dev_col])) / max(dynamic_grid_step, 1e-9)
    entry_level = int(np.clip(np.floor(entry_signal), 1, int(params["max_grid_levels"])))
    ref_step = max(dynamic_grid_step, float(params["base_grid_pct"]))
    tp_threshold = entry_level * ref_step * float(params["take_profit_grid"])
    sl_threshold = entry_level * ref_step * float(params["stop_loss_grid"])
    reasons: list[str] = []
    if holding_days >= int(params["max_holding_days"]):
        reasons.append(f"max_holding_days({int(params['max_holding_days'])})")
    if dev_pct >= tp_threshold:
        reasons.append(f"take_profit_grid(dev={dev_pct:.4%}>=tp={tp_threshold:.4%})")
    if dev_pct <= -sl_threshold:
        reasons.append(f"stop_loss_grid(dev={dev_pct:.4%}<=-{sl_threshold:.4%})")
    return _join_reasons(
        reasons,
        (
            "grid_exit_no_rule_match("
            f"dev={dev_pct:.4%};holding_days={holding_days};"
            f"tp={tp_threshold:.4%};sl={sl_threshold:.4%})"
        ),
    )


def _infer_switch_entry_reason(row: pd.Series, params: dict) -> str:
    close = float(row["Close"])
    base_value = float(row["PolyBasePred"])
    dev_pct = float(row["PolyDevPct"])
    fast_window = int(params["switch_fast_ma_window"])
    slow_window = int(params["switch_slow_ma_window"])
    fast_ma = float(row[f"MA{fast_window}"])
    slow_ma = float(row[f"MA{slow_window}"])

    if close > base_value and dev_pct > float(params["switch_deviation_m1"]) and fast_ma > slow_ma:
        return (
            f"deviation_ma_switch_buy;base={base_value:.4f};close={close:.4f};dev={dev_pct:.4%};"
            f"ma{fast_window}={fast_ma:.4f};ma{slow_window}={slow_ma:.4f};m1={float(params['switch_deviation_m1']):.4%}"
        )

    return _infer_grid_entry_reason(row, params, "PolyBasePred", "PolyDevPct", "PolyDevTrend")


def _infer_switch_exit_reason(row: pd.Series, entry_row: pd.Series, params: dict, holding_days: int) -> str:
    close = float(row["Close"])
    base_value = float(row["PolyBasePred"])
    dev_pct = float(row["PolyDevPct"])
    fast_window = int(params["switch_fast_ma_window"])
    slow_window = int(params["switch_slow_ma_window"])
    fast_ma = float(row[f"MA{fast_window}"])
    slow_ma = float(row[f"MA{slow_window}"])

    if close > base_value and dev_pct > float(params["switch_deviation_m1"]) and fast_ma < slow_ma:
        return (
            f"deviation_ma_switch_sell;base={base_value:.4f};close={close:.4f};dev={dev_pct:.4%};"
            f"ma{fast_window}={fast_ma:.4f};ma{slow_window}={slow_ma:.4f};m1={float(params['switch_deviation_m1']):.4%}"
        )

    return _infer_grid_exit_reason(row, entry_row, params, "PolyDevPct", "PolyDevTrend", holding_days)


def infer_trade_record_reasons(
    trades: pd.DataFrame | None,
    bt_data: pd.DataFrame,
    strategy_name: str,
    params: dict,
) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["EntryReason", "ExitReason"])

    trade_df = trades.copy()
    entry_reasons: list[str] = []
    exit_reasons: list[str] = []
    index = bt_data.index

    for _, trade in trade_df.iterrows():
        entry_bar = int(trade["EntryBar"]) if "EntryBar" in trade else _nearest_bar_index(index, pd.Timestamp(trade["EntryTime"]))
        exit_bar = int(trade["ExitBar"]) if "ExitBar" in trade else _nearest_bar_index(index, pd.Timestamp(trade["ExitTime"]))
        entry_bar = int(np.clip(entry_bar, 0, len(bt_data) - 1))
        exit_bar = int(np.clip(exit_bar, 0, len(bt_data) - 1))
        entry_row = bt_data.iloc[entry_bar]
        exit_row = bt_data.iloc[exit_bar]
        holding_days = exit_bar - entry_bar

        if strategy_name == "polyfit":
            entry_reason = "initial_position_carry" if entry_bar == 0 else _infer_grid_entry_reason(entry_row, params, "PolyBasePred", "PolyDevPct", "PolyDevTrend")
            exit_reason = _infer_grid_exit_reason(exit_row, entry_row, params, "PolyDevPct", "PolyDevTrend", holding_days)
        elif strategy_name == "ma":
            entry_reason = "initial_position_carry" if entry_bar == 0 else _infer_grid_entry_reason(entry_row, params, "MABase", "MADevPct", "MADevTrend")
            exit_reason = _infer_grid_exit_reason(exit_row, entry_row, params, "MADevPct", "MADevTrend", holding_days)
        elif strategy_name == "polyfit_switch":
            entry_reason = "initial_position_carry" if entry_bar == 0 else _infer_switch_entry_reason(entry_row, params)
            exit_reason = _infer_switch_exit_reason(exit_row, entry_row, params, holding_days)
        else:
            entry_reason = "entry_unclassified"
            exit_reason = "exit_unclassified"

        entry_reasons.append(entry_reason)
        exit_reasons.append(exit_reason)

    return pd.DataFrame({"EntryReason": entry_reasons, "ExitReason": exit_reasons}, index=trade_df.index)


def export_trade_records_csv(
    trades: pd.DataFrame | None,
    output_path: Path,
    bt_data: pd.DataFrame | None = None,
    equity_curve: pd.DataFrame | None = None,
    strategy_name: str | None = None,
    params: dict | None = None,
    native_reason_records: list[dict] | None = None,
    strategy_obj: object | None = None,
) -> pd.DataFrame:
    open_trade_df = _build_open_trade_frame(strategy_obj, bt_data) if bt_data is not None else pd.DataFrame()

    if (trades is None or len(trades) == 0) and len(open_trade_df) == 0:
        empty = pd.DataFrame(
            columns=[
                "EntryTime",
                "ExitTime",
                "Size",
                "EntryPositionPct",
                "EntryPrice",
                "ExitPrice",
                "PnL",
                "ReturnPct",
                "HoldingDays",
                "TradeStatus",
                "EntryReason",
                "ExitReason",
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_path, index=False, encoding="utf-8-sig")
        return empty
    trade_frames: list[pd.DataFrame] = []
    if trades is not None and len(trades) > 0:
        trade_df = trades.copy()
        trade_df["EntryTime"] = pd.to_datetime(trade_df["EntryTime"])
        trade_df["ExitTime"] = pd.to_datetime(trade_df["ExitTime"])
        trade_df["TradeStatus"] = "closed"
        trade_df["IsOpen"] = False
        trade_frames.append(trade_df)
    if len(open_trade_df) > 0:
        trade_frames.append(open_trade_df)

    trade_df = pd.concat(trade_frames, ignore_index=True, sort=False)
    trade_df["HoldingDays"] = (trade_df["ExitTime"] - trade_df["EntryTime"]).dt.days.clip(lower=1)
    trade_df["EntryPositionPct"] = np.nan

    if equity_curve is not None and len(equity_curve) > 0 and "EntryBar" in trade_df.columns and "Equity" in equity_curve.columns:
        equity_values = equity_curve["Equity"].reset_index(drop=True)
        entry_bars = trade_df["EntryBar"].fillna(-1).astype(int)
        valid_mask = entry_bars.between(0, len(equity_values) - 1)
        if valid_mask.any():
            entry_equity = entry_bars.loc[valid_mask].map(equity_values)
            entry_position_value = trade_df.loc[valid_mask, "Size"].abs() * trade_df.loc[valid_mask, "EntryPrice"]
            trade_df.loc[valid_mask, "EntryPositionPct"] = entry_position_value / entry_equity.replace(0, np.nan)

    if native_reason_records is not None and len(native_reason_records) > 0:
        native_df = pd.DataFrame(native_reason_records).copy()
        for col in ["EntryTime", "ExitTime"]:
            if col in native_df.columns:
                native_df[col] = pd.to_datetime(native_df[col])
        native_keep = [col for col in ["EntryTime", "ExitTime", "EntryReason", "ExitReason"] if col in native_df.columns]
        if {"EntryTime", "ExitTime", "EntryReason", "ExitReason"}.issubset(native_keep):
            if bt_data is not None and len(bt_data.index) > 1:
                native_df = _normalize_native_reason_records(native_df[native_keep], bt_data.index)
            else:
                native_df = native_df[native_keep].copy()
            native_df = native_df.drop_duplicates(subset=["EntryTime", "ExitTime"], keep="last")
            trade_df = trade_df.merge(native_df, on=["EntryTime", "ExitTime"], how="left", suffixes=("", "_native"))
            if "EntryReason_native" in trade_df.columns:
                trade_df["EntryReason"] = trade_df.get("EntryReason").combine_first(trade_df["EntryReason_native"])
            if "ExitReason_native" in trade_df.columns:
                trade_df["ExitReason"] = trade_df.get("ExitReason").combine_first(trade_df["ExitReason_native"])
            trade_df = trade_df.drop(columns=[col for col in ["EntryReason_native", "ExitReason_native"] if col in trade_df.columns])

    if bt_data is not None and strategy_name is not None and params is not None:
        reason_df = infer_trade_record_reasons(trade_df, bt_data, strategy_name, params)
        if "EntryReason" in trade_df.columns:
            trade_df["EntryReason"] = trade_df["EntryReason"].fillna(reason_df["EntryReason"])
            trade_df["ExitReason"] = trade_df["ExitReason"].fillna(reason_df["ExitReason"])
        else:
            trade_df = pd.concat([trade_df, reason_df], axis=1)

    output_cols = [
        col
        for col in [
            "EntryTime",
            "ExitTime",
            "Size",
            "EntryPositionPct",
            "EntryPrice",
            "ExitPrice",
            "PnL",
            "ReturnPct",
            "HoldingDays",
            "TradeStatus",
            "EntryReason",
            "ExitReason",
            "Tag",
            "Duration",
        ]
        if col in trade_df.columns
    ]
    export_df = trade_df[output_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return export_df


def write_window_comparison_summary_markdown(
    output_path: Path,
    title: str,
    sections: list[tuple[str, list[Path]]],
) -> None:
    lines = [f"# {title}", ""]
    for section_title, image_paths in sections:
        if not image_paths:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        for image_path in image_paths:
            try:
                image_ref = image_path.resolve().relative_to(output_path.parent.resolve()).as_posix()
            except ValueError:
                image_ref = image_path.as_posix()
            lines.append(f"### {image_path.stem}")
            lines.append("")
            lines.append(f"![{image_path.stem}]({image_ref})")
            lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def summarize_backtest_metrics(stats: pd.Series, benchmark_close: pd.Series) -> dict[str, float]:
    strategy_total = float(stats["Return [%]"]) / 100.0
    max_dd = float(stats["Max. Drawdown [%]"]) / 100.0
    trade_count = int(stats["# Trades"])

    bench = benchmark_close.dropna()
    buy_hold_total = float(bench.iloc[-1] / bench.iloc[0] - 1)
    days = len(bench)
    years = max(days / 252.0, 1e-9)
    months = max(days / 21.0, 1e-9)

    strategy_ann = float(stats["Return (Ann.) [%]"]) / 100.0
    buy_hold_ann = float((1 + buy_hold_total) ** (1 / years) - 1)
    strategy_month = float((1 + strategy_total) ** (1 / months) - 1)
    buy_hold_month = float((1 + buy_hold_total) ** (1 / months) - 1)

    trades = stats["_trades"]
    holding_avg = np.nan
    holding_median = np.nan
    holding_total = np.nan
    if isinstance(trades, pd.DataFrame) and len(trades) > 0:
        entry_time = pd.to_datetime(trades["EntryTime"])
        exit_time = pd.to_datetime(trades["ExitTime"])
        holding_days = (exit_time - entry_time).dt.days.clip(lower=1)
        holding_avg = float(holding_days.mean())
        holding_median = float(holding_days.median())
        holding_total = float(holding_days.sum())

    return {
        "总收益率": strategy_total,
        "最大回撤": max_dd,
        "超额收益": strategy_total - buy_hold_total,
        "年化收益率": strategy_ann,
        "年化超额收益": strategy_ann - buy_hold_ann,
        "月化收益率": strategy_month,
        "月化超额收益": strategy_month - buy_hold_month,
        "交易次数": float(trade_count),
        "年均交易频率": float(trade_count / years),
        "平均持有天数": holding_avg,
        "持有天数中位数": holding_median,
        "持有天数合计": holding_total,
    }