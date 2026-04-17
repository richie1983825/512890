# AGENTS.md

## Project Conventions

- Development environment: macOS
- Python and dependency management: use `uv`
- Primary development surface: Jupyter Notebook (`.ipynb`)
- Add dependencies with `uv add <package>`
- Run project commands with `uv run <command>`
- Use the existing uv-managed environment for this repository; do not create a separate manual virtual environment

## Repository Notes

- Prefer implementing exploratory and analysis work in `main.ipynb`
- Keep changes minimal and aligned with the current Python 3.14.3 project setup
- Store all source and intermediate datasets under the `data/` directory
- Save all backtest reports, csv summaries, and chart outputs under the `reports/` directory

## Plot Font Warning Prevention (Matplotlib)

- Goal: avoid repeated warnings like `Glyph xxxx missing from font(s) DejaVu Sans` when chart titles/labels contain Chinese.
- Before generating any chart with Chinese text, always call `configure_chinese_font()` once at runtime.
- If writing standalone scripts (outside `main.py` flow), explicitly set Matplotlib font fallback before plotting:
	- `plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]`
	- `plt.rcParams["axes.unicode_minus"] = False`
- For quick one-off scripts where Chinese font availability is uncertain, use English chart title/axis text to prevent glyph warnings.
- When producing report images under `reports/`, verify there are no font warnings in stderr; if warnings exist, re-run after font config/fallback adjustment.

## Backtest Output Requirements

- After every backtest run, always generate and save these two comparison charts:
	- Strategy vs Buy-and-Hold annual return comparison chart.
	- Strategy vs Buy-and-Hold daily cumulative return comparison chart, with buy/sell markers.
- After every backtest run, always append the run result to `records.md` and record one entry per strategy/run.
	- Required fields for each entry: round/backtest identifier, return, max drawdown, excess return, number of trades, and holding days.
- When yearly validation produces 4 comparison images (one per year/window), always generate an additional Markdown summary file (under `reports/`) that embeds the 4 images together in one page for quick side-by-side review.
- After every backtest run, always call `print_daily_cumulative_returns_with_signals(...)` to print strategy daily cumulative return, buy-and-hold daily cumulative return, and buy/sell points.
- Every backtest summary report must include these metrics:
	- Max drawdown.
	- Total return.
	- Excess return (strategy minus buy-and-hold).
	- Annual return and annual excess return.
	- Monthly return and monthly excess return.
	- Trading frequency.
	- Holding days (per trade and/or aggregate statistics).

## Implemented Strategies

- Strategy 1: `WeeklyReversalStrategy`
	- Type: Weekly signal + daily execution reversal strategy (EMA/ATR dynamic thresholds).
	- Core behavior: Threshold pullback entry, tiered TP, hard stop, trailing stop, max holding days.
	- Default parameters:
		- `ma_window_weeks = 8`
		- `atr_window_weeks = 8`
		- `buy_atr_mult = 0.75`
		- `tier1_tp_atr_mult = 0.75`
		- `tier2_tp_atr_mult = 1.50`
		- `stop_atr_mult = 1.50`
		- `trailing_atr_mult = 2.00`
		- `max_holding_days = 30`
		- `trend_filter = false`
		- `reversal_rsi_max = 55`
	- Main optimization/search functions:
		- `build_param_space()`
		- `scan_parameters(base_data, param_space, max_evals=..., random_seed=...)`

- Strategy 2: `AdaptiveShockReversalStrategy`
	- Type: Enhanced reversal strategy with panic-exit, momentum-entry, volatility regime switching, and execution optimization.
	- Core behavior:
		- Panic exit on gap/down-shock conditions.
		- Momentum entry for strong up moves.
		- Dynamic stop/holding limits under high-volatility regime.
		- Anti-overstay / anti-idle execution controls.
	- Default parameters:
		- `buy_atr_mult = 0.75`
		- `tier1_tp_atr_mult = 0.75`
		- `tier2_tp_atr_mult = 1.50`
		- `stop_atr_mult = 1.50`
		- `trailing_atr_mult = 2.00`
		- `max_holding_days = 30`
		- `reversal_rsi_max = 55`
		- `panic_daily_drop = -0.02`
		- `panic_gap_drop = -0.015`
		- `panic_atr_mult = 1.5`
		- `momentum_breakout_atr_mult = 0.35`
		- `momentum_rsi_max = 72`
		- `high_vol_ratio_threshold = 1.2`
		- `high_vol_stop_atr_mult = 1.2`
		- `high_vol_trailing_atr_mult = 1.8`
		- `high_vol_max_holding_days = 20`
		- `max_flat_days = 20`
	- Main optimization/search scripts:
		- `scans/adaptive_scan_mp.py` (multi-process random large-sample scan)
		- `scans/adaptive_scan_full_mp.py` (multi-process exhaustive scan framework)

## Shared Core Functions

- Data and features:
	- `load_and_forward_adjust(parquet_path)`
	- `add_strategy_features(df, ma_window_weeks, atr_window_weeks)`
	- `_rsi(series, window=14)`

- Reporting and charting:
	- `calc_independent_annual_returns(series)`
	- `plot_annual_return_comparison(strategy_equity_curve, benchmark_close, ...)`
	- `plot_daily_cumulative_return_comparison(strategy_equity_curve, benchmark_close, ..., trades=...)`

- Runtime utilities:
	- `configure_chinese_font()`
	- `main()` (main workflow entry in `main.py`)

## Latest Polyfit vs MA Test Snapshot

- Source report: `reports/wf3y1y_polyfit_ma_strategy_summary.csv`
- Run context: dual-strategy walk-forward with carry-over initial position logic.
- Window 1:
	- Polyfit return: 7.65%
	- MA return: 7.58%
	- Polyfit position: 0.00% -> 0.00%
	- MA position: 0.00% -> 99.00%
- Window 2:
	- Polyfit return: 8.81%
	- MA return: 0.17%
	- Polyfit position: 0.00% -> 0.00%
	- MA position: 99.00% -> 97.00%
- Window 3:
	- Polyfit return: 7.01%
	- MA return: 27.33%
	- Polyfit position: 0.00% -> 0.00%
	- MA position: 97.00% -> 0.00%
- Window 4:
	- Polyfit return: 15.07%
	- MA return: 12.16%
	- Polyfit position: 0.00% -> 95.00%
	- MA position: 0.00% -> 94.00%
- Aggregate:
	- Polyfit average annual return: 10.04%
	- MA average annual return: 12.33%