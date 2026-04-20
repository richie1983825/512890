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
- `main.py` is the parameter-scanning workflow entry (walk-forward scan + compare), not a fixed-seed optimal-parameter replay script.

## Which Script To Run For Seed 142 Best Params

- If the goal is "run the strategy using seed 142 best parameters", use `run_formal_report.py`.
- Run command: `uv run run_formal_report.py`
- `run_formal_report.py` is the report-oriented strategy runner for the global-best switch workflow.
- In current code, this script is configured with seed 342 by default (`report_root` folder name and `random_seed=342 + idx`).
- To run seed 142 outputs, update those seed settings in `run_formal_report.py` from 342 to 142 before execution.

## Plot Font Warning Prevention (Matplotlib)

- Goal: avoid repeated warnings like `Glyph xxxx missing from font(s) DejaVu Sans` when chart titles/labels contain Chinese.
- Before generating any chart with Chinese text, always call `configure_chinese_font()` once at runtime.
- If writing standalone scripts (outside `main.py` flow), explicitly set Matplotlib font fallback before plotting:
	- `plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]`
	- `plt.rcParams["axes.unicode_minus"] = False`
- For quick one-off scripts where Chinese font availability is uncertain, use English chart title/axis text to prevent glyph warnings.
- When producing report images under `reports/`, verify there are no font warnings in stderr; if warnings exist, re-run after font config/fallback adjustment.

## Backtest Output Requirements

- All chart X-axes must be plotted by trading days (bar index order), not natural calendar-day spacing.

- After every backtest run, always generate and save these two comparison charts:
	- Strategy vs Buy-and-Hold annual return comparison chart.
	- Strategy vs Buy-and-Hold daily cumulative return comparison chart, with buy/sell markers.
- After every backtest run, always export a dedicated trade-record CSV under `reports/` for each strategy/run, so fills can be queried later.
	- Recommended fields: entry time, exit time, size, entry price, exit price, pnl, return pct, and holding days.
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

- Strategy 0: `PolyfitDynamicGridStrategy`
	- Type: Polyfit baseline dynamic grid mean-reversion strategy.
	- Core behavior:
		- Uses `PolyBasePred` as trading baseline.
		- Uses `PolyDevPct`, `PolyDevTrend`, and `RollingVolPct` to dynamically scale grid thresholds.
		- Supports runtime trade reason capture and trade-record CSV export.
	- Main optimization/search functions:
		- `build_param_space()`
		- `scan_parameters(base_data, param_space, max_evals=..., random_seed=...)`

- Strategy 1: `MovingAverageDynamicGridStrategy`
	- Type: Moving-average baseline dynamic grid mean-reversion strategy.
	- Core behavior:
		- Uses `MABase` as trading baseline.
		- Uses `MADevPct`, `MADevTrend`, and `RollingVolPct` to dynamically scale grid thresholds.
		- Supports runtime trade reason capture and trade-record CSV export.
	- Main optimization/search functions:
		- `build_ma_param_space()`
		- `scan_ma_parameters(base_data, param_space, max_evals=..., random_seed=...)`

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