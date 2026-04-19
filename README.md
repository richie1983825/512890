# 512890 策略回测项目

当前仓库只保留两个策略：

- `PolyfitDynamicGridStrategy`
- `MovingAverageDynamicGridStrategy`

主流程执行这两个策略的 3 年训练 + 1 年验证滚动对比，并输出参数扫描、对比图、成交记录和汇总结果。

## 运行方式

```bash
uv run main.py
```

## 主流程输出

- 汇总文件：`reports/wf3y1y_polyfit_ma_strategy_summary.csv`
- 窗口级 Polyfit 曲线图：`reports/wf3y1y_XX_polyfit_daily_cumulative_return_comparison.png`
- 窗口级 MA 曲线图：`reports/wf3y1y_XX_ma_daily_cumulative_return_comparison.png`
- 窗口级双策略对比图：`reports/wf3y1y_XX_strategy_pair_daily_comparison.png`
- 窗口级成交记录：`reports/wf3y1y_XX_polyfit_trade_records.csv`、`reports/wf3y1y_XX_ma_trade_records.csv`
- 窗口级训练扫描：`reports/wf3y1y_XX_polyfit_train_scan_top50.csv`、`reports/wf3y1y_XX_ma_train_scan_top50.csv`

## 策略说明

### PolyfitDynamicGridStrategy

- 使用线性拟合基准 `PolyBasePred`
- 根据 `PolyDevPct`、`PolyDevTrend` 和 `RollingVolPct` 动态调整网格阈值
- 仅做多
- 支持成交原因追踪与成交记录 CSV 导出

### MovingAverageDynamicGridStrategy

- 使用移动平均基准 `MABase`
- 根据 `MADevPct`、`MADevTrend` 和 `RollingVolPct` 动态调整网格阈值
- 仅做多
- 支持成交原因追踪与成交记录 CSV 导出

## 当前参数空间

### Polyfit

- `fit_window_days`: `[252]`
- `trend_window_days`: `[10, 15, 20]`
- `vol_window_days`: `[5, 10, 15, 20, 30]`
- `base_grid_pct`: `[0.008, 0.010, 0.012, 0.015]`
- `volatility_scale`: `[0.0, 0.5, 1.0, 1.5, 2.0]`
- `trend_sensitivity`: `[4.0, 6.0, 8.0, 10.0]`
- `max_grid_levels`: `[2, 3, 4]`
- `take_profit_grid`: `[0.6, 0.8, 1.0]`
- `stop_loss_grid`: `[1.2, 1.6, 2.0]`
- `max_holding_days`: `[15, 25, 35]`
- `cooldown_days`: `[1]`
- `min_signal_strength`: `[0.30, 0.45, 0.60]`
- `position_size`: `[0.00, 0.01, ..., 1.00]`
- `position_sizing_coef`: `[10.0, 20.0, 30.0, 40.0, 60.0]`

### MA

- `ma_window_days`: `[60]`
- `trend_window_days`: `[5, 10, 15, 20]`
- `vol_window_days`: `[5, 10, 15, 20, 30]`
- `base_grid_pct`: `[0.008, 0.010, 0.012, 0.015]`
- `volatility_scale`: `[0.0, 0.5, 1.0, 1.5, 2.0]`
- `trend_sensitivity`: `[4.0, 6.0, 8.0, 10.0]`
- `max_grid_levels`: `[2, 3, 4]`
- `take_profit_grid`: `[0.6, 0.8, 1.0]`
- `stop_loss_grid`: `[1.2, 1.6, 2.0]`
- `max_holding_days`: `[15, 25, 35]`
- `cooldown_days`: `[0, 1, 2]`
- `min_signal_strength`: `[0.30, 0.45, 0.60]`
- `position_size`: `[0.00, 0.01, ..., 1.00]`
- `position_sizing_coef`: `[10.0, 20.0, 30.0, 40.0, 60.0]`

## 关键文件

- `main.py`
- `strategies/polyfit_dynamic_grid_strategy.py`
- `strategies/moving_average_dynamic_grid_strategy.py`