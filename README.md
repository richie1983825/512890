# 512890 策略回测项目

当前项目已切换为 **PolyfitDynamicGridStrategy**（`strategies/polyfit_dynamic_grid_strategy.py`）。

原先的实验性趋势策略 `PureTrendFollowingStrategy` 已删除。
并新增事件型启动策略：
## 最新四策略回测（强制可交易约束后）

最新汇总文件：
- `reports/wf3y1y_event_breakout_strategy_summary.csv`

窗口级结果（动量三策略）：

| 窗口 | 启动突破交易次数 | 回踩确认交易次数 | 事件型启动交易次数 | 启动突破总收益率 | 回踩确认总收益率 | 事件型启动总收益率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## 当前策略


同时新增了一个基于回归策略变体的均线基准策略：
- `MovingAverageDynamicGridStrategy`（`strategies/moving_average_dynamic_grid_strategy.py`）

当前主流程会同时比较以下五个策略：
- 回归基准动态网格：`PolyfitDynamicGridStrategy`
- MA 基准动态网格：`MovingAverageDynamicGridStrategy`
- 启动突破：`LaunchBreakoutMomentumStrategy`
- 突破回踩确认：`BreakoutRetestMomentumStrategy`
- 事件型启动：`EventDrivenLaunchStrategy`
策略名称：`PolyfitDynamicGridStrategy`

策略逻辑（仅做多）：
- 每天使用过去约 3 年行情（参数 `fit_window_days`，默认搜索为 630/756/882）做一元一次多项式拟合（线性拟合）。
- 以拟合当日预测值 `PolyBasePred` 作为交易基准。
- 计算价格偏离 `PolyDevPct` 与偏离趋势 `PolyDevTrend`。
- 计算滚动波动率 `RollingVolPct`，并将其作为高波动放缓因子。
- 围绕基准构建网格，网格阈值为动态值：
  - `dynamic_grid_step = base_grid_pct * (1 + trend_sensitivity * abs(PolyDevTrend)) * (1 + volatility_scale * RollingVolPct)`
- 当偏离趋势更陡或周期内波动更大时，买入和卖出阈值都会同步放大，整体交易节奏变慢。
- 出场由以下条件触发：
  - 网格止盈（`take_profit_grid`）
  - 网格止损（`stop_loss_grid`）
  - 最大持有天数（`max_holding_days`）
  - 冷却期（`cooldown_days`）

## 参数空间

当前 `main.py` 中参数搜索空间如下：

- `fit_window_days`: `[630, 756, 882]`
- `trend_window_days`: `[10, 15, 20]`
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

说明：当样本长度不足完整 3 年时，特征计算会自动降级到可用窗口（不少于 252 日），保证训练/回测可执行。

## 最新回测输出（来自 `reports/wf3y1y_summary.csv`）

口径：3 年训练 + 1 年验证，逐窗口先训练扫描参数，再验证回测。

| 窗口 | 验证区间 | 总收益率 | 超额收益 | 最大回撤 | 年化收益率 | 年化超额收益 | 交易次数 | 平均持有天数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2022-01-04 ~ 2022-12-30 | 7.20% | 6.08% | -9.25% | 7.51% | 6.35% | 32 | 7.88 |
| 2 | 2023-01-03 ~ 2023-12-29 | 8.29% | -2.91% | -4.03% | 8.64% | -3.04% | 32 | 3.28 |
| 3 | 2024-01-02 ~ 2024-12-31 | 7.01% | -14.51% | -7.50% | 7.31% | -15.19% | 12 | 6.75 |
| 4 | 2025-01-02 ~ 2025-12-31 | 15.15% | 8.34% | -2.77% | 15.75% | 8.68% | 16 | 10.44 |

汇总（4 窗口均值）：
- 平均总收益率：**9.09%**
- 平均超额收益：**-0.79%**
- 平均年化收益率：**9.62%**
- 平均年化超额收益：**-0.43%**
- 平均最大回撤：**-5.89%**
- 平均交易次数：**23.00 次/年**

## 输出文件

每个验证窗口会输出：
- `reports/wf3y1y_XX_annual_return_comparison.png`
- `reports/wf3y1y_XX_daily_cumulative_return_comparison.png`
- `reports/wf3y1y_XX_annual_return_comparison.csv`
- `reports/wf3y1y_XX_daily_cumulative_return_comparison.csv`
- `reports/wf3y1y_XX_train_scan_top50.csv`

汇总文件：
- `reports/wf3y1y_summary.csv`

同时，每个窗口都会调用 `print_daily_cumulative_returns_with_signals(...)` 输出：
- 策略每日累计收益
- 长期持有每日累计收益
- 买卖点

## 运行方式

```bash
uv run python main.py
```

## 关键实现位置

- `main.py`
  - 参数空间：`build_param_space()`
  - 特征构建：`add_strategy_features()`
  - 参数扫描：`scan_parameters()`
  - 三策略滚动验证：`run_breakout_strategy_comparison_3y1y()`
  - 主流程：`main()`
- `strategies/polyfit_dynamic_grid_strategy.py`
  - 当前策略类：`PolyfitDynamicGridStrategy`
- `strategies/launch_breakout_momentum_strategy.py`
  - 启动突破策略：`LaunchBreakoutMomentumStrategy`
- `strategies/breakout_retest_momentum_strategy.py`
  - 突破回踩确认策略：`BreakoutRetestMomentumStrategy`

## 动量策略替代实验

当前 `main.py` 已改为执行三策略 walk-forward 对比：
- 回归策略：`PolyfitDynamicGridStrategy`
- 启动突破策略：`LaunchBreakoutMomentumStrategy`
- 突破回踩确认策略：`BreakoutRetestMomentumStrategy`

启动突破策略逻辑：
- 只在 `FastEMA > SlowEMA` 且 `TrendSlope` 为正时考虑入场
- 需要真实突破前高，并满足最小单日涨幅 `min_breakout_return`
- 要求收盘位置强，避免冲高回落的弱 breakout
- 以初始 ATR 止损、跟踪止损、跌破 `FastEMA`/`ExitLow` 和最大持有天数退出

突破回踩确认策略逻辑：
- 先识别一次有效突破，不立即追价
- 在 `retest_window_days` 窗口内等待第一次回踩确认
- 要求回踩后重新转强，再做低风险二次上车
- 加入 `max_extension_pct` 过热过滤，减少高位追价

五策略汇总文件：
- `reports/wf3y1y_multi_strategy_summary.csv`

最新三策略对比结果：

| 窗口 | 回归策略总收益率 | 启动突破策略总收益率 | 突破回踩确认策略总收益率 |
| --- | ---: | ---: | ---: |
| 1 | 7.20% | -5.70% | -8.82% |
| 2 | 8.29% | 4.17% | -1.41% |
| 3 | 7.01% | 0.56% | 0.07% |
| 4 | 15.15% | 2.02% | 2.00% |

结论：
- 旧的纯趋势跟踪策略已被移除，因为它与“抓启动段”的目标不匹配。
- 两个新动量策略比旧趋势策略更贴近需求，但当前参数和规则下仍显著弱于回归策略。
- 启动突破策略在 2023 年有一定改善，突破回踩确认策略在当前标的上更保守，整体仍然偏弱。
- 当前实验说明：你要的方向更接近“启动段捕捉”，但单纯突破类规则还不够，需要继续加入市场状态过滤、成交量或波动压缩等前置条件。

## 参数含义与调参建议

以下建议以当前实现为准（`main.py` + `strategies/polyfit_dynamic_grid_strategy.py`）。

### 拟合与趋势参数

- `fit_window_days`
  - 含义：线性拟合使用的历史窗口长度。
  - 调大：基准更平滑、交易信号更慢，通常换手下降、对短期拐点更迟钝。
  - 调小：基准更敏感、信号更快，通常换手上升，容易在噪声中频繁触发。

- `trend_window_days`
  - 含义：偏离趋势 `PolyDevTrend` 的平滑窗口。
  - 调大：动态阈值变化更慢，风格更稳健。
  - 调小：动态阈值更灵敏，能更快反映偏离加速，但也更易抖动。

- `vol_window_days`
  - 含义：滚动波动率计算周期，用于识别近期波动环境。
  - 调大：波动识别更平滑，阈值调整更慢。
  - 调小：更快响应波动升高，但也更容易随短期噪声摆动。

### 网格与触发参数

- `base_grid_pct`
  - 含义：基础网格间距，决定最小阈值尺度。
  - 调大：入场更谨慎，交易次数下降，容易错过小幅回归。
  - 调小：入场更积极，交易次数上升，噪声交易增加风险更高。

- `volatility_scale`
  - 含义：高波动对买卖阈值放缓的强度系数。
  - 调大：波动越高，入场和出场阈值放得越远，交易更慢。
  - 调小：高波动放缓效果减弱，更接近原始动态网格。

- `trend_sensitivity`
  - 含义：偏离趋势对阈值放大的系数。
  - 调大：偏离越陡阈值抬升越明显，减少“追着陡峭偏离开仓”。
  - 调小：动态放大作用减弱，更接近静态网格。

- `max_grid_levels`
  - 含义：按偏离强度分层时可使用的最大层级。
  - 调大：允许更深偏离分层，入场阈值区分更细。
  - 调小：分层更粗，策略更简化。

- `min_signal_strength`
  - 含义：最小触发强度阈值（偏离/动态网格步长）。
  - 调大：过滤弱信号，减少交易；
  - 调小：允许更多边缘信号，增加交易频率。

- `position_size`
  - 含义：单次开仓仓位比例。
  - 调大：放大收益和回撤。
  - 调小：降低组合波动和单笔风险暴露。

### 出场与风险参数

- `take_profit_grid`
  - 含义：以网格步长计的止盈阈值倍数。
  - 调大：止盈更远，单笔潜在收益提升但回吐风险增大。
  - 调小：更快落袋为安，胜率可能提高但盈亏比可能下降。

- `stop_loss_grid`
  - 含义：以网格步长计的止损阈值倍数。
  - 调大：止损更宽，容忍波动更高但尾部损失可能放大。
  - 调小：止损更紧，回撤控制更快但容易被震荡扫损。

- `max_holding_days`
  - 含义：单笔最大持有天数。
  - 调大：给均值回归更多时间，但资金占用与不确定性增加。
  - 调小：提升资金周转，降低滞留风险，但可能提前离场。

- `cooldown_days`
  - 含义：平仓后冷却期，限制连续触发。
  - 调大：抑制过度交易，降低连亏簇拥风险。
  - 调小：提高再入场速度，适合趋势切换更快阶段。

### 实操调参顺序（推荐）

1. 先定风险边界：`stop_loss_grid`、`max_holding_days`、`cooldown_days`。
2. 再定交易频率：`base_grid_pct`、`min_signal_strength`。
3. 再调高波动放缓强度：`vol_window_days`、`volatility_scale`。
4. 最后调自适应强度：`trend_sensitivity`、`trend_window_days`。
5. 在同一口径下对比：优先看 `超额收益`、`最大回撤`、`年均交易频率` 三者平衡，不只看收益率。
