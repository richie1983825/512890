from __future__ import annotations

import numpy as np
from backtesting import Strategy


class LaunchBreakoutMomentumStrategy(Strategy):
    """启动突破策略：只做有趋势过滤的强势启动段。"""

    breakout_buffer_atr = 0.25
    min_breakout_return = 0.015
    min_trend_slope = 0.0
    min_close_strength = 0.55
    min_volume_ratio = 1.2
    max_compression_ratio = 0.85
    min_range_to_atr = 1.1
    min_continuation_return = 0.002
    min_continuation_volume_ratio = 0.8
    initial_stop_atr_mult = 1.8
    trailing_atr_mult = 2.5
    max_holding_days = 30
    position_size = 0.5

    def init(self) -> None:
        self.entry_bar_index: int | None = None
        self.entry_price = np.nan
        self.peak_close = np.nan

    def _reset_state_if_flat(self) -> None:
        if not self.position:
            self.entry_bar_index = None
            self.entry_price = np.nan
            self.peak_close = np.nan

    def next(self) -> None:
        self._reset_state_if_flat()

        open_price = float(self.data.Open[-1])
        high = float(self.data.High[-1])
        low = float(self.data.Low[-1])
        close = float(self.data.Close[-1])
        fast_ema = float(self.data.FastEMA[-1])
        slow_ema = float(self.data.SlowEMA[-1])
        trend_slope = float(self.data.TrendSlope[-1])
        breakout_high = float(self.data.BreakoutHigh[-1])
        exit_low = float(self.data.ExitLow[-1])
        atr = float(self.data.ATR[-1])
        daily_return = float(self.data.DailyReturn[-1])
        volume_ratio = float(self.data.VolumeRatio[-1])
        compression_ratio = float(self.data.CompressionRatio[-1])
        range_to_atr = float(self.data.RangeToATR[-1])
        bar_index = len(self.data) - 1

        required = [
            open_price,
            high,
            low,
            close,
            fast_ema,
            slow_ema,
            trend_slope,
            breakout_high,
            exit_low,
            atr,
            daily_return,
            volume_ratio,
            compression_ratio,
            range_to_atr,
        ]
        if any(np.isnan(v) for v in required) or atr <= 0:
            return

        range_size = max(high - low, 1e-9)
        close_strength = (close - low) / range_size

        if not self.position:
            size = float(np.clip(self.position_size, 0.0, 1.0))
            breakout_level = breakout_high + float(self.breakout_buffer_atr) * atr
            trend_ok = close >= fast_ema and fast_ema >= slow_ema and trend_slope >= float(self.min_trend_slope)
            breakout_ok = close >= breakout_level
            thrust_ok = daily_return >= float(self.min_breakout_return)
            close_ok = close_strength >= float(self.min_close_strength)
            volume_ok = volume_ratio >= float(self.min_volume_ratio)
            setup_ok = compression_ratio <= float(self.max_compression_ratio)
            expansion_ok = range_to_atr >= float(self.min_range_to_atr)

            continuation_ok = (
                trend_ok
                and daily_return >= float(self.min_continuation_return)
                and close >= fast_ema
                and volume_ratio >= float(self.min_continuation_volume_ratio)
                and range_to_atr >= 0.5
            )

            breakout_entry_ok = breakout_ok and thrust_ok and close_ok and volume_ok and setup_ok and expansion_ok
            if size > 0 and (breakout_entry_ok or continuation_ok):
                self.buy(size=size)
                self.entry_bar_index = bar_index
                self.entry_price = close
                self.peak_close = close
            return

        if self.entry_bar_index is None:
            self.entry_bar_index = bar_index
        if not np.isfinite(self.entry_price):
            self.entry_price = close
        self.peak_close = max(close, float(self.peak_close) if np.isfinite(self.peak_close) else close)

        holding_days = bar_index - self.entry_bar_index
        protective_stop = float(self.entry_price) - float(self.initial_stop_atr_mult) * atr
        trailing_stop = float(self.peak_close) - float(self.trailing_atr_mult) * atr
        exit_signal = (
            close <= max(protective_stop, trailing_stop)
            or close <= fast_ema
            or close <= exit_low
            or holding_days >= int(self.max_holding_days)
        )
        if exit_signal:
            self.position.close()
