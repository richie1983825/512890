from __future__ import annotations

import numpy as np
from backtesting import Strategy


class BreakoutRetestMomentumStrategy(Strategy):
    """突破回踩确认策略：先确认突破，再等待首次低风险回踩反弹。"""

    breakout_buffer_atr = 0.15
    retest_tolerance_atr = 0.6
    retest_window_days = 8
    min_trend_slope = 0.0
    max_extension_pct = 0.05
    min_breakout_volume_ratio = 1.2
    min_retest_volume_ratio = 0.9
    max_setup_compression_ratio = 0.85
    min_breakout_range_to_atr = 1.0
    initial_stop_atr_mult = 1.5
    trailing_atr_mult = 2.2
    max_holding_days = 35
    position_size = 0.5

    def init(self) -> None:
        self.pending_breakout_level = np.nan
        self.pending_deadline: int | None = None
        self.entry_bar_index: int | None = None
        self.entry_price = np.nan
        self.peak_close = np.nan

    def _reset_if_flat(self) -> None:
        if not self.position:
            self.entry_bar_index = None
            self.entry_price = np.nan
            self.peak_close = np.nan

    def _clear_pending(self) -> None:
        self.pending_breakout_level = np.nan
        self.pending_deadline = None

    def next(self) -> None:
        self._reset_if_flat()

        open_price = float(self.data.Open[-1])
        low = float(self.data.Low[-1])
        close = float(self.data.Close[-1])
        prev_close = float(self.data.PrevClose[-1])
        fast_ema = float(self.data.FastEMA[-1])
        slow_ema = float(self.data.SlowEMA[-1])
        trend_slope = float(self.data.TrendSlope[-1])
        breakout_high = float(self.data.BreakoutHigh[-1])
        exit_low = float(self.data.ExitLow[-1])
        atr = float(self.data.ATR[-1])
        volume_ratio = float(self.data.VolumeRatio[-1])
        compression_ratio = float(self.data.CompressionRatio[-1])
        range_to_atr = float(self.data.RangeToATR[-1])
        bar_index = len(self.data) - 1

        required = [
            open_price,
            low,
            close,
            prev_close,
            fast_ema,
            slow_ema,
            trend_slope,
            breakout_high,
            exit_low,
            atr,
            volume_ratio,
            compression_ratio,
            range_to_atr,
        ]
        if any(np.isnan(v) for v in required) or atr <= 0:
            return

        breakout_level = breakout_high + float(self.breakout_buffer_atr) * atr
        trend_ok = close >= fast_ema and fast_ema >= slow_ema and trend_slope >= float(self.min_trend_slope)
        extension_pct = close / fast_ema - 1.0

        if not self.position:
            if np.isfinite(self.pending_breakout_level):
                deadline_passed = self.pending_deadline is not None and bar_index > self.pending_deadline
                invalidated = close < slow_ema
                if deadline_passed or invalidated:
                    self._clear_pending()
                else:
                    retest_floor = float(self.pending_breakout_level) - float(self.retest_tolerance_atr) * atr
                    retest_touched = low <= float(self.pending_breakout_level) and close >= retest_floor
                    bounce_ok = close > open_price and close > prev_close and close >= fast_ema
                    extension_ok = extension_pct <= float(self.max_extension_pct)
                    volume_ok = volume_ratio >= float(self.min_retest_volume_ratio)
                    size = float(np.clip(self.position_size, 0.0, 1.0))
                    if size > 0 and trend_ok and retest_touched and bounce_ok and extension_ok and volume_ok:
                        self.buy(size=size)
                        self.entry_bar_index = bar_index
                        self.entry_price = close
                        self.peak_close = close
                        self._clear_pending()
                    return

            breakout_setup_ok = (
                trend_ok
                and close >= breakout_level
                and volume_ratio >= float(self.min_breakout_volume_ratio)
                and compression_ratio <= float(self.max_setup_compression_ratio)
                and range_to_atr >= float(self.min_breakout_range_to_atr)
            )
            if breakout_setup_ok:
                self.pending_breakout_level = breakout_level
                self.pending_deadline = bar_index + int(self.retest_window_days)
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

