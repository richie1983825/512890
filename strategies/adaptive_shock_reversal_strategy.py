from __future__ import annotations

import numpy as np
from backtesting import Strategy


class AdaptiveShockReversalStrategy(Strategy):
    """急涨跟随 + 急跌风控 + 快速反弹重入的自适应策略（仅做多）。"""

    z_entry = 1.0
    z_exit = 0.15
    z_stop = 3.5
    high_vol_q = 0.94
    max_holding_days = 45
    min_days_between_trades = 1
    trend_entry_z = 0.1
    min_flat_days_for_trend_entry = 8
    min_trend_up_days = 3
    trend_exit_z = -0.1
    min_holding_protect_days = 3
    momentum_breakout_z = 0.6
    panic_daily_drop = -0.02
    panic_z = 2.3
    entry_no_fall_daily_drop = -0.012
    panic_cooldown_days = 3
    rebound_max_days_after_panic = 12
    rebound_recovery_pct = 0.012
    rebound_daily_ret = 0.005
    rebound_exit_z = -0.1

    buy_atr_mult = 0.8
    tier1_tp_atr_mult = 0.9
    tier2_tp_atr_mult = 1.8
    stop_atr_mult = 1.4
    trailing_atr_mult = 1.7
    reversal_rsi_max = 54
    panic_gap_drop = -0.012
    panic_atr_mult = 1.4
    momentum_breakout_atr_mult = 0.3
    momentum_rsi_max = 74
    high_vol_ratio_threshold = 1.2
    high_vol_stop_atr_mult = 1.1
    high_vol_trailing_atr_mult = 1.4
    high_vol_max_holding_days = 18
    max_flat_days = 12

    def init(self) -> None:
        self.entry_bar_index: int | None = None
        self.entry_price = np.nan
        self.peak_close = np.nan
        self.cooldown = 0
        self.flat_days = 0
        self.entry_mode = "none"
        self.panic_ref_low = np.nan
        self.days_since_panic = 9999

    def _reset_state_if_flat(self) -> None:
        if not self.position:
            self.entry_bar_index = None
            self.entry_price = np.nan
            self.peak_close = np.nan
            self.entry_mode = "none"

    def _close_position(self, close_ref: float, panic: bool = False) -> None:
        self.position.close()
        self.cooldown = max(self.min_days_between_trades, int(self.panic_cooldown_days if panic else self.min_days_between_trades))
        if panic:
            self.days_since_panic = 0
            self.panic_ref_low = close_ref

    def next(self) -> None:
        self._reset_state_if_flat()

        z = float(self.data.ZScore[-1])
        daily_ret = float(self.data.DailyRet[-1])
        gap_ret = float(self.data.GapRet[-1])
        vol_rank = float(self.data.VolRank252[-1])
        close = float(self.data.Close[-1])
        low = float(self.data.Low[-1])
        high = float(self.data.High[-1])
        prev_close = float(self.data.PrevClose[-1])
        mid = float(self.data.Mid[-1])
        mid_slope = float(self.data.MidSlope[-1])
        fast_ema = float(self.data.FastEMA[-1])
        atr = float(self.data.ATR[-1])
        atr_ratio = float(self.data.ATRRatio[-1])
        trend_up_streak = float(self.data.TrendUpStreak[-1])
        rsi = float(self.data.RSI[-1])
        breakout_high = float(self.data.BreakoutHigh[-1])
        bar_index = len(self.data) - 1

        required = [z, daily_ret, gap_ret, vol_rank, close, low, high, prev_close, mid, mid_slope, fast_ema, atr, atr_ratio, trend_up_streak, rsi, breakout_high]
        if any(np.isnan(v) for v in required) or atr <= 0 or prev_close <= 0:
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.days_since_panic < 9999:
            self.days_since_panic += 1

        atr_pct = atr / prev_close
        price_vs_mid_atr = (close - mid) / atr
        high_vol = vol_rank >= self.high_vol_q or atr_ratio >= self.high_vol_ratio_threshold
        trend_up = close >= mid and fast_ema >= mid and mid_slope > 0
        panic_bar = (
            daily_ret <= self.panic_daily_drop
            or gap_ret <= self.panic_gap_drop
            or (prev_close - close) >= self.panic_atr_mult * atr
            or z <= -self.panic_z
        )

        if not self.position:
            self.flat_days += 1
            if panic_bar:
                self.days_since_panic = 0
                self.panic_ref_low = min(close, low)
            elif self.days_since_panic <= int(self.rebound_max_days_after_panic):
                self.panic_ref_low = min(float(self.panic_ref_low), low) if np.isfinite(self.panic_ref_low) else low

            reversal_entry = (
                self.cooldown == 0
                and not high_vol
                and (z <= -self.z_entry or price_vs_mid_atr <= -self.buy_atr_mult)
                and daily_ret >= self.entry_no_fall_daily_drop
                and gap_ret > self.panic_gap_drop
                and rsi <= self.reversal_rsi_max
                and z > -self.panic_z
            )
            trend_follow_entry = (
                self.cooldown == 0
                and self.flat_days >= int(self.min_flat_days_for_trend_entry)
                and trend_up_streak >= int(self.min_trend_up_days)
                and z >= self.trend_entry_z
                and trend_up
            )
            momentum_entry = (
                self.cooldown == 0
                and trend_up_streak >= 2
                and trend_up
                and rsi <= self.momentum_rsi_max
                and (
                    z >= self.momentum_breakout_z
                    or close >= breakout_high
                    or daily_ret >= max(self.rebound_daily_ret, self.momentum_breakout_atr_mult * atr_pct)
                    or gap_ret >= 0.8 * self.momentum_breakout_atr_mult * atr_pct
                )
            )
            rebound_entry = (
                self.days_since_panic <= int(self.rebound_max_days_after_panic)
                and np.isfinite(self.panic_ref_low)
                and close >= float(self.panic_ref_low) * (1 + self.rebound_recovery_pct)
                and close >= fast_ema
                and daily_ret >= max(self.rebound_daily_ret, 0.55 * atr_pct)
                and gap_ret > self.panic_gap_drop / 2
                and rsi <= self.momentum_rsi_max
                and self.cooldown <= 1
            )

            if rebound_entry:
                self.buy()
                self.flat_days = 0
                self.entry_mode = "rb"
                self.entry_bar_index = bar_index
            elif momentum_entry:
                self.buy()
                self.flat_days = 0
                self.entry_mode = "mo"
                self.entry_bar_index = bar_index
            elif trend_follow_entry:
                self.buy()
                self.flat_days = 0
                self.entry_mode = "tf"
                self.entry_bar_index = bar_index
            elif reversal_entry:
                self.buy()
                self.flat_days = 0
                self.entry_mode = "mr"
                self.entry_bar_index = bar_index
            return

        self.flat_days = 0
        if self.entry_bar_index is None:
            self.entry_bar_index = bar_index
        if np.isnan(self.entry_price):
            self.entry_price = float(self.trades[-1].entry_price) if self.trades else close
        self.peak_close = max(close, high, float(self.peak_close) if np.isfinite(self.peak_close) else close)

        holding_days = bar_index - self.entry_bar_index
        protected = holding_days < int(self.min_holding_protect_days)
        stop_atr_mult = self.high_vol_stop_atr_mult if high_vol else self.stop_atr_mult
        trailing_atr_mult = self.high_vol_trailing_atr_mult if high_vol else self.trailing_atr_mult
        max_hold = min(int(self.max_holding_days), int(self.high_vol_max_holding_days)) if high_vol else int(self.max_holding_days)
        pnl_atr = (close - self.entry_price) / atr
        gained_room = self.peak_close - self.entry_price

        panic_exit = (
            daily_ret <= self.panic_daily_drop
            or gap_ret <= self.panic_gap_drop
            or close <= self.entry_price - self.panic_atr_mult * atr
            or z <= -self.panic_z
        )
        if panic_exit:
            self._close_position(close_ref=min(close, low), panic=True)
            return

        hard_stop = close <= self.entry_price - stop_atr_mult * atr or z <= -self.z_stop
        if hard_stop:
            self._close_position(close_ref=close)
            return

        trailing_stop = (
            gained_room >= max(0.6, self.tier1_tp_atr_mult * 0.75) * atr
            and close <= self.peak_close - trailing_atr_mult * atr
        )
        if trailing_stop:
            self._close_position(close_ref=close)
            return

        if not protected:
            if self.entry_mode == "mr":
                reversal_done = z >= -self.z_exit or close >= mid or pnl_atr >= self.tier1_tp_atr_mult
                reversal_fail = close < fast_ema and daily_ret < 0 and mid_slope <= 0
                if reversal_done or reversal_fail:
                    self._close_position(close_ref=close)
                    return
            elif self.entry_mode == "rb":
                rebound_fail = z <= self.rebound_exit_z or (close < fast_ema and daily_ret < 0)
                strong_profit_fade = pnl_atr >= self.tier1_tp_atr_mult and close < fast_ema
                if rebound_fail or strong_profit_fade:
                    self._close_position(close_ref=close)
                    return
            else:
                trend_reversal = z <= self.trend_exit_z or (close < mid and mid_slope <= 0) or (close < fast_ema and daily_ret < 0)
                take_profit = (
                    (pnl_atr >= self.tier2_tp_atr_mult and (daily_ret <= 0 or rsi >= 78))
                    or (pnl_atr >= self.tier1_tp_atr_mult and close < fast_ema)
                )
                if trend_reversal or take_profit:
                    self._close_position(close_ref=close)
                    return

            anti_idle = holding_days >= int(self.max_flat_days) and pnl_atr < 0.35 and close < fast_ema
            if anti_idle:
                self._close_position(close_ref=close)
                return

        if holding_days >= max_hold:
            self._close_position(close_ref=close)
