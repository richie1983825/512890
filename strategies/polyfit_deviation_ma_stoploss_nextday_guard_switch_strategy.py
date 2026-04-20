from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


class PolyfitDeviationMAStoplossNextdayGuardSwitchStrategy(Strategy):
    """Polyfit 动态网格策略，止损后次日买回要求偏离度优于止损卖出当日。"""

    base_grid_pct = 0.012
    volatility_scale = 1.0
    trend_sensitivity = 8.0
    max_grid_levels = 3
    take_profit_grid = 0.85
    stop_loss_grid = 1.6
    max_holding_days = 30
    cooldown_days = 1
    min_signal_strength = 0.45
    position_size = 0.5
    position_sizing_coef = 30.0
    initial_position = 0.0

    flat_wait_days = 8
    switch_deviation_m1 = 0.03
    switch_deviation_m2 = 0.01
    switch_fast_ma_window = 5
    switch_slow_ma_window = 10

    def init(self) -> None:
        self.entry_bar_index: int | None = None
        self.entry_price = np.nan
        self.entry_level = 1
        self.entry_grid_step = np.nan
        self.cooldown = 0
        self.initialized_position = False
        self.ending_position = 0.0
        self.current_entry_reason = ""
        self.trade_reason_records: list[dict[str, str | int | float]] = []
        self.flat_days = 0
        self.switch_mode_active = False
        self.last_stoploss_exit_bar_index: int | None = None
        self.last_stoploss_exit_dev_pct = np.nan

    def _build_poly_entry_reason(self, close: float, pred: float, dev_pct: float, dynamic_grid_step: float, entry_level: int) -> str:
        return (
            f"mean_reversion_grid;base={pred:.4f};dev={dev_pct:.4%};"
            f"grid_step={dynamic_grid_step:.4%};level={entry_level};close={close:.4f}"
        )

    def _build_poly_exit_reason(
        self,
        dev_pct: float,
        tp_threshold: float,
        sl_threshold: float,
        hold_limit: bool,
    ) -> str:
        reasons: list[str] = []
        if hold_limit:
            reasons.append(f"max_holding_days({int(self.max_holding_days)})")
        if dev_pct >= tp_threshold:
            reasons.append(f"take_profit_grid(dev={dev_pct:.4%}>=tp={tp_threshold:.4%})")
        if dev_pct <= -sl_threshold:
            reasons.append(f"stop_loss_grid(dev={dev_pct:.4%}<=-{sl_threshold:.4%})")
        return "; ".join(reasons) if reasons else "grid_exit_runtime_unknown"

    def _build_switch_entry_reason(self, close: float, pred: float, dev_pct: float, fast_ma: float, slow_ma: float) -> str:
        return (
            f"deviation_ma_switch_buy;base={pred:.4f};close={close:.4f};dev={dev_pct:.4%};"
            f"ma{int(self.switch_fast_ma_window)}={fast_ma:.4f};ma{int(self.switch_slow_ma_window)}={slow_ma:.4f};"
            f"m1={float(self.switch_deviation_m1):.4%}"
        )

    def _build_switch_exit_reason(self, close: float, pred: float, dev_pct: float, fast_ma: float, slow_ma: float) -> str:
        return (
            f"deviation_ma_switch_sell;base={pred:.4f};close={close:.4f};dev={dev_pct:.4%};"
            f"ma{int(self.switch_fast_ma_window)}={fast_ma:.4f};ma{int(self.switch_slow_ma_window)}={slow_ma:.4f};"
            f"m1={float(self.switch_deviation_m1):.4%}"
        )

    def _get_switch_ma_values(self) -> tuple[float, float]:
        fast_col = f"MA{int(self.switch_fast_ma_window)}"
        slow_col = f"MA{int(self.switch_slow_ma_window)}"
        fast_ma = float(getattr(self.data, fast_col)[-1])
        slow_ma = float(getattr(self.data, slow_col)[-1])
        return fast_ma, slow_ma

    def _switch_full_position_size(self) -> float:
        return float(np.nextafter(1.0, 0.0))

    def _record_exit_reason(self, bar_index: int, close: float, portion: float, exit_reason: str) -> None:
        exit_time = self.data.index[-1]
        entry_time = self.data.index[self.entry_bar_index] if self.entry_bar_index is not None else exit_time
        self.trade_reason_records.append(
            {
                "EntryTime": pd.Timestamp(entry_time),
                "ExitTime": pd.Timestamp(exit_time),
                "EntryReason": self.current_entry_reason or "entry_runtime_unknown",
                "ExitReason": exit_reason,
                "ExitPortion": float(portion),
                "SignalBar": int(bar_index),
                "SignalClose": float(close),
            }
        )

    def _reset_state_if_flat(self) -> None:
        if self.position:
            self.flat_days = 0
            return

        self.entry_bar_index = None
        self.entry_price = np.nan
        self.entry_level = 1
        self.entry_grid_step = np.nan
        self.flat_days += 1

    def _passes_stoploss_nextday_guard(self, bar_index: int, dev_pct: float) -> bool:
        if self.last_stoploss_exit_bar_index is None or np.isnan(self.last_stoploss_exit_dev_pct):
            return True
        if bar_index != self.last_stoploss_exit_bar_index + 1:
            return True
        return dev_pct > float(self.last_stoploss_exit_dev_pct)

    def _clear_stoploss_nextday_guard_if_expired(self, bar_index: int) -> None:
        if self.last_stoploss_exit_bar_index is None:
            return
        if bar_index > self.last_stoploss_exit_bar_index + 1:
            self.last_stoploss_exit_bar_index = None
            self.last_stoploss_exit_dev_pct = np.nan

    def _polyfit_entry(self, close: float, pred: float, dev_pct: float, dynamic_grid_step: float, rolling_vol_pct: float, bar_index: int) -> bool:
        signal_strength = abs(dev_pct) / max(dynamic_grid_step, 1e-9)
        entry_level = int(np.clip(np.floor(signal_strength), 1, int(self.max_grid_levels)))
        entry_threshold = -entry_level * dynamic_grid_step

        if dev_pct > entry_threshold or signal_strength < float(self.min_signal_strength):
            return False

        size = float(
            np.clip(
                abs(dev_pct) * (1.0 + max(rolling_vol_pct, 0.0)) * float(self.position_sizing_coef),
                0.0,
                float(np.clip(self.position_size, 0.0, 1.0)),
            )
        )
        if size <= 0:
            return False

        self.buy(size=size)
        self.entry_bar_index = bar_index
        self.entry_price = close
        self.entry_level = entry_level
        self.entry_grid_step = dynamic_grid_step
        self.ending_position = size
        self.current_entry_reason = self._build_poly_entry_reason(close, pred, dev_pct, dynamic_grid_step, entry_level)
        self.flat_days = 0
        self.last_stoploss_exit_bar_index = None
        self.last_stoploss_exit_dev_pct = np.nan
        return True

    def _polyfit_exit(self, close: float, dev_pct: float, dynamic_grid_step: float, rolling_vol_pct: float, bar_index: int) -> bool:
        if self.entry_bar_index is None:
            self.entry_bar_index = bar_index
        if np.isnan(self.entry_price):
            self.entry_price = float(self.trades[-1].entry_price) if self.trades else close
        if np.isnan(self.entry_grid_step):
            self.entry_grid_step = dynamic_grid_step

        holding_days = bar_index - self.entry_bar_index
        hold_limit = holding_days >= int(self.max_holding_days)

        ref_step = max(dynamic_grid_step, float(self.entry_grid_step))
        tp_threshold = self.entry_level * ref_step * float(self.take_profit_grid)
        sl_threshold = self.entry_level * ref_step * float(self.stop_loss_grid)

        take_profit = dev_pct >= tp_threshold
        stop_loss = dev_pct <= -sl_threshold
        # If stop-loss is triggered but current price is still above entry,
        # keep holding and do not execute the stop-loss exit.
        if stop_loss and close > float(self.entry_price):
            stop_loss = False
        if not (hold_limit or take_profit or stop_loss):
            return False

        exit_size = float(
            np.clip(
                abs(dev_pct) * (1.0 + max(rolling_vol_pct, 0.0)) * float(self.position_sizing_coef),
                0.0,
                max(float(self.ending_position), 0.0),
            )
        )
        if exit_size <= 0:
            return False

        portion = float(np.clip(exit_size / max(float(self.ending_position), 1e-9), 0.0, 1.0))
        exit_reason = self._build_poly_exit_reason(dev_pct, tp_threshold, sl_threshold, hold_limit)
        self._record_exit_reason(bar_index, close, portion, exit_reason)
        self.position.close(portion=portion)
        self.ending_position = max(0.0, float(self.ending_position) * (1.0 - portion))
        if self.ending_position <= 1e-6:
            self.ending_position = 0.0
            self.cooldown = int(self.cooldown_days)
            self.current_entry_reason = ""
            if stop_loss:
                self.last_stoploss_exit_bar_index = bar_index
                self.last_stoploss_exit_dev_pct = dev_pct
            else:
                self.last_stoploss_exit_bar_index = None
                self.last_stoploss_exit_dev_pct = np.nan
        return True

    def _should_activate_switch_mode(self, close: float, pred: float, dev_pct: float) -> bool:
        return (
            not self.switch_mode_active
            and not self.position
            and self.flat_days >= int(self.flat_wait_days)
            and close > pred
            and dev_pct > float(self.switch_deviation_m1)
        )

    def _handle_switch_mode(self, close: float, pred: float, dev_pct: float, fast_ma: float, slow_ma: float, bar_index: int) -> bool:
        if self._should_activate_switch_mode(close, pred, dev_pct):
            self.switch_mode_active = True

        if not self.switch_mode_active:
            return False

        if dev_pct < float(self.switch_deviation_m2):
            self.switch_mode_active = False
            return False

        if close <= pred or dev_pct <= float(self.switch_deviation_m1):
            return True

        if fast_ma > slow_ma:
            if not self.position:
                full_size = self._switch_full_position_size()
                self.buy(size=full_size)
                self.entry_bar_index = bar_index
                self.entry_price = close
                self.entry_level = 1
                self.entry_grid_step = max(float(self.base_grid_pct), 1e-9)
                self.ending_position = full_size
                self.current_entry_reason = self._build_switch_entry_reason(close, pred, dev_pct, fast_ma, slow_ma)
                self.flat_days = 0
                self.last_stoploss_exit_bar_index = None
                self.last_stoploss_exit_dev_pct = np.nan
            return True

        if fast_ma < slow_ma:
            if self.position:
                exit_reason = self._build_switch_exit_reason(close, pred, dev_pct, fast_ma, slow_ma)
                self._record_exit_reason(bar_index, close, 1.0, exit_reason)
                self.position.close()
                self.ending_position = 0.0
                self.current_entry_reason = ""
                self.last_stoploss_exit_bar_index = None
                self.last_stoploss_exit_dev_pct = np.nan
            return True

        return True

    def next(self) -> None:
        self._reset_state_if_flat()

        close = float(self.data.Close[-1])
        pred = float(self.data.PolyBasePred[-1])
        dev_pct = float(self.data.PolyDevPct[-1])
        dev_trend = float(self.data.PolyDevTrend[-1])
        rolling_vol_pct = float(self.data.RollingVolPct[-1])
        fast_ma, slow_ma = self._get_switch_ma_values()
        bar_index = len(self.data) - 1

        if any(np.isnan(v) for v in [close, pred, dev_pct, dev_trend, rolling_vol_pct, fast_ma, slow_ma]) or pred <= 0:
            return

        if not self.initialized_position:
            self.initialized_position = True
            init_size = float(np.clip(self.initial_position, 0.0, 1.0))
            if init_size > 0 and not self.position:
                self.buy(size=init_size)
                self.entry_bar_index = bar_index
                self.entry_price = close
                self.entry_level = 1
                self.entry_grid_step = max(float(self.base_grid_pct), 1e-9)
                self.ending_position = init_size
                self.current_entry_reason = "initial_position_carry"
                self.flat_days = 0
                return

        if self.cooldown > 0:
            self.cooldown -= 1

        self._clear_stoploss_nextday_guard_if_expired(bar_index)

        vol_multiplier = 1.0 + float(self.volatility_scale) * max(rolling_vol_pct, 0.0)
        dynamic_grid_step = (
            float(self.base_grid_pct)
            * (1.0 + float(self.trend_sensitivity) * abs(dev_trend))
            * vol_multiplier
        )
        dynamic_grid_step = max(dynamic_grid_step, float(self.base_grid_pct) * 0.3)

        if not self.position and not self._passes_stoploss_nextday_guard(bar_index, dev_pct):
            return

        if self._handle_switch_mode(close, pred, dev_pct, fast_ma, slow_ma, bar_index):
            return

        if not self.position:
            if self.cooldown > 0:
                return
            self._polyfit_entry(close, pred, dev_pct, dynamic_grid_step, rolling_vol_pct, bar_index)
            return

        self._polyfit_exit(close, dev_pct, dynamic_grid_step, rolling_vol_pct, bar_index)