"""Event-driven backtesting engine."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

import config


@dataclass
class Trade:
    asset: str
    direction: int          # +1 long, -1 short
    signal_type: str        # "pullback" or "breakout"
    entry_date: object
    exit_date: object = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    size: float = 0.0
    pnl_gross: float = 0.0
    cost: float = 0.0
    pnl_net: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""


def calc_trading_cost(asset_cfg: dict, price: float, size: float) -> float:
    """Calculate one-way trading cost."""
    if asset_cfg["cost_type"] == "fixed":
        return asset_cfg["cost_per_side"] * abs(size)
    else:
        return asset_cfg["cost_pct"] * price * abs(size)


class Backtest:
    def __init__(self, asset_name: str, asset_cfg: dict,
                 entry_df: pd.DataFrame, signals_df: pd.DataFrame):
        self.asset_name = asset_name
        self.asset_cfg = asset_cfg
        self.entry_df = entry_df
        self.signals_df = signals_df

        self.capital = config.INITIAL_CAPITAL
        self.equity = config.INITIAL_CAPITAL
        self.position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple] = []

    def run(self) -> List[Trade]:
        """Run the backtest bar-by-bar."""
        df = self.entry_df
        sig = self.signals_df

        for i in range(1, len(df)):
            bar = df.iloc[i]
            prev_bar = df.iloc[i - 1]
            date = df.index[i]

            # Track mark-to-market equity
            if self.position is not None:
                mtm = self._mark_to_market(bar)
                self.equity_curve.append((date, mtm))
            else:
                self.equity_curve.append((date, self.equity))

            # 1. Check exits on current bar
            if self.position is not None:
                self._check_exit(bar, prev_bar, date, i)

            # 2. Check entries (only if flat, using previous bar's signal)
            if self.position is None and i >= 2:
                prev_signal = sig.iloc[i - 1]
                if prev_signal["signal"] != 0:
                    self._enter(bar, prev_signal, date)

        # Close any remaining position at last bar
        if self.position is not None:
            last_bar = df.iloc[-1]
            self._close_position(last_bar["Close"], df.index[-1], "end_of_data")

        return self.trades

    def _enter(self, bar, signal, date):
        """Open a new position at bar's Open price."""
        direction = signal["signal"]
        entry_price = bar["Open"]
        stop_distance = signal["stop_distance"]

        if np.isnan(stop_distance) or stop_distance <= 0:
            stop_distance = entry_price * 0.02  # fallback 2% stop

        # Position sizing: risk 1% of equity
        risk_amount = self.equity * config.RISK_PER_TRADE
        size = risk_amount / stop_distance

        # Cap position size
        max_size = (self.equity * config.MAX_POSITION_PCT) / entry_price
        size = min(size, max_size)

        if size <= 0:
            return

        # Stop loss
        if direction == 1:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Entry cost
        entry_cost = calc_trading_cost(self.asset_cfg, entry_price, size)

        self.position = Trade(
            asset=self.asset_name,
            direction=direction,
            signal_type=signal["signal_type"],
            entry_date=date,
            entry_price=entry_price,
            stop_loss=stop_loss,
            size=size,
            cost=entry_cost,
        )

    def _check_exit(self, bar, prev_bar, date, bar_idx):
        """Check exit conditions for current position."""
        pos = self.position
        bars_held = bar_idx - self._entry_bar_idx()

        # Exit 1: Stop loss hit
        if pos.direction == 1 and bar["Low"] <= pos.stop_loss:
            self._close_position(pos.stop_loss, date, "stop_loss")
            return
        if pos.direction == -1 and bar["High"] >= pos.stop_loss:
            self._close_position(pos.stop_loss, date, "stop_loss")
            return

        # Exit 2: BB middle band exit (mean reversion target)
        if "entry_ma20" in bar.index:
            ma20 = bar["entry_ma20"]
            if not np.isnan(ma20):
                if pos.direction == 1 and bar["Close"] < ma20 and bars_held >= 3:
                    self._close_position(bar["Close"], date, "ma_cross")
                    return
                if pos.direction == -1 and bar["Close"] > ma20 and bars_held >= 3:
                    self._close_position(bar["Close"], date, "ma_cross")
                    return

        # Exit 3: Max holding period
        if bars_held >= config.MAX_HOLDING_BARS:
            self._close_position(bar["Close"], date, "max_hold")
            return

        # Trailing stop: once in profit by 1x risk, trail stop to entry
        risk = abs(pos.entry_price - pos.stop_loss)
        if pos.direction == 1:
            unrealized = bar["High"] - pos.entry_price
            if unrealized >= 2 * risk:
                new_stop = pos.entry_price + risk
                pos.stop_loss = max(pos.stop_loss, new_stop)
            elif unrealized >= risk:
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
        else:
            unrealized = pos.entry_price - bar["Low"]
            if unrealized >= 2 * risk:
                new_stop = pos.entry_price - risk
                pos.stop_loss = min(pos.stop_loss, new_stop)
            elif unrealized >= risk:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price)

    def _close_position(self, exit_price: float, date, reason: str):
        """Close current position and record trade."""
        pos = self.position
        exit_cost = calc_trading_cost(self.asset_cfg, exit_price, pos.size)

        if pos.direction == 1:
            pnl_gross = (exit_price - pos.entry_price) * pos.size
        else:
            pnl_gross = (pos.entry_price - exit_price) * pos.size

        total_cost = pos.cost + exit_cost
        pnl_net = pnl_gross - total_cost

        pos.exit_date = date
        pos.exit_price = exit_price
        pos.pnl_gross = pnl_gross
        pos.cost = total_cost
        pos.pnl_net = pnl_net
        pos.exit_reason = reason
        pos.bars_held = self._bars_since_entry(date)

        self.trades.append(pos)
        self.equity += pnl_net
        self.position = None

    def _mark_to_market(self, bar) -> float:
        """Calculate current equity including unrealized P&L."""
        pos = self.position
        if pos.direction == 1:
            unrealized = (bar["Close"] - pos.entry_price) * pos.size
        else:
            unrealized = (pos.entry_price - bar["Close"]) * pos.size
        return self.equity + unrealized

    def _entry_bar_idx(self) -> int:
        """Find the bar index of position entry."""
        try:
            return self.entry_df.index.get_loc(self.position.entry_date)
        except KeyError:
            return 0

    def _bars_since_entry(self, current_date) -> int:
        try:
            entry_idx = self.entry_df.index.get_loc(self.position.entry_date)
            current_idx = self.entry_df.index.get_loc(current_date)
            return current_idx - entry_idx
        except KeyError:
            return 0
