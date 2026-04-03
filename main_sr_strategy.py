"""Backtest: Buy at Resistance / Sell at Support on BTC 15m candles.

Strategy:
  - Identify support and resistance levels using recent swing highs/lows
  - BUY when price touches/breaks resistance (breakout long)
  - SELL when price touches/breaks support (breakdown short)
  - Fixed 1% take profit, 1% stop loss (1:1 risk-reward)
  - Timeframe: 15-minute chart
  - Data: Binance BTCUSDT 1-minute candles resampled to 15m

Support/Resistance identification:
  - Resistance: highest high of last N bars (lookback window)
  - Support: lowest low of last N bars (lookback window)
  - A level is "tested" when price approaches within a threshold
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

import config
from binance_data import load_or_fetch_binance, resample_ohlcv

# ── Strategy Parameters ────────────────────────────────────────────────

# Support/Resistance lookback (in 15m bars)
SR_LOOKBACK = 96          # 96 bars * 15m = 24 hours lookback for S/R levels
SR_MIN_TOUCHES = 2        # Minimum times a level was tested to count as S/R
SR_PROXIMITY_PCT = 0.002  # Price within 0.2% of level counts as "at" the level
SR_BUFFER_BARS = 4        # Bars to wait after a level forms before trading it

# Trade parameters
TP_PCT = 0.02   # 2% take profit
SL_PCT = 0.01   # 1% stop loss

# Cost
COST_PCT = 0.001  # 0.1% per side (Binance taker fee)

# Capital
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.02  # 2% of equity per trade
MAX_POSITION_PCT = 0.25

# Data
BTC_START = "2017-08-17"
BTC_END = "2026-04-01"


@dataclass
class Trade:
    direction: int  # +1 long, -1 short
    entry_date: object
    entry_price: float
    tp_price: float
    sl_price: float
    size: float
    exit_date: object = None
    exit_price: float = 0.0
    pnl_gross: float = 0.0
    cost: float = 0.0
    pnl_net: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    level_type: str = ""  # "resistance" or "support"


def find_support_resistance(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Identify rolling support and resistance levels.

    Resistance = rolling max of High over lookback period
    Support = rolling min of Low over lookback period
    """
    df = df.copy()
    df["resistance"] = df["High"].rolling(window=lookback, min_periods=lookback).max()
    df["support"] = df["Low"].rolling(window=lookback, min_periods=lookback).min()

    # Shift by buffer bars so we don't trade the bar that sets the level
    df["resistance"] = df["resistance"].shift(SR_BUFFER_BARS)
    df["support"] = df["support"].shift(SR_BUFFER_BARS)

    return df


def run_backtest(df: pd.DataFrame) -> tuple:
    """Run the support/resistance backtest on 15m data."""
    df = find_support_resistance(df, SR_LOOKBACK)

    equity = INITIAL_CAPITAL
    position: Optional[Trade] = None
    trades: List[Trade] = []
    equity_curve = []

    for i in range(SR_LOOKBACK + SR_BUFFER_BARS + 1, len(df)):
        bar = df.iloc[i]
        date = df.index[i]
        resistance = bar["resistance"]
        support = bar["support"]

        # Track equity
        if position is not None:
            if position.direction == 1:
                mtm = equity + (bar["Close"] - position.entry_price) * position.size
            else:
                mtm = equity + (position.entry_price - bar["Close"]) * position.size
            equity_curve.append((date, mtm))
        else:
            equity_curve.append((date, equity))

        # ── Check exits ────────────────────────────────────────────
        if position is not None:
            hit_tp = False
            hit_sl = False

            if position.direction == 1:  # Long
                if bar["High"] >= position.tp_price:
                    hit_tp = True
                if bar["Low"] <= position.sl_price:
                    hit_sl = True
            else:  # Short
                if bar["Low"] <= position.tp_price:
                    hit_tp = True
                if bar["High"] >= position.sl_price:
                    hit_sl = True

            # If both hit in same bar, assume SL hit first (conservative)
            if hit_sl:
                exit_price = position.sl_price
                exit_reason = "stop_loss"
            elif hit_tp:
                exit_price = position.tp_price
                exit_reason = "take_profit"
            else:
                position.bars_held += 1
                continue  # Still in trade

            # Close position
            if position.direction == 1:
                pnl_gross = (exit_price - position.entry_price) * position.size
            else:
                pnl_gross = (position.entry_price - exit_price) * position.size

            exit_cost = COST_PCT * exit_price * position.size
            total_cost = position.cost + exit_cost
            pnl_net = pnl_gross - total_cost

            position.exit_date = date
            position.exit_price = exit_price
            position.pnl_gross = pnl_gross
            position.cost = total_cost
            position.pnl_net = pnl_net
            position.exit_reason = exit_reason
            position.bars_held += 1

            trades.append(position)
            equity += pnl_net
            position = None

        # ── Check entries (only if flat) ───────────────────────────
        if position is None and not np.isnan(resistance) and not np.isnan(support):
            close = bar["Close"]
            high = bar["High"]
            low = bar["Low"]

            # Buy at resistance: price reaches/breaks resistance level
            near_resistance = high >= resistance * (1 - SR_PROXIMITY_PCT)
            # Sell at support: price reaches/breaks support level
            near_support = low <= support * (1 + SR_PROXIMITY_PCT)

            signal = 0
            entry_price = close
            level_type = ""

            if near_resistance and not near_support:
                signal = 1  # Buy at resistance (breakout)
                entry_price = close
                level_type = "resistance"
            elif near_support and not near_resistance:
                signal = -1  # Sell at support (breakdown)
                entry_price = close
                level_type = "support"

            if signal != 0:
                # Position sizing
                risk_amount = equity * RISK_PER_TRADE
                risk_per_unit = entry_price * SL_PCT
                size = risk_amount / risk_per_unit
                max_size = (equity * MAX_POSITION_PCT) / entry_price
                size = min(size, max_size)

                if size <= 0 or equity <= 0:
                    continue

                # Set TP and SL
                if signal == 1:
                    tp_price = entry_price * (1 + TP_PCT)
                    sl_price = entry_price * (1 - SL_PCT)
                else:
                    tp_price = entry_price * (1 - TP_PCT)
                    sl_price = entry_price * (1 + SL_PCT)

                entry_cost = COST_PCT * entry_price * size

                position = Trade(
                    direction=signal,
                    entry_date=date,
                    entry_price=entry_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    size=size,
                    cost=entry_cost,
                    level_type=level_type,
                )

    # Close any remaining position at last bar's close
    if position is not None:
        last = df.iloc[-1]
        if position.direction == 1:
            pnl_gross = (last["Close"] - position.entry_price) * position.size
        else:
            pnl_gross = (position.entry_price - last["Close"]) * position.size
        exit_cost = COST_PCT * last["Close"] * position.size
        position.exit_date = df.index[-1]
        position.exit_price = last["Close"]
        position.pnl_gross = pnl_gross
        position.cost = position.cost + exit_cost
        position.pnl_net = pnl_gross - position.cost
        position.exit_reason = "end_of_data"
        trades.append(position)
        equity += position.pnl_net

    return trades, equity_curve, equity


def compute_metrics(trades: List[Trade], equity_curve: list, initial_capital: float) -> dict:
    """Compute backtest metrics."""
    if not trades:
        return {"total_trades": 0}

    pnls = [t.pnl_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    final_equity = initial_capital + total_pnl
    total_costs = sum(t.cost for t in trades)

    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses)) if losses else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Max drawdown
    eq_vals = [e[1] for e in equity_curve]
    eq_s = pd.Series(eq_vals)
    rolling_max = eq_s.cummax()
    dd = (eq_s - rolling_max) / rolling_max * 100
    max_dd = dd.min()

    # CAGR
    dates = [e[0] for e in equity_curve]
    years = (dates[-1] - dates[0]).days / 365.25 if len(dates) > 1 else 1
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and final_equity > 0 else 0

    # Sharpe
    eq_s2 = pd.Series(eq_vals)
    rets = eq_s2.pct_change().dropna()
    # 15m bars: ~35,040 bars/year (24*4*365)
    sharpe = (rets.mean() / rets.std() * np.sqrt(35040)) if rets.std() > 0 else 0

    avg_bars = np.mean([t.bars_held for t in trades])

    # By type
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]
    tp_exits = [t for t in trades if t.exit_reason == "take_profit"]
    sl_exits = [t for t in trades if t.exit_reason == "stop_loss"]

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_costs": round(total_costs, 2),
        "final_equity": round(final_equity, 2),
        "return_pct": round((total_pnl / initial_capital) * 100, 2),
        "cagr_pct": round(cagr, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "avg_bars_held": round(avg_bars, 1),
        "long_trades": len(long_trades),
        "long_pnl": round(sum(t.pnl_net for t in long_trades), 2),
        "short_trades": len(short_trades),
        "short_pnl": round(sum(t.pnl_net for t in short_trades), 2),
        "tp_exits": len(tp_exits),
        "sl_exits": len(sl_exits),
    }


def print_report(metrics: dict, title: str):
    """Print formatted report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    if metrics["total_trades"] == 0:
        print("  No trades.")
        return

    m = metrics
    print(f"  Total Trades:      {m['total_trades']}")
    print(f"  Win Rate:          {m['win_rate']}%")
    print(f"  Profit Factor:     {m['profit_factor']}")
    print(f"  Net P&L:           ${m['total_pnl']:,.2f}")
    print(f"  Total Costs:       ${m['total_costs']:,.2f}")
    print(f"  Final Equity:      ${m['final_equity']:,.2f}")
    print(f"  Return:            {m['return_pct']}%")
    print(f"  CAGR:              {m['cagr_pct']}%")
    print(f"  Max Drawdown:      {m['max_drawdown_pct']}%")
    print(f"  Sharpe Ratio:      {m['sharpe']}")
    print(f"  Avg Win:           ${m['avg_win']:,.2f}")
    print(f"  Avg Loss:          ${m['avg_loss']:,.2f}")
    print(f"  Avg Bars Held:     {m['avg_bars_held']} ({m['avg_bars_held']*15:.0f} min)")
    print(f"  ---")
    print(f"  TP exits:          {m['tp_exits']}")
    print(f"  SL exits:          {m['sl_exits']}")
    print(f"  Long trades:       {m['long_trades']}  (P&L: ${m['long_pnl']:,.2f})")
    print(f"  Short trades:      {m['short_trades']}  (P&L: ${m['short_pnl']:,.2f})")


def save_results(trades, equity_curve, metrics, output_dir):
    """Save trade log, equity curve plot, and charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Trade log CSV
    rows = []
    for t in trades:
        rows.append({
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "level_type": t.level_type,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "tp_price": round(t.tp_price, 2),
            "sl_price": round(t.sl_price, 2),
            "size": round(t.size, 6),
            "pnl_gross": round(t.pnl_gross, 2),
            "cost": round(t.cost, 2),
            "pnl_net": round(t.pnl_net, 2),
            "bars_held": t.bars_held,
            "exit_reason": t.exit_reason,
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "trades_sr.csv"), index=False)
    print(f"  Saved {len(rows)} trades to {output_dir}/trades_sr.csv")

    # Equity curve
    if equity_curve:
        dates = [e[0] for e in equity_curve]
        vals = [e[1] for e in equity_curve]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        ax1.plot(dates, vals, linewidth=0.5)
        ax1.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("BTC Support/Resistance Strategy - Equity Curve (15m)")
        ax1.set_ylabel("Equity ($)")
        ax1.grid(True, alpha=0.3)

        # Drawdown
        eq_s = pd.Series(vals, index=dates)
        dd = (eq_s - eq_s.cummax()) / eq_s.cummax() * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "equity_sr.png"), dpi=150)
        plt.close()
        print(f"  Saved equity curve to {output_dir}/equity_sr.png")

    # P&L distribution
    if trades:
        pnls = [t.pnl_net for t in trades]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(pnls, bins=min(80, max(20, len(pnls) // 5)),
                color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_title("Trade P&L Distribution")
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distribution_sr.png"), dpi=150)
        plt.close()
        print(f"  Saved distribution to {output_dir}/distribution_sr.png")

    # Monthly returns heatmap
    if trades:
        monthly = {}
        for t in trades:
            if t.exit_date is not None:
                key = (t.exit_date.year, t.exit_date.month)
                monthly[key] = monthly.get(key, 0) + t.pnl_net

        if monthly:
            years = sorted(set(k[0] for k in monthly))
            data = np.full((len(years), 12), np.nan)
            for (y, m), pnl in monthly.items():
                data[years.index(y), m - 1] = pnl

            fig, ax = plt.subplots(figsize=(14, max(4, len(years) * 0.5)))
            im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(12))
            ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"])
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels(years)
            ax.set_title("Monthly P&L ($)")
            plt.colorbar(im, ax=ax)
            for i in range(len(years)):
                for j in range(12):
                    if not np.isnan(data[i, j]):
                        ax.text(j, i, f"{data[i,j]:,.0f}", ha="center", va="center",
                                fontsize=6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "monthly_sr.png"), dpi=150)
            plt.close()
            print(f"  Saved monthly returns to {output_dir}/monthly_sr.png")


def main():
    print("=" * 60)
    print("  BTC SUPPORT/RESISTANCE STRATEGY BACKTEST")
    print("  Buy at Resistance | Sell at Support | 15m Chart")
    print("=" * 60)
    print()
    print(f"  Take Profit:     {TP_PCT*100}%")
    print(f"  Stop Loss:       {SL_PCT*100}%")
    print(f"  Risk/Reward:     1:2")
    print(f"  Trading Cost:    {COST_PCT*100}% per side")
    print(f"  S/R Lookback:    {SR_LOOKBACK} bars ({SR_LOOKBACK*15/60:.0f} hours)")
    print(f"  Capital:         ${INITIAL_CAPITAL:,}")
    print()

    # 1. Load 1m data from Binance
    print("Loading Binance BTCUSDT 1-minute data...")
    df_1m = load_or_fetch_binance("BTCUSDT", "1m", BTC_START, BTC_END)
    print(f"  {len(df_1m):,} candles loaded")
    print()

    # 2. Resample to 15m
    print("Resampling to 15-minute bars...")
    df_15m = resample_ohlcv(df_1m, "15min")
    print(f"  {len(df_15m):,} bars from {df_15m.index[0]} to {df_15m.index[-1]}")
    years = (df_15m.index[-1] - df_15m.index[0]).days / 365.25
    print(f"  Period: {years:.1f} years")
    print()

    # 3. Run backtest
    print("Running backtest...")
    trades, equity_curve, final_equity = run_backtest(df_15m)
    print(f"  {len(trades)} trades completed")
    print(f"  Final equity: ${final_equity:,.2f}")

    # 4. Compute metrics
    metrics = compute_metrics(trades, equity_curve, INITIAL_CAPITAL)
    print_report(metrics, "BTC S/R Strategy (15m) — 1% TP / 1% SL")

    # 5. Yearly breakdown
    if trades:
        print(f"\n{'='*60}")
        print("  YEARLY BREAKDOWN")
        print(f"{'='*60}")
        yearly = {}
        for t in trades:
            yr = (t.exit_date or t.entry_date).year
            if yr not in yearly:
                yearly[yr] = {"n": 0, "w": 0, "pnl": 0.0, "cost": 0.0}
            yearly[yr]["n"] += 1
            yearly[yr]["pnl"] += t.pnl_net
            yearly[yr]["cost"] += t.cost
            if t.pnl_net > 0:
                yearly[yr]["w"] += 1

        print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'Net P&L':>12} {'Costs':>10} {'Cum P&L':>12}")
        print(f"  {'-'*56}")
        cum = 0
        for yr in sorted(yearly):
            y = yearly[yr]
            wr = y["w"] / y["n"] * 100 if y["n"] > 0 else 0
            cum += y["pnl"]
            print(f"  {yr:<6} {y['n']:>7} {wr:>6.1f}% ${y['pnl']:>10,.2f} ${y['cost']:>8,.2f} ${cum:>10,.2f}")

    # 6. Save results
    output_dir = os.path.join(config.OUTPUT_DIR, "sr_strategy")
    print(f"\nSaving results to {output_dir}/...")
    save_results(trades, equity_curve, metrics, output_dir)

    # 7. Verdict
    print(f"\n{'#'*60}")
    print(f"  VERDICT")
    print(f"{'#'*60}")
    if metrics["total_trades"] > 0:
        print(f"  Strategy: Buy at Resistance / Sell at Support")
        print(f"  Timeframe: 15-minute | TP: 1% | SL: 1%")
        print(f"  Period: {years:.1f} years of real Binance data")
        print(f"  Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate']}%")
        print(f"  Net P&L: ${metrics['total_pnl']:,.2f} | Return: {metrics['return_pct']}%")
        print(f"  Costs: ${metrics['total_costs']:,.2f}")
        if metrics["total_pnl"] > 0:
            print(f"\n  PROFITABLE after costs")
        else:
            print(f"\n  NOT PROFITABLE after costs")
    print()


if __name__ == "__main__":
    main()
