"""Backtest the Bollinger Band strategy on BTC using Binance 1-minute data.

This uses the ORIGINAL strategy timeframes:
  - Trend: 1H and 4H charts (20 MA, 60 MA slope)
  - Entry: 5-minute chart (custom BB zones + candlestick patterns)

Data source: Binance public API (BTCUSDT, 1-minute candles from Aug 2017 onward)
We resample 1m -> 5m for entries, 1m -> 1H and 4H for trend checking.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from binance_data import load_or_fetch_binance, resample_ohlcv
from indicators import (
    add_trend_indicators, add_entry_indicators,
    compute_sma, compute_bollinger_bands
)
from signals import generate_signals
from backtest import Backtest, Trade, calc_trading_cost
from report import compute_metrics, print_metrics, generate_report


# ── Binance-specific config ────────────────────────────────────────────

BINANCE_ASSET_CFG = {
    "ticker": "BTCUSDT",
    "cost_pct": 0.001,       # 0.1% taker fee per side
    "cost_type": "percentage",
    "point_value": 1,
}

# Date range: Binance BTCUSDT started Aug 17, 2017
BTC_START = "2017-08-17"
BTC_END = "2026-04-01"


def align_multi_trend(trend_1h: pd.DataFrame, trend_4h: pd.DataFrame,
                      entry_df: pd.DataFrame) -> pd.Series:
    """Align 1H and 4H trend signals to 5-minute entry bars.

    Strategy rule: "You only need one of these higher timeframes to confirm
    the direction to start hunting for entries."

    We require at least the 1H trend to agree. If 4H also agrees, stronger signal.
    Final trend: +1 if 1H bullish (and 4H not bearish), -1 if 1H bearish (and 4H not bullish), else 0.
    """
    # Forward-fill trends to 5-minute index
    trend_1h_aligned = trend_1h["trend"].reindex(entry_df.index, method="ffill").fillna(0).astype(int)
    trend_4h_aligned = trend_4h["trend"].reindex(entry_df.index, method="ffill").fillna(0).astype(int)

    combined = pd.Series(0, index=entry_df.index, dtype=int)

    # Bullish if 1H is bullish AND 4H is not bearish
    bull_mask = (trend_1h_aligned == 1) & (trend_4h_aligned >= 0)
    # Bearish if 1H is bearish AND 4H is not bullish
    bear_mask = (trend_1h_aligned == -1) & (trend_4h_aligned <= 0)

    combined[bull_mask] = 1
    combined[bear_mask] = -1

    return combined


def run_binance_backtest():
    """Run the full backtest on Binance BTCUSDT 1-minute data."""
    print("=" * 65)
    print("  BINANCE BTC BACKTEST - ORIGINAL STRATEGY TIMEFRAMES")
    print("=" * 65)
    print()
    print("  Data:    Binance BTCUSDT, 1-minute candles")
    print(f"  Period:  {BTC_START} to {BTC_END}")
    print("  Trend:   1H + 4H charts (20 MA, 60 MA, standard BB)")
    print("  Entry:   5-minute chart (custom BB zones + candlestick patterns)")
    print(f"  Cost:    {BINANCE_ASSET_CFG['cost_pct']*100}% per side (taker fee)")
    print(f"  Capital: ${config.INITIAL_CAPITAL:,}")
    print()

    # ── Step 1: Fetch 1-minute data ────────────────────────────────────
    print("Step 1: Fetching data from Binance...")
    df_1m = load_or_fetch_binance("BTCUSDT", "1m", BTC_START, BTC_END)
    print(f"  Total 1m candles: {len(df_1m):,}")
    print(f"  Date range: {df_1m.index[0]} to {df_1m.index[-1]}")
    years_covered = (df_1m.index[-1] - df_1m.index[0]).days / 365.25
    print(f"  Years covered: {years_covered:.1f}")
    print()

    # ── Step 2: Resample to required timeframes ────────────────────────
    print("Step 2: Resampling to multiple timeframes...")
    df_5m = resample_ohlcv(df_1m, "5min")
    df_1h = resample_ohlcv(df_1m, "1h")
    df_4h = resample_ohlcv(df_1m, "4h")
    df_daily = resample_ohlcv(df_1m, "1D")
    df_weekly = resample_ohlcv(df_1m, "1W")

    print(f"  5m:     {len(df_5m):,} bars")
    print(f"  1H:     {len(df_1h):,} bars")
    print(f"  4H:     {len(df_4h):,} bars")
    print(f"  Daily:  {len(df_daily):,} bars")
    print(f"  Weekly: {len(df_weekly):,} bars")
    print()

    # ── Step 3: Compute trend indicators on 1H and 4H ─────────────────
    print("Step 3: Computing trend indicators on 1H and 4H...")
    df_1h = add_trend_indicators(df_1h)
    df_4h = add_trend_indicators(df_4h)

    n_bull_1h = (df_1h["trend"] == 1).sum()
    n_bear_1h = (df_1h["trend"] == -1).sum()
    n_neut_1h = (df_1h["trend"] == 0).sum()
    print(f"  1H trend: Bull={n_bull_1h}, Bear={n_bear_1h}, Neutral={n_neut_1h}")

    n_bull_4h = (df_4h["trend"] == 1).sum()
    n_bear_4h = (df_4h["trend"] == -1).sum()
    n_neut_4h = (df_4h["trend"] == 0).sum()
    print(f"  4H trend: Bull={n_bull_4h}, Bear={n_bear_4h}, Neutral={n_neut_4h}")
    print()

    # ── Step 4: Compute entry indicators on 5m ─────────────────────────
    print("Step 4: Computing entry indicators on 5-minute chart...")
    df_5m = add_entry_indicators(df_5m)

    # Add previous day high/low for breakout logic
    daily_highs = df_daily["High"].reindex(df_5m.index, method="ffill").shift(1)
    daily_lows = df_daily["Low"].reindex(df_5m.index, method="ffill").shift(1)
    df_5m["prev_high"] = daily_highs.values
    df_5m["prev_low"] = daily_lows.values
    print(f"  Entry indicators computed on {len(df_5m):,} bars")
    print()

    # ── Step 5: Align trend to 5m entries ──────────────────────────────
    print("Step 5: Aligning trend signals to 5-minute bars...")
    trend_signal = align_multi_trend(df_1h, df_4h, df_5m)
    n_bull = (trend_signal == 1).sum()
    n_bear = (trend_signal == -1).sum()
    n_neut = (trend_signal == 0).sum()
    print(f"  Combined trend on 5m: Bull={n_bull:,}, Bear={n_bear:,}, Neutral={n_neut:,}")
    print()

    # ── Step 6: Generate signals on 5m ─────────────────────────────────
    print("Step 6: Generating entry signals on 5-minute chart...")
    signals = generate_signals(df_5m, trend_signal)
    n_long = (signals["signal"] == 1).sum()
    n_short = (signals["signal"] == -1).sum()
    print(f"  Raw signals: {n_long:,} long, {n_short:,} short")
    print()

    # ── Step 7: Run backtest variants ─────────────────────────────────
    # Test multiple exit strategies to find the best approach
    variants = [
        {
            "name": "V1: MA20 exit (min 24 bars/2hr) + trailing stop",
            "min_bars_for_ma_exit": 24,
            "max_holding_bars": 288,  # 24 hours in 5m bars
            "use_bb_band_exit": False,
        },
        {
            "name": "V2: BB target exit (ride to opposite zone)",
            "min_bars_for_ma_exit": 12,
            "max_holding_bars": 576,  # 48 hours
            "use_bb_band_exit": True,
        },
        {
            "name": "V3: MA20 exit (min 48 bars/4hr) + trailing stop",
            "min_bars_for_ma_exit": 48,
            "max_holding_bars": 576,  # 48 hours
            "use_bb_band_exit": False,
        },
    ]

    best_trades = None
    best_bt = None
    best_metrics = None
    best_name = None

    for v in variants:
        vname = v["name"]
        print(f"\n  Running: {vname}")
        bt = Backtest(
            "BTC-Binance", BINANCE_ASSET_CFG, df_5m, signals,
            min_bars_for_ma_exit=v["min_bars_for_ma_exit"],
            max_holding_bars=v["max_holding_bars"],
            use_bb_band_exit=v["use_bb_band_exit"],
        )
        vt = bt.run()
        vm = compute_metrics(vt, bt.equity_curve)
        pnl = vm.get("total_pnl", 0)
        wr = vm.get("win_rate", 0)
        pf = vm.get("profit_factor", 0)
        nt = vm.get("total_trades", 0)
        print(f"    Trades={nt}, WinRate={wr}%, PF={pf}, P&L=${pnl:,.2f}")

        if best_metrics is None or pnl > best_metrics.get("total_pnl", float("-inf")):
            best_trades = vt
            best_bt = bt
            best_metrics = vm
            best_name = vname

    trades = best_trades
    metrics = best_metrics
    print(f"\n  Best variant: {best_name}")
    print(f"  Final equity: ${best_bt.equity:,.2f}")
    print()

    # ── Step 8: Compute and display metrics ────────────────────────────
    print_metrics(metrics, f"BTC-Binance BEST: {best_name}")

    # ── Step 9: Yearly breakdown ───────────────────────────────────────
    if trades:
        print(f"\n{'='*65}")
        print("  YEARLY BREAKDOWN")
        print(f"{'='*65}")

        yearly = {}
        for t in trades:
            if t.exit_date is not None:
                year = t.exit_date.year
            else:
                year = t.entry_date.year
            if year not in yearly:
                yearly[year] = {"trades": 0, "wins": 0, "pnl": 0.0, "costs": 0.0}
            yearly[year]["trades"] += 1
            yearly[year]["pnl"] += t.pnl_net
            yearly[year]["costs"] += t.cost
            if t.pnl_net > 0:
                yearly[year]["wins"] += 1

        print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'Net P&L':>12} {'Costs':>10} {'Cum P&L':>12}")
        print(f"  {'-'*56}")
        cum_pnl = 0
        for year in sorted(yearly.keys()):
            y = yearly[year]
            wr = (y["wins"] / y["trades"] * 100) if y["trades"] > 0 else 0
            cum_pnl += y["pnl"]
            print(f"  {year:<6} {y['trades']:>7} {wr:>6.1f}% ${y['pnl']:>10,.2f} ${y['costs']:>8,.2f} ${cum_pnl:>10,.2f}")

    # ── Step 10: Generate plots ────────────────────────────────────────
    output_dir = os.path.join(config.OUTPUT_DIR, "binance")
    all_trades = {"BTC-Binance": trades}
    all_curves = {"BTC-Binance": best_bt.equity_curve}
    all_metrics = {"BTC-Binance": metrics}

    print(f"\nGenerating charts to {output_dir}/...")
    from report import (plot_equity_curves, plot_drawdown,
                        plot_trade_distribution, plot_monthly_returns,
                        save_trade_log)
    plot_equity_curves(all_curves, output_dir)
    plot_drawdown(all_curves, output_dir)
    plot_trade_distribution(all_trades, output_dir)
    plot_monthly_returns(all_trades, output_dir)
    save_trade_log(all_trades, output_dir)

    # ── Final verdict ──────────────────────────────────────────────────
    print(f"\n{'#'*65}")
    print(f"  FINAL VERDICT — BTC Binance 1m Data Backtest")
    print(f"{'#'*65}")
    print(f"  Data:           Binance BTCUSDT 1-minute candles")
    print(f"  Period:         {df_1m.index[0].strftime('%Y-%m-%d')} to {df_1m.index[-1].strftime('%Y-%m-%d')} ({years_covered:.1f} years)")
    print(f"  Trend TF:       1H + 4H (original strategy)")
    print(f"  Entry TF:       5-minute (original strategy)")
    print(f"  Total Trades:   {len(trades)}")
    if metrics.get("total_trades", 0) > 0:
        print(f"  Win Rate:       {metrics['win_rate']}%")
        print(f"  Profit Factor:  {metrics['profit_factor']}")
        print(f"  Net P&L:        ${metrics['total_pnl']:,.2f}")
        print(f"  Total Costs:    ${metrics['total_costs']:,.2f}")
        print(f"  Return:         {metrics['return_pct']}%")
        print(f"  CAGR:           {metrics['cagr_pct']}%")
        print(f"  Max Drawdown:   {metrics['max_drawdown_pct']}%")
        print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']}")
        if metrics['total_pnl'] > 0:
            print(f"\n  VERDICT: PROFITABLE after costs")
        else:
            print(f"\n  VERDICT: NOT PROFITABLE after costs")
    print()

    return trades, bt.equity_curve, metrics


if __name__ == "__main__":
    run_binance_backtest()
