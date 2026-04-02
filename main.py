"""Main orchestrator: fetch data, compute indicators, generate signals, backtest, report."""

import sys
import warnings
warnings.filterwarnings("ignore")

import config
from data import get_asset_data
from indicators import add_trend_indicators, add_entry_indicators
from signals import align_trend_to_entries, generate_signals
from backtest import Backtest
from report import compute_metrics, generate_report


def run_backtest(mode: str = "primary"):
    """Run full backtest for all assets in the given mode."""
    print(f"\n{'='*60}")
    print(f"  Running {mode.upper()} backtest")
    print(f"  Timeframes: {config.TIMEFRAMES[mode]}")
    if mode == "primary":
        print(f"  Period: {config.DATA_START} to {config.DATA_END}")
    else:
        print(f"  Period: {config.VALIDATION_START} to {config.VALIDATION_END}")
    print(f"  Initial Capital: ${config.INITIAL_CAPITAL:,} per asset")
    print(f"{'='*60}")

    all_trades = {}
    all_curves = {}
    all_metrics = {}

    for asset_name, asset_cfg in config.ASSETS.items():
        print(f"\n--- {asset_name} ({asset_cfg['ticker']}) ---")

        try:
            # 1. Fetch data
            trend_df, entry_df = get_asset_data(asset_name, asset_cfg, mode)

            if len(entry_df) < 100:
                print(f"  WARNING: Only {len(entry_df)} entry bars. Skipping.")
                continue

            # 2. Compute indicators
            print(f"  Computing indicators...")
            trend_df = add_trend_indicators(trend_df)
            entry_df = add_entry_indicators(entry_df)

            # 3. Align trend to entry timeframe
            trend_signal = align_trend_to_entries(trend_df, entry_df)

            # Trend distribution
            n_bull = (trend_signal == 1).sum()
            n_bear = (trend_signal == -1).sum()
            n_neut = (trend_signal == 0).sum()
            print(f"  Trend distribution: Bull={n_bull}, Bear={n_bear}, Neutral={n_neut}")

            # 4. Generate signals
            print(f"  Generating signals...")
            signals = generate_signals(entry_df, trend_signal)
            n_long = (signals["signal"] == 1).sum()
            n_short = (signals["signal"] == -1).sum()
            print(f"  Raw signals: {n_long} long, {n_short} short")

            # 5. Run backtest
            print(f"  Running backtest...")
            bt = Backtest(asset_name, asset_cfg, entry_df, signals)
            trades = bt.run()

            print(f"  Completed: {len(trades)} trades, Final equity: ${bt.equity:,.2f}")

            all_trades[asset_name] = trades
            all_curves[asset_name] = bt.equity_curve
            all_metrics[asset_name] = compute_metrics(trades, bt.equity_curve)

        except Exception as e:
            print(f"  ERROR processing {asset_name}: {e}")
            import traceback
            traceback.print_exc()

    # 6. Generate report
    if all_trades:
        generate_report(all_trades, all_curves, all_metrics, mode)

    return all_trades, all_curves, all_metrics


def main():
    print("=" * 60)
    print("  MULTI-TIMEFRAME BOLLINGER BAND STRATEGY BACKTESTER")
    print("  Assets: Gold (GC=F), NASDAQ (NQ=F), Bitcoin (BTC-USD)")
    print("=" * 60)
    print()
    print("Strategy: Custom Bollinger Band zones (EMA/WMA on High/Low)")
    print("          + Moving Average trend filter (20/60 MA)")
    print("          + Candlestick pattern confirmation")
    print("          + Pullback and Breakout entries")
    print()
    print("Trading Costs:")
    for name, cfg in config.ASSETS.items():
        if cfg["cost_type"] == "fixed":
            print(f"  {name}: ${cfg['cost_per_side']}/side (fixed)")
        else:
            print(f"  {name}: {cfg['cost_pct']*100}%/trade (percentage)")
    print()

    # Run primary backtest (weekly/daily, full history)
    primary_trades, primary_curves, primary_metrics = run_backtest("primary")

    # Run validation backtest (daily/hourly, last ~2 years)
    print("\n\n")
    val_trades, val_curves, val_metrics = run_backtest("validation")

    # Final summary
    print(f"\n\n{'#'*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*60}")

    for mode_name, metrics in [("PRIMARY (Weekly/Daily, ~20yr)", primary_metrics),
                                ("VALIDATION (Daily/Hourly, ~2yr)", val_metrics)]:
        print(f"\n  {mode_name}:")
        total_pnl = sum(m.get("total_pnl", 0) for m in metrics.values())
        total_trades = sum(m.get("total_trades", 0) for m in metrics.values())
        total_costs = sum(m.get("total_costs", 0) for m in metrics.values())
        n = len(metrics) or 1
        total_initial = config.INITIAL_CAPITAL * n

        print(f"    Total trades:  {total_trades}")
        print(f"    Total P&L:     ${total_pnl:,.2f}")
        print(f"    Total costs:   ${total_costs:,.2f}")
        print(f"    Return:        {(total_pnl/total_initial)*100:.2f}%")
        if total_pnl > 0:
            print(f"    Verdict:       PROFITABLE")
        else:
            print(f"    Verdict:       NOT PROFITABLE")

    print(f"\nOutput files saved to: {config.OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
