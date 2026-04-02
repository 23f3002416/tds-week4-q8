"""Reporting: metrics calculation and visualization."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict

from backtest import Trade
import config


def compute_metrics(trades: List[Trade], equity_curve: List[tuple],
                    initial_capital: float = None) -> dict:
    """Compute comprehensive backtest metrics from trade list."""
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    if not trades:
        return {"total_trades": 0, "note": "No trades generated"}

    pnls = [t.pnl_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    final_equity = initial_capital + total_pnl

    # Win rate
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0

    # Average win/loss
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Profit factor
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses)) if losses else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Total costs
    total_costs = sum(t.cost for t in trades)

    # Max drawdown from equity curve
    if equity_curve:
        eq_values = [e[1] for e in equity_curve]
        eq_series = pd.Series(eq_values)
        rolling_max = eq_series.cummax()
        drawdowns = (eq_series - rolling_max) / rolling_max * 100
        max_dd = drawdowns.min()
        max_dd_abs = (eq_series - rolling_max).min()
    else:
        max_dd = 0
        max_dd_abs = 0

    # CAGR
    if equity_curve:
        dates = [e[0] for e in equity_curve]
        years = (dates[-1] - dates[0]).days / 365.25
        if years > 0 and final_equity > 0:
            cagr = (final_equity / initial_capital) ** (1 / years) - 1
        else:
            cagr = 0
    else:
        cagr = 0

    # Sharpe ratio (daily returns)
    if equity_curve and len(equity_curve) > 1:
        eq_series = pd.Series([e[1] for e in equity_curve])
        daily_returns = eq_series.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Average bars held
    avg_bars = np.mean([t.bars_held for t in trades])

    # Win/loss streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for p in pnls:
        if p > 0:
            streak = max(1, streak + 1) if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = min(-1, streak - 1) if streak < 0 else -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    # Breakdown by signal type
    pullback_trades = [t for t in trades if t.signal_type == "pullback"]
    breakout_trades = [t for t in trades if t.signal_type == "breakout"]
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_costs": round(total_costs, 2),
        "final_equity": round(final_equity, 2),
        "return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "max_drawdown_abs": round(max_dd_abs, 2),
        "sharpe_ratio": round(sharpe, 3),
        "avg_bars_held": round(avg_bars, 1),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "pullback_trades": len(pullback_trades),
        "breakout_trades": len(breakout_trades),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "pullback_pnl": round(sum(t.pnl_net for t in pullback_trades), 2),
        "breakout_pnl": round(sum(t.pnl_net for t in breakout_trades), 2),
        "long_pnl": round(sum(t.pnl_net for t in long_trades), 2),
        "short_pnl": round(sum(t.pnl_net for t in short_trades), 2),
    }


def print_metrics(metrics: dict, title: str = ""):
    """Print metrics summary to console."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    if metrics.get("total_trades", 0) == 0:
        print("  No trades generated.")
        return

    print(f"  Total Trades:       {metrics['total_trades']}")
    print(f"  Win Rate:           {metrics['win_rate']}%")
    print(f"  Profit Factor:      {metrics['profit_factor']}")
    print(f"  Total P&L (net):    ${metrics['total_pnl']:,.2f}")
    print(f"  Total Costs:        ${metrics['total_costs']:,.2f}")
    print(f"  Final Equity:       ${metrics['final_equity']:,.2f}")
    print(f"  Return:             {metrics['return_pct']}%")
    print(f"  CAGR:               {metrics['cagr_pct']}%")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']}%")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']}")
    print(f"  Avg Win:            ${metrics['avg_win']:,.2f}")
    print(f"  Avg Loss:           ${metrics['avg_loss']:,.2f}")
    print(f"  Avg Bars Held:      {metrics['avg_bars_held']}")
    print(f"  Max Win Streak:     {metrics['max_win_streak']}")
    print(f"  Max Loss Streak:    {metrics['max_loss_streak']}")
    print(f"  ---")
    print(f"  Pullback trades:    {metrics['pullback_trades']}  (P&L: ${metrics['pullback_pnl']:,.2f})")
    print(f"  Breakout trades:    {metrics['breakout_trades']}  (P&L: ${metrics['breakout_pnl']:,.2f})")
    print(f"  Long trades:        {metrics['long_trades']}  (P&L: ${metrics['long_pnl']:,.2f})")
    print(f"  Short trades:       {metrics['short_trades']}  (P&L: ${metrics['short_pnl']:,.2f})")


def plot_equity_curves(all_curves: Dict[str, List[tuple]], output_dir: str = None):
    """Plot equity curves for all assets and combined."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(len(all_curves) + 1, 1,
                              figsize=(14, 4 * (len(all_curves) + 1)))
    if len(all_curves) + 1 == 1:
        axes = [axes]

    # Individual equity curves
    combined_equity = {}
    for idx, (name, curve) in enumerate(all_curves.items()):
        if not curve:
            continue
        dates = [c[0] for c in curve]
        values = [c[1] for c in curve]
        axes[idx].plot(dates, values, linewidth=0.8)
        axes[idx].set_title(f"{name} - Equity Curve")
        axes[idx].set_ylabel("Equity ($)")
        axes[idx].axhline(y=config.INITIAL_CAPITAL, color="gray",
                          linestyle="--", alpha=0.5, label="Initial Capital")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

        for d, v in zip(dates, values):
            if d not in combined_equity:
                combined_equity[d] = config.INITIAL_CAPITAL * len(all_curves)
            combined_equity[d] += (v - config.INITIAL_CAPITAL)

    # Combined
    if combined_equity:
        sorted_dates = sorted(combined_equity.keys())
        combined_values = [combined_equity[d] for d in sorted_dates]
        axes[-1].plot(sorted_dates, combined_values, linewidth=0.8, color="purple")
        axes[-1].set_title("Combined Portfolio - Equity Curve")
        axes[-1].set_ylabel("Equity ($)")
        total_initial = config.INITIAL_CAPITAL * len(all_curves)
        axes[-1].axhline(y=total_initial, color="gray", linestyle="--", alpha=0.5)
        axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "equity_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved equity curves to {path}")


def plot_drawdown(all_curves: Dict[str, List[tuple]], output_dir: str = None):
    """Plot drawdown curves."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(len(all_curves), 1,
                              figsize=(14, 3 * len(all_curves)))
    if len(all_curves) == 1:
        axes = [axes]

    for idx, (name, curve) in enumerate(all_curves.items()):
        if not curve:
            continue
        eq = pd.Series([c[1] for c in curve], index=[c[0] for c in curve])
        rolling_max = eq.cummax()
        dd = (eq - rolling_max) / rolling_max * 100
        axes[idx].fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
        axes[idx].set_title(f"{name} - Drawdown")
        axes[idx].set_ylabel("Drawdown (%)")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "drawdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved drawdown chart to {path}")


def plot_trade_distribution(all_trades: Dict[str, List[Trade]], output_dir: str = None):
    """Histogram of P&L per trade."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(all_trades), figsize=(6 * len(all_trades), 5))
    if len(all_trades) == 1:
        axes = [axes]

    for idx, (name, trades) in enumerate(all_trades.items()):
        if not trades:
            continue
        pnls = [t.pnl_net for t in trades]
        axes[idx].hist(pnls, bins=min(50, max(10, len(pnls) // 3)),
                       color="steelblue", edgecolor="black", alpha=0.7)
        axes[idx].axvline(x=0, color="red", linestyle="--")
        axes[idx].set_title(f"{name} - Trade P&L Distribution")
        axes[idx].set_xlabel("P&L ($)")
        axes[idx].set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(output_dir, "trade_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved trade distribution to {path}")


def plot_monthly_returns(all_trades: Dict[str, List[Trade]], output_dir: str = None):
    """Monthly returns heatmap."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for name, trades in all_trades.items():
        if not trades:
            continue

        # Aggregate monthly P&L
        monthly = {}
        for t in trades:
            if t.exit_date is not None:
                key = (t.exit_date.year, t.exit_date.month)
                monthly[key] = monthly.get(key, 0) + t.pnl_net

        if not monthly:
            continue

        years = sorted(set(k[0] for k in monthly))
        months = list(range(1, 13))
        data = np.full((len(years), 12), np.nan)
        for (y, m), pnl in monthly.items():
            yi = years.index(y)
            data[yi, m - 1] = pnl

        fig, ax = plt.subplots(figsize=(14, max(4, len(years) * 0.4)))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.set_title(f"{name} - Monthly Returns ($)")
        plt.colorbar(im, ax=ax, label="P&L ($)")

        # Annotate cells
        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:,.0f}", ha="center", va="center",
                            fontsize=6, color="black")

        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        path = os.path.join(output_dir, f"monthly_returns_{safe_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved monthly returns heatmap to {path}")


def save_trade_log(all_trades: Dict[str, List[Trade]], output_dir: str = None):
    """Save all trades to CSV."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for name, trades in all_trades.items():
        for t in trades:
            rows.append({
                "asset": t.asset,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "signal_type": t.signal_type,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "size": round(t.size, 6),
                "pnl_gross": round(t.pnl_gross, 2),
                "cost": round(t.cost, 2),
                "pnl_net": round(t.pnl_net, 2),
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "trades.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(rows)} trades to {path}")


def generate_report(all_trades: Dict[str, List[Trade]],
                    all_curves: Dict[str, List[tuple]],
                    all_metrics: Dict[str, dict],
                    mode: str = "primary"):
    """Generate full report: console output + plots + CSV."""
    print(f"\n{'#'*60}")
    print(f"  BACKTEST REPORT - {mode.upper()} MODE")
    print(f"{'#'*60}")

    for name, metrics in all_metrics.items():
        print_metrics(metrics, f"{name} ({mode})")

    # Combined metrics
    total_trades_all = sum(m.get("total_trades", 0) for m in all_metrics.values())
    total_pnl_all = sum(m.get("total_pnl", 0) for m in all_metrics.values())
    total_costs_all = sum(m.get("total_costs", 0) for m in all_metrics.values())
    n_assets = len(all_metrics)

    print(f"\n{'='*60}")
    print(f"  COMBINED PORTFOLIO SUMMARY ({mode})")
    print(f"{'='*60}")
    print(f"  Total Trades (all):    {total_trades_all}")
    print(f"  Total P&L (all):       ${total_pnl_all:,.2f}")
    print(f"  Total Costs (all):     ${total_costs_all:,.2f}")
    print(f"  Total Initial Capital: ${config.INITIAL_CAPITAL * n_assets:,.2f}")
    total_initial = config.INITIAL_CAPITAL * n_assets
    print(f"  Final Combined Equity: ${total_initial + total_pnl_all:,.2f}")
    if total_initial > 0:
        print(f"  Combined Return:       {(total_pnl_all / total_initial) * 100:.2f}%")

    # Verdict
    print(f"\n  --- PROFITABILITY VERDICT ---")
    if total_pnl_all > 0:
        print(f"  The strategy is NET PROFITABLE across all assets.")
    else:
        print(f"  The strategy is NET UNPROFITABLE across all assets.")

    # Generate plots and CSV
    print(f"\nGenerating charts...")
    plot_equity_curves(all_curves)
    plot_drawdown(all_curves)
    plot_trade_distribution(all_trades)
    plot_monthly_returns(all_trades)
    save_trade_log(all_trades)
