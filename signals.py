"""Signal generation: align trend with entry-timeframe triggers."""

import numpy as np
import pandas as pd


def align_trend_to_entries(trend_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.Series:
    """Forward-fill weekly/daily trend signal to entry timeframe index.

    trend_df must have a 'trend' column (+1, -1, 0).
    Returns a Series aligned to entry_df's index.
    """
    trend_series = trend_df["trend"].copy()
    # Normalize timezones: make both tz-naive for alignment
    if trend_series.index.tz is not None:
        trend_series.index = trend_series.index.tz_localize(None)
    entry_index = entry_df.index
    if entry_index.tz is not None:
        entry_index = entry_index.tz_localize(None)
    # Reindex to entry timeframe and forward-fill
    aligned = trend_series.reindex(entry_index, method="ffill")
    aligned.index = entry_df.index  # restore original index
    # Fill any leading NaN with 0 (no trade)
    aligned = aligned.fillna(0).astype(int)
    return aligned


def generate_signals(entry_df: pd.DataFrame, trend: pd.Series) -> pd.DataFrame:
    """Generate trade signals based on entry indicators and trend alignment.

    Returns a DataFrame with columns: signal (+1/-1/0), signal_type, stop_distance.
    Signals are generated on bar i, to be acted upon at bar i+1's Open.
    """
    n = len(entry_df)
    signals = pd.DataFrame(index=entry_df.index, data={
        "signal": np.zeros(n, dtype=int),
        "signal_type": "",
        "stop_distance": np.nan,
    })

    for i in range(1, n):
        t = trend.iloc[i]
        if t == 0:
            continue

        row = entry_df.iloc[i]

        # --- LONG signals (trend == +1) ---
        if t == 1 and row.get("entry_ma20_slope", 0) > 0:
            # Pullback long: price touches buy zone + bullish candle
            if (row["Low"] <= row["buy_zone_upper"] and
                    row["Low"] >= row["buy_zone_lower"] and
                    row["bullish_candle"]):
                signals.iloc[i, signals.columns.get_loc("signal")] = 1
                signals.iloc[i, signals.columns.get_loc("signal_type")] = "pullback"
                stop = row["buy_zone_lower"] - row["Close"] * 0.002  # just below zone
                signals.iloc[i, signals.columns.get_loc("stop_distance")] = abs(row["Close"] - stop)

            # Breakout long: close above buy zone upper + breaks prev high
            elif (row["Close"] > row["buy_breakout_level"] and
                  not np.isnan(row["prev_high"]) and
                  row["Close"] > row["prev_high"]):
                signals.iloc[i, signals.columns.get_loc("signal")] = 1
                signals.iloc[i, signals.columns.get_loc("signal_type")] = "breakout"
                stop = row["Low"]  # stop at breakout bar's low
                signals.iloc[i, signals.columns.get_loc("stop_distance")] = abs(row["Close"] - stop)

        # --- SHORT signals (trend == -1) ---
        elif t == -1 and row.get("entry_ma20_slope", 0) < 0:
            # Pullback short: price touches sell zone + bearish candle
            if (row["High"] >= row["sell_zone_lower"] and
                    row["High"] <= row["sell_zone_upper"] and
                    row["bearish_candle"]):
                signals.iloc[i, signals.columns.get_loc("signal")] = -1
                signals.iloc[i, signals.columns.get_loc("signal_type")] = "pullback"
                stop = row["sell_zone_upper"] + row["Close"] * 0.002
                signals.iloc[i, signals.columns.get_loc("stop_distance")] = abs(stop - row["Close"])

            # Breakout short: close below sell zone lower + breaks prev low
            elif (row["Close"] < row["sell_breakout_level"] and
                  not np.isnan(row["prev_low"]) and
                  row["Close"] < row["prev_low"]):
                signals.iloc[i, signals.columns.get_loc("signal")] = -1
                signals.iloc[i, signals.columns.get_loc("signal_type")] = "breakout"
                stop = row["High"]
                signals.iloc[i, signals.columns.get_loc("stop_distance")] = abs(stop - row["Close"])

    return signals
