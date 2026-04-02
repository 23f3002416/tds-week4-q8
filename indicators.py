"""Technical indicator calculations: MAs, Bollinger Bands, candlestick patterns."""

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(window):
        return np.dot(window, weights) / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(_wma, raw=True)


def compute_ma(series: pd.Series, period: int, ma_type: str = "sma") -> pd.Series:
    if ma_type == "sma":
        return compute_sma(series, period)
    elif ma_type == "ema":
        return compute_ema(series, period)
    elif ma_type == "wma":
        return compute_wma(series, period)
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def compute_bollinger_bands(series: pd.Series, period: int = 20,
                            std_dev: float = 2.0, ma_type: str = "sma"):
    """Compute Bollinger Bands.

    Returns (middle, upper, lower) Series tuple.
    """
    middle = compute_ma(series, period, ma_type)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return middle, upper, lower


# ---------------------------------------------------------------------------
# Trend Indicators (applied to weekly/trend timeframe)
# ---------------------------------------------------------------------------

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard BB, 20 MA, 60 MA, and slope indicators to trend timeframe."""
    df = df.copy()

    # Standard Bollinger Bands on Close
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(
        df["Close"], config.TREND_BB_PERIOD, config.TREND_BB_STD, "sma"
    )

    # Moving averages
    df["ma_fast"] = compute_sma(df["Close"], config.TREND_MA_FAST)
    df["ma_slow"] = compute_sma(df["Close"], config.TREND_MA_SLOW)

    # Slope of MAs (positive = uptrend, negative = downtrend)
    lb = config.TREND_SLOPE_LOOKBACK
    df["ma_fast_slope"] = df["ma_fast"] - df["ma_fast"].shift(lb)
    df["ma_slow_slope"] = df["ma_slow"] - df["ma_slow"].shift(lb)

    # Trend signal: +1 bullish, -1 bearish, 0 neutral
    df["trend"] = 0
    df.loc[(df["ma_fast_slope"] > 0) & (df["ma_slow_slope"] > 0), "trend"] = 1
    df.loc[(df["ma_fast_slope"] < 0) & (df["ma_slow_slope"] < 0), "trend"] = -1

    return df


# ---------------------------------------------------------------------------
# Entry Indicators (applied to daily/entry timeframe)
# ---------------------------------------------------------------------------

def add_entry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom Buy/Sell zone BBands, entry MA, and candlestick patterns."""
    df = df.copy()
    period = config.ENTRY_BB_PERIOD
    std = config.ENTRY_BB_STD

    # Entry timeframe MA (for confirming entry direction)
    df["entry_ma20"] = compute_sma(df["Close"], 20)
    df["entry_ma20_slope"] = df["entry_ma20"] - df["entry_ma20"].shift(5)

    # --- Buy Zone (computed on High prices) ---
    # EMA-based BB on High
    _, ema_high_upper, ema_high_lower = compute_bollinger_bands(
        df["High"], period, std, "ema"
    )
    # WMA-based BB on High
    _, wma_high_upper, wma_high_lower = compute_bollinger_bands(
        df["High"], period, std, "wma"
    )
    # Buy zone = region between the two lower bands of both BB sets on High
    df["buy_zone_upper"] = pd.concat([ema_high_lower, wma_high_lower], axis=1).max(axis=1)
    df["buy_zone_lower"] = pd.concat([ema_high_lower, wma_high_lower], axis=1).min(axis=1)
    # Also track the upper bands of Buy zone BB (for breakout)
    df["buy_breakout_level"] = pd.concat([ema_high_upper, wma_high_upper], axis=1).max(axis=1)

    # --- Sell Zone (computed on Low prices) ---
    # EMA-based BB on Low
    _, ema_low_upper, ema_low_lower = compute_bollinger_bands(
        df["Low"], period, std, "ema"
    )
    # WMA-based BB on Low
    _, wma_low_upper, wma_low_lower = compute_bollinger_bands(
        df["Low"], period, std, "wma"
    )
    # Sell zone = region between the two upper bands of both BB sets on Low
    df["sell_zone_upper"] = pd.concat([ema_low_upper, wma_low_upper], axis=1).max(axis=1)
    df["sell_zone_lower"] = pd.concat([ema_low_upper, wma_low_upper], axis=1).min(axis=1)
    # Also track the lower bands of Sell zone BB (for breakout)
    df["sell_breakout_level"] = pd.concat([ema_low_lower, wma_low_lower], axis=1).min(axis=1)

    # Previous day's high/low (for breakout confirmation)
    df["prev_high"] = df["High"].shift(1)
    df["prev_low"] = df["Low"].shift(1)

    # --- Candlestick Patterns ---
    df["is_hammer"] = _is_hammer(df)
    df["is_inv_hammer"] = _is_inverted_hammer(df)
    df["is_doji"] = _is_doji(df)
    df["is_bullish_engulfing"] = _is_bullish_engulfing(df)
    df["is_bearish_engulfing"] = _is_bearish_engulfing(df)

    # Composite bullish/bearish candle signals
    df["bullish_candle"] = df["is_hammer"] | df["is_bullish_engulfing"] | (
        df["is_doji"] & (df["Close"] > df["Open"])
    )
    df["bearish_candle"] = df["is_inv_hammer"] | df["is_bearish_engulfing"] | (
        df["is_doji"] & (df["Close"] < df["Open"])
    )

    return df


# ---------------------------------------------------------------------------
# Candlestick Pattern Detection
# ---------------------------------------------------------------------------

def _body(df: pd.DataFrame) -> pd.Series:
    return (df["Close"] - df["Open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df["Low"]


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df[["Close", "Open"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["Close", "Open"]].min(axis=1) - df["Low"]


def _is_doji(df: pd.DataFrame) -> pd.Series:
    rng = _range(df)
    body = _body(df)
    return (rng > 0) & (body / rng < config.DOJI_BODY_RATIO)


def _is_hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish hammer: small body in upper portion, long lower shadow."""
    body = _body(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    rng = _range(df)
    return (
        (rng > 0) &
        (lower >= config.HAMMER_WICK_RATIO * body) &
        (upper < body) &
        (body / rng < config.HAMMER_BODY_POSITION)
    )


def _is_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """Bearish inverted hammer/shooting star: small body in lower portion, long upper shadow."""
    body = _body(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    rng = _range(df)
    return (
        (rng > 0) &
        (upper >= config.HAMMER_WICK_RATIO * body) &
        (lower < body) &
        (body / rng < config.HAMMER_BODY_POSITION)
    )


def _is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current green candle body fully engulfs previous red candle body."""
    curr_green = df["Close"] > df["Open"]
    prev_red = df["Close"].shift(1) < df["Open"].shift(1)
    curr_body_low = df[["Close", "Open"]].min(axis=1)
    curr_body_high = df[["Close", "Open"]].max(axis=1)
    prev_body_low = pd.concat([df["Close"].shift(1), df["Open"].shift(1)], axis=1).min(axis=1)
    prev_body_high = pd.concat([df["Close"].shift(1), df["Open"].shift(1)], axis=1).max(axis=1)
    return (
        curr_green & prev_red &
        (curr_body_low <= prev_body_low) &
        (curr_body_high >= prev_body_high)
    )


def _is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current red candle body fully engulfs previous green candle body."""
    curr_red = df["Close"] < df["Open"]
    prev_green = df["Close"].shift(1) > df["Open"].shift(1)
    curr_body_low = df[["Close", "Open"]].min(axis=1)
    curr_body_high = df[["Close", "Open"]].max(axis=1)
    prev_body_low = pd.concat([df["Close"].shift(1), df["Open"].shift(1)], axis=1).min(axis=1)
    prev_body_high = pd.concat([df["Close"].shift(1), df["Open"].shift(1)], axis=1).max(axis=1)
    return (
        curr_red & prev_green &
        (curr_body_low <= prev_body_low) &
        (curr_body_high >= prev_body_high)
    )
