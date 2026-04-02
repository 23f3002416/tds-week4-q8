"""Central configuration for the multi-timeframe Bollinger Band backtesting system."""

ASSETS = {
    "Gold": {
        "ticker": "GC=F",
        "cost_per_side": 2.50,
        "cost_type": "fixed",
        "point_value": 100,
    },
    "NASDAQ": {
        "ticker": "NQ=F",
        "cost_per_side": 2.50,
        "cost_type": "fixed",
        "point_value": 20,
    },
    "Bitcoin": {
        "ticker": "BTC-USD",
        "cost_pct": 0.001,  # 0.1% per trade
        "cost_type": "percentage",
        "point_value": 1,
    },
}

# Timeframe configuration
# Primary: Weekly trend + Daily entries (for 10-20 year backtest)
# Validation: Daily trend + Hourly entries (for ~2 year backtest)
TIMEFRAMES = {
    "primary": {"trend": "1wk", "entry": "1d"},
    "validation": {"trend": "1d", "entry": "1h"},
}

# Standard Bollinger Bands for trend timeframe
TREND_BB_PERIOD = 20
TREND_BB_STD = 2.0

# Moving averages for trend identification
TREND_MA_FAST = 20
TREND_MA_SLOW = 60
TREND_SLOPE_LOOKBACK = 5  # bars to measure slope direction

# Custom Bollinger Bands for entry timeframe
ENTRY_BB_PERIOD = 20
ENTRY_BB_STD = 2.0

# Candlestick pattern thresholds
DOJI_BODY_RATIO = 0.1        # body < 10% of total range = doji
HAMMER_WICK_RATIO = 2.0      # lower shadow >= 2x body for hammer
HAMMER_BODY_POSITION = 0.33  # body in upper/lower third of range
ENGULFING_MIN_RATIO = 1.0    # current body must fully engulf prior body

# Backtesting parameters
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE = 0.01        # risk 1% of equity per trade
MAX_POSITION_PCT = 0.20      # max 20% of equity in single position
MAX_HOLDING_BARS = 40        # max bars to hold a position (safety exit)

# Date range
DATA_START = "2006-01-01"
DATA_END = "2026-04-01"

# Validation date range (hourly data limited to ~730 days)
VALIDATION_START = "2024-06-01"
VALIDATION_END = "2026-03-30"

# Data cache directory
CACHE_DIR = "data"

# Output directory
OUTPUT_DIR = "output"
