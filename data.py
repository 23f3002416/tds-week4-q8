"""Data acquisition and caching using yfinance."""

import os
import time
import pandas as pd
import yfinance as yf

import config


def fetch_data(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from yfinance."""
    print(f"  Downloading {ticker} ({interval}) from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({interval})")
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Ensure standard column names
    df = df.rename(columns={"Adj Close": "Adj_Close"})
    # Drop rows with NaN in OHLC
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def load_or_fetch(ticker: str, interval: str, start: str, end: str,
                  cache_dir: str = None) -> pd.DataFrame:
    """Load from CSV cache if available and fresh, otherwise download."""
    if cache_dir is None:
        cache_dir = config.CACHE_DIR

    os.makedirs(cache_dir, exist_ok=True)
    safe_ticker = ticker.replace("=", "_").replace("-", "_")
    cache_file = os.path.join(cache_dir, f"{safe_ticker}_{interval}.csv")

    # Use cache if it exists and is less than 24 hours old
    if os.path.exists(cache_file):
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 24:
            print(f"  Loading {ticker} ({interval}) from cache...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df

    df = fetch_data(ticker, interval, start, end)
    df.to_csv(cache_file)
    return df


def resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV data to weekly bars."""
    weekly = daily_df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return weekly


def get_asset_data(asset_name: str, asset_cfg: dict, mode: str = "primary"):
    """Fetch both trend and entry timeframe data for an asset.

    Returns (trend_df, entry_df) tuple.
    """
    if mode == "primary":
        start, end = config.DATA_START, config.DATA_END
        # Always download daily, resample to weekly for trend
        entry_df = load_or_fetch(asset_cfg["ticker"], "1d", start, end)
        trend_df = resample_to_weekly(entry_df)
    elif mode == "validation":
        start, end = config.VALIDATION_START, config.VALIDATION_END
        # Daily for trend, hourly for entries
        trend_df = load_or_fetch(asset_cfg["ticker"], "1d", start, end)
        entry_df = load_or_fetch(asset_cfg["ticker"], "1h", start, end)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"  {asset_name} ({mode}): trend={len(trend_df)} bars, entry={len(entry_df)} bars")
    return trend_df, entry_df
