"""Binance historical klines (candlestick) data fetcher.

Uses Binance's public bulk data download service (data.binance.vision)
to download historical 1-minute candle data for BTCUSDT going back to 2017.

Data format: Monthly ZIP files containing CSV with OHLCV data.
URL pattern: https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{YYYY}-{MM}.zip
"""

import os
import io
import time
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import config

BINANCE_DATA_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# CSV columns in Binance bulk download files
KLINE_COLUMNS = [
    "open_time", "Open", "High", "Low", "Close", "Volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]


def _download_monthly_zip(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame:
    """Download and parse a single monthly ZIP file from data.binance.vision."""
    filename = f"{symbol}-{interval}-{year}-{month:02d}"
    url = f"{BINANCE_DATA_BASE}/{symbol}/{interval}/{filename}.zip"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(4):
        try:
            with urlopen(req, timeout=60) as resp:
                zip_data = resp.read()
            break
        except (URLError, HTTPError, TimeoutError, ConnectionError) as e:
            if isinstance(e, HTTPError) and e.code == 404:
                return pd.DataFrame()  # Month not available yet
            wait = 2 ** (attempt + 1)
            print(f"    Retry {attempt+1}/4 for {filename}: {e}. Waiting {wait}s...")
            time.sleep(wait)
    else:
        print(f"    Failed to download {filename} after retries, skipping.")
        return pd.DataFrame()

    # Extract CSV from ZIP
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
    except (zipfile.BadZipFile, IndexError) as e:
        print(f"    Bad ZIP for {filename}: {e}")
        return pd.DataFrame()

    return df


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "1m",
                         start_date: str = "2017-08-17",
                         end_date: str = "2026-04-01") -> pd.DataFrame:
    """Download full historical klines from Binance data.binance.vision.

    Downloads month-by-month ZIP files and concatenates them.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    print(f"  Fetching {symbol} {interval} from {start_date} to {end_date}...")
    print(f"  Source: data.binance.vision (monthly bulk downloads)")

    all_dfs = []
    current_year = start_dt.year
    current_month = start_dt.month
    total_months = 0

    while (current_year < end_dt.year or
           (current_year == end_dt.year and current_month <= end_dt.month)):

        df = _download_monthly_zip(symbol, interval, current_year, current_month)
        if not df.empty:
            all_dfs.append(df)
            total_months += 1
            if total_months % 12 == 0:
                print(f"    Downloaded {total_months} months ({len(pd.concat(all_dfs)):,} candles so far)...")

        # Next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

        # Small delay between downloads
        time.sleep(0.05)

    if not all_dfs:
        raise ValueError(f"No data returned for {symbol} {interval}")

    # Concatenate all months
    df = pd.concat(all_dfs, ignore_index=True)

    # Convert types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    # Set datetime index
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df.index.name = "Date"

    # Keep only OHLCV
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Filter to requested date range
    df = df[start_date:end_date]

    print(f"  Downloaded {len(df):,} candles across {total_months} months")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    return df


def load_or_fetch_binance(symbol: str = "BTCUSDT", interval: str = "1m",
                          start_date: str = "2017-08-17",
                          end_date: str = "2026-04-01",
                          cache_dir: str = None) -> pd.DataFrame:
    """Load from cache or fetch from Binance."""
    if cache_dir is None:
        cache_dir = config.CACHE_DIR

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"binance_{symbol}_{interval}.csv")

    if os.path.exists(cache_file):
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 48:
            print(f"  Loading {symbol} {interval} from cache ({cache_file})...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index, format="mixed")
                print(f"  Loaded {len(df):,} candles from {df.index[0]} to {df.index[-1]}")
                return df

    df = fetch_binance_klines(symbol, interval, start_date, end_date)
    print(f"  Saving to cache: {cache_file}")
    df.to_csv(cache_file)
    return df


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe.

    Args:
        df: Source OHLCV DataFrame (e.g., 1-minute data)
        timeframe: Target timeframe string compatible with pandas resample
                   (e.g., '5min', '15min', '1h', '4h', '1D', '1W')
    """
    resampled = df.resample(timeframe).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return resampled
