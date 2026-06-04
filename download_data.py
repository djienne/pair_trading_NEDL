import argparse
import datetime
import json
import os
import time
from typing import Dict, Optional

import pandas as pd
import requests

from utils import load_config


def load_download_cache(cache_path: str) -> dict:
    """Load the download timestamp cache."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_download_cache(cache_path: str, cache: dict) -> None:
    """Save the download timestamp cache."""
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def is_within_grace_period(cache_key: str, cache: dict, grace_hours: float) -> bool:
    """Check if symbol was downloaded within the grace period."""
    if grace_hours <= 0 or cache_key not in cache:
        return False
    last_download = cache[cache_key].get("last_download", 0)
    elapsed_hours = (time.time() - last_download) / 3600
    return elapsed_hours < grace_hours


def get_download_cache_path(config: dict, feather_dir: str) -> str:
    """Get the download cache path from config or derive a stable default."""
    if config.get("download_cache_file"):
        return config["download_cache_file"]

    parent_dir = os.path.dirname(feather_dir)
    if not parent_dir:
        parent_dir = "."
    return os.path.join(parent_dir, "download_cache.json")


def feather_file_is_usable(path: str) -> bool:
    """Return True when a feather file exists and contains at least one row."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        return not pd.read_feather(path).empty
    except Exception:
        return False


COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base",
    "taker_buy_quote",
]

INT_COLUMNS = ["open_time", "close_time", "num_trades"]


def interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60 * 1000
    if unit == "h":
        return value * 60 * 60 * 1000
    if unit == "d":
        return value * 24 * 60 * 60 * 1000
    if unit == "w":
        return value * 7 * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_klines(
    base_url: str,
    symbol: str,
    interval: str,
    start_time: Optional[int],
    end_time: Optional[int],
    limit: int,
) -> list:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    response = requests.get(
        f"{base_url}/fapi/v1/klines", params=params, timeout=30
    )
    response.raise_for_status()
    return response.json()


def klines_to_frame(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=COLUMNS)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    df["open_time_dt"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    )
    df.drop(columns=["ignore"], inplace=True)
    return df


def update_symbol_data(
    symbol: str,
    interval: str,
    out_path: str,
    base_url: str,
    limit: int,
    start_date: Optional[str] = None,
    exclude_incomplete_candles: bool = True,
) -> pd.DataFrame:
    interval_ms = interval_to_millis(interval)
    now_ms = int(time.time() * 1000)
    if exclude_incomplete_candles:
        current_open = (now_ms // interval_ms) * interval_ms
        end_time = current_open - 1
    else:
        end_time = now_ms

    existing: Optional[pd.DataFrame] = None
    if os.path.exists(out_path):
        existing = pd.read_feather(out_path)

    # Calculate start_time from start_date if provided
    config_start_time = None
    if start_date:
        config_start_time = int(pd.Timestamp(start_date).timestamp() * 1000)

    if existing is None or existing.empty:
        start_time = config_start_time if config_start_time else 0
    else:
        last_open = int(existing["open_time"].max())
        start_time = last_open + interval_ms
        # If config start_date is earlier than existing data, we may want to backfill
        if config_start_time and config_start_time < int(existing["open_time"].min()):
            # Need to fetch older data too
            start_time = config_start_time

    if start_time >= end_time:
        print(f"{symbol} {interval}: already up to date.")
        return existing

    all_rows = []
    current_start = start_time
    while True:
        data = fetch_klines(
            base_url=base_url,
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_time,
            limit=limit,
        )
        if not data:
            break

        all_rows.extend(data)
        last_open = data[-1][0]
        next_start = last_open + interval_ms
        if next_start >= end_time or len(data) < limit:
            break

        current_start = next_start
        time.sleep(0.2)

    if not all_rows:
        print(f"{symbol} {interval}: no new data.")
        return existing

    new_df = klines_to_frame(all_rows)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time"]).sort_values(
            "open_time"
        )
    else:
        combined = new_df.sort_values("open_time")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_feather(out_path)
    print(f"{symbol} {interval}: saved {len(combined)} rows to {out_path}")
    return combined


def download_symbols(
    config: dict,
    symbols: list[str],
    full_symbol_names: bool = False,
    force: bool = False,
) -> list[str]:
    """
    Download data for a list of symbols using config settings.

    Args:
        config: Configuration dictionary with interval, quote, base_url, etc.
        symbols: List of symbol names (e.g., ["BTC", "ETH"] or ["BTCUSDT", "ETHUSDT"])
        full_symbol_names: If True, symbols are full names (e.g., BTCUSDT).
                          If False, quote currency is appended.
        force: If True, bypass grace period and force download.

    Returns:
        List of symbols that were successfully downloaded
    """
    interval = config.get("interval", "1d")
    quote = config.get("quote", "USDT")
    base_url = config.get("binance_base_url", "https://fapi.binance.com")
    limit = int(config.get("max_klines_per_request", 1500))
    start_date = config.get("start_date")
    feather_dir = config.get("feather_dir") or config.get("data_dir", "data/feather")
    grace_hours = float(config.get("download_grace_hours", 0))
    exclude_incomplete_candles = config.get("exclude_incomplete_candles", True)

    os.makedirs(feather_dir, exist_ok=True)

    # Load download cache
    cache_path = get_download_cache_path(config, feather_dir)
    cache = load_download_cache(cache_path)

    successful = []
    for symbol in sorted(set(symbols)):
        full_symbol = symbol if full_symbol_names else f"{symbol}{quote}"
        cache_key = f"{full_symbol}_{interval}"
        out_path = os.path.join(feather_dir, f"{cache_key}.feather")

        # Check grace period unless force is True
        if not force and is_within_grace_period(cache_key, cache, grace_hours):
            if feather_file_is_usable(out_path):
                print(f"Skipping {full_symbol} - downloaded within grace period")
                successful.append(symbol)
                continue
            print(f"{full_symbol}: cache entry is fresh but data file is missing or empty; downloading.")

        try:
            updated = update_symbol_data(
                full_symbol,
                interval,
                out_path,
                base_url,
                limit,
                start_date,
                exclude_incomplete_candles=exclude_incomplete_candles,
            )
            if updated is None or updated.empty or not feather_file_is_usable(out_path):
                print(f"  Warning: No usable data saved for {full_symbol}")
                continue

            successful.append(symbol)

            # Update cache with current timestamp
            cache[cache_key] = {
                "last_download": time.time(),
                "last_download_dt": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as e:
            print(f"  Warning: Failed to download {full_symbol}: {e}")

    # Save updated cache
    save_download_cache(cache_path, cache)

    return successful


def main(
    config_path: str = "config.json",
    symbols_override: Optional[list] = None,
    force: bool = False,
) -> None:
    config = load_config(config_path)

    if symbols_override:
        # CLI override: assume full symbol names (e.g., BTCUSDT)
        symbols = list(symbols_override)
        full_symbol_names = True
    else:
        # Config: simple list like ["BTC", "ETH"] -> append quote
        quote = config.get("quote", "USDT")
        symbols = [f"{s}{quote}" for s in config.get("symbols", [])]
        full_symbol_names = True

    if not symbols:
        raise ValueError("No symbols provided. Add 'symbols' list to config or use --symbols flag.")

    download_symbols(config, symbols, full_symbol_names=full_symbol_names, force=force)
    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and cache Binance futures klines."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional list of symbols to download (e.g., BTCUSDT ETHUSDT).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download, bypassing the grace period check.",
    )
    args = parser.parse_args()

    main(args.config, args.symbols, force=args.force)
