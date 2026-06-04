"""
Symbol Ranking Module

Fetches top cryptocurrency symbols from Binance Futures API ranked by 24-hour
trading volume. Includes caching to avoid excessive API calls.
"""

import json
import os
import time
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

# Fallback list if API fails
TOP_20_SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "AVAX", "DOGE", "DOT", "TRX",
    "LINK", "MATIC", "LTC", "BCH", "UNI",
    "ATOM", "XLM", "ETC", "FIL", "APT",
]

# Stablecoins to exclude from ranking
STABLECOINS = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDT", "USDP", "GUSD", "FRAX"}


def _fetch_json(url: str, timeout: int = 30) -> dict | list:
    """Fetch JSON data from a URL."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_perpetual_symbols(base_url: str, quote: str) -> set[str]:
    """
    Get all available perpetual contract symbols from Binance Futures.

    Args:
        base_url: Binance Futures API base URL
        quote: Quote currency (e.g., "USDT")

    Returns:
        Set of base symbols that have perpetual contracts
    """
    url = f"{base_url}/fapi/v1/exchangeInfo"
    data = _fetch_json(url)

    perpetual_symbols = set()
    for symbol_info in data.get("symbols", []):
        if (
            symbol_info.get("contractType") == "PERPETUAL"
            and symbol_info.get("quoteAsset") == quote
            and symbol_info.get("status") == "TRADING"
        ):
            perpetual_symbols.add(symbol_info.get("baseAsset"))

    return perpetual_symbols


def _get_24hr_volumes(base_url: str) -> list[dict]:
    """
    Get 24-hour ticker data for all symbols.

    Args:
        base_url: Binance Futures API base URL

    Returns:
        List of ticker data dictionaries
    """
    url = f"{base_url}/fapi/v1/ticker/24hr"
    return _fetch_json(url)


def _load_cache(cache_file: str, cache_hours: float, top_n: int) -> Optional[list[str]]:
    """
    Load symbols from cache if fresh and top_n matches.

    Args:
        cache_file: Path to cache file
        cache_hours: Maximum age of cache in hours
        top_n: Number of symbols requested (cache invalidates if different)

    Returns:
        List of symbols if cache is valid, None otherwise
    """
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        cache_time = cache_data.get("timestamp", 0)
        cache_age_hours = (time.time() - cache_time) / 3600

        # Check if top_n matches cached value
        cached_top_n = cache_data.get("top_n")
        if cached_top_n != top_n:
            print(f"Cache invalidated: top_n changed from {cached_top_n} to {top_n}")
            return None

        if cache_age_hours < cache_hours:
            symbols = cache_data.get("symbols", [])
            if symbols:
                return symbols
    except (json.JSONDecodeError, KeyError, TypeError):
        # Cache corrupted, delete it
        try:
            os.remove(cache_file)
        except OSError:
            pass

    return None


def _save_cache(cache_file: str, symbols: list[str], top_n: int) -> None:
    """
    Save symbols to cache file.

    Args:
        cache_file: Path to cache file
        symbols: List of symbols to cache
        top_n: Number of symbols requested (stored for cache invalidation)
    """
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    cache_data = {
        "timestamp": time.time(),
        "symbols": symbols,
        "top_n": top_n,
    }

    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def get_top_symbols(
    base_url: str = "https://fapi.binance.com",
    top_n: int = 20,
    quote: str = "USDT",
    cache_file: str = "data/symbol_cache.json",
    cache_hours: float = 24,
) -> list[str]:
    """
    Get top N cryptocurrency symbols by 24-hour trading volume.

    Fetches data from Binance Futures API and ranks symbols by quote volume.
    Uses caching to avoid excessive API calls.

    Args:
        base_url: Binance Futures API base URL
        top_n: Number of top symbols to return
        quote: Quote currency (e.g., "USDT")
        cache_file: Path to cache file for storing results
        cache_hours: Hours before cache expires

    Returns:
        List of base symbols (e.g., ["BTC", "ETH", "SOL", ...])
    """
    # Check cache first
    cached = _load_cache(cache_file, cache_hours, top_n)
    if cached is not None:
        print(f"Using cached symbols from {cache_file}")
        return cached[:top_n]

    try:
        print("Fetching top symbols from Binance Futures API...")

        # Get available perpetual contracts
        perpetual_symbols = _get_perpetual_symbols(base_url, quote)

        if not perpetual_symbols:
            print("Warning: No perpetual symbols found, using fallback list")
            return TOP_20_SYMBOLS[:top_n]

        # Get 24hr volume data
        ticker_data = _get_24hr_volumes(base_url)

        if not ticker_data:
            print("Warning: Empty ticker data, using fallback list")
            return TOP_20_SYMBOLS[:top_n]

        # Filter and sort by volume
        symbol_volumes = []
        for ticker in ticker_data:
            symbol = ticker.get("symbol", "")

            # Extract base asset from symbol (e.g., "BTCUSDT" -> "BTC")
            if symbol.endswith(quote):
                base_asset = symbol[:-len(quote)]
            else:
                continue

            # Skip if not a perpetual contract
            if base_asset not in perpetual_symbols:
                continue

            # Skip stablecoins
            if base_asset in STABLECOINS:
                continue

            try:
                volume = float(ticker.get("quoteVolume", 0))
                symbol_volumes.append((base_asset, volume))
            except (ValueError, TypeError):
                continue

        # Sort by volume descending
        symbol_volumes.sort(key=lambda x: x[1], reverse=True)

        # Extract top N symbols
        top_symbols = [sym for sym, _ in symbol_volumes[:top_n]]

        if not top_symbols:
            print("Warning: No valid symbols after filtering, using fallback list")
            return TOP_20_SYMBOLS[:top_n]

        # Save to cache
        _save_cache(cache_file, top_symbols, top_n)
        print(f"Fetched top {len(top_symbols)} symbols by 24h volume")

        return top_symbols

    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to fetch from API ({e}), checking cache...")

        # Try to use expired cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                symbols = cache_data.get("symbols", [])
                if symbols:
                    print(f"Using expired cache ({len(symbols)} symbols)")
                    return symbols[:top_n]
            except (json.JSONDecodeError, KeyError):
                pass

        print("Using fallback hardcoded list")
        return TOP_20_SYMBOLS[:top_n]

    except Exception as e:
        print(f"Warning: Unexpected error ({e}), using fallback list")
        return TOP_20_SYMBOLS[:top_n]


if __name__ == "__main__":
    # Test the module
    print("Testing symbol ranking module...")
    print()

    symbols = get_top_symbols(top_n=20)

    print()
    print("Top 20 symbols by 24h volume:")
    for i, sym in enumerate(symbols, 1):
        print(f"  {i:2d}. {sym}")
