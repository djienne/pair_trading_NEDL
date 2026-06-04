"""
Shared Test Utilities for NEDL Pair Trading Tests

Common helper functions and fixtures to avoid code duplication
across test_strategy.py and test_scan_pairs.py.
"""

import numpy as np
import pandas as pd


def create_cointegrated_prices(n: int = 100, seed: int = 42) -> tuple:
    """
    Create synthetic cointegrated price data for testing.

    Generates two price series where Y = a + b*X + stationary noise,
    following the NEDL cointegration model (stock2 = a + b*stock1).

    Args:
        n: Number of data points to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prices0, prices1) as pd.Series with datetime index
    """
    np.random.seed(seed)
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

    # X follows random walk
    x = np.cumsum(np.random.randn(n)) + 100
    prices0 = pd.Series(x, index=dates)

    # Y = a + b*X + stationary noise (cointegrated)
    noise = np.random.randn(n) * 0.5
    y = 10 + 1.5 * x + noise
    prices1 = pd.Series(y, index=dates)

    return prices0, prices1


def get_default_mock_config() -> dict:
    """
    Return standard mock configuration dictionary for testing.

    Uses NEDL strategy default parameters:
    - window: 21 (rolling window for cointegration)
    - kpss_threshold: 0.463 (95% critical value)
    - entry_threshold: 0.02 (minimum divergence for entry)
    - stop_loss_threshold: -0.05 (stop-loss threshold)
    - transaction_fee: 0.0001 (fee per transaction)

    Returns:
        Dictionary with standard test configuration
    """
    return {
        "feather_dir": "data/feather",
        "interval": "1d",
        "quote": "USDT",
        "output_dir": "output",
        "window": 21,
        "kpss_threshold": 0.463,
        "entry_threshold": 0.02,
        "stop_loss_threshold": -0.05,
        "transaction_fee": 0.0001,
    }
