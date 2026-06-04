"""
Unit Tests for Multi-Pair Scanner

Tests for the PairScanner class and helper functions in scan_pairs.py.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from io import StringIO

from test_helpers import create_cointegrated_prices, get_default_mock_config
from scan_pairs import (
    PairScanner,
    format_results_table,
    print_scan_results,
    TOP_20_SYMBOLS,
)


class TestGeneratePairs(unittest.TestCase):
    """Tests for pair generation logic."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the config loading to avoid needing actual config file
        self.config_patcher = patch('scan_pairs.load_config')
        self.mock_load_config = self.config_patcher.start()
        self.mock_load_config.return_value = get_default_mock_config()

    def tearDown(self):
        """Clean up patches."""
        self.config_patcher.stop()

    def test_generate_pairs_basic(self):
        """Verify C(n,2) combinations are generated correctly."""
        scanner = PairScanner()
        symbols = ["BTC", "ETH", "SOL", "AVAX"]
        pairs = scanner.generate_pairs(symbols)

        # C(4,2) = 6 pairs
        self.assertEqual(len(pairs), 6)

        # Verify all expected pairs are present
        expected_pairs = [
            ("BTC", "ETH"),
            ("BTC", "SOL"),
            ("BTC", "AVAX"),
            ("ETH", "SOL"),
            ("ETH", "AVAX"),
            ("SOL", "AVAX"),
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_generate_pairs_empty(self):
        """Handle empty symbol list."""
        scanner = PairScanner()
        pairs = scanner.generate_pairs([])

        self.assertEqual(len(pairs), 0)
        self.assertEqual(pairs, [])

    def test_generate_pairs_single(self):
        """Handle single symbol (should return empty)."""
        scanner = PairScanner()
        pairs = scanner.generate_pairs(["BTC"])

        self.assertEqual(len(pairs), 0)
        self.assertEqual(pairs, [])

    def test_generate_pairs_order(self):
        """Verify pairs maintain order (A,B not B,A)."""
        scanner = PairScanner()
        symbols = ["ETH", "BTC", "SOL"]
        pairs = scanner.generate_pairs(symbols)

        # First element should always come before second in original list
        for pair in pairs:
            idx0 = symbols.index(pair[0])
            idx1 = symbols.index(pair[1])
            self.assertLess(idx0, idx1)

        # Verify no reversed duplicates
        for pair in pairs:
            reversed_pair = (pair[1], pair[0])
            self.assertNotIn(reversed_pair, pairs)


class TestRunSinglePair(unittest.TestCase):
    """Tests for single pair backtest execution."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock config loading
        self.config_patcher = patch('scan_pairs.load_config')
        self.mock_load_config = self.config_patcher.start()
        self.mock_load_config.return_value = get_default_mock_config()

    def tearDown(self):
        """Clean up patches."""
        self.config_patcher.stop()

    @patch.object(PairScanner, '__init__', lambda self, config_path=None: None)
    def test_run_single_pair_success(self):
        """Verify result dict structure with real data."""
        from pair_trading import StrategyConfig

        scanner = PairScanner.__new__(PairScanner)
        scanner.config = {
            "feather_dir": "data/feather",
            "interval": "1d",
            "quote": "USDT",
            "window": 21,
        }
        scanner.strategy_config = StrategyConfig(
            window=21,
            kpss_threshold=0.463,
            entry_threshold=0.02,
            stop_loss_threshold=-0.05,
            transaction_fee=0.0001,
        )
        scanner.data_loader = Mock()

        # Mock data loader to return synthetic data
        prices0, prices1 = create_cointegrated_prices(n=100, seed=42)
        scanner.data_loader.load_pair.return_value = (prices0, prices1)

        result = scanner.run_single_pair("BTC", "ETH")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["pair"], "BTC/ETH")
        self.assertEqual(result["symbol0"], "BTC")
        self.assertEqual(result["symbol1"], "ETH")

    @patch.object(PairScanner, '__init__', lambda self, config_path=None: None)
    def test_run_single_pair_file_not_found(self):
        """Returns None when data missing."""
        from pair_trading import StrategyConfig

        scanner = PairScanner.__new__(PairScanner)
        scanner.config = {"window": 21}
        scanner.strategy_config = StrategyConfig(window=21)
        scanner.data_loader = Mock()

        # Mock data loader to raise FileNotFoundError
        scanner.data_loader.load_pair.side_effect = FileNotFoundError("Data not found")

        result = scanner.run_single_pair("INVALID1", "INVALID2")

        self.assertIsNone(result)

    @patch.object(PairScanner, '__init__', lambda self, config_path=None: None)
    def test_run_single_pair_insufficient_data(self):
        """Returns None when not enough rows."""
        from pair_trading import StrategyConfig

        scanner = PairScanner.__new__(PairScanner)
        scanner.config = {"window": 21}
        scanner.strategy_config = StrategyConfig(window=21)
        scanner.data_loader = Mock()

        # Create insufficient data (less than window + 10)
        dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
        prices0 = pd.Series(np.random.randn(20) + 100, index=dates)
        prices1 = pd.Series(np.random.randn(20) + 100, index=dates)

        scanner.data_loader.load_pair.return_value = (prices0, prices1)

        result = scanner.run_single_pair("BTC", "ETH")

        self.assertIsNone(result)

    @patch.object(PairScanner, '__init__', lambda self, config_path=None: None)
    def test_run_single_pair_result_keys(self):
        """Verify all expected keys in result dict."""
        from pair_trading import StrategyConfig

        scanner = PairScanner.__new__(PairScanner)
        scanner.config = {"window": 21}
        scanner.strategy_config = StrategyConfig(
            window=21,
            kpss_threshold=0.463,
            entry_threshold=0.02,
            stop_loss_threshold=-0.05,
            transaction_fee=0.0001,
        )
        scanner.data_loader = Mock()

        prices0, prices1 = create_cointegrated_prices(n=100, seed=42)
        scanner.data_loader.load_pair.return_value = (prices0, prices1)

        result = scanner.run_single_pair("BTC", "ETH")

        expected_keys = [
            "pair",
            "symbol0",
            "symbol1",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades",
            "calmar_ratio",
            "data_points",
        ]

        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")


class TestSaveResults(unittest.TestCase):
    """Tests for result file saving."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample results DataFrame
        self.sample_df = pd.DataFrame({
            "pair": ["BTC/ETH", "BTC/SOL"],
            "symbol0": ["BTC", "BTC"],
            "symbol1": ["ETH", "SOL"],
            "total_return": [0.15, 0.22],
            "annualized_return": [0.35, 0.48],
            "annualized_volatility": [0.20, 0.25],
            "sharpe_ratio": [1.75, 1.92],
            "sortino_ratio": [2.10, 2.35],
            "max_drawdown": [-0.08, -0.12],
            "win_rate": [0.55, 0.58],
            "profit_factor": [1.8, 2.1],
            "total_trades": [45, 52],
            "calmar_ratio": [4.38, 4.0],
            "data_points": [365, 365],
        })

        # Mock config loading
        self.config_patcher = patch('scan_pairs.load_config')
        self.mock_load_config = self.config_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.config_patcher.stop()

    def test_save_results_creates_files(self):
        """CSV and feather files created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_load_config.return_value = {
                "feather_dir": "data/feather",
                "interval": "1d",
                "quote": "USDT",
                "output_dir": tmpdir,
                "window": 21,
            }

            scanner = PairScanner()
            csv_path, feather_path = scanner.save_results(self.sample_df)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(feather_path))
            self.assertTrue(csv_path.endswith(".csv"))
            self.assertTrue(feather_path.endswith(".feather"))

            # Verify content can be read back
            loaded_csv = pd.read_csv(csv_path)
            loaded_feather = pd.read_feather(feather_path)

            self.assertEqual(len(loaded_csv), 2)
            self.assertEqual(len(loaded_feather), 2)

    def test_save_results_custom_path(self):
        """Custom output path works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.mock_load_config.return_value = {
                "feather_dir": "data/feather",
                "interval": "1d",
                "quote": "USDT",
                "output_dir": tmpdir,
                "window": 21,
            }

            scanner = PairScanner()
            custom_path = os.path.join(tmpdir, "custom_results.csv")
            csv_path, feather_path = scanner.save_results(self.sample_df, custom_path)

            self.assertIn("custom_results", csv_path)
            self.assertIn("custom_results", feather_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(feather_path))

    def test_save_results_creates_directory(self):
        """Creates output dir if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "output")

            self.mock_load_config.return_value = {
                "feather_dir": "data/feather",
                "interval": "1d",
                "quote": "USDT",
                "output_dir": nested_dir,
                "window": 21,
            }

            scanner = PairScanner()
            csv_path, feather_path = scanner.save_results(self.sample_df)

            self.assertTrue(os.path.exists(nested_dir))
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(feather_path))


class TestFormatResultsTable(unittest.TestCase):
    """Tests for table formatting."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            "pair": ["BTC/ETH", "BTC/SOL", "ETH/SOL", "BTC/AVAX"],
            "total_return": [0.15, 0.22, 0.10, 0.30],
            "annualized_return": [0.35, 0.48, 0.25, 0.65],
            "annualized_volatility": [0.20, 0.25, 0.15, 0.30],
            "sharpe_ratio": [1.75, 1.92, 1.67, 2.17],
            "sortino_ratio": [2.10, 2.35, 1.90, 2.60],
            "max_drawdown": [-0.08, -0.12, -0.05, -0.15],
            "total_trades": [45, 52, 38, 60],
        })

    def test_format_empty_dataframe(self):
        """Returns 'No results' message."""
        empty_df = pd.DataFrame()
        result = format_results_table(empty_df, "sharpe_ratio")

        self.assertEqual(result, "No results to display.")

    def test_format_sorts_by_sharpe(self):
        """Correct sorting by sharpe_ratio."""
        result = format_results_table(self.sample_df, "sharpe_ratio", top_n=4)

        # Should be sorted descending by sharpe_ratio
        lines = result.split("\n")
        # First data line (after header and separator) should be BTC/AVAX (highest sharpe: 2.17)
        self.assertIn("BTC/AVAX", lines[2])

    def test_format_sorts_by_total_return(self):
        """Correct sorting by total_return."""
        result = format_results_table(self.sample_df, "total_return", top_n=4)

        lines = result.split("\n")
        # First data line should be BTC/AVAX (highest total_return: 0.30)
        self.assertIn("BTC/AVAX", lines[2])

    def test_format_top_n_limit(self):
        """Respects top_n parameter."""
        result = format_results_table(self.sample_df, "sharpe_ratio", top_n=2)

        lines = result.split("\n")
        # Should have header (1) + separator (1) + 2 data rows = 4 lines
        self.assertEqual(len(lines), 4)

    def test_format_output_structure(self):
        """Header and row format correct."""
        result = format_results_table(self.sample_df, "sharpe_ratio", top_n=1)

        lines = result.split("\n")

        # Check header contains expected columns
        header = lines[0]
        self.assertIn("Rank", header)
        self.assertIn("Pair", header)
        self.assertIn("Total Ret", header)
        self.assertIn("Sharpe", header)
        self.assertIn("Max DD", header)
        self.assertIn("Trades", header)

        # Check separator line
        self.assertTrue(lines[1].startswith("-"))


class TestPrintScanResults(unittest.TestCase):
    """Tests for console output formatting."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            "pair": ["BTC/ETH", "BTC/SOL", "ETH/SOL"],
            "total_return": [0.15, 0.22, 0.10],
            "annualized_return": [0.35, 0.48, 0.25],
            "annualized_volatility": [0.20, 0.25, 0.15],
            "sharpe_ratio": [1.75, 1.92, np.nan],  # One NaN for testing
            "sortino_ratio": [2.10, 2.35, 1.90],
            "max_drawdown": [-0.08, -0.12, -0.05],
            "total_trades": [45, 52, 38],
        })

    def test_print_empty_dataframe(self):
        """Handles empty results gracefully."""
        empty_df = pd.DataFrame()

        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_scan_results(empty_df)
            output = mock_stdout.getvalue()

        self.assertIn("No results to display", output)

    def test_print_counts_successful(self):
        """Correctly counts valid Sharpe ratios."""
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_scan_results(self.sample_df)
            output = mock_stdout.getvalue()

        # Should count 2 successful (non-NaN sharpe_ratio)
        self.assertIn("Successful: 2", output)
        self.assertIn("Scanned 3 pairs", output)


class TestTop20Symbols(unittest.TestCase):
    """Tests for the TOP_20_SYMBOLS constant."""

    def test_top_20_symbols_count(self):
        """Verify exactly 20 symbols."""
        self.assertEqual(len(TOP_20_SYMBOLS), 20)

    def test_top_20_symbols_unique(self):
        """Verify all symbols are unique."""
        self.assertEqual(len(TOP_20_SYMBOLS), len(set(TOP_20_SYMBOLS)))

    def test_top_20_symbols_format(self):
        """Verify symbols are uppercase strings."""
        for symbol in TOP_20_SYMBOLS:
            self.assertIsInstance(symbol, str)
            self.assertEqual(symbol, symbol.upper())


if __name__ == '__main__':
    unittest.main(verbosity=2)
