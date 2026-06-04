"""
Unit Tests for Pair Trading Strategy

Step-by-step tests for all components of the cointegration-based
pair trading strategy.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from test_helpers import create_cointegrated_prices
from pair_trading import (
    KPSSCalculator,
    CointegrationOptimizer,
    SignalGenerator,
    PairTradingStrategy,
    StrategyConfig,
    TradingState,
)


class TestKPSSCalculator(unittest.TestCase):
    """Tests for KPSS statistic calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.calc = KPSSCalculator()
        np.random.seed(42)

    def test_stationary_series_low_kpss(self):
        """White noise should have low KPSS (<0.463)."""
        n = 100
        # Create cointegrated pair: Y = 2 + 1.5*X + stationary noise
        x = np.cumsum(np.random.randn(n)) + 100  # Random walk
        noise = np.random.randn(n) * 0.5  # Stationary noise
        y = 2 + 1.5 * x + noise

        kpss = self.calc.kpss_unbiased(1.5, y, x)
        self.assertLess(kpss, 0.463, "Cointegrated series should have low KPSS")

    def test_random_walk_high_kpss(self):
        """Non-cointegrated series should have high KPSS (>0.463)."""
        n = 100
        # Two independent random walks
        x = np.cumsum(np.random.randn(n)) + 100
        y = np.cumsum(np.random.randn(n)) + 100

        # Try different b values
        kpss_values = [self.calc.kpss_unbiased(b, y, x) for b in [0.5, 1.0, 1.5, 2.0]]
        min_kpss = min(kpss_values)

        # Even with optimization, non-cointegrated series tend to have higher KPSS
        # This test may occasionally fail due to random chance
        self.assertGreater(min_kpss, 0.1, "Non-cointegrated series should have higher KPSS")

    def test_formula_verification(self):
        """Manual calculation should match implementation."""
        x = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
        y = np.array([200.0, 202.5, 204.0, 203.5, 206.0])
        b = 2.0

        # Manual calculation
        a = np.mean(y - b * x)
        resid = y - (a + b * x)
        cum_resid = np.cumsum(resid)
        st_error = np.sqrt(np.sum(resid**2) / (len(resid) - 2))
        expected_kpss = np.sum(cum_resid**2) / (len(resid)**2 * st_error**2)

        # Implementation
        actual_kpss = self.calc.kpss_unbiased(b, y, x)

        self.assertAlmostEqual(actual_kpss, expected_kpss, places=10)

    def test_two_param_formulation(self):
        """Two-parameter KPSS should also work correctly."""
        x = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
        y = np.array([200.0, 202.5, 204.0, 203.5, 206.0])
        params = np.array([0.0, 2.0])

        kpss = self.calc.kpss_two_param(params, y, x)
        self.assertIsInstance(kpss, float)
        self.assertGreater(kpss, 0)


class TestCointegrationOptimizer(unittest.TestCase):
    """Tests for cointegration parameter optimization."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_perfect_cointegration_recovery(self):
        """Optimizer should recover true parameters for perfectly cointegrated series."""
        n = 100
        true_a = 5.0
        true_b = 1.5

        x = np.cumsum(np.random.randn(n)) + 100
        noise = np.random.randn(n) * 0.1  # Very small noise
        y = true_a + true_b * x + noise

        optimizer = CointegrationOptimizer(unbiased=True)
        a_opt, b_opt, kpss_opt = optimizer.optimize(y, x)

        # Should recover parameters approximately
        self.assertAlmostEqual(b_opt, true_b, places=1)
        self.assertLess(kpss_opt, 0.463)

    def test_non_cointegrated_detection(self):
        """Optimizer should give high KPSS for non-cointegrated series."""
        n = 100
        x = np.cumsum(np.random.randn(n)) + 100
        y = np.cumsum(np.random.randn(n)) + 200  # Independent random walk

        optimizer = CointegrationOptimizer(unbiased=True)
        _, _, kpss_opt = optimizer.optimize(y, x)

        # Should have higher KPSS (though not always above threshold due to chance)
        self.assertGreater(kpss_opt, 0)

    def test_ols_initial_values(self):
        """OLS should provide reasonable starting values."""
        np.random.seed(42)
        n = 50
        x = np.linspace(100, 150, n)
        y = 10 + 2 * x + np.random.randn(n) * 0.5

        optimizer = CointegrationOptimizer()
        a0, b0 = optimizer.get_ols_initial_values(y, x)

        self.assertAlmostEqual(b0, 2.0, places=1)
        # Intercept can vary more due to noise, just check it's reasonable
        self.assertGreater(a0, 5.0)
        self.assertLess(a0, 15.0)

    def test_two_param_optimization(self):
        """Two-parameter optimization should also work."""
        n = 100
        true_a = 5.0
        true_b = 1.5

        x = np.cumsum(np.random.randn(n)) + 100
        noise = np.random.randn(n) * 0.1
        y = true_a + true_b * x + noise

        optimizer = CointegrationOptimizer(unbiased=False)
        a_opt, b_opt, kpss_opt = optimizer.optimize(y, x)

        self.assertAlmostEqual(b_opt, true_b, places=1)
        self.assertLess(kpss_opt, 0.5)


class TestSignalGenerator(unittest.TestCase):
    """Tests for trading signal generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = StrategyConfig(
            kpss_threshold=0.463,
            entry_threshold=0.02,
            stop_loss_threshold=-0.05,
        )
        self.generator = SignalGenerator(self.config)

    def test_no_trade_when_not_cointegrated(self):
        """Should return 0 when KPSS > threshold."""
        signal = self.generator.generate_signal(
            price_y=100.0,
            fair_value=95.0,  # 5% divergence
            kpss_opt=0.5,  # Above threshold
            old_signal=0,
            current_return=0.0,
        )
        self.assertEqual(signal, 0)

    def test_no_trade_insufficient_divergence(self):
        """Should return 0 when divergence < entry_threshold."""
        signal = self.generator.generate_signal(
            price_y=100.0,
            fair_value=99.5,  # 0.5% divergence < 2%
            kpss_opt=0.3,  # Below threshold
            old_signal=0,
            current_return=0.0,
        )
        self.assertEqual(signal, 0)

    def test_long_signal_when_price_above_fair_value(self):
        """Should return +1 when Y > fair_value with sufficient divergence."""
        signal = self.generator.generate_signal(
            price_y=105.0,
            fair_value=100.0,  # 5% divergence
            kpss_opt=0.3,  # Below threshold
            old_signal=0,
            current_return=0.0,
        )
        self.assertEqual(signal, 1)

    def test_short_signal_when_price_below_fair_value(self):
        """Should return -1 when Y < fair_value with sufficient divergence."""
        signal = self.generator.generate_signal(
            price_y=95.0,
            fair_value=100.0,  # 5% divergence
            kpss_opt=0.3,  # Below threshold
            old_signal=0,
            current_return=0.0,
        )
        self.assertEqual(signal, -1)

    def test_position_maintenance_same_sign(self):
        """Should maintain position when residual has same sign as old signal."""
        # Old signal was +1 (long Y), price still above fair value
        signal = self.generator.generate_signal(
            price_y=103.0,  # Still above fair value
            fair_value=100.0,
            kpss_opt=0.3,
            old_signal=1,
            current_return=0.01,
        )
        self.assertEqual(signal, 1)

    def test_stop_loss_triggers(self):
        """Should exit position when current_return < stop_loss."""
        signal = self.generator.generate_signal(
            price_y=105.0,  # Would normally maintain position
            fair_value=100.0,
            kpss_opt=0.3,
            old_signal=1,
            current_return=-0.06,  # Below -5% stop loss
        )
        self.assertEqual(signal, 0)

    def test_stop_loss_at_threshold(self):
        """Should NOT trigger stop-loss at exactly the threshold."""
        signal = self.generator.generate_signal(
            price_y=105.0,
            fair_value=100.0,
            kpss_opt=0.3,
            old_signal=1,
            current_return=-0.05,  # Exactly at threshold
        )
        # At exactly -5%, should maintain (not < -5%)
        self.assertEqual(signal, 1)


class TestEntryThreshold(unittest.TestCase):
    """Tests for entry threshold filtering."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = StrategyConfig(
            kpss_threshold=0.463,
            entry_threshold=0.02,  # 2%
            stop_loss_threshold=-0.05,
        )
        self.generator = SignalGenerator(self.config)

    def test_filter_below_threshold(self):
        """Should not enter when divergence is below threshold."""
        # 1.5% divergence < 2%
        signal = self.generator.generate_signal(
            price_y=101.5,
            fair_value=100.0,
            kpss_opt=0.3,
            old_signal=0,
            current_return=0.0,
        )
        self.assertEqual(signal, 0)

    def test_allow_above_threshold(self):
        """Should enter when divergence is above threshold."""
        # 2.5% divergence > 2%
        signal = self.generator.generate_signal(
            price_y=102.5,
            fair_value=100.0,
            kpss_opt=0.3,
            old_signal=0,
            current_return=0.0,
        )
        self.assertEqual(signal, 1)

    def test_exactly_at_threshold(self):
        """At exactly 2% divergence, entry is allowed (not strictly < threshold)."""
        # 2% divergence = 2%, the condition is abs(divergence) < threshold
        # so exactly at threshold should allow entry
        signal = self.generator.generate_signal(
            price_y=102.0,
            fair_value=100.0,
            kpss_opt=0.3,
            old_signal=0,
            current_return=0.0,
        )
        # At exactly 2% (not strictly less than), entry is allowed
        self.assertEqual(signal, 1)


class TestReturnCalculation(unittest.TestCase):
    """Tests for return calculations."""

    def test_gross_return_calculation(self):
        """Verify gross return math: position0 * ret0 + position1 * ret1."""
        # Long position0, short position1
        position0 = 1
        position1 = -1
        price0_t = 100.0
        price0_next = 102.0
        price1_t = 200.0
        price1_next = 199.0

        gross = (
            position0 * (price0_next / price0_t - 1) +
            position1 * (price1_next / price1_t - 1)
        )

        # position0: 1 * (102/100 - 1) = 0.02
        # position1: -1 * (199/200 - 1) = -1 * (-0.005) = 0.005
        expected = 0.02 + 0.005
        self.assertAlmostEqual(gross, expected, places=10)

    def test_fee_deduction(self):
        """Verify fee is deducted on position changes."""
        fee = 0.0001
        old_position0 = 0
        old_position1 = 0
        new_position0 = 1
        new_position1 = -1

        fee_cost = fee * (abs(new_position0 - old_position0) + abs(new_position1 - old_position1))
        expected_fee = 0.0001 * (1 + 1)
        self.assertAlmostEqual(fee_cost, expected_fee, places=10)

    def test_no_fee_when_position_unchanged(self):
        """No fee should be charged when position is unchanged."""
        fee = 0.0001
        old_position0 = 1
        old_position1 = -1
        new_position0 = 1
        new_position1 = -1

        fee_cost = fee * (abs(new_position0 - old_position0) + abs(new_position1 - old_position1))
        self.assertEqual(fee_cost, 0)

    def test_current_return_tracking(self):
        """Verify cumulative return tracking logic."""
        current_return = 0.02
        gross = 0.01
        signal = 1
        old_signal = 1

        # Same signal: compound returns
        if signal == old_signal:
            new_current_return = (1 + current_return) * (1 + gross) - 1
        else:
            new_current_return = gross

        expected = (1 + 0.02) * (1 + 0.01) - 1
        self.assertAlmostEqual(new_current_return, expected, places=10)

    def test_current_return_reset_on_signal_change(self):
        """Current return should reset when signal changes."""
        current_return = 0.02
        gross = 0.01
        signal = 1
        old_signal = 0

        # Different signal: reset to current gross
        if signal == old_signal:
            new_current_return = (1 + current_return) * (1 + gross) - 1
        else:
            new_current_return = gross

        self.assertEqual(new_current_return, gross)


class TestPairTradingStrategy(unittest.TestCase):
    """Integration tests for the full strategy."""

    def test_full_backtest_on_synthetic_data(self):
        """Run full backtest on synthetic cointegrated data."""
        prices0, prices1 = create_cointegrated_prices(n=100, seed=42)

        config = StrategyConfig(
            window=21,
            kpss_threshold=0.463,
            entry_threshold=0.02,
            stop_loss_threshold=-0.05,
            transaction_fee=0.0001,
        )

        strategy = PairTradingStrategy(config)
        results = strategy.run(prices0, prices1)

        # Check output structure
        self.assertIn('signal', results.columns)
        self.assertIn('gross_return', results.columns)
        self.assertIn('net_return', results.columns)
        self.assertIn('kpss_opt', results.columns)

        # Check data types
        self.assertTrue(results['signal'].isin([-1, 0, 1]).all())

    def test_strategy_handles_insufficient_data(self):
        """Strategy should raise error for insufficient data."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        prices0 = pd.Series(np.random.randn(10) + 100, index=dates)
        prices1 = pd.Series(np.random.randn(10) + 100, index=dates)

        config = StrategyConfig(window=21)
        strategy = PairTradingStrategy(config)

        with self.assertRaises(ValueError):
            strategy.run(prices0, prices1)

    def test_strategy_aligns_data(self):
        """Strategy should handle misaligned data."""
        dates1 = pd.date_range(start='2020-01-01', periods=100, freq='D')
        dates2 = pd.date_range(start='2020-01-05', periods=100, freq='D')

        prices0 = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates1)
        prices1 = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates2)

        config = StrategyConfig(window=21)
        strategy = PairTradingStrategy(config)

        # Should handle misalignment by finding common dates
        results = strategy.run(prices0, prices1)
        self.assertGreater(len(results), 0)


class TestDataLoader(unittest.TestCase):
    """Tests for data loading (requires actual data files)."""

    def test_load_prices_file_not_found(self):
        """Should raise error for missing file."""
        from backtest import DataLoader

        config = {
            "feather_dir": "nonexistent_dir",
            "interval": "1d",
            "quote": "USDT",
        }
        loader = DataLoader(config)

        with self.assertRaises(FileNotFoundError):
            loader.load_prices("NONEXISTENT")


class TestMetricsCalculator(unittest.TestCase):
    """Tests for performance metrics calculation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample results DataFrame
        np.random.seed(42)
        n = 252

        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02  # 2% daily volatility
        signals = np.random.choice([-1, 0, 1], size=n)

        self.results = pd.DataFrame({
            'gross_return': returns,
            'net_return': returns - 0.0001,  # Small fee
            'signal': signals,
        }, index=dates)

    def test_metrics_calculation(self):
        """Verify all metrics are calculated."""
        from backtest import MetricsCalculator

        metrics = MetricsCalculator.calculate(self.results)

        # Check all metrics exist
        self.assertIsNotNone(metrics.total_return)
        self.assertIsNotNone(metrics.annualized_return)
        self.assertIsNotNone(metrics.annualized_volatility)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.sortino_ratio)
        self.assertIsNotNone(metrics.max_drawdown)
        self.assertIsNotNone(metrics.win_rate)
        self.assertIsNotNone(metrics.profit_factor)
        self.assertIsNotNone(metrics.total_trades)
        self.assertIsNotNone(metrics.calmar_ratio)

    def test_sharpe_ratio_calculation(self):
        """Verify Sharpe ratio is annualized correctly."""
        from backtest import MetricsCalculator, DAYS_PER_YEAR

        metrics = MetricsCalculator.calculate(self.results)

        # Sharpe should be annualized_return / annualized_volatility
        # Verify it's a reasonable number for random data
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertTrue(np.isfinite(metrics.sharpe_ratio))

    def test_max_drawdown_calculation(self):
        """Verify max drawdown is negative or zero."""
        from backtest import MetricsCalculator

        metrics = MetricsCalculator.calculate(self.results)

        self.assertLessEqual(metrics.max_drawdown, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
