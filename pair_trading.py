"""
Cointegration-based Pair Trading Strategy

Implementation of the NEDL cointegration pair trading strategy from Part 2,
adapted for cryptocurrency pairs using Binance Futures data.
"""

import numpy as np
import pandas as pd
import scipy.optimize as spop
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StrategyConfig:
    """Configuration for the pair trading strategy."""
    window: int = 21
    kpss_threshold: float = 0.463
    entry_threshold: float = 0.02
    stop_loss_threshold: float = -0.05
    transaction_fee: float = 0.0001
    unbiased_formulation: bool = True
    beta_loading: bool = False
    beta_proxy_symbol: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "StrategyConfig":
        """Create StrategyConfig from a dictionary."""
        strategy_config = config.get("strategy", {})
        return cls(
            window=strategy_config.get("window", 21),
            kpss_threshold=strategy_config.get("kpss_threshold", 0.463),
            entry_threshold=strategy_config.get("entry_threshold", 0.02),
            stop_loss_threshold=strategy_config.get("stop_loss_threshold", -0.05),
            transaction_fee=strategy_config.get("transaction_fee", 0.0001),
            unbiased_formulation=strategy_config.get("unbiased_formulation", True),
            beta_loading=strategy_config.get("beta_loading", False),
            beta_proxy_symbol=strategy_config.get("beta_proxy_symbol", None),
        )


class KPSSCalculator:
    """
    Compute KPSS statistic for cointegration testing.

    The KPSS test is used to determine if a series is stationary.
    Lower KPSS values indicate stationarity (cointegration).
    """

    @staticmethod
    def kpss_unbiased(b: float, prices_y: np.ndarray, prices_x: np.ndarray) -> float:
        """
        Calculate KPSS statistic using unbiased one-parameter formulation.

        In this formulation, 'a' is calculated from 'b' to minimize KPSS.

        Args:
            b: Slope coefficient for the cointegration relationship
            prices_y: Price series of the dependent asset
            prices_x: Price series of the independent asset

        Returns:
            KPSS statistic value
        """
        a = np.mean(prices_y - b * prices_x)
        resid = prices_y - (a + b * prices_x)
        cum_resid = np.cumsum(resid)
        st_error = np.sqrt(np.sum(resid**2) / (len(resid) - 2))
        kpss = np.sum(cum_resid**2) / (len(resid)**2 * st_error**2)
        return kpss

    @staticmethod
    def kpss_two_param(params: np.ndarray, prices_y: np.ndarray, prices_x: np.ndarray) -> float:
        """
        Calculate KPSS statistic using two-parameter formulation.

        Both 'a' and 'b' are optimized directly.

        Args:
            params: Array of [a, b] coefficients
            prices_y: Price series of the dependent asset
            prices_x: Price series of the independent asset

        Returns:
            KPSS statistic value
        """
        a, b = params
        resid = prices_y - (a + b * prices_x)
        cum_resid = np.cumsum(resid)
        st_error = np.sqrt(np.sum(resid**2) / (len(resid) - 2))
        kpss = np.sum(cum_resid**2) / (len(resid)**2 * st_error**2)
        return kpss


class CointegrationOptimizer:
    """
    Optimize cointegration parameters using Nelder-Mead optimization.

    Finds the optimal coefficients (a, b) that minimize the KPSS statistic,
    i.e., maximize the stationarity of the residuals.
    """

    def __init__(self, unbiased: bool = True):
        """
        Initialize the optimizer.

        Args:
            unbiased: If True, use one-parameter unbiased formulation.
                     If False, use two-parameter formulation.
        """
        self.unbiased = unbiased
        self.kpss_calc = KPSSCalculator()

    def get_ols_initial_values(self, prices_y: np.ndarray, prices_x: np.ndarray) -> Tuple[float, float]:
        """
        Get OLS regression coefficients as starting values for optimization.

        Args:
            prices_y: Price series of the dependent asset
            prices_x: Price series of the independent asset

        Returns:
            Tuple of (intercept, slope) from OLS regression
        """
        X = sm.add_constant(prices_x)
        reg = sm.OLS(prices_y, X)
        res = reg.fit()
        return res.params[0], res.params[1]

    def optimize(self, prices_y: np.ndarray, prices_x: np.ndarray) -> Tuple[float, float, float]:
        """
        Find optimal cointegration parameters.

        Args:
            prices_y: Price series of the dependent asset
            prices_x: Price series of the independent asset

        Returns:
            Tuple of (a_opt, b_opt, kpss_opt)
        """
        a0, b0 = self.get_ols_initial_values(prices_y, prices_x)

        if self.unbiased:
            # One-parameter optimization (optimize b only, derive a)
            def objective(b):
                return self.kpss_calc.kpss_unbiased(b, prices_y, prices_x)

            result = spop.minimize(objective, b0, method='Nelder-Mead')
            kpss_opt = result.fun
            b_opt = float(result.x.item() if hasattr(result.x, 'item') else result.x)
            a_opt = np.mean(prices_y - b_opt * prices_x)
        else:
            # Two-parameter optimization
            def objective(params):
                return self.kpss_calc.kpss_two_param(params, prices_y, prices_x)

            result = spop.minimize(objective, [a0, b0], method='Nelder-Mead')
            kpss_opt = result.fun
            a_opt, b_opt = result.x

        return a_opt, b_opt, kpss_opt


class SignalGenerator:
    """
    Generate trading signals based on cointegration analysis.

    Signal logic (from Part 2):
    1. Check stop-loss first
    2. If in position and residual same sign as old signal, maintain position
    3. Otherwise, evaluate new entry:
       - No trade if not cointegrated (KPSS > threshold)
       - No trade if insufficient divergence
       - Otherwise, signal based on sign of residual
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize the signal generator.

        Args:
            config: Strategy configuration parameters
        """
        self.config = config

    def generate_signal(
        self,
        price_y: float,
        fair_value: float,
        kpss_opt: float,
        old_signal: int,
        current_return: float,
    ) -> int:
        """
        Generate trading signal for the current period.

        Args:
            price_y: Current price of dependent asset
            fair_value: Fair value = a + b * price_x
            kpss_opt: Optimal KPSS statistic for the window
            old_signal: Previous period's signal
            current_return: Cumulative return of current position

        Returns:
            Signal: +1 (long X, short Y), -1 (short X, long Y), 0 (no position)
        """
        # 1. Stop-loss check first
        if current_return < self.config.stop_loss_threshold:
            return 0

        # Calculate residual sign
        residual_sign = np.sign(price_y - fair_value)

        # 2. If in position and residual same sign as old signal, maintain position
        if residual_sign == old_signal:
            return old_signal

        # 3. New evaluation
        # Not cointegrated
        if kpss_opt > self.config.kpss_threshold:
            return 0

        # Insufficient divergence
        divergence = abs(price_y / fair_value - 1)
        if divergence < self.config.entry_threshold:
            return 0

        # Enter position based on residual sign
        return int(residual_sign)


class BetaCalculator:
    """Calculate beta coefficients for beta-loading strategy."""

    @staticmethod
    def calculate_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Calculate beta coefficient relative to market.

        Args:
            returns: Asset returns
            market_returns: Market returns

        Returns:
            Beta coefficient
        """
        X = sm.add_constant(market_returns)
        reg = sm.OLS(returns, X)
        res = reg.fit()
        return res.params[1]


@dataclass
class TradingState:
    """Current state of the trading strategy."""
    signal: int = 0
    current_return: float = 0.0
    position0: float = 0.0
    position1: float = 0.0


@dataclass
class PeriodResult:
    """Results for a single trading period."""
    date: pd.Timestamp
    price0: float
    price1: float
    a_opt: float
    b_opt: float
    kpss_opt: float
    fair_value: float
    signal: int
    position0: float
    position1: float
    gross_return: float
    net_return: float
    current_return: float
    stop_loss_triggered: bool = False


class PairTradingStrategy:
    """
    Main orchestrator for the pair trading strategy.

    Implements the complete cointegration-based pair trading strategy
    from NEDL Part 2, adapted for cryptocurrency pairs.
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.

        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.optimizer = CointegrationOptimizer(unbiased=config.unbiased_formulation)
        self.signal_generator = SignalGenerator(config)
        self.beta_calculator = BetaCalculator()

    def run(
        self,
        prices0: pd.Series,
        prices1: pd.Series,
        market_prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Run the pair trading strategy.

        Convention from Part 2:
        - ticker[0] is the independent variable (X) in cointegration
        - ticker[1] is the dependent variable (Y) in cointegration
        - Y = a + b * X
        - position0 = signal (long ticker[0] when signal > 0)
        - position1 = -signal (short ticker[1] when signal > 0)

        Args:
            prices0: Price series for first asset (independent, X)
            prices1: Price series for second asset (dependent, Y)
            market_prices: Optional market prices for beta loading

        Returns:
            DataFrame with strategy results
        """
        # Align data
        df = pd.DataFrame({
            'price0': prices0,
            'price1': prices1,
        }).dropna()

        if market_prices is not None:
            df['market'] = market_prices
            df = df.dropna()

        if len(df) < self.config.window + 1:
            raise ValueError(
                f"Insufficient data: {len(df)} rows, need at least {self.config.window + 1}"
            )

        # Initialize state
        state = TradingState()
        results = []

        # Loop through the sample (matching notebook exactly)
        for t in range(self.config.window, len(df) - 1):
            old_signal = state.signal
            old_position0 = state.position0
            old_position1 = state.position1

            # Get subsample for this window
            window_data = df.iloc[t - self.config.window:t]
            prices_x = np.array(window_data['price0'])  # ticker[0]
            prices_y = np.array(window_data['price1'])  # ticker[1]

            # Optimize cointegration parameters
            a_opt, b_opt, kpss_opt = self.optimizer.optimize(prices_y, prices_x)

            # Current prices
            price0_t = df['price0'].iloc[t]
            price1_t = df['price1'].iloc[t]
            fair_value = a_opt + b_opt * price0_t

            # Check stop-loss
            stop_loss_triggered = state.current_return < self.config.stop_loss_threshold

            # Generate signal
            signal = self.signal_generator.generate_signal(
                price_y=price1_t,
                fair_value=fair_value,
                kpss_opt=kpss_opt,
                old_signal=old_signal,
                current_return=state.current_return,
            )

            # Calculate positions
            if self.config.beta_loading and market_prices is not None:
                # Beta loading
                window_prices0 = df['price0'].iloc[t - self.config.window:t - 1]
                window_prices1 = df['price1'].iloc[t - self.config.window:t - 1]
                window_market = df['market'].iloc[t - self.config.window:t - 1]

                rets0 = window_prices0.values[:-1] / window_prices0.values[1:] - 1
                rets1 = window_prices1.values[:-1] / window_prices1.values[1:] - 1
                rets_mkt = window_market.values[:-1] / window_market.values[1:] - 1

                beta0 = self.beta_calculator.calculate_beta(rets0, rets_mkt)
                beta1 = self.beta_calculator.calculate_beta(rets1, rets_mkt)

                position0 = beta1 * signal
                position1 = -beta0 * signal
            else:
                # Standard positions
                position0 = signal
                position1 = -signal

            # Calculate returns
            price0_next = df['price0'].iloc[t + 1]
            price1_next = df['price1'].iloc[t + 1]

            gross_return = (
                position0 * (price0_next / price0_t - 1) +
                position1 * (price1_next / price1_t - 1)
            )

            # Transaction costs on position changes
            net_return = gross_return - self.config.transaction_fee * (
                abs(position0 - old_position0) + abs(position1 - old_position1)
            )

            # Update current return tracking
            if signal == old_signal:
                state.current_return = (1 + state.current_return) * (1 + gross_return) - 1
            else:
                state.current_return = gross_return

            # Update state
            state.signal = signal
            state.position0 = position0
            state.position1 = position1

            # Store results
            results.append(PeriodResult(
                date=df.index[t],
                price0=price0_t,
                price1=price1_t,
                a_opt=a_opt,
                b_opt=b_opt,
                kpss_opt=kpss_opt,
                fair_value=fair_value,
                signal=signal,
                position0=position0,
                position1=position1,
                gross_return=gross_return,
                net_return=net_return,
                current_return=state.current_return,
                stop_loss_triggered=stop_loss_triggered,
            ))

        # Convert to DataFrame
        return self._results_to_dataframe(results)

    def _results_to_dataframe(self, results: list[PeriodResult]) -> pd.DataFrame:
        """Convert list of PeriodResult to DataFrame."""
        df = pd.DataFrame([
            {
                'date': r.date,
                'price0': r.price0,
                'price1': r.price1,
                'a_opt': r.a_opt,
                'b_opt': r.b_opt,
                'kpss_opt': r.kpss_opt,
                'fair_value': r.fair_value,
                'signal': r.signal,
                'position0': r.position0,
                'position1': r.position1,
                'gross_return': r.gross_return,
                'net_return': r.net_return,
                'current_return': r.current_return,
                'stop_loss_triggered': r.stop_loss_triggered,
            }
            for r in results
        ])
        df.set_index('date', inplace=True)
        return df
