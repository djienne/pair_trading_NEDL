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
    position_sizing_mode: str = "gross_normalized"
    stop_loss_cooldown_bars: int = 5
    exit_on_decointegration: bool = False
    min_fair_value: float = 1e-12
    divergence_mode: str = "residual_over_price"

    @classmethod
    def from_dict(cls, config: dict) -> "StrategyConfig":
        """Create StrategyConfig from a dictionary."""
        strategy_config = config.get("strategy", {})
        position_sizing_mode = strategy_config.get(
            "position_sizing_mode", "gross_normalized"
        )
        if strategy_config.get("legacy_notebook_sizing", False):
            position_sizing_mode = "legacy_notebook"

        return cls(
            window=strategy_config.get("window", 21),
            kpss_threshold=strategy_config.get("kpss_threshold", 0.463),
            entry_threshold=strategy_config.get("entry_threshold", 0.02),
            stop_loss_threshold=strategy_config.get("stop_loss_threshold", -0.05),
            transaction_fee=strategy_config.get("transaction_fee", 0.0001),
            unbiased_formulation=strategy_config.get("unbiased_formulation", True),
            beta_loading=strategy_config.get("beta_loading", False),
            beta_proxy_symbol=strategy_config.get("beta_proxy_symbol", None),
            position_sizing_mode=position_sizing_mode,
            stop_loss_cooldown_bars=strategy_config.get("stop_loss_cooldown_bars", 5),
            exit_on_decointegration=strategy_config.get("exit_on_decointegration", False),
            min_fair_value=strategy_config.get("min_fair_value", 1e-12),
            divergence_mode=strategy_config.get("divergence_mode", "residual_over_price"),
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
        if len(prices_y) <= 2 or not np.all(np.isfinite(prices_y)) or not np.all(np.isfinite(prices_x)):
            return float("inf")

        a = np.mean(prices_y - b * prices_x)
        resid = prices_y - (a + b * prices_x)
        cum_resid = np.cumsum(resid)
        st_error = np.sqrt(np.sum(resid**2) / (len(resid) - 2))
        if not np.isfinite(st_error) or st_error <= 0:
            return float("inf")
        kpss = np.sum(cum_resid**2) / (len(resid)**2 * st_error**2)
        return float(kpss) if np.isfinite(kpss) else float("inf")

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
        if len(prices_y) <= 2 or not np.all(np.isfinite(prices_y)) or not np.all(np.isfinite(prices_x)):
            return float("inf")

        a, b = params
        resid = prices_y - (a + b * prices_x)
        cum_resid = np.cumsum(resid)
        st_error = np.sqrt(np.sum(resid**2) / (len(resid) - 2))
        if not np.isfinite(st_error) or st_error <= 0:
            return float("inf")
        kpss = np.sum(cum_resid**2) / (len(resid)**2 * st_error**2)
        return float(kpss) if np.isfinite(kpss) else float("inf")


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
        if len(prices_x) <= 1 or not np.all(np.isfinite(prices_x)) or not np.all(np.isfinite(prices_y)):
            return float(np.nanmean(prices_y)), 0.0

        if np.nanstd(prices_x) <= 0:
            return float(np.nanmean(prices_y)), 0.0

        X = sm.add_constant(prices_x)
        reg = sm.OLS(prices_y, X)
        res = reg.fit()
        if len(res.params) < 2 or not np.all(np.isfinite(res.params)):
            return float(np.nanmean(prices_y)), 0.0
        return float(res.params[0]), float(res.params[1])

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

        if not np.isfinite(a_opt) or not np.isfinite(b_opt) or not np.isfinite(kpss_opt):
            return float("nan"), float("nan"), float("inf")

        return float(a_opt), float(b_opt), float(kpss_opt)


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

        if not self._is_tradable_fair_value(price_y, fair_value):
            return 0

        # Calculate residual sign
        residual_sign = np.sign(price_y - fair_value)
        if residual_sign == 0 or not np.isfinite(residual_sign):
            return 0

        if old_signal != 0 and self.config.exit_on_decointegration:
            if not np.isfinite(kpss_opt) or kpss_opt > self.config.kpss_threshold:
                return 0

        # 2. If in position and residual same sign as old signal, maintain position
        if residual_sign == old_signal:
            return old_signal

        # 3. New evaluation
        # Not cointegrated
        if not np.isfinite(kpss_opt) or kpss_opt > self.config.kpss_threshold:
            return 0

        # Insufficient divergence
        divergence = self._calculate_divergence(price_y, fair_value)
        if not np.isfinite(divergence):
            return 0
        if divergence < self.config.entry_threshold:
            return 0

        # Enter position based on residual sign
        return int(residual_sign)

    def _is_tradable_fair_value(self, price_y: float, fair_value: float) -> bool:
        """Return True when price and fair value can support a meaningful signal."""
        if not np.isfinite(price_y) or not np.isfinite(fair_value):
            return False
        if price_y <= self.config.min_fair_value:
            return False
        return fair_value > self.config.min_fair_value

    def _calculate_divergence(self, price_y: float, fair_value: float) -> float:
        """Calculate the configured divergence measure."""
        residual = price_y - fair_value
        if self.config.divergence_mode == "price_over_fair_value":
            return abs(price_y / fair_value - 1)
        return abs(residual) / abs(price_y)


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
        if len(returns) < 2 or len(market_returns) < 2:
            return 0.0
        if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(market_returns)):
            return 0.0
        if np.nanstd(market_returns) <= 0:
            return 0.0

        X = sm.add_constant(market_returns)
        reg = sm.OLS(returns, X)
        res = reg.fit()
        if len(res.params) < 2 or not np.isfinite(res.params[1]):
            return 0.0
        return float(res.params[1])


@dataclass
class TradingState:
    """Current state of the trading strategy."""
    signal: int = 0
    current_return: float = 0.0
    position0: float = 0.0
    position1: float = 0.0
    cooldown_bars_remaining: int = 0
    cooldown_direction: int = 0


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
    exit_reason: str = ""
    cooldown_bars_remaining: int = 0
    cooldown_direction: int = 0
    entry_blocked_by_cooldown: bool = False


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
        if config.position_sizing_mode not in {"gross_normalized", "legacy_notebook"}:
            raise ValueError(
                "position_sizing_mode must be 'gross_normalized' or 'legacy_notebook'"
            )
        self.optimizer = CointegrationOptimizer(unbiased=config.unbiased_formulation)
        self.signal_generator = SignalGenerator(config)
        self.beta_calculator = BetaCalculator()

    def _calculate_positions(
        self,
        signal: int,
        beta0: Optional[float] = None,
        beta1: Optional[float] = None,
    ) -> tuple[float, float]:
        """Calculate position weights for the configured sizing mode."""
        if beta0 is not None and beta1 is not None:
            position0 = beta1 * signal
            position1 = -beta0 * signal
        else:
            position0 = float(signal)
            position1 = float(-signal)

        if self.config.position_sizing_mode == "gross_normalized":
            gross_exposure = abs(position0) + abs(position1)
            if gross_exposure > 0:
                position0 /= gross_exposure
                position1 /= gross_exposure

        return position0, position1

    def _calculate_beta_positions(self, df: pd.DataFrame, t: int, signal: int) -> tuple[float, float]:
        """Calculate beta-loaded positions from the historical window."""
        window_prices0 = df['price0'].iloc[t - self.config.window:t]
        window_prices1 = df['price1'].iloc[t - self.config.window:t]
        window_market = df['market'].iloc[t - self.config.window:t]

        rets0 = window_prices0.values[1:] / window_prices0.values[:-1] - 1
        rets1 = window_prices1.values[1:] / window_prices1.values[:-1] - 1
        rets_mkt = window_market.values[1:] / window_market.values[:-1] - 1

        beta0 = self.beta_calculator.calculate_beta(rets0, rets_mkt)
        beta1 = self.beta_calculator.calculate_beta(rets1, rets_mkt)

        return self._calculate_positions(signal, beta0=beta0, beta1=beta1)

    def _is_decointegrated(self, kpss_opt: float) -> bool:
        """Return True when the current window is not tradable cointegration."""
        return not np.isfinite(kpss_opt) or kpss_opt > self.config.kpss_threshold

    def _unsafe_fair_value(self, price_y: float, fair_value: float) -> bool:
        """Return True when fair value cannot support a meaningful open position."""
        return not self.signal_generator._is_tradable_fair_value(price_y, fair_value)

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

        if len(df) < self.config.window + 2:
            raise ValueError(
                f"Insufficient data: {len(df)} rows, need at least {self.config.window + 2}"
            )

        # Initialize state
        state = TradingState()
        results = []

        # Fit on t-window:t, signal at t, and realize return over t -> t+1.
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
            stop_loss_triggered = (
                old_signal != 0
                and state.current_return < self.config.stop_loss_threshold
            )
            cooldown_active = state.cooldown_bars_remaining > 0
            exit_reason = ""
            entry_blocked_by_cooldown = False

            # Generate signal
            signal = self.signal_generator.generate_signal(
                price_y=price1_t,
                fair_value=fair_value,
                kpss_opt=kpss_opt,
                old_signal=old_signal,
                current_return=state.current_return,
            )

            if (
                cooldown_active
                and signal == state.cooldown_direction
                and signal != 0
            ):
                signal = 0
                entry_blocked_by_cooldown = True
                exit_reason = "cooldown"

            if stop_loss_triggered:
                exit_reason = "stop_loss"
            elif old_signal != 0 and signal == 0:
                if self.config.exit_on_decointegration and self._is_decointegrated(kpss_opt):
                    exit_reason = "decointegration"
                elif self._unsafe_fair_value(price1_t, fair_value):
                    exit_reason = "invalid_fair_value"
                else:
                    exit_reason = "mean_reversion"

            # Calculate positions
            if self.config.beta_loading and market_prices is not None:
                position0, position1 = self._calculate_beta_positions(df, t, signal)
            else:
                position0, position1 = self._calculate_positions(signal)

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
            if stop_loss_triggered:
                state.cooldown_bars_remaining = max(self.config.stop_loss_cooldown_bars, 0)
                state.cooldown_direction = old_signal if state.cooldown_bars_remaining > 0 else 0
            elif state.cooldown_bars_remaining > 0:
                state.cooldown_bars_remaining -= 1
                if state.cooldown_bars_remaining == 0:
                    state.cooldown_direction = 0

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
                exit_reason=exit_reason,
                cooldown_bars_remaining=state.cooldown_bars_remaining,
                cooldown_direction=state.cooldown_direction,
                entry_blocked_by_cooldown=entry_blocked_by_cooldown,
            ))

        # Convert to DataFrame
        return self._results_to_dataframe(results)

    def _results_to_dataframe(self, results: list[PeriodResult]) -> pd.DataFrame:
        """Convert list of PeriodResult to DataFrame."""
        columns = [
            'date', 'price0', 'price1', 'a_opt', 'b_opt', 'kpss_opt',
            'fair_value', 'signal', 'position0', 'position1', 'gross_return',
            'net_return', 'current_return', 'stop_loss_triggered',
            'exit_reason', 'cooldown_bars_remaining', 'cooldown_direction',
            'entry_blocked_by_cooldown',
        ]
        if not results:
            df = pd.DataFrame(columns=columns)
            df.set_index('date', inplace=True)
            return df

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
                'exit_reason': r.exit_reason,
                'cooldown_bars_remaining': r.cooldown_bars_remaining,
                'cooldown_direction': r.cooldown_direction,
                'entry_blocked_by_cooldown': r.entry_blocked_by_cooldown,
            }
            for r in results
        ], columns=columns)
        df.set_index('date', inplace=True)
        return df
