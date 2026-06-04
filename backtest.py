"""
Backtesting Engine with Visualizations

Provides comprehensive backtesting, performance metrics, and visualization
for the cointegration-based pair trading strategy.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pair_trading import PairTradingStrategy, StrategyConfig
from utils import load_config


# Crypto markets trade 24/7, so we use 365 days for annualization
DAYS_PER_YEAR = 365


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0

    @classmethod
    def from_dict(cls, config: dict) -> "BacktestConfig":
        """Create BacktestConfig from a dictionary."""
        backtest_config = config.get("backtest", {})
        return cls(
            initial_capital=backtest_config.get("initial_capital", 10000.0),
        )


class DataLoader:
    """Load and align price data from feather files."""

    def __init__(self, config: dict):
        """
        Initialize the data loader.

        Args:
            config: Configuration dictionary with data paths
        """
        self.config = config
        self.feather_dir = config.get("feather_dir", "data/feather")
        self.interval = config.get("interval", "1d")
        self.quote = config.get("quote", "USDT")

    def load_prices(self, symbol: str) -> pd.Series:
        """
        Load closing prices for a symbol.

        Args:
            symbol: Symbol name (e.g., "BTC")

        Returns:
            Series of closing prices indexed by date
        """
        filename = f"{symbol}{self.quote}_{self.interval}.feather"
        path = os.path.join(self.feather_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_feather(path)
        if self.interval != "1d":
            import warnings
            warnings.warn(
                f"Interval '{self.interval}' is not '1d'. The .dt.normalize() call "
                f"will collapse intraday candles to the same date.",
                UserWarning
            )
        df['date'] = pd.to_datetime(df['open_time_dt']).dt.normalize()
        df.set_index('date', inplace=True)

        # Filter by date range if specified
        start_date = self.config.get("start_date")
        end_date = self.config.get("end_date")

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df['close']

    def load_pair(self, symbol0: str, symbol1: str) -> Tuple[pd.Series, pd.Series]:
        """
        Load aligned price series for a pair.

        Args:
            symbol0: First symbol (independent variable X)
            symbol1: Second symbol (dependent variable Y)

        Returns:
            Tuple of (prices0, prices1) aligned by date
        """
        prices0 = self.load_prices(symbol0)
        prices1 = self.load_prices(symbol1)

        # Align by date
        combined = pd.DataFrame({
            'price0': prices0,
            'price1': prices1,
        }).dropna()

        return combined['price0'], combined['price1']


@dataclass
class PerformanceMetrics:
    """Performance metrics for strategy evaluation."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    calmar_ratio: float

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "Total Return (%)": self.total_return * 100,
            "Annualized Return (%)": self.annualized_return * 100,
            "Annualized Volatility (%)": self.annualized_volatility * 100,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Max Drawdown (%)": self.max_drawdown * 100,
            "Win Rate (%)": self.win_rate * 100,
            "Profit Factor": self.profit_factor,
            "Total Trades": self.total_trades,
            "Calmar Ratio": self.calmar_ratio,
        }

    def __str__(self) -> str:
        """Format metrics as string."""
        lines = []
        for name, value in self.to_dict().items():
            if isinstance(value, int):
                lines.append(f"{name}: {value}")
            else:
                lines.append(f"{name}: {value:.4f}")
        return "\n".join(lines)


class MetricsCalculator:
    """Calculate performance metrics from strategy results."""

    @staticmethod
    def calculate(results: pd.DataFrame, use_net: bool = True) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            results: Strategy results DataFrame
            use_net: If True, use net returns (after fees), else gross returns

        Returns:
            PerformanceMetrics instance
        """
        returns_col = 'net_return' if use_net else 'gross_return'
        returns = results[returns_col].fillna(0)

        # Total return
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        # Annualized return
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (DAYS_PER_YEAR / n_periods) - 1

        # Annualized volatility
        daily_vol = returns.std()
        annualized_volatility = daily_vol * np.sqrt(DAYS_PER_YEAR)

        # Sharpe ratio (assuming 0 risk-free rate for crypto)
        sharpe_ratio = (annualized_return / annualized_volatility
                       if annualized_volatility > 0 else 0)

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_dev = np.sqrt((negative_returns ** 2).mean()) * np.sqrt(DAYS_PER_YEAR)
        sortino_ratio = annualized_return / downside_dev if downside_dev > 0 else 0

        # Max drawdown
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        non_zero_returns = returns[returns != 0]
        win_rate = (non_zero_returns > 0).mean() if len(non_zero_returns) > 0 else 0

        # Profit factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

        # Total trades (count signal changes)
        signal_changes = results['signal'].diff().fillna(0) != 0
        total_trades = signal_changes.sum()

        # Calmar ratio
        calmar_ratio = (annualized_return / abs(max_drawdown)
                       if max_drawdown != 0 else float('inf'))

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=int(total_trades),
            calmar_ratio=calmar_ratio,
        )


class Visualizer:
    """Create comprehensive visualizations for backtest results."""

    def __init__(self, results: pd.DataFrame, metrics: PerformanceMetrics,
                 symbol0: str, symbol1: str, config: StrategyConfig):
        """
        Initialize the visualizer.

        Args:
            results: Strategy results DataFrame
            metrics: Calculated performance metrics
            symbol0: First symbol name
            symbol1: Second symbol name
            config: Strategy configuration
        """
        self.results = results
        self.metrics = metrics
        self.symbol0 = symbol0
        self.symbol1 = symbol1
        self.config = config

    def plot_equity_curves(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot equity curves: gross, net, and buy-hold benchmarks."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Strategy equity curves
        gross_equity = (1 + self.results['gross_return']).cumprod()
        net_equity = (1 + self.results['net_return']).cumprod()

        # Buy-hold benchmarks
        buyhold0 = self.results['price0'] / self.results['price0'].iloc[0]
        buyhold1 = self.results['price1'] / self.results['price1'].iloc[0]

        ax.plot(self.results.index, gross_equity, label='Strategy (Gross)', linewidth=2)
        ax.plot(self.results.index, net_equity, label='Strategy (Net)', linewidth=2)
        ax.plot(self.results.index, buyhold0, label=f'Buy-Hold {self.symbol0}',
                alpha=0.7, linestyle='--')
        ax.plot(self.results.index, buyhold1, label=f'Buy-Hold {self.symbol1}',
                alpha=0.7, linestyle='--')

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Equity Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_prices(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot price charts with trade markers."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Normalize prices for comparison
        price0_norm = self.results['price0'] / self.results['price0'].iloc[0]
        price1_norm = self.results['price1'] / self.results['price1'].iloc[0]

        ax.plot(self.results.index, price0_norm, label=f'{self.symbol0} (normalized)')
        ax.plot(self.results.index, price1_norm, label=f'{self.symbol1} (normalized)')

        # Mark trade entries and exits
        signal_changes = self.results['signal'].diff().fillna(0)
        entries = self.results.index[signal_changes != 0]

        for entry in entries:
            ax.axvline(x=entry, color='gray', alpha=0.3, linestyle=':')

        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price')
        ax.set_title('Price Charts')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_spread(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot spread/residuals over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        residual = self.results['price1'] - self.results['fair_value']

        ax.plot(self.results.index, residual, label='Residual', linewidth=1)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        ax.fill_between(self.results.index, 0, residual,
                       where=residual > 0, alpha=0.3, color='green')
        ax.fill_between(self.results.index, 0, residual,
                       where=residual < 0, alpha=0.3, color='red')

        ax.set_xlabel('Date')
        ax.set_ylabel('Residual')
        ax.set_title('Spread (Residual)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_price_ratio(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot price ratio with entry threshold bands."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ratio = self.results['price1'] / self.results['fair_value']

        ax.plot(self.results.index, ratio, label='Price/Fair Value', linewidth=1)
        ax.axhline(y=1, color='black', linestyle='-', alpha=0.5)

        # Entry threshold bands
        threshold = self.config.entry_threshold
        ax.axhline(y=1 + threshold, color='green', linestyle='--',
                  alpha=0.7, label=f'+{threshold*100:.0f}% threshold')
        ax.axhline(y=1 - threshold, color='red', linestyle='--',
                  alpha=0.7, label=f'-{threshold*100:.0f}% threshold')

        ax.fill_between(self.results.index, 1 - threshold, 1 + threshold,
                       alpha=0.1, color='gray')

        ax.set_xlabel('Date')
        ax.set_ylabel('Ratio')
        ax.set_title('Price Ratio (Y / Fair Value)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_kpss(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot KPSS statistic over time with threshold."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        kpss = self.results['kpss_opt']

        ax.plot(self.results.index, kpss, label='KPSS Statistic', linewidth=1)
        ax.axhline(y=self.config.kpss_threshold, color='red', linestyle='--',
                  label=f'Threshold ({self.config.kpss_threshold})')

        # Shade cointegration regions
        ax.fill_between(self.results.index, 0, kpss,
                       where=kpss < self.config.kpss_threshold,
                       alpha=0.3, color='green', label='Cointegrated')

        ax.set_xlabel('Date')
        ax.set_ylabel('KPSS Statistic')
        ax.set_title('KPSS Statistic Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_signal(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot trading signal over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        signal = self.results['signal']

        ax.fill_between(self.results.index, 0, signal,
                       where=signal > 0, alpha=0.5, color='green', label='Long X/Short Y')
        ax.fill_between(self.results.index, 0, signal,
                       where=signal < 0, alpha=0.5, color='red', label='Short X/Long Y')
        ax.plot(self.results.index, signal, color='black', linewidth=0.5)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Short', 'Flat', 'Long'])

        ax.set_xlabel('Date')
        ax.set_ylabel('Position')
        ax.set_title('Trading Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_drawdown(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot drawdown chart (underwater curve)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        cumulative = (1 + self.results['net_return']).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        ax.fill_between(self.results.index, 0, drawdown,
                       alpha=0.5, color='red')
        ax.plot(self.results.index, drawdown, color='darkred', linewidth=1)

        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_value*100:.1f}%',
                   xy=(max_dd_idx, max_dd_value),
                   xytext=(max_dd_idx, max_dd_value - 0.05),
                   fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color='black'))

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown (Underwater Curve)')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_returns_distribution(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot histogram of daily returns."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        returns = self.results['net_return'].dropna()

        ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(x=returns.mean(), color='red', linestyle='--',
                  label=f'Mean: {returns.mean()*100:.3f}%')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_rolling_metrics(self, ax: Optional[plt.Axes] = None, window: int = 30) -> plt.Axes:
        """Plot rolling Sharpe ratio and volatility."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        returns = self.results['net_return']

        # Rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(DAYS_PER_YEAR)

        # Rolling Sharpe
        rolling_mean = returns.rolling(window=window).mean() * DAYS_PER_YEAR
        rolling_sharpe = rolling_mean / rolling_vol

        ax2 = ax.twinx()

        line1, = ax.plot(self.results.index, rolling_vol,
                        color='blue', alpha=0.7, label='Rolling Volatility')
        line2, = ax2.plot(self.results.index, rolling_sharpe,
                         color='orange', alpha=0.7, label='Rolling Sharpe')

        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility', color='blue')
        ax2.set_ylabel('Sharpe Ratio', color='orange')
        ax.set_title(f'Rolling Metrics ({window}-day window)')

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)

        return ax

    def create_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive dashboard with all plots and metrics table."""
        fig = plt.figure(figsize=(20, 24))

        # Create grid for plots
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.2)

        # Row 1
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_equity_curves(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_prices(ax2)

        # Row 2
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_spread(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_price_ratio(ax4)

        # Row 3
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_kpss(ax5)

        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_signal(ax6)

        # Row 4
        ax7 = fig.add_subplot(gs[3, 0])
        self.plot_drawdown(ax7)

        ax8 = fig.add_subplot(gs[3, 1])
        self.plot_returns_distribution(ax8)

        # Row 5
        ax9 = fig.add_subplot(gs[4, 0])
        self.plot_rolling_metrics(ax9)

        # Metrics table
        ax10 = fig.add_subplot(gs[4, 1])
        ax10.axis('off')

        metrics_dict = self.metrics.to_dict()
        table_data = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)]
                     for k, v in metrics_dict.items()]

        table = ax10.table(cellText=table_data,
                          colLabels=['Metric', 'Value'],
                          loc='center',
                          cellLoc='left',
                          colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax10.set_title('Performance Metrics', fontsize=12, fontweight='bold')

        # Main title
        fig.suptitle(f'Pair Trading Backtest: {self.symbol0}/{self.symbol1}',
                    fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")

        return fig


class BacktestEngine:
    """Main backtesting engine that orchestrates strategy execution and analysis."""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the backtest engine.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.strategy_config = StrategyConfig.from_dict(self.config)
        self.backtest_config = BacktestConfig.from_dict(self.config)
        self.data_loader = DataLoader(self.config)

    def run(self, symbol0: Optional[str] = None, symbol1: Optional[str] = None,
            save_results: bool = True, show_plots: bool = True) -> Tuple[pd.DataFrame, PerformanceMetrics]:
        """
        Run the backtest.

        Args:
            symbol0: First symbol (defaults to config symbols[0])
            symbol1: Second symbol (defaults to config symbols[1])
            save_results: If True, save results to feather file
            show_plots: If True, display the dashboard

        Returns:
            Tuple of (results DataFrame, performance metrics)
        """
        # Get symbols from config if not provided
        symbols = self.config.get("symbols", ["BTC", "ETH"])
        symbol0 = symbol0 or symbols[0]
        symbol1 = symbol1 or symbols[1]

        print(f"Running backtest: {symbol0}/{symbol1}")
        print(f"Strategy config: window={self.strategy_config.window}, "
              f"KPSS threshold={self.strategy_config.kpss_threshold}, "
              f"entry threshold={self.strategy_config.entry_threshold}")

        # Load data
        print("Loading data...")
        prices0, prices1 = self.data_loader.load_pair(symbol0, symbol1)
        print(f"Loaded {len(prices0)} data points from {prices0.index[0]} to {prices0.index[-1]}")

        # Load market prices for beta loading if enabled
        market_prices = None
        if self.strategy_config.beta_loading and self.strategy_config.beta_proxy_symbol:
            market_prices = self.data_loader.load_prices(self.strategy_config.beta_proxy_symbol)

        # Run strategy
        print("Running strategy...")
        strategy = PairTradingStrategy(self.strategy_config)
        results = strategy.run(prices0, prices1, market_prices)

        # Calculate metrics
        print("Calculating metrics...")
        metrics = MetricsCalculator.calculate(results)
        print("\n" + str(metrics))

        # Save results
        if save_results:
            output_dir = self.config.get("output_dir", "output")
            os.makedirs(output_dir, exist_ok=True)

            results_path = os.path.join(output_dir, f"backtest_{symbol0}_{symbol1}.feather")
            results.reset_index().to_feather(results_path)
            print(f"\nResults saved to: {results_path}")

        # Create visualizations
        if show_plots:
            print("\nGenerating dashboard...")
            visualizer = Visualizer(results, metrics, symbol0, symbol1, self.strategy_config)

            output_dir = self.config.get("output_dir", "output")
            if not save_results:
                os.makedirs(output_dir, exist_ok=True)
            dashboard_path = os.path.join(output_dir, f"dashboard_{symbol0}_{symbol1}.png")

            visualizer.create_dashboard(save_path=dashboard_path)
            plt.show()

        return results, metrics


def main():
    """Main entry point for backtesting."""
    import argparse

    parser = argparse.ArgumentParser(description="Run pair trading backtest")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--symbol0", help="First symbol (e.g., BTC)")
    parser.add_argument("--symbol1", help="Second symbol (e.g., ETH)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--no-plots", action="store_true", help="Don't show plots")

    args = parser.parse_args()

    engine = BacktestEngine(args.config)
    results, metrics = engine.run(
        symbol0=args.symbol0,
        symbol1=args.symbol1,
        save_results=not args.no_save,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
