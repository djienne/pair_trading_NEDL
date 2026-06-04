"""
Multi-Pair Crypto Scanner

Scans all pair combinations from a list of cryptocurrencies using the
existing pair trading strategy and backtest infrastructure.
"""

import argparse
import itertools
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from backtest import DataLoader, MetricsCalculator, PerformanceMetrics, Visualizer
from download_data import download_symbols
from pair_trading import PairTradingStrategy, StrategyConfig
from utils import load_config


# Default top 20 cryptocurrencies by market cap (available on Binance Futures)
# Note: SHIB excluded due to limited history, replaced with APT
TOP_20_SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "AVAX", "DOGE", "DOT", "TRX",
    "LINK", "MATIC", "LTC", "BCH", "UNI",
    "ATOM", "XLM", "ETC", "FIL", "APT",
]


class PairScanner:
    """Scanner for testing all pair combinations from a list of symbols."""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the scanner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.strategy_config = StrategyConfig.from_dict(self.config)
        self.data_loader = DataLoader(self.config)
        self.results: list[dict] = []

    def download_all_data(self, symbols: list[str]) -> list[str]:
        """
        Download data for all symbols.

        Args:
            symbols: List of symbol names (e.g., ["BTC", "ETH"])

        Returns:
            List of symbols that were successfully downloaded
        """
        print(f"Downloading data for {len(symbols)} symbols...")
        successful = download_symbols(self.config, symbols, full_symbol_names=False)
        print(f"Download complete: {len(successful)}/{len(symbols)} symbols")
        return successful

    def generate_pairs(self, symbols: list[str]) -> list[tuple[str, str]]:
        """
        Generate all unique pairs from the symbol list.

        Args:
            symbols: List of symbol names

        Returns:
            List of (symbol0, symbol1) tuples
        """
        return list(itertools.combinations(symbols, 2))

    def run_single_pair(self, sym0: str, sym1: str) -> Optional[dict]:
        """
        Run backtest on a single pair.

        Args:
            sym0: First symbol (independent variable X)
            sym1: Second symbol (dependent variable Y)

        Returns:
            Dictionary with pair metrics or None on error
        """
        try:
            # Load data
            prices0, prices1 = self.data_loader.load_pair(sym0, sym1)

            # Check minimum data requirement
            min_required = self.strategy_config.window + 10
            if len(prices0) < min_required:
                print(f"  Warning: {sym0}/{sym1} has insufficient data ({len(prices0)} rows)")
                return None

            # Run strategy
            strategy = PairTradingStrategy(self.strategy_config)
            results = strategy.run(prices0, prices1)

            # Calculate metrics
            metrics = MetricsCalculator.calculate(results)

            return {
                "pair": f"{sym0}/{sym1}",
                "symbol0": sym0,
                "symbol1": sym1,
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "annualized_volatility": metrics.annualized_volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
                "calmar_ratio": metrics.calmar_ratio,
                "data_points": len(results),
            }

        except FileNotFoundError as e:
            print(f"  Warning: Data not found for {sym0}/{sym1}: {e}")
            return None
        except ValueError as e:
            print(f"  Warning: Error processing {sym0}/{sym1}: {e}")
            return None
        except Exception as e:
            print(f"  Warning: Unexpected error for {sym0}/{sym1}: {e}")
            return None

    def scan(
        self,
        symbols: Optional[list[str]] = None,
        download: bool = True,
        auto_fetch: bool = True,
    ) -> pd.DataFrame:
        """
        Run full scan on all pairs.

        Args:
            symbols: List of symbols to scan (defaults to config scan_symbols or TOP_20)
            download: If True, download data before scanning
            auto_fetch: If True and no symbols specified, fetch from Binance API

        Returns:
            DataFrame with results sorted by Sharpe ratio
        """
        # Get symbols
        if symbols is None:
            config_symbols = self.config.get("scan_symbols")
            if config_symbols:
                symbols = config_symbols
            elif auto_fetch and self.config.get("auto_fetch_symbols", True):
                from symbol_ranking import get_top_symbols
                symbols = get_top_symbols(
                    base_url=self.config.get("binance_base_url", "https://fapi.binance.com"),
                    top_n=self.config.get("top_n_symbols", 20),
                    cache_file=self.config.get("symbol_cache_file", "data/symbol_cache.json"),
                    cache_hours=self.config.get("symbol_cache_hours", 24),
                )
            else:
                symbols = TOP_20_SYMBOLS

        print(f"Scanning {len(symbols)} symbols...")

        # Download data if requested
        if download:
            successful_symbols = self.download_all_data(symbols)
            # Only use symbols that were successfully downloaded
            symbols = [s for s in symbols if s in successful_symbols]
            print(f"Using {len(symbols)} symbols with available data")

        pairs = self.generate_pairs(symbols)
        print(f"Generated {len(pairs)} unique pairs")

        # Run backtest on each pair
        self.results = []
        successful = 0
        failed = 0

        for i, (sym0, sym1) in enumerate(pairs):
            print(f"[{i+1}/{len(pairs)}] Testing {sym0}/{sym1}...", end=" ")
            result = self.run_single_pair(sym0, sym1)

            if result is not None:
                self.results.append(result)
                print(f"Sharpe: {result['sharpe_ratio']:.2f}, Return: {result['total_return']*100:.1f}%")
                successful += 1
            else:
                failed += 1

        print(f"\nCompleted: {successful} successful, {failed} failed")

        # Convert to DataFrame
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        return df

    def save_results(self, df: pd.DataFrame, output_path: Optional[str] = None) -> tuple[str, str]:
        """
        Save results to CSV and feather files.

        Args:
            df: Results DataFrame
            output_path: Base output path (without extension)

        Returns:
            Tuple of (csv_path, feather_path)
        """
        output_dir = self.config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        if output_path is None:
            base_path = os.path.join(output_dir, "pair_scan_results")
        else:
            base_path = output_path.rsplit(".", 1)[0]  # Remove extension if present

        csv_path = f"{base_path}.csv"
        feather_path = f"{base_path}.feather"

        df.to_csv(csv_path, index=False)
        df.to_feather(feather_path)

        return csv_path, feather_path


def format_results_table(df: pd.DataFrame, sort_by: str, top_n: int = 10) -> str:
    """
    Format results as a printable table.

    Args:
        df: Results DataFrame
        sort_by: Column to sort by
        top_n: Number of rows to show

    Returns:
        Formatted table string
    """
    if df.empty:
        return "No results to display."

    # Sort and take top N
    ascending = sort_by == "max_drawdown"  # Lower is better for max_drawdown
    sorted_df = df.sort_values(sort_by, ascending=ascending).head(top_n)

    lines = []
    header = f"{'Rank':<6}{'Pair':<14}{'Total Ret':>12}{'Ann. Ret':>12}{'Sharpe':>10}{'Sortino':>10}{'Max DD':>12}{'Trades':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        total_ret = f"{row['total_return']*100:+.2f}%"
        ann_ret = f"{row['annualized_return']*100:+.2f}%"
        sharpe = f"{row['sharpe_ratio']:.2f}"
        sortino = f"{row['sortino_ratio']:.2f}"
        max_dd = f"{row['max_drawdown']*100:.2f}%"
        trades = f"{row['total_trades']}"

        line = f"{i:<6}{row['pair']:<14}{total_ret:>12}{ann_ret:>12}{sharpe:>10}{sortino:>10}{max_dd:>12}{trades:>8}"
        lines.append(line)

    return "\n".join(lines)


def _get_unique_best_pairs(df: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    """
    Get unique best pairs by Sharpe ratio and total return.

    Args:
        df: Results DataFrame

    Returns:
        List of (row, label) tuples for unique best pairs
    """
    best_sharpe = df.loc[df["sharpe_ratio"].idxmax()]
    best_return = df.loc[df["total_return"].idxmax()]

    pairs = [(best_sharpe, "best_sharpe")]
    if best_return["pair"] != best_sharpe["pair"]:
        pairs.append((best_return, "best_pnl"))

    return pairs


def _run_pair_backtest(scanner: PairScanner, sym0: str, sym1: str) -> tuple[pd.DataFrame, PerformanceMetrics]:
    """
    Load data and run backtest for a pair.

    Args:
        scanner: PairScanner instance
        sym0: First symbol
        sym1: Second symbol

    Returns:
        Tuple of (results DataFrame, metrics)
    """
    prices0, prices1 = scanner.data_loader.load_pair(sym0, sym1)
    strategy = PairTradingStrategy(scanner.strategy_config)
    results = strategy.run(prices0, prices1)
    metrics = MetricsCalculator.calculate(results)
    return results, metrics


def plot_scan_summary(
    df: pd.DataFrame,
    scanner: PairScanner,
    save_path: Optional[str] = None,
    generate_dashboards: bool = False,
    output_dir: Optional[str] = None,
) -> tuple[Optional[plt.Figure], list[str]]:
    """
    Create equity curve plots and optionally full dashboards for best pairs.

    Args:
        df: Results DataFrame
        scanner: PairScanner instance (to re-run backtests for PnL curves)
        save_path: Optional path to save the summary figure
        generate_dashboards: If True, also generate full dashboard plots
        output_dir: Directory for dashboard files (required if generate_dashboards=True)

    Returns:
        Tuple of (summary Figure, list of dashboard paths)
    """
    if df.empty:
        print("No results to plot.")
        return None, []

    best_pairs = _get_unique_best_pairs(df)
    dashboard_paths = []

    if generate_dashboards and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run backtests and optionally generate dashboards
    pair_results = {}
    for row, label in best_pairs:
        sym0, sym1 = row["symbol0"], row["symbol1"]
        try:
            results, metrics = _run_pair_backtest(scanner, sym0, sym1)
            pair_results[row["pair"]] = (results, metrics, row)

            if generate_dashboards and output_dir:
                print(f"Generating dashboard for {row['pair']} ({label})...")
                visualizer = Visualizer(results, metrics, sym0, sym1, scanner.strategy_config)
                dashboard_path = os.path.join(output_dir, f"dashboard_{sym0}_{sym1}.png")
                visualizer.create_dashboard(save_path=dashboard_path)
                dashboard_paths.append(dashboard_path)
                plt.close()

        except Exception as e:
            print(f"  Warning: Failed to process {row['pair']}: {e}")

    # Create summary plot
    if len(pair_results) == 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        pair_name = list(pair_results.keys())[0]
        results, _, row = pair_results[pair_name]
        _plot_pnl_curve(ax, results)
        ax.set_title(f"Best Pair: {pair_name} (Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']*100:.1f}%)")
    elif len(pair_results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, (row, label) in enumerate(best_pairs[:2]):
            if row["pair"] in pair_results:
                results, _, row_data = pair_results[row["pair"]]
                _plot_pnl_curve(axes[i], results)
                if label == "best_sharpe":
                    axes[i].set_title(f"Best by Sharpe: {row['pair']} (Sharpe: {row_data['sharpe_ratio']:.2f})")
                else:
                    axes[i].set_title(f"Best by PnL: {row['pair']} (Return: {row_data['total_return']*100:.1f}%)")
    else:
        return None, dashboard_paths

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Summary plot saved to: {save_path}")

    return fig, dashboard_paths


def _plot_pnl_curve(ax: plt.Axes, results: pd.DataFrame) -> None:
    """Plot PnL curve from strategy results."""
    cumulative = (1 + results["net_return"]).cumprod()

    ax.plot(results.index, cumulative, linewidth=1.5)
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax.fill_between(
        results.index,
        1,
        cumulative,
        where=cumulative >= 1,
        alpha=0.3,
        color="green",
    )
    ax.fill_between(
        results.index,
        1,
        cumulative,
        where=cumulative < 1,
        alpha=0.3,
        color="red",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)


def print_scan_results(df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print formatted scan results to console.

    Args:
        df: Results DataFrame
        top_n: Number of top results to show
    """
    if df.empty:
        print("No results to display.")
        return

    total_pairs = len(df)
    successful = len(df[df["sharpe_ratio"].notna()])

    print()
    print("=" * 80)
    print("                        PAIR TRADING SCAN RESULTS")
    print("=" * 80)
    print(f"Scanned {total_pairs} pairs | Successful: {successful}")
    print()

    # Top by Sharpe Ratio
    print(f"TOP {top_n} BY SHARPE RATIO (Risk-Adjusted Performance)")
    print("-" * 80)
    print(format_results_table(df, "sharpe_ratio", top_n))
    print()

    # Top by Total Return
    print(f"TOP {top_n} BY TOTAL RETURN (Absolute PnL)")
    print("-" * 80)
    print(format_results_table(df, "total_return", top_n))
    print()


def main():
    """Main entry point for the pair scanner."""
    parser = argparse.ArgumentParser(
        description="Scan all pair combinations for cointegration trading opportunities"
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        help="Symbols to scan (e.g., BTC ETH SOL AVAX)"
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top results to display (default: 10)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (e.g., results/scan.csv)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip data download (use cached data only)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--no-auto-fetch", action="store_true",
        help="Disable auto-fetching symbols from Binance (use hardcoded list)"
    )

    args = parser.parse_args()

    # Initialize scanner
    scanner = PairScanner(args.config)

    # Run scan
    results_df = scanner.scan(
        symbols=args.symbols,
        download=not args.no_download,
        auto_fetch=not args.no_auto_fetch,
    )

    if results_df.empty:
        print("No valid results. Check that data is available for the requested symbols.")
        return

    # Print results
    print_scan_results(results_df, top_n=args.top)

    # Save results
    csv_path, feather_path = scanner.save_results(results_df, args.output)
    print(f"Results saved to: {csv_path}")
    print(f"Results saved to: {feather_path}")

    # Generate plots
    if not args.no_plots:
        output_dir = scanner.config.get("output_dir", "output")
        plot_path = os.path.join(output_dir, "pair_scan_summary.png")

        _, dashboard_paths = plot_scan_summary(
            results_df,
            scanner,
            save_path=plot_path,
            generate_dashboards=True,
            output_dir=output_dir,
        )
        if dashboard_paths:
            print(f"Generated {len(dashboard_paths)} dashboard(s)")

        plt.show()


if __name__ == "__main__":
    main()
