"""
Multi-Pair Crypto Scanner

Scans all pair combinations from a list of cryptocurrencies using the
existing pair trading strategy and backtest infrastructure.
"""

import argparse
import itertools
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import DataLoader, MetricsCalculator, PerformanceMetrics, Visualizer
from download_data import download_symbols
from pair_trading import PairTradingStrategy, StrategyConfig
from utils import load_config


def _strategy_config_to_dict(config: StrategyConfig) -> dict:
    """Convert StrategyConfig to a picklable dict for multiprocessing."""
    return {
        "window": config.window,
        "kpss_threshold": config.kpss_threshold,
        "entry_threshold": config.entry_threshold,
        "stop_loss_threshold": config.stop_loss_threshold,
        "transaction_fee": config.transaction_fee,
        "unbiased_formulation": config.unbiased_formulation,
        "beta_loading": config.beta_loading,
        "beta_proxy_symbol": config.beta_proxy_symbol,
        "position_sizing_mode": config.position_sizing_mode,
        "stop_loss_cooldown_bars": config.stop_loss_cooldown_bars,
        "exit_on_decointegration": config.exit_on_decointegration,
        "min_fair_value": config.min_fair_value,
        "divergence_mode": config.divergence_mode,
    }


def _load_market_prices(
    data_loader: DataLoader,
    strategy_config: StrategyConfig,
) -> Optional[pd.Series]:
    """Load beta proxy prices when beta loading is enabled."""
    if strategy_config.beta_loading and strategy_config.beta_proxy_symbol:
        return data_loader.load_prices(strategy_config.beta_proxy_symbol)
    return None


def _metrics_to_scan_result(
    sym0: str,
    sym1: str,
    metrics: PerformanceMetrics,
    data_points: int,
) -> dict:
    """Convert metrics into the scanner row format."""
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
        "daily_win_rate": metrics.daily_win_rate,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
        "signal_changes": metrics.signal_changes,
        "entries": metrics.entries,
        "exits": metrics.exits,
        "round_trips": metrics.round_trips,
        "calmar_ratio": metrics.calmar_ratio,
        "valid_metrics": metrics.valid_metrics,
        "ruin_event": metrics.ruin_event,
        "data_points": data_points,
    }


def _error_result(sym0: str, sym1: str, error: Exception, debug_errors: bool = False) -> dict:
    """Return a structured scanner failure."""
    result = {
        "pair": f"{sym0}/{sym1}",
        "symbol0": sym0,
        "symbol1": sym1,
        "error": f"{type(error).__name__}: {error}",
    }
    if debug_errors:
        result["traceback"] = traceback.format_exc()
    return result


def _sort_results(df: pd.DataFrame, sort_by: str, top_n: Optional[int] = None) -> pd.DataFrame:
    """Sort valid metric rows ahead of invalid rows for stable rankings."""
    if df.empty:
        return df

    ascending = False
    sorted_df = df.copy()
    valid = sorted_df.get("valid_metrics", pd.Series(True, index=sorted_df.index)).fillna(False)
    values = pd.to_numeric(sorted_df[sort_by], errors="coerce")
    sorted_df["_valid_sort"] = valid & values.notna()

    sorted_df = sorted_df.sort_values(
        ["_valid_sort", sort_by],
        ascending=[False, ascending],
        na_position="last",
    ).drop(columns=["_valid_sort"])

    return sorted_df.head(top_n) if top_n is not None else sorted_df


def _format_metric(value: float, suffix: str = "", signed: bool = False) -> str:
    """Format finite and non-finite metric values for tables."""
    if pd.isna(value):
        return "n/a"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    sign = "+" if signed else ""
    return f"{value:{sign}.2f}{suffix}"


def _scan_single_pair_worker(args: tuple) -> Optional[dict]:
    """
    Worker function to scan a single pair in a subprocess.

    Args:
        args: Tuple of (sym0, sym1, config_dict, strategy_config_dict, debug_errors)

    Returns:
        Dictionary with pair metrics or None on failure
    """
    sym0, sym1, config_dict, strategy_config_dict, debug_errors = args

    # Set matplotlib backend to avoid GUI issues in subprocess
    matplotlib.use("Agg")

    try:
        # Create own instances (not shared across processes)
        data_loader = DataLoader(config_dict)
        strategy_config = StrategyConfig(**strategy_config_dict)

        # Load data
        prices0, prices1 = data_loader.load_pair(sym0, sym1)
        market_prices = _load_market_prices(data_loader, strategy_config)

        # Check minimum data requirement
        min_required = strategy_config.window + 10
        if len(prices0) < min_required:
            return None

        # Run strategy
        strategy = PairTradingStrategy(strategy_config)
        results = strategy.run(prices0, prices1, market_prices)

        # Calculate metrics
        metrics = MetricsCalculator.calculate(results)

        return _metrics_to_scan_result(sym0, sym1, metrics, len(results))

    except Exception as e:
        return _error_result(sym0, sym1, e, debug_errors)


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
        self.scan_config = self.config.get("scan", {})
        self.debug_errors = self.scan_config.get("debug_errors", False)
        self.fail_fast = self.scan_config.get("fail_fast", False)

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

    def generate_pairs(
        self,
        symbols: list[str],
        bidirectional: Optional[bool] = None,
    ) -> list[tuple[str, str]]:
        """
        Generate all unique pairs from the symbol list.

        Args:
            symbols: List of symbol names
            bidirectional: If True, include both A/B and B/A

        Returns:
            List of (symbol0, symbol1) tuples
        """
        if bidirectional is None:
            bidirectional = self.scan_config.get("scan_bidirectional", False)

        pairs = list(itertools.combinations(symbols, 2))
        if bidirectional:
            pairs.extend((sym1, sym0) for sym0, sym1 in list(pairs))
        return pairs

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
            market_prices = _load_market_prices(self.data_loader, self.strategy_config)

            # Check minimum data requirement
            min_required = self.strategy_config.window + 10
            if len(prices0) < min_required:
                print(f"  Warning: {sym0}/{sym1} has insufficient data ({len(prices0)} rows)")
                return None

            # Run strategy
            strategy = PairTradingStrategy(self.strategy_config)
            results = strategy.run(prices0, prices1, market_prices)

            # Calculate metrics
            metrics = MetricsCalculator.calculate(results)

            return _metrics_to_scan_result(sym0, sym1, metrics, len(results))

        except FileNotFoundError as e:
            print(f"  Warning: Data not found for {sym0}/{sym1}: {e}")
            if getattr(self, "fail_fast", False):
                raise
            return None
        except ValueError as e:
            print(f"  Warning: Error processing {sym0}/{sym1}: {e}")
            if getattr(self, "fail_fast", False):
                raise
            return None
        except Exception as e:
            print(f"  Warning: Unexpected error for {sym0}/{sym1}: {e}")
            if getattr(self, "debug_errors", False):
                print(traceback.format_exc())
            if getattr(self, "fail_fast", False):
                raise
            return None

    def _scan_sequential(self, pairs: list[tuple[str, str]]) -> tuple[list[dict], int, int]:
        """
        Run backtest on pairs sequentially.

        Args:
            pairs: List of (symbol0, symbol1) tuples

        Returns:
            Tuple of (results list, successful count, failed count)
        """
        results = []
        successful = 0
        failed = 0

        for i, (sym0, sym1) in enumerate(pairs):
            print(f"[{i+1}/{len(pairs)}] Testing {sym0}/{sym1}...", end=" ")
            result = self.run_single_pair(sym0, sym1)

            if result is not None:
                results.append(result)
                print(f"Sharpe: {result['sharpe_ratio']:.2f}, Return: {result['total_return']*100:.1f}%")
                successful += 1
            else:
                failed += 1

        return results, successful, failed

    def _scan_parallel(self, pairs: list[tuple[str, str]], workers: int) -> tuple[list[dict], int, int]:
        """
        Run backtest on pairs in parallel using multiprocessing.

        Args:
            pairs: List of (symbol0, symbol1) tuples
            workers: Number of parallel workers

        Returns:
            Tuple of (results list, successful count, failed count)
        """
        results = []
        successful = 0
        failed = 0

        # Prepare picklable arguments
        strategy_config_dict = _strategy_config_to_dict(self.strategy_config)
        work_items = [
            (sym0, sym1, self.config, strategy_config_dict, self.debug_errors)
            for sym0, sym1 in pairs
        ]

        print(f"Running parallel scan with {workers} workers...")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all work items
            future_to_pair = {
                executor.submit(_scan_single_pair_worker, item): (item[0], item[1])
                for item in work_items
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_pair):
                completed += 1
                sym0, sym1 = future_to_pair[future]

                try:
                    result = future.result()
                    if result is not None and "error" not in result:
                        results.append(result)
                        successful += 1
                    else:
                        if result is not None and "error" in result:
                            print(f"  Warning: {result['pair']} failed: {result['error']}")
                            if self.debug_errors and result.get("traceback"):
                                print(result["traceback"])
                            if self.fail_fast:
                                raise RuntimeError(result["error"])
                        failed += 1
                except Exception as e:
                    if self.fail_fast:
                        raise
                    print(f"  Warning: Unexpected worker failure for {sym0}/{sym1}: {e}")
                    failed += 1

                # Report progress every 10 pairs
                if completed % 10 == 0 or completed == len(pairs):
                    print(f"Progress: {completed}/{len(pairs)} pairs processed...")

        return results, successful, failed

    def scan(
        self,
        symbols: Optional[list[str]] = None,
        download: bool = True,
        auto_fetch: bool = True,
        min_trades: int = 60,
        workers: int = 4,
    ) -> pd.DataFrame:
        """
        Run full scan on all pairs.

        Args:
            symbols: List of symbols to scan (defaults to config scan_symbols or TOP_20)
            download: If True, download data before scanning
            auto_fetch: If True and no symbols specified, fetch from Binance API
            min_trades: Minimum number of signal changes required (default: 60)
            workers: Number of parallel workers (1 for sequential)

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
        print(f"Generated {len(pairs)} pair tests")

        # Run backtest on each pair (parallel or sequential)
        if workers > 1:
            self.results, successful, failed = self._scan_parallel(pairs, workers)
        else:
            self.results, successful, failed = self._scan_sequential(pairs)

        print(f"\nCompleted: {successful} successful, {failed} failed")

        # Convert to DataFrame
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Filter by minimum trades
        if min_trades > 0:
            before_count = len(df)
            df = df[df["total_trades"] >= min_trades]
            filtered_count = before_count - len(df)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} pairs with fewer than {min_trades} trades")

        return _sort_results(df, "sharpe_ratio")

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
    sorted_df = _sort_results(df, sort_by, top_n)

    lines = []
    header = f"{'Rank':<6}{'Pair':<14}{'Total Ret':>12}{'Ann. Ret':>12}{'Sharpe':>10}{'Sortino':>10}{'Max DD':>12}{'Trades':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        total_ret = _format_metric(row['total_return'] * 100, "%", signed=True)
        ann_ret = _format_metric(row['annualized_return'] * 100, "%", signed=True)
        sharpe = _format_metric(row['sharpe_ratio'])
        sortino = _format_metric(row['sortino_ratio'])
        max_dd = _format_metric(row['max_drawdown'] * 100, "%")
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
    best_sharpe = _sort_results(df, "sharpe_ratio", 1).iloc[0]
    best_return = _sort_results(df, "total_return", 1).iloc[0]

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
    market_prices = _load_market_prices(scanner.data_loader, scanner.strategy_config)
    strategy = PairTradingStrategy(scanner.strategy_config)
    results = strategy.run(prices0, prices1, market_prices)
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
    if "valid_metrics" in df.columns:
        successful = int(df["valid_metrics"].fillna(False).sum())
    else:
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
    parser.add_argument(
        "--min-trades", type=int, default=None,
        help="Minimum number of trades required (default: from config or 60)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4, use 1 for sequential)"
    )
    parser.add_argument(
        "--bidirectional", action="store_true",
        help="Scan both A/B and B/A directions"
    )
    parser.add_argument(
        "--debug-errors", action="store_true",
        help="Print tracebacks for failed pair scans"
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop scanning on the first pair-processing error"
    )

    args = parser.parse_args()

    # Initialize scanner
    scanner = PairScanner(args.config)
    if args.bidirectional:
        scanner.scan_config["scan_bidirectional"] = True
    if args.debug_errors:
        scanner.debug_errors = True
    if args.fail_fast:
        scanner.fail_fast = True

    # Get min_trades from CLI or config
    min_trades = args.min_trades
    if min_trades is None:
        scan_config = scanner.config.get("scan", {})
        min_trades = scan_config.get("min_trades", 60)

    # Run scan
    results_df = scanner.scan(
        symbols=args.symbols,
        download=not args.no_download,
        auto_fetch=not args.no_auto_fetch,
        min_trades=min_trades,
        workers=args.workers,
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
