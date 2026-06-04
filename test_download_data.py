"""Tests for Binance data download caching behavior."""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd

from download_data import (
    download_symbols,
    feather_file_is_usable,
    get_download_cache_path,
    update_symbol_data,
)


class TestDownloadCache(unittest.TestCase):
    """Tests for download cache validation."""

    def _config(self, tmpdir: str) -> dict:
        return {
            "quote": "USDT",
            "interval": "1d",
            "feather_dir": os.path.join(tmpdir, "feather"),
            "download_cache_file": os.path.join(tmpdir, "download_cache.json"),
            "download_grace_hours": 12,
            "exclude_incomplete_candles": True,
        }

    def _write_fresh_cache(self, path: str, key: str) -> None:
        with open(path, "w") as f:
            json.dump({key: {"last_download": time.time()}}, f)

    def test_grace_period_skip_requires_existing_feather(self):
        """A fresh cache entry should not skip a missing feather file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            cache_key = "BTCUSDT_1d"
            self._write_fresh_cache(config["download_cache_file"], cache_key)

            with patch("download_data.update_symbol_data", return_value=pd.DataFrame()) as mock_update:
                successful = download_symbols(config, ["BTC"], full_symbol_names=False)

            self.assertEqual(successful, [])
            mock_update.assert_called_once()

    def test_grace_period_skips_when_feather_is_usable(self):
        """A fresh cache entry may skip only when the feather has data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            os.makedirs(config["feather_dir"], exist_ok=True)
            out_path = os.path.join(config["feather_dir"], "BTCUSDT_1d.feather")
            pd.DataFrame({
                "open_time": [1],
                "open_time_dt": [pd.Timestamp("2020-01-01")],
                "close": [1.0],
            }).to_feather(out_path)
            self._write_fresh_cache(config["download_cache_file"], "BTCUSDT_1d")

            with patch("download_data.update_symbol_data") as mock_update:
                successful = download_symbols(config, ["BTC"], full_symbol_names=False)

            self.assertEqual(successful, ["BTC"])
            mock_update.assert_not_called()
            self.assertTrue(feather_file_is_usable(out_path))

    def test_cache_path_handles_bare_feather_dir(self):
        """Bare feather_dir values should place the cache in the current directory."""
        path = get_download_cache_path({}, "feather")

        self.assertEqual(path, os.path.join(".", "download_cache.json"))

    def test_update_symbol_data_excludes_incomplete_current_candle(self):
        """The fetch end_time should stop before the current interval open."""
        captured = {}

        def fake_fetch_klines(base_url, symbol, interval, start_time, end_time, limit):
            captured["end_time"] = end_time
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "BTCUSDT_1d.feather")
            now = pd.Timestamp("2020-01-02 12:00:00", tz="UTC").timestamp()
            current_open = int(pd.Timestamp("2020-01-02", tz="UTC").timestamp() * 1000)

            with patch("download_data.time.time", return_value=now):
                with patch("download_data.fetch_klines", side_effect=fake_fetch_klines):
                    update_symbol_data(
                        "BTCUSDT",
                        "1d",
                        out_path,
                        "https://example.test",
                        1500,
                        start_date="2020-01-01",
                    )

        self.assertEqual(captured["end_time"], current_open - 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
