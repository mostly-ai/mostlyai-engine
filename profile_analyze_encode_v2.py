# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Profiling script for analyze and encode operations with mixed-type tabular data.
Saves detailed profiling results to files for analysis.
"""

import cProfile
import io
import logging
import pstats
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mostlyai.engine import analyze, encode, init_logging, set_random_state, split
from mostlyai.engine.domain import ModelEncodingType

# Setup logging
init_logging()
_LOG = logging.getLogger(__name__)

# Set random seed for reproducibility
set_random_state(random_state=42)


def generate_dummy_data(n_rows: int = 100_000, missing_rate: float = 0.15) -> pd.DataFrame:
    """Generate large dummy dataset with mixed types and missing values."""
    _LOG.info(f"Generating {n_rows:,} rows of dummy data with {missing_rate:.0%} missing rate...")

    rng = np.random.default_rng(42)

    # Helper to add missing values
    def add_missing(series: pd.Series, rate: float = missing_rate) -> pd.Series:
        mask = rng.random(len(series)) < rate
        series = series.copy()
        series[mask] = None
        return series

    data = {
        # Categorical columns
        "cat_low_card": add_missing(pd.Series(rng.choice(["A", "B", "C", "D", "E"], size=n_rows))),
        "cat_medium_card": add_missing(pd.Series(rng.choice([f"Category_{i}" for i in range(100)], size=n_rows))),
        "cat_high_card": add_missing(pd.Series([f"ID_{i}" for i in rng.integers(0, 10000, size=n_rows)])),
        # Numeric columns
        "num_int": add_missing(pd.Series(rng.integers(0, 1000, size=n_rows))),
        "num_float": add_missing(pd.Series(rng.normal(100, 25, size=n_rows))),
        "num_discrete": add_missing(pd.Series(rng.choice([1, 2, 3, 5, 10, 20], size=n_rows))),
        "num_skewed": add_missing(pd.Series(rng.exponential(scale=50, size=n_rows))),
        "num_binary": add_missing(pd.Series(rng.choice([0, 1], size=n_rows))),
        # Datetime columns
        "date_recent": add_missing(pd.Series(pd.date_range("2020-01-01", "2024-12-31", periods=n_rows))),
        "date_wide_range": add_missing(pd.Series(pd.date_range("1990-01-01", "2024-12-31", periods=n_rows))),
        # Character/String columns
        "char_short": add_missing(
            pd.Series(["".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=5)) for _ in range(n_rows)])
        ),
        "char_medium": add_missing(
            pd.Series(
                ["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=20)) for _ in range(n_rows)]
            )
        ),
        # Lat/Long
        "latitude": add_missing(pd.Series(rng.uniform(-90, 90, size=n_rows))),
        "longitude": add_missing(pd.Series(rng.uniform(-180, 180, size=n_rows))),
    }

    df = pd.DataFrame(data)
    _LOG.info(f"Generated DataFrame: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    _LOG.info(f"Missing values per column:\n{df.isnull().sum()}")

    return df


def profile_and_time(func, name, output_file, *args, **kwargs):
    """Profile a function, time it, and save results."""
    _LOG.info(f"\n{'=' * 80}")
    _LOG.info(f"PROFILING {name}")
    _LOG.info(f"{'=' * 80}")

    profiler = cProfile.Profile()

    t0 = time.time()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    elapsed = time.time() - t0

    _LOG.info(f"✓ {name} completed in {elapsed:.2f}s")

    # Save detailed profile
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    # Write to file
    with open(output_file, "w") as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"{name} PROFILING RESULTS\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Total time: {elapsed:.2f}s\n\n")

        f.write(f"\n{'=' * 80}\n")
        f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
        f.write(f"{'=' * 80}\n")
        stream = io.StringIO()
        stats_obj = pstats.Stats(profiler, stream=stream)
        stats_obj.strip_dirs()
        stats_obj.sort_stats("cumulative")
        stats_obj.print_stats(50)
        f.write(stream.getvalue())

        f.write(f"\n{'=' * 80}\n")
        f.write("TOP 30 FUNCTIONS BY TOTAL (SELF) TIME\n")
        f.write(f"{'=' * 80}\n")
        stream = io.StringIO()
        stats_obj = pstats.Stats(profiler, stream=stream)
        stats_obj.strip_dirs()
        stats_obj.sort_stats("tottime")
        stats_obj.print_stats(30)
        f.write(stream.getvalue())

        f.write(f"\n{'=' * 80}\n")
        f.write("CALLERS OF SLOWEST FUNCTIONS\n")
        f.write(f"{'=' * 80}\n")
        stream = io.StringIO()
        stats_obj = pstats.Stats(profiler, stream=stream)
        stats_obj.strip_dirs()
        stats_obj.sort_stats("tottime")
        stats_obj.print_callers(20)
        f.write(stream.getvalue())

    _LOG.info(f"Detailed profile saved to: {output_file}")

    return result, elapsed, stats


def print_summary_stats(stats, name):
    """Print summary statistics."""
    print(f"\n{'=' * 80}")
    print(f"{name} - TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print(f"{'=' * 80}")
    stream = io.StringIO()
    stats.stream = stream
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())


def main():
    """Main profiling workflow."""
    print("=" * 80)
    print("PROFILING ANALYZE AND ENCODE OPERATIONS")
    print("=" * 80)

    # Generate dummy data
    n_rows = 100_000  # Large partition
    df = generate_dummy_data(n_rows=n_rows)

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_dir = Path(tmpdir) / "profile-ws"
        workspace_dir.mkdir(exist_ok=True)

        # Output directory for profiles
        output_dir = Path.cwd() / "profiling_results"
        output_dir.mkdir(exist_ok=True)

        _LOG.info(f"\nWorkspace directory: {workspace_dir}")
        _LOG.info(f"Output directory: {output_dir}")

        # Define encoding types for mixed data
        encoding_types = {
            "cat_low_card": ModelEncodingType.tabular_categorical,
            "cat_medium_card": ModelEncodingType.tabular_categorical,
            "cat_high_card": ModelEncodingType.tabular_categorical,
            "num_int": ModelEncodingType.tabular_numeric_discrete,
            "num_float": ModelEncodingType.tabular_numeric_binned,
            "num_discrete": ModelEncodingType.tabular_numeric_discrete,
            "num_skewed": ModelEncodingType.tabular_numeric_binned,
            "num_binary": ModelEncodingType.tabular_categorical,
            "date_recent": ModelEncodingType.tabular_datetime,
            "date_wide_range": ModelEncodingType.tabular_datetime,
            "char_short": ModelEncodingType.tabular_character,
            "char_medium": ModelEncodingType.tabular_character,
            "latitude": ModelEncodingType.tabular_numeric_binned,
            "longitude": ModelEncodingType.tabular_numeric_binned,
        }

        # Split data
        _LOG.info("\n" + "=" * 80)
        _LOG.info("SPLITTING DATA")
        _LOG.info("=" * 80)

        split(
            tgt_data=df,
            tgt_encoding_types=encoding_types,
            workspace_dir=workspace_dir,
            trn_val_split=0.8,
        )
        _LOG.info("✓ Split completed")

        # Patch to force n_jobs=1
        from mostlyai.engine import analysis
        from mostlyai.engine._tabular import encoding as tabular_encoding

        original_analyze_partition = analysis._analyze_partition
        original_encode_partition = tabular_encoding._encode_partition

        def patched_analyze_partition(*args, **kwargs):
            kwargs["n_jobs"] = 1
            return original_analyze_partition(*args, **kwargs)

        def patched_encode_partition(*args, **kwargs):
            kwargs["n_jobs"] = 1
            return original_encode_partition(*args, **kwargs)

        analysis._analyze_partition = patched_analyze_partition
        tabular_encoding._encode_partition = patched_encode_partition

        # Profile ANALYZE
        _, analyze_time, analyze_stats = profile_and_time(
            analyze,
            "ANALYZE",
            output_dir / "analyze_profile.txt",
            value_protection=False,
            differential_privacy=None,
            workspace_dir=workspace_dir,
        )

        # Profile ENCODE
        _, encode_time, encode_stats = profile_and_time(
            encode,
            "ENCODE",
            output_dir / "encode_profile.txt",
            workspace_dir=workspace_dir,
        )

        # Restore original functions
        analysis._analyze_partition = original_analyze_partition
        tabular_encoding._encode_partition = original_encode_partition

        # Print summaries
        print_summary_stats(analyze_stats, "ANALYZE")
        print_summary_stats(encode_stats, "ENCODE")

        # Print overall summary
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Dataset size: {n_rows:,} rows x {len(df.columns)} columns")
        print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\nTiming Results:")
        print(f"  ANALYZE: {analyze_time:.2f}s")
        print(f"  ENCODE:  {encode_time:.2f}s")
        print(f"  TOTAL:   {analyze_time + encode_time:.2f}s")
        print("\nThroughput:")
        print(f"  ANALYZE: {n_rows / analyze_time:,.0f} rows/sec")
        print(f"  ENCODE:  {n_rows / encode_time:,.0f} rows/sec")
        print(f"\nDetailed profiles saved to: {output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
