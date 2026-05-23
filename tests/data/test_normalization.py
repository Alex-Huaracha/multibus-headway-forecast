"""Tests for src.data.normalization — AC-NORM-1..5, AC-LEAK-1.

Strict TDD: this file is the RED commit. src/data/normalization.py does not exist yet;
all tests fail with ImportError. Run: uv run pytest tests/data/test_normalization.py -q
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.fixtures.headways_factory import make_split_fixture


class TestComputeStats:
    """AC-NORM-1: compute_normalization_stats uses train rows only.
    AC-LEAK-1: leakage guard — train mean != full-frame mean.
    """

    def test_stats_per_direction_from_train_only(self) -> None:
        """AC-NORM-1: stats dict has keys for each (empresaid, direction) in train_df.

        Two directions with different train means. Stats must reflect only train values.
        """
        from src.data.normalization import NormalizationStats, compute_normalization_stats

        df = make_split_fixture(
            empresaid=2,
            train_means={-1: 2.0, 1: 3.0},
            test_means={-1: 10.0, 1: 12.0},
        )
        train_df = df.filter(pl.col("split") == "train")
        stats = compute_normalization_stats(train_df)

        assert isinstance(stats, NormalizationStats)
        assert (2, -1) in stats.means
        assert (2, 1) in stats.means
        # Means must match train values (all rows have same value per direction)
        assert abs(stats.means[(2, -1)] - 2.0) < 1e-9
        assert abs(stats.means[(2, 1)] - 3.0) < 1e-9

    def test_stats_ignore_null_delta(self) -> None:
        """AC-NORM-1: null delta_t_min rows do not contribute to mean/std.

        Train frame has some null values; mean must be computed over non-null only.
        """
        from src.data.normalization import compute_normalization_stats

        df = pl.DataFrame({
            "empresaid": [2, 2, 2, 2],
            "direction": [-1, -1, -1, -1],
            "pair_rank": [0, 0, 0, 0],
            "delta_t_min": [2.0, 4.0, None, 6.0],
            "split": ["train"] * 4,
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        stats = compute_normalization_stats(df)
        # mean of [2.0, 4.0, 6.0] = 4.0 (null excluded)
        assert abs(stats.means[(2, -1)] - 4.0) < 1e-9

    def test_leakage_train_vs_full_diverge(self) -> None:
        """AC-LEAK-1: normalization leakage guard.

        train mean for direction=-1 = 2.0.
        Full frame includes test rows with delta=10.0, shifting the full mean.
        compute_normalization_stats(train_df).means != compute_normalization_stats(full_df).means
        """
        from src.data.normalization import compute_normalization_stats

        df = make_split_fixture(
            empresaid=2,
            train_means={-1: 2.0, 1: 3.0},
            test_means={-1: 10.0, 1: 12.0},
        )
        train_df = df.filter(pl.col("split") == "train")
        stats_train = compute_normalization_stats(train_df)
        stats_full = compute_normalization_stats(df)

        # train mean is 2.0; full-frame mean includes test rows (10.0) → higher
        assert stats_train.means[(2, -1)] != stats_full.means[(2, -1)]
        assert stats_train.means[(2, -1)] < stats_full.means[(2, -1)]

    def test_no_torch_import(self) -> None:
        """AC-NORM-5: normalization module has zero torch imports."""
        import inspect
        import src.data.normalization as nmod

        source = inspect.getsource(nmod)
        assert "import torch" not in source
        assert "from torch" not in source


class TestApplyZscore:
    """AC-NORM-2..4: apply_zscore correctness, null passthrough, no clipping."""

    def _make_known_df(self) -> pl.DataFrame:
        """Fixture: one direction, delta_t_min values [1.0, 2.0, 3.0, 4.0, 5.0].

        Mean = 3.0, std = sqrt(2) ≈ 1.4142.
        z-score of 3.0 → (3-3)/std = 0.0
        z-score of 5.0 → (5-3)/std ≈ 1.414
        """
        return pl.DataFrame({
            "empresaid": [2] * 5,
            "direction": [-1] * 5,
            "pair_rank": [0] * 5,
            "delta_t_min": [1.0, 2.0, 3.0, 4.0, 5.0],
            "split": ["train"] * 5,
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )

    def test_zscore_value_for_known_inputs(self) -> None:
        """AC-NORM-2: z-score = (x - mean) / (std + Z_EPS) per direction.

        For values [1..5], mean=3, std=sqrt(2). z(3) ≈ 0.0, z(5) ≈ 1.414.
        """
        from src.data.normalization import Z_EPS, apply_zscore, compute_normalization_stats

        import math
        df = self._make_known_df()
        stats = compute_normalization_stats(df)
        result = apply_zscore(df, stats)

        assert "delta_t_min_z" in result.columns
        z_values = result["delta_t_min_z"].to_list()

        mean_val = stats.means[(2, -1)]
        std_val = stats.stds[(2, -1)]

        expected_z3 = (3.0 - mean_val) / (std_val + Z_EPS)
        expected_z5 = (5.0 - mean_val) / (std_val + Z_EPS)

        assert abs(z_values[2] - expected_z3) < 1e-6  # value at index 2 is 3.0
        assert abs(z_values[4] - expected_z5) < 1e-6  # value at index 4 is 5.0

    def test_null_delta_remains_null_after_zscore(self) -> None:
        """AC-NORM-3: null delta_t_min stays null in the z-scored output column.

        Design note: no imputation. Null means the slot was empty at that timestep.
        """
        from src.data.normalization import apply_zscore, compute_normalization_stats

        df = pl.DataFrame({
            "empresaid": [2, 2, 2],
            "direction": [-1, -1, -1],
            "pair_rank": [0, 0, 0],
            "delta_t_min": [2.0, None, 4.0],
            "split": ["train"] * 3,
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        # Compute stats from non-null only
        stats = compute_normalization_stats(df.filter(pl.col("delta_t_min").is_not_null()))
        result = apply_zscore(df, stats)

        z_col = result["delta_t_min_z"]
        assert z_col[1] is None  # null row must remain null

    def test_no_clipping_pass_through(self) -> None:
        """AC-NORM-4 + DL-8: z-scored values with |z| > 5 are NOT clipped.

        A test row with delta_t_min far outside the training distribution
        should produce a z-score > 5 without being clipped.
        """
        from src.data.normalization import apply_zscore, compute_normalization_stats

        # Train: tight distribution around 2.0 (very small std)
        train_df = pl.DataFrame({
            "empresaid": [2, 2, 2],
            "direction": [-1, -1, -1],
            "pair_rank": [0, 0, 0],
            "delta_t_min": [1.9, 2.0, 2.1],
            "split": ["train"] * 3,
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        stats = compute_normalization_stats(train_df)

        # Val row with a huge outlier.
        val_df = pl.DataFrame({
            "empresaid": [2],
            "direction": [-1],
            "pair_rank": [0],
            "delta_t_min": [100.0],  # extreme outlier → |z| >> 5
            "split": ["val"],
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        result = apply_zscore(val_df, stats)
        z_outlier = result["delta_t_min_z"][0]
        # Must NOT be clipped — value should be >> 5
        assert z_outlier is not None
        assert z_outlier > 5.0
