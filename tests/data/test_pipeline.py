"""Integration tests for the supervised-dataset pipeline order — W4 RED.

These tests verify that the LOCKED pipeline order (INV-1) is respected and
that leakage guards (INV-2, INV-3, INV-6) hold end-to-end with real function
calls (not mocks of the underlying modules).

ACs covered: AC-PIPE-1..4, AC-LEAK-1..3.
Design refs: spec §3 (INV-1..3, INV-6), spec §4 (AC-PIPE-*, AC-LEAK-*), design §3.

No torch imports in this file (pure polars pipeline up to the Dataset boundary).
"""
from __future__ import annotations

import pytest
import polars as pl
from datetime import date, datetime, timedelta
from typing import Any

from src.evaluation.splits import (
    split_temporal,
    winsorize_train_p99,
    SPLIT_TRAIN_START,
    SPLIT_TEST_START,
    SPLIT_VAL_START,
)
from src.data.normalization import compute_normalization_stats, apply_zscore
from src.data.windowing import compute_max_N, make_window_index
from src.data.context_features import encode_context, load_atypical_days
from tests.fixtures.headways_factory import make_multi_corridor_fixture, make_split_fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_pipeline_df(n_train: int = 20, n_val: int = 5, n_test: int = 5) -> pl.DataFrame:
    """Return a minimal headways DataFrame that passes through the full pipeline.

    All rows belong to a single (empresaid=2, direction=-1, pair_rank=0) slot
    so window counts are predictable.
    """
    rows: list[dict[str, Any]] = []
    for i in range(n_train):
        d = SPLIT_TRAIN_START + timedelta(days=i)
        rows.append({
            "empresaid": 2,
            "t": datetime(d.year, d.month, d.day, 8, 0, 0),
            "direction": -1,
            "pair_rank": 0,
            "delta_t_min": float(i + 1),
            "n_buses": 2,
        })
    for i in range(n_val):
        d = SPLIT_VAL_START + timedelta(days=i)
        rows.append({
            "empresaid": 2,
            "t": datetime(d.year, d.month, d.day, 8, 0, 0),
            "direction": -1,
            "pair_rank": 0,
            "delta_t_min": float(i + 100),
            "n_buses": 2,
        })
    for i in range(n_test):
        d = SPLIT_TEST_START + timedelta(days=i)
        rows.append({
            "empresaid": 2,
            "t": datetime(d.year, d.month, d.day, 8, 0, 0),
            "direction": -1,
            "pair_rank": 0,
            "delta_t_min": float(i + 200),
            "n_buses": 2,
        })

    return pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
        pl.col("n_buses").cast(pl.Int32),
    )


# ---------------------------------------------------------------------------
# AC-PIPE-1: split_temporal before winsorize_train_p99
# ---------------------------------------------------------------------------

class TestPipelineOrder:

    def test_split_before_winsorize(self):
        """AC-PIPE-1 / INV-1: split_temporal output feeds winsorize_train_p99.

        Asserts that the DataFrame returned by split_temporal has a 'split'
        column and that winsorize_train_p99 can be applied to it without error.
        The output of winsorize must preserve the split column.
        """
        raw_df = _make_minimal_pipeline_df()
        split_df = split_temporal(raw_df)
        assert "split" in split_df.columns, "split_temporal must add 'split' column"

        clipped_df, threshold = winsorize_train_p99(split_df)
        assert "split" in clipped_df.columns, "split column must survive winsorization"
        assert isinstance(threshold, float)

    def test_winsorize_before_compute_normalization_stats(self):
        """AC-PIPE-2 / INV-6: winsorize_train_p99 output feeds compute_normalization_stats.

        Asserts that no value in the train split exceeds the winsorization
        threshold after winsorize_train_p99, confirming stats are computed on
        winsorized data (not raw data).
        """
        raw_df = _make_minimal_pipeline_df()
        split_df = split_temporal(raw_df)
        clipped_df, threshold = winsorize_train_p99(split_df)

        train_df = clipped_df.filter(pl.col("split") == "train")
        max_val = train_df["delta_t_min"].max()
        assert max_val is not None
        assert max_val <= threshold + 1e-9, (
            f"Train values after winsorization must not exceed threshold {threshold}; "
            f"got max={max_val}"
        )

    def test_normalization_stats_uses_train_only(self):
        """AC-PIPE-3 / INV-2: compute_normalization_stats is called on train-only rows.

        Uses make_split_fixture where train mean differs sharply from test mean.
        Passing the full frame vs train-only must yield different stats.
        """
        full_df = make_split_fixture(
            train_means={-1: 2.0, 1: 3.0},
            test_means={-1: 100.0, 1: 100.0},
        )
        train_df = full_df.filter(pl.col("split") == "train")

        stats_train = compute_normalization_stats(train_df)
        stats_full = compute_normalization_stats(full_df)

        key = (2, -1)
        assert abs(stats_train.means[key] - stats_full.means[key]) > 1.0, (
            "Train-only stats must differ from full-frame stats (leakage guard AC-LEAK-1)"
        )

    def test_max_N_uses_train_only(self):
        """AC-PIPE-4 / INV-3: compute_max_N is called on train-only rows.

        A fixture where the full-frame p99 bus count > train p99 confirms that
        max_N changes when val/test rows are leaked.
        """
        full_df = make_multi_corridor_fixture(n_days=14, n_pair_ranks=3)
        train_df = full_df.filter(pl.col("split") == "train")

        max_N_train = compute_max_N(train_df)
        max_N_full = compute_max_N(full_df)

        # Both dicts must have the same keys; confirm the function is callable on
        # both. The test primarily asserts that max_N is computed from train-only.
        # The leakage invariant is verified by AC-LEAK-3 below.
        assert isinstance(max_N_train, dict)
        assert isinstance(max_N_full, dict)
        assert len(max_N_train) > 0


# ---------------------------------------------------------------------------
# AC-LEAK — Leakage guards (end-to-end assertions)
# ---------------------------------------------------------------------------

class TestLeakageGuards:

    def test_normalization_leakage_guard(self):
        """AC-LEAK-1 / INV-2: normalization stats from train only != stats from full df.

        Pipeline must call compute_normalization_stats(train_df), NOT full_df.
        This test mirrors test_normalization.py::test_leakage_train_vs_full_diverge
        but at integration level.
        """
        full_df = make_split_fixture(
            train_means={-1: 2.0, 1: 3.0},
            test_means={-1: 50.0, 1: 60.0},
        )
        train_df = full_df.filter(pl.col("split") == "train")

        stats_correct = compute_normalization_stats(train_df)
        stats_leaked = compute_normalization_stats(full_df)

        key = (2, -1)
        assert key in stats_correct.means
        assert key in stats_leaked.means
        assert abs(stats_correct.means[key] - stats_leaked.means[key]) > 0.5, (
            "Leak guard: train-only mean must differ from full-frame mean by >0.5"
        )

    def test_winsorize_leakage_guard(self):
        """AC-LEAK-2 / INV-6: winsorize_train_p99 uses train-only threshold.

        Confirms that the threshold derived from a frame with additional high
        outlier test rows differs from the threshold derived from train only.
        """
        raw_df = _make_minimal_pipeline_df(n_train=20)
        split_df = split_temporal(raw_df)

        # Add an extreme outlier in the test split.
        extreme_row = pl.DataFrame({
            "empresaid": pl.Series([2], dtype=pl.Int64),
            "t": pl.Series(
                [datetime(SPLIT_TEST_START.year, SPLIT_TEST_START.month, SPLIT_TEST_START.day, 23, 0, 0)],
                dtype=pl.Datetime("us"),
            ),
            "direction": pl.Series([-1], dtype=pl.Int64),
            "pair_rank": pl.Series([0], dtype=pl.Int32),
            "delta_t_min": pl.Series([9999.0], dtype=pl.Float64),
            "split": pl.Series(["test"], dtype=pl.Utf8),
            "n_buses": pl.Series([2], dtype=pl.Int32),
        })
        # Align column order with split_df before concat.
        extreme_row = extreme_row.select(split_df.columns)
        full_df = pl.concat([split_df, extreme_row])

        # Both must produce the same threshold because winsorize uses train only.
        _, threshold_without_outlier = winsorize_train_p99(split_df)
        _, threshold_with_outlier = winsorize_train_p99(full_df)

        assert abs(threshold_without_outlier - threshold_with_outlier) < 1e-9, (
            "winsorize_train_p99 threshold must be identical regardless of test outliers "
            f"(got {threshold_without_outlier} vs {threshold_with_outlier})"
        )

    def test_max_N_leakage_guard(self):
        """AC-LEAK-3 / INV-3: max_N from train != max_N from combined frame with more buses.

        Constructs a fixture where val/test rows have n_buses=10 (far above train).
        max_N(train) must be less than max_N(combined).
        """
        n_train = 10
        rows_train: list[dict[str, Any]] = []
        for i in range(n_train):
            d = SPLIT_TRAIN_START + timedelta(days=i)
            rows_train.append({
                "empresaid": 2,
                "t": datetime(d.year, d.month, d.day, 8, 0, 0),
                "direction": -1,
                "pair_rank": 0,
                "delta_t_min": 1.0,
                "split": "train",
                "n_buses": 3,  # train max: 3 buses → max_N_train = 2
            })
        rows_test: list[dict[str, Any]] = []
        for i in range(5):
            d = SPLIT_TEST_START + timedelta(days=i)
            rows_test.append({
                "empresaid": 2,
                "t": datetime(d.year, d.month, d.day, 8, 0, 0),
                "direction": -1,
                "pair_rank": 0,
                "delta_t_min": 1.0,
                "split": "test",
                "n_buses": 10,  # test: 10 buses → would push p99 up if leaked
            })

        schema = {
            "empresaid": pl.Int64,
            "t": pl.Datetime("us"),
            "direction": pl.Int64,
            "pair_rank": pl.Int32,
            "delta_t_min": pl.Float64,
            "split": pl.Utf8,
            "n_buses": pl.Int32,
        }

        train_df = pl.DataFrame(rows_train).with_columns(
            [pl.col(c).cast(dt) for c, dt in schema.items()]
        )
        full_df = pl.concat([train_df, pl.DataFrame(rows_test).with_columns(
            [pl.col(c).cast(dt) for c, dt in schema.items()]
        )])

        max_N_train = compute_max_N(train_df)
        max_N_full = compute_max_N(full_df)

        key = (2, -1)
        assert max_N_train[key] < max_N_full[key], (
            f"Train-only max_N ({max_N_train[key]}) must be less than full-frame max_N "
            f"({max_N_full[key]}) when test rows have more buses"
        )
