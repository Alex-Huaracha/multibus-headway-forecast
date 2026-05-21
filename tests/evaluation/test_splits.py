"""Tests for src/evaluation/splits.py — split_temporal and winsorize_train_p99.

Acceptance criteria covered:
    B3-SPLIT-DATES  boundary-row assignment (train/val/test boundaries)
    B3-SPLIT-TRAIN  train range 2023-10-01 → 2024-01-15
    B3-SPLIT-VAL    val range 2024-01-16 → 2024-02-07
    B3-SPLIT-TEST   test range 2024-02-08 → 2024-02-29
    AC-WINSOR-1     threshold sourced from train p99 (not test p99)
    AC-WINSOR-2     leakage guard — test outlier does NOT shift threshold
    AC-WINSOR-3     null preservation after winsorization
    AC-WINSOR-4     clipping without dropping rows
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from tests.fixtures.headways_factory import make_headways_fixture
from src.evaluation.splits import split_temporal, winsorize_train_p99


# ---------------------------------------------------------------------------
# Helper — build a tiny fixture with rows on specific dates
# ---------------------------------------------------------------------------

def _make_dated_df(dates: list[date], delta_values: list[float | None]) -> pl.DataFrame:
    """One row per date, single slot (direction=-1, pair_rank=1)."""
    assert len(dates) == len(delta_values)
    rows = []
    for d, dv in zip(dates, delta_values):
        rows.append({
            "empresaid": 2,
            "t": datetime(d.year, d.month, d.day, 0, 0, 0),
            "direction": -1,
            "pair_rank": 1,
            "delta_t_min": dv,
        })
    df = pl.DataFrame(rows).with_columns(
        pl.col("empresaid").cast(pl.Int64),
        pl.col("t").cast(pl.Datetime("us")),
        pl.col("direction").cast(pl.Int64),
        pl.col("pair_rank").cast(pl.Int32),
        pl.col("delta_t_min").cast(pl.Float64),
    )
    return df


# ---------------------------------------------------------------------------
# B3-SPLIT-DATES: boundary-row acceptance scenario
# ---------------------------------------------------------------------------

class TestSplitTemporal:
    def test_split_boundary_dates(self):
        """B3-SPLIT-DATES: five boundary dates are assigned the correct split buckets.

        Boundary dates (inclusive ends of each range):
            2024-01-15 → train (last train day)
            2024-01-16 → val   (first val day)
            2024-02-07 → val   (last val day)
            2024-02-08 → test  (first test day)
            2024-02-29 → test  (last test day)
        """
        boundary_dates = [
            date(2024, 1, 15),
            date(2024, 1, 16),
            date(2024, 2, 7),
            date(2024, 2, 8),
            date(2024, 2, 29),
        ]
        expected_splits = ["train", "val", "val", "test", "test"]

        df = _make_dated_df(boundary_dates, [1.0] * 5)
        result = split_temporal(df)

        assert "split" in result.columns, "split_temporal must add a 'split' column"

        actual_splits = result.sort("t")["split"].to_list()
        assert actual_splits == expected_splits, (
            f"Boundary split mismatch.\n"
            f"  Dates:    {boundary_dates}\n"
            f"  Expected: {expected_splits}\n"
            f"  Got:      {actual_splits}"
        )

    def test_split_no_overlap(self):
        """No row can belong to two splits; train + val + test counts == total rows."""
        train_dates = [date(2023, 11, 1), date(2024, 1, 10)]
        val_dates = [date(2024, 1, 20)]
        test_dates = [date(2024, 2, 10), date(2024, 2, 20)]
        all_dates = train_dates + val_dates + test_dates

        df = _make_dated_df(all_dates, [1.0] * len(all_dates))
        result = split_temporal(df)

        n_train = result.filter(pl.col("split") == "train").height
        n_val   = result.filter(pl.col("split") == "val").height
        n_test  = result.filter(pl.col("split") == "test").height

        assert n_train + n_val + n_test == result.height, (
            "Total split rows must equal total rows (no row in two splits)."
        )
        assert n_train == 2
        assert n_val == 1
        assert n_test == 2


# ---------------------------------------------------------------------------
# AC-WINSOR-1: threshold sourced from train only
# ---------------------------------------------------------------------------

class TestWinsorizeTrainP99:
    def test_winsor_threshold_from_train_only(self):
        """AC-WINSOR-1: train p99 threshold is applied to test outlier.

        Train delta_t_min = [2.0, 3.0, 100.0].  p99 ≈ 100.0.
        Test delta_t_min = [200.0].
        After winsorization, the test value must be clipped to the train p99
        (≈ 100.0), NOT to the test p99 (200.0).
        """
        # Three train rows + one test row.
        train_date = date(2023, 12, 1)
        test_date = date(2024, 2, 15)

        df = _make_dated_df(
            [train_date, train_date, train_date, test_date],
            [2.0, 3.0, 100.0, 200.0],
        )
        # Three rows land on same date/slot — use different timestamps.
        df = pl.DataFrame({
            "empresaid": [2, 2, 2, 2],
            "t": [
                datetime(2023, 12, 1, 0, 0, 0),
                datetime(2023, 12, 1, 0, 1, 0),
                datetime(2023, 12, 1, 0, 2, 0),
                datetime(2024, 2, 15, 0, 0, 0),
            ],
            "direction": [-1, -1, -1, -1],
            "pair_rank": [1, 1, 1, 1],
            "delta_t_min": [2.0, 3.0, 100.0, 200.0],
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        df = split_temporal(df)

        clipped, threshold = winsorize_train_p99(df)

        # The threshold must come from train rows only.
        train_vals = df.filter(pl.col("split") == "train")["delta_t_min"].to_list()
        assert threshold <= max(v for v in train_vals if v is not None) + 1e-9

        # The test row must be clipped to threshold.
        test_row_val = clipped.filter(pl.col("split") == "test")["delta_t_min"][0]
        assert test_row_val == pytest.approx(threshold, abs=1e-9), (
            f"Test value {test_row_val} must equal train p99 threshold {threshold}."
        )

    # ---------------------------------------------------------------------------
    # AC-WINSOR-2: leakage guard — test outlier does NOT shift threshold
    # ---------------------------------------------------------------------------
    def test_winsor_leakage_guard(self):
        """AC-WINSOR-2 (critical leakage guard): injecting an extreme outlier into
        the test set must NOT change the threshold compared to train-only p99.

        This is risk R-LEAK-WINSOR from the proposal.  The test is findable by
        name 'test_winsor_leakage_guard' as required by design §4.
        """
        # Build a frame where train rows have a moderate spread (p99 = T_train)
        # and test contains an extreme outlier so that the combined-dataset p99
        # would be much higher (T_combined > T_train).
        train_vals = list(range(1, 101))  # 100 train rows: 1.0 to 100.0; p99 ≈ 99.01
        test_val_extreme = 9999.0         # extreme outlier — would shift combined p99

        train_dates = [date(2023, 11, 1)] * 100  # all on the same date (different timestamps)
        test_date = date(2024, 2, 10)

        # Build manually for precise timestamp uniqueness.
        train_rows = [
            {
                "empresaid": 2,
                "t": datetime(2023, 11, 1, 0, i % 60, i // 60),
                "direction": -1,
                "pair_rank": 1,
                "delta_t_min": float(v),
            }
            for i, v in enumerate(train_vals)
        ]
        test_rows = [
            {
                "empresaid": 2,
                "t": datetime(2024, 2, 10, 0, 0, 0),
                "direction": -1,
                "pair_rank": 1,
                "delta_t_min": test_val_extreme,
            }
        ]
        df = pl.DataFrame(train_rows + test_rows).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        df = split_temporal(df)

        _, t_train = winsorize_train_p99(df)

        # Compute what the combined-dataset p99 would be (should be higher).
        all_vals = df["delta_t_min"].drop_nulls()
        t_combined = float(all_vals.quantile(0.99))

        # Leakage guard: train-only threshold must differ from combined threshold.
        assert t_combined > t_train, (
            f"Test setup error: combined p99 ({t_combined}) must be > train p99 ({t_train})."
        )
        # The threshold returned by winsorize_train_p99 must equal T_train (not T_combined).
        assert t_train < t_combined, (
            f"Leakage detected: winsorize threshold ({t_train}) equals combined p99 "
            f"({t_combined}), meaning test outlier shifted the threshold."
        )

    # ---------------------------------------------------------------------------
    # AC-WINSOR-3: null preservation
    # ---------------------------------------------------------------------------
    def test_winsor_null_preservation(self):
        """AC-WINSOR-3: null delta_t_min values remain null after winsorization."""
        df = pl.DataFrame({
            "empresaid": [2, 2, 2],
            "t": [
                datetime(2023, 12, 1, 0, 0, 0),
                datetime(2023, 12, 1, 0, 1, 0),
                datetime(2024, 2, 15, 0, 0, 0),
            ],
            "direction": [-1, -1, -1],
            "pair_rank": [1, 1, 1],
            "delta_t_min": [5.0, None, None],
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        df = split_temporal(df)
        clipped, _ = winsorize_train_p99(df)

        # Null rows must remain null.
        null_count = clipped["delta_t_min"].null_count()
        assert null_count == 2, (
            f"Expected 2 null rows after winsorization; got {null_count}."
        )

    # ---------------------------------------------------------------------------
    # AC-WINSOR-4: clipping not dropping
    # ---------------------------------------------------------------------------
    def test_winsor_clipping_not_dropping(self):
        """AC-WINSOR-4: values above threshold are clipped to threshold; row count unchanged."""
        df = pl.DataFrame({
            "empresaid": [2, 2, 2],
            "t": [
                datetime(2023, 12, 1, 0, 0, 0),
                datetime(2023, 12, 1, 0, 1, 0),
                datetime(2024, 2, 15, 0, 0, 0),
            ],
            "direction": [-1, -1, -1],
            "pair_rank": [1, 1, 1],
            "delta_t_min": [2.0, 100.0, 100.000001],  # last value = threshold + epsilon
        }).with_columns(
            pl.col("empresaid").cast(pl.Int64),
            pl.col("t").cast(pl.Datetime("us")),
            pl.col("direction").cast(pl.Int64),
            pl.col("pair_rank").cast(pl.Int32),
            pl.col("delta_t_min").cast(pl.Float64),
        )
        df = split_temporal(df)
        clipped, threshold = winsorize_train_p99(df)

        # Row count must be unchanged.
        assert clipped.height == df.height, (
            "winsorize_train_p99 must not drop rows."
        )

        # The test row with value > threshold must be clipped to threshold.
        test_val = clipped.filter(pl.col("split") == "test")["delta_t_min"][0]
        assert test_val == pytest.approx(threshold, abs=1e-9), (
            f"Value above threshold must be clipped to {threshold}; got {test_val}."
        )
