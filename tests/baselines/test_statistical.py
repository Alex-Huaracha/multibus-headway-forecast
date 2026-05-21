"""TDD tests for src/baselines/statistical.py — Fase 3.

Test classes:
    TestB0  — predict_b0 (global mean per slot)   AC-B0-1, AC-B0-2, AC-B0-3
    TestB1  — predict_b1 (naive persistence)       AC-B1-1, AC-B1-2, AC-B1-3
    TestB2  — predict_b2 (moving average)          AC-B2-1..5
    TestB3  — predict_b3 (SES α=0.3)              AC-B3-1..5

All fixtures use make_headways_fixture + split_temporal so the frame has a
`split` column before being handed to baseline functions.
"""
from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from tests.fixtures.headways_factory import make_headways_fixture
from src.evaluation.splits import split_temporal


# ---------------------------------------------------------------------------
# Helper: build a minimal single-slot frame ready for baselines
# ---------------------------------------------------------------------------

def _frame(
    train_deltas: list[float | None],
    test_deltas: list[float | None],
    *,
    empresaid: int = 2,
    direction: int = -1,
    pair_rank: int = 1,
    extra_slots: dict[tuple[int, int], list[float | None]] | None = None,
) -> pl.DataFrame:
    """Return a split-annotated headways frame for one (or more) slots.

    Values in train_deltas are placed on dates 2023-12-01, 2023-12-02, …
    Values in test_deltas are placed on dates 2024-02-10, 2024-02-11, …
    These dates are firmly inside the train / test windows from splits.py.
    """
    n_train = len(train_deltas)
    n_test = len(test_deltas)

    train_dates = [
        date(2023, 12, 1 + i) for i in range(n_train)
    ]
    test_dates = [
        date(2024, 2, 10 + i) for i in range(n_test)
    ]

    delta_map: dict[tuple[int, int], list[float | None]] = {
        (direction, pair_rank): train_deltas + test_deltas
    }
    if extra_slots:
        delta_map.update(extra_slots)

    df = make_headways_fixture(
        empresaid=empresaid,
        train_dates=train_dates,
        test_dates=test_dates,
        delta_values_per_slot=delta_map,
    )
    return split_temporal(df)


# ===========================================================================
# Wave 3: B0 — global mean per slot
# ===========================================================================

class TestB0:
    """predict_b0: per-slot mean of train delta_t_min, propagated to test rows."""

    def test_b0_deterministic_mean(self):
        """AC-B0-1: slot with train [3.0, 5.0, 7.0] → test prediction = 5.0."""
        from src.baselines.statistical import predict_b0

        df = _frame([3.0, 5.0, 7.0], [0.0])  # test ground-truth is irrelevant here
        result = predict_b0(df)

        test_rows = result.filter(pl.col("split") == "test")
        assert len(test_rows) == 1
        pred = test_rows["y_pred_b0"][0]
        assert abs(pred - 5.0) < 1e-9, f"Expected 5.0, got {pred}"

    def test_b0_null_train_emits_null(self):
        """AC-B0-2: slot with all-null train → prediction is null for test rows."""
        from src.baselines.statistical import predict_b0

        df = _frame([None, None], [0.0])
        result = predict_b0(df)

        test_rows = result.filter(pl.col("split") == "test")
        assert test_rows["y_pred_b0"][0] is None

    def test_b0_multi_slot_independence(self):
        """AC-B0-3: two slots have independent means; slot A doesn't affect slot B."""
        from src.baselines.statistical import predict_b0

        df = _frame(
            train_deltas=[10.0, 20.0],
            test_deltas=[0.0],
            direction=-1,
            pair_rank=1,
            extra_slots={(1, 1): [1.0, 1.0, 0.0]},  # slot B: mean=1.0
        )
        result = predict_b0(df)

        test_a = (
            result
            .filter((pl.col("split") == "test") & (pl.col("direction") == -1))
            ["y_pred_b0"][0]
        )
        test_b = (
            result
            .filter((pl.col("split") == "test") & (pl.col("direction") == 1))
            ["y_pred_b0"][0]
        )
        assert abs(test_a - 15.0) < 1e-9, f"Slot A mean should be 15.0, got {test_a}"
        assert abs(test_b - 1.0) < 1e-9, f"Slot B mean should be 1.0, got {test_b}"


# ===========================================================================
# Wave 4: B1 — naive / persistence baseline
# ===========================================================================

class TestB1:
    """predict_b1: last non-null observed delta_t_min carried forward (shift-by-1)."""

    def test_b1_basic_persistence(self):
        """AC-B1-1: train [4.0, null, 6.0], test row at t=4 → prediction 6.0."""
        from src.baselines.statistical import predict_b1

        df = _frame([4.0, None, 6.0], [0.0])
        result = predict_b1(df)

        test_rows = result.filter(pl.col("split") == "test")
        assert len(test_rows) == 1
        pred = test_rows["y_pred_b1"][0]
        assert abs(pred - 6.0) < 1e-9, f"Expected 6.0, got {pred}"

    def test_b1_all_null_train_emits_null(self):
        """AC-B1-2: all train observations null → first test row prediction is null."""
        from src.baselines.statistical import predict_b1

        df = _frame([None, None], [0.0])
        result = predict_b1(df)

        test_rows = result.filter(pl.col("split") == "test")
        assert test_rows["y_pred_b1"][0] is None

    def test_b1_null_mid_train_carries_forward(self):
        """AC-B1-3: train [3.0, null], test row → prediction 3.0 (null doesn't reset)."""
        from src.baselines.statistical import predict_b1

        df = _frame([3.0, None], [0.0])
        result = predict_b1(df)

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows["y_pred_b1"][0]
        assert abs(pred - 3.0) < 1e-9, f"Expected 3.0, got {pred}"


# ===========================================================================
# Wave 5: B2 — moving average (last w NON-NULL observations)
# ===========================================================================

class TestB2:
    """predict_b2: trailing mean of last w non-null observations, min_periods=w//2."""

    def test_b2_full_window(self):
        """AC-B2-1: w=5, history [2,4,6,8,10] all non-null → prediction = 6.0."""
        from src.baselines.statistical import predict_b2

        df = _frame([2.0, 4.0, 6.0, 8.0, 10.0], [0.0])
        result = predict_b2(df, window=5)
        col = "y_pred_b2_w5"

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows[col][0]
        assert abs(pred - 6.0) < 1e-9, f"Expected 6.0, got {pred}"

    def test_b2_minimum_non_null_passes(self):
        """AC-B2-2: w=5, floor(5/2)=2, exactly 2 non-null → prediction emitted (not null)."""
        from src.baselines.statistical import predict_b2

        # 4 nulls then 2 non-null values in train, 1 test row
        df = _frame([None, None, None, None, 3.0, 5.0], [0.0])
        result = predict_b2(df, window=5)
        col = "y_pred_b2_w5"

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows[col][0]
        assert pred is not None, "Expected a prediction (2 non-null >= min_periods=2)"
        # mean of [3.0, 5.0] = 4.0
        assert abs(pred - 4.0) < 1e-9, f"Expected 4.0, got {pred}"

    def test_b2_below_minimum_non_null_emits_null(self):
        """AC-B2-3: w=5, floor(5/2)=2, only 1 non-null → prediction is null."""
        from src.baselines.statistical import predict_b2

        # 4 nulls then 1 non-null in train, 1 test row
        df = _frame([None, None, None, None, 5.0], [0.0])
        result = predict_b2(df, window=5)
        col = "y_pred_b2_w5"

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows[col][0]
        assert pred is None, f"Expected null (only 1 non-null < min_periods=2), got {pred}"

    def test_b2_causal_no_future_leakage(self):
        """AC-B2-4: prediction at time T must not use observations after T."""
        from src.baselines.statistical import predict_b2

        # Train: [1.0, 2.0, 3.0, 4.0, 5.0], Test: [999.0]
        # The test observation value (999.0) must NOT influence the prediction.
        df = _frame([1.0, 2.0, 3.0, 4.0, 5.0], [999.0])
        result = predict_b2(df, window=5)
        col = "y_pred_b2_w5"

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows[col][0]
        # Prediction must be mean of train [1,2,3,4,5] = 3.0, NOT influenced by 999
        assert abs(pred - 3.0) < 1e-9, f"Expected 3.0 (no leakage), got {pred}"

    def test_b2_independence_across_windows(self):
        """AC-B2-5: w=5, w=10, w=15 produce separate columns; they share no state."""
        from src.baselines.statistical import predict_b2

        # Make 15 non-null train observations with different values so the
        # means will differ across window sizes.
        train_vals = list(range(1, 16))  # 1..15
        df = _frame(train_vals, [0.0])

        r5 = predict_b2(df, window=5)
        r10 = predict_b2(df, window=10)
        r15 = predict_b2(df, window=15)

        test5 = r5.filter(pl.col("split") == "test")["y_pred_b2_w5"][0]
        test10 = r10.filter(pl.col("split") == "test")["y_pred_b2_w10"][0]
        test15 = r15.filter(pl.col("split") == "test")["y_pred_b2_w15"][0]

        # mean of last 5: [11,12,13,14,15] = 13.0
        assert abs(test5 - 13.0) < 1e-9, f"w=5: expected 13.0, got {test5}"
        # mean of last 10: [6..15] = 10.5
        assert abs(test10 - 10.5) < 1e-9, f"w=10: expected 10.5, got {test10}"
        # mean of last 15: [1..15] = 8.0
        assert abs(test15 - 8.0) < 1e-9, f"w=15: expected 8.0, got {test15}"
        # Columns are separate
        assert "y_pred_b2_w5" in r5.columns
        assert "y_pred_b2_w10" in r10.columns
        assert "y_pred_b2_w15" in r15.columns


# ===========================================================================
# Wave 6: B3 — Simple Exponential Smoothing (α=0.3)
# ===========================================================================

class TestB3:
    """predict_b3: online per-slot SES with α=0.3, null-skip, initialized from first non-null."""

    def test_b3_two_step_recursion(self):
        """AC-B3-1: train [10.0, null, 5.0] → test prediction = 0.3*5 + 0.7*10 = 8.5."""
        from src.baselines.statistical import predict_b3

        df = _frame([10.0, None, 5.0], [0.0])
        result = predict_b3(df)

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows["y_pred_b3"][0]
        expected = 0.3 * 5.0 + 0.7 * 10.0  # = 8.5
        assert abs(pred - expected) < 1e-9, f"Expected {expected}, got {pred}"

    def test_b3_null_skip_does_not_update_state(self):
        """AC-B3-2: null observations in train do not change the smoothing state."""
        from src.baselines.statistical import predict_b3

        # Train: [8.0, null, null], one test row.
        # State after train: s=8.0 (nulls are skipped).
        # Test prediction = 8.0.
        df = _frame([8.0, None, None], [0.0])
        result = predict_b3(df)

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows["y_pred_b3"][0]
        assert abs(pred - 8.0) < 1e-9, f"Expected 8.0 (null-skip), got {pred}"

    def test_b3_initialization_from_first_non_null(self):
        """AC-B3-3: single train obs y=8.0 → state after warm-up == 8.0."""
        from src.baselines.statistical import predict_b3

        df = _frame([8.0], [0.0])
        result = predict_b3(df)

        test_rows = result.filter(pl.col("split") == "test")
        pred = test_rows["y_pred_b3"][0]
        assert abs(pred - 8.0) < 1e-9, f"Expected 8.0 (init from first obs), got {pred}"

    def test_b3_all_null_train_emits_null(self):
        """AC-B3-4: all-null train → B3 emits null for all test rows."""
        from src.baselines.statistical import predict_b3

        df = _frame([None, None, None], [0.0])
        result = predict_b3(df)

        test_rows = result.filter(pl.col("split") == "test")
        assert test_rows["y_pred_b3"][0] is None

    def test_b3_determinism(self):
        """AC-B3-5: calling predict_b3 twice on the same input produces bit-identical results."""
        from src.baselines.statistical import predict_b3

        df = _frame([1.0, 2.0, None, 3.0], [0.0, 0.0])
        result1 = predict_b3(df)
        result2 = predict_b3(df)

        # Sort both by same key to ensure row order alignment
        key = ["direction", "pair_rank", "t"]
        r1 = result1.sort(key)["y_pred_b3"].to_list()
        r2 = result2.sort(key)["y_pred_b3"].to_list()
        assert r1 == r2, "predict_b3 must be deterministic"
