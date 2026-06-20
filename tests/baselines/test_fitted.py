"""TDD tests for src/baselines/fitted.py — fitted ML baseline (B5_XGB).

These are EXECUTION-level tests (the model is actually trained and asked to
predict), not source-structure assertions. They protect the properties a
reviewer cares about for a fitted baseline:

    AC-B5-1: output contract — input frame + one new column `y_pred_b5_xgb`
             (Float64), same row count, original columns preserved.
    AC-B5-2: the model learns — on a constant series it predicts that constant.
    AC-B5-3: determinism — two identical calls produce identical predictions.
    AC-B5-4: NO LEAKAGE — the model is fit on TRAIN rows only; test-period
             target values do not bleed into the fit (train=2, test=100 →
             test predictions stay near the TRAIN pattern, not 100).
    AC-B5-5: degenerate input — empty train split yields an all-null column,
             no crash.
    AC-B5-6: horizon-aware — runs at h>1 and produces finite test predictions.
"""
from __future__ import annotations

from datetime import date

import math

import polars as pl

from tests.fixtures.headways_factory import make_headways_fixture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _frame(train_vals: list[float], test_vals: list[float]) -> pl.DataFrame:
    """Single-slot frame (direction=-1, pair_rank=1) with a `split` column.

    train_vals / test_vals are assigned chronologically (train dates first).
    One row per date; dates are spread across distinct days so timestamps are
    unique and the slot forms a proper time series.
    """
    from datetime import timedelta

    from src.evaluation.splits import split_temporal

    base_train = date(2023, 11, 1)
    base_test = date(2024, 2, 10)
    train_dates = [base_train + timedelta(days=i) for i in range(len(train_vals))]
    test_dates = [base_test + timedelta(days=i) for i in range(len(test_vals))]

    delta_map = {(-1, 1): [float(v) for v in (train_vals + test_vals)]}
    df = make_headways_fixture(
        empresaid=2,
        train_dates=train_dates,
        test_dates=test_dates,
        delta_values_per_slot=delta_map,
    )
    return split_temporal(df)


# ---------------------------------------------------------------------------
# AC-B5-1: output contract
# ---------------------------------------------------------------------------

class TestB5Contract:
    def test_adds_prediction_column_preserving_rows(self):
        """AC-B5-1: returns input frame + y_pred_b5_xgb (Float64), rows preserved."""
        from src.baselines.fitted import predict_b5_xgb

        df = _frame(train_vals=[5.0] * 30, test_vals=[5.0] * 8)
        result = predict_b5_xgb(df, horizon=1)

        assert "y_pred_b5_xgb" in result.columns
        assert result.schema["y_pred_b5_xgb"] == pl.Float64
        assert len(result) == len(df), "row count must be preserved"
        # Original columns must survive.
        for col in df.columns:
            assert col in result.columns, f"original column {col!r} dropped"


# ---------------------------------------------------------------------------
# AC-B5-2: the model learns
# ---------------------------------------------------------------------------

class TestB5Learns:
    def test_constant_series_predicts_constant(self):
        """AC-B5-2: a constant headway series → test predictions near the constant."""
        from src.baselines.fitted import predict_b5_xgb

        df = _frame(train_vals=[5.0] * 40, test_vals=[5.0] * 10)
        result = predict_b5_xgb(df, horizon=1)

        test_preds = (
            result.filter(pl.col("split") == "test")["y_pred_b5_xgb"]
            .drop_nulls()
            .to_list()
        )
        assert test_preds, "expected non-null test predictions"
        for p in test_preds:
            assert abs(p - 5.0) < 0.5, f"expected ~5.0, got {p}"


# ---------------------------------------------------------------------------
# AC-B5-3: determinism
# ---------------------------------------------------------------------------

class TestB5Deterministic:
    def test_two_calls_identical(self):
        """AC-B5-3: identical inputs → identical predictions."""
        from src.baselines.fitted import predict_b5_xgb

        df = _frame(train_vals=[float(i % 7 + 1) for i in range(50)], test_vals=[3.0] * 10)
        r1 = predict_b5_xgb(df, horizon=1)["y_pred_b5_xgb"].to_list()
        r2 = predict_b5_xgb(df, horizon=1)["y_pred_b5_xgb"].to_list()

        for a, b in zip(r1, r2):
            if a is None or b is None:
                assert a is b
            else:
                assert a == b, "predictions must be deterministic"


# ---------------------------------------------------------------------------
# AC-B5-4: no leakage — fit on train only
# ---------------------------------------------------------------------------

class TestB5NoLeakage:
    def test_fits_train_only_not_test_target(self):
        """AC-B5-4: train=2, test=100. The model never saw 100 at fit time, so
        test predictions must stay near the TRAIN pattern (~2), proving the test
        target did not leak into training.
        """
        from src.baselines.fitted import predict_b5_xgb

        df = _frame(train_vals=[2.0] * 40, test_vals=[100.0] * 12)
        result = predict_b5_xgb(df, horizon=1)

        test_preds = (
            result.filter(pl.col("split") == "test")["y_pred_b5_xgb"]
            .drop_nulls()
            .to_list()
        )
        assert test_preds, "expected non-null test predictions"
        for p in test_preds:
            # Must be far closer to the train value (2) than the test value (100).
            assert abs(p - 2.0) < abs(p - 100.0), (
                f"prediction {p} is closer to the test target (100) than to the "
                "train pattern (2) — possible leakage"
            )


# ---------------------------------------------------------------------------
# AC-B5-5: degenerate input
# ---------------------------------------------------------------------------

class TestB5Degenerate:
    def test_empty_train_returns_null_column(self):
        """AC-B5-5: no train rows → all-null y_pred_b5_xgb, no exception."""
        from src.baselines.fitted import predict_b5_xgb
        from src.evaluation.splits import split_temporal
        from datetime import timedelta

        # Only test rows (no train/val): build directly.
        base_test = date(2024, 2, 10)
        test_dates = [base_test + timedelta(days=i) for i in range(6)]
        df = make_headways_fixture(
            empresaid=2,
            train_dates=[],
            test_dates=test_dates,
            delta_values_per_slot={(-1, 1): [5.0] * 6},
        )
        df = split_temporal(df)

        result = predict_b5_xgb(df, horizon=1)
        assert "y_pred_b5_xgb" in result.columns
        assert result["y_pred_b5_xgb"].null_count() == len(result), (
            "with no train rows, all predictions must be null"
        )


# ---------------------------------------------------------------------------
# AC-B5-6: horizon-aware
# ---------------------------------------------------------------------------

class TestB5Horizon:
    def test_runs_at_horizon_3(self):
        """AC-B5-6: produces finite, non-null test predictions at h=3."""
        from src.baselines.fitted import predict_b5_xgb

        df = _frame(train_vals=[float(i % 5 + 1) for i in range(60)], test_vals=[3.0] * 12)
        result = predict_b5_xgb(df, horizon=3)

        test_preds = (
            result.filter(pl.col("split") == "test")["y_pred_b5_xgb"]
            .drop_nulls()
            .to_list()
        )
        assert test_preds, "expected non-null test predictions at h=3"
        for p in test_preds:
            assert math.isfinite(p), f"prediction must be finite, got {p}"
