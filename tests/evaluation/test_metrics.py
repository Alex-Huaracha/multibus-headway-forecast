"""Tests for src/evaluation/metrics.py — MAE, RMSE, null-masking contract.

Acceptance criteria covered:
    AC-MAE-1  basic mae
    AC-MAE-2  null in y_true
    AC-MAE-3  null in y_pred
    AC-MAE-4  perfect prediction
    AC-MAE-5  all-null raises ValueError
    AC-MAE-6  empty input raises ValueError
    AC-RMSE-1 basic rmse
    AC-RMSE-2 null masking rmse
    AC-RMSE-3 perfect prediction rmse
    AC-RMSE-4 rmse >= mae (property)
    AC-NO-MAPE-1 mape/smape/percentage_error must not exist
    AC-MOD-M-1  mae and rmse importable
    AC-MOD-M-2  mae and rmse accept polars Series and numpy arrays
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import polars as pl
import pytest

from src.evaluation.metrics import mae, rmse


# ---------------------------------------------------------------------------
# AC-MOD-M-1: importable (implicit — if the import above fails, all tests fail)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AC-MAE-1: basic
# ---------------------------------------------------------------------------
class TestMaeBasic:
    def test_mae_basic(self):
        """AC-MAE-1: mae([3.0, 5.0], [2.0, 7.0]) == 1.5."""
        result = mae(np.array([3.0, 5.0]), np.array([2.0, 7.0]))
        assert result == pytest.approx(1.5)

    # ---------------------------------------------------------------------------
    # AC-MAE-2: null in y_true
    # ---------------------------------------------------------------------------
    def test_mae_null_in_y_true(self):
        """AC-MAE-2: null row in y_true is dropped; remaining mean is 1.0."""
        y_true = pl.Series([3.0, None, 5.0], dtype=pl.Float64)
        y_pred = pl.Series([2.0, 4.0, 6.0], dtype=pl.Float64)
        # |3-2| = 1, |5-6| = 1 → mean = 1.0  (null row dropped)
        result = mae(y_true, y_pred)
        assert result == pytest.approx(1.0)

    # ---------------------------------------------------------------------------
    # AC-MAE-3: null in y_pred
    # ---------------------------------------------------------------------------
    def test_mae_null_in_y_pred(self):
        """AC-MAE-3: null row in y_pred is dropped; remaining mean is 1.0."""
        y_true = pl.Series([3.0, 5.0, 7.0], dtype=pl.Float64)
        y_pred = pl.Series([2.0, None, 6.0], dtype=pl.Float64)
        # |3-2| = 1, |7-6| = 1 → mean = 1.0  (null row dropped)
        result = mae(y_true, y_pred)
        assert result == pytest.approx(1.0)

    # ---------------------------------------------------------------------------
    # AC-MAE-4: perfect prediction
    # ---------------------------------------------------------------------------
    def test_mae_perfect(self):
        """AC-MAE-4: mae([4.0, 6.0], [4.0, 6.0]) == 0.0."""
        result = mae(np.array([4.0, 6.0]), np.array([4.0, 6.0]))
        assert result == pytest.approx(0.0)

    # ---------------------------------------------------------------------------
    # AC-MAE-5: all-null raises ValueError
    # ---------------------------------------------------------------------------
    def test_mae_all_null_raises(self):
        """AC-MAE-5: all-null y_true raises ValueError (design locks to ValueError)."""
        y_true = pl.Series([None, None], dtype=pl.Float64)
        y_pred = pl.Series([1.0, 2.0], dtype=pl.Float64)
        with pytest.raises(ValueError, match="empty|null"):
            mae(y_true, y_pred)

    # ---------------------------------------------------------------------------
    # AC-MAE-6: empty input raises ValueError
    # ---------------------------------------------------------------------------
    def test_mae_empty_raises(self):
        """AC-MAE-6: empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="empty|null"):
            mae(np.array([]), np.array([]))

    # ---------------------------------------------------------------------------
    # AC-MOD-M-2: type flexibility — polars Series and numpy arrays
    # ---------------------------------------------------------------------------
    def test_mae_accepts_numpy_array(self):
        """AC-MOD-M-2 (numpy path): mae accepts np.ndarray without TypeError."""
        result = mae(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        assert result == pytest.approx(0.5)

    def test_mae_accepts_polars_series(self):
        """AC-MOD-M-2 (polars path): mae accepts pl.Series without TypeError."""
        result = mae(pl.Series([1.0, 2.0]), pl.Series([1.0, 3.0]))
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# AC-RMSE-1: basic
# ---------------------------------------------------------------------------
class TestRmseBasic:
    def test_rmse_basic(self):
        """AC-RMSE-1: rmse([3.0, 5.0], [2.0, 7.0]) == sqrt(2.5) within 1e-9."""
        result = rmse(np.array([3.0, 5.0]), np.array([2.0, 7.0]))
        expected = math.sqrt((1.0**2 + 2.0**2) / 2)  # sqrt(2.5)
        assert abs(result - expected) < 1e-9

    # ---------------------------------------------------------------------------
    # AC-RMSE-2: null masking
    # ---------------------------------------------------------------------------
    def test_rmse_null_masking(self):
        """AC-RMSE-2: null row dropped; rmse([3,5], [2,6]) == sqrt(mean(1,1)) = 1.0."""
        y_true = pl.Series([3.0, None, 5.0], dtype=pl.Float64)
        y_pred = pl.Series([2.0, 4.0, 6.0], dtype=pl.Float64)
        # |3-2|^2 = 1, |5-6|^2 = 1 → sqrt(mean(1,1)) = 1.0
        result = rmse(y_true, y_pred)
        assert result == pytest.approx(1.0)

    # ---------------------------------------------------------------------------
    # AC-RMSE-3: perfect prediction
    # ---------------------------------------------------------------------------
    def test_rmse_perfect(self):
        """AC-RMSE-3: rmse([4.0, 6.0], [4.0, 6.0]) == 0.0."""
        result = rmse(np.array([4.0, 6.0]), np.array([4.0, 6.0]))
        assert result == pytest.approx(0.0)

    # ---------------------------------------------------------------------------
    # AC-RMSE-4: RMSE >= MAE (property)
    # ---------------------------------------------------------------------------
    def test_rmse_ge_mae(self):
        """AC-RMSE-4: rmse >= mae for any 3-element fixture with non-zero variance."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 4.0])
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


# ---------------------------------------------------------------------------
# AC-NO-MAPE-1: mape/smape/percentage_error must NOT exist in the module
# ---------------------------------------------------------------------------
class TestNoMape:
    def test_no_mape_in_metrics_module(self):
        """AC-NO-MAPE-1: mape, smape, percentage_error must not be defined in metrics.py."""
        import src.evaluation.metrics as metrics_module

        source = inspect.getsource(metrics_module)
        for forbidden in ("mape", "smape", "percentage_error"):
            assert forbidden not in source, (
                f"Forbidden name '{forbidden}' found in src/evaluation/metrics.py. "
                "MAPE is explicitly out of scope (spec §2 B3-NO-MAPE)."
            )
