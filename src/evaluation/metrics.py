"""Evaluation metrics for headway forecasting — Fase 3.

Public API:
    mae(y_true, y_pred) -> float
    rmse(y_true, y_pred) -> float

Both functions accept polars Series (Float64) or numpy arrays (float64).
Null / NaN masking: rows where EITHER y_true or y_pred is null/NaN are
dropped before aggregation.  If no valid rows remain, ValueError is raised.

Design decisions locked in design §4:
  - ValueError on empty/all-null input (NOT silent NaN return).
  - Only MAE and RMSE are in scope (spec B3-NO-MAPE — ratio-based metrics
    are out of scope because near-zero headways cause denominator blow-up).
  - No new pyproject.toml dependencies (polars + numpy already present).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def _to_numpy_with_mask(
    y_true: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce both inputs to float64 numpy arrays and apply the null/NaN mask.

    Polars Series with dtype Float64: null cells become NaN via .to_numpy().
    numpy arrays: assumed to already use NaN for missing values.

    Returns
    -------
    (y_true_masked, y_pred_masked) — two 1-D float64 arrays of equal length,
    containing no NaN values.  May be empty if all rows were masked.
    """
    # Coerce to numpy.
    if isinstance(y_true, pl.Series):
        yt = y_true.to_numpy(allow_copy=True).astype(np.float64)
    else:
        yt = np.asarray(y_true, dtype=np.float64).ravel()

    if isinstance(y_pred, pl.Series):
        yp = y_pred.to_numpy(allow_copy=True).astype(np.float64)
    else:
        yp = np.asarray(y_pred, dtype=np.float64).ravel()

    # Elementwise mask: keep row only if BOTH sides are finite (not NaN).
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


def mae(
    y_true: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
) -> float:
    """Mean Absolute Error in minutes, with null/NaN masking.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted headway values in minutes.
        Accepts polars Series (Float64) or numpy arrays (float64).
        Null / NaN positions in either input are dropped before computation.

    Returns
    -------
    float — MAE in minutes.

    Raises
    ------
    ValueError
        If the masked input is empty (all-null or zero-length).
    """
    yt, yp = _to_numpy_with_mask(y_true, y_pred)
    if len(yt) == 0:
        raise ValueError(
            "mae: metric on empty/all-null input — no valid (y_true, y_pred) pairs."
        )
    return float(np.mean(np.abs(yt - yp)))


def rmse(
    y_true: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
) -> float:
    """Root Mean Squared Error in minutes, with null/NaN masking.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted headway values in minutes.
        Accepts polars Series (Float64) or numpy arrays (float64).
        Null / NaN positions in either input are dropped before computation.

    Returns
    -------
    float — RMSE in minutes.

    Raises
    ------
    ValueError
        If the masked input is empty (all-null or zero-length).
    """
    yt, yp = _to_numpy_with_mask(y_true, y_pred)
    if len(yt) == 0:
        raise ValueError(
            "rmse: metric on empty/all-null input — no valid (y_true, y_pred) pairs."
        )
    return float(np.sqrt(np.mean((yt - yp) ** 2)))
