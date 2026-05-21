"""Classical statistical baselines for headway forecasting — Fase 3.

Public API:
    predict_b0(headways: pl.DataFrame) -> pl.DataFrame
    predict_b1(headways: pl.DataFrame) -> pl.DataFrame
    predict_b2(headways: pl.DataFrame, *, window: int) -> pl.DataFrame
    predict_b3(headways: pl.DataFrame, *, alpha: float = SES_ALPHA) -> pl.DataFrame

Input contract (all four functions):
    The DataFrame must have a `split` column (Utf8) added by split_temporal.
    Columns consumed: empresaid, t, direction, pair_rank, delta_t_min, split.

Output contract:
    Each function returns the input frame with ONE additional column:
        B0 → y_pred_b0
        B1 → y_pred_b1
        B2 → y_pred_b2_w{window}
        B3 → y_pred_b3

Predictions are filled for ALL rows (train + test); the evaluation harness
consumes test rows only.  Filling train rows costs negligibly more and lets
future SDDs reuse predictions if needed (design §3).

Design decisions locked in design §3 and §9:
  - Functions, not classes (consistent with project precedent).
  - Slot key: (empresaid, direction, pair_rank).
  - B2 `window` = count of last NON-NULL observations (not a time window).
  - B2 min_periods = window // 2  (floor division).
  - B3 alpha = SES_ALPHA = 0.3, per-slot online recursion, null-skip.
  - B3 state init: NaN until first non-null train obs; first non-null sets s directly.
  - No new pyproject.toml dependencies (polars + numpy only).
"""
from __future__ import annotations

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Module-level constants (locked in design §9)
# ---------------------------------------------------------------------------

BASELINE_B2_WINDOWS: tuple[int, ...] = (5, 10, 15)
SES_ALPHA: float = 0.3

_SLOT_COLS: list[str] = ["empresaid", "direction", "pair_rank"]


# ===========================================================================
# B0 — Global mean per slot (train rows only)
# ===========================================================================

def predict_b0(headways: pl.DataFrame) -> pl.DataFrame:
    """Add column `y_pred_b0`: per-slot mean of train delta_t_min.

    The prediction is constant within a slot — the arithmetic mean of all
    non-null delta_t_min values in the train split for that slot.  Slots
    with no non-null train observations receive null (AC-B0-2).

    Parameters
    ----------
    headways:
        headways DataFrame with `split` column attached.

    Returns
    -------
    pl.DataFrame — input frame with `y_pred_b0` (Float64 nullable) added.
    """
    train_means = (
        headways
        .filter(pl.col("split") == "train")
        .group_by(_SLOT_COLS)
        .agg(pl.col("delta_t_min").mean().alias("y_pred_b0"))
    )
    return headways.join(train_means, on=_SLOT_COLS, how="left")


# ===========================================================================
# B1 — Naive / persistence baseline
# ===========================================================================

def predict_b1(headways: pl.DataFrame) -> pl.DataFrame:
    """Add column `y_pred_b1`: last non-null delta_t_min seen before each row.

    Uses forward_fill().shift(1).over(slot) — the canonical polars pattern for
    ŷ_{t+1} = y_t with null gaps.  Causal by construction (shift prevents
    the current-row value from appearing as its own prediction).

    Parameters
    ----------
    headways:
        headways DataFrame with `split` column attached.

    Returns
    -------
    pl.DataFrame — input frame sorted by (slot, t), with `y_pred_b1` added.
    """
    return (
        headways
        .sort(_SLOT_COLS + ["t"])
        .with_columns(
            pl.col("delta_t_min")
              .forward_fill()
              .shift(1)
              .over(_SLOT_COLS)
              .alias("y_pred_b1")
        )
    )


# ===========================================================================
# B2 — Trailing moving average of last w NON-NULL observations
# ===========================================================================

def predict_b2(headways: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Add column `y_pred_b2_w{window}`: mean of last `window` non-null observations.

    Semantics (locked, design §3):
      - `window` is a COUNT of non-null observations, not a time window.
      - min_periods = window // 2  (floor).
      - Prediction at row i uses only observations strictly before row i (causal).

    Implementation:
      - group_by(slot).map_groups(lambda g: _b2_one_slot(g, window))
      - Within each group: extract non-null values, compute rolling_mean with
        shift(1) (causal), then join_asof(strategy="backward") back to
        original group rows on `t`.

    Parameters
    ----------
    headways:
        headways DataFrame with `split` column attached.
    window:
        Number of non-null observations in the trailing window.

    Returns
    -------
    pl.DataFrame — input frame with `y_pred_b2_w{window}` (Float64 nullable) added.
    """
    col_name = f"y_pred_b2_w{window}"
    min_periods = window // 2

    def _b2_one_slot(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("t")

        # Extract non-null rows only, in time order.
        non_null = group.filter(pl.col("delta_t_min").is_not_null())

        if len(non_null) == 0:
            # No non-null observations: all predictions are null.
            return group.with_columns(pl.lit(None, dtype=pl.Float64).alias(col_name))

        # Compute rolling mean on the non-null sub-series, then shift(1) for
        # causality: the prediction at position i uses observations 0..i-1.
        non_null = non_null.with_columns(
            pl.col("delta_t_min")
              .rolling_mean(window_size=window, min_samples=min_periods)
              .shift(1)
              .alias(col_name)
        )

        # Align back to the full group (including null rows) via join_asof.
        # strategy="backward" finds the most recent non-null rolling mean at or
        # before each timestamp in the original group.
        result = group.join_asof(
            non_null.select(["t", col_name]),
            on="t",
            strategy="backward",
        )
        return result

    result = (
        headways
        .sort(_SLOT_COLS + ["t"])
        .group_by(_SLOT_COLS, maintain_order=True)
        .map_groups(_b2_one_slot)
    )
    return result


# ===========================================================================
# B3 — Simple Exponential Smoothing (α=0.3, per slot, online)
# ===========================================================================

def _ses_one_slot(slot_df: pl.DataFrame, alpha: float) -> pl.DataFrame:
    """Online SES recursion for a single slot.

    s_t = α·y_t + (1-α)·s_{t-1}  (null observations skip the update).
    pred[i] = s before observing y[i]  (causal: shift-1 semantics).

    Initialization: s = NaN until the first non-null y_t; the first non-null
    value sets s directly (no prior needed) — AC-B3-3.  The prediction at
    that initialization row is NaN (no prior state), so the first test
    prediction for a slot with at least one train observation is the state
    after consuming ALL train rows.
    """
    slot_df = slot_df.sort("t")
    y = slot_df["delta_t_min"].to_numpy(allow_copy=True).astype(np.float64)
    pred = np.full(len(y), np.nan)
    s = np.nan  # smoothing state; NaN until first non-null

    for i in range(len(y)):
        # Prediction at row i is the state BEFORE observing y[i].
        pred[i] = s
        # Update state if current observation is not null/NaN.
        if not np.isnan(y[i]):
            if np.isnan(s):
                s = y[i]  # initialization: first non-null sets state directly
            else:
                s = alpha * y[i] + (1.0 - alpha) * s

    # Convert float NaN → polars null so downstream is_null() works correctly.
    pred_series = pl.Series("y_pred_b3", pred, dtype=pl.Float64)
    return slot_df.with_columns(pred_series.set(pred_series.is_nan(), None))


def predict_b3(headways: pl.DataFrame, *, alpha: float = SES_ALPHA) -> pl.DataFrame:
    """Add column `y_pred_b3`: online SES predictions, α=0.3 (default).

    Applies the causal recursion s_t = α·y_t + (1-α)·s_{t-1} per slot.
    Null observations do not update the state (AC-B3-2).
    State is initialized from the first non-null observation (AC-B3-3).
    Slots with all-null values emit null for all rows (AC-B3-4).

    Parameters
    ----------
    headways:
        headways DataFrame with `split` column attached.
    alpha:
        Smoothing parameter.  Default SES_ALPHA = 0.3 (locked, design §3).
        Tests may pass alternative values for edge-case verification.

    Returns
    -------
    pl.DataFrame — input frame with `y_pred_b3` (Float64 nullable) added.

    Design note (D-PL-OVER-VS-MAPGROUPS):
        polars .over() does not support stateful numpy loops; map_groups
        materializes one Python frame per slot (~30–100 per corridor — trivially fast).
    """
    return (
        headways
        .sort(_SLOT_COLS + ["t"])
        .group_by(_SLOT_COLS, maintain_order=True)
        .map_groups(lambda g: _ses_one_slot(g, alpha))
    )
