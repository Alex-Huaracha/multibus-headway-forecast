"""Fitted ML baseline for headway forecasting — gradient-boosted regressor (B5_XGB).

Why this module is separate from `statistical.py`:
    B0-B4 are closed-form/recursive predictors with NO learned parameters and a
    "no new dependencies" design lock. B5_XGB is a *fitted* learner (XGBoost) —
    a different category. It answers the reviewer reflex "where is a fitted/ML
    baseline?" that pure naive baselines (persistence, moving average, SES,
    historical average) do not.

Design — fair comparison to the DL models (NB11-13):
    The DL models consume an input window of T_in = 12 consecutive 1-minute
    steps and predict the headway HORIZON steps after the last input step. The
    XGBoost baseline is given the SAME information: 12 lagged headway values
    ending HORIZON steps before the target, so `lag_1` equals the B1 persistence
    prediction (`shift(horizon)`) and the model strictly extends the naive
    baselines rather than seeing extra future data. Calendar context (hour,
    weekday) and static slot keys (direction, pair_rank) round out the features.

Contract (mirrors statistical.py):
    predict_b5_xgb(headways, *, horizon=1, seed=42) -> headways + y_pred_b5_xgb
    Input must have the `split` column (added by split_temporal). The model is
    fit on TRAIN rows only; predictions are produced for ALL rows. Validation
    rows are used for early stopping ONLY when there are enough of them
    (>= _MIN_VAL_ROWS); otherwise a fixed number of trees is used.

Determinism:
    Single-threaded (`n_jobs=1`), fixed `random_state`, `tree_method="hist"` →
    repeated calls on the same machine produce identical predictions.
"""
from __future__ import annotations

import numpy as np
import polars as pl

_SLOT_COLS: list[str] = ["empresaid", "direction", "pair_rank"]

# Number of lagged headway steps fed to the model = DL input window (T_in).
N_LAGS: int = 12

# Use validation rows for early stopping only when there are at least this many;
# tiny test fixtures (and corridors with no val rows) fall back to fixed trees.
_MIN_VAL_ROWS: int = 50

# Fixed gradient-boosting hyperparameters (native xgboost API, no sklearn dep).
# Deliberately modest and regularized: a credible fitted competitor, not an
# over-tuned one. nthread=1 + fixed seed + hist tree method → deterministic.
_NUM_BOOST_ROUND: int = 400
_EARLY_STOPPING_ROUNDS: int = 30

_XGB_PARAMS: dict = {
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "nthread": 1,
}


def _build_features(headways: pl.DataFrame, *, horizon: int) -> tuple[pl.DataFrame, list[str]]:
    """Return the frame sorted by (slot, t) with lag + calendar feature columns
    added, plus the list of feature column names.

    lag_k (k = 1..N_LAGS) = headway value (forward-filled within slot) observed
    `horizon + k - 1` steps before the target row. lag_1 == B1 persistence.
    """
    lag_exprs = [
        pl.col("delta_t_min")
        .forward_fill()
        .shift(horizon + k - 1)
        .over(_SLOT_COLS)
        .alias(f"_lag_{k}")
        for k in range(1, N_LAGS + 1)
    ]
    df = (
        headways
        .sort(_SLOT_COLS + ["t"])
        .with_columns(
            *lag_exprs,
            pl.col("t").dt.hour().alias("_hour"),
            pl.col("t").dt.weekday().alias("_weekday"),
        )
    )
    feature_cols = (
        [f"_lag_{k}" for k in range(1, N_LAGS + 1)]
        + ["_hour", "_weekday", "direction", "pair_rank"]
    )
    return df, feature_cols


def predict_b5_xgb(
    headways: pl.DataFrame,
    *,
    horizon: int = 1,
    seed: int = 42,
) -> pl.DataFrame:
    """Add column `y_pred_b5_xgb`: gradient-boosted forecast of delta_t_min.

    Parameters
    ----------
    headways:
        headways DataFrame with the `split` column attached. Columns consumed:
        empresaid, t, direction, pair_rank, delta_t_min, split.
    horizon:
        Forecast horizon in steps. lag_1 = shift(horizon) so the 1-lag feature
        equals B1 persistence; horizon=1 is the default.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame — input frame (sorted by slot, t) with `y_pred_b5_xgb`
        (Float64 nullable) added. If the train split has no usable rows, the
        column is all-null.
    """
    import xgboost as xgb

    original_cols = headways.columns
    df, feature_cols = _build_features(headways, horizon=horizon)

    is_train = df["split"] == "train"
    is_val = df["split"] == "val"
    target_present = df["delta_t_min"].is_not_null()

    train_mask = (is_train & target_present).to_numpy()
    n_train = int(train_mask.sum())

    # Degenerate: nothing to fit on → null predictions (mirrors B0 on empty slots).
    if n_train == 0:
        return df.select(original_cols).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("y_pred_b5_xgb")
        )

    X_all = df.select(feature_cols).to_numpy().astype(np.float64)
    y_all = df["delta_t_min"].to_numpy().astype(np.float64)

    dtrain = xgb.DMatrix(X_all[train_mask], label=y_all[train_mask], missing=np.nan)
    dall = xgb.DMatrix(X_all, missing=np.nan)

    params = dict(_XGB_PARAMS, seed=seed)

    val_mask = (is_val & target_present).to_numpy()
    if int(val_mask.sum()) >= _MIN_VAL_ROWS:
        # Use validation for early stopping (the DL models also tuned on val).
        dval = xgb.DMatrix(X_all[val_mask], label=y_all[val_mask], missing=np.nan)
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=_NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
    else:
        booster = xgb.train(params, dtrain, num_boost_round=_NUM_BOOST_ROUND)

    preds = booster.predict(dall).astype(np.float64)

    return df.select(original_cols).with_columns(
        pl.Series("y_pred_b5_xgb", preds, dtype=pl.Float64)
    )
