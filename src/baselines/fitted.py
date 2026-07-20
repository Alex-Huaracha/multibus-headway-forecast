"""Fitted ML baseline for headway forecasting — gradient-boosted regressor (B5_XGB).

Why this module is separate from `statistical.py`:
    B0-B4 are closed-form/recursive predictors with NO learned parameters and a
    "no new dependencies" design lock. B5_XGB is a *fitted* learner (XGBoost) —
    a different category. It answers the reviewer reflex "where is a fitted/ML
    baseline?" that pure naive baselines (persistence, moving average, SES,
    historical average) do not.

Design — fair comparison to the DL models (NB11-13, NB17-19):
    The DL models consume an input window of T_in = 12 consecutive 1-minute
    steps and predict the headway HORIZON steps after the last input step. The
    XGBoost baseline is given the SAME information: 12 lagged headway values
    ending HORIZON steps before the target, so `lag_1` equals the B1 persistence
    prediction (`shift(horizon)`) and the model strictly extends the naive
    baselines rather than seeing extra future data. Calendar context (hour,
    weekday), static slot keys (direction, pair_rank) and the atypical-day flag
    round out the features.

    Two asymmetries versus the DL models were removed (peer-review fix):
      1. ATYPICAL-DAY FLAG. The DL models receive `atypical_flag` as a required,
         hash-pinned context feature. B5_XGB now receives the same binary flag,
         built with the SAME `encode_context` helper so the semantics cannot
         drift between the two model families.
      2. HYPERPARAMETER SEARCH. The DL models were tuned; B5_XGB used a single
         hardcoded configuration. It now runs a seeded random search of
         `SEARCH_N_CONFIGS` configurations selected STRICTLY on the validation
         split (see `_random_search`), with the winning configuration reported
         back to the caller so it can be persisted and audited.

Contract (mirrors statistical.py):
    predict_b5_xgb(headways, *, horizon=1, seed=42, atypical_dates=None,
                   search=True) -> headways + y_pred_b5_xgb
    fit_predict_b5_xgb(...) -> B5FitResult (predictions + search provenance)

    Input must have the `split` column (added by split_temporal). The model is
    fit on TRAIN rows only; predictions are produced for ALL rows. Validation
    rows are used for hyperparameter selection and early stopping ONLY when
    there are enough of them (>= _MIN_VAL_ROWS); otherwise the frozen default
    configuration and a fixed number of trees are used.

Leakage contract (hard):
    The `test` split NEVER influences training, early stopping, or
    hyperparameter selection. Selection reads the validation loss only.

Atypical-flag contract (mirrors the DL notebooks):
    `atypical_dates=None` means "no atypical calendar supplied" (library/fixture
    use) and yields an all-zero flag column. Passing an EXPLICIT EMPTY SET is a
    configuration error — a CSV that parsed to nothing must fail closed instead
    of silently disabling the feature — and raises ValueError.

Determinism:
    Fixed `seed`, fixed `SEARCH_SEED` for the configuration sampler, and
    `tree_method="hist"` with a pinned `nthread` → repeated calls on the same
    machine with the same inputs produce identical predictions and select the
    same configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from ..data.context_features import encode_context

_SLOT_COLS: list[str] = ["empresaid", "direction", "pair_rank"]

# Number of lagged headway steps fed to the model = DL input window (T_in).
N_LAGS: int = 12

# Use validation rows for search + early stopping only when there are at least
# this many; tiny test fixtures (and corridors with no val rows) fall back to
# the frozen default configuration and a fixed number of trees.
_MIN_VAL_ROWS: int = 50

# Threads for the Kaggle CPU kernel (4 vCPU). Pinned in source: XGBoost `hist`
# is reproducible for a FIXED thread count, so this value is part of the
# determinism contract and must not be made environment-dependent.
_NTHREAD: int = 4

_NUM_BOOST_ROUND: int = 400
_EARLY_STOPPING_ROUNDS: int = 30

# Cheaper budget for the search sweep; the winner is refit at the full budget.
_SEARCH_NUM_BOOST_ROUND: int = 200
_SEARCH_EARLY_STOPPING_ROUNDS: int = 20

# Frozen fallback configuration (the pre-search hardcoded baseline). Used when
# there is no usable validation split, or when `search=False`.
_XGB_PARAMS: dict = {
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "nthread": _NTHREAD,
}

# ---------------------------------------------------------------------------
# Hyperparameter random search — validation-only selection.
# ---------------------------------------------------------------------------

# EXACTLY 24 configurations: the agreed budget for a Kaggle CPU kernel that
# must fit 2 corridors x 4 horizons within the session runtime limit.
SEARCH_N_CONFIGS: int = 24

# Fixed in source so the search is reproducible and cannot be silently
# re-rolled between runs. Changing this value changes the paper's numbers.
SEARCH_SEED: int = 20240718

# Discrete search space (|space| = 6*6*5*5*5*5 = 22500 >> 24 draws).
SEARCH_SPACE: dict[str, list] = {
    "eta": [0.02, 0.03, 0.05, 0.08, 0.12, 0.20],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}


def sample_search_configs(
    n_configs: int = SEARCH_N_CONFIGS, *, seed: int = SEARCH_SEED
) -> list[dict]:
    """Draw `n_configs` DISTINCT hyperparameter configurations, deterministically.

    Sampling depends only on `seed` and `SEARCH_SPACE` — never on the data — so
    every corridor and horizon evaluates the same candidate set and the sweep is
    exactly reproducible.

    Returns
    -------
    list[dict] — each dict holds only the searched keys (eta, max_depth,
    min_child_weight, subsample, colsample_bytree, lambda).
    """
    rng = np.random.default_rng(seed)
    keys = sorted(SEARCH_SPACE)  # sorted → draw order independent of dict order
    seen: set[tuple] = set()
    configs: list[dict] = []
    # Bounded loop: the space is ~22500 wide, so 24 distinct draws are reached
    # almost immediately; the cap only guards against a shrunken space.
    for _ in range(n_configs * 1000):
        if len(configs) == n_configs:
            break
        values = tuple(
            SEARCH_SPACE[k][int(rng.integers(len(SEARCH_SPACE[k])))] for k in keys
        )
        if values in seen:
            continue
        seen.add(values)
        configs.append(dict(zip(keys, values)))
    if len(configs) != n_configs:
        raise ValueError(
            f"sample_search_configs: could only draw {len(configs)} distinct "
            f"configurations out of {n_configs} requested"
        )
    return configs


@dataclass(frozen=True)
class B5FitResult:
    """Predictions plus the provenance needed to audit the fitted baseline.

    Attributes
    ----------
    predictions:
        Input frame (sorted by slot, t) with `y_pred_b5_xgb` added.
    best_params:
        The full parameter dict handed to XGBoost for the final fit.
    best_val_rmse:
        Validation RMSE of the selected configuration (``nan`` when no search
        ran, i.e. no usable validation split).
    best_iteration:
        Boosting iteration chosen by early stopping (-1 when unavailable).
    n_configs_evaluated:
        How many configurations the search actually fit (0 when it was skipped).
    search_seed:
        Seed used to draw the candidate configurations.
    used_atypical_flag:
        True when a non-empty atypical calendar was supplied.
    """

    predictions: pl.DataFrame
    best_params: dict = field(default_factory=dict)
    best_val_rmse: float = float("nan")
    best_iteration: int = -1
    n_configs_evaluated: int = 0
    search_seed: int = SEARCH_SEED
    used_atypical_flag: bool = False


def _build_features(
    headways: pl.DataFrame,
    *,
    horizon: int,
    atypical_dates: set[date] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Return the frame sorted by (slot, t) with lag + context feature columns
    added, plus the list of feature column names.

    lag_k (k = 1..N_LAGS) = headway value (forward-filled within slot) observed
    `horizon + k - 1` steps before the target row. lag_1 == B1 persistence.

    `_atypical` is the SAME binary flag the DL models consume: it is produced by
    `encode_context`, not reimplemented here, so the two model families cannot
    diverge on what counts as an atypical day.

    Raises
    ------
    ValueError
        If `atypical_dates` is an explicit empty set (fail closed — see the
        module docstring's atypical-flag contract).
    """
    if atypical_dates is not None and len(atypical_dates) == 0:
        raise ValueError(
            "_build_features: atypical_dates parsed to an EMPTY set. The "
            "atypical-day feature must not be silently disabled — pass None "
            "only when no atypical calendar exists at all."
        )

    lag_exprs = [
        pl.col("delta_t_min")
        .forward_fill()
        .shift(horizon + k - 1)
        .over(_SLOT_COLS)
        .alias(f"_lag_{k}")
        for k in range(1, N_LAGS + 1)
    ]
    df = (
        encode_context(headways, atypical_dates=atypical_dates)
        .sort(_SLOT_COLS + ["t"])
        .with_columns(
            *lag_exprs,
            pl.col("t").dt.hour().alias("_hour"),
            pl.col("t").dt.weekday().alias("_weekday"),
            pl.col("atypical_flag").alias("_atypical"),
        )
    )
    feature_cols = (
        [f"_lag_{k}" for k in range(1, N_LAGS + 1)]
        + ["_hour", "_weekday", "direction", "pair_rank", "_atypical"]
    )
    return df, feature_cols


def _random_search(
    xgb,
    dtrain,
    dval,
    *,
    seed: int,
    n_configs: int,
    search_seed: int,
) -> tuple[dict, float, int, int]:
    """Fit `n_configs` candidates and keep the one with the lowest VALIDATION RMSE.

    The only signal read here is `booster.best_score` on `dval` — the test split
    is not part of either DMatrix, so selection cannot see it.

    Returns
    -------
    (best_params, best_val_rmse, best_iteration, n_evaluated)
    """
    best_params: dict = {}
    best_score = float("inf")
    best_iteration = -1
    n_evaluated = 0

    for candidate in sample_search_configs(n_configs, seed=search_seed):
        params = dict(_XGB_PARAMS, **candidate, seed=seed)
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=_SEARCH_NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=_SEARCH_EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        n_evaluated += 1
        score = float(booster.best_score)
        # Strict `<` → first-drawn config wins ties, keeping selection
        # deterministic for a fixed candidate order.
        if score < best_score:
            best_score = score
            best_params = params
            best_iteration = int(getattr(booster, "best_iteration", -1))

    return best_params, best_score, best_iteration, n_evaluated


def fit_predict_b5_xgb(
    headways: pl.DataFrame,
    *,
    horizon: int = 1,
    seed: int = 42,
    atypical_dates: set[date] | None = None,
    search: bool = True,
    n_configs: int = SEARCH_N_CONFIGS,
    search_seed: int = SEARCH_SEED,
) -> B5FitResult:
    """Fit B5_XGB and return predictions plus the auditable search provenance.

    Parameters
    ----------
    headways:
        headways DataFrame with the `split` column attached. Columns consumed:
        empresaid, t, direction, pair_rank, delta_t_min, split.
    horizon:
        Forecast horizon in steps. lag_1 = shift(horizon) so the 1-lag feature
        equals B1 persistence; horizon=1 is the default.
    seed:
        XGBoost training seed (reproducible tree construction).
    atypical_dates:
        Atypical-day calendar (the same set the DL models receive). None means
        "not supplied" → all-zero flag; an explicit empty set raises.
    search:
        When True (default) run the seeded validation-only random search. When
        False, or when there is no usable validation split, the frozen default
        configuration is used.
    n_configs / search_seed:
        Search budget and sampler seed. Both default to the frozen constants;
        overriding them is a test/debug affordance, not a production path.

    Returns
    -------
    B5FitResult
    """
    import xgboost as xgb

    original_cols = headways.columns
    df, feature_cols = _build_features(
        headways, horizon=horizon, atypical_dates=atypical_dates
    )
    used_atypical = bool(atypical_dates)

    is_train = df["split"] == "train"
    is_val = df["split"] == "val"
    target_present = df["delta_t_min"].is_not_null()

    train_mask = (is_train & target_present).to_numpy()
    n_train = int(train_mask.sum())

    # Degenerate: nothing to fit on → null predictions (mirrors B0 on empty slots).
    if n_train == 0:
        return B5FitResult(
            predictions=df.select(original_cols).with_columns(
                pl.lit(None, dtype=pl.Float64).alias("y_pred_b5_xgb")
            ),
            used_atypical_flag=used_atypical,
        )

    X_all = df.select(feature_cols).to_numpy().astype(np.float64)
    y_all = df["delta_t_min"].to_numpy().astype(np.float64)

    dtrain = xgb.DMatrix(X_all[train_mask], label=y_all[train_mask], missing=np.nan)
    dall = xgb.DMatrix(X_all, missing=np.nan)

    val_mask = (is_val & target_present).to_numpy()
    has_val = int(val_mask.sum()) >= _MIN_VAL_ROWS

    params = dict(_XGB_PARAMS, seed=seed)
    best_val_rmse = float("nan")
    best_iteration = -1
    n_evaluated = 0

    if has_val:
        # NOTE: dval holds VALIDATION rows only. The test split is absent from
        # every DMatrix built here, so neither early stopping nor hyperparameter
        # selection can read it.
        dval = xgb.DMatrix(X_all[val_mask], label=y_all[val_mask], missing=np.nan)
        if search:
            params, best_val_rmse, _search_iter, n_evaluated = _random_search(
                xgb,
                dtrain,
                dval,
                seed=seed,
                n_configs=n_configs,
                search_seed=search_seed,
            )
        # Refit the selected configuration at the full boosting budget, keeping
        # the existing early-stopping-on-validation behaviour.
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=_NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_val_rmse = float(booster.best_score)
        best_iteration = int(getattr(booster, "best_iteration", -1))
    else:
        booster = xgb.train(params, dtrain, num_boost_round=_NUM_BOOST_ROUND)

    preds = booster.predict(dall).astype(np.float64)

    return B5FitResult(
        predictions=df.select(original_cols).with_columns(
            pl.Series("y_pred_b5_xgb", preds, dtype=pl.Float64)
        ),
        best_params=dict(params),
        best_val_rmse=best_val_rmse,
        best_iteration=best_iteration,
        n_configs_evaluated=n_evaluated,
        search_seed=search_seed,
        used_atypical_flag=used_atypical,
    )


def predict_b5_xgb(
    headways: pl.DataFrame,
    *,
    horizon: int = 1,
    seed: int = 42,
    atypical_dates: set[date] | None = None,
    search: bool = True,
) -> pl.DataFrame:
    """Add column `y_pred_b5_xgb`: gradient-boosted forecast of delta_t_min.

    Thin wrapper over :func:`fit_predict_b5_xgb` for callers that only need the
    predictions. See that function for the full parameter documentation.

    Returns
    -------
    pl.DataFrame — input frame (sorted by slot, t) with `y_pred_b5_xgb`
        (Float64 nullable) added. If the train split has no usable rows, the
        column is all-null.
    """
    return fit_predict_b5_xgb(
        headways,
        horizon=horizon,
        seed=seed,
        atypical_dates=atypical_dates,
        search=search,
    ).predictions
