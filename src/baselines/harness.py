"""Evaluation harness for classical baseline comparison — Fase 3.

Public API:
    run_corridor(headways, corridor_name, ...) -> CorridorRun
        metrics + per-sample XGB residuals + fitted-baseline provenance.
    evaluate_corridor(headways: pl.DataFrame, corridor_name: str) -> pl.DataFrame
        Metrics only (thin wrapper over run_corridor; unchanged contract).

The function composes the full pipeline for one corridor:
    split_temporal → winsorize_train_p99 → predict_b0/b1/b2(×3)/b3/b4_ha
    [→ predict_b5_xgb when include_fitted]
    → filter test rows → compute MAE + RMSE per (direction, baseline)
    → return tidy long-form DataFrame.

Output schema (design §6):
    corridor   Utf8
    direction  Utf8   — "-1", "+1", "aggregate"
    baseline   Utf8   — "B0", "B1", "B2_w5", "B2_w10", "B2_w15", "B3", "B4_HA"
                        [, "B5_XGB" when include_fitted]
    metric     Utf8   — "MAE", "RMSE"
    value      Float64 — minutes

Rows per corridor: 3 directions × N baselines × 2 metrics.
    include_fitted=True  (default): N = 8 → 48 rows per corridor.
    include_fitted=False (formulaic-only): N = 7 → 42 rows per corridor.

Design decisions (locked in design §6 and §9):
  - "aggregate" direction = MAE/RMSE over POOLED test rows (both directions
    concatenated), NOT mean of per-direction metrics.
  - val rows are NEVER consumed by the formulaic baselines B0-B4 (B3-VAL-UNUSED).
    The fitted baseline B5_XGB MAY use val rows for early stopping (only when
    there are enough), which is correct practice for a learned model and mirrors
    how the DL models were tuned.
  - B5_XGB (the fitted ML baseline) adds an xgboost dependency; it lives in
    fitted.py and is opt-out via include_fitted=False.
  - harness.py does NOT read parquets or write CSV (notebook does those).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from ..evaluation.metrics import mae, rmse
from ..evaluation.splits import split_temporal, winsorize_train_p99
from .fitted import B5FitResult, fit_predict_b5_xgb
from .statistical import (
    BASELINE_B2_WINDOWS,
    predict_b0,
    predict_b1,
    predict_b2,
    predict_b3,
    predict_b4_ha,
)

# Map from prediction column name → display name for the output DataFrame.
_BASELINE_MAP: list[tuple[str, str]] = [
    ("y_pred_b0", "B0"),
    ("y_pred_b1", "B1"),
    ("y_pred_b2_w5", "B2_w5"),
    ("y_pred_b2_w10", "B2_w10"),
    ("y_pred_b2_w15", "B2_w15"),
    ("y_pred_b3", "B3"),
    ("y_pred_b4_ha", "B4_HA"),
]

# The fitted ML baseline is appended only when include_fitted=True.
_FITTED_ENTRY: tuple[str, str] = ("y_pred_b5_xgb", "B5_XGB")

# Per-sample residual export for the paired significance tests (DM / Wilcoxon).
# `t` is the join key: without it the XGBoost residuals cannot be paired with
# any other model's per-sample errors.
XGB_RESIDUAL_COLUMNS: list[str] = [
    "corridor",
    "direction",
    "horizon",
    "t",
    "y_true",
    "y_pred_xgb",
    "y_pred_persist",
]


@dataclass(frozen=True)
class CorridorRun:
    """Everything one corridor x horizon run produces.

    Attributes
    ----------
    metrics:
        Tidy long-form MAE/RMSE table (the historical `evaluate_corridor` output).
    residuals:
        Per-sample TEST residuals for B5_XGB paired with B1 persistence, with the
        `t` join key. Empty frame when `include_fitted=False`.
    fit_result:
        Provenance of the fitted baseline (winning hyperparameters, validation
        RMSE, search budget). None when `include_fitted=False`.
    """

    metrics: pl.DataFrame
    residuals: pl.DataFrame = field(default_factory=lambda: pl.DataFrame([]))
    fit_result: B5FitResult | None = None


def _direction_label(direction_val: int) -> str:
    """Signed direction label ("-1" / "+1") shared with the DL residual exports."""
    return f"+{direction_val}" if direction_val > 0 else str(direction_val)


def _build_xgb_residuals(
    test_df: pl.DataFrame, corridor_name: str, horizon: int
) -> pl.DataFrame:
    """Per-sample paired TEST residuals: B5_XGB vs B1 persistence.

    Keeps only samples where the target AND both predictions are present — the
    paired set the significance tests require.
    """
    return (
        test_df.filter(
            pl.col("delta_t_min").is_not_null()
            & pl.col("y_pred_b5_xgb").is_not_null()
            & pl.col("y_pred_b1").is_not_null()
        )
        .with_columns(
            pl.lit(corridor_name).alias("corridor"),
            pl.col("direction")
            .map_elements(_direction_label, return_dtype=pl.Utf8)
            .alias("direction"),
            pl.lit(horizon, dtype=pl.Int64).alias("horizon"),
            pl.col("delta_t_min").cast(pl.Float64).alias("y_true"),
            pl.col("y_pred_b5_xgb").cast(pl.Float64).alias("y_pred_xgb"),
            pl.col("y_pred_b1").cast(pl.Float64).alias("y_pred_persist"),
        )
        .select(XGB_RESIDUAL_COLUMNS)
        .sort(["corridor", "direction", "t"])
    )


def run_corridor(
    headways: pl.DataFrame,
    corridor_name: str,
    *,
    horizon: int = 1,
    include_fitted: bool = True,
    atypical_dates: set[date] | None = None,
) -> CorridorRun:
    """Full pipeline for one corridor: metrics + XGB residuals + fit provenance.

    Same pipeline and contracts as :func:`evaluate_corridor` (which delegates
    here and returns only `metrics`), plus the two artifacts the paper needs to
    defend the fitted baseline: the per-sample paired residuals and the winning
    hyperparameter configuration.

    Parameters
    ----------
    atypical_dates:
        Atypical-day calendar forwarded to B5_XGB so the fitted baseline sees
        the same context feature as the DL models. An explicit empty set raises
        (fail closed); None means no calendar was supplied.
    """
    # --- Pipeline: split → winsorize → all baselines ---
    # INV-6: the p99 threshold is computed on TRAIN only and applied to ALL
    # splits — winsorize_train_p99 receives the full split-tagged frame.
    df = split_temporal(headways)
    df, _threshold = winsorize_train_p99(df)

    df = predict_b0(df)
    df = predict_b1(df, horizon=horizon)
    for w in BASELINE_B2_WINDOWS:
        df = predict_b2(df, window=w, horizon=horizon)
    df = predict_b3(df, horizon=horizon)
    df = predict_b4_ha(df)

    baseline_map = list(_BASELINE_MAP)
    fit_result: B5FitResult | None = None
    if include_fitted:
        fit_result = fit_predict_b5_xgb(
            df, horizon=horizon, atypical_dates=atypical_dates
        )
        df = fit_result.predictions
        baseline_map = baseline_map + [_FITTED_ENTRY]

    # --- Filter to test rows only (B3-VAL-UNUSED) ---
    test_df = df.filter(pl.col("split") == "test")

    metrics = _metrics_table(test_df, corridor_name, baseline_map)
    residuals = (
        _build_xgb_residuals(test_df, corridor_name, horizon)
        if include_fitted
        else pl.DataFrame([])
    )
    return CorridorRun(metrics=metrics, residuals=residuals, fit_result=fit_result)


def _metrics_table(
    test_df: pl.DataFrame,
    corridor_name: str,
    baseline_map: list[tuple[str, str]],
) -> pl.DataFrame:
    """MAE/RMSE per (direction x baseline) over the TEST rows, long-form."""
    rows: list[dict] = []

    for pred_col, baseline_name in baseline_map:
        for direction_val in (-1, 1, "aggregate"):
            if direction_val == "aggregate":
                # Pool all test rows regardless of direction.
                subset = test_df
                direction_str = "aggregate"
            else:
                subset = test_df.filter(pl.col("direction") == direction_val)
                direction_str = _direction_label(direction_val)

            y_true = subset["delta_t_min"]
            y_pred = subset[pred_col]

            # Compute MAE and RMSE (null rows are masked inside the functions).
            mae_val = mae(y_true, y_pred)
            rmse_val = rmse(y_true, y_pred)

            rows.append(
                {
                    "corridor": corridor_name,
                    "direction": direction_str,
                    "baseline": baseline_name,
                    "metric": "MAE",
                    "value": mae_val,
                }
            )
            rows.append(
                {
                    "corridor": corridor_name,
                    "direction": direction_str,
                    "baseline": baseline_name,
                    "metric": "RMSE",
                    "value": rmse_val,
                }
            )

    return pl.DataFrame(rows).with_columns(
        pl.col("corridor").cast(pl.Utf8),
        pl.col("direction").cast(pl.Utf8),
        pl.col("baseline").cast(pl.Utf8),
        pl.col("metric").cast(pl.Utf8),
        pl.col("value").cast(pl.Float64),
    )


def evaluate_corridor(
    headways: pl.DataFrame,
    corridor_name: str,
    *,
    horizon: int = 1,
    include_fitted: bool = True,
    atypical_dates: set[date] | None = None,
) -> pl.DataFrame:
    """Run all classical baselines on one corridor and return a tidy metrics table.

    Thin wrapper over :func:`run_corridor` kept for the existing callers that
    only need the metrics table.

    Parameters
    ----------
    headways:
        Raw headways DataFrame with R7 v4 schema columns:
        empresaid, t, direction, pair_rank, delta_t_min.
        Must NOT already have a `split` column (this function adds it).
    corridor_name:
        Label for the `corridor` column in the output (e.g. "E2", "E59").
    horizon:
        Prediction horizon in steps. Default 1 reproduces the original behavior
        exactly. B1, B2, B3, and B5_XGB are horizon-aware and receive this value.
        B0 and B4_HA are horizon-agnostic (constant/lookup predictors) and are
        not affected.
    include_fitted:
        When True (default), also runs the fitted ML baseline B5_XGB (xgboost).
        When False, only the formulaic baselines B0-B4 run (no xgboost import).

    Returns
    -------
    pl.DataFrame — tidy long-form table (48 rows with include_fitted, else 42):
        [corridor, direction, baseline, metric, value]

    Notes
    -----
    - val rows are ignored at prediction time (baselines consume train only)
      and are never included in metric computation (metrics use test rows only).
    - The "aggregate" direction row pools test rows from both directions before
      computing MAE/RMSE — it is NOT the mean of the two per-direction metrics.
    """
    return run_corridor(
        headways,
        corridor_name,
        horizon=horizon,
        include_fitted=include_fitted,
        atypical_dates=atypical_dates,
    ).metrics
