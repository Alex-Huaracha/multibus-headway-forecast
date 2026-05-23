"""Evaluation harness for classical baseline comparison — Fase 3.

Public API:
    evaluate_corridor(headways: pl.DataFrame, corridor_name: str) -> pl.DataFrame

The function composes the full pipeline for one corridor:
    split_temporal → winsorize_train_p99 → predict_b0/b1/b2(×3)/b3/b4_ha
    → filter test rows → compute MAE + RMSE per (direction, baseline)
    → return tidy long-form DataFrame.

Output schema (design §6):
    corridor   Utf8
    direction  Utf8   — "-1", "+1", "aggregate"
    baseline   Utf8   — "B0", "B1", "B2_w5", "B2_w10", "B2_w15", "B3", "B4_HA"
    metric     Utf8   — "MAE", "RMSE"
    value      Float64 — minutes

Rows per corridor: 3 directions × 7 baselines × 2 metrics = 42.
Both corridors together (notebook caller): 84 rows.

Design decisions (locked in design §6 and §9):
  - "aggregate" direction = MAE/RMSE over POOLED test rows (both directions
    concatenated), NOT mean of per-direction metrics.
  - val rows are NEVER consumed (B3-VAL-UNUSED).
  - No new pyproject.toml dependencies.
  - harness.py does NOT read parquets or write CSV (notebook does those).
"""
from __future__ import annotations

import polars as pl

from ..evaluation.metrics import mae, rmse
from ..evaluation.splits import split_temporal, winsorize_train_p99
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


def evaluate_corridor(
    headways: pl.DataFrame,
    corridor_name: str,
) -> pl.DataFrame:
    """Run all classical baselines on one corridor and return a tidy metrics table.

    Parameters
    ----------
    headways:
        Raw headways DataFrame with R7 v4 schema columns:
        empresaid, t, direction, pair_rank, delta_t_min.
        Must NOT already have a `split` column (this function adds it).
    corridor_name:
        Label for the `corridor` column in the output (e.g. "E2", "E59").

    Returns
    -------
    pl.DataFrame — 42-row tidy long-form table with schema:
        [corridor, direction, baseline, metric, value]

    Notes
    -----
    - val rows are ignored at prediction time (baselines consume train only)
      and are never included in metric computation (metrics use test rows only).
    - The "aggregate" direction row pools test rows from both directions before
      computing MAE/RMSE — it is NOT the mean of the two per-direction metrics.
    """
    # --- Pipeline: split → winsorize → all baselines ---
    df = split_temporal(headways)
    df, _threshold = winsorize_train_p99(df)

    df = predict_b0(df)
    df = predict_b1(df)
    for w in BASELINE_B2_WINDOWS:
        df = predict_b2(df, window=w)
    df = predict_b3(df)
    df = predict_b4_ha(df)

    # --- Filter to test rows only (B3-VAL-UNUSED) ---
    test_df = df.filter(pl.col("split") == "test")

    # --- Compute metrics per (direction × baseline) ---
    rows: list[dict] = []

    for pred_col, baseline_name in _BASELINE_MAP:
        for direction_val in (-1, 1, "aggregate"):
            if direction_val == "aggregate":
                # Pool all test rows regardless of direction.
                subset = test_df
                direction_str = "aggregate"
            else:
                subset = test_df.filter(pl.col("direction") == direction_val)
                direction_str = f"+{direction_val}" if direction_val > 0 else str(direction_val)

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
