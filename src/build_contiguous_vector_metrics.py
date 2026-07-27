"""Vector-level evaluation of the retrained pipeline — audit pending #5.

The claim under test is the project's own: that this forecasts «el vector
completo de headways» rather than N independent scalars. Grouped MAE cannot
distinguish the two, so this builder reports the three vector metrics the audit
named — positional error profile, service regularity, joint bunching detection —
computed over the same three-way paired population as every other verdict.

The metrics are constructed so a negative answer is expressible. If the error
profile is flat, if predicted regularity tracks the truth no better than
persistence does, and if bunching recall is negligible, then the vector framing
is not supported and the claim has to be narrowed. That outcome is a result, not
a build failure.

Outputs
-------
``docs/resultados/csv-multihorizon/contiguous_error_profile.csv``
    MAE per corridor x horizon x pair_rank x model.
``docs/resultados/csv-multihorizon/contiguous_vector_metrics.csv``
    Per corridor x horizon x model: CV fidelity and bunching detection.

Usage
-----
    uv run python -m src.build_contiguous_vector_metrics
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_significance import (  # noqa: E402
    CORRIDORS,
    HORIZONS,
    XGB_CSV,
    load_lstm,
)
from src.evaluation.residual_export import RESIDUAL_KEY_COLUMNS  # noqa: E402
from src.evaluation.vector_metrics import (  # noqa: E402
    MIN_VECTOR_LEN,
    bunching_flags,
    detection_scores,
    error_profile,
    regularity_error,
    vector_frame,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
PROFILE_CSV = OUT_DIR / "contiguous_error_profile.csv"
VECTOR_CSV = OUT_DIR / "contiguous_vector_metrics.csv"

# (display name, prediction column). Persistence is a full competitor here: it
# propagates the observed vector unchanged, which is a strong regularity
# predictor and a useless bunching predictor, and both show up below.
MODELS: tuple[tuple[str, str], ...] = (
    ("LSTM", "y_pred_model"),
    ("XGBoost", "y_pred_xgb"),
    ("Persistence", "y_pred_persist"),
)


def load_paired() -> pl.DataFrame:
    """The three-way paired population, the same one every verdict uses."""
    lstm = load_lstm()
    xgb = pl.read_csv(XGB_CSV, try_parse_dates=True)
    return lstm.join(
        xgb.select(RESIDUAL_KEY_COLUMNS + ["y_pred_model"]).rename(
            {"y_pred_model": "y_pred_xgb"}
        ),
        on=RESIDUAL_KEY_COLUMNS,
        how="inner",
    )


def build_profile(paired: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for name, column in MODELS:
        frames.append(
            error_profile(paired, column).with_columns(pl.lit(name).alias("model"))
        )
    return pl.concat(frames).select(
        ["model", "corridor", "horizon", "pair_rank", "n", "mean_headway", "mae"]
    ).sort(["model", "corridor", "horizon", "pair_rank"])


def build_vector_metrics(paired: pl.DataFrame) -> pl.DataFrame:
    value_cols = ["y_true"] + [column for _, column in MODELS]

    # Bunching is per cell, so the flags are computed on the residual frame and
    # then restricted to the vectors that survive the length filter.
    flagged = paired.with_columns(
        [
            bunching_flags(paired, column).alias(f"_bunch_{column}")
            for column in value_cols
        ]
    )
    lengths = paired.group_by(["corridor", "direction", "horizon", "start_ts"]).agg(
        pl.len().alias("vector_len")
    )
    flagged = flagged.join(
        lengths, on=["corridor", "direction", "horizon", "start_ts"], how="inner"
    ).filter(pl.col("vector_len") >= MIN_VECTOR_LEN)

    vectors = vector_frame(paired, value_cols)

    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            where = (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            cell_vectors = vectors.filter(where)
            cell_cells = flagged.filter(where)
            if cell_vectors.height == 0:
                continue

            truth = cell_cells.get_column("_bunch_y_true").to_numpy()
            for name, column in MODELS:
                regularity = regularity_error(cell_vectors, column)
                scores = detection_scores(
                    truth, cell_cells.get_column(f"_bunch_{column}").to_numpy()
                )
                rows.append(
                    {
                        "model": name,
                        "corridor": corridor,
                        "horizon": horizon,
                        **regularity,
                        "n_cells": scores.n,
                        "bunching_rate_true": scores.true_rate,
                        "bunching_rate_pred": scores.pred_rate,
                        "bunching_tp": scores.tp,
                        "bunching_fp": scores.fp,
                        "bunching_fn": scores.fn,
                        "bunching_precision": scores.precision,
                        "bunching_recall": scores.recall,
                        "bunching_f1": scores.f1,
                    }
                )

    return pl.DataFrame(rows).sort(["model", "corridor", "horizon"])


def profile_shape(profile: pl.DataFrame) -> pl.DataFrame:
    """Is the positional error profile flat, or does position carry information?

    Reported as the spread between the best and worst position relative to the
    cell's mean MAE. A few percent is flat; a large ratio means the vector has
    structure a scalar metric was averaging away.
    """
    return (
        profile.group_by(["model", "corridor", "horizon"])
        .agg(
            pl.col("mae").min().alias("mae_min_position"),
            pl.col("mae").max().alias("mae_max_position"),
            pl.col("mae").mean().alias("mae_mean"),
            pl.len().alias("n_positions"),
        )
        .with_columns(
            (
                (pl.col("mae_max_position") - pl.col("mae_min_position"))
                / pl.col("mae_mean")
            ).alias("relative_spread")
        )
        .sort(["model", "corridor", "horizon"])
    )


def vector_verdict(vector: pl.DataFrame) -> pl.DataFrame:
    """Which model wins each vector metric, per cell.

    Kept separate from the scalar verdicts on purpose: a model can own the MAE
    table and lose every column here, and that dissociation is the result.
    """
    best_cv = (
        vector.sort("cv_mae")
        .group_by(["corridor", "horizon"], maintain_order=True)
        .first()
        .select(["corridor", "horizon", pl.col("model").alias("best_regularity")])
    )
    best_f1 = (
        vector.sort("bunching_f1", descending=True)
        .group_by(["corridor", "horizon"], maintain_order=True)
        .first()
        .select(
            ["corridor", "horizon",
             pl.col("model").alias("best_bunching"),
             pl.col("bunching_f1").alias("best_f1")]
        )
    )
    lstm = vector.filter(pl.col("model") == "LSTM").select(
        ["corridor", "horizon",
         pl.col("cv_bias").alias("lstm_cv_bias"),
         pl.col("bunching_recall").alias("lstm_recall"),
         pl.col("bunching_f1").alias("lstm_f1")]
    )
    return (
        best_cv.join(best_f1, on=["corridor", "horizon"])
        .join(lstm, on=["corridor", "horizon"])
        .with_columns(
            (pl.col("best_f1") / pl.col("lstm_f1")).alias("f1_ratio_over_lstm")
        )
        .sort(["corridor", "horizon"])
    )


def main() -> None:
    paired = load_paired()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profile = build_profile(paired)
    profile.write_csv(PROFILE_CSV)

    vector = build_vector_metrics(paired)
    vector.write_csv(VECTOR_CSV)

    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=200):
        print("Positional error profile — is it flat?")
        print(
            profile_shape(profile)
            .filter(pl.col("model") == "LSTM")
            .select(["corridor", "horizon", "n_positions", "mae_min_position",
                     "mae_max_position", "relative_spread"])
        )
        print("\nService regularity (CV of the headway vector):")
        print(
            vector.select(
                ["model", "corridor", "horizon", "n_vectors", "mean_cv_true",
                 "mean_cv_pred", "cv_bias", "cv_mae", "cv_correlation"]
            )
        )
        print("\nJoint bunching detection:")
        print(
            vector.select(
                ["model", "corridor", "horizon", "bunching_rate_true",
                 "bunching_rate_pred", "bunching_precision", "bunching_recall",
                 "bunching_f1"]
            )
        )

        print("\nWho wins each vector metric:")
        print(vector_verdict(vector))

    print(f"\nWrote {PROFILE_CSV.relative_to(REPO_ROOT)} ({profile.height} rows)")
    print(f"Wrote {VECTOR_CSV.relative_to(REPO_ROOT)} ({vector.height} rows)")


if __name__ == "__main__":
    main()
