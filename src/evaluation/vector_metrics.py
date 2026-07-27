"""Metrics that evaluate the headway VECTOR, not N independent scalars.

Audit pending #5: the work claims to forecast «el vector completo de headways»
while every reported number is a grouped scalar MAE or RMSE. As evaluated, the
result is indistinguishable from N separate scalar forecasts — so either a
vector-level metric appears, or the claim has to be reformulated.

This module supplies the three the audit named, and they are chosen so that each
one *could* come out against the claim:

``error_profile``
    MAE per ``pair_rank``. If the error is flat along the vector there is no
    positional structure to speak of and the vector framing adds nothing; if it
    is shaped, the position matters and the claim has something behind it.

``regularity``
    The coefficient of variation of a snapshot's headway vector — the standard
    service-regularity measure in transit operations. It is a property of the
    vector as a whole: a model can predict every individual headway acceptably
    and still destroy the SHAPE, which is what an operator actually acts on.

``bunching``
    Joint event detection. ``propuesta.md`` defines bunching as one headway
    collapsing while its neighbour grows — an anomaly that «solo existe en el
    patrón colectivo». A cell is flagged when it falls below
    ``BUNCHING_RATIO`` of its own vector's mean, so the flag is relative to the
    corridor's current state rather than to an absolute minute count. Crucially,
    the predicted flag is derived from the PREDICTED vector's own mean: an
    operator has no access to the true mean, so scoring against it would measure
    something nobody could deploy.

Scalar MAE cannot produce any of these. That is the point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# Columns identifying one predicted vector: a corridor/direction snapshot at one
# horizon. ``pair_rank`` indexes position WITHIN it.
VECTOR_KEY: list[str] = ["corridor", "direction", "horizon", "start_ts"]

# A headway below half the vector's current mean is bunched. The relative form
# is what makes the flag comparable across corridors running different
# frequencies; ``propuesta.md``'s worked example (5 min -> 2 min against a ~6 min
# mean) sits comfortably inside it.
BUNCHING_RATIO = 0.5

# Below three positions a coefficient of variation is dominated by its own
# sampling noise and "the shape of the vector" is not a meaningful notion.
MIN_VECTOR_LEN = 3


@dataclass(frozen=True)
class DetectionScores:
    """Confusion-matrix summary for one predictor's bunching flags."""

    n: int
    true_rate: float
    pred_rate: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def error_profile(residuals: pl.DataFrame, pred_col: str) -> pl.DataFrame:
    """MAE per ``(corridor, horizon, pair_rank)`` for one prediction column.

    The profile is the cheapest test of whether position carries information at
    all. A flat profile is a real answer, not a failed measurement.
    """
    return (
        residuals.with_columns(
            (pl.col("y_true") - pl.col(pred_col)).abs().alias("_ae")
        )
        .group_by(["corridor", "horizon", "pair_rank"])
        .agg(
            pl.len().alias("n"),
            pl.col("_ae").mean().alias("mae"),
            pl.col("y_true").mean().alias("mean_headway"),
        )
        .sort(["corridor", "horizon", "pair_rank"])
    )


def vector_frame(residuals: pl.DataFrame, value_cols: list[str]) -> pl.DataFrame:
    """Per-vector mean, std and length for every requested column.

    Vectors shorter than :data:`MIN_VECTOR_LEN` are dropped. They are not a
    defect — a snapshot with two active buses genuinely has no shape — but they
    would otherwise dominate the CV distribution with noise.
    """
    aggregations = [pl.len().alias("vector_len")]
    for col in value_cols:
        aggregations.append(pl.col(col).mean().alias(f"{col}_mean"))
        # ddof=1: the vector is a sample of the corridor's headways, and with
        # three to six positions the population form is visibly biased low.
        aggregations.append(pl.col(col).std(ddof=1).alias(f"{col}_std"))

    frame = (
        residuals.group_by(VECTOR_KEY)
        .agg(aggregations)
        .filter(pl.col("vector_len") >= MIN_VECTOR_LEN)
    )

    return frame.with_columns(
        [
            # Guard the ratio: a vector whose mean is non-positive has no
            # meaningful CV, and headways at or below zero are not physical.
            pl.when(pl.col(f"{col}_mean") > 0)
            .then(pl.col(f"{col}_std") / pl.col(f"{col}_mean"))
            .otherwise(None)
            .alias(f"{col}_cv")
            for col in value_cols
        ]
    ).sort(VECTOR_KEY)


def regularity_error(vectors: pl.DataFrame, pred_col: str) -> dict:
    """How well a model reproduces the vector's coefficient of variation.

    ``bias`` is signed: negative means the model predicts a SMOOTHER corridor
    than the real one. That is the failure mode worth naming — a model that
    regresses toward the mean scores well on scalar MAE while systematically
    under-reporting the irregularity an operator needs to see.
    """
    true_cv = vectors.get_column("y_true_cv").to_numpy()
    pred_cv = vectors.get_column(f"{pred_col}_cv").to_numpy()
    ok = np.isfinite(true_cv) & np.isfinite(pred_cv)
    true_cv, pred_cv = true_cv[ok], pred_cv[ok]
    if true_cv.size < 2:
        raise ValueError("regularity_error: need at least 2 comparable vectors")

    correlation = float(np.corrcoef(true_cv, pred_cv)[0, 1])
    return {
        "n_vectors": int(true_cv.size),
        "mean_cv_true": float(true_cv.mean()),
        "mean_cv_pred": float(pred_cv.mean()),
        "cv_bias": float((pred_cv - true_cv).mean()),
        "cv_mae": float(np.abs(pred_cv - true_cv).mean()),
        "cv_correlation": correlation,
    }


def bunching_flags(residuals: pl.DataFrame, value_col: str) -> pl.Series:
    """True where a cell's headway is below ``BUNCHING_RATIO`` of its vector mean.

    The mean is taken over the SAME column being flagged. For a prediction that
    means the predicted vector's own mean — the only quantity available at
    prediction time.
    """
    mean_over_vector = pl.col(value_col).mean().over(VECTOR_KEY)
    return residuals.select(
        (pl.col(value_col) < BUNCHING_RATIO * mean_over_vector).alias("_flag")
    ).get_column("_flag")


def detection_scores(truth: np.ndarray, predicted: np.ndarray) -> DetectionScores:
    """Precision / recall / F1 of predicted bunching against realized bunching."""
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError(
            f"detection_scores: shape mismatch {truth.shape} vs {predicted.shape}"
        )

    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))

    # A model that never fires has undefined precision; reporting 0.0 is the
    # honest reading — it detected nothing — and keeps F1 well defined.
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return DetectionScores(
        n=int(truth.size),
        true_rate=float(truth.mean()) if truth.size else 0.0,
        pred_rate=float(predicted.mean()) if predicted.size else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
