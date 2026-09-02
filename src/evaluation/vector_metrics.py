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
from scipy.stats import beta

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


def bunching_score(residuals: pl.DataFrame, value_col: str) -> pl.Series:
    """How far below its vector's mean a cell sits, as a CONTINUOUS score.

    Higher means more bunched. ``bunching_flags`` is exactly
    ``bunching_score(...) > -BUNCHING_RATIO``, so this is the same detector with
    the decision left open.

    Why this exists
    ---------------
    ``BUNCHING_RATIO`` is an operating point calibrated on OBSERVED vectors,
    where the ratio has the realized dispersion. Transplanting it onto a
    predicted vector is not neutral: a conditional-mean forecast is compressed
    (CV 0.16 against a real 0.79), so the same cut lands three standard
    deviations into its left tail and the detector never fires. Scoring that as
    "the model cannot detect bunching" confuses a mis-set threshold with missing
    information.

    Ranking metrics computed on this score answer the question the fixed
    threshold cannot: does the model ORDER cells by bunching risk correctly,
    whatever scale its outputs happen to live on?
    """
    mean_over_vector = pl.col(value_col).mean().over(VECTOR_KEY)
    return residuals.select(
        pl.when(mean_over_vector > 0)
        .then(-pl.col(value_col) / mean_over_vector)
        .otherwise(None)
        .alias("_score")
    ).get_column("_score")


def ranking_scores(truth: np.ndarray, score: np.ndarray) -> dict:
    """Threshold-free discrimination: ROC-AUC and average precision.

    These are the honest way to ask whether a model carries information about an
    event, because they are invariant to any monotone rescaling of the score —
    exactly the transformation that a fixed relative threshold is NOT invariant
    to. ``ap_lift`` divides average precision by the base rate, so 1.0 means "no
    better than firing at random".
    """
    truth = np.asarray(truth, dtype=bool)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    truth, score = truth[ok], score[ok]

    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"auc": float("nan"), "average_precision": float("nan"),
                "ap_lift": float("nan")}

    # AUC via the Mann-Whitney U identity, with ties given average ranks so a
    # constant score scores exactly 0.5 instead of an artifact of sort order.
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=float)
    sorted_score = score[order]
    i = 0
    while i < sorted_score.size:
        j = i
        while j + 1 < sorted_score.size and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    auc = (ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # Average precision, descending score, ties broken pessimistically by taking
    # the step at the end of each tie group.
    desc = np.argsort(-score, kind="mergesort")
    t_sorted = truth[desc]
    tp_cum = np.cumsum(t_sorted)
    precision = tp_cum / np.arange(1, t_sorted.size + 1)
    average_precision = float(precision[t_sorted].sum() / n_pos)
    base_rate = n_pos / truth.size

    return {
        "auc": float(auc),
        "average_precision": average_precision,
        "ap_lift": average_precision / base_rate,
    }


def matthews_corrcoef(truth: np.ndarray, predicted: np.ndarray) -> float:
    """MCC — the summary F1 should have been all along for this comparison.

    F1 ignores true negatives, so it rewards any detector that merely fires at
    roughly the base rate. On these corridors a constant "always bunched"
    classifier outscores every real detector at h=10 on F1.

    For that degenerate rule MCC returns 0 here, and the justification has to be
    stated carefully because the loose version is wrong: always-fire gives
    FN = TN = 0, so the numerator AND the denominator are both zero and the ratio
    is indeterminate, not zero. Returning 0.0 is the continuity extension and the
    standard convention for degenerate confusion matrices; it also coincides with
    the expected MCC of a chance classifier. Do not write "0 by construction".
    """
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = float(np.sum(truth & predicted))
    tn = float(np.sum(~truth & ~predicted))
    fp = float(np.sum(~truth & predicted))
    fn = float(np.sum(truth & ~predicted))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def trivial_f1(base_rate: float) -> float:
    """F1 of the degenerate detector that flags EVERY cell as bunched.

    The floor every reported F1 must clear. Precision is the base rate, recall
    is 1, so F1 = 2b/(1+b). Any detector below this line is worse than not
    thinking at all, and a table without this column cannot show that.
    """
    return 2 * base_rate / (1 + base_rate) if base_rate > 0 else 0.0


def best_threshold(
    truth: np.ndarray, score: np.ndarray, objective: str = "mcc"
) -> float:
    """The score cut maximising ``objective`` — FIT on one window, APPLY to another.

    Never fit and score on the same rows: the point of this function is to give
    the learner the one free parameter that persistence gets for free by copying
    an observed vector, not to hand it an oracle.

    ``objective`` defaults to ``"mcc"`` rather than ``"f1"`` because F1
    maximisation degenerates at these base rates. On E2 the realized bunching
    rate is 30 %, so "flag everything" already scores F1 = 0.46, and the
    F1-optimal cut collapses to almost exactly that rule for BOTH models — a
    threshold with no discriminative content that nonetheless posts a
    respectable-looking F1. MCC is 0 for that rule, so maximising it cannot
    select it. ``"f1"`` remains available to reproduce the degeneracy on purpose.

    Candidate cuts are evaluated only at TIE-GROUP boundaries. Picking an
    interior point of a run of equal scores would return a cut whose ``>=``
    behaviour does not match the confusion matrix that selected it — the failure
    mode that matters exactly when a model's scores pile up, which is the
    situation under study here.
    """
    if objective not in {"mcc", "f1"}:
        raise ValueError(f"best_threshold: unknown objective {objective!r}")

    truth = np.asarray(truth, dtype=bool)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    truth, score = truth[ok], score[ok]
    if truth.size == 0 or truth.sum() == 0 or truth.all():
        return -BUNCHING_RATIO

    desc = np.argsort(-score, kind="mergesort")
    t_sorted, s_sorted = truth[desc], score[desc]

    # Last index of each run of equal scores: cutting there is the only choice
    # consistent with applying the returned value as ``score >= threshold``.
    boundary = np.flatnonzero(np.r_[np.diff(s_sorted) != 0, True])

    n_pos = float(t_sorted.sum())
    n = float(t_sorted.size)
    tp = np.cumsum(t_sorted)[boundary].astype(float)
    fired = (boundary + 1).astype(float)
    fp = fired - tp
    fn = n_pos - tp
    tn = n - fired - fn

    if objective == "f1":
        value = np.divide(2 * tp, 2 * tp + fp + fn,
                          out=np.zeros(tp.shape), where=(2 * tp + fp + fn) > 0)
    else:
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        value = np.divide(tp * tn - fp * fn, denom,
                          out=np.zeros(tp.shape), where=denom > 0)

    return float(s_sorted[boundary[int(np.argmax(value))]])


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


def precision_interval(
    hits: int, fires: int, level: float = 0.95
) -> tuple[float, float]:
    """Clopper-Pearson interval for precision, given ``hits`` of ``fires``.

    A precision computed off fourteen fires is a regime, not a value, and the
    tables report the point estimate alone. This bounds it.

    Clopper-Pearson rather than Wald or Wilson because the counts that need
    bounding here are the small ones: at fourteen trials Wald's normal
    approximation puts part of its interval outside [0, 1], and both
    approximations undercover. The exact interval is conservative instead, which
    is the direction an interval quoted in a paper should err.

    ``fires == 0`` raises rather than returning [0, 1]: a detector that never
    fired has no precision to bound, and the wide interval would read as a
    measurement.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"precision_interval: level must be in (0, 1), got {level}")
    if fires < 0 or hits < 0:
        raise ValueError(
            f"precision_interval: counts must be non-negative, got {hits}/{fires}"
        )
    if fires == 0:
        raise ValueError(
            "precision_interval: a detector that never fired has undefined precision"
        )
    if hits > fires:
        raise ValueError(
            f"precision_interval: {hits} hits cannot exceed {fires} fires"
        )

    alpha = 1.0 - level
    low = 0.0 if hits == 0 else float(beta.ppf(alpha / 2, hits, fires - hits + 1))
    high = (
        1.0
        if hits == fires
        else float(beta.ppf(1 - alpha / 2, hits + 1, fires - hits))
    )
    return low, high
