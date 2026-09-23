"""Which property of an event rule the compression reaches, and which it cannot.

Section V-B reports that the rule of Section III-B, carried without change onto
the forecast, stops firing. Section V-D closes one escape route: an absolute cut
in minutes, calibrated on an earlier window, does not rescue it either. Between
those two results sits a question neither answers. The published rule divides by
the mean of the very vector it is thresholding, and a reader is entitled to
suspect that self-reference is the whole mechanism. If it is, the finding is
about one invented rule; if it is not, it is about every rule shaped like it.

This builder runs the same event under three denominators, each applied
identically to the observed vector and to the forecast, on the same population
and the same origin as the rest of Section V.

``pred_mean``
    Half the mean of the vector being thresholded — the published rule. The
    denominator moves with whatever is being scored, so on the forecast side it
    is the forecast's own mean.

``obs_mean``
    Half the mean of the vector observed at the forecast origin. That vector is
    ``y_pred_persist``: the last step of the input window, which is what an
    operator holds at the moment of deciding. The denominator is therefore
    recomputed at every instant and still follows the corridor, but it is the
    same number on both sides and it does not follow the forecast. This is the
    arm that isolates self-reference.

``rank``
    The shortest positions of each vector, as many of them as the published rule
    marked on average in that cell at :data:`CALIBRATION_ORIGIN`, the earlier
    origin the recalibrated threshold is fitted on. No denominator at all: the
    cut is an order
    statistic. Because the count is fixed before either vector is looked at, this
    arm fires exactly as often as the event occurs, on both sides, by
    construction. It is stated here as algebra rather than offered as a result —
    it is the control that isolates level against rank.

What the arms are for
---------------------
The three pick out nearly the same event on observed vectors, which is what
makes their divergence on forecasts worth reporting; the table carries that
overlap so the claim is measured rather than asserted. Two of them name a LEVEL
in minutes and one names a POSITION, and that is the only property that
separates them. The realized cuts are carried in minutes for the same reason:
the rank rule raises its own level onto wherever the forecast's distribution
happens to sit, and the level rules cannot.

The last column is the uncomfortable one. Firing at the right volume is not the
same as firing at the right cells, so the rule that cannot collapse still has to
show what its detection is worth against the vector it failed to beat.

Outputs
-------
``docs/resultados/csv-multihorizon/threshold_denominators.csv``

Usage
-----
    uv run python -m src.build_threshold_denominators
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
    load_lstm,
)
from src.evaluation.vector_metrics import (  # noqa: E402
    BUNCHING_RATIO,
    MIN_VECTOR_LEN,
    VECTOR_KEY,
    bunching_score,
    detection_scores,
    matthews_corrcoef,
    ranking_scores,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "threshold_denominators.csv"

SCORING_ORIGIN = "main"

# The rank rule needs one number, the share of positions it marks, and it is
# read from the origin the recalibrated threshold of Section IV-A is fitted on.
# Read from the scored period instead, the quota would be set by the event rate
# it is then graded against.
CALIBRATION_ORIGIN = "r2"

# The published rule first: the table reads as a ladder away from it.
RULES: tuple[str, ...] = ("pred_mean", "obs_mean", "rank")

MODELS: tuple[tuple[str, str], ...] = (
    ("LSTM", "y_pred_model"),
    ("Persistence", "y_pred_persist"),
)

TRUTH = "y_true"
# The observed vector at the forecast origin. Persistence IS that vector carried
# forward, so its prediction column doubles as the causal denominator: nothing
# about the target instant enters it.
ORIGIN_VECTOR = "y_pred_persist"


def prepared(origin: str) -> pl.DataFrame:
    """Residuals restricted to vectors that have a shape, with their means.

    The filter is the one every other vector-level table applies: below
    :data:`MIN_VECTOR_LEN` positions there is no dispersion to threshold
    against, and the rule would be deciding on noise.
    """
    residuals = load_lstm(origin)
    lengths = residuals.group_by(VECTOR_KEY).agg(pl.len().alias("vector_len"))
    residuals = residuals.join(lengths, on=VECTOR_KEY, how="inner").filter(
        pl.col("vector_len") >= MIN_VECTOR_LEN
    )
    columns = [TRUTH] + [column for _, column in MODELS]
    return residuals.with_columns(
        [pl.col(c).mean().over(VECTOR_KEY).alias(f"_mean_{c}") for c in columns]
    )


def rank_rule_flags(
    frame: pl.DataFrame, value_col: str, base_rate: float
) -> pl.Series:
    """The ``round(base_rate * length)`` shortest positions of each vector.

    The count is decided by the vector's length and a rate fixed beforehand, so
    it is identical whether the column holds observations or forecasts. That is
    the property the arm exists to demonstrate: the number of alarms cannot
    depend on the scale the forecast happens to live on.

    Ties are broken by row order (``method="ordinal"``), which keeps the count
    exact and the output reproducible. A vector too short for the rate to reach
    one position fires nowhere, which is the honest reading and not a defect.
    """
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError(f"rank_rule_flags: base_rate out of range: {base_rate}")

    quota = (pl.col("vector_len") * base_rate).round(0)
    position = pl.col(value_col).rank(method="ordinal").over(VECTOR_KEY)
    return frame.select((position <= quota).alias("_flag")).get_column("_flag")


def _rank_cut(frame: pl.DataFrame, value_col: str, flag: pl.Series) -> pl.Series:
    """The largest value the rank rule fires on, per vector.

    The rank rule never names a level, so this is the only cut it has: the
    order statistic its quota happens to land on. Vectors whose quota rounds to
    zero fire nowhere and carry no cut, and they are the SAME vectors on both
    sides — the quota depends on the vector's length, not on its contents — so
    dropping them leaves the two sides comparable.
    """
    return (
        frame.with_columns(flag.alias("_f"))
        .select(
            pl.when(pl.col("_f"))
            .then(pl.col(value_col))
            .otherwise(None)
            .max()
            .over(VECTOR_KEY)
            .alias("_cut")
        )
        .get_column("_cut")
    )


def _mean_cut(cut: pl.Series) -> float:
    """One cut in minutes per arm, averaged over positions."""
    values = cut.drop_nulls()
    return float(values.mean()) if values.len() else float("nan")


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Overlap of two event definitions over the same positions."""
    union = int(np.sum(left | right))
    return float(np.sum(left & right)) / union if union else float("nan")


def _arm(
    cell: pl.DataFrame, rule: str, value_col: str, base_rate: float
) -> tuple[pl.Series, pl.Series, float, float]:
    """One rule's (truth, alarm) flags and the two cuts in minutes behind them.

    Both sides of an arm are built the same way; what changes between arms is
    only what the cut is made of.
    """
    if rule == "pred_mean":
        truth_cut = BUNCHING_RATIO * pl.col(f"_mean_{TRUTH}")
        alarm_cut = BUNCHING_RATIO * pl.col(f"_mean_{value_col}")
    elif rule == "obs_mean":
        # One denominator, shared. Nothing about the scored column enters it.
        truth_cut = alarm_cut = BUNCHING_RATIO * pl.col(f"_mean_{ORIGIN_VECTOR}")
    elif rule == "rank":
        truth = rank_rule_flags(cell, TRUTH, base_rate)
        alarm = rank_rule_flags(cell, value_col, base_rate)
        return (
            truth,
            alarm,
            _mean_cut(_rank_cut(cell, TRUTH, truth)),
            _mean_cut(_rank_cut(cell, value_col, alarm)),
        )
    else:
        raise ValueError(f"unknown rule: {rule!r}")

    computed = cell.select(
        (pl.col(TRUTH) < truth_cut).alias("t"),
        (pl.col(value_col) < alarm_cut).alias("a"),
        truth_cut.alias("tc"),
        alarm_cut.alias("ac"),
    )
    return (
        computed.get_column("t"),
        computed.get_column("a"),
        _mean_cut(computed.get_column("tc")),
        _mean_cut(computed.get_column("ac")),
    )


def _published_event(frame: pl.DataFrame) -> pl.Series:
    """The event of Section III-B on the observed vector."""
    return frame.select(
        (pl.col(TRUTH) < BUNCHING_RATIO * pl.col(f"_mean_{TRUTH}")).alias("t")
    ).get_column("t")


def calibration_rates() -> dict[tuple[str, int], float]:
    """Per cell, the published event rate on :data:`CALIBRATION_ORIGIN`."""
    fit = prepared(CALIBRATION_ORIGIN)
    rates: dict[tuple[str, int], float] = {}
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            cell = fit.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            if cell.height:
                rates[(corridor, horizon)] = float(_published_event(cell).mean())
    return rates


def build() -> pl.DataFrame:
    score = prepared(SCORING_ORIGIN)
    quota_rates = calibration_rates()
    rows: list[dict] = []

    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            cell = score.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            if cell.height == 0:
                continue

            # The published event on the scored period: what every arm's
            # overlap and the threshold-free verdict are measured against.
            published = _published_event(cell)
            published_np = published.to_numpy()
            quota_rate = quota_rates[(corridor, horizon)]

            # The verdict of Section V-C, carried alongside so each arm can be
            # asked the question that matters: does thresholding this way
            # reproduce what the same residuals say with no threshold at all?
            # Recomputed here rather than read from another table, so this
            # builder depends on residuals and on nothing generated.
            free = {
                name: ranking_scores(
                    published_np, bunching_score(cell, column).to_numpy()
                )["auc"]
                for name, column in MODELS
            }

            for rule in RULES:
                for name, value_col in MODELS:
                    truth, alarm, cut_truth, cut_alarm = _arm(
                        cell, rule, value_col, quota_rate
                    )
                    truth_np = truth.to_numpy()
                    alarm_np = alarm.to_numpy()
                    scores = detection_scores(truth_np, alarm_np)

                    rows.append(
                        {
                            "corridor": corridor,
                            "horizon": horizon,
                            "rule": rule,
                            "model": name,
                            "n_cells": scores.n,
                            "base_rate": scores.true_rate,
                            "quota_rate": (
                                quota_rate if rule == "rank" else float("nan")
                            ),
                            "fire_rate": scores.pred_rate,
                            "rate_ratio": (
                                scores.pred_rate / scores.true_rate
                                if scores.true_rate > 0 else float("nan")
                            ),
                            "cut_truth_min": cut_truth,
                            "cut_alarm_min": cut_alarm,
                            "tp": scores.tp,
                            "fp": scores.fp,
                            "fn": scores.fn,
                            "precision": scores.precision,
                            "recall": scores.recall,
                            "f1": scores.f1,
                            "mcc": matthews_corrcoef(truth_np, alarm_np),
                            "auc_published": free[name],
                            "jaccard_truth_vs_published": _jaccard(
                                truth_np, published_np
                            ),
                        }
                    )

    order = {rule: i for i, rule in enumerate(RULES)}
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("rule").replace_strict(order, return_dtype=pl.Int64).alias("_o")
        )
        .sort(["_o", "corridor", "horizon", "model"])
        .drop("_o")
    )


def threshold_free_agreement(table: pl.DataFrame) -> pl.DataFrame:
    """Per rule and cell: does the thresholded verdict match the unthresholded one?

    Both sides compare the same two methods over the same residuals. The only
    thing that differs is whether a cut was applied, so a disagreement is a
    property of the cut and of nothing else. This is what the three arms are
    finally judged on: an event rule that inverts the verdict its own data gives
    without a threshold is reporting the rule, not the corridor.
    """
    verdict = (
        table.pivot(values="mcc", index=["rule", "corridor", "horizon"], on="model")
        .with_columns(
            (pl.col("LSTM") > pl.col("Persistence")).alias("_thresholded")
        )
        .select("rule", "corridor", "horizon", "_thresholded")
    )
    free = (
        table.filter(pl.col("rule") == RULES[0])
        .pivot(values="auc_published", index=["corridor", "horizon"], on="model")
        .with_columns((pl.col("LSTM") > pl.col("Persistence")).alias("_free"))
        .select("corridor", "horizon", "_free")
    )
    return (
        verdict.join(free, on=["corridor", "horizon"], how="inner")
        .with_columns((pl.col("_thresholded") == pl.col("_free")).alias("agrees"))
        .sort(["rule", "corridor", "horizon"])
    )


def main() -> None:
    table = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=190, float_precision=4):
        for rule in RULES:
            print(f"\n=== {rule} — LSTM ===")
            print(
                table.filter(
                    (pl.col("rule") == rule) & (pl.col("model") == "LSTM")
                ).select(
                    "corridor", "horizon", "base_rate", "fire_rate", "rate_ratio",
                    "cut_truth_min", "cut_alarm_min", "precision", "recall",
                    "mcc", "jaccard_truth_vs_published",
                )
            )

    agreement = threshold_free_agreement(table)
    print("\nMediana de la razon disparo/evento (1.0 = dispara tan seguido como ocurre)")
    for rule in RULES:
        lstm = table.filter((pl.col("rule") == rule) & (pl.col("model") == "LSTM"))
        persist = table.filter(
            (pl.col("rule") == rule) & (pl.col("model") == "Persistence")
        )
        agrees = int(
            agreement.filter(pl.col("rule") == rule).get_column("agrees").sum()
        )
        print(
            f"  {rule:<10} LSTM {lstm.get_column('rate_ratio').median():.4f}"
            f"   persistencia {persist.get_column('rate_ratio').median():.4f}"
            f"   MCC LSTM {lstm.get_column('mcc').median():.4f}"
            f"   solape del evento {lstm.get_column('jaccard_truth_vs_published').median():.4f}"
            f"   coincide con el veredicto sin umbral {agrees}/12"
        )

    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
