"""Two things the threshold argument needs before it can be published.

Section 5.3 shows that a bunching cut calibrated in observation space is not
transportable onto a point forecast. The literature survey of 2026-07-29 exposed
two holes in that argument, and this builder closes both. Neither needs a GPU:
the residual exports already carry every column.

HOLE 1 — the rule is OURS, not the field's
------------------------------------------
The relative form that dominates the literature is a fraction of the SCHEDULED
headway (a quarter in Yu et al. 2016 and Moreira-Matias et al. 2016; a half in
the TCQSM). The "fraction of the vector's own mean" form does not exist as a
published event definition. We use it because these are raw GPS with no GTFS
schedule, so the vector mean is the available stand-in.

That makes the finding scoped to SELF-REFERENTIAL thresholds — and a reviewer
will ask whether the collapse is an artifact of the self-reference alone. So the
same detection is recomputed here under an ABSOLUTE cut in minutes, calibrated on
an earlier disjoint window and identical for truth and forecast:

    K = ratio * median(y_true on the calibration origin), per (corridor, direction)

Two ratios are reported: 0.5 to stay comparable with our own rule, and 0.25 to
match the field's dominant convention. The absolute cut is NOT self-referential —
its denominator does not move with the forecast — so:

    if the learner still under-fires   -> the collapse is about the compressed
                                          marginal distribution, and the finding
                                          survives beyond our invented rule
    if the learner fires at ~base rate -> the self-reference was the whole story,
                                          and the scope claim must shrink

Either outcome is worth reporting. The one thing we cannot do is leave it unasked.

HOLE 2 — no published comparison of threshold-fitting stability
---------------------------------------------------------------
Lipton et al. (2014) prove F1-maximising threshold selection degenerates to
"flag everything" for an uninformative classifier, and note it is high variance.
Nobody has compared the STABILITY of F1- versus MCC-fitted thresholds. We have
three disjoint test windows and both objectives already implemented, so the
measurement is nearly free and it is the empirical warrant for a choice that is
currently justified only by citation.

Reported per (corridor, horizon, model): the spread of the fitted cut across the
three origins under each objective, plus the out-of-sample cost — fit on r1 and
on r2, apply both to main, and see how much the achieved MCC moves depending on
which window you happened to calibrate on.

Outputs
-------
``docs/resultados/csv-multihorizon/threshold_absolute_comparison.csv``
``docs/resultados/csv-multihorizon/threshold_stability.csv``

Usage
-----
    uv run python -m src.build_threshold_robustness
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
from src.build_rolling_origin_significance import ORIGINS  # noqa: E402
from src.evaluation.vector_metrics import (  # noqa: E402
    BUNCHING_RATIO,
    MIN_VECTOR_LEN,
    VECTOR_KEY,
    best_threshold,
    bunching_flags,
    bunching_score,
    detection_scores,
    matthews_corrcoef,
    ranking_scores,
    trivial_f1,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_ABSOLUTE = OUT_DIR / "threshold_absolute_comparison.csv"
OUT_STABILITY = OUT_DIR / "threshold_stability.csv"

MODELS: tuple[tuple[str, str], ...] = (
    ("LSTM", "y_pred_model"),
    ("Persistence", "y_pred_persist"),
)

CALIBRATION_ORIGIN = "r2"
SCORING_ORIGIN = "main"

# 0.5 mirrors our own relative rule; 0.25 is the field's dominant convention
# (fraction of the scheduled headway) with the train-window median standing in
# for a schedule we do not have.
ABSOLUTE_RATIOS: tuple[float, ...] = (0.5, 0.25)

OBJECTIVES: tuple[str, ...] = ("mcc", "f1")


def _with_vector_length(residuals: pl.DataFrame) -> pl.DataFrame:
    """Restrict to vectors long enough to have a shape, as every other table does."""
    lengths = residuals.group_by(VECTOR_KEY).agg(pl.len().alias("vector_len"))
    return residuals.join(lengths, on=VECTOR_KEY, how="inner").filter(
        pl.col("vector_len") >= MIN_VECTOR_LEN
    )


def prepared(origin: str) -> pl.DataFrame:
    """Residuals carrying the self-referential flag and score for every column."""
    residuals = _with_vector_length(load_lstm(origin))
    columns = ["y_true"] + [column for _, column in MODELS]
    return residuals.with_columns(
        [bunching_flags(residuals, c).alias(f"_bunch_{c}") for c in columns]
        + [bunching_score(residuals, c).alias(f"_score_{c}") for c in columns]
    )


def absolute_cuts(origin: str, ratio: float) -> dict[tuple[str, int], float]:
    """K = ratio * median(observed headway), per (corridor, direction).

    Calibrated on an EARLIER window, so nothing about the scored window informs
    its own cut — the same discipline the relative threshold gets. Keyed by
    direction because the two directions of a corridor run different lengths and
    therefore different headway levels.
    """
    medians = (
        _with_vector_length(load_lstm(origin))
        .group_by(["corridor", "direction"])
        .agg(pl.col("y_true").median().alias("median_headway"))
    )
    return {
        (row["corridor"], row["direction"]): ratio * row["median_headway"]
        for row in medians.iter_rows(named=True)
    }


def build_absolute() -> pl.DataFrame:
    """Detection under the self-referential cut vs an absolute cut in minutes."""
    score = prepared(SCORING_ORIGIN)
    rows: list[dict] = []

    for ratio in ABSOLUTE_RATIOS:
        cuts = absolute_cuts(CALIBRATION_ORIGIN, ratio)
        # The absolute cut is a per-(corridor, direction) constant, so it can be
        # attached as a column and compared elementwise. Directions absent from
        # the calibration window get a null cut and drop out of the comparison
        # rather than silently inheriting another direction's level.
        cut_frame = pl.DataFrame(
            [
                {"corridor": c, "direction": d, "_cut": k}
                for (c, d), k in sorted(cuts.items())
            ]
        )
        tagged = score.join(cut_frame, on=["corridor", "direction"], how="inner")

        columns = ["y_true"] + [column for _, column in MODELS]
        tagged = tagged.with_columns(
            [(pl.col(c) < pl.col("_cut")).alias(f"_abs_{c}") for c in columns]
        )

        for corridor in CORRIDORS:
            for horizon in HORIZONS:
                cell = tagged.filter(
                    (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
                )
                if cell.height == 0:
                    continue

                rel_truth = cell.get_column("_bunch_y_true").to_numpy()
                abs_truth = cell.get_column("_abs_y_true").to_numpy()

                for name, column in MODELS:
                    rel = detection_scores(
                        rel_truth, cell.get_column(f"_bunch_{column}").to_numpy()
                    )
                    absolute = detection_scores(
                        abs_truth, cell.get_column(f"_abs_{column}").to_numpy()
                    )
                    # Threshold-free reading of the absolute event: the score is
                    # simply -value, monotone in "how short is this headway",
                    # with no vector-mean normalisation anywhere.
                    ranked = ranking_scores(
                        abs_truth, -cell.get_column(column).to_numpy()
                    )

                    rows.append(
                        {
                            "corridor": corridor,
                            "horizon": horizon,
                            "model": name,
                            "absolute_ratio": ratio,
                            "cut_minutes": float(
                                cell.get_column("_cut").mean()
                            ),
                            "n_cells": rel.n,
                            # --- self-referential rule (the published one) ---
                            "base_rate_relative": rel.true_rate,
                            "fire_rate_relative": rel.pred_rate,
                            "underfire_relative": (
                                rel.pred_rate / rel.true_rate
                                if rel.true_rate > 0 else None
                            ),
                            "f1_relative": rel.f1,
                            "mcc_relative": matthews_corrcoef(
                                rel_truth,
                                cell.get_column(f"_bunch_{column}").to_numpy(),
                            ),
                            # --- absolute cut in minutes ---
                            "base_rate_absolute": absolute.true_rate,
                            "fire_rate_absolute": absolute.pred_rate,
                            "underfire_absolute": (
                                absolute.pred_rate / absolute.true_rate
                                if absolute.true_rate > 0 else None
                            ),
                            "f1_absolute": absolute.f1,
                            "trivial_f1_absolute": trivial_f1(absolute.true_rate),
                            "mcc_absolute": matthews_corrcoef(
                                abs_truth,
                                cell.get_column(f"_abs_{column}").to_numpy(),
                            ),
                            "auc_absolute": ranked["auc"],
                        }
                    )

    return pl.DataFrame(rows).sort(
        ["absolute_ratio", "corridor", "horizon", "model"]
    )


def build_stability() -> pl.DataFrame:
    """How much does the fitted cut move between windows, per objective?

    ``spread`` is the range across the three origins. The out-of-sample columns
    answer the question an operator actually faces: if I had calibrated on r1
    instead of r2, how different would my deployed performance be?
    """
    per_origin = {origin: prepared(origin) for origin in ORIGINS}
    rows: list[dict] = []

    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            where = (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            cells = {o: f.filter(where) for o, f in per_origin.items()}
            if any(c.height == 0 for c in cells.values()):
                continue

            scored = cells[SCORING_ORIGIN]
            scored_truth = scored.get_column("_bunch_y_true").to_numpy()

            for name, column in MODELS:
                scored_values = scored.get_column(f"_score_{column}").to_numpy()
                row: dict = {
                    "corridor": corridor,
                    "horizon": horizon,
                    "model": name,
                }

                for objective in OBJECTIVES:
                    fitted = {
                        origin: best_threshold(
                            cell.get_column("_bunch_y_true").to_numpy(),
                            cell.get_column(f"_score_{column}").to_numpy(),
                            objective=objective,
                        )
                        for origin, cell in cells.items()
                    }
                    values = np.array(list(fitted.values()), dtype=float)
                    for origin, cut in fitted.items():
                        row[f"cut_{objective}_{origin}"] = cut
                    row[f"spread_{objective}"] = float(values.max() - values.min())
                    row[f"std_{objective}"] = float(values.std(ddof=0))

                    # Out-of-sample cost of the calibration-window choice: apply
                    # each earlier window's cut forward to the published window.
                    achieved = {
                        origin: matthews_corrcoef(
                            scored_truth, scored_values >= fitted[origin]
                        )
                        for origin in ORIGINS
                        if origin != SCORING_ORIGIN
                    }
                    for origin, mcc in achieved.items():
                        row[f"mcc_on_main_fit_{origin}_{objective}"] = mcc
                    got = np.array(list(achieved.values()), dtype=float)
                    row[f"mcc_spread_{objective}"] = float(got.max() - got.min())

                rows.append(row)

    return pl.DataFrame(rows).sort(["corridor", "horizon", "model"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    absolute = build_absolute()
    absolute.write_csv(OUT_ABSOLUTE)

    stability = build_stability()
    stability.write_csv(OUT_STABILITY)

    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=210, float_precision=4):
        for ratio in ABSOLUTE_RATIOS:
            sub = absolute.filter(
                (pl.col("absolute_ratio") == ratio) & (pl.col("model") == "LSTM")
            )
            print(f"\n=== Cut = {ratio} x median headway (calibrated on "
                  f"{CALIBRATION_ORIGIN!r}) — LSTM ===")
            print(sub.select([
                "corridor", "horizon", "cut_minutes",
                "base_rate_relative", "fire_rate_relative", "underfire_relative",
                "base_rate_absolute", "fire_rate_absolute", "underfire_absolute",
                "mcc_relative", "mcc_absolute", "auc_absolute",
            ]))

        print("\n=== Threshold stability across the three origins ===")
        print(stability.select([
            "corridor", "horizon", "model",
            "spread_mcc", "spread_f1", "mcc_spread_mcc", "mcc_spread_f1",
        ]))

    # --- the two headline comparisons, computed rather than eyeballed ---
    lstm = absolute.filter(pl.col("model") == "LSTM")
    for ratio in ABSOLUTE_RATIOS:
        sub = lstm.filter(pl.col("absolute_ratio") == ratio)
        print(f"\nRatio {ratio}: mediana de sub-disparo del LSTM "
              f"(1.0 = dispara tan seguido como ocurre el evento)")
        print(f"  regla auto-referencial : "
              f"{sub.get_column('underfire_relative').median():.4f}")
        print(f"  corte absoluto         : "
              f"{sub.get_column('underfire_absolute').median():.4f}")

    print("\nEstabilidad del umbral ajustado, mediana del rango entre los tres origenes:")
    print(f"  objetivo MCC : {stability.get_column('spread_mcc').median():.4f}")
    print(f"  objetivo F1  : {stability.get_column('spread_f1').median():.4f}")
    print("Costo fuera de muestra (rango del MCC en 'main' segun la ventana de calibracion):")
    print(f"  objetivo MCC : {stability.get_column('mcc_spread_mcc').median():.4f}")
    print(f"  objetivo F1  : {stability.get_column('mcc_spread_f1').median():.4f}")

    print(f"\nWrote {OUT_ABSOLUTE.relative_to(REPO_ROOT)} ({absolute.height} rows)")
    print(f"Wrote {OUT_STABILITY.relative_to(REPO_ROOT)} ({stability.height} rows)")


if __name__ == "__main__":
    main()
