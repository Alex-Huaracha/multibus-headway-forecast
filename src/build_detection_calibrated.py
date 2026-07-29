"""Does the learner MISS bunching, or was it never given a usable threshold?

The previous version of Section 5 answered "misses it" and reported ratios up to
2299x. That answer does not survive its own data, and this builder is the
correction.

The defect
----------
``bunching_flags`` fires when a value falls below ``BUNCHING_RATIO = 0.5`` of
its vector's mean. That operating point is calibrated on OBSERVED vectors.
Persistence copies an observed vector, so it inherits the realized dispersion
(CV about 0.79) and the cut lands where it was designed to land. An L1-trained
point forecast emits a compressed vector (CV about 0.16), so the SAME relative
cut sits roughly three standard deviations into its left tail and the detector
fires fourteen times in fifty thousand opportunities. Reporting that as a
detection failure confuses a mis-set threshold with missing information.

Three columns settle it
-----------------------
``auc`` / ``average_precision``
    Threshold-free. Invariant to any monotone rescaling of the score — exactly
    the transformation a fixed relative cut is NOT invariant to. If the learner
    were blind, these would be at chance. They are not.

``f1_calibrated``
    The threshold FITTED on an earlier origin and APPLIED forward to the
    published window. One scalar, out of sample, in the deployable direction —
    the same calibrate-early / score-later discipline Section 6 already accepts
    for the router. This gives the learner the free parameter persistence gets
    by construction.

``trivial_f1``
    F1 of flagging EVERY cell. The floor. Without this column a table cannot
    reveal that the previously declared winner loses to a constant at h=10 in
    all three corridors, which is what made F1 the wrong summary here. ``mcc``
    is reported alongside for the same reason: it is 0 for the constant rule.

What survives and what does not
-------------------------------
The under-dispersion itself is real, robust, and unchanged (CV bias negative in
36 of 36 cells). What does not survive is reading it as blindness. The honest
claim is narrower and more useful: a conditional-mean forecast is systematically
under-dispersed, so an event rule whose threshold was calibrated in observation
space cannot be transplanted onto it — and a naive deployment therefore fails
for reasons that have nothing to do with what the model knows.

Calibration origin
------------------
The threshold is fitted on ``r2`` and applied to ``main``. Those are disjoint
test windows produced by SEPARATELY TRAINED models, so nothing about the
published window informs its own threshold.

Outputs
-------
``docs/resultados/csv-multihorizon/contiguous_detection_calibrated.csv``

Usage
-----
    uv run python -m src.build_detection_calibrated
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

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
OUT_CSV = OUT_DIR / "contiguous_detection_calibrated.csv"

MODELS: tuple[tuple[str, str], ...] = (
    ("LSTM", "y_pred_model"),
    ("Persistence", "y_pred_persist"),
)

# Fit the operating point here, score it there. Disjoint windows, separately
# trained models, and the fit window is EARLIER — the only direction a deployed
# system could actually calibrate in.
CALIBRATION_ORIGIN = "r2"
SCORING_ORIGIN = "main"


def prepared(origin: str) -> pl.DataFrame:
    """Residuals restricted to vectors with a shape, carrying flags and scores."""
    residuals = load_lstm(origin)
    lengths = residuals.group_by(VECTOR_KEY).agg(pl.len().alias("vector_len"))
    residuals = residuals.join(lengths, on=VECTOR_KEY, how="inner").filter(
        pl.col("vector_len") >= MIN_VECTOR_LEN
    )
    columns = ["y_true"] + [column for _, column in MODELS]
    return residuals.with_columns(
        [bunching_flags(residuals, c).alias(f"_bunch_{c}") for c in columns]
        + [bunching_score(residuals, c).alias(f"_score_{c}") for c in columns]
    )


def build() -> pl.DataFrame:
    fit = prepared(CALIBRATION_ORIGIN)
    score = prepared(SCORING_ORIGIN)

    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            where = (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            fit_cell, score_cell = fit.filter(where), score.filter(where)
            if score_cell.height == 0 or fit_cell.height == 0:
                continue

            truth = score_cell.get_column("_bunch_y_true").to_numpy()
            fit_truth = fit_cell.get_column("_bunch_y_true").to_numpy()
            base_rate = float(truth.mean())

            for name, column in MODELS:
                flags = score_cell.get_column(f"_bunch_{column}").to_numpy()
                values = score_cell.get_column(f"_score_{column}").to_numpy()
                fit_values = fit_cell.get_column(f"_score_{column}").to_numpy()
                fixed = detection_scores(truth, flags)

                # One scalar, fitted on the earlier window, applied here. MCC is
                # the fitting objective; the F1-fitted cut is carried alongside
                # only to document that it degenerates (see below).
                threshold = best_threshold(fit_truth, fit_values, objective="mcc")
                recalibrated = detection_scores(truth, values >= threshold)

                threshold_f1 = best_threshold(fit_truth, fit_values, objective="f1")
                degenerate = detection_scores(truth, values >= threshold_f1)

                rows.append(
                    {
                        "corridor": corridor,
                        "horizon": horizon,
                        "model": name,
                        "n_cells": fixed.n,
                        "base_rate": base_rate,
                        "trivial_f1": trivial_f1(base_rate),
                        # --- the published operating point ---
                        "fire_rate_fixed": fixed.pred_rate,
                        "f1_fixed": fixed.f1,
                        "mcc_fixed": matthews_corrcoef(truth, flags),
                        # --- threshold-free ---
                        **ranking_scores(truth, values),
                        # --- out-of-sample calibrated operating point (MCC-fitted) ---
                        "threshold_fitted": threshold,
                        "fire_rate_calibrated": recalibrated.pred_rate,
                        "f1_calibrated": recalibrated.f1,
                        "mcc_calibrated": matthews_corrcoef(truth, values >= threshold),
                        # --- the same exercise fitted on F1, which degenerates ---
                        "threshold_f1fit": threshold_f1,
                        "fire_rate_f1fit": degenerate.pred_rate,
                        "f1_f1fit": degenerate.f1,
                    }
                )

    return pl.DataFrame(rows).sort(["corridor", "horizon", "model"])


def verdicts(table: pl.DataFrame) -> pl.DataFrame:
    """Per cell: who wins under each of the three readings."""
    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            cell = {
                r["model"]: r
                for r in table.filter(
                    (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
                ).iter_rows(named=True)
            }
            if not cell:
                continue
            lstm, persist = cell["LSTM"], cell["Persistence"]
            rows.append(
                {
                    "corridor": corridor,
                    "horizon": horizon,
                    "f1_ratio_fixed": (
                        persist["f1_fixed"] / lstm["f1_fixed"]
                        if lstm["f1_fixed"] > 0 else None
                    ),
                    "winner_fixed": (
                        "PERSIST" if persist["f1_fixed"] > lstm["f1_fixed"] else "LSTM"
                    ),
                    "winner_calibrated": (
                        "PERSIST"
                        if persist["f1_calibrated"] > lstm["f1_calibrated"]
                        else "LSTM"
                    ),
                    "winner_auc": (
                        "PERSIST" if persist["auc"] > lstm["auc"] else "LSTM"
                    ),
                    "winner_mcc_cal": (
                        "PERSIST"
                        if persist["mcc_calibrated"] > lstm["mcc_calibrated"]
                        else "LSTM"
                    ),
                    "f1_cal_lstm": lstm["f1_calibrated"],
                    "f1_cal_persist": persist["f1_calibrated"],
                    "mcc_cal_lstm": lstm["mcc_calibrated"],
                    "mcc_cal_persist": persist["mcc_calibrated"],
                    "auc_lstm": lstm["auc"],
                    "auc_persist": persist["auc"],
                    "trivial_f1": lstm["trivial_f1"],
                    # Fitting the cut on F1 instead of MCC collapses it to
                    # "flag everything" — recorded so the choice of objective
                    # is auditable rather than asserted.
                    "f1fit_fire_rate_lstm": lstm["fire_rate_f1fit"],
                    "f1fit_fire_rate_persist": persist["fire_rate_f1fit"],
                    # The question no previous table asked: does the declared
                    # winner even beat flagging everything?
                    "persist_beats_trivial": persist["f1_fixed"] > lstm["trivial_f1"],
                }
            )
    return pl.DataFrame(rows).sort(["corridor", "horizon"])


def main() -> None:
    table = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    summary = verdicts(table)
    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=230):
        print(f"Bunching detection at origin {SCORING_ORIGIN!r}, threshold fitted "
              f"on {CALIBRATION_ORIGIN!r}\n")
        print(summary.select(
            ["corridor", "horizon", "f1_ratio_fixed", "winner_fixed",
             "winner_auc", "winner_mcc_cal", "auc_lstm", "auc_persist",
             "mcc_cal_lstm", "mcc_cal_persist", "trivial_f1",
             "persist_beats_trivial"]
        ))
        print("\nUmbral ajustado por F1 (degenera: dispara casi siempre):")
        print(summary.select(
            ["corridor", "horizon", "f1fit_fire_rate_lstm",
             "f1fit_fire_rate_persist", "f1_cal_lstm", "f1_cal_persist"]
        ))

    n = summary.height
    print(f"\nGanador con el umbral publicado : "
          f"{summary.filter(pl.col('winner_fixed') == 'LSTM').height}/{n} LSTM")
    print(f"Ganador por AUC (sin umbral)    : "
          f"{summary.filter(pl.col('winner_auc') == 'LSTM').height}/{n} LSTM")
    print(f"Ganador por MCC calibrado       : "
          f"{summary.filter(pl.col('winner_mcc_cal') == 'LSTM').height}/{n} LSTM")
    print(f"Ganador por F1 calibrado        : "
          f"{summary.filter(pl.col('winner_calibrated') == 'LSTM').height}/{n} LSTM")
    print(f"Celdas donde la persistencia NO supera al detector trivial: "
          f"{summary.filter(~pl.col('persist_beats_trivial')).height}/{n}")
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
