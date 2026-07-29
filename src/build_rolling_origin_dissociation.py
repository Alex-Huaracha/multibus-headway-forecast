"""Does the DISSOCIATION survive outside the published window?

``build_rolling_origin_significance`` answered that question for the scalar
result — the MAE crossover holds in three test windows. But the crossover is not
this project's contribution; the dissociation is. And the dissociation was
measured on ONE 22-day window, which the results document itself flagged as its
most important remaining limitation.

This builder closes that gap. It re-computes the two vector metrics that carry
the claim — regularity fidelity (the CV of the headway vector) and joint
bunching detection — at all three origins:

    r1    test 2023-12-23 .. 2024-01-13
    r2    test 2024-01-14 .. 2024-02-04
    main  test 2024-02-08 .. 2024-02-29   (the published window)

No GPU, no Kaggle. The rolling residual exports already carry every column the
metrics need (``pair_rank`` indexes position inside the vector, ``y_true`` and
``y_pred_persist`` travel alongside ``y_pred_model`` on identical rows), so the
whole thing is a local recompute over bytes that are already on disk and already
guarded by ``tests/test_residual_freshness.py``.

Two models, not three
---------------------
The XGBoost was not re-run at the rolling origins, so it cannot appear here. The
comparison that matters is unaffected: persistence travels INSIDE the LSTM's own
export, on the same rows, so LSTM-vs-persistence is paired by construction at
every origin — no join, no coverage loss.

Population caveat, same as the significance builder
---------------------------------------------------
``build_contiguous_vector_metrics`` inner-joins against the XGBoost export
first, so its numbers describe the LSTM∩XGB population. This builder has no
XGBoost to join against at r1/r2, so it scores the FULL LSTM population at every
origin, ``main`` included. Comparability ACROSS origins is what this table
exists for, and restricting one origin but not the other two would destroy it.
The cost is that ``main`` here is not byte-comparable with the published vector
table — quote the published table for the published window, and this one only
for the comparison between windows.

What would falsify the claim
----------------------------
If persistence's bunching advantage were an artifact of February 2024, some
cells would put the win on the learner's side at an earlier origin. ``agrees``
reports exactly that, per (corridor, horizon), and a cell that flips belongs in
threats-to-validity rather than being smoothed away.

Outputs
-------
``docs/resultados/csv-multihorizon/rolling_origin_dissociation.csv``
    One row per (origin, corridor, horizon, model).
``docs/resultados/csv-multihorizon/rolling_origin_dissociation_agreement.csv``
    One row per (corridor, horizon): does every origin agree on the winner?

Usage
-----
    uv run python -m src.build_rolling_origin_dissociation
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
from src.build_rolling_origin_significance import ORIGINS  # noqa: E402
from src.evaluation.vector_metrics import (  # noqa: E402
    MIN_VECTOR_LEN,
    bunching_flags,
    bunching_score,
    detection_scores,
    ranking_scores,
    regularity_error,
    trivial_f1,
    vector_frame,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "rolling_origin_dissociation.csv"
OUT_SUMMARY_CSV = OUT_DIR / "rolling_origin_dissociation_agreement.csv"

# (display name, prediction column). Persistence is the competitor, not a
# formality: it propagates the observed vector, so it preserves shape by
# construction — a strong regularity predictor that scalar MAE calls a loser.
MODELS: tuple[tuple[str, str], ...] = (
    ("LSTM", "y_pred_model"),
    ("Persistence", "y_pred_persist"),
)

VECTOR_GROUP = ["corridor", "direction", "horizon", "start_ts"]


def cell_metrics(residuals: pl.DataFrame) -> list[dict]:
    """Regularity and detection scores per (corridor, horizon, model).

    Mirrors ``build_contiguous_vector_metrics.build_vector_metrics`` — same
    helpers, same ``MIN_VECTOR_LEN`` filter, same flag semantics — so the two
    tables are read the same way. The only difference is the model list.
    """
    value_cols = ["y_true"] + [column for _, column in MODELS]

    # Bunching is a per-cell flag, so it is computed on the residual frame and
    # only then restricted to the vectors long enough to have a shape.
    flagged = residuals.with_columns(
        [
            bunching_flags(residuals, column).alias(f"_bunch_{column}")
            for column in value_cols
        ]
        # The continuous score behind the flag. Carried alongside because the
        # flag alone cannot separate "the model has no information" from "the
        # cut is in the wrong place", and that distinction is the whole point.
        + [
            bunching_score(residuals, column).alias(f"_score_{column}")
            for column in value_cols
        ]
    )
    lengths = residuals.group_by(VECTOR_GROUP).agg(pl.len().alias("vector_len"))
    flagged = flagged.join(lengths, on=VECTOR_GROUP, how="inner").filter(
        pl.col("vector_len") >= MIN_VECTOR_LEN
    )

    vectors = vector_frame(residuals, value_cols)

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
                        "corridor": corridor,
                        "horizon": horizon,
                        "model": name,
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
                        # The threshold-free reading of the same rows. Without
                        # these two columns the table can only report the
                        # operating point it was handed.
                        "trivial_f1": trivial_f1(scores.true_rate),
                        **ranking_scores(
                            truth, cell_cells.get_column(f"_score_{column}").to_numpy()
                        ),
                    }
                )
    return rows


def build() -> pl.DataFrame:
    """One row per (origin, corridor, horizon, model)."""
    rows: list[dict] = []
    for origin in ORIGINS:
        residuals = load_lstm(origin)
        for row in cell_metrics(residuals):
            rows.append({"origin": origin, **row})
    return pl.DataFrame(rows).sort(["corridor", "horizon", "origin", "model"])


def agreement(table: pl.DataFrame) -> pl.DataFrame:
    """Per (corridor, horizon): do all three origins name the same winner?

    Reported on bunching F1, the metric that carries the dissociation. The CV
    bias travels alongside because a sign flip there would undercut the
    mechanism even if the F1 ranking held.
    """
    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            cell = table.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            if cell.height == 0:
                continue

            row: dict = {"corridor": corridor, "horizon": horizon}
            winners: list[str] = []
            for origin in ORIGINS:
                at = {
                    r["model"]: r
                    for r in cell.filter(pl.col("origin") == origin).iter_rows(
                        named=True
                    )
                }
                lstm, persist = at["LSTM"], at["Persistence"]
                winner = (
                    "PERSIST"
                    if persist["bunching_f1"] > lstm["bunching_f1"]
                    else "LSTM"
                )
                winners.append(winner)
                row[f"winner_{origin}"] = winner
                # The same cell judged without the cut. Reported next to the
                # F1 winner rather than instead of it: the pair is the finding.
                row[f"winner_auc_{origin}"] = (
                    "PERSIST" if persist["auc"] > lstm["auc"] else "LSTM"
                )
                row[f"auc_lstm_{origin}"] = lstm["auc"]
                row[f"auc_persist_{origin}"] = persist["auc"]
                # Does the F1 winner even beat flagging every cell?
                row[f"persist_beats_trivial_{origin}"] = (
                    persist["bunching_f1"] > persist["trivial_f1"]
                )
                row[f"f1_lstm_{origin}"] = lstm["bunching_f1"]
                row[f"f1_persist_{origin}"] = persist["bunching_f1"]
                # The headline ratio. Guarded: a learner that never fires scores
                # exactly 0.0, and the honest reading of that is "unbounded",
                # not a division error.
                row[f"f1_ratio_{origin}"] = (
                    persist["bunching_f1"] / lstm["bunching_f1"]
                    if lstm["bunching_f1"] > 0
                    else None
                )
                row[f"cv_bias_lstm_{origin}"] = lstm["cv_bias"]

            row["agrees"] = len(set(winners)) == 1
            row["winner"] = winners[0] if row["agrees"] else "SPLIT"

            auc_winners = {row[f"winner_auc_{origin}"] for origin in ORIGINS}
            row["agrees_auc"] = len(auc_winners) == 1
            row["winner_auc"] = (
                next(iter(auc_winners)) if row["agrees_auc"] else "SPLIT"
            )
            rows.append(row)
    return pl.DataFrame(rows).sort(["corridor", "horizon"])


def main() -> None:
    table = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    summary = agreement(table)
    summary.write_csv(OUT_SUMMARY_CSV)

    with pl.Config(tbl_rows=60, tbl_cols=16, tbl_width_chars=220):
        print("Bunching F1 at the FIXED cut — who wins, per origin:")
        print(
            summary.select(
                ["corridor", "horizon", "winner_r1", "winner_r2", "winner_main",
                 "f1_ratio_r1", "f1_ratio_r2", "f1_ratio_main", "agrees"]
            )
        )
        print("\nSame cells, THRESHOLD-FREE (AUC) — the corrected verdict:")
        print(
            summary.select(
                ["corridor", "horizon", "winner_auc_r1", "winner_auc_r2",
                 "winner_auc_main", "auc_lstm_main", "auc_persist_main",
                 "agrees_auc"]
            )
        )
        print("\nLSTM CV bias (negative = predicts a smoother corridor):")
        print(
            summary.select(
                ["corridor", "horizon", "cv_bias_lstm_r1", "cv_bias_lstm_r2",
                 "cv_bias_lstm_main"]
            )
        )

    n_cells = summary.height
    n_agree = int(summary.get_column("agrees").sum())
    lstm_rows = table.filter(pl.col("model") == "LSTM")
    n_negative_bias = int((lstm_rows.get_column("cv_bias") < 0).sum())

    print(f"\n{n_agree}/{n_cells} celdas coinciden en los tres origenes (F1, corte fijo)")
    print(
        f"{int(summary.get_column('agrees_auc').sum())}/{n_cells} celdas coinciden "
        "en los tres origenes (AUC, sin umbral)"
    )
    print(
        f"{n_negative_bias}/{lstm_rows.height} celdas (origen x corredor x horizonte) "
        "tienen sesgo de CV negativo"
    )
    at_10 = summary.filter(pl.col("horizon") == 10)
    print(
        f"A h=10, el LSTM gana el AUC en "
        f"{int((at_10.get_column('winner_auc') == 'LSTM').sum())}/{at_10.height} "
        "celdas en los tres origenes"
    )
    below = sum(
        int((~summary.get_column(f"persist_beats_trivial_{o}")).sum())
        for o in ORIGINS
    )
    print(
        f"{below}/{n_cells * len(ORIGINS)} celdas donde la persistencia NO supera "
        "al detector trivial"
    )
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")
    print(f"Wrote {OUT_SUMMARY_CSV.relative_to(REPO_ROOT)} ({summary.height} rows)")


if __name__ == "__main__":
    main()
