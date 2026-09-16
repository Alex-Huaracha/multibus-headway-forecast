"""The floor the threshold-free verdict was missing.

Section IV-D fixes a floor for F1 — the trivial detector that flags every
position — because F1 ignores true negatives and rewards a rule with no content.
The AUC was reported without any equivalent, and that is a real gap rather than
a stylistic one.

The reason is the event rule itself. A position counts as bunching when its
headway falls below half the mean of its OWN vector, and the positions of a
vector are not exchangeable: the ones nearer the front carry systematically
shorter headways (``contiguous_error_profile.csv`` shows a 51% spread across
ranks in E2 at h=1). So some positions sit below half their vector mean because
of WHERE they are, not because of anything the input window contained. Any model
that learns the positional profile earns AUC above chance without anticipating
anything.

This builder measures exactly how much. The null answers with the mean observed
headway of each ``(corridor, direction, horizon, pair_rank)``, fitted on the
earlier disjoint origin and held constant for every minute of the scored period.
It never reads the input window; it cannot anticipate by construction. Its AUC
is therefore the part of the published ranking advantage that position alone
explains.

What it found, and why the document has to say so: the null is at chance in E4
(~0.52) and E59 (~0.49), so the learner's advantage in those two corridors is
not positional. E2 is different — the null holds ~0.58 at every horizon, and at
h=10 the learner falls below it (0.566 against 0.579). That is the cell Sections
V-C and VII use as their showcase.

The fit window is the same one Section IV-D calibrates the operating point on,
for the same reason: it is earlier than the scored period and disjoint from it,
which is the only direction a deployed system could calibrate in.
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
    MIN_VECTOR_LEN,
    VECTOR_KEY,
    bunching_flags,
    bunching_score,
    ranking_scores,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "positional_null.csv"

# Deliberately free of any timestamp: one constant per position is what makes
# this a null rather than a weak forecaster.
PROFILE_KEY: list[str] = ["corridor", "direction", "horizon", "pair_rank"]

PREDICTORS: tuple[tuple[str, str], ...] = (
    ("null", "y_pred_prior"),
    ("persist", "y_pred_persist"),
    ("lstm", "y_pred_model"),
)

FIT_ORIGIN = "r2"
SCORING_ORIGIN = "main"


def positional_profile(origin: str) -> pl.DataFrame:
    """Mean observed headway per position, fitted on one origin.

    The mean is taken over the whole window, so the profile knows the corridor's
    average shape and nothing about any particular minute.
    """
    return (
        load_lstm(origin)
        .group_by(PROFILE_KEY)
        .agg(
            pl.col("y_true").mean().alias("y_pred_prior"),
            pl.len().alias("n_fit"),
        )
        .sort(PROFILE_KEY)
    )


def prepared() -> pl.DataFrame:
    """Scored-origin residuals carrying the null alongside the two predictors."""
    profile = positional_profile(FIT_ORIGIN).select(PROFILE_KEY + ["y_pred_prior"])
    residuals = load_lstm(SCORING_ORIGIN).join(profile, on=PROFILE_KEY, how="left")

    lengths = residuals.group_by(VECTOR_KEY).agg(pl.len().alias("vector_len"))
    residuals = residuals.join(lengths, on=VECTOR_KEY, how="inner").filter(
        pl.col("vector_len") >= MIN_VECTOR_LEN
    )

    columns = ["y_true"] + [column for _, column in PREDICTORS]
    return residuals.with_columns(
        [bunching_flags(residuals, c).alias(f"_bunch_{c}") for c in columns]
        + [bunching_score(residuals, c).alias(f"_score_{c}") for c in columns]
    )


def build() -> pl.DataFrame:
    scored = prepared()

    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            cell = scored.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            if cell.height == 0:
                continue

            truth = cell.get_column("_bunch_y_true").to_numpy()
            row = {
                "corridor": corridor,
                "horizon": horizon,
                "n_cells": cell.height,
                "base_rate": float(truth.mean()),
                # A position the fit window never resolved has no profile, so it
                # cannot be scored. Recorded rather than silently dropped.
                "n_unprofiled": int(
                    cell.get_column("y_pred_prior").is_null().sum()
                ),
            }
            for name, column in PREDICTORS:
                scores = ranking_scores(
                    truth, cell.get_column(f"_score_{column}").to_numpy()
                )
                row[f"auc_{name}"] = scores["auc"]
                row[f"ap_lift_{name}"] = scores["ap_lift"]
            row["null_beats_lstm"] = row["auc_null"] > row["auc_lstm"]
            row["null_beats_persist"] = row["auc_null"] > row["auc_persist"]
            rows.append(row)

    return pl.DataFrame(rows).sort(["corridor", "horizon"])


def main() -> None:
    table = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=200):
        print(
            f"Positional floor: profile fitted on origin {FIT_ORIGIN!r}, "
            f"scored on {SCORING_ORIGIN!r}\n"
        )
        print(
            table.select(
                "corridor", "horizon", "base_rate",
                "auc_null", "auc_persist", "auc_lstm",
                "null_beats_lstm",
            )
        )

    beaten = table.filter(pl.col("null_beats_lstm"))
    print(
        f"\nCeldas donde el perfil posicional supera al LSTM: "
        f"{beaten.height}/{table.height}"
    )
    for row in beaten.iter_rows(named=True):
        print(
            f"  {row['corridor']} h={row['horizon']}: "
            f"piso {row['auc_null']:.4f} > LSTM {row['auc_lstm']:.4f}"
        )
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
