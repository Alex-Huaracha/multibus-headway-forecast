"""The floor the threshold-free verdict was missing.

Section IV-D fixes a floor for F1 — the trivial detector that flags every
position — because F1 ignores true negatives and rewards a rule with no content.
The AUC was reported without any equivalent, and that is a real gap rather than
a stylistic one.

The reason is the event rule itself. A position counts as bunching when its
headway falls below half the mean of its OWN vector, so what matters is not how
much headways vary across positions but how much they vary across positions
AFTER each vector's own mean is divided out. Where a residue survives that
division, some positions sit below half their vector mean because of WHERE they
are, and a rule that knows only the position earns AUC above chance without
anticipating anything.

That distinction decides which corridor is exposed, and the raw spread does not.
Measured on ``contiguous_error_profile.csv``, the across-rank spread of the mean
headway is about the same in all three corridors (33% to 55% of the corridor
mean). What separates them is the shape of the profile once the vector mean is
removed. In E4 and E59 it falls monotonically with rank, and rank depth is
confounded with vector length: the long vectors are the ones that reach the high
ranks and they carry lower mean headways, so each vector's own mean absorbs the
trend and it cancels. E2's profile is U-shaped instead, its vectors are short
(mean length 3.79 against E59's 5.60), and the deviation survives. The relative
profile swings 48 points across ranks in E2 against 11 in E59.

This builder measures what that is worth. The null answers with the mean
observed headway of each ``(corridor, direction, horizon, pair_rank)``, fitted on
the earlier disjoint origin and held constant for every minute of the scored
period. It never reads the input window, so it cannot anticipate; it is not
information-free, because scoring divides by the evaluated vector's own mean and
therefore reads which rank slots that instant filled. The two competitors read
exactly the same thing.

What it found, and why the document has to say so: the null is at chance in E4
(~0.52) and E59 (~0.49), so the learner's advantage in those two corridors is
not positional. E2 is different — the null holds ~0.58 at every horizon, and at
h=10 the learner falls below it (0.566 against 0.579).

That loss is bounded here rather than left as a bare sign. Section IV-E defines a
verdict as three parts — who wins, by how much, and whether the difference
survives its test — and a floor published without the last two asserts a sign at
a finer resolution than every interval the paper prints. The same day-clustered
bootstrap that bounds the rest of the table bounds these differences.

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

import numpy as np  # noqa: E402

from src.build_contiguous_significance import (  # noqa: E402
    CORRIDORS,
    HORIZONS,
    load_lstm,
)
from src.build_detection_ranking_ci import LEVEL, N_BOOT, SEED  # noqa: E402
from src.evaluation.ranking_significance import (  # noqa: E402
    RankingSignificanceError,
    clustered_bootstrap_delta,
    fast_auc,
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


def _bounded(truth, better, worse, day) -> dict[str, float | None]:
    """One difference of areas, with the interval the rest of the paper uses.

    A cell with a single class present has no area to compare, which is an
    absent measurement rather than a zero difference and is written as null.
    """
    try:
        result = clustered_bootstrap_delta(
            truth, better, worse, day, statistic=fast_auc,
            n_boot=N_BOOT, seed=SEED, level=LEVEL,
        )
    except RankingSignificanceError:
        return {"delta": None, "ci_low": None, "ci_high": None, "survives": None}
    return {
        "delta": result.delta,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "survives": not result.crosses_zero,
    }


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

            # A position the fit window never resolved has no profile. Dropping
            # it from the cell rather than from the null alone is what keeps
            # Section IV-C's contract: every method scored on the same rows.
            unprofiled = int(cell.get_column("y_pred_prior").is_null().sum())
            cell = cell.filter(pl.col("y_pred_prior").is_not_null())

            truth = cell.get_column("_bunch_y_true").to_numpy()
            day = cell.get_column("target_ts").dt.date().to_numpy()
            row = {
                "corridor": corridor,
                "horizon": horizon,
                "n_cells": cell.height,
                "base_rate": float(truth.mean()),
                "n_unprofiled": unprofiled,
            }
            series = {}
            for name, column in PREDICTORS:
                series[name] = cell.get_column(f"_score_{column}").to_numpy()
                scores = ranking_scores(truth, series[name])
                row[f"auc_{name}"] = scores["auc"]
                row[f"ap_lift_{name}"] = scores["ap_lift"]

            row["null_beats_lstm"] = row["auc_null"] > row["auc_lstm"]
            row["null_beats_persist"] = row["auc_null"] > row["auc_persist"]

            # Signed toward the competitor, so a negative entry reads as "this
            # method ranks worse than knowing only the position".
            for name in ("lstm", "persist"):
                bound = _bounded(truth, series[name], series["null"], day)
                for key, value in bound.items():
                    row[f"{name}_vs_null_{key}"] = value

            row["n_boot"] = N_BOOT
            row["seed"] = SEED
            row["level"] = LEVEL
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
