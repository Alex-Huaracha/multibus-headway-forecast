"""Winsorization sensitivity — audit pending #3, and the paper's first line of attack.

The training contract clips ``delta_t_min`` at the train p99 and applies that
ceiling to every split. The clipped 1% is not noise: it is the extreme bunching
and service-gap tail — precisely the regime the whole argument now rests on. A
reviewer's first question is whether the reported gap survives when the ceiling
comes off, so it gets measured rather than argued.

Two scorings, both against RAW un-clipped targets recovered from the parquet:

``as_produced``
    Predictions exactly as the models emitted them, including persistence, whose
    prediction is a clipped observation. This is what a deployed system scored
    against reality would look like.
``raw_persistence``
    Persistence recomputed from the raw observation at the same timestamp. It is
    the fair competitor — persistence is trivially recomputable without a ceiling,
    so leaving one on it would be an arbitrary handicap.

Both are reported. If the DL-vs-persistence verdict only survives under one of
them, that is the finding.

The ceiling turns out to *help* persistence rather than handicap it, which is
worth stating because it is the paper's own thesis appearing from a second
direction. Persistence propagates the last observation; when that observation is
an extreme 35 min, clipping it to 28.5 moves the prediction toward the bulk of
targets and lowers MAE. Winsorization is a shrinkage, and shrinkage is what MAE
rewards — the same mechanism that makes the learners flatten the vector, arriving
here through preprocessing instead of through a loss function.

The bunching detection is recomputed against raw targets for the same reason:
clipping compresses the true vector's spread, so it could be manufacturing part
of the flattening result. Under raw targets the true irregularity is larger, and
the question is whether the learners fall further behind or catch up.

A terminology correction the audit's phrasing invites: the clipped top 1% are
**service gaps**, not bunching. Bunching is a headway collapsing toward zero, and
a ceiling cannot touch it. So the ceiling is a threat to the gap side of the
irregularity claim and essentially inert on the bunching side — which is what
the numbers show, and which is worth stating rather than leaving as a
coincidence.

Output
------
``docs/resultados/csv-multihorizon/contiguous_winsorization_sensitivity.csv``

Usage
-----
    uv run python -m src.build_contiguous_winsorization_sensitivity
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_contiguous_volatility import CORRIDOR_IDS  # noqa: E402
from src.build_contiguous_vector_metrics import MODELS, load_paired  # noqa: E402
from src.build_sample_index import T_IN, load_corridor  # noqa: E402
from src.evaluation.significance_clustered import dm_clustered  # noqa: E402
from src.evaluation.splits import split_temporal, winsorize_train_p99  # noqa: E402
from src.evaluation.vector_metrics import (  # noqa: E402
    MIN_VECTOR_LEN,
    bunching_flags,
    detection_scores,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "contiguous_winsorization_sensitivity.csv"
)


def raw_observations(empresaid: int) -> pl.DataFrame:
    """Un-clipped ``delta_t_min`` per ``(direction, t, pair_rank)``.

    ``prepare`` applies the ceiling; this deliberately does not. Only the split
    tagging is shared, so the two frames index the same rows.
    """
    df = split_temporal(load_corridor(empresaid))
    return df.select(
        pl.col("direction"),
        pl.col("t"),
        pl.col("pair_rank"),
        pl.col("delta_t_min").alias("y_true_raw"),
    ).filter(pl.col("y_true_raw").is_not_null())


def with_raw_targets(paired: pl.DataFrame, corridor: str) -> tuple[pl.DataFrame, float]:
    """Attach raw targets and the raw persistence value; return the frame and ceiling."""
    empresaid = CORRIDOR_IDS[corridor]
    _clipped, threshold = winsorize_train_p99(split_temporal(load_corridor(empresaid)))

    raw = raw_observations(empresaid)
    cell = paired.filter(pl.col("corridor") == corridor)

    # Raw target: the observation at target_ts for this vector position.
    out = cell.join(
        raw,
        left_on=["direction", "target_ts", "pair_rank"],
        right_on=["direction", "t", "pair_rank"],
        how="inner",
    )
    # Raw persistence: the observation at the last input snapshot, which contract
    # C2 places exactly (T_in - 1) minutes after start_ts on the same run.
    return (
        out.with_columns(
            (pl.col("start_ts") + pl.duration(minutes=T_IN - 1)).alias("_persist_ts")
        )
        .join(
            raw.rename({"y_true_raw": "y_pred_persist_raw"}),
            left_on=["direction", "_persist_ts", "pair_rank"],
            right_on=["direction", "t", "pair_rank"],
            how="inner",
        )
        .drop("_persist_ts"),
        threshold,
    )


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).mean())


def bunching_row(frame: pl.DataFrame, truth_col: str, pred_col: str) -> dict:
    """Bunching detection restricted to vectors long enough to have a shape."""
    lengths = frame.group_by(
        ["corridor", "direction", "horizon", "start_ts"]
    ).agg(pl.len().alias("_len"))
    work = frame.join(
        lengths, on=["corridor", "direction", "horizon", "start_ts"], how="inner"
    ).filter(pl.col("_len") >= MIN_VECTOR_LEN)
    scores = detection_scores(
        bunching_flags(work, truth_col).to_numpy(),
        bunching_flags(work, pred_col).to_numpy(),
    )
    return {
        "bunching_rate_true": scores.true_rate,
        "bunching_recall": scores.recall,
        "bunching_precision": scores.precision,
        "bunching_f1": scores.f1,
    }


def build() -> pl.DataFrame:
    paired = load_paired()

    rows: list[dict] = []
    for corridor in CORRIDORS:
        cell_all, threshold = with_raw_targets(paired, corridor)

        for horizon in HORIZONS:
            frame = cell_all.filter(pl.col("horizon") == horizon)
            if frame.height == 0:
                continue

            y_clipped = frame.get_column("y_true").to_numpy()
            y_raw = frame.get_column("y_true_raw").to_numpy()
            persist_raw = frame.get_column("y_pred_persist_raw").to_numpy()
            day = frame.get_column("target_ts").dt.date().to_numpy()

            clipped_cells = int(np.sum(y_raw > threshold + 1e-9))

            for name, column in MODELS:
                pred = frame.get_column(column).to_numpy()
                # `raw_persistence` differs from `as_produced` only for
                # persistence itself; for the learners the two are the same
                # prediction and the column is emitted for symmetry.
                pred_fair = persist_raw if name == "Persistence" else pred

                # Paired differential against the FAIR persistence, so the test
                # answers "does the learner still beat a persistence that was
                # never handicapped by the ceiling".
                d_raw = np.abs(y_raw - pred_fair) - np.abs(y_raw - persist_raw)
                verdict = (
                    dm_clustered(d_raw, day, horizon=horizon)
                    if name != "Persistence"
                    else None
                )

                row = {
                    "model": name,
                    "corridor": corridor,
                    "horizon": horizon,
                    "n": frame.height,
                    "p99_threshold": threshold,
                    "n_clipped_targets": clipped_cells,
                    "pct_clipped_targets": 100.0 * clipped_cells / frame.height,
                    "mae_vs_clipped_target": _mae(y_clipped, pred),
                    "mae_vs_raw_target_as_produced": _mae(y_raw, pred),
                    "mae_vs_raw_target_fair": _mae(y_raw, pred_fair),
                    "delta_vs_persist_clipped": _mae(y_clipped, pred)
                    - _mae(y_clipped, frame.get_column("y_pred_persist").to_numpy()),
                    "delta_vs_persist_raw_fair": _mae(y_raw, pred_fair)
                    - _mae(y_raw, persist_raw),
                    "dm_p_raw_fair": verdict.p_value if verdict is not None else None,
                }
                row.update(
                    {
                        f"raw_{key}": value
                        for key, value in bunching_row(
                            frame, "y_true_raw",
                            "y_pred_persist_raw" if name == "Persistence" else column,
                        ).items()
                    }
                )
                rows.append(row)

    return pl.DataFrame(rows).sort(["model", "corridor", "horizon"])


def main() -> None:
    table = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=210):
        print("Clipping footprint and the scalar verdict under a raw ceiling:")
        print(
            table.filter(pl.col("model") == "LSTM").select(
                ["corridor", "horizon", "p99_threshold", "pct_clipped_targets",
                 "mae_vs_clipped_target", "mae_vs_raw_target_fair",
                 "delta_vs_persist_clipped", "delta_vs_persist_raw_fair",
                 "dm_p_raw_fair"]
            )
        )
        print("\nBunching detection against RAW targets:")
        print(
            table.select(
                ["model", "corridor", "horizon", "raw_bunching_rate_true",
                 "raw_bunching_recall", "raw_bunching_precision", "raw_bunching_f1"]
            )
        )

    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
