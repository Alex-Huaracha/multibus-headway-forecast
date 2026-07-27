"""Materialize and freeze the canonical sample index (plan step 2).

The retrained pipeline's central contract is that the LSTM, the XGBoost baseline
and persistence all consume **the same population**. Historically they did not:
XGBoost predicted one row of ``headways_E*.parquet`` per sample while the LSTM
predicted one window of ``T_in + horizon`` rows, and the two were reconciled by a
post-hoc join (audit §2.1). That join corrected the report, not the cause.

This builder closes it. It materializes the index defined by
``src/data/sample_index.py`` (contracts C1 and C2) for every corridor, horizon
and split, and freezes a SHA-256 over a canonical serialization of each.

The frozen digest is the enforcement mechanism: each Kaggle kernel recomputes the
index in-kernel from the hash-pinned parquet and asserts its digest matches this
manifest. Same code plus same input bytes yields the same index, so "shared
population" becomes verifiable rather than assumed — no index file needs to ship
alongside the data.

Outputs
-------
``docs/resultados/csv-multihorizon/sample_index_manifest.csv`` — one row per
(corridor, horizon, split) with row counts, effective scalar n, and the digest.

Usage
-----
    uv run python -m src.build_sample_index
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from src.data.sample_index import make_sample_index  # noqa: E402
from src.evaluation.splits import (  # noqa: E402
    MAIN_FOLD,
    ROLLING_FOLDS,
    Fold,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "sample_index_manifest.csv"

CORRIDORS: tuple[tuple[int, str], ...] = ((2, "E2"), (59, "E59"), (4, "E4"))
HORIZONS: tuple[int, ...] = (1, 3, 5, 10)
T_IN: int = 12

#: Bounds of the PUBLISHED split. Kept as a module constant because several
#: analysis builders import it directly; fold-aware callers should use
#: ``fold.bounds()`` instead.
SPLIT_BOUNDS = MAIN_FOLD.bounds()


def load_corridor(empresaid: int) -> pl.DataFrame:
    """Headway parquet with the implicit ``empresaid`` materialized as a column.

    The parquets are one file per company, so ``empresaid`` is not stored in
    them. Every consumer must add it back the same way or the join keys differ.
    """
    path = DATA_DIR / f"headways_E{empresaid}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Download it first:\n"
            "  uv run kaggle kernels output alexhuaracha/04-preprocessing -p <dir>"
        )
    return pl.read_parquet(path).with_columns(
        pl.lit(empresaid, dtype=pl.Int64).alias("empresaid")
    )


def index_digest(index: pl.DataFrame) -> str:
    """SHA-256 over a canonical serialization of the index.

    Parquet is not byte-stable across writer versions, so the digest is taken
    over a CSV rendering of the already-sorted index. That is stable as long as
    the column order and the datetime format are, both of which are fixed here.
    """
    canonical = index.select(
        ["empresaid", "direction", "start_ts", "target_ts", "horizon"]
    ).sort(["empresaid", "direction", "horizon", "start_ts"])
    payload = canonical.write_csv(datetime_format="%Y-%m-%dT%H:%M:%S")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_scalar_n(index: pl.DataFrame, frame: pl.DataFrame) -> int:
    """Count of scalar predictions: one per (sample, pair_rank) with a target.

    A sample predicts the whole headway vector, so the unit the significance
    tests operate on is the (sample, vector position) pair, not the sample.

    This counts cells with a valid TARGET only. It is an upper bound on the
    usable n — see ``paired_scalar_n``.
    """
    if index.height == 0:
        return 0
    joined = index.join(
        frame.select(["empresaid", "direction", "t", "pair_rank", "delta_t_min"]),
        left_on=["empresaid", "direction", "target_ts"],
        right_on=["empresaid", "direction", "t"],
        how="inner",
    )
    return joined.filter(pl.col("delta_t_min").is_not_null()).height


def paired_scalar_n(index: pl.DataFrame, frame: pl.DataFrame) -> int:
    """Count of cells where BOTH the target and persistence are observed.

    This is the n the paired significance tests actually run on, and it is
    materially smaller than ``effective_scalar_n``: a cell whose target exists
    but whose last input snapshot is missing cannot be compared against B1, so
    it is dropped from every paired comparison. Reporting only the target count
    overstates statistical power — measured 2026-07-27 on E2 h=3 test, the
    target count is 134 931 while the paired count is 81 695.

    Persistence is the value at the last input snapshot, i.e.
    ``start_ts + (T_in - 1)`` minutes, which contract C2 guarantees exists on the
    same contiguous run as the target.

    Caveat: this count does NOT apply the ``max_N`` truncation the models apply
    (AC-MAXN-2 drops ``pair_rank >= max_N``), so it is a slight upper bound on
    what a model actually consumes — measured at 2 rows out of 81 697 for
    E2 h=3 test. ``max_N`` is a train-derived quantity and pulling it in here
    would couple the index manifest to the normalization layer.
    """
    if index.height == 0:
        return 0
    values = frame.select(["empresaid", "direction", "t", "pair_rank", "delta_t_min"])
    with_persist_ts = index.with_columns(
        (pl.col("start_ts") + pl.duration(minutes=T_IN - 1)).alias("persist_ts")
    )
    targets = with_persist_ts.join(
        values.rename({"delta_t_min": "y_true"}),
        left_on=["empresaid", "direction", "target_ts"],
        right_on=["empresaid", "direction", "t"],
        how="inner",
    ).filter(pl.col("y_true").is_not_null())

    paired = targets.join(
        values.rename({"delta_t_min": "y_persist"}),
        left_on=["empresaid", "direction", "persist_ts", "pair_rank"],
        right_on=["empresaid", "direction", "t", "pair_rank"],
        how="inner",
    )
    return paired.filter(pl.col("y_persist").is_not_null()).height


def build(folds: tuple[Fold, ...] = ROLLING_FOLDS) -> pl.DataFrame:
    """One row per (fold, corridor, horizon, split). Deterministic.

    The digest is taken over the index CONTENT, so the published fold's digests
    are unaffected by the arrival of the rolling ones — the manifest gains a
    column and rows, and contract C1 keeps holding for work already done.
    """
    rows: list[dict] = []

    for empresaid, corridor in CORRIDORS:
        frame = load_corridor(empresaid)
        day = pl.col("t").dt.date()

        for fold in folds:
            for split, (lo, hi) in fold.bounds().items():
                part = frame.filter((day >= lo) & (day <= hi))
                n_snapshots = part.select(["direction", "t"]).unique().height

                for horizon in HORIZONS:
                    index = make_sample_index(part, horizon=horizon, T_in=T_IN)
                    rows.append(
                        {
                            "fold": fold.name,
                            "corridor": corridor,
                            "empresaid": empresaid,
                            "horizon": horizon,
                            "split": split,
                            "n_days": (hi - lo).days + 1,
                            "n_snapshots": n_snapshots,
                            "n_samples": index.height,
                            "pct_snapshots_usable": (
                                round(100.0 * index.height / n_snapshots, 4)
                                if n_snapshots
                                else 0.0
                            ),
                            "n_scalar_effective": effective_scalar_n(index, part),
                            "n_scalar_paired": paired_scalar_n(index, part),
                            "sha256": index_digest(index),
                        }
                    )

    return pl.DataFrame(rows).sort(["fold", "corridor", "split", "horizon"])


def main() -> None:
    manifest = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=200, tbl_cols=12):
        print(manifest.select(
            ["fold", "corridor", "split", "horizon", "n_days", "n_samples",
             "pct_snapshots_usable", "n_scalar_paired"]
        ))
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({manifest.height} rows)")


if __name__ == "__main__":
    main()
