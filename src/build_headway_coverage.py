"""Headway coverage per corridor, measured off the headway parquets.

Section III-A evaluates Equation (3) on every consecutive bus pair, and the
equation does not always answer: a pair with no crossing, or one whose headway
exceeds thirty minutes, is emitted as "no data". Section IV-A reports how often
that happened, and this builder is where that number comes from.

The counting rule is one expression — non-null ``delta_t_min`` over every pair
row — but the denominator is what carries the meaning. Every pair the pipeline
attempted stays in it, so the percentage answers "of the pairs we tried to
measure, how many answered", and not "of the answers we got, how many are
answers". The second question is a tautology; only the first one bounds the
population the results describe.

Input is the ``headways_E*.parquet`` set published by the Kaggle kernel
``alexhuaracha/04-preprocessing`` and pinned in ``docs/dataset-manifest.md``.
Those files do not live in Git, so download them first:

    uv run --env-file .env kaggle kernels output alexhuaracha/04-preprocessing \\
        -p data/processed/

Output is ``docs/resultados/csv-multihorizon/headway_coverage.csv``, one row per
``(corridor, direction)`` with raw counts only. The percentage is derived at
render time by ``build_paper_tables``, so the CSV keeps the primitive and the
manuscript can never carry a rounded number with no counts behind it.

Usage
-----
    uv run python -m src.build_headway_coverage
    uv run python -m src.build_headway_coverage --data-dir data/processed
"""

from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_NAME = "headway_coverage.csv"

CORRIDORS = ("E2", "E4", "E59")
REQUIRED_COLUMNS = ("direction", "delta_t_min")


class CoverageError(RuntimeError):
    """The input is not a pair table, so any count off it would be a fiction."""


def coverage_rows(frame: pl.DataFrame, corridor: str) -> list[dict[str, object]]:
    """Count answered pairs against attempted pairs, per direction."""
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise CoverageError(
            f"{corridor}: headway frame is missing {missing} — a pair table "
            f"carries {list(REQUIRED_COLUMNS)}"
        )
    if frame.height == 0:
        raise CoverageError(f"{corridor}: headway frame has no rows")
    if frame.filter(pl.col("direction") == 0).height:
        raise CoverageError(
            f"{corridor}: frame carries direction 0, which compute_pairs drops. "
            "Counting it would inflate the denominator with rows that were "
            "never pairs"
        )

    grouped = (
        frame.group_by("direction")
        .agg(
            pl.len().alias("total_pairs"),
            pl.col("delta_t_min").is_not_null().sum().alias("valid_pairs"),
        )
        .sort("direction")
    )
    return [
        {
            "corridor": corridor,
            "direction": int(row["direction"]),
            "valid_pairs": int(row["valid_pairs"]),
            "total_pairs": int(row["total_pairs"]),
        }
        for row in grouped.to_dicts()
    ]


def aggregate_coverage(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse the directions into one coverage figure per corridor.

    Both directions of a corridor are the same corpus seen twice, so the paper
    quotes the corridor and keeps the directional split for the audit trail.
    """
    aggregate = frame.group_by("corridor").agg(
        pl.col("valid_pairs").sum(),
        pl.col("total_pairs").sum(),
    )
    if aggregate.filter(pl.col("total_pairs") == 0).height:
        raise CoverageError("a corridor reports no pairs at all; nothing to divide")

    order = pl.DataFrame(
        {"corridor": list(CORRIDORS), "_order": list(range(len(CORRIDORS)))}
    )
    return (
        aggregate.with_columns(
            (100 * pl.col("valid_pairs") / pl.col("total_pairs")).alias("coverage_pct")
        )
        .join(order, on="corridor", how="left")
        .sort("_order", nulls_last=True)
        .drop("_order")
    )


def build(data_dir: Path | str | None = None) -> pl.DataFrame:
    """Read every corridor's pair table and return the per-direction counts."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    rows: list[dict[str, object]] = []
    for corridor in CORRIDORS:
        path = directory / f"headways_{corridor}.parquet"
        if not path.exists():
            raise CoverageError(
                f"missing {path} — download the kernel outputs first "
                "(see this module's docstring)"
            )
        frame = pl.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        rows.extend(coverage_rows(frame, corridor))
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="directory holding headways_E*.parquet (default: data/processed)",
    )
    args = parser.parse_args()

    per_direction = build(args.data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    per_direction.write_csv(out_path)
    print(f"Conteos escritos en {out_path.relative_to(REPO_ROOT)}")
    # Plain ASCII, not the polars repr: its box-drawing characters cannot be
    # encoded by the cp1252 console this project is run from.
    for row in aggregate_coverage(per_direction).to_dicts():
        print(
            f"  {row['corridor']:<4} {row['valid_pairs']:>9} / "
            f"{row['total_pairs']:>9} = {row['coverage_pct']:.2f} %"
        )


if __name__ == "__main__":
    main()
