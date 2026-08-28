"""Reporting cadence of the GPS fleet, measured off the cleaned position data.

Section IV-A opens on two claims about the raw feed: that each unit reports every
20 seconds, and that the cadence is regular rather than bursty. The second one is
what the grid of Section III-A rests on, and it is carried by a single statistic
— the median gap and the 95th percentile of the gap agree, so there is no long
tail of silence hiding behind a healthy average.

Gaps are measured **within a bus**. Two units that report at different hours
would otherwise produce a gap that is the corridor's idle time rather than
anyone's cadence. Each file already holds one company, so ``unidadid`` alone
identifies the bus here; anywhere else the composite key ``(empresaid,
unidadid)`` is required, because ``unidadid`` repeats across companies.

Input is the ``cleaned_gps_E*.parquet`` set published by the Kaggle kernel
``alexhuaracha/04-preprocessing`` and pinned in ``docs/dataset-manifest.md``.
Those files do not live in Git, so download them first:

    uv run --env-file .env kaggle kernels output alexhuaracha/04-preprocessing \\
        -p data/processed/

Output is ``docs/resultados/csv-multihorizon/gps_cadence.csv``, one row per
corridor.

Usage
-----
    uv run python -m src.build_gps_cadence
    uv run python -m src.build_gps_cadence --data-dir data/processed
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
OUT_NAME = "gps_cadence.csv"

CORRIDORS = ("E2", "E4", "E59")
REQUIRED_COLUMNS = ("unidadid", "t")
TAIL_QUANTILE = 0.95


class CadenceError(RuntimeError):
    """The input is not a ping table, so any interval off it would be a fiction."""


def cadence_row(frame: pl.DataFrame, corridor: str) -> dict[str, object]:
    """Median and tail of the interval between consecutive pings of one bus."""
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise CadenceError(
            f"{corridor}: ping frame is missing {missing} — a ping table carries "
            f"{list(REQUIRED_COLUMNS)}"
        )
    if frame.height == 0:
        raise CadenceError(f"{corridor}: ping frame has no rows")

    gaps = (
        frame.sort(["unidadid", "t"])
        .with_columns(
            pl.col("t").diff().over("unidadid").dt.total_seconds().alias("gap_s")
        )
        .drop_nulls("gap_s")
    )
    if gaps.height == 0:
        raise CadenceError(
            f"{corridor}: no consecutive pings from any single bus, so there is "
            "no interval to measure"
        )

    median = float(gaps["gap_s"].median())
    return {
        "corridor": corridor,
        "n_buses": int(frame["unidadid"].n_unique()),
        "n_gaps": int(gaps.height),
        "median_gap_s": median,
        "p95_gap_s": float(gaps["gap_s"].quantile(TAIL_QUANTILE, "nearest")),
        "emissions_per_minute": 60.0 / median,
    }


def build(data_dir: Path | str | None = None) -> pl.DataFrame:
    """Measure every corridor's cadence and return one row per corridor."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    rows: list[dict[str, object]] = []
    for corridor in CORRIDORS:
        path = directory / f"cleaned_gps_{corridor}.parquet"
        if not path.exists():
            raise CadenceError(
                f"missing {path} — download the kernel outputs first "
                "(see this module's docstring)"
            )
        frame = pl.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        rows.append(cadence_row(frame, corridor))
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="directory holding cleaned_gps_E*.parquet (default: data/processed)",
    )
    args = parser.parse_args()

    cadence = build(args.data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    cadence.write_csv(out_path)
    print(f"Cadencia escrita en {out_path.relative_to(REPO_ROOT)}")
    # Plain ASCII, not the polars repr: its box-drawing characters cannot be
    # encoded by the cp1252 console this project is run from.
    for row in cadence.to_dicts():
        print(
            f"  {row['corridor']:<4} mediana {row['median_gap_s']:>6.1f} s | "
            f"p95 {row['p95_gap_s']:>6.1f} s | "
            f"{row['emissions_per_minute']:.2f} emisiones/min | "
            f"n={row['n_gaps']}"
        )


if __name__ == "__main__":
    main()
