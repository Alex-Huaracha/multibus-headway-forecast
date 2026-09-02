"""Confidence bounds on detection precision, per model x corridor x horizon.

Section V-C reports that the learner's bunching detector collapsed in coverage
but not in precision, and the sharpest instance of that is also the thinnest:
on E2 at ten minutes it fired fourteen times and hit ten real events. Seventy-one
percent off fourteen trials is not a value, and quoting it bare invites the
reader to treat it as one.

This builder bounds it. The interval is Clopper-Pearson at 95 %, computed by
``src.evaluation.vector_metrics.precision_interval``, over the hit and fire
counts already published in ``contiguous_vector_metrics.csv``. Reading the
counts rather than the residuals keeps this CSV consistent with the F1 column
of Table 1 by construction: both descend from the same confusion matrix, so a
bound here can never disagree with a score there.

Cells where the detector never fired carry null bounds. Clopper-Pearson would
return [0, 1] for zero trials, and a full-width interval printed next to real
ones reads as a measurement of total ignorance rather than the absence of a
measurement.

Output is ``docs/resultados/csv-multihorizon/detection_precision_ci.csv``.

Usage
-----
    uv run python -m src.build_detection_precision_ci
"""

from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from src.evaluation.vector_metrics import precision_interval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
SOURCE_NAME = "contiguous_vector_metrics.csv"
OUT_NAME = "detection_precision_ci.csv"

LEVEL = 0.95
CORRIDORS = ("E2", "E4", "E59")
HORIZONS = (1, 3, 5, 10)
REQUIRED_COLUMNS = ("model", "corridor", "horizon", "bunching_tp", "bunching_fp")


class PrecisionCIError(RuntimeError):
    """The source is not the vector-metrics table, so any bound off it is a fiction."""


def _load(path: Path | None = None) -> pl.DataFrame:
    source = path if path is not None else OUT_DIR / SOURCE_NAME
    if not source.exists():
        raise PrecisionCIError(
            f"missing {source} — run `uv run python -m "
            "src.build_contiguous_vector_metrics` first"
        )
    frame = pl.read_csv(source)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise PrecisionCIError(f"{source.name} is missing {missing}")
    return frame


def build(path: Path | None = None) -> pl.DataFrame:
    """Bound the precision of every published detection cell."""
    frame = _load(path)

    order = {corridor: index for index, corridor in enumerate(CORRIDORS)}
    rows: list[dict[str, object]] = []
    for row in frame.iter_rows(named=True):
        fires = int(row["bunching_tp"]) + int(row["bunching_fp"])
        hits = int(row["bunching_tp"])
        if fires:
            low, high = precision_interval(hits, fires, level=LEVEL)
            precision = hits / fires
        else:
            low = high = precision = None
        rows.append({
            "model": row["model"],
            "corridor": row["corridor"],
            "horizon": int(row["horizon"]),
            "fires": fires,
            "hits": hits,
            "precision": precision,
            "ci_low": low,
            "ci_high": high,
            "level": LEVEL,
        })

    if not rows:
        raise PrecisionCIError("the vector-metrics table has no detection cells")

    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("corridor").replace_strict(order).alias("_order"))
        .sort("model", "_order", "horizon")
        .drop("_order")
    )


def render(frame: pl.DataFrame) -> str:
    """The CSV text, so determinism can be asserted without touching the disk."""
    return frame.write_csv()


def main() -> None:
    frame = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    out_path.write_text(render(frame), encoding="utf-8", newline="")
    print(f"Intervalos escritos en {out_path.relative_to(REPO_ROOT)}")
    # Plain ASCII, not the polars repr: its box-drawing characters cannot be
    # encoded by the cp1252 console this project is run from.
    for row in frame.iter_rows(named=True):
        if row["fires"]:
            bounds = f"{row['ci_low'] * 100:5.1f} % a {row['ci_high'] * 100:5.1f} %"
            point = f"{row['precision'] * 100:5.1f} %"
        else:
            bounds = "sin disparos"
            point = "    -"
        print(
            f"  {row['model']:<12} {row['corridor']:<4} h={row['horizon']:<3} "
            f"{row['hits']:>6}/{row['fires']:<6} {point}  [{bounds}]"
        )


if __name__ == "__main__":
    main()
