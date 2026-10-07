"""Why do routes A and C get a second pass and route B does not?

Why this exists
---------------
The two-pass centerline (one axis per direction) runs on E2 and E59 only
(``config.EMPRESA_CONFIG``); E4 keeps a single axis. The config records that
choice as provisional, with no measurement behind E4. A reviewer will ask why
the second pass is not applied everywhere. The answer is geometric: the second
pass only matters where the two directions run on different streets. This
builder measures how far apart the two directions run on each route.

What it does
------------
For each route it fits one centerline per direction with the production
function ``build_centerline_per_direction``, on the direction labels the
published pipeline produced (``cleaned_gps_E{n}.parquet``). It then samples
the direction +1 axis densely (20 points per segment) and measures the
distance from each point to the direction -1 axis with the production
projection ``_project_arc_length``. The distribution of that distance is the
separation between the two directions along the route.

Result (2026-10-07): median separation 35 m on A (E2), 4 m on B (E4), 19 m on
C (E59); share of the route more than 50 m apart 41 %, 3 % and 33 %. On B the
two directions share the streets, so a per-direction axis reproduces the
single one.

Requires ``data/processed/cleaned_gps_E{2,4,59}.parquet`` (outputs of
notebooks 04 and 16; see ``docs/dataset-manifest.md``).

Usage
-----
    uv run python -m src.build_direction_separation
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.preprocessing.config import PRODUCTIVE_PARAMS  # noqa: E402
from src.preprocessing.corridor import build_centerline_per_direction  # noqa: E402
from src.preprocessing.projection import _project_arc_length  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_CSV = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "direction_separation.csv"

ROUTES = [("A", "E2", 2), ("B", "E4", 4), ("C", "E59", 59)]
POINTS_PER_SEGMENT = 20
APART_M = 50.0


def separation(empresaid: int) -> np.ndarray:
    """Distance (m) from points along the direction +1 axis to the direction -1 axis."""
    gps = pl.read_parquet(
        DATA_DIR / f"cleaned_gps_E{empresaid}.parquet",
        columns=["lat", "lon", "direction", "speed_kmh"],
    ).with_columns(pl.lit(empresaid, dtype=pl.Int64).alias("empresaid"))
    axes = build_centerline_per_direction(
        gps, empresaid=empresaid,
        min_pings_per_dir=PRODUCTIVE_PARAMS.centerline_min_pings_per_direction,
    )
    forward, backward = axes[1], axes[-1]
    points = np.concatenate([
        np.linspace(forward[k], forward[k + 1], POINTS_PER_SEGMENT)
        for k in range(len(forward) - 1)
    ])
    _s, lateral = _project_arc_length(points, backward, chunk_size=10_000)
    return lateral.astype(np.float64)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    rows = []
    for route, corridor, empresaid in ROUTES:
        dist = separation(empresaid)
        rows.append({
            "route": route,
            "corridor": corridor,
            "two_pass": empresaid in (2, 59),
            "median_m": float(np.median(dist)),
            "p75_m": float(np.percentile(dist, 75)),
            "p90_m": float(np.percentile(dist, 90)),
            "max_m": float(dist.max()),
            f"share_over_{int(APART_M)}m": float((dist > APART_M).mean()),
        })
    table = pl.DataFrame(rows)
    table.write_csv(OUT_CSV)
    print(table)


if __name__ == "__main__":
    sys.exit(main())
