"""How much do the two pass-2 defects of routes A and C move the headway?

Why this exists
---------------
In the two-pass pipeline (E2, E59; ``build_notebook_04.py:210-229``) the second
pass does two things the paper's Appendix A.1 would not describe:

D1. ``project_per_direction`` applies no 300 m lateral filter, so pings far off
    the route keep a valid arc coordinate ``s``.
D2. Pings that pass 1 labelled direction 0 get ``s = NaN``, and the second
    ``infer_direction`` runs over rows concatenated by direction block rather
    than by time. ``NaN > 0`` is True in polars, so those pings come out as
    direction +1 with no position.

Describing the pipeline so a reviewer can replicate it forces a choice between
declaring the defects and fixing them, and fixing them means rerunning every
kernel downstream. This builder measures what the fix changes before that choice
is made.

What it does
------------
1. Reruns the preprocessing of ``build_notebook_04.py`` locally, twice per
   two-pass corridor:
   - ``asis``: the exact running path. Its parquet must equal the published
     ``headways_E{n}.parquet``, or this local run is not the published one.
   - ``fixed``: the same path through the repaired ``project_per_direction``:
     direction-0 pings take the closer per-direction centerline, the 300 m
     filter is applied, and rows are sorted by time before the second
     ``infer_direction``.
2. Compares the two headway sets: coverage and the headway on shared keys.
3. Refits the XGBoost of notebook 22 (published configuration, 12-minute
   window) on each set and scores both with the paper's measurements, so the
   question "does the finding survive the fix" is answered on one model.

Every other behaviour of the pipeline is left as it runs; this isolates D1 and D2.

Requires ``data/processed/clean_gps.parquet`` (Kaggle dataset
``alexhuaracha/multibus-headway-forecast-clean``) and the published
``data/processed/headways_E{2,59}.parquet``.

Usage
-----
    uv run python -m src.build_pipeline_fix_check
"""
from __future__ import annotations

import os

# No single-thread pin: the preprocessing is heavy and its output is compared
# by key, not by bytes.

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.preprocessing.config import PRODUCTIVE_PARAMS  # noqa: E402
from src.preprocessing.corridor import (  # noqa: E402
    build_centerline,
    build_centerline_per_direction,
)
from src.preprocessing.direction import infer_direction  # noqa: E402
from src.preprocessing.headways import compute_headways_c2  # noqa: E402
from src.preprocessing.projection import (  # noqa: E402
    _project_arc_length,
    attach_observed_speed,
    project_per_direction,
    project_to_centerline,
)
from src.preprocessing.trips import assign_trip_ids, build_snapshots  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CLEAN_GPS = DATA_DIR / "clean_gps.parquet"
WORK_DIR = DATA_DIR / "pipeline-fix"
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_HEADWAY_CSV = CSV_DIR / "pipeline_fix_headways.csv"
OUT_XGB_CSV = CSV_DIR / "pipeline_fix_xgb.csv"

CORRIDORS = [("E2", 2), ("E59", 59)]
VARIANTS = ("asis", "fixed")
# The published parquet carries no ``empresaid`` or ``day`` column; one corridor
# per file and ``t`` make the key without them.
KEY = ["t", "direction", "pair_rank"]


def load(empresaid: int) -> pl.DataFrame:
    """The load cell of ``build_notebook_04.py:189-199``, for one corridor."""
    return (
        pl.scan_parquet(CLEAN_GPS)
        .filter(
            (pl.col("empresaid") == empresaid)
            & pl.col("time").is_not_null()
            & pl.col("lat").is_not_null()
            & pl.col("lon").is_not_null()
            & (pl.col("lat") != 0)
            & (pl.col("lon") != 0)
        )
        .with_columns(pl.col("time").dt.date().alias("day"))
        .sort(["empresaid", "unidadid", "time"])
        .collect()
    )


def _legacy_project_per_direction(gps: pl.DataFrame, centerlines: dict) -> pl.DataFrame:
    """``project_per_direction`` as it ran before the fix: the published path.

    Direction-0 pings get NaN s, no off-route filter, rows concatenated by
    direction block. Kept here so the ``asis`` headways stay regenerable after
    the production function was repaired.
    """
    parts = []
    for direction, cl in centerlines.items():
        subset = gps.filter(pl.col("direction") == direction)
        if subset.is_empty():
            continue
        s_arr, lat_arr = _project_arc_length(subset.select(["lat", "lon"]).to_numpy(), cl, 10_000)
        parts.append(subset.with_columns(
            pl.Series("s", s_arr.astype(float), dtype=pl.Float64),
            pl.Series("lateral_m", lat_arr.astype(float), dtype=pl.Float64),
        ))
    unknown = gps.filter(~pl.col("direction").is_in(list(centerlines)))
    if not unknown.is_empty():
        parts.append(unknown.with_columns(
            pl.lit(float("nan"), dtype=pl.Float64).alias("s"),
            pl.lit(float("nan"), dtype=pl.Float64).alias("lateral_m"),
        ))
    return pl.concat(parts)


def preprocess(empresaid: int, variant: str) -> pl.DataFrame:
    """The two-pass branch of ``build_notebook_04.py:210-229``, with or without the fix."""
    sub = attach_observed_speed(load(empresaid))

    centerline = build_centerline(sub, empresaid=empresaid)
    sub = project_to_centerline(sub, centerline, empresaid=empresaid)
    sub = infer_direction(sub)

    cls = build_centerline_per_direction(
        sub, empresaid=empresaid,
        min_pings_per_dir=PRODUCTIVE_PARAMS.centerline_min_pings_per_direction,
    )
    if variant == "asis":
        sub = _legacy_project_per_direction(sub, cls)
    else:
        sub = project_per_direction(sub, cls, empresaid=empresaid)
    sub = infer_direction(sub)
    sub = assign_trip_ids(sub)

    snaps = build_snapshots(sub)
    heads, _buckets = compute_headways_c2(snaps, sub)
    return heads


def headway_path(name: str, variant: str) -> Path:
    return WORK_DIR / f"headways_{name}_{variant}.parquet"


def build_headways() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for name, emp in CORRIDORS:
        for variant in VARIANTS:
            path = headway_path(name, variant)
            if path.exists():
                print(f"{name} {variant}: cached", flush=True)
                continue
            t0 = time.time()
            heads = preprocess(emp, variant)
            heads.write_parquet(path)
            print(f"{name} {variant}: {heads.height:,} rows ({time.time() - t0:.0f}s)",
                  flush=True)


def compare_headways() -> pl.DataFrame:
    rows = []
    for name, _emp in CORRIDORS:
        published = pl.read_parquet(DATA_DIR / f"headways_{name}.parquet")
        frames = {v: pl.read_parquet(headway_path(name, v)) for v in VARIANTS}

        asis = frames["asis"].sort(KEY)
        reproduced = asis.select(KEY + ["delta_t_min"]).equals(
            published.sort(KEY).select(KEY + ["delta_t_min"])
        )

        joined = frames["asis"].select(KEY + ["delta_t_min"]).join(
            frames["fixed"].select(KEY + [pl.col("delta_t_min").alias("fixed")]),
            on=KEY, how="inner",
        ).drop_nulls(["delta_t_min", "fixed"])
        diff = (joined["fixed"] - joined["delta_t_min"]).abs()

        for variant, frame in frames.items():
            valid = frame.get_column("delta_t_min").is_not_null()
            rows.append({
                "corridor": name,
                "variant": variant,
                "pairs": frame.height,
                "valid_pairs": int(valid.sum()),
                "coverage": float(valid.mean()),
                "median_headway_min": float(frame.get_column("delta_t_min").median()),
                "reproduces_published": reproduced if variant == "asis" else None,
                "shared_valid_pairs": joined.height,
                "share_identical": float((diff < 1e-9).mean()),
                "median_abs_change_min": float(diff.median()),
            })
    return pl.DataFrame(rows)


def refit_xgb() -> pl.DataFrame:
    """The paper's measurements on each headway set, from one XGBoost refit each.

    Each set is scored on its own population: the fix changes which pings exist,
    so a shared population would exclude exactly what the fix adds.
    """
    from src.build_window_sensitivity import (
        HORIZONS,
        PUBLISHED_WINDOW,
        fit_predict,
        prepare,
        published_configs,
        score,
    )

    configs = published_configs()
    rows = []
    for name, emp in CORRIDORS:
        for variant in VARIANTS:
            frame, max_n = prepare(name, emp, headway_path(name, variant))
            for horizon in HORIZONS:
                residuals = fit_predict(
                    frame, max_n, name, horizon, PUBLISHED_WINDOW, configs[(name, horizon)]
                )
                for row in score(residuals):
                    rows.append({"corridor": name, "variant": variant,
                                 "horizon": horizon, **row})
                print(f"{name} {variant} h{horizon}: refit done", flush=True)
    return pl.DataFrame(rows).sort(["corridor", "horizon", "model", "variant"])


def main() -> None:
    build_headways()
    table = compare_headways()
    table.write_csv(OUT_HEADWAY_CSV)
    print(table)
    xgb_table = refit_xgb()
    xgb_table.write_csv(OUT_XGB_CSV)
    print(xgb_table)


if __name__ == "__main__":
    sys.exit(main())
