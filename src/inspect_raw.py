"""Inspect the raw CSVs to decide merge/discard strategy.

The export arrived as two CSVs, but the paths are the caller's: they live outside
the repository and differ per machine, so they are arguments rather than
constants.

Usage
-----
    uv run python src/inspect_raw.py <csv> [<csv> ...]
"""
import sys
from pathlib import Path

import polars as pl

if len(sys.argv) < 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} <csv> [<csv> ...]")

FILES = {Path(p).stem: p for p in sys.argv[1:]}

for name, path in FILES.items():
    print(f"\n========== {name} ==========")
    lf = pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10000)

    summary = lf.select([
        pl.len().alias("rows"),
        pl.col("time").min().alias("date_min"),
        pl.col("time").max().alias("date_max"),
        pl.col("empresaid").n_unique().alias("n_empresas"),
        pl.col("unidadid").n_unique().alias("n_unidades"),
        pl.col("lat").is_null().sum().alias("null_lat"),
        pl.col("time").is_null().sum().alias("null_time"),
    ]).collect(engine="streaming")
    print(summary)

    empresas = (
        lf.group_by("empresaid")
        .agg(pl.len().alias("rows"))
        .sort("empresaid")
        .collect(engine="streaming")
    )
    print(f"Empresas en {name}:")
    print(empresas)
