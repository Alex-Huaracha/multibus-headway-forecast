"""Inspect the two raw CSVs to decide merge/discard strategy."""
import polars as pl
from pathlib import Path

FILES = {
    "satchek1": "/mnt/c/Users/Programador/Downloads/satchek csvs/satchek1.csv",
    "satchek2": "/mnt/c/Users/Programador/Downloads/satchek csvs/satchek2.csv",
}

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
