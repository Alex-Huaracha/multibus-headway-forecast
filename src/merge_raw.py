"""One-shot: merge the two satchek CSVs into a single parquet.

Why this exists:
The raw export from the university came split in 2 CSVs (~6.2GB total) due to
DB pagination. The split is not methodologically meaningful — empresas 56 and 58
appear in both files. This script unifies them so the rest of the pipeline
(viability analysis, filtering, modeling) sees a single coherent raw dataset.

Output: data/raw/raw_gps.parquet

The CSVs live outside the repository and their location differs per machine, so
they are arguments rather than constants.

Usage
-----
    uv run python src/merge_raw.py <csv> [<csv> ...]
"""
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "raw_gps.parquet"

if len(sys.argv) < 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} <csv> [<csv> ...]")

INPUTS = sys.argv[1:]

t0 = time.time()
print(f"Reading and merging {len(INPUTS)} CSVs (streaming)...")

lf = pl.concat(
    [pl.scan_csv(p, try_parse_dates=True, infer_schema_length=10000) for p in INPUTS],
    how="vertical",
)

# Sink directly to parquet without materializing in memory.
lf.sink_parquet(OUT, compression="zstd", compression_level=3)

elapsed = time.time() - t0
size_gb = OUT.stat().st_size / 1e9
print(f"Done in {elapsed:.1f}s -> {OUT} ({size_gb:.2f} GB)")

# Quick sanity check
lf_out = pl.scan_parquet(OUT)
summary = lf_out.select([
    pl.len().alias("rows"),
    pl.col("empresaid").n_unique().alias("n_empresas"),
    pl.col("time").min().alias("date_min"),
    pl.col("time").max().alias("date_max"),
]).collect()
print(summary)
