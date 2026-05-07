"""Generate the 01_viability_and_filter.ipynb file for Kaggle."""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "01_viability_and_filter" / "01_viability_and_filter.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip()))


md("""
# 01 — Viability Analysis and Corridor Filtering

This notebook validates the corridor-selection criteria stated in the proposal
(section 4.3) directly from the raw GPS data, and produces the filtered clean
dataset consumed by the rest of the pipeline.

**Input**: `multibus-headway-forecast-raw` (single Parquet, ~100M rows, 12 empresas).

**Outputs** (to `/kaggle/working/`):
- `viability.csv` — viability table per empresa (linearity + simultaneous-bus stats).
- `clean_gps.parquet` — GPS records filtered to the 4 selected corridors.

## Viability criteria

A corridor is considered viable if **both** hold:

1. **Linearity**: the GPS cloud of the empresa is dominated by a single direction.
   Operationalized as `PC1_var / PC2_var ≥ 4` (PCA on `(lat, lon)`).
2. **Simultaneous fleet**: enough buses circulate at the same time so that a vector
   of headways is meaningful. Operationalized as `median simultaneous buses ≥ 5`.

These criteria justify selecting empresas **2, 4, 58, 59** and excluding the other 8.

## Composite key reminder

`unidadid` values are reused across empresas. Always group/join with the composite
key `(empresaid, unidadid)`.
""")

code("""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("/kaggle/input/multibus-headway-forecast-raw/raw_gps.parquet")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(exist_ok=True)

SELECTED_EMPRESAS = [2, 4, 58, 59]
LINEARITY_THRESHOLD = 4.0
SIMULTANEOUS_THRESHOLD = 5

assert INPUT.exists(), f"Raw dataset not found at {INPUT}"
print(f"Reading from: {INPUT}")
""")

md("## 1. Dataset overview")

code("""
lf = pl.scan_parquet(INPUT)

overview = lf.select([
    pl.len().alias("rows"),
    pl.col("empresaid").n_unique().alias("n_empresas"),
    pl.struct(["empresaid", "unidadid"]).n_unique().alias("n_unidades_compkey"),
    pl.col("time").min().alias("date_min"),
    pl.col("time").max().alias("date_max"),
    pl.col("lat").is_null().sum().alias("null_lat"),
]).collect(engine="streaming")
overview
""")

code("""
per_empresa = (
    lf.group_by("empresaid")
    .agg([
        pl.len().alias("rows"),
        pl.struct(["empresaid", "unidadid"]).n_unique().alias("n_unidades"),
        pl.col("lat").is_null().sum().alias("null_lat"),
    ])
    .sort("empresaid")
    .collect(engine="streaming")
)
per_empresa
""")

md("""
## 2. Linearity per empresa (PCA)

For each empresa we sample up to 200k GPS points and compute the PCA of `(lat, lon)`.
The ratio `eig1 / eig2` (variance along principal direction over the orthogonal one)
quantifies linearity. Values ≥ 4 indicate a corridor-shaped trajectory; values close
to 1 indicate a 2D spread (city-wide or non-linear route).
""")

code("""
def pca_ratio(empresa_id: int, sample_size: int = 200_000) -> tuple[float | None, int]:
    pts = (
        lf.filter(pl.col("empresaid") == empresa_id)
        .filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
        .filter(pl.col("lat") != 0.0)
        .select(["lat", "lon"])
        .collect(engine="streaming")
    )
    n = pts.height
    if n < 100:
        return None, n
    if n > sample_size:
        pts = pts.sample(n=sample_size, seed=42)
    arr = pts.to_numpy()
    arr = arr - arr.mean(axis=0)
    cov = np.cov(arr.T)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    return float(eigvals[1] / eigvals[0]), n

empresas = sorted(per_empresa["empresaid"].to_list())
linearity_rows = []
for e in empresas:
    ratio, n = pca_ratio(e)
    linearity_rows.append({"empresaid": e, "pca_ratio": ratio, "points": n})

linearity_df = pl.DataFrame(linearity_rows).sort("pca_ratio", descending=True, nulls_last=True)
linearity_df
""")

md("""
## 3. Simultaneous buses per empresa

For each empresa we count distinct active units per minute, then summarize the
distribution. The median over all observed minutes captures typical operation;
the p95 and max approximate peak-hour fleet size.
""")

code("""
simultaneous = (
    lf.filter(pl.col("time").is_not_null())
    .with_columns(pl.col("time").dt.truncate("1m").alias("minute"))
    .group_by(["empresaid", "minute"])
    .agg(pl.struct(["empresaid", "unidadid"]).n_unique().alias("active_buses"))
    .group_by("empresaid")
    .agg([
        pl.col("active_buses").median().alias("median_active"),
        pl.col("active_buses").quantile(0.95).alias("p95_active"),
        pl.col("active_buses").max().alias("max_active"),
    ])
    .sort("empresaid")
    .collect(engine="streaming")
)
simultaneous
""")

md("## 4. Viability table")

code("""
viability = (
    per_empresa.select(["empresaid", "rows", "n_unidades"])
    .join(linearity_df, on="empresaid", how="left")
    .join(simultaneous, on="empresaid", how="left")
    .with_columns([
        (pl.col("pca_ratio") >= LINEARITY_THRESHOLD).alias("pass_linearity"),
        (pl.col("median_active") >= SIMULTANEOUS_THRESHOLD).alias("pass_simultaneous"),
    ])
    .with_columns(
        (pl.col("pass_linearity") & pl.col("pass_simultaneous")).alias("viable")
    )
    .sort("empresaid")
)
viability_pd = viability.to_pandas()
viability_pd.to_csv(OUTPUT_DIR / "viability.csv", index=False)
viability
""")

code("""
print("Empresas viable per criteria:", sorted(viability.filter(pl.col("viable"))["empresaid"].to_list()))
print("Empresas selected in paper :", SELECTED_EMPRESAS)
""")

md("""
## 5. Filter and save clean dataset

Filter the raw GPS records to the 4 viable empresas and write the result as Parquet
in `/kaggle/working/`. The output of this notebook becomes the input dataset for the
rest of the pipeline (preprocessing → headways → models).
""")

code("""
clean_path = OUTPUT_DIR / "clean_gps.parquet"
(
    lf.filter(pl.col("empresaid").is_in(SELECTED_EMPRESAS))
    .sink_parquet(clean_path, compression="zstd", compression_level=3)
)

final = pl.scan_parquet(clean_path).select([
    pl.len().alias("rows"),
    pl.col("empresaid").n_unique().alias("n_empresas"),
    pl.struct(["empresaid", "unidadid"]).n_unique().alias("n_unidades"),
    pl.col("time").min().alias("date_min"),
    pl.col("time").max().alias("date_max"),
]).collect(engine="streaming")

size_gb = clean_path.stat().st_size / 1e9
print(f"Saved {clean_path} ({size_gb:.2f} GB)")
final
""")

md("""
## Next step

The output `clean_gps.parquet` should be saved as a new Kaggle dataset
(`multibus-headway-forecast-clean`) and consumed by `02_preprocessing`.
""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT}")
