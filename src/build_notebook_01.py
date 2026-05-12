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
- `viability.csv` — viability table per empresa with linearity, simultaneous-bus stats, and exclusion reason.
- `sensitivity.csv` — how empresa selection changes under different thresholds.
- `clean_gps.parquet` — GPS records filtered to the 4 selected corridors.
- `figuras/pca_por_empresa.png` — PCA scatter per empresa.
- `figuras/buses_simultaneos_por_hora.png` — active fleet by hour-of-day per empresa.

## Viability criteria

A corridor is considered viable if **both** hold:

1. **Linearity**: the route shape is dominated by a single direction.
   Operationalized as `PC1_var / PC2_var ≥ 4` (PCA on `(lat, lon)`),
   computed **after removing stationary points** so terminal idle time
   does not bias the ratio.
2. **Simultaneous fleet**: enough *active* buses circulate at the same time so a
   vector of headways is meaningful. Operationalized as `median simultaneous
   active buses ≥ 5`. A bus is "active" if it moved more than 50 m in the last
   5 minutes (parked/garaged buses do not count).

These criteria justify selecting empresas **2, 4, 58, 59** and excluding the others.

## Composite key reminder

`unidadid` values are reused across empresas (28 of 126 appear in 3+ empresas).
Every per-bus aggregation in this notebook uses the composite key
`(empresaid, unidadid)`.
""")

code("""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("/kaggle/input/multibus-headway-forecast-raw/raw_gps.parquet")
OUTPUT_DIR = Path("/kaggle/working")
FIG_DIR = OUTPUT_DIR / "figuras"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SELECTED_EMPRESAS = [2, 4, 58, 59]
LINEARITY_THRESHOLD = 4.0
SIMULTANEOUS_THRESHOLD = 5
STATIONARY_DIST_M = 50.0   # threshold for "moved" in the last window
STATIONARY_WINDOW_MIN = 5  # window size to assess movement

# Local meters-per-degree at Arequipa (lat ≈ -16.4°)
LAT_DEG_M = 111_000.0
LON_DEG_M = 111_000.0 * np.cos(np.deg2rad(-16.4))

assert INPUT.exists(), f"Raw dataset not found at {INPUT}"
print(f"Reading from: {INPUT}")
""")

md("## 1. Dataset overview and sanity checks")

code("""
lf = pl.scan_parquet(INPUT)

overview = lf.select([
    pl.len().alias("rows"),
    pl.col("empresaid").n_unique().alias("n_empresas"),
    pl.struct(["empresaid", "unidadid"]).n_unique().alias("n_unidades_compkey"),
    pl.col("unidadid").n_unique().alias("n_unidadid_naive"),
    pl.col("time").min().alias("date_min"),
    pl.col("time").max().alias("date_max"),
    pl.col("lat").is_null().sum().alias("null_lat"),
    ((pl.col("lat") == 0) | (pl.col("lon") == 0)).sum().alias("zero_coords"),
]).collect(engine="streaming")
overview
""")

code("""
# Composite-key sanity: confirm unidadid reuse and no duplicate (empresaid, unidadid, time).
n_unidad_naive = overview["n_unidadid_naive"][0]
n_unidad_comp = overview["n_unidades_compkey"][0]
print(f"unidadid únicos (sin empresaid): {n_unidad_naive}")
print(f"(empresaid, unidadid) únicos   : {n_unidad_comp}")
print(f"Diferencia (reúso entre empresas): {n_unidad_comp - n_unidad_naive}")

dup_check = (
    lf.group_by(["empresaid", "unidadid", "time"])
    .agg(pl.len().alias("n"))
    .filter(pl.col("n") > 1)
    .select(pl.len().alias("dup_rows"))
    .collect(engine="streaming")
)
print(f"Filas duplicadas con clave (empresaid, unidadid, time): {dup_check['dup_rows'][0]}")
""")

code("""
per_empresa = (
    lf.group_by("empresaid")
    .agg([
        pl.len().alias("rows"),
        pl.struct(["empresaid", "unidadid"]).n_unique().alias("n_unidades"),
    ])
    .sort("empresaid")
    .collect(engine="streaming")
)
per_empresa
""")

md("""
## 2. Build a clean working frame: drop bad coords, mark stationary points

We collect a per-empresa working frame in memory (after dropping null/zero
coordinates) and flag each record as **stationary** if the bus moved less than
`STATIONARY_DIST_M` meters in the last `STATIONARY_WINDOW_MIN` minutes,
computed per `(empresaid, unidadid)`.

Stationary points are excluded from both the PCA (so terminal idle time does
not inflate or shrink the linearity ratio) and the simultaneous-fleet count
(so parked buses do not count as active).
""")

code("""
def load_empresa(empresa_id: int) -> pl.DataFrame:
    df = (
        lf.filter(pl.col("empresaid") == empresa_id)
        .filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
        .filter((pl.col("lat") != 0) & (pl.col("lon") != 0))
        .filter(pl.col("time").is_not_null())
        .select(["empresaid", "unidadid", "time", "lat", "lon"])
        .sort(["empresaid", "unidadid", "time"])
        .collect(engine="streaming")
    )
    if df.is_empty():
        return df
    # Per-bus shifted coordinates over the composite key.
    df = df.with_columns([
        pl.col("lat").shift(1).over(["empresaid", "unidadid"]).alias("lat_prev"),
        pl.col("lon").shift(1).over(["empresaid", "unidadid"]).alias("lon_prev"),
        pl.col("time").shift(1).over(["empresaid", "unidadid"]).alias("time_prev"),
    ])
    df = df.with_columns([
        (((pl.col("lat") - pl.col("lat_prev")) * LAT_DEG_M) ** 2
         + ((pl.col("lon") - pl.col("lon_prev")) * LON_DEG_M) ** 2).sqrt().alias("step_m"),
        (pl.col("time") - pl.col("time_prev")).dt.total_seconds().alias("dt_s"),
    ])
    # Cumulative distance and point-count over a rolling 5-min window per bus.
    # Use `by=` (not `.over()`) — combining rolling_*_by with over is invalid and
    # can cross bus boundaries silently.
    df = df.with_columns([
        pl.col("step_m").fill_null(0.0).rolling_sum_by(
            "time",
            window_size=f"{STATIONARY_WINDOW_MIN}m",
            by=["empresaid", "unidadid"],
        ).alias("dist_5m"),
        pl.col("step_m").rolling_count_by(
            "time",
            window_size=f"{STATIONARY_WINDOW_MIN}m",
            by=["empresaid", "unidadid"],
        ).alias("n_5m"),
    ])
    # Mark stationary only when the rolling window has enough points to decide.
    # An immature window (e.g. the first minutes of a bus's day) is left as null
    # so it doesn't bias either the PCA or the simultaneous count.
    df = df.with_columns(
        pl.when(pl.col("n_5m") < 5)
          .then(None)
          .otherwise(pl.col("dist_5m") < STATIONARY_DIST_M)
          .alias("stationary")
    )
    return df
""")

md("""
## 3. Per-empresa pass: PCA + simultaneous buses + figure data

We process empresas one at a time to keep peak memory low. For each empresa we
compute the PCA ratio (on moving points only), the simultaneous-fleet stats
(active buses per minute), and we keep aside two small artifacts for the
figures: a 40k scatter sample and the per-hour median of active buses. The
full per-empresa frame is dropped before moving to the next.

**Treatment of immature windows**: points whose stationary flag is `null`
(fewer than 5 points in their 5-minute window — typically the first minutes
of a bus's day) are treated as follows:
- **PCA**: excluded. We only PCA on points we're confident are moving.
- **Simultaneous count**: included. The bus is reporting GPS, so it counts
  toward "active fleet" even if we don't yet know if it's moving.
""")

code("""
def pca_ratio(df: pl.DataFrame, sample_size: int = 200_000) -> tuple[float | None, int]:
    # PCA only on points confidently moving (stationary == False, not null).
    moving = df.filter(pl.col("stationary") == False).select(["lat", "lon"])
    n_moving = moving.height
    if n_moving < 100:
        return None, n_moving
    if n_moving > sample_size:
        moving = moving.sample(n=sample_size, seed=42)
    arr = moving.to_numpy()
    arr = np.column_stack([arr[:, 0] * LAT_DEG_M, arr[:, 1] * LON_DEG_M])
    arr = arr - arr.mean(axis=0)
    cov = np.cov(arr.T)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    return float(eigvals[1] / eigvals[0]), n_moving


def per_minute_active(df: pl.DataFrame) -> pl.DataFrame:
    # Active = not stationary (null included: reporting GPS, indeterminate movement).
    active = df.filter(pl.col("stationary") != True)
    if active.is_empty():
        return pl.DataFrame({"minute": [], "active_buses": []})
    return (
        active.with_columns(pl.col("time").dt.truncate("1m").alias("minute"))
        .group_by("minute")
        .agg(pl.col("unidadid").n_unique().alias("active_buses"))
        .sort("minute")
    )


empresas = sorted(per_empresa["empresaid"].to_list())
linearity_rows = []
simultaneous_rows = []
figure_samples = {}     # empresa -> small pandas DF for scatter
hourly_active = {}      # empresa -> pandas (hour, median_n) for line plot

for e in empresas:
    df_e = load_empresa(e)
    # --- PCA ---
    ratio, n_moving = pca_ratio(df_e)
    linearity_rows.append({
        "empresaid": e, "pca_ratio": ratio,
        "points_total": df_e.height, "points_moving": n_moving,
    })
    # --- Simultaneous-fleet stats ---
    per_min = per_minute_active(df_e)
    if per_min.is_empty():
        simultaneous_rows.append({"empresaid": e, "median_active": 0.0,
                                  "p95_active": 0.0, "max_active": 0.0})
    else:
        simultaneous_rows.append({
            "empresaid": e,
            "median_active": float(per_min["active_buses"].median()),
            "p95_active": float(per_min["active_buses"].quantile(0.95)),
            "max_active": float(per_min["active_buses"].max()),
        })
    # --- Artifacts for figures (small) ---
    sample_n = min(40_000, df_e.height)
    figure_samples[e] = (
        df_e.select(["lat", "lon", "stationary"])
        .sample(n=sample_n, seed=42)
        .to_pandas()
    )
    if not per_min.is_empty():
        hourly_active[e] = (
            per_min.with_columns(pl.col("minute").dt.hour().alias("hour"))
            .group_by("hour")
            .agg(pl.col("active_buses").median().alias("median_n"))
            .sort("hour")
            .to_pandas()
        )
    # Free the per-empresa frame before the next iteration.
    del df_e

linearity_df = pl.DataFrame(linearity_rows).sort("pca_ratio", descending=True, nulls_last=True)
simultaneous = pl.DataFrame(simultaneous_rows).sort("empresaid")
print("Linearity:"); print(linearity_df)
print("\\nSimultaneous fleet:"); print(simultaneous)
""")

md("## 5. Viability table with exclusion reason")

code("""
def exclusion_reason(pass_lin: bool, pass_sim: bool) -> str:
    if pass_lin and pass_sim:
        return ""
    if not pass_lin and not pass_sim:
        return "ratio PCA bajo y flota simultánea insuficiente"
    if not pass_lin:
        return "ratio PCA bajo (ruta no lineal)"
    return "flota simultánea insuficiente"

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
    .with_columns(
        pl.struct(["pass_linearity", "pass_simultaneous"]).map_elements(
            lambda s: exclusion_reason(s["pass_linearity"], s["pass_simultaneous"]),
            return_dtype=pl.String,
        ).alias("motivo_exclusion")
    )
    .sort("empresaid")
)
viability_pd = viability.to_pandas()
viability_pd.to_csv(OUTPUT_DIR / "viability.csv", index=False)
viability
""")

code("""
viable_set = sorted(viability.filter(pl.col("viable"))["empresaid"].to_list())
print("Empresas viables por criterio:", viable_set)
print("Empresas seleccionadas en propuesta:", SELECTED_EMPRESAS)

mismatch = viable_set != SELECTED_EMPRESAS
if mismatch:
    print()
    print("⚠ MISMATCH: el resultado de los criterios NO coincide con la propuesta.")
    print("  No es un error de código. Inspecciona viability.csv y sensitivity.csv")
    print("  para entender qué empresa(s) cambiaron y por qué (filtro de buses")
    print("  parados puede mover los números). Decide si actualizar la propuesta")
    print("  o ajustar los criterios — NO 'forzar' el resultado al valor esperado.")
    print()
    print("  El notebook seguirá ejecutándose para producir todos los diagnósticos.")
    print("  NO subas clean_gps.parquet como dataset oficial hasta resolver esto.")
else:
    print("OK: los criterios reproducen exactamente la selección de la propuesta.")
""")

md("""
## 6. Sensitivity analysis

A reviewer will ask why we chose `PCA ≥ 4` and `simultaneous ≥ 5`. The table
below shows which empresas would pass under nearby threshold combinations.
""")

code("""
sensitivity_rows = []
for lin_thr in [3.0, 4.0, 5.0, 6.0]:
    for sim_thr in [3, 5, 7, 10]:
        passing = (
            viability
            .filter((pl.col("pca_ratio") >= lin_thr) & (pl.col("median_active") >= sim_thr))
            ["empresaid"].to_list()
        )
        sensitivity_rows.append({
            "linearity_thr": lin_thr,
            "simultaneous_thr": sim_thr,
            "n_empresas": len(passing),
            "empresas": ",".join(map(str, sorted(passing))),
        })
sensitivity_df = pl.DataFrame(sensitivity_rows)
sensitivity_df.to_pandas().to_csv(OUTPUT_DIR / "sensitivity.csv", index=False)
sensitivity_df
""")

md("""
## 7. Figures

Two figures used in the paper to defend the selection visually.
""")

code("""
# Figure 1: PCA scatter per empresa, coloring stationary points lighter.
# Uses figure_samples (small per-empresa pandas DFs collected in section 3).
n = len(empresas)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)
for ax, e in zip(axes, empresas):
    sample = figure_samples[e]
    moving = sample[sample["stationary"] == False]
    stopped = sample[sample["stationary"] == True]
    indet = sample[sample["stationary"].isna()]
    ax.scatter(stopped["lon"], stopped["lat"], s=0.5, alpha=0.15, color="lightgray", label="parado")
    ax.scatter(indet["lon"], indet["lat"], s=0.5, alpha=0.15, color="khaki", label="indet.")
    ax.scatter(moving["lon"], moving["lat"], s=0.5, alpha=0.4, color="C0", label="activo")
    row = viability.filter(pl.col("empresaid") == e).to_dicts()[0]
    ratio = row["pca_ratio"]
    med = row["median_active"]
    ok = "OK" if row["viable"] else "X"
    ratio_txt = f"{ratio:.2f}" if ratio is not None else "N/A"
    ax.set_title(f"Empresa {e} [{ok}]\\nPCA={ratio_txt}  med_sim={med:.0f}")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect("equal")
for ax in axes[n:]:
    ax.axis("off")
fig.suptitle("Linealidad por empresa (parado=gris, indeterminado=amarillo)", y=1.0)
fig.tight_layout()
fig.savefig(FIG_DIR / "pca_por_empresa.png", dpi=120, bbox_inches="tight")
plt.show()
""")

code("""
# Figure 2: simultaneous active buses by hour-of-day per empresa.
# Uses hourly_active (small per-empresa pandas DFs collected in section 3).
fig, ax = plt.subplots(figsize=(10, 5))
for e in empresas:
    if e not in hourly_active:
        continue
    per_min = hourly_active[e]
    is_sel = e in SELECTED_EMPRESAS
    ax.plot(per_min["hour"], per_min["median_n"],
            label=f"E{e}", linewidth=2 if is_sel else 1,
            alpha=1.0 if is_sel else 0.4)
ax.axhline(SIMULTANEOUS_THRESHOLD, color="red", linestyle="--", label=f"umbral={SIMULTANEOUS_THRESHOLD}")
ax.set_xlabel("Hora del día")
ax.set_ylabel("Mediana de buses activos simultáneos / minuto")
ax.set_title("Flota activa por hora — empresas seleccionadas en negrita")
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "buses_simultaneos_por_hora.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 8. Filter and save clean dataset

Filter the raw GPS records to the 4 viable empresas and write the result as
Parquet. We use the lazy frame (no stationary filter applied here — the
stationary flag was a per-empresa diagnostic; the clean dataset preserves all
GPS records of selected empresas so downstream notebooks can re-derive their
own filters).
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
(`multibus-headway-forecast-clean`) and consumed by `02_preprocessing`,
which will reconstruct the corridor centerline, project each GPS record onto
it, separate ida/vuelta, and compute the headway time series.
""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT}")
