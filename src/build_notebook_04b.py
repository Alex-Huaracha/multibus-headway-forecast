"""Generate the 04b_preprocessing_visual.ipynb file for Kaggle.

Visual validation of Fase 2 outputs (`cleaned_gps_E{2,59}.parquet`,
`headways_E{2,59}.parquet`). Reads them from the kernel 04 v2 output as a
kernel_source — no recomputation of the pipeline.

Output: notebooks/04b_preprocessing_visual/04b_preprocessing_visual.ipynb
Kaggle kernel: alexhuaracha/04b-preprocessing-visual
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "04b_preprocessing_visual" / "04b_preprocessing_visual.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)


nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.rstrip()))


# ---------------------------------------------------------------------------
# Header + setup
# ---------------------------------------------------------------------------

md("""# 04b — Validación visual del preprocesamiento

Cierra el item "Validación visual sobre muestras (lado a lado con GPS crudo)"
del `plan-de-desarrollo.md` Fase 2.

Lee los 4 parquets producidos por el kernel `alexhuaracha/04-preprocessing` v2
(referenciado como `kernel_source` en `kernel-metadata.json`) y genera 6
figuras que permiten inspeccionar visualmente la calidad de cleaned_gps y
headways.

Empresas en alcance: 2 y 59 (mismo scope que Fase 2).
""")

code("""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Diagnostic — discover where kernel_sources mounted the parquets
KAGGLE_INPUT = Path("/kaggle/input")
print("=== /kaggle/input contents ===")
for p in sorted(KAGGLE_INPUT.rglob("*.parquet")):
    print(f"  {p}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
print()

# Find the directory containing the cleaned_gps parquets
candidates = list(KAGGLE_INPUT.rglob("cleaned_gps_E2.parquet"))
if not candidates:
    raise FileNotFoundError(
        f"cleaned_gps_E2.parquet not found under /kaggle/input. "
        f"Available: {list(KAGGLE_INPUT.iterdir())}"
    )
INPUT_DIR = candidates[0].parent
print(f"Resolved INPUT_DIR: {INPUT_DIR}")

OUTPUT_DIR = Path("/kaggle/working")
FIGURAS_DIR = OUTPUT_DIR / "figuras"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

EMPRESAS = [2, 59]

# Load all 4 parquets
data = {}
for e in EMPRESAS:
    gps = pl.read_parquet(INPUT_DIR / f"cleaned_gps_E{e}.parquet")
    hw = pl.read_parquet(INPUT_DIR / f"headways_E{e}.parquet")
    data[e] = {"gps": gps, "hw": hw}
    print(f"E{e}: cleaned_gps {gps.height:,} rows | headways {hw.height:,} rows")
""")

# ---------------------------------------------------------------------------
# Figure 1 — centerline + pings overlay
# ---------------------------------------------------------------------------

md("""## Figura 1 — Trazado del corredor (lat/lon) con pings proyectados

Scatter de los pings de un día típico (2024-01-23, martes). Los pings con
`lateral_m` chico marcan visualmente la centerline del corredor.
""")

code("""
SAMPLE_DAY = pl.lit("2024-01-23").str.to_date()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, e in zip(axes, EMPRESAS):
    gps = data[e]["gps"]
    sub = gps.filter(pl.col("t").dt.date() == SAMPLE_DAY)
    # Subsample to 8000 points max for readability
    if sub.height > 8000:
        sub = sub.sample(n=8000, seed=42)
    sc = ax.scatter(
        sub["lon"].to_numpy(),
        sub["lat"].to_numpy(),
        c=sub["lateral_m"].to_numpy(),
        cmap="viridis",
        s=2,
        alpha=0.5,
        vmin=0,
        vmax=300,
    )
    plt.colorbar(sc, ax=ax, label="lateral_m")
    ax.set_title(f"Empresa {e} — pings de 2024-01-23 (sample {sub.height:,})")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "01_corridor_overlay.png", dpi=120, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
# Figure 2 — lateral offset distribution
# ---------------------------------------------------------------------------

md("""## Figura 2 — Distribución de `lateral_m`

Muestra cómo se reparten los pings respecto a la centerline. El threshold
productivo es 300 m (`LATERAL_OFFSET_THRESHOLD_M` en `config.py`); todo lo
que pasa de ahí ya fue descartado por `projection.py`.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, e in zip(axes, EMPRESAS):
    lat_m = data[e]["gps"]["lateral_m"].to_numpy()
    ax.hist(lat_m, bins=60, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(300, color="crimson", linestyle="--", label="threshold 300m")
    ax.set_title(f"Empresa {e} — lateral_m (n={len(lat_m):,})")
    ax.set_xlabel("lateral_m")
    ax.set_ylabel("count")
    ax.legend()
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "02_lateral_m_dist.png", dpi=120, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
# Figure 3 — sample trajectory: t vs s with direction color
# ---------------------------------------------------------------------------

md("""## Figura 3 — Trayectoria muestral (`t` vs `s`)

Un bus elegido de cada empresa en un día típico. Color = `direction` (ida = +1
azul, vuelta = −1 naranja). Los trazos diagonales son viajes; los saltos
verticales son cambios de dirección o gaps.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, e in zip(axes, EMPRESAS):
    gps = data[e]["gps"]
    day_sub = gps.filter(pl.col("t").dt.date() == SAMPLE_DAY)
    # Pick the unidad with most pings on that day
    top_bus = (
        day_sub.group_by("unidadid").len().sort("len", descending=True).head(1)
    )
    if top_bus.height == 0:
        ax.set_title(f"Empresa {e} — sin datos para sample day")
        continue
    bus_id = top_bus["unidadid"][0]
    sub = day_sub.filter(pl.col("unidadid") == bus_id).sort("t")
    t = sub["t"].to_numpy()
    s = sub["s"].to_numpy()
    d = sub["direction"].to_numpy()
    colors = np.where(d > 0, "tab:blue", np.where(d < 0, "tab:orange", "gray"))
    ax.scatter(t, s, c=colors, s=4, alpha=0.8)
    ax.set_title(f"Empresa {e} — bus {bus_id} en 2024-01-23 (n={sub.height})")
    ax.set_xlabel("t")
    ax.set_ylabel("s (m)")
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "03_sample_trajectory.png", dpi=120, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
# Figure 4 — headway timeline for one day
# ---------------------------------------------------------------------------

md("""## Figura 4 — Timeline de `delta_t_min` (un día típico)

Para 2024-01-23, dispersión de los headways computados por C.2 a lo largo del
día. Se ven los picos del horario de mayor demanda (mañana / tarde) y las
colas de bajo tráfico.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, e in zip(axes, EMPRESAS):
    hw = data[e]["hw"]
    day_sub = (
        hw.filter(pl.col("t").dt.date() == SAMPLE_DAY)
        .filter(pl.col("delta_t_min").is_not_null())
    )
    if day_sub.height == 0:
        ax.set_title(f"Empresa {e} — sin headways para sample day")
        continue
    t = day_sub["t"].to_numpy()
    dt = day_sub["delta_t_min"].to_numpy()
    d = day_sub["direction"].to_numpy()
    colors = np.where(d > 0, "tab:blue", "tab:orange")
    ax.scatter(t, dt, c=colors, s=3, alpha=0.4)
    ax.set_title(f"Empresa {e} — delta_t_min en 2024-01-23 (n={day_sub.height:,})")
    ax.set_xlabel("t")
    ax.set_ylabel("delta_t (min)")
    ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "04_headway_timeline.png", dpi=120, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
# Figure 5 — delta_t distribution per empresa+direction (log scale)
# ---------------------------------------------------------------------------

md("""## Figura 5 — Distribución de `delta_t_min` (escala log)

Histograma por empresa y dirección. La cola larga es esperada (Caveat 2 de
`decisiones-headway-fase2.md`); la winsorización p99 se aplicará en Fase 5,
no acá.
""")

code("""
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for col, e in enumerate(EMPRESAS):
    hw = data[e]["hw"].filter(pl.col("delta_t_min").is_not_null())
    for row, direction in enumerate([1, -1]):
        ax = axes[row, col]
        sub = hw.filter(pl.col("direction") == direction)
        if sub.height == 0:
            ax.set_title(f"E{e} dir={direction} — sin datos")
            continue
        vals = sub["delta_t_min"].to_numpy()
        ax.hist(vals, bins=80, range=(0, 120), color="darkcyan",
                edgecolor="black", alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"E{e} direction={direction} — n={sub.height:,}")
        ax.set_xlabel("delta_t_min")
        ax.set_ylabel("count (log)")
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "05_delta_t_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
# Figure 6 — pairs efectivo per day (daily timeline)
# ---------------------------------------------------------------------------

md("""## Figura 6 — `n_pairs_efectivo` por día

Timeline diario del conteo de pares con `delta_t_min` no nulo. Permite
ubicar visualmente los días con baja cobertura (domingos, feriados, eventos
sistémicos) documentados en `eventos-anomalos.md §3` y `§4`.
""")

code("""
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, e in zip(axes, EMPRESAS):
    hw = data[e]["hw"]
    daily = (
        hw.filter(pl.col("delta_t_min").is_not_null())
        .with_columns(pl.col("t").dt.date().alias("day"))
        .group_by("day").len().sort("day")
    )
    days = daily["day"].to_numpy()
    cnt = daily["len"].to_numpy()
    ax.bar(days, cnt, width=1.0, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axhline(10000, color="crimson", linestyle="--",
               label="Caveat 3 threshold (10k)")
    ax.set_title(f"Empresa {e} — pairs efectivo / día (n_days={daily.height})")
    ax.set_ylabel("count")
    ax.legend()
axes[-1].set_xlabel("día")
plt.tight_layout()
plt.savefig(FIGURAS_DIR / "06_pairs_efectivo_per_day.png", dpi=120, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
# Closing summary
# ---------------------------------------------------------------------------

md("""## Cierre

6 figuras guardadas en `/kaggle/working/figuras/`. Resultado esperado:
trazados coherentes con la geografía de Arequipa Cercado, distribuciones
acotadas (lateral_m < 300, delta_t mayormente < 30 min), y timeline diaria
que refleja los patrones de demanda ya conocidos.
""")

code("""
print("Figures generated:")
for f in sorted(FIGURAS_DIR.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
""")

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}
with OUT.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook written: {OUT}  ({len(cells)} cells)")
