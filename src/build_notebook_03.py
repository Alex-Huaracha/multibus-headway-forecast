"""Generate the 03_headway_viability.ipynb file for Kaggle.

Viability probe to decide which headway formulation is feasible for Phase 2+
before committing the pipeline. Three candidates are compared on the SAME
preprocessed GPS data:

  A) Virtual passage points       — Δt at fixed locations along the route.
                                    Nodes = points, vector size = N_points.
  B) Spatial snapshot in meters   — Δs between consecutive buses at time T.
                                    Nodes = buses, vector size = N(T) - 1.
  C) Temporal snapshot per bus    — Δt between consecutive buses at time T.
                                    Nodes = buses, vector size = N(T) - 1.
                                    This is the formulation that propuesta.md
                                    §3.2 / §5.2 already describes.
       C.1 forward projection: Δt = (s_i - s_{i+1}) / v_{i+1}(T)
       C.2 trailing crossing:  Δt = T - t_cross(bus_{i+1}, s_i)

Decision matrix produced from 7 measurable dimensions × {A, B, C.1, C.2}
× 2 representative empresas × 3 days.

This notebook does NOT commit to any formulation; it produces the evidence
needed to choose. The decision is written to docs/decisiones-headway-fase2.md
AFTER reading viability_matrix.csv.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "03_headway_viability" / "03_headway_viability.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip()))


# ============================================================================
# Stage 0 — Setup
# ============================================================================

md("""
# 03 — Headway viability probe

**Goal**: empirically determine which headway formulation is feasible for
Phase 2+ BEFORE committing to a pipeline architecture. Decision is data-driven,
not by intuition.

**Three formulations under test, computed on the same preprocessed GPS data**:

| ID | Definition | Vector size | Graph nodes (Phase 6) |
|----|------------|-------------|------------------------|
| A   | Δt at N fixed virtual points along the route | `N_points` | points |
| B   | Δs (meters) between consecutive buses at time T | `N(T) − 1` | buses (dynamic) |
| C.1 | Δt (min) at time T via forward projection: `(s_i − s_{i+1}) / v_{i+1}` | `N(T) − 1` | buses (dynamic) |
| C.2 | Δt (min) at time T via trailing crossing: `T − t_cross(bus_{i+1}, s_i)` | `N(T) − 1` | buses (dynamic) |

Formulation **C** (sub-options C.1 and C.2) is what `docs/propuesta.md` §3.2
and §5.2 already describe. The probe verifies it is actually viable on the
real data.

**Scope**:
- Empresas: **2** (rich, PCA=33.55, median 16 buses) and **59** (hard case,
  no `direccion` column, median 20 buses).
- Days: **2024-01-23** (typical Tuesday), **2024-01-27** (Saturday),
  **2023-10-28** (systemic atypical day — Procesión del Señor de los Milagros).
- 3 days × 2 empresas × 4 formulations = 24 artifacts.

**Outputs to `/kaggle/working/`**:
- `viability_matrix.csv` — go/no-go table over 7 dimensions × 4 formulations × 2 empresas.
- `headways_<ID>_<empresa>_<day>.parquet` — 24 files for traceability.
- `figuras/signal_distributions.png` — histograms per formulation.
- `figuras/autocorrelation.png` — temporal ACF per formulation.
- `figuras/spatial_mi_heatmap.png` — neighbor mutual information.
- `figuras/stability_kl.png` — sensitivity to parameter choice.
- `viability_log.txt` — wall-time and memory per stage.

**Conventions inherited from notebook 02**:
- Composite key `(empresaid, unidadid)` for all per-unit aggregations.
- Observed speed `step_m / dt_s`, NOT the reported `velocidad` field.
- Direction inferred from sign of `ds/dt` (smoothed), NOT from `direccion`
  (per propuesta §1 finding 6 — empresas 58 and 59 do not report it).
""")

code("""
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, datetime
import os
import time
import psutil

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy

# Locate the clean parquet anywhere under /kaggle/input/.
print("Tree under /kaggle/input/:")
for root, dirs, files in os.walk("/kaggle/input"):
    rel = root.replace("/kaggle/input", "") or "/"
    for f in files:
        print(f"  {rel}/{f}")
    if not files and not dirs:
        print(f"  {rel}/ (empty)")

candidates = list(Path("/kaggle/input").rglob("clean_gps.parquet"))
assert candidates, "clean_gps.parquet not found anywhere under /kaggle/input/"
INPUT = candidates[0]
print(f"\\nUsing INPUT = {INPUT}")

OUTPUT_DIR = Path("/kaggle/working")
FIG_DIR = OUTPUT_DIR / "figuras"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# === Probe scope ===
EMPRESAS_PROBE = [2, 59]
TARGET_DATES = [date(2024, 1, 23), date(2024, 1, 27), date(2023, 10, 28)]

# === Formulation parameters ===
N_POINTS_A = 20            # virtual passage points along the route (Opción A)
GRID_SECONDS = 60          # snapshot resampling grid (Opción B and C)

# === Stability variants (Stage 4) ===
N_POINTS_VARIANTS = [10, 20, 40]
GRID_VARIANTS_S = [30, 60, 120]

# === Centerline construction ===
CENTERLINE_N_BINS = 50          # number of bins along principal axis
CENTERLINE_TRIM_PCT = 0.025     # trim 2.5% from each end (noisy edges)
CENTERLINE_SMOOTH_WIN = 5       # rolling-mean window in bins
MIN_SPEED_FOR_CENTERLINE_KMH = 10.0  # use only moving points (filters terminals & traffic)
CENTERLINE_LATLON_QUANTILE = (0.005, 0.995)  # IQR-style geographic outlier filter
LATERAL_OFFSET_THRESHOLD_M = 300.0   # drop pings projected >300m off the centerline

# === Quality thresholds (inherited from notebook 02) ===
MAX_PLAUSIBLE_SPEED_KMH = 80
LAT_DEG_M = 111_000.0
LON_DEG_M = 111_000.0 * np.cos(np.deg2rad(-16.4))

# === Direction inference ===
DIRECTION_SMOOTH_WIN = 5   # rolling window for sign(ds/dt)

# === Reproducibility ===
RNG_SEED = 42

# === Logging ===
log_lines = []
def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    log_lines.append(line)

def mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

log(f"Setup complete. RSS = {mem_mb():.0f} MB")
""")

# ============================================================================
# Stage 0 — Load + filter
# ============================================================================

md("""
## 0. Load and filter to probe scope

We work on a small subset (~3M rows) — materialise once, reuse across all
stages. Avoids the lazy re-read smell observed in notebook 02.
""")

code("""
t0 = time.perf_counter()

lf_raw = pl.scan_parquet(INPUT)

# Filter: probe empresas + probe dates + valid rows (drop nulls and zero coords).
lf = (
    lf_raw
    .filter(pl.col("empresaid").is_in(EMPRESAS_PROBE))
    .filter(pl.col("time").dt.date().is_in(TARGET_DATES))
    .filter(
        pl.col("time").is_not_null()
        & pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
        & (pl.col("lat") != 0) & (pl.col("lon") != 0)
    )
    .with_columns(pl.col("time").dt.date().alias("day"))
    .sort(["empresaid", "unidadid", "time"])
)

gps = lf.collect(engine="streaming")
log(f"Loaded {gps.height:,} rows in {time.perf_counter()-t0:.1f}s. RSS = {mem_mb():.0f} MB")

# Per (empresa, day): row count + unique units.
audit = (
    gps.group_by(["empresaid", "day"])
    .agg([pl.len().alias("rows"), pl.col("unidadid").n_unique().alias("units")])
    .sort(["empresaid", "day"])
)
print(audit)
""")

# ============================================================================
# Stage 1 — Shared preprocessing
# ============================================================================

md("""
## 1. Shared preprocessing — centerline, arc-length s, direction, speed

All three formulations need the **same** four products:
1. **Centerline polyline** per empresa — ordered (lat, lon) vertices.
2. **Arc-length s** per ping — projection onto the centerline.
3. **Direction** per ping — sign of smoothed `ds/dt` (+1 ida, −1 vuelta).
4. **Observed speed** per ping — `step_m / dt_s`.

If this stage is wrong, all four formulations downstream are wrong. We
verify with diagnostic plots.

**Centerline construction** (PCA + binned median + smoothing):
- Sample moving points (speed > 10 km/h) — excludes parked clusters at
  terminals AND quasi-stopped pings in traffic / red lights that bias PCA.
- **Filter geographic outliers** (lat/lon p0.5–p99.5 quantile box) before
  PCA — removes depots, GPS jumps, alternate-route trips that pulled the
  principal axis off-corridor in v1. This was the BUG that broke E2 in v1.
- Project to principal axis via PCA.
- Bin along principal axis (50 bins); take median secondary coord per bin.
- Trim 2.5% noise from each end; smooth with a 5-bin rolling mean.
- Back-transform to (lat, lon). Result: ordered polyline.

**Off-route filtering**: after projection, drop pings with lateral offset
> 300 m. These are buses driving on parallel streets or returning to depot
— they don't belong in the corridor model for Phase 2.

**Arc-length projection** is chunked (10k points per chunk) to keep memory
bounded when computing point-to-segment distances against a 50-vertex
polyline.
""")

code("""
# --- Step 1a: observed-speed column (needed before centerline filter) ---
t0 = time.perf_counter()

gps = gps.with_columns([
    pl.col("lat").shift(1).over(["empresaid", "unidadid"]).alias("lat_prev"),
    pl.col("lon").shift(1).over(["empresaid", "unidadid"]).alias("lon_prev"),
    pl.col("time").shift(1).over(["empresaid", "unidadid"]).alias("time_prev"),
])
gps = gps.with_columns([
    (((pl.col("lat") - pl.col("lat_prev")) * LAT_DEG_M) ** 2
     + ((pl.col("lon") - pl.col("lon_prev")) * LON_DEG_M) ** 2).sqrt().alias("step_m"),
    (pl.col("time") - pl.col("time_prev")).dt.total_seconds().alias("dt_s"),
])
gps = gps.with_columns(
    pl.when(pl.col("dt_s").is_not_null() & (pl.col("dt_s") > 0))
      .then(pl.col("step_m") / pl.col("dt_s") * 3.6)
      .otherwise(None)
      .alias("speed_kmh")
)
# Cap implausible speeds (GPS jumps) so they don't poison aggregates.
gps = gps.with_columns(
    pl.when(pl.col("speed_kmh") > MAX_PLAUSIBLE_SPEED_KMH)
      .then(None)
      .otherwise(pl.col("speed_kmh"))
      .alias("speed_kmh")
)
log(f"Speed computed in {time.perf_counter()-t0:.1f}s. RSS = {mem_mb():.0f} MB")
""")

code("""
# --- Step 1b: build centerline per empresa ---

def filter_geographic_outliers(points_latlon, q=CENTERLINE_LATLON_QUANTILE):
    \"\"\"Trim pings outside [q_lo, q_hi] quantile box of lat and lon.
    Removes geographic outliers (depots, GPS jumps, alternate routes) that
    pull the PCA principal axis off-corridor. Returns filtered (n_kept, 2) array.\"\"\"
    pts = np.asarray(points_latlon, dtype=float)
    lat_lo, lat_hi = np.quantile(pts[:, 0], q)
    lon_lo, lon_hi = np.quantile(pts[:, 1], q)
    mask = (
        (pts[:, 0] >= lat_lo) & (pts[:, 0] <= lat_hi)
        & (pts[:, 1] >= lon_lo) & (pts[:, 1] <= lon_hi)
    )
    return pts[mask]


def build_centerline(points_latlon, n_bins=CENTERLINE_N_BINS,
                      trim_pct=CENTERLINE_TRIM_PCT,
                      smooth_win=CENTERLINE_SMOOTH_WIN):
    \"\"\"PCA + binned median centerline. Returns ordered polyline (m, 2) in (lat, lon).
    Applies geographic outlier filter BEFORE PCA to avoid contaminated principal axis.\"\"\"
    pts = filter_geographic_outliers(points_latlon)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA via eigen-decomp of 2x2 covariance.
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    projected = centered @ eigvecs            # (n, 2)
    t1 = projected[:, 0]
    t2 = projected[:, 1]

    # Trim extreme percentiles of t1 (noisy at ends).
    lo, hi = np.quantile(t1, [trim_pct, 1 - trim_pct])
    mask = (t1 >= lo) & (t1 <= hi)
    t1, t2 = t1[mask], t2[mask]

    # Bin along principal axis; take median secondary coord per bin.
    bins = np.linspace(t1.min(), t1.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(t1, bins) - 1, 0, n_bins - 1)

    cl_proj = []
    for i in range(n_bins):
        m = (bin_idx == i)
        if m.sum() < 5:
            continue
        cl_proj.append([0.5 * (bins[i] + bins[i + 1]), np.median(t2[m])])
    cl_proj = np.array(cl_proj)

    # Smooth secondary coord with rolling mean.
    if smooth_win > 1 and len(cl_proj) >= smooth_win:
        kernel = np.ones(smooth_win) / smooth_win
        cl_proj[:, 1] = np.convolve(cl_proj[:, 1], kernel, mode="same")

    # Back to (lat, lon).
    cl_latlon = cl_proj @ eigvecs.T + centroid
    return cl_latlon


# Build one centerline per empresa using only moving pings.
# Note: HDBSCAN-based cluster filtering was tested in probe v4 and discarded —
# it over-restricts E59 (splits one corridor into halves by GPS density) and
# the geographic outliers in E2 are sparse noise, not a second corridor. The
# IQR-style quantile filter + post-projection lateral cutoff handle both.
centerlines = {}
rng = np.random.default_rng(RNG_SEED)
for e in EMPRESAS_PROBE:
    moving = gps.filter(
        (pl.col("empresaid") == e)
        & (pl.col("speed_kmh") >= MIN_SPEED_FOR_CENTERLINE_KMH)
    ).select(["lat", "lon"])
    if moving.height > 50_000:
        idx = rng.choice(moving.height, size=50_000, replace=False)
        sample = moving.to_numpy()[idx]
    else:
        sample = moving.to_numpy()
    n_pre = len(sample)
    cl = build_centerline(sample)
    centerlines[e] = cl
    n_filtered = n_pre - len(filter_geographic_outliers(sample))
    log(f"Centerline E{e}: {len(cl)} vertices from {moving.height:,} moving pings "
        f"({n_filtered:,} geographic outliers filtered, {100*n_filtered/n_pre:.1f}%)")
""")

code("""
# --- Step 1c: visualise centerlines to verify ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, e in zip(axes, EMPRESAS_PROBE):
    sample = (
        gps.filter(pl.col("empresaid") == e)
        .select(["lat", "lon"])
        .sample(n=min(100_000, gps.filter(pl.col("empresaid") == e).height), seed=RNG_SEED)
        .to_numpy()
    )
    ax.scatter(sample[:, 1], sample[:, 0], s=0.5, alpha=0.2, color="C0", label="GPS pings")
    cl = centerlines[e]
    ax.plot(cl[:, 1], cl[:, 0], color="red", lw=2, label="centerline")
    ax.scatter(cl[0, 1], cl[0, 0], color="green", s=80, marker="^", label="start", zorder=10)
    ax.scatter(cl[-1, 1], cl[-1, 0], color="orange", s=80, marker="v", label="end", zorder=10)
    ax.set_title(f"Empresa {e}: centerline ({len(cl)} vertices)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect("equal")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "centerlines.png", dpi=120, bbox_inches="tight")
plt.show()
""")

code("""
# --- Step 1d: project pings onto centerline → arc-length s ---

def project_to_centerline(points_latlon, centerline_latlon,
                          lat_m=LAT_DEG_M, lon_m=LON_DEG_M,
                          chunk_size=10_000):
    \"\"\"Per point: find closest segment of polyline, project, return cumulative
    arc-length s (in meters) and lateral offset (in meters).
    Chunked to keep peak memory bounded.\"\"\"
    pts = np.asarray(points_latlon, dtype=float)
    cl = np.asarray(centerline_latlon, dtype=float)

    # Convert to meters (local flat-Earth at Arequipa latitude).
    pts_m = np.stack([pts[:, 0] * lat_m, pts[:, 1] * lon_m], axis=1)
    cl_m = np.stack([cl[:, 0] * lat_m, cl[:, 1] * lon_m], axis=1)

    seg_starts = cl_m[:-1]                              # (m-1, 2)
    seg_vecs = np.diff(cl_m, axis=0)                    # (m-1, 2)
    seg_norms_sq = (seg_vecs ** 2).sum(axis=1)          # (m-1,)
    seg_lengths = np.sqrt(seg_norms_sq)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])  # (m,)

    n = pts_m.shape[0]
    s_out = np.zeros(n, dtype=np.float32)
    lateral_out = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = pts_m[start:end]                        # (c, 2)
        diff = chunk[:, None, :] - seg_starts[None, :, :]   # (c, m-1, 2)
        t = (diff * seg_vecs[None, :, :]).sum(axis=2) / seg_norms_sq[None, :]
        t = np.clip(t, 0.0, 1.0)                        # (c, m-1)
        proj = seg_starts[None, :, :] + t[:, :, None] * seg_vecs[None, :, :]
        dist_sq = ((chunk[:, None, :] - proj) ** 2).sum(axis=2)  # (c, m-1)
        best_seg = dist_sq.argmin(axis=1)               # (c,)
        best_t = np.take_along_axis(t, best_seg[:, None], axis=1).squeeze(1)
        s_out[start:end] = cum_s[best_seg] + best_t * seg_lengths[best_seg]
        lateral_out[start:end] = np.sqrt(
            np.take_along_axis(dist_sq, best_seg[:, None], axis=1).squeeze(1)
        )
    return s_out, lateral_out


t0 = time.perf_counter()
s_all = np.zeros(gps.height, dtype=np.float32)
lat_off_all = np.zeros(gps.height, dtype=np.float32)
empresa_arr = gps["empresaid"].to_numpy()
latlon_arr = gps.select(["lat", "lon"]).to_numpy()
for e in EMPRESAS_PROBE:
    mask = empresa_arr == e
    s_e, lat_e = project_to_centerline(latlon_arr[mask], centerlines[e])
    s_all[mask] = s_e
    lat_off_all[mask] = lat_e
    log(f"Projected E{e}: {mask.sum():,} pings, max s = {s_e.max():,.0f} m, p95 lateral = {np.percentile(lat_e, 95):.0f} m")
gps = gps.with_columns([
    pl.Series("s", s_all, dtype=pl.Float32),
    pl.Series("lateral_m", lat_off_all, dtype=pl.Float32),
])
log(f"Total projection time {time.perf_counter()-t0:.1f}s. RSS = {mem_mb():.0f} MB")

# Off-route filter: drop pings projected too far from the centerline.
# These are buses on parallel streets or returning to depot — not part of the
# corridor model for Phase 2. v2 fix to address the E2 contamination from v1.
n_pre = gps.height
off_route_summary = (
    gps.group_by("empresaid")
    .agg([
        pl.len().alias("n_total"),
        (pl.col("lateral_m") > LATERAL_OFFSET_THRESHOLD_M).sum().alias("n_off_route"),
        pl.col("lateral_m").quantile(0.5).alias("lat_p50"),
        pl.col("lateral_m").quantile(0.95).alias("lat_p95"),
        pl.col("lateral_m").max().alias("lat_max"),
    ])
    .with_columns(
        (pl.col("n_off_route") / pl.col("n_total") * 100).alias("pct_off_route")
    )
    .sort("empresaid")
)
print(off_route_summary)
gps = gps.filter(pl.col("lateral_m") <= LATERAL_OFFSET_THRESHOLD_M)
log(f"On-route filter: {n_pre:,} → {gps.height:,} rows "
    f"({100*(1 - gps.height/n_pre):.1f}% dropped as off-route)")
""")

code("""
# --- Step 1e: direction inference via signed ds/dt ---
# Smoothed difference of s over rolling window of DIRECTION_SMOOTH_WIN pings.
t0 = time.perf_counter()
gps = gps.with_columns([
    (pl.col("s") - pl.col("s").shift(1).over(["empresaid", "unidadid"])).alias("ds_raw"),
])
gps = gps.with_columns([
    pl.col("ds_raw").rolling_mean(window_size=DIRECTION_SMOOTH_WIN, min_samples=1)
       .over(["empresaid", "unidadid"]).alias("ds_smooth"),
])
gps = gps.with_columns([
    pl.when(pl.col("ds_smooth") > 0).then(1)
      .when(pl.col("ds_smooth") < 0).then(-1)
      .otherwise(0)
      .cast(pl.Int8).alias("direction"),
])
log(f"Direction inferred in {time.perf_counter()-t0:.1f}s")

# Sanity: distribution of direction per empresa.
dir_summary = (
    gps.group_by("empresaid")
    .agg([
        (pl.col("direction") == 1).sum().alias("n_ida"),
        (pl.col("direction") == -1).sum().alias("n_vuelta"),
        (pl.col("direction") == 0).sum().alias("n_zero"),
    ])
    .with_columns([
        (pl.col("n_ida") / (pl.col("n_ida") + pl.col("n_vuelta") + pl.col("n_zero")) * 100).alias("pct_ida"),
        (pl.col("n_vuelta") / (pl.col("n_ida") + pl.col("n_vuelta") + pl.col("n_zero")) * 100).alias("pct_vuelta"),
    ])
    .sort("empresaid")
)
print(dir_summary)
""")

# ============================================================================
# Stage 2 — Compute the four headway artifacts
# ============================================================================

md("""
## 2. Compute headways — formulations A, B, C.1, C.2

Each function takes the preprocessed `gps` frame plus parameters and returns
a long-format `pl.DataFrame`. The four artifacts share the same indexing
philosophy: explicit `(empresa, day)` columns, `direction` where applicable.

### Opción A — virtual passage points
For each of `N_points` points along the centerline, detect when each bus
crosses by interpolating `(t, s)` around `s = point`. The headway at that
point is the time between consecutive crossings (in the same direction).

### Opción B — spatial snapshot in meters
Resample to a uniform time grid (`GRID_SECONDS`). At each timestamp T, sort
active buses by `s` within the same `direction` and compute `Δs` between
consecutive pairs.

### Opción C — temporal snapshot per bus
Same resampling and ordering as B. For each consecutive pair (front, back):
- **C.1**: `Δt = (s_front − s_back) / v_back(T)` — forward projection.
- **C.2**: `Δt = T − interp(t, traj_back, s_front)` — trailing crossing
  (search past trajectory of `bus_back` for when it crossed `s_front`).
""")

code("""
# === Opción A — virtual passage points ===

def compute_headways_A(gps_df, n_points=N_POINTS_A):
    \"\"\"Long-format headways at fixed points.
    Returns columns: (empresaid, day, point_id, direction, bus_id,
                      t_cross, delta_t_prev_s).\"\"\"
    out_rows = []
    for (e, day), sub_eday in gps_df.group_by(["empresaid", "day"], maintain_order=True):
        s_max_e = float(sub_eday["s"].max())
        points = np.linspace(0.05 * s_max_e, 0.95 * s_max_e, n_points)
        for (bus,), sub in sub_eday.group_by(["unidadid"], maintain_order=True):
            s_arr = sub["s"].to_numpy()
            t_arr = sub["time"].to_numpy()
            d_arr = sub["direction"].to_numpy()
            if len(s_arr) < 2:
                continue
            # For each point, find indices i where sign(s[i] - point) != sign(s[i+1] - point).
            for pid, P in enumerate(points):
                diff = s_arr - P
                signs = np.sign(diff)
                cross_mask = (signs[:-1] * signs[1:]) < 0
                if not cross_mask.any():
                    continue
                idxs = np.where(cross_mask)[0]
                for i in idxs:
                    if s_arr[i + 1] == s_arr[i]:
                        continue
                    frac = (P - s_arr[i]) / (s_arr[i + 1] - s_arr[i])
                    # Interpolate time of crossing.
                    dt_ns = (t_arr[i + 1] - t_arr[i]).astype("timedelta64[ns]").astype(np.int64)
                    t_cross_ns = t_arr[i].astype("datetime64[ns]").astype(np.int64) + int(frac * dt_ns)
                    dir_cross = int(d_arr[i]) if d_arr[i] != 0 else int(d_arr[i + 1])
                    out_rows.append((int(e), day, int(pid), dir_cross, int(bus),
                                     t_cross_ns, float(P)))

    if not out_rows:
        return pl.DataFrame(schema={
            "empresaid": pl.Int64, "day": pl.Date, "point_id": pl.Int64,
            "direction": pl.Int64, "unidadid": pl.Int64,
            "t_cross": pl.Datetime("ns"), "s_point": pl.Float64,
            "delta_t_prev_s": pl.Float64,
        })

    df = pl.DataFrame(out_rows, schema=[
        "empresaid", "day", "point_id", "direction", "unidadid",
        "t_cross_ns", "s_point",
    ], orient="row")
    df = df.with_columns(pl.col("t_cross_ns").cast(pl.Datetime("ns")).alias("t_cross"))
    # Sort by (empresa, day, point, direction, t_cross) and compute delta_t to previous crossing.
    df = df.sort(["empresaid", "day", "point_id", "direction", "t_cross"])
    df = df.with_columns(
        (pl.col("t_cross") - pl.col("t_cross").shift(1)
            .over(["empresaid", "day", "point_id", "direction"]))
            .dt.total_seconds().alias("delta_t_prev_s")
    )
    return df.drop("t_cross_ns").select([
        "empresaid", "day", "point_id", "direction", "unidadid",
        "t_cross", "s_point", "delta_t_prev_s",
    ])


t0 = time.perf_counter()
heads_A = compute_headways_A(gps)
log(f"Opción A: {heads_A.height:,} crossings in {time.perf_counter()-t0:.1f}s")
print(heads_A.head())
""")

code("""
# === Build snapshot table — common helper for B, C.1, C.2 ===

def build_snapshots(gps_df, grid_s=GRID_SECONDS):
    \"\"\"Resample each bus to a uniform time grid and return
    (empresaid, day, t_grid, unidadid, direction, s, v_kmh).
    Linear interpolation in s; nearest direction; rolling-mean speed.\"\"\"
    snaps_per_eday = []
    for (e, day), sub_eday in gps_df.group_by(["empresaid", "day"], maintain_order=True):
        # Snapshot grid: floor(min) to ceil(max) at grid_s seconds.
        t_min = sub_eday["time"].min()
        t_max = sub_eday["time"].max()
        # Round t_min down and t_max up to grid_s.
        t_min_s = int(t_min.timestamp())
        t_max_s = int(t_max.timestamp())
        t_grid_s = np.arange(
            (t_min_s // grid_s) * grid_s,
            ((t_max_s // grid_s) + 1) * grid_s + 1,
            grid_s,
        )
        t_grid = np.array(t_grid_s, dtype="datetime64[s]").astype("datetime64[ns]")
        n_grid = len(t_grid)

        for (bus,), sub in sub_eday.group_by(["unidadid"], maintain_order=True):
            t_arr = sub["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
            s_arr = sub["s"].to_numpy().astype(np.float64)
            v_arr = sub["speed_kmh"].to_numpy().astype(np.float64)
            d_arr = sub["direction"].to_numpy().astype(np.int64)
            if len(t_arr) < 2:
                continue
            t_grid_ns = t_grid.astype(np.int64)
            # Only interpolate within the bus's reported window.
            in_window = (t_grid_ns >= t_arr[0]) & (t_grid_ns <= t_arr[-1])
            if not in_window.any():
                continue
            tg = t_grid_ns[in_window]
            s_interp = np.interp(tg, t_arr, s_arr)
            # Speed: nan-aware linear interp.
            v_clean = np.where(np.isnan(v_arr), 0.0, v_arr)
            v_interp = np.interp(tg, t_arr, v_clean)
            # Direction: nearest valid (non-zero) by left-search.
            # Use right=False so we get the latest known direction at tg.
            idx_left = np.searchsorted(t_arr, tg, side="right") - 1
            idx_left = np.clip(idx_left, 0, len(d_arr) - 1)
            d_interp = d_arr[idx_left]
            snaps_per_eday.append(pl.DataFrame({
                "empresaid": np.full(len(tg), int(e), dtype=np.int64),
                "day": [day] * len(tg),
                "t_grid": tg.astype("datetime64[ns]"),
                "unidadid": np.full(len(tg), int(bus), dtype=np.int64),
                "s": s_interp.astype(np.float32),
                "speed_kmh": v_interp.astype(np.float32),
                "direction": d_interp.astype(np.int8),
            }))
    if not snaps_per_eday:
        return pl.DataFrame()
    snaps = pl.concat(snaps_per_eday)
    return snaps.with_columns(pl.col("t_grid").cast(pl.Datetime("ns")))


t0 = time.perf_counter()
snaps = build_snapshots(gps, grid_s=GRID_SECONDS)
log(f"Snapshots: {snaps.height:,} rows in {time.perf_counter()-t0:.1f}s. RSS = {mem_mb():.0f} MB")
""")

code("""
# === Opción B — spatial snapshot (Δs in meters) ===

def compute_headways_B(snaps_df):
    \"\"\"For each (empresa, day, t_grid, direction): sort buses by s,
    return Δs between consecutive pairs.\"\"\"
    # Only keep snapshots with a determinate direction.
    s = snaps_df.filter(pl.col("direction") != 0)
    s = s.sort(["empresaid", "day", "t_grid", "direction", "s"])
    s = s.with_columns([
        pl.col("s").shift(1).over(["empresaid", "day", "t_grid", "direction"]).alias("s_back"),
        pl.col("unidadid").shift(1).over(["empresaid", "day", "t_grid", "direction"]).alias("bus_back"),
        pl.cum_count("s").over(["empresaid", "day", "t_grid", "direction"]).alias("rank"),
    ])
    s = s.filter(pl.col("s_back").is_not_null()).with_columns(
        (pl.col("s") - pl.col("s_back")).alias("delta_s_m")
    )
    return s.select([
        "empresaid", "day", "t_grid", "direction",
        pl.col("rank").alias("pair_rank"),
        pl.col("unidadid").alias("bus_front"),
        "bus_back",
        pl.col("s").alias("s_front"),
        "s_back",
        "delta_s_m",
    ])


t0 = time.perf_counter()
heads_B = compute_headways_B(snaps)
log(f"Opción B: {heads_B.height:,} pairs in {time.perf_counter()-t0:.1f}s")
print(heads_B.head())
""")

code("""
# === Opción C.1 — forward projection: Δt = (s_front - s_back) / v_back ===

def compute_headways_C1(snaps_df):
    \"\"\"Δt in minutes via forward projection. Same pair structure as B but
    divides by v_back(T). Replaces v == 0 with NaN to flag undetermined cases.\"\"\"
    s = snaps_df.filter(pl.col("direction") != 0)
    s = s.sort(["empresaid", "day", "t_grid", "direction", "s"])
    s = s.with_columns([
        pl.col("s").shift(1).over(["empresaid", "day", "t_grid", "direction"]).alias("s_back"),
        pl.col("unidadid").shift(1).over(["empresaid", "day", "t_grid", "direction"]).alias("bus_back"),
        pl.col("speed_kmh").shift(1).over(["empresaid", "day", "t_grid", "direction"]).alias("v_back_kmh"),
        pl.cum_count("s").over(["empresaid", "day", "t_grid", "direction"]).alias("rank"),
    ])
    s = s.filter(pl.col("s_back").is_not_null())
    # v in m/s; Δt = (s_front - s_back) / v_back
    s = s.with_columns([
        (pl.col("v_back_kmh") / 3.6).alias("v_back_ms"),
    ])
    s = s.with_columns(
        pl.when(pl.col("v_back_ms") > 0.5)   # below 0.5 m/s ≈ stopped → undefined
          .then((pl.col("s") - pl.col("s_back")) / pl.col("v_back_ms") / 60.0)
          .otherwise(None)
          .alias("delta_t_min")
    )
    return s.select([
        "empresaid", "day", "t_grid", "direction",
        pl.col("rank").alias("pair_rank"),
        pl.col("unidadid").alias("bus_front"),
        "bus_back",
        pl.col("s").alias("s_front"),
        "s_back",
        "v_back_ms",
        "delta_t_min",
    ])


t0 = time.perf_counter()
heads_C1 = compute_headways_C1(snaps)
log(f"Opción C.1: {heads_C1.height:,} pairs in {time.perf_counter()-t0:.1f}s")
print(heads_C1.head())
""")

code("""
# === Opción C.2 — trailing crossing ===
# For each pair (bus_front at s_front, bus_back) at time T: search the past
# trajectory of bus_back for when it last crossed s_front (in the SAME
# direction). Δt = T - t_cross.

def compute_headways_C2(snaps_df, gps_df):
    \"\"\"Δt in minutes via trailing crossing. Requires the full per-bus
    trajectory (gps_df), not just the snapshot.\"\"\"
    pairs = compute_headways_B(snaps_df)   # reuse pair structure

    # Build per-(empresa, unidad, direction) sorted (s, t) trajectory once.
    # Then for each pair, look up the (bus_back, direction) trajectory and
    # interpolate/search for t when bus_back was at s_front.
    traj_index = {}
    for (e, bus, dirc), sub in gps_df.filter(pl.col("direction") != 0).group_by(
        ["empresaid", "unidadid", "direction"], maintain_order=True
    ):
        s_arr = sub["s"].to_numpy().astype(np.float64)
        t_arr = sub["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        # We need monotonic-in-time for searching by time and monotonic-in-s
        # within a single trip. Buses do many trips so s is NOT monotonic globally.
        # Approach: for trailing crossing, scan t backwards from T and find the
        # most recent index i where (s[i] - s_front) and (s[i+1] - s_front) have
        # opposite signs. Store (t_arr, s_arr) sorted by t.
        order = np.argsort(t_arr)
        traj_index[(int(e), int(bus), int(dirc))] = (t_arr[order], s_arr[order])

    pairs_pd = pairs.to_pandas()
    n = len(pairs_pd)
    delta_t_min = np.full(n, np.nan, dtype=np.float64)

    t_grid_ns_all = pairs_pd["t_grid"].astype("datetime64[ns]").astype("int64").to_numpy()
    s_front_all = pairs_pd["s_front"].to_numpy(dtype=np.float64)
    e_all = pairs_pd["empresaid"].to_numpy(dtype=np.int64)
    bus_back_all = pairs_pd["bus_back"].to_numpy(dtype=np.int64)
    dir_all = pairs_pd["direction"].to_numpy(dtype=np.int64)

    for k in range(n):
        key = (int(e_all[k]), int(bus_back_all[k]), int(dir_all[k]))
        if key not in traj_index:
            continue
        t_arr, s_arr = traj_index[key]
        T = t_grid_ns_all[k]
        # Restrict to past pings (t <= T).
        cutoff = np.searchsorted(t_arr, T, side="right")
        if cutoff < 2:
            continue
        s_past = s_arr[:cutoff]
        t_past = t_arr[:cutoff]
        sf = s_front_all[k]
        diff = s_past - sf
        signs = np.sign(diff)
        cross_mask = (signs[:-1] * signs[1:]) < 0
        if not cross_mask.any():
            continue
        # Most recent crossing.
        i = np.where(cross_mask)[0][-1]
        if s_past[i + 1] == s_past[i]:
            continue
        frac = (sf - s_past[i]) / (s_past[i + 1] - s_past[i])
        t_cross = t_past[i] + frac * (t_past[i + 1] - t_past[i])
        delta_t_min[k] = (T - t_cross) / 1e9 / 60.0   # ns → s → min

    out = pairs.with_columns(pl.Series("delta_t_min", delta_t_min, dtype=pl.Float64))
    return out.select([
        "empresaid", "day", "t_grid", "direction", "pair_rank",
        "bus_front", "bus_back", "s_front", "s_back", "delta_s_m", "delta_t_min",
    ])


t0 = time.perf_counter()
heads_C2 = compute_headways_C2(snaps, gps)
log(f"Opción C.2: {heads_C2.height:,} pairs ({(~heads_C2['delta_t_min'].is_null()).sum():,} valid) "
    f"in {time.perf_counter()-t0:.1f}s")
print(heads_C2.head())
""")

code("""
# === Persist all four artifacts to /kaggle/working ===
ARTIFACTS = {"A": heads_A, "B": heads_B, "C1": heads_C1, "C2": heads_C2}

for fid, df in ARTIFACTS.items():
    for (e, day), sub in df.group_by(["empresaid", "day"], maintain_order=True):
        path = OUTPUT_DIR / f"headways_{fid}_E{int(e)}_{day}.parquet"
        sub.write_parquet(path)
log(f"Wrote {len(ARTIFACTS) * len(EMPRESAS_PROBE) * len(TARGET_DATES)} artifact files")
""")

# ============================================================================
# Stage 3 — Diagnostics
# ============================================================================

md("""
## 3. Diagnostics — 7 dimensions per formulation

We evaluate each formulation on:

| # | Dim | Metric | Pass |
|---|-----|--------|------|
| 1 | Computability | fraction of valid headway records | ≥ 80% |
| 2 | Signal richness | coefficient of variation `std / mean` | ≥ 0.2 |
| 3 | Temporal predictability | autocorrelation at 5-min lag | ≥ 0.3 |
| 4 | Spatial structure | mutual information between neighbours | ≥ 0.1 bits |
| 5 | Persistence baseline | R² of `ŷ(T+5min) = y(T)` | between 0.5 and 0.85 |
| 6 | Sample volume | # (X, y) pairs at 5-min horizon (per empresa, scaled to 151 days) | ≥ 50k |
| 7 | Compute cost | wall-time per (empresa, day) | ≤ 10 min |

Neighbour semantics:
- **A**: adjacent points on the route, paired by matching 5-min time bucket.
- **B / C.1 / C.2**: adjacent bus pairs in the same snapshot (consecutive `pair_rank`).

MI uses the KSG estimator (`mutual_info_regression`, k=5). Bootstrap with
5 resamples to report `mi ± std`.
""")

code("""
# === Diagnostic functions ===

def _value_col(df, fid):
    return {"A": "delta_t_prev_s", "B": "delta_s_m",
            "C1": "delta_t_min", "C2": "delta_t_min"}[fid]


def diag_computability(df, fid):
    col = _value_col(df, fid)
    n_total = df.height
    if n_total == 0:
        return 0.0
    n_valid = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & (pl.col(col) > 0)).height
    return n_valid / n_total


def diag_cv(df, fid):
    col = _value_col(df, fid)
    arr = df[col].drop_nulls().to_numpy()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < 10:
        return np.nan
    m = arr.mean()
    if m == 0:
        return np.nan
    return float(arr.std() / m)


def diag_autocorr_5min(df, fid):
    \"\"\"For A: ACF over consecutive crossings within (point, direction).
    For B/C: ACF over consecutive snapshots at fixed (pair_rank, direction).\"\"\"
    col = _value_col(df, fid)
    sub = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & (pl.col(col) > 0))
    if sub.height < 100:
        return np.nan
    if fid == "A":
        sub = sub.sort(["empresaid", "day", "point_id", "direction", "t_cross"])
        groups = ["empresaid", "day", "point_id", "direction"]
    else:
        # 5-min lag at 60s grid = lag 5.
        sub = sub.sort(["empresaid", "day", "direction", "pair_rank", "t_grid"])
        groups = ["empresaid", "day", "direction", "pair_rank"]
    sub = sub.with_columns(
        pl.col(col).shift(5).over(groups).alias("lag5")
    ).filter(pl.col("lag5").is_not_null() & pl.col("lag5").is_finite() & (pl.col("lag5") > 0))
    if sub.height < 100:
        return np.nan
    x = sub[col].to_numpy()
    y = sub["lag5"].to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


def diag_mi_neighbours(df, fid, n_bootstrap=5):
    \"\"\"Mutual information between Δ_i and Δ_{i+1} (neighbour pair).
    A: pair (Δ at point_i, Δ at point_{i+1}) within same 5-min bucket and direction.
    B/C: pair (Δ at pair_rank, Δ at pair_rank+1) within same snapshot.\"\"\"
    col = _value_col(df, fid)
    sub = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & (pl.col(col) > 0))
    if fid == "A":
        sub = sub.with_columns(
            (pl.col("t_cross").dt.truncate("5m")).alias("bucket")
        ).group_by(["empresaid", "day", "direction", "bucket", "point_id"]).agg(
            pl.col(col).mean().alias("v")
        ).sort(["empresaid", "day", "direction", "bucket", "point_id"])
        sub = sub.with_columns([
            pl.col("v").shift(-1).over(["empresaid", "day", "direction", "bucket"]).alias("v_next"),
            pl.col("point_id").shift(-1).over(["empresaid", "day", "direction", "bucket"]).alias("pid_next"),
        ]).filter(pl.col("v_next").is_not_null() & (pl.col("pid_next") == pl.col("point_id") + 1))
        pairs = sub.select(["v", "v_next"]).to_numpy()
    else:
        sub = sub.sort(["empresaid", "day", "t_grid", "direction", "pair_rank"])
        sub = sub.with_columns([
            pl.col(col).shift(-1).over(["empresaid", "day", "t_grid", "direction"]).alias("v_next"),
            pl.col("pair_rank").shift(-1).over(["empresaid", "day", "t_grid", "direction"]).alias("rank_next"),
        ]).filter(pl.col("v_next").is_not_null() & (pl.col("rank_next") == pl.col("pair_rank") + 1))
        pairs = sub.select([col, "v_next"]).to_numpy()

    if len(pairs) < 100:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(pairs), size=int(0.8 * len(pairs)), replace=False)
        sample = pairs[idx]
        mi_nats = mutual_info_regression(
            sample[:, 0].reshape(-1, 1), sample[:, 1],
            n_neighbors=5, random_state=int(rng.integers(0, 2**31 - 1)),
        )[0]
        estimates.append(mi_nats / np.log(2))   # nats → bits
    return float(np.mean(estimates)), float(np.std(estimates))


def diag_persistence_r2(df, fid):
    \"\"\"R² of ŷ(t+5) = y(t). Positive R² means persistence is some baseline;
    we want the value to be in [0.5, 0.85] — not trivial, but predictable.\"\"\"
    col = _value_col(df, fid)
    sub = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & (pl.col(col) > 0))
    if sub.height < 100:
        return np.nan
    if fid == "A":
        sub = sub.sort(["empresaid", "day", "point_id", "direction", "t_cross"])
        groups = ["empresaid", "day", "point_id", "direction"]
    else:
        sub = sub.sort(["empresaid", "day", "direction", "pair_rank", "t_grid"])
        groups = ["empresaid", "day", "direction", "pair_rank"]
    sub = sub.with_columns(pl.col(col).shift(5).over(groups).alias("pred"))
    sub = sub.filter(pl.col("pred").is_not_null() & pl.col("pred").is_finite() & (pl.col("pred") > 0))
    if sub.height < 100:
        return np.nan
    y = sub[col].to_numpy()
    yhat = sub["pred"].to_numpy()
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def diag_sample_count(df, fid):
    \"\"\"# (X, y) pairs available for ML, scaled from 3 probe days to 151 days.\"\"\"
    col = _value_col(df, fid)
    sub = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & (pl.col(col) > 0))
    if fid == "A":
        groups = ["empresaid", "day", "point_id", "direction"]
    else:
        groups = ["empresaid", "day", "direction", "pair_rank"]
    # n_pairs_per_series = max(0, length - 5) summed over groups.
    counts = sub.group_by(groups).len()
    n_pairs_3d = int(np.maximum(counts["len"].to_numpy() - 5, 0).sum())
    return int(n_pairs_3d * 151 / 3)
""")

code("""
# === Run all diagnostics ===
t0 = time.perf_counter()
diag_rows = []
for fid in ["A", "B", "C1", "C2"]:
    df_full = ARTIFACTS[fid]
    for e in EMPRESAS_PROBE:
        df = df_full.filter(pl.col("empresaid") == e)
        mi_mean, mi_std = diag_mi_neighbours(df, fid)
        row = {
            "formulation": fid,
            "empresa": e,
            "n_records": df.height,
            "pct_valid": round(diag_computability(df, fid) * 100, 1),
            "cv": round(diag_cv(df, fid), 3),
            "autocorr_5min": round(diag_autocorr_5min(df, fid), 3),
            "mi_bits": round(mi_mean, 3) if not np.isnan(mi_mean) else np.nan,
            "mi_std": round(mi_std, 3) if not np.isnan(mi_std) else np.nan,
            "r2_persistence": round(diag_persistence_r2(df, fid), 3),
            "n_pairs_151d": diag_sample_count(df, fid),
        }
        diag_rows.append(row)

diag_df = pd.DataFrame(diag_rows)
print(diag_df.to_string(index=False))
log(f"Diagnostics computed in {time.perf_counter()-t0:.1f}s")
""")

code("""
# === Go/no-go evaluation per cell ===
THRESHOLDS = {
    "pct_valid": (80.0, "≥"),
    "cv": (0.2, "≥"),
    "autocorr_5min": (0.3, "≥"),
    "mi_bits": (0.1, "≥"),
    "r2_persistence": ((0.5, 0.85), "between"),
    "n_pairs_151d": (50_000, "≥"),
}

def verdict(metric, value):
    if pd.isna(value):
        return "?"
    rule = THRESHOLDS[metric]
    if rule[1] == "≥":
        return "✓" if value >= rule[0] else "✗"
    if rule[1] == "between":
        lo, hi = rule[0]
        return "✓" if (lo <= value <= hi) else "✗"
    return "?"

verdict_df = diag_df.copy()
for m in THRESHOLDS:
    verdict_df[f"v_{m}"] = diag_df[m].apply(lambda v, m=m: verdict(m, v))
verdict_df["pass_count"] = verdict_df[[f"v_{m}" for m in THRESHOLDS]].apply(
    lambda r: (r == "✓").sum(), axis=1
)
print(verdict_df[["formulation", "empresa", "pass_count"] + [f"v_{m}" for m in THRESHOLDS]]
      .to_string(index=False))
""")

# ============================================================================
# Stage 4 — Stability
# ============================================================================

md("""
## 4. Stability — sensitivity to parametrisation

A formulation that flips its distribution shape when we change a knob is
fragile. We re-compute with alternative parameters and measure the
Kullback-Leibler divergence between distributions of headway values.

- **A**: re-run with `N_points ∈ {10, 20, 40}`.
- **B / C.1 / C.2**: re-run with `grid ∈ {30, 60, 120}` seconds.

Pass criterion: max pairwise KL divergence < 0.1.
""")

code("""
# === Stability for A ===
def hist_density(arr, bins):
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < 50:
        return None
    h, _ = np.histogram(arr, bins=bins, density=True)
    h = h + 1e-9
    h = h / h.sum()
    return h

t0 = time.perf_counter()
stab_rows = []

# A: vary N_points
bins_A = np.linspace(0, 1800, 50)   # 0–30 min in seconds
hists_A = {}
for n_pts in N_POINTS_VARIANTS:
    if n_pts == N_POINTS_A:
        h_df = heads_A
    else:
        h_df = compute_headways_A(gps, n_points=n_pts)
    arr = h_df["delta_t_prev_s"].drop_nulls().to_numpy()
    h = hist_density(arr, bins_A)
    hists_A[n_pts] = h

for i, k1 in enumerate(N_POINTS_VARIANTS):
    for k2 in N_POINTS_VARIANTS[i + 1:]:
        if hists_A[k1] is None or hists_A[k2] is None:
            kl = np.nan
        else:
            kl = float(entropy(hists_A[k1], hists_A[k2]))
        stab_rows.append({"formulation": "A", "var_a": k1, "var_b": k2, "kl": round(kl, 4)})

# B / C.1 / C.2: vary grid_s
bins_B = np.linspace(0, 5000, 50)   # 0–5 km
bins_C = np.linspace(0, 30, 50)     # 0–30 min

snaps_variants = {}
for g in GRID_VARIANTS_S:
    if g == GRID_SECONDS:
        snaps_variants[g] = snaps
    else:
        snaps_variants[g] = build_snapshots(gps, grid_s=g)

heads_B_var = {g: compute_headways_B(snaps_variants[g]) for g in GRID_VARIANTS_S}
heads_C1_var = {g: compute_headways_C1(snaps_variants[g]) for g in GRID_VARIANTS_S}
heads_C2_var = {g: compute_headways_C2(snaps_variants[g], gps) for g in GRID_VARIANTS_S}

for fid, var_dict, bins, col in [
    ("B", heads_B_var, bins_B, "delta_s_m"),
    ("C1", heads_C1_var, bins_C, "delta_t_min"),
    ("C2", heads_C2_var, bins_C, "delta_t_min"),
]:
    hists = {g: hist_density(var_dict[g][col].drop_nulls().to_numpy(), bins) for g in GRID_VARIANTS_S}
    for i, k1 in enumerate(GRID_VARIANTS_S):
        for k2 in GRID_VARIANTS_S[i + 1:]:
            if hists[k1] is None or hists[k2] is None:
                kl = np.nan
            else:
                kl = float(entropy(hists[k1], hists[k2]))
            stab_rows.append({"formulation": fid, "var_a": k1, "var_b": k2, "kl": round(kl, 4)})

stab_df = pd.DataFrame(stab_rows)
print(stab_df.to_string(index=False))

# Max KL per formulation.
stab_max = stab_df.groupby("formulation")["kl"].max().reset_index().rename(columns={"kl": "kl_max"})
stab_max["stable"] = stab_max["kl_max"] < 0.1
print()
print(stab_max.to_string(index=False))
log(f"Stability computed in {time.perf_counter()-t0:.1f}s")
""")

# ============================================================================
# Stage 5 — Decision matrix + plots
# ============================================================================

md("""
## 5. Decision — viability matrix and plots

Final outputs:
- `viability_matrix.csv` — verdict per dimension per formulation per empresa,
  including stability and the per-formulation pass count.
- Four figures comparing distributions, autocorrelation, neighbour MI, and
  stability.
- An executive printout that names the recommended formulation (or escalates
  to Option D if none pass enough dimensions).
""")

code("""
# === Assemble final viability matrix ===
final = verdict_df.merge(stab_max, left_on="formulation", right_on="formulation", how="left")
final["v_stability"] = final["stable"].map(lambda b: "✓" if b else "✗" if pd.notna(b) else "?")
final["pass_count_total"] = final["pass_count"] + (final["v_stability"] == "✓").astype(int)

col_order = ["formulation", "empresa", "n_records",
             "pct_valid", "v_pct_valid",
             "cv", "v_cv",
             "autocorr_5min", "v_autocorr_5min",
             "mi_bits", "mi_std", "v_mi_bits",
             "r2_persistence", "v_r2_persistence",
             "n_pairs_151d", "v_n_pairs_151d",
             "kl_max", "v_stability",
             "pass_count_total"]
final = final[col_order]
final.to_csv(OUTPUT_DIR / "viability_matrix.csv", index=False)
print(final.to_string(index=False))
""")

code("""
# === Figure 1: signal distributions per formulation (across both empresas) ===
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
specs = [
    ("A",  heads_A,  "delta_t_prev_s", "Δt (s) — point crossings", (0, 1800)),
    ("B",  heads_B,  "delta_s_m",      "Δs (m) — spatial snapshot", (0, 5000)),
    ("C1", heads_C1, "delta_t_min",    "Δt (min) — forward proj.",  (0, 30)),
    ("C2", heads_C2, "delta_t_min",    "Δt (min) — trailing cross.", (0, 30)),
]
for ax, (fid, df, col, label, xlim) in zip(axes, specs):
    arr = df[col].drop_nulls().to_numpy()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        ax.set_title(f"{fid} (no data)"); continue
    arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
    ax.hist(arr, bins=60, color=f"C{specs.index((fid, df, col, label, xlim))}", alpha=0.85)
    ax.set_xlim(*xlim)
    ax.set_xlabel(label); ax.set_title(f"Opción {fid} (n={len(arr):,})")
axes[0].set_ylabel("Frequency")
fig.suptitle("Signal distributions per formulation (both empresas pooled)")
fig.tight_layout()
fig.savefig(FIG_DIR / "signal_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
""")

code("""
# === Figure 2: autocorrelation as a function of lag (per formulation, E2) ===
def acf_series(df, fid, max_lag=20):
    col = _value_col(df, fid)
    sub = df.filter(pl.col("empresaid") == 2)
    sub = sub.filter(pl.col(col).is_not_null() & pl.col(col).is_finite() & (pl.col(col) > 0))
    if fid == "A":
        sub = sub.sort(["day", "point_id", "direction", "t_cross"])
        groups = ["day", "point_id", "direction"]
    else:
        sub = sub.sort(["day", "direction", "pair_rank", "t_grid"])
        groups = ["day", "direction", "pair_rank"]
    out = []
    arr0 = sub[col].to_numpy()
    for lag in range(1, max_lag + 1):
        lagged = sub.with_columns(pl.col(col).shift(lag).over(groups).alias("lag"))
        lagged = lagged.filter(pl.col("lag").is_not_null() & pl.col("lag").is_finite() & (pl.col("lag") > 0))
        if lagged.height < 50:
            out.append(np.nan); continue
        x = lagged[col].to_numpy()
        y = lagged["lag"].to_numpy()
        out.append(float(np.corrcoef(x, y)[0, 1]))
    return out

fig, ax = plt.subplots(figsize=(10, 5))
for fid, df, color in [("A", heads_A, "C0"), ("B", heads_B, "C1"),
                       ("C1", heads_C1, "C2"), ("C2", heads_C2, "C3")]:
    vals = acf_series(df, fid, max_lag=15)
    ax.plot(range(1, len(vals) + 1), vals, marker="o", label=f"Opción {fid}", color=color)
ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="threshold 0.3")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_xlabel("Lag (units of the series — for B/C: × 1 min)")
ax.set_ylabel("Autocorrelation (Pearson)")
ax.set_title("Autocorrelation per formulation (E2)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "autocorrelation.png", dpi=120, bbox_inches="tight")
plt.show()
""")

code("""
# === Figure 3: MI heatmap (formulation × empresa) ===
mi_pivot = final.pivot(index="formulation", columns="empresa", values="mi_bits")
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(mi_pivot.values, cmap="viridis", aspect="auto")
for i in range(mi_pivot.shape[0]):
    for j in range(mi_pivot.shape[1]):
        v = mi_pivot.values[i, j]
        ax.text(j, i, f"{v:.2f}" if pd.notna(v) else "NaN",
                ha="center", va="center", color="white", fontsize=11)
ax.set_xticks(range(mi_pivot.shape[1])); ax.set_xticklabels([f"E{e}" for e in mi_pivot.columns])
ax.set_yticks(range(mi_pivot.shape[0])); ax.set_yticklabels(mi_pivot.index)
ax.set_title("Mutual information between neighbour headways (bits)")
plt.colorbar(im, ax=ax, label="MI (bits)")
fig.tight_layout()
fig.savefig(FIG_DIR / "spatial_mi_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()
""")

code("""
# === Figure 4: stability KL ===
fig, ax = plt.subplots(figsize=(8, 4))
for fid in stab_df["formulation"].unique():
    sub = stab_df[stab_df["formulation"] == fid]
    labels = [f"{a}↔{b}" for a, b in zip(sub["var_a"], sub["var_b"])]
    ax.bar([f"{fid}-{lbl}" for lbl in labels], sub["kl"].values, label=f"Opción {fid}")
ax.axhline(0.1, color="red", linestyle="--", alpha=0.6, label="threshold 0.1")
ax.set_ylabel("KL divergence")
ax.set_title("Parametrisation stability — pairwise KL between distribution variants")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "stability_kl.png", dpi=120, bbox_inches="tight")
plt.show()
""")

code("""
# === Executive summary ===
def summarize_formulation(fid):
    rows = final[final["formulation"] == fid]
    pass_counts = rows["pass_count_total"].tolist()
    return f"Opción {fid}: pass_count_total = {pass_counts} (per empresa {list(rows['empresa'])})"

print("=" * 70)
print("VIABILITY PROBE — EXECUTIVE SUMMARY")
print("=" * 70)
for fid in ["A", "B", "C1", "C2"]:
    print(summarize_formulation(fid))
print()
print("Decision rule:")
print("  pass_count_total ≥ 6 of 7 dimensions in BOTH empresas → viable.")
print("  pass_count_total ≥ 5 in BOTH empresas → conditional (document risk).")
print("  Otherwise → not viable, escalate to Opción D (semantic checkpoints +")
print("    bus nodes) or reformulate propuesta.")
print()
print("This notebook does NOT commit to a formulation. The decision is to be")
print("recorded in docs/decisiones-headway-fase2.md after inspecting these")
print("outputs and discussing with advisors.")
print("=" * 70)

# Persist log.
(OUTPUT_DIR / "viability_log.txt").write_text("\\n".join(log_lines))
print(f"\\nLog saved to {OUTPUT_DIR / 'viability_log.txt'}")
""")

# ============================================================================
# Footer
# ============================================================================

md("""
## Next steps

After running this notebook in Kaggle and downloading the outputs:

1. Read `viability_matrix.csv` and inspect the four figures.
2. If **C.1 or C.2** pass ≥ 6 of 7 dimensions in both empresas → confirm
   Opción C as the formulation. Lock the sub-option (C.1 or C.2) based on
   `pct_valid` and `pass_count_total`. Write `docs/decisiones-headway-fase2.md`
   with the formal definition and parameters.
3. If **only A or B** pass and C does not → re-read `docs/propuesta.md` §3.2 /
   §5.2 and discuss with advisors whether to rewrite the propuesta to match
   the viable formulation.
4. If **none pass** → escalate to Opción D (semantic checkpoints with bus
   nodes) or replan the project scope.
5. Only after the decision is recorded, scaffold `src/preprocessing/` and
   start producing `cleaned_gps_<empresa>.parquet` and `headways_<empresa>.parquet`
   as the durable Phase 2 artifacts.
""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT} ({len(cells)} cells)")
