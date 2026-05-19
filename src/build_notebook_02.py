"""Generate the 02_eda_corredores.ipynb file for Kaggle."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "02_eda_corredores" / "02_eda_corredores.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip()))


md("""
# 02 — Targeted EDA on selected corridors

Goal: understand the quality and the temporal/spatial dynamics of the GPS data
for the 4 selected empresas (2, 4, 58, 59) before computing headways. Detect
quality issues that will impact preprocessing in Phase 2.

**Input**: `multibus-headway-forecast-clean` (Parquet, ~47.68M rows, 4
empresas, 2023-10-01 → 2024-02-29).

**Outputs** (to `/kaggle/working/`):
- `quality_gps.csv` — per-empresa quality summary table.
- `atypical_days.csv` — per-(empresa, day) rows flagged as low-volume or low-fleet.
- `figuras/temporal_distribution.png` — records by hour / weekday / month.
- `figuras/gaps_distribution.png` — distribution of inter-record gaps.
- `figuras/spatial_heatmap.png` — spatial density per empresa.
- `figuras/unit_statistics.png` — activity statistics per unit.
- `figuras/gps_quality.png` — observed speed distribution.
- `figuras/heading_distribution.png` — heading rose plot per empresa.
- `figuras/atypical_days.png` — daily records timeline with atypical days highlighted.

## Conventions

- The composite key `(empresaid, unidadid)` is used for every per-unit
  aggregation. `unidadid` is reused across empresas (34 of 150 ids appear
  in 3+ empresas).
- Observed speed is computed from consecutive coordinates and time deltas,
  not from the reported `velocidad` field. The reported field may carry
  spurious zeros or be uncalibrated.
- Quality thresholds are documented explicitly in the setup cell. Any change
  must remain in the log so reviewers can audit it.
""")

code("""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Locate the clean parquet anywhere under /kaggle/input/ — mount path may
# vary depending on dataset visibility/version.
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
print()
print(f"Using INPUT = {INPUT}")

OUTPUT_DIR = Path("/kaggle/working")
FIG_DIR = OUTPUT_DIR / "figuras"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

EMPRESAS = [2, 4, 58, 59]

# Local meters-per-degree at Arequipa (lat ≈ -16.4°)
LAT_DEG_M = 111_000.0
LON_DEG_M = 111_000.0 * np.cos(np.deg2rad(-16.4))

# GPS-quality thresholds (documented for reviewer audit).
MAX_PLAUSIBLE_SPEED_KMH = 80     # urban-transport ceiling for Arequipa
MAX_PLAUSIBLE_JUMP_M = 500       # >500m at 20s sampling implies >90 km/h
SAMPLING_TARGET_S = 20           # nominal source sampling rate
""")

md("""
## 0. Data quality preflight

`clean_gps.parquet` is "clean" only in the sense that Phase 0 deduplicated on
`(empresaid, unidadid, time)` and filtered to the selected empresas. By design
it preserves all rows, including those with null `time` / `lat` / `lon` or
`(lat, lon) == (0, 0)`, so each downstream notebook re-derives its own
row-level filters. We count those rows here and drop them before every
aggregation below — the counts in this cell are the audit trail of what we
discarded.
""")

code("""
lf_raw = pl.scan_parquet(INPUT)

preflight = lf_raw.select([
    pl.len().alias("rows"),
    pl.col("time").is_null().sum().alias("null_time"),
    pl.col("lat").is_null().sum().alias("null_lat"),
    pl.col("lon").is_null().sum().alias("null_lon"),
    pl.col("empresaid").is_null().sum().alias("null_empresaid"),
    pl.col("unidadid").is_null().sum().alias("null_unidadid"),
    ((pl.col("lat") == 0) | (pl.col("lon") == 0)).sum().alias("zero_coords"),
]).collect(engine="streaming").to_pandas().iloc[0]
print("Pre-filter counts:")
print(preflight.to_string())

lf = lf_raw.filter(
    pl.col("time").is_not_null()
    & pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
    & (pl.col("lat") != 0) & (pl.col("lon") != 0)
)
post_rows = lf.select(pl.len().alias("rows")).collect(engine="streaming")["rows"][0]
print(f"\\nRows after filter: {post_rows:,}  (dropped: {int(preflight['rows']) - post_rows:,})")
""")

md("## 1. Temporal distribution: records by hour, weekday, month")

code("""
# Per (empresa, hour-of-day): how does activity distribute through the day?
by_hour = (
    lf.with_columns(pl.col("time").dt.hour().alias("hour"))
    .group_by(["empresaid", "hour"])
    .agg(pl.len().alias("n"))
    .sort(["empresaid", "hour"])
    .collect(engine="streaming")
    .to_pandas()
)

# polars dt.weekday(): 1=Mon..7=Sun
by_weekday = (
    lf.with_columns(pl.col("time").dt.weekday().alias("weekday"))
    .group_by(["empresaid", "weekday"])
    .agg(pl.len().alias("n"))
    .sort(["empresaid", "weekday"])
    .collect(engine="streaming")
    .to_pandas()
)

by_month = (
    lf.with_columns(pl.col("time").dt.strftime("%Y-%m").alias("month"))
    .group_by(["empresaid", "month"])
    .agg(pl.len().alias("n"))
    .sort(["empresaid", "month"])
    .collect(engine="streaming")
    .to_pandas()
)

print("Total records per empresa:")
print(by_month.groupby("empresaid")["n"].sum().to_string())
print()
print("Months observed:")
print(sorted(by_month["month"].unique()))
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

for e in EMPRESAS:
    sub_h = by_hour[by_hour["empresaid"] == e]
    axes[0].plot(sub_h["hour"], sub_h["n"], marker="o", label=f"E{e}")
    sub_w = by_weekday[by_weekday["empresaid"] == e]
    axes[1].plot(sub_w["weekday"], sub_w["n"], marker="o", label=f"E{e}")
    sub_m = by_month[by_month["empresaid"] == e]
    axes[2].plot(sub_m["month"], sub_m["n"], marker="o", label=f"E{e}")

axes[0].set_xlabel("Hour of day"); axes[0].set_ylabel("Records")
axes[0].set_title("Records by hour of day"); axes[0].legend()
axes[0].set_xticks(range(0, 24, 2))

axes[1].set_xlabel("Weekday"); axes[1].set_ylabel("Records")
axes[1].set_title("Records by weekday"); axes[1].legend()
axes[1].set_xticks(range(1, 8)); axes[1].set_xticklabels(weekday_labels)

axes[2].set_xlabel("Month"); axes[2].set_ylabel("Records")
axes[2].set_title("Records by month"); axes[2].legend()
axes[2].tick_params(axis="x", rotation=45)

fig.tight_layout()
fig.savefig(FIG_DIR / "temporal_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 2. Inter-record gaps per unit

Nominal GPS sampling is 20 s. A gap noticeably larger than that means the
unit stopped reporting (GPS blackout, end of trip, communication failure).
We classify the intervals `(t_i − t_{i-1})` per bus into buckets of growing
severity.

This statistic informs the **variable-cardinality strategy** in Phase 2: if
an empresa has many gaps > 5 min, the model will see rapidly fluctuating
fleets.
""")

code("""
gaps = (
    lf.sort(["empresaid", "unidadid", "time"])
    .with_columns(
        (pl.col("time") - pl.col("time").shift(1).over(["empresaid", "unidadid"]))
        .dt.total_seconds().alias("gap_s")
    )
    .filter(pl.col("gap_s").is_not_null() & (pl.col("gap_s") > 0))
    .select(["empresaid", "gap_s"])
    .collect(engine="streaming")
)

gap_summary = (
    gaps.with_columns([
        (pl.col("gap_s") <= 60).alias("le_1min"),
        ((pl.col("gap_s") > 60) & (pl.col("gap_s") <= 300)).alias("1to5min"),
        ((pl.col("gap_s") > 300) & (pl.col("gap_s") <= 1800)).alias("5to30min"),
        ((pl.col("gap_s") > 1800) & (pl.col("gap_s") <= 3600)).alias("30to60min"),
        (pl.col("gap_s") > 3600).alias("gt_60min"),
    ])
    .group_by("empresaid")
    .agg([
        pl.len().alias("n_intervals"),
        pl.col("le_1min").sum(),
        pl.col("1to5min").sum(),
        pl.col("5to30min").sum(),
        pl.col("30to60min").sum(),
        pl.col("gt_60min").sum(),
        pl.col("gap_s").median().alias("gap_median_s"),
        pl.col("gap_s").quantile(0.95).alias("gap_p95_s"),
        pl.col("gap_s").max().alias("gap_max_s"),
    ])
    .sort("empresaid")
)
print(gap_summary)
""")

code("""
# Distribution of gaps (log scale) per empresa, capped at 1h for readability.
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, e in zip(axes, EMPRESAS):
    sub = gaps.filter(pl.col("empresaid") == e)
    if sub.is_empty():
        ax.set_title(f"E{e} (no data)"); continue
    arr = sub["gap_s"].to_numpy()
    arr = arr[arr <= 3600]
    ax.hist(arr, bins=np.logspace(0, np.log10(3600), 60), color="C0", alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(60, color="green", linestyle="--", alpha=0.5, label="1 min")
    ax.axvline(300, color="orange", linestyle="--", alpha=0.5, label="5 min")
    ax.axvline(1800, color="red", linestyle="--", alpha=0.5, label="30 min")
    ax.set_xlabel("Gap (seconds, log)"); ax.set_title(f"Empresa {e}")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Frequency (log)")
fig.suptitle("Inter-record gap distribution per empresa (≤ 1h)")
fig.tight_layout()
fig.savefig(FIG_DIR / "gaps_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 3. Spatial heatmaps per empresa

GPS-point density per empresa, on a sample. Used to:
- Visually confirm the corridor's shape.
- Detect geographic outliers (points far from Arequipa — GPS errors).
- Identify terminals and busier segments.

We use a 200k-point sample per empresa with `seed=42` for reproducibility.
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, e in zip(axes, EMPRESAS):
    df_e = (
        lf.filter(pl.col("empresaid") == e)
        .select(["lat", "lon"])
        .collect(engine="streaming")
    )
    n = df_e.height
    if n > 200_000:
        df_e = df_e.sample(n=200_000, seed=42)
    arr = df_e.to_numpy()
    hb = ax.hexbin(arr[:, 1], arr[:, 0], gridsize=80, cmap="viridis", mincnt=1, bins="log")
    ax.set_title(f"Empresa {e} (n={n:,})")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect("equal")
    plt.colorbar(hb, ax=ax, label="log(count)")
fig.suptitle("Spatial density of GPS records per empresa")
fig.tight_layout()
fig.savefig(FIG_DIR / "spatial_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 4. Per-unit statistics: activity, span, volume

For each bus `(empresaid, unidadid)` we compute:
- `active_days`: distinct days on which the bus reported at least one point.
- `span_days`: days between first and last report (inclusive).
- `activity_ratio = active_days / span_days`: fraction of its lifespan in
  which it was active. Useful to detect buses retired mid-period or new ones.
- `n_records`: total reported volume.
""")

code("""
per_unit = (
    lf.group_by(["empresaid", "unidadid"])
    .agg([
        pl.col("time").min().alias("first_seen"),
        pl.col("time").max().alias("last_seen"),
        pl.col("time").dt.date().n_unique().alias("active_days"),
        pl.len().alias("n_records"),
    ])
    .with_columns(
        ((pl.col("last_seen").dt.date() - pl.col("first_seen").dt.date())
         .dt.total_days() + 1).alias("span_days")
    )
    .with_columns(
        (pl.col("active_days") / pl.col("span_days")).alias("activity_ratio")
    )
    .sort(["empresaid", "unidadid"])
    .collect(engine="streaming")
)

per_unit_summary = (
    per_unit.group_by("empresaid")
    .agg([
        pl.len().alias("n_unidades"),
        pl.col("active_days").median().alias("active_days_median"),
        pl.col("active_days").min().alias("active_days_min"),
        pl.col("active_days").max().alias("active_days_max"),
        pl.col("activity_ratio").median().alias("activity_ratio_median"),
        pl.col("n_records").median().alias("records_median"),
    ])
    .sort("empresaid")
)
print(per_unit_summary)
""")

code("""
# One column per empresa: histogram of active_days (top) and activity_ratio (bottom).
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for col, e in enumerate(EMPRESAS):
    sub = per_unit.filter(pl.col("empresaid") == e)
    if sub.is_empty():
        continue
    ad = sub["active_days"].to_numpy()
    ar = sub["activity_ratio"].to_numpy()
    axes[0, col].hist(ad, bins=30, color="C0", alpha=0.85)
    axes[0, col].set_title(f"Empresa {e}: active days per unit")
    axes[0, col].set_xlabel("Active days"); axes[0, col].set_ylabel("# units")
    axes[1, col].hist(ar, bins=30, color="C1", alpha=0.85)
    axes[1, col].set_title(f"Empresa {e}: activity ratio")
    axes[1, col].set_xlabel("active_days / span_days"); axes[1, col].set_ylabel("# units")
fig.tight_layout()
fig.savefig(FIG_DIR / "unit_statistics.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 5. GPS quality diagnostics

Three checks:
1. **Implausible speeds**: observed speed (`step_m / dt_s`) above the urban
   ceiling (80 km/h). Indicates spurious GPS jumps.
2. **Spatial jumps**: steps with `step_m > 500 m` and `dt_s ≤ 60 s` (would
   imply > 30 km/h but the threshold is geometric — catches teleports).
3. **Residual duplicates**: Phase 0 deduped on `(empresaid, unidadid, time)`.
   We re-check here for safety.

The reported `velocidad` field is compared against the observed speed to
flag inconsistencies (`velocidad == 0` while the bus moved ≥ 50 m).
""")

code("""
quality_pairs = (
    lf.sort(["empresaid", "unidadid", "time"])
    .with_columns([
        pl.col("lat").shift(1).over(["empresaid", "unidadid"]).alias("lat_prev"),
        pl.col("lon").shift(1).over(["empresaid", "unidadid"]).alias("lon_prev"),
        pl.col("time").shift(1).over(["empresaid", "unidadid"]).alias("time_prev"),
    ])
    .with_columns([
        (((pl.col("lat") - pl.col("lat_prev")) * LAT_DEG_M) ** 2
         + ((pl.col("lon") - pl.col("lon_prev")) * LON_DEG_M) ** 2).sqrt().alias("step_m"),
        (pl.col("time") - pl.col("time_prev")).dt.total_seconds().alias("dt_s"),
    ])
    .filter(pl.col("dt_s").is_not_null() & (pl.col("dt_s") > 0))
    .with_columns(
        (pl.col("step_m") / pl.col("dt_s") * 3.6).alias("speed_obs_kmh")
    )
    .select(["empresaid", "step_m", "dt_s", "speed_obs_kmh", "velocidad"])
)

quality_summary = (
    quality_pairs.group_by("empresaid")
    .agg([
        pl.len().alias("n_pairs"),
        (pl.col("speed_obs_kmh") > MAX_PLAUSIBLE_SPEED_KMH).sum().alias("n_overspeed"),
        ((pl.col("step_m") > MAX_PLAUSIBLE_JUMP_M) & (pl.col("dt_s") <= 60))
            .sum().alias("n_jumps"),
        ((pl.col("velocidad") == 0) & (pl.col("step_m") >= 50)).sum().alias("n_vel0_but_moved"),
        pl.col("speed_obs_kmh").median().alias("speed_median_kmh"),
        pl.col("speed_obs_kmh").quantile(0.95).alias("speed_p95_kmh"),
    ])
    .with_columns([
        (pl.col("n_overspeed") / pl.col("n_pairs") * 100).alias("pct_overspeed"),
        (pl.col("n_jumps") / pl.col("n_pairs") * 100).alias("pct_jumps"),
        (pl.col("n_vel0_but_moved") / pl.col("n_pairs") * 100).alias("pct_vel0_inconsistent"),
    ])
    .sort("empresaid")
    .collect(engine="streaming")
)
print(quality_summary)
""")

code("""
# Residual duplicate check (expected: 0 after Phase 0 dedup).
dup_residual = (
    lf.group_by(["empresaid", "unidadid", "time"])
    .agg(pl.len().alias("n"))
    .filter(pl.col("n") > 1)
    .group_by("empresaid")
    .agg(pl.len().alias("n_residual_dups"))
    .sort("empresaid")
    .collect(engine="streaming")
)
print("Residual duplicates per empresa (expected: 0 for all):")
print(dup_residual)
""")

code("""
# Observed-speed distribution per empresa, capped at 120 km/h for plotting.
fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
for ax, e in zip(axes, EMPRESAS):
    sample = (
        quality_pairs.filter(pl.col("empresaid") == e)
        .select("speed_obs_kmh")
        .collect(engine="streaming")
    )
    if sample.is_empty():
        ax.set_title(f"E{e} (no data)"); continue
    arr = sample["speed_obs_kmh"].to_numpy()
    arr = arr[(arr >= 0) & (arr <= 120)]
    ax.hist(arr, bins=60, color="C2", alpha=0.85)
    ax.axvline(MAX_PLAUSIBLE_SPEED_KMH, color="red", linestyle="--",
               label=f"ceiling {MAX_PLAUSIBLE_SPEED_KMH} km/h")
    ax.set_xlabel("Observed speed (km/h)"); ax.set_title(f"Empresa {e}")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Frequency")
fig.suptitle("Observed speed distribution between consecutive records")
fig.tight_layout()
fig.savefig(FIG_DIR / "gps_quality.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 6. Heading analysis: can we distinguish inbound/outbound?

The `direccion` field reports bus heading in degrees (0–360). For a linear
corridor we expect a **bimodal** distribution with two peaks ~180° apart.
A flat distribution would mean heading is unreliable for direction labeling
and Phase 2 would need to fall back on the derivative of the linear
projection `s`.

A large share of `direccion == 0` likely indicates a sentinel value (heading
undefined while parked), not literal "north".
""")

code("""
heading_summary = (
    lf.group_by("empresaid")
    .agg([
        pl.len().alias("n"),
        (pl.col("direccion") == 0).sum().alias("n_zero"),
        pl.col("direccion").is_null().sum().alias("n_null"),
        pl.col("direccion").min().alias("min"),
        pl.col("direccion").max().alias("max"),
    ])
    .with_columns(
        (pl.col("n_zero") / pl.col("n") * 100).alias("pct_zero")
    )
    .sort("empresaid")
    .collect(engine="streaming")
)
print(heading_summary)
""")

code("""
# Rose plot per empresa (heading distribution on a polar axis). Excludes 0°
# which we suspect is a sentinel.
fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection="polar"))
n_bins = 36   # 10° per bin
bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
for ax, e in zip(axes, EMPRESAS):
    sample = (
        lf.filter(pl.col("empresaid") == e)
        .filter(pl.col("direccion") != 0)
        .select("direccion")
        .collect(engine="streaming")
    )
    if sample.is_empty():
        ax.set_title(f"E{e} (no data)"); continue
    n_total = sample.height
    n_plot = min(500_000, n_total)
    sub = sample.sample(n=n_plot, seed=42) if n_total > n_plot else sample
    radians = np.deg2rad(sub["direccion"].to_numpy())
    counts, _ = np.histogram(radians, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 2 * np.pi / n_bins
    ax.bar(bin_centers, counts, width=width, bottom=0.0, color="C3", alpha=0.85, edgecolor="white")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title(f"Empresa {e} (n={n_total:,}, excl. 0°)", pad=15)
fig.suptitle("Heading distribution per empresa (rose plot, bin = 10°)")
fig.tight_layout()
fig.savefig(FIG_DIR / "heading_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## 7. Per-empresa quality summary table

Consolidates the key metrics into a single CSV: `quality_gps.csv`. This is
the verifiable artifact of Phase 1 per the development plan, and the basis
for the **cleaning decisions** to be applied in Phase 2.
""")

code("""
# Pull everything computed above into one row per empresa.
import pandas as pd

gap_pd = gap_summary.to_pandas().set_index("empresaid")
quality_pd = quality_summary.to_pandas().set_index("empresaid")
heading_pd = heading_summary.to_pandas().set_index("empresaid")
per_unit_pd = per_unit_summary.to_pandas().set_index("empresaid")
dup_pd = dup_residual.to_pandas().set_index("empresaid") if dup_residual.height > 0 else None

rows = []
for e in EMPRESAS:
    row = {
        "empresaid": e,
        "n_unidades": int(per_unit_pd.loc[e, "n_unidades"]) if e in per_unit_pd.index else None,
        "active_days_median": per_unit_pd.loc[e, "active_days_median"] if e in per_unit_pd.index else None,
        "activity_ratio_median": round(float(per_unit_pd.loc[e, "activity_ratio_median"]), 3) if e in per_unit_pd.index else None,
        "gap_median_s": float(gap_pd.loc[e, "gap_median_s"]) if e in gap_pd.index else None,
        "gap_p95_s": float(gap_pd.loc[e, "gap_p95_s"]) if e in gap_pd.index else None,
        "n_gaps_gt_5min": int(gap_pd.loc[e, "5to30min"] + gap_pd.loc[e, "30to60min"] + gap_pd.loc[e, "gt_60min"]) if e in gap_pd.index else None,
        "speed_median_kmh": round(float(quality_pd.loc[e, "speed_median_kmh"]), 2) if e in quality_pd.index else None,
        "speed_p95_kmh": round(float(quality_pd.loc[e, "speed_p95_kmh"]), 2) if e in quality_pd.index else None,
        "pct_overspeed": round(float(quality_pd.loc[e, "pct_overspeed"]), 3) if e in quality_pd.index else None,
        "pct_jumps": round(float(quality_pd.loc[e, "pct_jumps"]), 3) if e in quality_pd.index else None,
        "pct_vel0_inconsistent": round(float(quality_pd.loc[e, "pct_vel0_inconsistent"]), 3) if e in quality_pd.index else None,
        "pct_heading_zero": round(float(heading_pd.loc[e, "pct_zero"]), 3) if e in heading_pd.index else None,
        "residual_dups": int(dup_pd.loc[e, "n_residual_dups"]) if dup_pd is not None and e in dup_pd.index else 0,
    }
    rows.append(row)

quality = pd.DataFrame(rows)
quality.to_csv(OUTPUT_DIR / "quality_gps.csv", index=False)
print(quality.to_string(index=False))
""")

md("""
## 8. Atypical days detection

A day is "atypical" for an empresa when its operational footprint differs
substantially from the typical day for that empresa. We flag two kinds:

- **Low-volume days**: total records < 50% of the median daily volume for
  that empresa. Suggests partial service (strike, holiday, system outage).
- **Low-fleet days**: active units < 50% of the median active units for that
  empresa. Stronger signal of operational disruption than low records alone.

Note: only days with at least one record are flagged. Total blackout days
(no records at all) are not in `by_day` and appear as missing dates in the
timeline plot. Phase 3 must materialize the full calendar to count them.

This list feeds:
- Phase 3 (split design): atypical days must be tagged so train/val/test
  do not concentrate disruptions on one side.
- Phase 7 (robustness analysis): models are evaluated on atypical days
  separately.
""")

code("""
# Daily aggregates per empresa.
by_day = (
    lf.with_columns(pl.col("time").dt.date().alias("day"))
    .group_by(["empresaid", "day"])
    .agg([
        pl.len().alias("records"),
        pl.col("unidadid").n_unique().alias("active_units"),
    ])
    .sort(["empresaid", "day"])
    .collect(engine="streaming")
)

# Median baselines per empresa.
baselines = (
    by_day.group_by("empresaid")
    .agg([
        pl.col("records").median().alias("records_median"),
        pl.col("active_units").median().alias("units_median"),
    ])
    .sort("empresaid")
)
print("Per-empresa daily baselines (median):")
print(baselines)

# Flag atypical days: < 50% of either baseline.
atypical = (
    by_day.join(baselines, on="empresaid")
    .with_columns([
        (pl.col("records") < 0.5 * pl.col("records_median")).alias("low_records"),
        (pl.col("active_units") < 0.5 * pl.col("units_median")).alias("low_fleet"),
    ])
    .filter(pl.col("low_records") | pl.col("low_fleet"))
    .sort(["empresaid", "day"])
)

print()
print("Atypical days per empresa (count):")
print(
    atypical.group_by("empresaid")
    .agg([
        pl.col("low_records").sum().alias("n_low_records"),
        pl.col("low_fleet").sum().alias("n_low_fleet"),
        pl.len().alias("n_total_flagged"),
    ])
    .sort("empresaid")
)

atypical.write_csv(OUTPUT_DIR / "atypical_days.csv")
print(f"\\nSaved {atypical.height} flagged (empresa, day) rows to atypical_days.csv")
""")

code("""
# Daily-records timeline per empresa, with atypical days highlighted.
import pandas as pd

by_day_pd = by_day.to_pandas()
atypical_pd = atypical.to_pandas()
by_day_pd["day"] = pd.to_datetime(by_day_pd["day"])
if not atypical_pd.empty:
    atypical_pd["day"] = pd.to_datetime(atypical_pd["day"])

fig, axes = plt.subplots(2, 2, figsize=(18, 8))
for ax, e in zip(axes.flat, EMPRESAS):
    sub = by_day_pd[by_day_pd["empresaid"] == e]
    if sub.empty:
        ax.set_title(f"E{e} (no data)"); continue
    ax.plot(sub["day"], sub["records"], color="C0", lw=1, label="records/day")
    flagged = atypical_pd[atypical_pd["empresaid"] == e]
    if not flagged.empty:
        ax.scatter(flagged["day"], flagged["records"], color="red", s=40,
                   zorder=5, label="atypical")
    median = sub["records"].median()
    ax.axhline(median, color="green", linestyle="--", alpha=0.5,
               label=f"median={int(median):,}")
    ax.axhline(0.5 * median, color="orange", linestyle=":", alpha=0.5,
               label="50% median")
    ax.set_title(f"Empresa {e}: daily records timeline")
    ax.set_ylabel("Records"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Daily records per empresa with atypical days flagged")
fig.tight_layout()
fig.savefig(FIG_DIR / "atypical_days.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
## Next steps for Phase 2

Concrete cleaning decisions (what to do with speeds > 80 km/h, with
`direccion == 0`, with gaps > 30 min, with residual duplicates if any) are
decided from this table and documented in `docs/decisiones-limpieza-fase2.md`
before starting preprocessing.

The `atypical_days.csv` list is consumed by Phase 3 (temporal split design)
and Phase 7 (robustness analysis), not by Phase 2 cleaning.

**Phase 1 closes** when `quality_gps.csv` and `atypical_days.csv` are
produced and the cleaning decisions are approved by advisors. Only then we
move on to Phase 2 (corridor centerline reconstruction and headway
computation).
""")


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT}")
