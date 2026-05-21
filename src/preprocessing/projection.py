"""Speed attachment and arc-length projection for the preprocessing pipeline.

Provides:
  attach_observed_speed — compute step_m, dt_s, speed_kmh per (empresaid, unidadid)
                          and DROP GPS-jump pairs per spec R11 (pair-level discard,
                          not row-level nulling).
  project_to_centerline — project pings onto a polyline, compute s and lateral_m,
                          drop off-route rows.

Source: derived from build_notebook_03.py lines 246-273 (speed) and
        390-468 (projection + off-route filter).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import (
    LAT_DEG_M,
    LON_DEG_M,
    MAX_PLAUSIBLE_JUMP_M,
    MAX_PLAUSIBLE_SPEED_KMH,
    lateral_threshold_for,
)


def attach_observed_speed(gps: pl.DataFrame) -> pl.DataFrame:
    """Add columns (lat_prev, lon_prev, time_prev, step_m, dt_s, speed_kmh) by
    diffing successive rows of the same (empresaid, unidadid), then discard
    GPS-jump pairs per spec R11.

    speed_kmh is computed as step_m / dt_s * 3.6 (observed speed from GPS
    displacement). The raw `velocidad` field is intentionally NOT used (spec R11,
    decisiones-limpieza-fase2 §2.3).

    Pair-level discard (spec R11) — rows are DROPPED (not nulled) when:
      1. speed_kmh > MAX_PLAUSIBLE_SPEED_KMH (80 km/h): GPS jump or data error.
      2. step_m > MAX_PLAUSIBLE_JUMP_M (500 m) AND dt_s <= 60 s: implausible jump.

    The first ping per bus has no previous ping, so step_m and dt_s are null
    and speed_kmh is null. These rows are KEPT (null speed is not an outlier —
    it is missing data for the leading ping only). The filter conditions
    explicitly preserve null-speed rows.

    Output frame has fewer rows than input when GPS jumps are present.

    Source: build_notebook_03.py lines 250-272 (extended for R11 pair-level discard).
    """
    gps = gps.with_columns([
        pl.col("lat").shift(1).over(["empresaid", "unidadid"]).alias("lat_prev"),
        pl.col("lon").shift(1).over(["empresaid", "unidadid"]).alias("lon_prev"),
        pl.col("time").shift(1).over(["empresaid", "unidadid"]).alias("time_prev"),
    ])
    gps = gps.with_columns([
        (
            ((pl.col("lat") - pl.col("lat_prev")) * LAT_DEG_M) ** 2
            + ((pl.col("lon") - pl.col("lon_prev")) * LON_DEG_M) ** 2
        ).sqrt().alias("step_m"),
        (pl.col("time") - pl.col("time_prev")).dt.total_seconds().alias("dt_s"),
    ])
    gps = gps.with_columns(
        pl.when(pl.col("dt_s").is_not_null() & (pl.col("dt_s") > 0))
          .then(pl.col("step_m") / pl.col("dt_s") * 3.6)
          .otherwise(None)
          .alias("speed_kmh")
    )
    # Pair-level discard criterion 1 (spec R11): drop rows where speed > 80 km/h.
    # Null speed (first ping per bus) is preserved — it is not a GPS-jump outlier.
    gps = gps.filter(
        pl.col("speed_kmh").is_null() | (pl.col("speed_kmh") <= MAX_PLAUSIBLE_SPEED_KMH)
    )
    # Pair-level discard criterion 2 (spec R11): drop rows where step_m > 500 m
    # AND dt_s <= 60 s. This catches teleporting pings that briefly exceed the
    # jump threshold within a 1-minute window.
    # The first ping per bus has step_m = null (no previous ping) — these must
    # be kept. Polars propagates null through comparisons, so we must explicitly
    # preserve null-step_m rows with step_m.is_null() as an OR guard.
    gps = gps.filter(
        pl.col("step_m").is_null()
        | ~(
            (pl.col("step_m") > MAX_PLAUSIBLE_JUMP_M)
            & (pl.col("dt_s") <= 60)
        )
    )
    return gps


def project_to_centerline(
    gps: pl.DataFrame,
    centerline_latlon: np.ndarray,
    empresaid: int,
    chunk_size: int = 10_000,
) -> pl.DataFrame:
    """Project each ping onto the centerline polyline, compute arc-length s and
    lateral_m, then drop pings where lateral_m > lateral_threshold_for(empresaid).

    Args:
        gps: rows belonging to a SINGLE empresa with columns
             (empresaid, unidadid, time, lat, lon, speed_kmh).
        centerline_latlon: (m, 2) array of (lat, lon) from corridor.build_centerline.
        empresaid: used to look up the lateral offset threshold.
        chunk_size: number of pings to process per numpy batch (bounds peak memory).

    Returns:
        pl.DataFrame with added columns (s: Float64, lateral_m: Float64) after
        applying the lateral off-route filter. Pings with lateral_m above the
        threshold are removed.

    Failure mode: if chunk boundaries produce s discontinuities, monotonicity
    of s for a straight on-route bus breaks. test_projection.py catches this.

    Source: build_notebook_03.py lines 390-468.
    """
    gps_e = gps.filter(pl.col("empresaid") == empresaid)
    if gps_e.is_empty():
        return gps_e.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("s"),
            pl.lit(None, dtype=pl.Float64).alias("lateral_m"),
        ])

    points_latlon = gps_e.select(["lat", "lon"]).to_numpy()
    s_arr, lateral_arr = _project_arc_length(points_latlon, centerline_latlon, chunk_size)

    threshold = lateral_threshold_for(empresaid)
    result = gps_e.with_columns([
        pl.Series("s", s_arr.astype(float), dtype=pl.Float64),
        pl.Series("lateral_m", lateral_arr.astype(float), dtype=pl.Float64),
    ]).filter(pl.col("lateral_m") <= threshold)

    return result


def project_per_direction(
    gps: pl.DataFrame,
    centerlines: dict[int, "np.ndarray"],
    *,
    empresaid: int,
    direction_col: str = "direction",
    chunk_size: int = 10_000,
) -> pl.DataFrame:
    """Project each direction subset onto its own centerline, then vertical_concat.

    Pings with direction not in centerlines (e.g. direction == 0) get NaN s,
    NaN lateral_m, and are kept (downstream filters handle them). Schema and dtypes
    match project_to_centerline.

    This function OVERWRITES the s and lateral_m columns in the returned DataFrame.
    It is designed for pass-2 of the two-pass pipeline: after pass-1 has already
    written s/lateral_m, call this to replace them with per-direction projections.

    Args:
        gps: DataFrame with columns including (direction, lat, lon) and existing
             s/lateral_m columns (will be overwritten). All rows are kept.
        centerlines: dict mapping direction int → (m, 2) centerline array.
                     Keys are typically {+1, -1}. Pings with unknown direction
                     keys receive NaN s and NaN lateral_m.
        empresaid: empresa identifier (for type consistency; not used for filtering
                   since gps is assumed to be already empresa-filtered).
        direction_col: name of the direction column (default "direction").
        chunk_size: number of pings per numpy batch (bounds peak memory).

    Returns:
        pl.DataFrame with same schema as input, same row count, with s and
        lateral_m overwritten by per-direction projections.
    """
    known_directions = set(centerlines.keys())
    parts: list[pl.DataFrame] = []

    for direction, cl in centerlines.items():
        subset = gps.filter(pl.col(direction_col) == direction)
        if subset.is_empty():
            continue
        pts = subset.select(["lat", "lon"]).to_numpy()
        s_arr, lateral_arr = _project_arc_length(pts, cl, chunk_size)
        subset = subset.with_columns([
            pl.Series("s", s_arr.astype(float), dtype=pl.Float64),
            pl.Series("lateral_m", lateral_arr.astype(float), dtype=pl.Float64),
        ])
        parts.append(subset)

    # Handle pings with unknown direction (not in centerlines) — assign NaN
    unknown_mask = ~pl.col(direction_col).is_in(list(known_directions))
    unknown_subset = gps.filter(unknown_mask)
    if not unknown_subset.is_empty():
        unknown_subset = unknown_subset.with_columns([
            pl.lit(float("nan"), dtype=pl.Float64).alias("s"),
            pl.lit(float("nan"), dtype=pl.Float64).alias("lateral_m"),
        ])
        parts.append(unknown_subset)

    if not parts:
        # Edge case: empty input
        return gps.with_columns([
            pl.lit(float("nan"), dtype=pl.Float64).alias("s"),
            pl.lit(float("nan"), dtype=pl.Float64).alias("lateral_m"),
        ])

    return pl.concat(parts)


def _project_arc_length(
    points_latlon: np.ndarray,
    centerline_latlon: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy point-to-polyline projection using local flat-Earth coordinates.

    For each point, finds the closest centerline segment, projects orthogonally,
    and computes cumulative arc-length s (meters from polyline start) plus
    lateral offset (perpendicular distance in meters).

    Memory: O(chunk_size × n_segments) intermediate tensor. chunk_size=10_000
    with 50 segments ≈ 4 MB float32 — bounded regardless of total ping count.

    Source: build_notebook_03.py lines 396-427.
    """
    pts = np.asarray(points_latlon, dtype=float)
    cl = np.asarray(centerline_latlon, dtype=float)

    # Convert to meters (local flat-Earth at Arequipa latitude).
    pts_m = np.stack([pts[:, 0] * LAT_DEG_M, pts[:, 1] * LON_DEG_M], axis=1)
    cl_m = np.stack([cl[:, 0] * LAT_DEG_M, cl[:, 1] * LON_DEG_M], axis=1)

    seg_starts = cl_m[:-1]                              # (m-1, 2)
    seg_vecs = np.diff(cl_m, axis=0)                    # (m-1, 2)
    seg_norms_sq = (seg_vecs ** 2).sum(axis=1)          # (m-1,)
    seg_lengths = np.sqrt(seg_norms_sq)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])   # (m,)

    n = pts_m.shape[0]
    s_out = np.zeros(n, dtype=np.float32)
    lateral_out = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = pts_m[start:end]                                       # (c, 2)
        diff = chunk[:, None, :] - seg_starts[None, :, :]             # (c, m-1, 2)
        t = (diff * seg_vecs[None, :, :]).sum(axis=2) / seg_norms_sq[None, :]
        t = np.clip(t, 0.0, 1.0)                                       # (c, m-1)
        proj = seg_starts[None, :, :] + t[:, :, None] * seg_vecs[None, :, :]
        dist_sq = ((chunk[:, None, :] - proj) ** 2).sum(axis=2)       # (c, m-1)
        best_seg = dist_sq.argmin(axis=1)                              # (c,)
        best_t = np.take_along_axis(t, best_seg[:, None], axis=1).squeeze(1)
        s_out[start:end] = cum_s[best_seg] + best_t * seg_lengths[best_seg]
        lateral_out[start:end] = np.sqrt(
            np.take_along_axis(dist_sq, best_seg[:, None], axis=1).squeeze(1)
        )

    return s_out, lateral_out
