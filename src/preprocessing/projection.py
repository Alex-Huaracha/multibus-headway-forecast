"""Speed attachment and arc-length projection for the preprocessing pipeline.

Provides:
  attach_observed_speed — compute step_m, dt_s, speed_kmh per (empresaid, unidadid).
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
    MAX_PLAUSIBLE_SPEED_KMH,
    lateral_threshold_for,
)


def attach_observed_speed(gps: pl.DataFrame) -> pl.DataFrame:
    """Add columns (lat_prev, lon_prev, time_prev, step_m, dt_s, speed_kmh) by
    diffing successive rows of the same (empresaid, unidadid).

    speed_kmh is computed as step_m / dt_s * 3.6 (observed speed from GPS
    displacement). Values exceeding MAX_PLAUSIBLE_SPEED_KMH (80 km/h) are set
    to null — these signal GPS jumps or data errors, NOT high-speed buses.

    The raw `velocidad` field is intentionally NOT used anywhere in the pipeline
    (decisiones-limpieza-fase2 §2.3).

    Source: build_notebook_03.py lines 250-272.
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
    # Cap implausible speeds: GPS jumps produce speed > 80 km/h → null.
    gps = gps.with_columns(
        pl.when(pl.col("speed_kmh") > MAX_PLAUSIBLE_SPEED_KMH)
          .then(None)
          .otherwise(pl.col("speed_kmh"))
          .alias("speed_kmh")
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
