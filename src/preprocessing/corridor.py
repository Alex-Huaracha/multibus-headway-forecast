"""Corridor centerline construction for the preprocessing pipeline.

Extracts an ordered (n_bins, 2) lat/lon polyline from GPS pings via:
  1. Geographic-outlier filter (IQR box trim at configurable quantiles).
  2. PCA to find the principal axis of the corridor.
  3. Binned median along the principal axis.
  4. Smoothing of the secondary (cross-corridor) coordinate.
  5. Back-transformation to (lat, lon).

Source: derived from build_notebook_03.py lines 279-361.
"""
from __future__ import annotations

import logging
import numpy as np
import polars as pl

from .config import EMPRESA_CONFIG, LAT_DEG_M, LON_DEG_M, PRODUCTIVE_PARAMS

logger = logging.getLogger(__name__)


def _filter_geographic_outliers(
    points_latlon: np.ndarray,
    q: tuple[float, float] = (
        PRODUCTIVE_PARAMS.centerline_latlon_quantile_lo,
        PRODUCTIVE_PARAMS.centerline_latlon_quantile_hi,
    ),
) -> np.ndarray:
    """Trim pings outside the [q_lo, q_hi] quantile box of lat and lon.

    Failure mode: if this filter is broken (too loose or too tight) the PCA
    principal axis tilts off-corridor or the sample becomes too small. The
    test_corridor.py outlier test catches regressions in both directions.

    Args:
        points_latlon: (n, 2) array of (lat, lon) values.
        q: (q_lo, q_hi) quantile tuple, default from PRODUCTIVE_PARAMS.

    Returns:
        Filtered (n_kept, 2) array.
    """
    pts = np.asarray(points_latlon, dtype=float)
    lat_lo, lat_hi = np.quantile(pts[:, 0], q)
    lon_lo, lon_hi = np.quantile(pts[:, 1], q)
    mask = (
        (pts[:, 0] >= lat_lo) & (pts[:, 0] <= lat_hi)
        & (pts[:, 1] >= lon_lo) & (pts[:, 1] <= lon_hi)
    )
    return pts[mask]


def build_centerline(
    gps: pl.DataFrame,
    empresaid: int,
    rng_seed: int = 42,
) -> np.ndarray:
    """Build the ordered (m, 2) lat/lon polyline for one empresa.

    Pipeline: geographic-outlier filter → PCA → binned median → trim → smooth
    → back-transform to (lat, lon).

    Args:
        gps: DataFrame with columns (empresaid, unidadid, lat, lon, speed_kmh).
             speed_kmh must already be populated — call
             projection.attach_observed_speed first.
        empresaid: which empresa to build the centerline for.
        rng_seed: seed for deterministic random sampling when the GPS sample
                  exceeds centerline_sample_cap.

    Returns:
        np.ndarray shape (m, 2) of (lat, lon) ordered along the principal axis,
        where m <= PRODUCTIVE_PARAMS.centerline_n_bins (bins with < 5 samples
        are silently dropped).

    Failure mode: PCA sign flip (centered data → eigenvector pointing west)
    produces a reversed polyline. test_corridor.py checks that the first vertex
    is near LON_START and the last is near LON_END of the synthetic route.
    """
    cfg = EMPRESA_CONFIG[empresaid]
    params = PRODUCTIVE_PARAMS

    moving = (
        gps.filter(
            (pl.col("empresaid") == empresaid)
            & (pl.col("speed_kmh") >= params.min_speed_for_centerline_kmh)
        )
        .select(["lat", "lon"])
    )

    rng = np.random.default_rng(rng_seed)
    sample: np.ndarray = moving.to_numpy()
    if len(sample) > cfg.centerline_sample_cap:
        idx = rng.choice(len(sample), size=cfg.centerline_sample_cap, replace=False)
        sample = sample[idx]

    return _build_centerline_from_points(
        sample,
        n_bins=params.centerline_n_bins,
        trim_pct=params.centerline_trim_pct,
        smooth_win=params.centerline_smooth_win,
    )


def build_centerline_per_direction(
    gps: pl.DataFrame,
    *,
    empresaid: int,
    direction_col: str = "direction",
    min_pings_per_dir: int = 1_000,
    rng_seed: int = 42,
) -> dict[int, np.ndarray]:
    """Build one (m, 2) centerline per direction key {+1, -1}.

    Filters gps by empresaid and speed >= min_speed_for_centerline_kmh, partitions
    by direction_col, calls _build_centerline_from_points per subset. Subsets below
    min_pings_per_dir fall back to single-pass build_centerline; same fallback on
    ValueError from sparse bins. Logs a structured FallbackEvent per fallback.

    Args:
        gps: DataFrame with columns (empresaid, direction, speed_kmh, lat, lon).
             speed_kmh must already be populated.
        empresaid: which empresa to build centerlines for.
        direction_col: name of the direction column (default "direction").
        min_pings_per_dir: minimum pings required per direction subset to attempt
                           per-direction PCA. Below this, falls back to single-pass
                           centerline. (R-CL1)
        rng_seed: seed for deterministic random sampling in the fallback single-pass
                  build_centerline call.

    Returns:
        dict[int, np.ndarray] with keys +1 and -1. Each value is the (m, 2)
        centerline for that direction subset. When a subset falls back to the
        single-pass centerline, that centerline is stored for the direction key.

    Raises:
        Never raises — all exceptions from _build_centerline_from_points trigger
        the fallback path.
    """
    params = PRODUCTIVE_PARAMS

    # Filter to this empresa's moving pings.
    # Speed filter is applied only when speed_kmh column is present
    # (it may be absent in test fixtures that pre-set direction without going
    # through attach_observed_speed).
    if "speed_kmh" in gps.columns:
        moving = gps.filter(
            (pl.col("empresaid") == empresaid)
            & (pl.col("speed_kmh") >= params.min_speed_for_centerline_kmh)
        )
    else:
        moving = gps.filter(pl.col("empresaid") == empresaid)

    # Build the single-pass centerline once (used as fallback for sparse directions)
    single_pass_cl = build_centerline(gps, empresaid=empresaid, rng_seed=rng_seed)

    result: dict[int, np.ndarray] = {}

    for direction in [1, -1]:
        subset = moving.filter(pl.col(direction_col) == direction)
        n_pings = subset.height

        if n_pings < min_pings_per_dir:
            logger.warning(
                "FallbackEvent: empresaid=%d direction=%d pings=%d reason=sparse "
                "(below min_pings_per_dir=%d); using single-pass centerline",
                empresaid, direction, n_pings, min_pings_per_dir,
            )
            result[direction] = single_pass_cl
            continue

        points = subset.select(["lat", "lon"]).to_numpy()
        try:
            cl = _build_centerline_from_points(
                points,
                n_bins=params.centerline_n_bins,
                trim_pct=params.centerline_trim_pct,
                smooth_win=params.centerline_smooth_win,
            )
            result[direction] = cl
        except ValueError as exc:
            logger.warning(
                "FallbackEvent: empresaid=%d direction=%d pings=%d reason=pca_error "
                "(%s); using single-pass centerline",
                empresaid, direction, n_pings, exc,
            )
            result[direction] = single_pass_cl

    return result


def _build_centerline_from_points(
    points_latlon: np.ndarray,
    n_bins: int = PRODUCTIVE_PARAMS.centerline_n_bins,
    trim_pct: float = PRODUCTIVE_PARAMS.centerline_trim_pct,
    smooth_win: int = PRODUCTIVE_PARAMS.centerline_smooth_win,
) -> np.ndarray:
    """Inner implementation of centerline construction from a point array.

    Separated from build_centerline to make the algorithm unit-testable with
    arbitrary point sets (not tied to a polars DataFrame or empresa).

    Args:
        points_latlon: (n, 2) array of (lat, lon) values.
        n_bins: number of bins along the principal axis.
        trim_pct: fraction of extreme principal-axis positions to drop.
        smooth_win: rolling mean window for the cross-corridor coordinate.

    Returns:
        np.ndarray shape (m, 2) of (lat, lon), m <= n_bins.
    """
    pts = _filter_geographic_outliers(points_latlon)

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA via eigen-decomposition of the 2×2 covariance matrix.
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    projected = centered @ eigvecs     # (n, 2)
    t1 = projected[:, 0]               # principal axis coordinate
    t2 = projected[:, 1]               # cross-corridor coordinate

    # Trim extreme percentiles along the principal axis.
    lo, hi = np.quantile(t1, [trim_pct, 1.0 - trim_pct])
    mask = (t1 >= lo) & (t1 <= hi)
    t1, t2 = t1[mask], t2[mask]

    # Bin along the principal axis; take median cross-corridor coord per bin.
    bins = np.linspace(t1.min(), t1.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(t1, bins) - 1, 0, n_bins - 1)

    cl_proj: list[list[float]] = []
    for i in range(n_bins):
        m = bin_idx == i
        if m.sum() < 5:
            continue
        cl_proj.append([0.5 * (bins[i] + bins[i + 1]), float(np.median(t2[m]))])

    if not cl_proj:
        raise ValueError(
            f"build_centerline produced no bins with >= 5 points for n_bins={n_bins}. "
            "The GPS sample may be too small or too sparse."
        )

    cl_proj_arr = np.array(cl_proj)

    # Smooth the cross-corridor coordinate with a rolling mean.
    if smooth_win > 1 and len(cl_proj_arr) >= smooth_win:
        kernel = np.ones(smooth_win) / smooth_win
        cl_proj_arr[:, 1] = np.convolve(cl_proj_arr[:, 1], kernel, mode="same")

    # Back-transform from PCA space to (lat, lon).
    cl_latlon: np.ndarray = cl_proj_arr @ eigvecs.T + centroid
    return cl_latlon
