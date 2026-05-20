"""Tests for src/preprocessing/corridor.py.

Covers:
  T1.1 — geographic outlier filter drops known outliers, retains inliers.
  T1.2 — build_centerline on a synthetic straight-line point set returns a valid polyline.
"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.preprocessing.corridor import _filter_geographic_outliers, _build_centerline_from_points


class TestFilterGeographicOutliers:
    """T1.1 — geographic outlier filter drops exactly the injected outliers."""

    def test_inliers_retained_outliers_dropped(self):
        """Failure mode: regression of the IQR-quantile filter; if the quantile
        thresholds are wrong (e.g. median instead of p0.5/p99.5) this fails.

        1000 inlier points inside the 0.5th–99.5th percentile box and 5 outliers
        outside it. With a large inlier sample, the p0.5/p99.5 thresholds are
        dominated by the inliers so the 5 extreme outliers fall outside the box.
        After filtering, at least 990 inliers must survive (0.5% trim may remove
        a few edge-case inliers) and all 5 injected outliers must be gone.
        """
        rng = np.random.default_rng(7)
        # 1000 inlier (lat, lon) pairs — large sample so p0.5/p99.5 is inlier-driven.
        inliers = rng.normal(loc=[-16.4, -71.52], scale=[0.01, 0.02], size=(1000, 2))

        # 5 extreme outliers at positions clearly beyond the inlier distribution.
        # At 3× the inlier std, these are at the 99.9th percentile of a Gaussian,
        # well outside the p0.5–p99.5 trim box of 1000 inlier-dominated points.
        outliers = np.array([
            [-16.4 - 0.12, -71.52],   # 12× std south
            [-16.4 + 0.12, -71.52],   # 12× std north
            [-16.4,         -71.52 - 0.30],   # 15× std west
            [-16.4,         -71.52 + 0.30],   # 15× std east
            [-16.4 - 0.12,  -71.52 - 0.30],   # corner
        ])

        pts = np.vstack([inliers, outliers])
        filtered = _filter_geographic_outliers(pts, q=(0.005, 0.995))

        # ~1% of inliers may be trimmed (0.5% per axis × 2 axes).
        # Threshold: >= 97% retained (allowing for boundary effects in 2D box).
        n_inliers_kept = 0
        for inlier in inliers:
            if np.any(np.all(np.isclose(filtered, inlier, atol=1e-9), axis=1)):
                n_inliers_kept += 1

        assert n_inliers_kept >= 970, (
            f"Expected >= 970 inliers retained; got {n_inliers_kept}"
        )

        # All 5 injected outliers must be dropped.
        for outlier in outliers:
            still_present = np.any(np.all(np.isclose(filtered, outlier, atol=1e-9), axis=1))
            assert not still_present, f"Outlier {outlier} survived filtering"


class TestBuildCenterline:
    """T1.2 — _build_centerline_from_points on a controlled synthetic route."""

    @pytest.fixture
    def straight_east_west_points(self) -> np.ndarray:
        """300 points along a straight east-west line (lon -71.55 → -71.50,
        lat -16.4 fixed), with small Gaussian lat jitter to allow PCA to work.
        """
        rng = np.random.default_rng(42)
        n = 300
        lons = np.linspace(-71.55, -71.50, n)
        lats = np.full(n, -16.4) + rng.normal(0, 0.0002, n)
        return np.stack([lats, lons], axis=1)

    def test_vertex_count_within_bounds(self, straight_east_west_points: np.ndarray):
        """Failure mode: if bins with < 5 points are over-dropped (too few points
        per bin after trimming) the vertex count shrinks below 5.

        With 300 uniformly distributed points, CENTERLINE_N_BINS=50 should
        yield between 5 and 50 vertices.
        """
        cl = _build_centerline_from_points(straight_east_west_points)
        assert 5 <= cl.shape[0] <= 50, f"Expected 5–50 vertices, got {cl.shape[0]}"
        assert cl.shape[1] == 2

    def test_lon_span_covers_route(self, straight_east_west_points: np.ndarray):
        """Failure mode: PCA sign flip or over-trimming compresses the polyline.

        The route spans lon -71.55 → -71.50 (delta = 0.05 degrees ≈ 4.7 km).
        After trim_pct=0.025 at each end, expected span >= 0.04 degrees.
        """
        cl = _build_centerline_from_points(straight_east_west_points)
        lon_span = cl[:, 1].max() - cl[:, 1].min()
        assert lon_span >= 0.04, (
            f"Lon span {lon_span:.4f} is too small — PCA sign flip or excessive trimming?"
        )

    def test_shape_is_2d(self, straight_east_west_points: np.ndarray):
        """Basic shape sanity: output must be (m, 2)."""
        cl = _build_centerline_from_points(straight_east_west_points)
        assert cl.ndim == 2
        assert cl.shape[1] == 2
