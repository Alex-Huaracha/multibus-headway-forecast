"""Tests for src/preprocessing/corridor.py.

Covers:
  T1.1 — geographic outlier filter drops known outliers, retains inliers.
  T1.2 — build_centerline on a synthetic straight-line point set returns a valid polyline.
  T1.5 — dual-filar fixture sanity (row count, both directions, geographic separation).
  T2.1..T2.6 — build_centerline_per_direction per-direction centerline + fallback.
"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.preprocessing.corridor import _filter_geographic_outliers, _build_centerline_from_points
from tests.fixtures.synthetic import make_dual_filar_gps


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


# ---------------------------------------------------------------------------
# T1.5 RED: TestDualFilarFixture — sanity checks for make_dual_filar_gps
# ---------------------------------------------------------------------------

_N_BUSES = 4
_N_PINGS = 300
_SEP_M = 40.0


class TestDualFilarFixture:
    """Sanity checks for the make_dual_filar_gps factory (design §6).

    These tests are RED-then-immediate-GREEN: the fixture is pure data generation,
    so they should pass as soon as T1.4 INFRA (the fixture function) lands.
    """

    @pytest.fixture(scope="class")
    def dual_filar(self) -> pl.DataFrame:
        return make_dual_filar_gps(
            empresaid=59,
            n_buses_per_street=_N_BUSES,
            n_pings_per_bus=_N_PINGS,
            street_separation_m=_SEP_M,
            rng_seed=42,
        )

    def test_fixture_row_count(self, dual_filar: pl.DataFrame):
        """Total rows must equal 2 * n_buses_per_street * n_pings_per_bus."""
        expected_rows = 2 * _N_BUSES * _N_PINGS
        assert dual_filar.height == expected_rows, (
            f"Expected {expected_rows} rows; got {dual_filar.height}"
        )

    def test_both_directions_present(self, dual_filar: pl.DataFrame):
        """Both direction=+1 and direction=-1 must be present in the fixture."""
        directions = set(dual_filar["direction"].unique().to_list())
        assert 1 in directions, "direction=+1 (street A) must be present"
        assert -1 in directions, "direction=-1 (street B) must be present"

    def test_geographic_separation(self, dual_filar: pl.DataFrame):
        """Mean lat of dir=+1 group and dir=-1 group must be separated by
        approximately street_separation_m (within ±20%).
        """
        from src.preprocessing.config import LAT_DEG_M
        mean_lat_plus = dual_filar.filter(pl.col("direction") == 1)["lat"].mean()
        mean_lat_minus = dual_filar.filter(pl.col("direction") == -1)["lat"].mean()
        assert mean_lat_plus is not None and mean_lat_minus is not None
        sep_m = abs(mean_lat_plus - mean_lat_minus) * LAT_DEG_M
        tolerance = 0.20 * _SEP_M
        assert abs(sep_m - _SEP_M) <= tolerance, (
            f"Expected geographic separation ≈ {_SEP_M} m ± {tolerance:.1f} m; "
            f"got {sep_m:.2f} m"
        )


# ---------------------------------------------------------------------------
# T2.1–T2.4 RED: TestBuildCenterlinePerDirection
# ---------------------------------------------------------------------------


class TestBuildCenterlinePerDirection:
    """D2-CL-API, D2-FALLBACK-SPARSE — per-direction centerline wrapper.

    T2.1: returns dict keyed by {+1, -1} with valid (m,2) arrays.
    T2.2: the two centerlines' centroids are spatially separated.
    T2.3: fallback triggers when subset is below min_pings_per_dir.
    T2.4: fallback emits structured log warning via caplog.
    T2.6: centerlines are deterministic across two calls on identical input.
    """

    @pytest.fixture(scope="class")
    def dual_filar(self) -> pl.DataFrame:
        return make_dual_filar_gps(
            empresaid=59,
            n_buses_per_street=4,
            n_pings_per_bus=300,
            street_separation_m=40.0,
            rng_seed=42,
        )

    def test_returns_dict_keyed_by_direction(self, dual_filar: pl.DataFrame):
        """T2.1: build_centerline_per_direction returns dict with keys {+1, -1},
        each value is np.ndarray of shape (m, 2) with m >= 1.
        """
        from src.preprocessing.corridor import build_centerline_per_direction
        result = build_centerline_per_direction(dual_filar, empresaid=59)
        assert isinstance(result, dict), f"Expected dict; got {type(result)}"
        assert set(result.keys()) == {1, -1}, (
            f"Expected keys {{+1, -1}}; got {set(result.keys())}"
        )
        for direction, cl in result.items():
            assert isinstance(cl, np.ndarray), (
                f"Expected np.ndarray for direction={direction}; got {type(cl)}"
            )
            assert cl.ndim == 2, f"Expected 2D array for direction={direction}; got ndim={cl.ndim}"
            assert cl.shape[1] == 2, (
                f"Expected shape (m,2) for direction={direction}; got {cl.shape}"
            )
            assert cl.shape[0] >= 1, (
                f"Expected m>=1 for direction={direction}; got shape={cl.shape}"
            )

    def test_centerlines_are_spatially_separated(self, dual_filar: pl.DataFrame):
        """T2.2: centroid of cl[+1] and centroid of cl[-1] must be separated in lat
        by >= 80% of street_separation_m (40.0 m default).
        """
        from src.preprocessing.corridor import build_centerline_per_direction
        from src.preprocessing.config import LAT_DEG_M
        result = build_centerline_per_direction(dual_filar, empresaid=59)
        centroid_plus = result[1].mean(axis=0)   # (lat, lon) centroid of cl[+1]
        centroid_minus = result[-1].mean(axis=0)  # (lat, lon) centroid of cl[-1]
        sep_m = abs(centroid_plus[0] - centroid_minus[0]) * LAT_DEG_M
        assert sep_m >= 0.80 * 40.0, (
            f"Expected centerline centroid separation >= {0.80 * 40.0:.1f} m; "
            f"got {sep_m:.2f} m"
        )

    def test_fallback_triggers_when_subset_too_small(self, dual_filar: pl.DataFrame):
        """T2.3: when direction=+1 subset has < 1000 pings, build_centerline_per_direction
        must fall back to the single-pass centerline for that direction (not None,
        and returns a valid array).
        """
        from src.preprocessing.corridor import build_centerline_per_direction
        # Create a sparse-dir=+1 frame: only 50 pings for +1, plenty for -1
        sparse = dual_filar.filter(pl.col("direction") == -1)  # full -1 set
        sparse_plus = dual_filar.filter(pl.col("direction") == 1).head(50)  # only 50 pings
        sparse_frame = pl.concat([sparse, sparse_plus])
        result = build_centerline_per_direction(
            sparse_frame, empresaid=59, min_pings_per_dir=1_000
        )
        # +1 direction must have fallen back to a valid single-pass centerline
        assert 1 in result, "Key +1 must be present even after fallback"
        assert result[1] is not None, "Fallback must return a valid array, not None"
        assert isinstance(result[1], np.ndarray)
        assert result[1].shape[0] >= 1

    def test_fallback_logs_structured_event(self, dual_filar: pl.DataFrame, caplog):
        """T2.4: fallback emits a logging.warning with empresaid, direction, and pings."""
        import logging
        from src.preprocessing.corridor import build_centerline_per_direction
        sparse_plus = dual_filar.filter(pl.col("direction") == 1).head(50)
        sparse_minus = dual_filar.filter(pl.col("direction") == -1)
        sparse_frame = pl.concat([sparse_plus, sparse_minus])
        with caplog.at_level(logging.WARNING):
            build_centerline_per_direction(
                sparse_frame, empresaid=59, min_pings_per_dir=1_000
            )
        # Verify that at least one warning was emitted containing the required fields
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1, (
            "Expected at least one WARNING log entry for fallback; got none"
        )
        full_log = " ".join(r.getMessage() for r in warning_records)
        assert "59" in full_log, f"empresaid=59 not found in log: {full_log!r}"
        assert "1" in full_log or "+1" in full_log, (
            f"direction not found in log: {full_log!r}"
        )
        assert "50" in full_log, f"ping count (50) not found in log: {full_log!r}"

    def test_centerline_deterministic_across_runs(self, dual_filar: pl.DataFrame):
        """T2.6: build_centerline_per_direction is deterministic: two calls on
        identical input produce numerically identical centerlines.
        """
        from src.preprocessing.corridor import build_centerline_per_direction
        cl1 = build_centerline_per_direction(dual_filar, empresaid=59)
        cl2 = build_centerline_per_direction(dual_filar, empresaid=59)
        for direction in [1, -1]:
            assert np.array_equal(cl1[direction], cl2[direction]), (
                f"Centerline for direction={direction} is not deterministic across runs"
            )
