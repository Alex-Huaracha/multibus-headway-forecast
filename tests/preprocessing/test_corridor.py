"""Tests for src/preprocessing/corridor.py.

Covers:
  T1.1 — geographic outlier filter drops known outliers, retains inliers.
  T1.2 — build_centerline on a synthetic straight-line point set returns a valid polyline.
  T1.5 — dual-filar fixture sanity (row count, both directions, geographic separation).
  T2.1..T2.6 — build_centerline_per_direction per-direction centerline + fallback.
  AC-ORIENT-1 — centerline orientation is invariant to input reversal.
  AC-ORIENT-2 — centerline is unchanged for already-forward input.
"""
import hashlib
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


# ---------------------------------------------------------------------------
# AC-ORIENT-1: Orientation invariant to input reversal
# AC-ORIENT-2: No mutation on already-forward input
# ---------------------------------------------------------------------------


class TestCenterlineOrientationDeterminism:
    """AC-ORIENT-1, AC-ORIENT-2 — centerline orientation is deterministic.

    Wave 1: these tests define the orientation-enforcement contract.

    The core bug: _build_centerline_from_points bins points along the PCA
    principal axis (PC1). The bins are in ascending PC1 order. After back-
    transforming to (lat, lon), the returned polyline's geographic direction
    depends on the SIGN of the PC1 eigenvector:

      - If eigvecs[:, 0] has a positive dominant component → low PC1 = geographic
        'start' (e.g., south end for a north-south route) → cl_latlon goes south→north.
      - If eigvecs[:, 0] has a negative dominant component → low PC1 = geographic
        'end' (e.g., north end for a north-south route) → cl_latlon goes north→south.

    numpy.linalg.eigh may return eigenvectors with either sign depending on the
    data distribution. For a north-south route (rng seed 7), it returns a
    south-pointing eigvec, making cl_latlon go north→south. Buses traveling
    south→north see decreasing s and get relabeled direction=-1 by infer_direction,
    causing trajectory misses in compute_headways_c2.

    The fix: after back-transform, if the dominant component of eigvecs[:, 0]
    is negative, reverse cl_latlon. This normalizes the orientation so the
    centerline always traverses from the 'negative-dominant-axis' end to the
    'positive-dominant-axis' end in geographic space.
    """

    @pytest.fixture
    def west_east_points(self) -> np.ndarray:
        """300 points along a strict west-east line.

        lon goes from -71.55 (west) to -71.50 (east), lat fixed at -16.4 with
        small Gaussian jitter. The PCA dominant axis is lon, and the eigvec
        points east (positive lon component), so the centerline naturally runs
        west→east for this fixture. Used as the 'already-canonical' fixture for
        AC-ORIENT-2 — the fix should be a no-op on this data.
        """
        rng = np.random.default_rng(42)
        n = 300
        lons = np.linspace(-71.55, -71.50, n)
        lats = np.full(n, -16.4) + rng.normal(0, 0.0002, n)
        return np.stack([lats, lons], axis=1)

    @pytest.fixture
    def north_south_points(self) -> np.ndarray:
        """300 points along a north-south route where numpy returns
        a south-pointing (negative lat) eigenvector, causing cl_latlon
        to run north→south WITHOUT the orientation fix.

        lat goes from -16.50 (south) to -16.30 (north), lon fixed near -71.52
        with very small jitter so lat is the clearly dominant axis.
        Seed 7 reliably triggers the negative-lat eigenvector sign.
        """
        rng = np.random.default_rng(7)
        n = 300
        lats = np.linspace(-16.50, -16.30, n) + rng.normal(0, 0.0002, n)
        lons = np.full(n, -71.52) + rng.normal(0, 0.00001, n)
        return np.stack([lats, lons], axis=1)

    def test_centerline_orientation_invariant_to_input_reversal(
        self, west_east_points: np.ndarray
    ):
        """AC-ORIENT-1 (regression guard): forward and reversed input must
        produce the same polyline (up to full reversal).

        For a west-east route, both orderings have the same covariance matrix
        (covariance is order-invariant), so they must produce identical results.
        This is already true without the fix. The test guards against any
        regression introduced by the orientation patch.
        """
        pts_fwd = west_east_points
        pts_rev = west_east_points[::-1]

        cl_fwd = _build_centerline_from_points(pts_fwd)
        cl_rev = _build_centerline_from_points(pts_rev)

        same_direction = np.allclose(cl_fwd, cl_rev, atol=1e-8)
        mirror_direction = np.allclose(cl_fwd, cl_rev[::-1], atol=1e-8)

        assert same_direction or mirror_direction, (
            f"Centerline is NOT orientation-invariant to input reversal.\n"
            f"cl_fwd[0]={cl_fwd[0]}, cl_fwd[-1]={cl_fwd[-1]}\n"
            f"cl_rev[0]={cl_rev[0]}, cl_rev[-1]={cl_rev[-1]}\n"
            "Neither np.allclose(fwd, rev) nor np.allclose(fwd, rev[::-1]) holds."
        )

    def test_centerline_canonical_orientation_for_north_south_route(
        self, north_south_points: np.ndarray
    ):
        """AC-ORIENT-1 (primary RED test): for a north-south route where numpy
        returns a NEGATIVE-lat eigenvector, _build_centerline_from_points MUST
        return a centerline where cl_latlon[0] has LOWER lat than cl_latlon[-1]
        (south→north, which is canonical = lower lat first).

        WITHOUT the fix (RED): the PCA principal eigvec points south (negative lat).
        Bins in ascending PC1 order correspond to north→south geographically.
        After back-transform, cl_latlon[0] is at lat ≈ -16.31 (NORTH) and
        cl_latlon[-1] is at lat ≈ -16.49 (SOUTH). The test assertion fails.

        WITH the fix (GREEN): the dominant eigvec component is checked. Since
        it is negative (lat component ≈ -1.0), cl_latlon is reversed so that
        cl_latlon[0] is at lat ≈ -16.49 (SOUTH) and cl_latlon[-1] at lat ≈ -16.31
        (NORTH). The assertion passes.
        """
        pts = north_south_points
        cl = _build_centerline_from_points(pts)

        # cl_latlon[:, 0] is the lat column.
        # For a south-to-north canonical orientation: first lat < last lat.
        assert cl[0, 0] < cl[-1, 0], (
            f"Centerline for a north-south route is NOT canonically oriented "
            f"(south→north).\n"
            f"cl[0] (lat={cl[0, 0]:.5f}) >= cl[-1] (lat={cl[-1, 0]:.5f}).\n"
            "The PCA eigenvector points south, causing the centerline to run "
            "north→south. Apply the orientation fix: when the dominant eigvec "
            "component is negative, reverse cl_latlon after back-transform."
        )

    def test_centerline_orientation_preserved_for_forward_input(
        self, west_east_points: np.ndarray
    ):
        """AC-ORIENT-2: for a west-east route (already canonical), the returned
        centerline must be byte-identical across repeated calls.

        Regression guard: the orientation patch must NOT mutate the output when
        the eigenvector already points in the canonical (positive) direction.
        The fix should be a no-op for this fixture.
        """
        pts = west_east_points

        cl1 = _build_centerline_from_points(pts)
        cl2 = _build_centerline_from_points(pts)

        hash1 = hashlib.md5(cl1.tobytes()).hexdigest()
        hash2 = hashlib.md5(cl2.tobytes()).hexdigest()

        assert hash1 == hash2, (
            f"_build_centerline_from_points is non-deterministic on repeated calls "
            f"with identical forward-oriented input.\nhash1={hash1}\nhash2={hash2}"
        )
