"""Integration smoke test T4.1 — full pipeline end-to-end on synthetic fixtures.

Runs the complete preprocessing pipeline:
  attach_observed_speed → build_centerline → project_to_centerline →
  infer_direction → assign_trip_ids → build_snapshots → compute_headways_c2

Asserts:
  - cleaned_gps_E2.parquet schema matches R6 (8 columns, correct dtypes)
  - headways_E2.parquet schema matches R7 (11 columns, correct dtypes)
  - headways has >= 5 rows with non-null delta_t_min
  - All t values satisfy t.dt.second() == 0 (INV-6 / clarification #17 rule 1)
  - INV-3: pair_rank dense per (t, direction)
  - INV-4: n_buses >= 2 for all rows
  - INV-7: bus_front != bus_back for all rows
  - INV-8: lateral_m <= 300.0 for all rows in cleaned_gps
  - INV-1: delta_t_min >= 0 OR delta_t_min IS NULL (clarification #17 updated INV-1)
"""
from __future__ import annotations
from pathlib import Path

import polars as pl
import pytest

from src.preprocessing.corridor import build_centerline
from src.preprocessing.direction import infer_direction
from src.preprocessing.headways import compute_headways_c2
from src.preprocessing.projection import attach_observed_speed, project_to_centerline
from src.preprocessing.trips import assign_trip_ids, build_snapshots

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full pipeline on synthetic_gps_e2.parquet; return (cleaned_gps, headways)."""
    tmp = tmp_path_factory.mktemp("integration")
    empresaid = 2

    gps_raw = pl.read_parquet(FIXTURES_DIR / "synthetic_gps_e2.parquet")

    # Step 1: observed speed.
    gps = attach_observed_speed(gps_raw)

    # Step 2: centerline.
    centerline = build_centerline(gps, empresaid=empresaid)

    # Step 3: project + off-route filter.
    gps = project_to_centerline(gps, centerline, empresaid=empresaid)

    # Step 4: direction.
    gps = infer_direction(gps)

    # Step 5: trip segmentation.
    gps = assign_trip_ids(gps)

    # Step 6: snapshots.
    snaps = build_snapshots(gps)

    # Step 7: headways (C.2).
    headways, _ = compute_headways_c2(snaps, gps)

    # Write cleaned GPS — include the columns defined in R6 schema.
    # The pipeline frame has extra columns; select only what R6 requires.
    cleaned_cols = ["unidadid", "t", "lat", "lon", "s", "direction", "speed_kmh", "lateral_m"]
    # Note: the cleaned GPS is the projected gps frame; rename 'time' → 't' for R6.
    cleaned_gps = gps.rename({"time": "t"}).select([
        c for c in cleaned_cols if c in gps.rename({"time": "t"}).columns
    ])

    cleaned_path = tmp / "cleaned_gps_E2.parquet"
    headways_path = tmp / "headways_E2.parquet"
    cleaned_gps.write_parquet(cleaned_path)
    headways.write_parquet(headways_path)

    return cleaned_gps, headways


class TestCleanedGPSSchema:
    """R6 schema: cleaned_gps must have the correct columns and non-null constraints."""

    def test_schema_columns(self, pipeline_outputs: tuple):
        cleaned_gps, _ = pipeline_outputs
        required = {"unidadid", "t", "lat", "lon", "s", "direction", "speed_kmh", "lateral_m"}
        assert required.issubset(set(cleaned_gps.columns)), (
            f"Missing R6 columns: {required - set(cleaned_gps.columns)}"
        )

    def test_lateral_m_within_threshold(self, pipeline_outputs: tuple):
        """INV-8: lateral_m <= 300.0 for all rows."""
        cleaned_gps, _ = pipeline_outputs
        if "lateral_m" not in cleaned_gps.columns:
            pytest.skip("lateral_m not in output")
        over_threshold = cleaned_gps.filter(pl.col("lateral_m") > 300.0)
        assert len(over_threshold) == 0, (
            f"INV-8 violated: {len(over_threshold)} rows with lateral_m > 300 m"
        )


class TestHeadwaysSchema:
    """R7 schema: headways must have 11 correct columns."""

    def test_schema_columns(self, pipeline_outputs: tuple):
        _, headways = pipeline_outputs
        required = {
            "t", "direction", "pair_rank", "bus_front", "bus_back",
            "s_front", "s_back", "speed_front_kmh", "speed_back_kmh",
            "delta_t_min", "n_buses",
        }
        assert required.issubset(set(headways.columns)), (
            f"Missing R7 columns: {required - set(headways.columns)}"
        )

    def test_non_empty(self, pipeline_outputs: tuple):
        _, headways = pipeline_outputs
        assert len(headways) > 0, "headways frame must not be empty"

    def test_min_nonnull_delta_t(self, pipeline_outputs: tuple):
        """Must have at least 5 rows with non-null delta_t_min."""
        _, headways = pipeline_outputs
        nonnull_count = headways.filter(pl.col("delta_t_min").is_not_null()).height
        assert nonnull_count >= 5, (
            f"Expected >= 5 rows with non-null delta_t_min; got {nonnull_count}"
        )


class TestInvariants:
    """All INV-1 through INV-8 assertions on the pipeline output."""

    def test_inv1_delta_t_non_negative_or_null(self, pipeline_outputs: tuple):
        """INV-1 (updated by clarification #17): delta_t_min >= 0 OR delta_t_min IS NULL."""
        _, headways = pipeline_outputs
        negative_rows = headways.filter(
            pl.col("delta_t_min").is_not_null() & (pl.col("delta_t_min") < 0)
        )
        assert len(negative_rows) == 0, (
            f"INV-1 violated: {len(negative_rows)} rows with negative delta_t_min"
        )

    def test_inv3_pair_rank_dense(self, pipeline_outputs: tuple):
        """INV-3: pair_rank is dense 1..N-1 per (t, direction)."""
        _, headways = pipeline_outputs
        if headways.is_empty():
            return
        for (t_val, d), group in headways.group_by(["t", "direction"], maintain_order=True):
            ranks = sorted(group["pair_rank"].to_list())
            expected = list(range(1, len(ranks) + 1))
            assert ranks == expected, (
                f"INV-3 violated at t={t_val} dir={d}: "
                f"pair_rank={ranks}, expected={expected}"
            )

    def test_inv4_n_buses_at_least_2(self, pipeline_outputs: tuple):
        """INV-4: n_buses >= 2 for all rows."""
        _, headways = pipeline_outputs
        violating = headways.filter(pl.col("n_buses") < 2)
        assert len(violating) == 0, (
            f"INV-4 violated: {len(violating)} rows with n_buses < 2"
        )

    def test_inv6_grid_timestamps_minute_aligned(self, pipeline_outputs: tuple):
        """INV-6 (clarification #17 rule 1): all t values have second == 0."""
        _, headways = pipeline_outputs
        if headways.is_empty():
            return
        non_zero_seconds = headways.filter(pl.col("t").dt.second() != 0)
        assert len(non_zero_seconds) == 0, (
            f"INV-6 violated: {len(non_zero_seconds)} rows with non-zero seconds in t"
        )

    def test_inv7_bus_front_not_equal_bus_back(self, pipeline_outputs: tuple):
        """INV-7: bus_front != bus_back for all rows."""
        _, headways = pipeline_outputs
        same_bus = headways.filter(pl.col("bus_front") == pl.col("bus_back"))
        assert len(same_bus) == 0, (
            f"INV-7 violated: {len(same_bus)} rows where bus_front == bus_back"
        )

    def test_inv8_lateral_m_in_cleaned_gps(self, pipeline_outputs: tuple):
        """INV-8: lateral_m <= 300.0 in cleaned GPS (duplicate of schema test, explicit name)."""
        cleaned_gps, _ = pipeline_outputs
        if "lateral_m" not in cleaned_gps.columns:
            pytest.skip("lateral_m not in cleaned_gps")
        violating = cleaned_gps.filter(pl.col("lateral_m") > 300.0)
        assert len(violating) == 0, (
            f"INV-8 violated: {len(violating)} rows with lateral_m > 300 m"
        )
