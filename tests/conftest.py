"""Shared pytest fixtures for the preprocessing test suite."""
from pathlib import Path

import polars as pl
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def gps_e2_fixture() -> pl.DataFrame:
    """Load the synthetic E2 GPS parquet committed in tests/fixtures/."""
    return pl.read_parquet(FIXTURES_DIR / "synthetic_gps_e2.parquet")


@pytest.fixture(scope="session")
def gps_e59_fixture() -> pl.DataFrame:
    """Load the synthetic E59 GPS parquet committed in tests/fixtures/."""
    return pl.read_parquet(FIXTURES_DIR / "synthetic_gps_e59.parquet")


@pytest.fixture(scope="session")
def centerline_e2_fixture(gps_e2_fixture: pl.DataFrame):
    """Build the E2 centerline once per session (build_centerline is expensive)."""
    from src.preprocessing.projection import attach_observed_speed
    from src.preprocessing.corridor import build_centerline

    gps = attach_observed_speed(gps_e2_fixture)
    return build_centerline(gps, empresaid=2)
