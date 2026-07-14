"""Tests for src.data.context_features — AC-CTX-1..6, AC-LEAK-2.

Strict TDD: this file is the RED commit. src/data/context_features.py does not exist yet;
all tests fail with ImportError. Run: uv run pytest tests/data/test_context_features.py -q
"""
from __future__ import annotations

import math
import warnings
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest


def _make_single_ts_df(dt: datetime) -> pl.DataFrame:
    """Single-row DataFrame with one timestamp — minimal fixture for context tests."""
    return pl.DataFrame({
        "t": [dt],
    }).with_columns(pl.col("t").cast(pl.Datetime("us")))


class TestEncodeContext:
    """AC-CTX-1..5: encode_context adds 5 cyclical + atypical columns correctly."""

    def test_hour_sin_cos_continuity_at_midnight(self) -> None:
        """AC-CTX-1: midnight Monday → hour_sin=0, hour_cos=1, dow_sin=0, dow_cos=1.

        sin(2π * 0/24) = 0.0, cos(2π * 0/24) = 1.0.
        sin(2π * 0/7) = 0.0, cos(2π * 0/7) = 1.0.
        Monday = weekday 0 in Python datetime.weekday().
        """
        from src.data.context_features import encode_context

        midnight_monday = datetime(2023, 11, 6, 0, 0, 0)  # 2023-11-06 is a Monday
        df = _make_single_ts_df(midnight_monday)
        result = encode_context(df)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns

        row = result.row(0, named=True)
        assert abs(row["hour_sin"] - 0.0) < 1e-9, f"hour_sin={row['hour_sin']}"
        assert abs(row["hour_cos"] - 1.0) < 1e-9, f"hour_cos={row['hour_cos']}"
        assert abs(row["dow_sin"] - 0.0) < 1e-9, f"dow_sin={row['dow_sin']}"
        assert abs(row["dow_cos"] - 1.0) < 1e-9, f"dow_cos={row['dow_cos']}"

    def test_hour_sin_cos_at_noon(self) -> None:
        """AC-CTX-1 extension: noon → hour_sin=sin(π)≈0, hour_cos=cos(π)=-1."""
        from src.data.context_features import encode_context

        noon = datetime(2023, 11, 6, 12, 0, 0)
        df = _make_single_ts_df(noon)
        result = encode_context(df)
        row = result.row(0, named=True)

        # hour=12, period=24: 2π*12/24 = π → sin(π)≈0, cos(π)=-1
        assert abs(row["hour_sin"] - math.sin(math.pi)) < 1e-9
        assert abs(row["hour_cos"] - math.cos(math.pi)) < 1e-9

    def test_emits_exactly_five_columns_with_names(self) -> None:
        """AC-CTX-2: encode_context adds exactly 5 named context columns.

        Names must match CONTEXT_FEATURE_NAMES constant.
        """
        from src.data.context_features import CONTEXT_FEATURE_NAMES, encode_context

        df = _make_single_ts_df(datetime(2023, 11, 6, 8, 0, 0))
        result = encode_context(df)

        for name in CONTEXT_FEATURE_NAMES:
            assert name in result.columns, f"Missing column: {name}"
        assert len(CONTEXT_FEATURE_NAMES) == 5

    def test_atypical_flag_one_when_date_in_set(self) -> None:
        """AC-CTX-5: atypical_flag=1.0 when the timestamp date is in atypical_dates."""
        from src.data.context_features import encode_context

        target_date = date(2023, 11, 6)
        dt = datetime(target_date.year, target_date.month, target_date.day, 8, 0, 0)
        df = _make_single_ts_df(dt)

        result = encode_context(df, atypical_dates={target_date})
        row = result.row(0, named=True)
        assert row["atypical_flag"] == 1.0, f"Expected 1.0, got {row['atypical_flag']}"

    def test_atypical_flag_zero_when_date_absent(self) -> None:
        """AC-CTX-4: atypical_flag=0.0 when the timestamp date is NOT in atypical_dates."""
        from src.data.context_features import encode_context

        other_date = date(2023, 11, 7)
        dt = datetime(2023, 11, 6, 8, 0, 0)
        df = _make_single_ts_df(dt)

        result = encode_context(df, atypical_dates={other_date})
        row = result.row(0, named=True)
        assert row["atypical_flag"] == 0.0, f"Expected 0.0, got {row['atypical_flag']}"

    def test_atypical_flag_zero_when_set_is_none(self) -> None:
        """AC-CTX-3 + DL-2: atypical_flag defaults to 0.0 when atypical_dates is None."""
        from src.data.context_features import encode_context

        dt = datetime(2023, 11, 6, 8, 0, 0)
        df = _make_single_ts_df(dt)

        result = encode_context(df, atypical_dates=None)
        row = result.row(0, named=True)
        assert row["atypical_flag"] == 0.0

    def test_no_torch_import(self) -> None:
        """AC-CTX-6: context_features module has zero torch imports."""
        import inspect
        import src.data.context_features as cmod

        source = inspect.getsource(cmod)
        assert "import torch" not in source
        assert "from torch" not in source


class TestLoadAtypicalDays:
    """AC-CTX-3 + DL-2: load_atypical_days graceful fallback."""

    def test_returns_empty_when_path_is_none(self) -> None:
        """AC-CTX-3: load_atypical_days(None) returns empty set, no exception."""
        from src.data.context_features import load_atypical_days

        result = load_atypical_days(None)
        assert isinstance(result, set)
        assert len(result) == 0

    def test_returns_empty_when_file_missing_with_warning(self, tmp_path: Path) -> None:
        """AC-CTX-3 + DL-2: load_atypical_days(missing_path) returns empty set.

        A warning must be emitted (not an exception).
        """
        from src.data.context_features import load_atypical_days

        nonexistent = tmp_path / "atypical_days.csv"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_atypical_days(nonexistent)

        assert isinstance(result, set)
        assert len(result) == 0
        # A warning must have been issued.
        assert len(caught) >= 1

    def test_returns_dates_when_file_exists(self, tmp_path: Path) -> None:
        """AC-CTX-4: load_atypical_days(path) returns set[date] from CSV.

        CSV must have at least a `date` column with ISO-8601 date strings.
        """
        from src.data.context_features import load_atypical_days

        csv_path = tmp_path / "atypical_days.csv"
        csv_path.write_text("date\n2023-11-06\n2023-12-25\n")

        result = load_atypical_days(csv_path)
        assert isinstance(result, set)
        assert date(2023, 11, 6) in result
        assert date(2023, 12, 25) in result
        assert len(result) == 2

    def test_returns_dates_when_day_column(self, tmp_path: Path) -> None:
        """AC-CTX-4: `day` column is accepted as the date column.

        The frozen `atypical_days.csv` produced by the 02-eda-corridors kernel
        names its date column `day` (not `date`); the loader must parse it.
        """
        from src.data.context_features import load_atypical_days

        csv_path = tmp_path / "atypical_days.csv"
        csv_path.write_text(
            "empresaid,day,records\n2,2023-10-28,53115\n2,2023-11-05,61527\n"
        )

        result = load_atypical_days(csv_path)
        assert result == {date(2023, 10, 28), date(2023, 11, 5)}
