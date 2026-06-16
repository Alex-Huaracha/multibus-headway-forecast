"""Tests for src/evaluation/degradation.py — multi-horizon consolidation + curve.

The multi-horizon experiment (Fase 6.5) leaves one CSV per model/horizon plus a
baselines CSV in docs/resultados/csv-multihorizon/, all sharing the schema
    corridor,direction,baseline,metric,value,horizon

This module consolidates them into a single tidy DataFrame and reshapes it into a
model x horizon degradation table (the central paper figure's data source).

Acceptance criteria:
    AC-LOAD-1  load_results concatenates every CSV in the dir into one frame
    AC-LOAD-2  load_results preserves the 6-column schema and all rows
    AC-LOAD-3  load_results raises if the dir has no CSVs
    AC-LOAD-4  load_results raises if a CSV has the wrong schema
    AC-CURVE-1 degradation_table pivots horizons to columns, one row per model
    AC-CURVE-2 degradation_table filters by metric and direction
    AC-CURVE-3 degradation_table sorts horizons ascending
    AC-CURVE-4 degradation_table raises on unknown metric/direction (empty result)
"""
from __future__ import annotations

import polars as pl
import pytest

from src.evaluation.degradation import load_results, degradation_table

SCHEMA = ["corridor", "direction", "baseline", "metric", "value", "horizon"]


def _write_csv(path, rows):
    pl.DataFrame(rows, schema=SCHEMA, orient="row").write_csv(path)


@pytest.fixture
def results_dir(tmp_path):
    """Two CSVs mimicking a baselines file and a single DL model/horizon file."""
    _write_csv(
        tmp_path / "baselines_results_multih.csv",
        [
            ("E2", "aggregate", "B1", "MAE", 4.78, 1),
            ("E2", "aggregate", "B1", "MAE", 5.50, 3),
            ("E2", "aggregate", "B1", "RMSE", 6.55, 1),
            ("E59", "aggregate", "B1", "MAE", 3.80, 1),
        ],
    )
    _write_csv(
        tmp_path / "lstm_results_h3.csv",
        [
            ("E2", "aggregate", "LSTM", "MAE", 5.10, 3),
            ("E2", "+1", "LSTM", "MAE", 4.20, 3),
            ("E2", "aggregate", "LSTM", "RMSE", 6.90, 3),
        ],
    )
    return tmp_path


# --- AC-LOAD ---------------------------------------------------------------
class TestLoadResults:
    def test_concatenates_all_csvs(self, results_dir):
        """AC-LOAD-1/2: every row from both CSVs present, schema preserved."""
        df = load_results(results_dir)
        assert df.columns == SCHEMA
        assert df.height == 7  # 4 baseline rows + 3 lstm rows
        assert set(df["baseline"].unique()) == {"B1", "LSTM"}

    def test_horizon_is_integer(self, results_dir):
        """AC-LOAD-2: horizon column is integer-typed for numeric sorting."""
        df = load_results(results_dir)
        assert df["horizon"].dtype in (pl.Int64, pl.Int32)

    def test_empty_dir_raises(self, tmp_path):
        """AC-LOAD-3: no CSVs -> ValueError."""
        with pytest.raises(ValueError, match="no CSV"):
            load_results(tmp_path)

    def test_wrong_schema_raises(self, tmp_path):
        """AC-LOAD-4: a CSV missing required columns -> ValueError."""
        pl.DataFrame({"foo": [1], "bar": [2]}).write_csv(tmp_path / "bad.csv")
        with pytest.raises(ValueError, match="schema"):
            load_results(tmp_path)


# --- AC-CURVE --------------------------------------------------------------
class TestDegradationTable:
    def test_pivots_horizons_to_columns(self, results_dir):
        """AC-CURVE-1/2: MAE/E2/aggregate -> one row per model, horizons as cols."""
        df = load_results(results_dir)
        table = degradation_table(df, metric="MAE", direction="aggregate", corridor="E2")
        # models present at E2/aggregate/MAE: B1 (h1,h3) and LSTM (h3)
        assert set(table["baseline"]) == {"B1", "LSTM"}
        b1 = table.filter(pl.col("baseline") == "B1")
        assert b1["h1"].item() == pytest.approx(4.78)
        assert b1["h3"].item() == pytest.approx(5.50)
        lstm = table.filter(pl.col("baseline") == "LSTM")
        assert lstm["h3"].item() == pytest.approx(5.10)
        assert lstm["h1"].item() is None  # LSTM has no h1 in this fixture

    def test_horizons_sorted_ascending(self, results_dir):
        """AC-CURVE-3: horizon columns appear in ascending order."""
        df = load_results(results_dir)
        table = degradation_table(df, metric="MAE", direction="aggregate", corridor="E2")
        horizon_cols = [c for c in table.columns if c.startswith("h")]
        order = [int(c[1:]) for c in horizon_cols]
        assert order == sorted(order)

    def test_unknown_filter_raises(self, results_dir):
        """AC-CURVE-4: a metric/direction with no rows -> ValueError."""
        df = load_results(results_dir)
        with pytest.raises(ValueError, match="no rows"):
            degradation_table(df, metric="MAPE", direction="aggregate", corridor="E2")
