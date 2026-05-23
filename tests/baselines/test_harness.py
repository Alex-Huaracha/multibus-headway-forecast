"""TDD tests for src/baselines/harness.py — Fase 3.

Tests:
    test_harness_output_schema     — AC-CSV-1, AC-CSV-2: columns, dtypes, direction as string
    test_harness_row_count         — AC-CSV-2: 36 rows per corridor, 72 for two corridors
    test_harness_deterministic     — AC-B3-5 / AC-CSV-3: two identical calls produce equal output

Design (design §6):
    evaluate_corridor(headways, corridor_name) runs:
        split_temporal → winsorize_train_p99 → predict_b0/b1/b2(w5,w10,w15)/b3
        → filter test rows → compute MAE + RMSE per (direction, baseline)
        → return tidy long DataFrame.

    42 rows per corridor (3 directions × 7 baselines × 2 metrics).
    direction column is Utf8 (strings: "-1", "+1", "aggregate").
    val rows are NEVER used.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from tests.fixtures.headways_factory import make_headways_fixture


# ---------------------------------------------------------------------------
# Minimal fixture: enough data for all baselines to produce predictions
# ---------------------------------------------------------------------------

def _make_corridor_frame() -> pl.DataFrame:
    """Return a synthetic headways frame sufficient for harness evaluation.

    Train: 20 rows per slot (well above B2 w=15 window).
    Test:  5 rows per slot.
    Two directions (-1, +1), one pair_rank → 2 slots.
    All delta_t_min values are non-null to keep math predictable.
    """
    n_train = 20
    n_test = 5

    train_dates = [date(2023, 11, 1 + i) for i in range(n_train)]
    test_dates = [date(2024, 2, 10 + i) for i in range(n_test)]

    # Two slots: (direction=-1, pair_rank=1) and (direction=+1, pair_rank=1)
    slot_a = (-1, 1)
    slot_b = (1, 1)
    # Non-null increasing values so predictions are valid
    vals_a = list(range(1, n_train + n_test + 1))     # 1..25
    vals_b = list(range(2, n_train + n_test + 2))     # 2..26

    delta_map = {
        slot_a: [float(v) for v in vals_a],
        slot_b: [float(v) for v in vals_b],
    }

    return make_headways_fixture(
        empresaid=2,
        train_dates=train_dates,
        test_dates=test_dates,
        delta_values_per_slot=delta_map,
    )


class TestEvaluateCorridorSchema:
    """Output schema: columns, types, direction as string, no extra columns."""

    def test_harness_output_schema(self):
        """AC-CSV-1: column names are exactly [corridor, direction, baseline, metric, value]."""
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result = evaluate_corridor(df, "E2")

        assert result.columns == ["corridor", "direction", "baseline", "metric", "value"], (
            f"Unexpected columns: {result.columns}"
        )

        # Type checks per design §6 output schema
        assert result.schema["corridor"] == pl.Utf8
        assert result.schema["direction"] == pl.Utf8
        assert result.schema["baseline"] == pl.Utf8
        assert result.schema["metric"] == pl.Utf8
        assert result.schema["value"] == pl.Float64

        # direction values must be strings: "-1", "+1", "aggregate"
        directions = set(result["direction"].unique().to_list())
        assert directions == {"-1", "+1", "aggregate"}, (
            f"Unexpected direction values: {directions}"
        )

    def test_harness_row_count(self):
        """AC-CSV-2: 42 rows per corridor (3 directions × 7 baselines × 2 metrics)."""
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result = evaluate_corridor(df, "E2")

        assert len(result) == 42, (
            f"Expected 42 rows per corridor, got {len(result)}"
        )

        # Verify the corridor name propagated correctly
        assert result["corridor"].unique().to_list() == ["E2"]

        # Verify metric values are only MAE and RMSE
        metrics = set(result["metric"].unique().to_list())
        assert metrics == {"MAE", "RMSE"}, f"Unexpected metrics: {metrics}"

        # Verify baselines
        baselines = set(result["baseline"].unique().to_list())
        expected_baselines = {"B0", "B1", "B2_w5", "B2_w10", "B2_w15", "B3", "B4_HA"}
        assert baselines == expected_baselines, f"Unexpected baselines: {baselines}"

    def test_harness_deterministic(self):
        """AC-B3-5 / B3-VAL-UNUSED: two identical calls produce equal DataFrames."""
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result1 = evaluate_corridor(df, "E2")
        result2 = evaluate_corridor(df, "E2")

        # Sort both by the same key before comparing.
        sort_key = ["corridor", "direction", "baseline", "metric"]
        r1 = result1.sort(sort_key)
        r2 = result2.sort(sort_key)

        # Compare value column numerically (no NaN expected in a well-formed fixture)
        v1 = r1["value"].to_list()
        v2 = r2["value"].to_list()
        assert v1 == v2, "evaluate_corridor must be deterministic"

        # Structural equality on non-numeric columns
        for col in ["corridor", "direction", "baseline", "metric"]:
            assert r1[col].to_list() == r2[col].to_list(), (
                f"Column {col} differs between two calls"
            )
