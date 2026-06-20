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
        """AC-CSV-2: 48 rows per corridor (3 directions × 8 baselines × 2 metrics),
        with the fitted baseline B5_XGB included by default."""
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result = evaluate_corridor(df, "E2")

        assert len(result) == 48, (
            f"Expected 48 rows per corridor (8 baselines), got {len(result)}"
        )

        # Verify the corridor name propagated correctly
        assert result["corridor"].unique().to_list() == ["E2"]

        # Verify metric values are only MAE and RMSE
        metrics = set(result["metric"].unique().to_list())
        assert metrics == {"MAE", "RMSE"}, f"Unexpected metrics: {metrics}"

        # Verify baselines (now includes the fitted B5_XGB).
        baselines = set(result["baseline"].unique().to_list())
        expected_baselines = {
            "B0", "B1", "B2_w5", "B2_w10", "B2_w15", "B3", "B4_HA", "B5_XGB"
        }
        assert baselines == expected_baselines, f"Unexpected baselines: {baselines}"

    def test_formulaic_only_excludes_fitted(self):
        """include_fitted=False → 42 rows, no B5_XGB (formulaic-only path)."""
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result = evaluate_corridor(df, "E2", include_fitted=False)

        assert len(result) == 42, (
            f"Expected 42 rows without the fitted baseline, got {len(result)}"
        )
        baselines = set(result["baseline"].unique().to_list())
        assert "B5_XGB" not in baselines, "B5_XGB must be absent when include_fitted=False"

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


# ===========================================================================
# Fase 6.5 Ola 3: horizon propagation to B2 and B3 in evaluate_corridor
# ===========================================================================

class TestEvaluateCorridorHorizon:
    """evaluate_corridor must thread horizon to B1, B2, and B3 (not B0/B4_HA)."""

    def test_horizon_propagates_to_b2_b3(self):
        """horizon=3 must produce different B2/B3 values than horizon=1.

        With 20 train rows and 5 test rows, horizon=3 shifts predictions
        by 2 extra steps relative to horizon=1 — metric values must differ
        for at least B2 and B3 baselines.
        """
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result_h1 = evaluate_corridor(df, "E2", horizon=1)
        result_h3 = evaluate_corridor(df, "E2", horizon=3)

        sort_key = ["corridor", "direction", "baseline", "metric"]
        r1 = result_h1.sort(sort_key)
        r3 = result_h3.sort(sort_key)

        # B2_w5 MAE must differ between horizon=1 and horizon=3.
        b2_mae_h1 = (
            r1.filter(
                (pl.col("baseline") == "B2_w5")
                & (pl.col("direction") == "aggregate")
                & (pl.col("metric") == "MAE")
            )["value"].to_list()
        )
        b2_mae_h3 = (
            r3.filter(
                (pl.col("baseline") == "B2_w5")
                & (pl.col("direction") == "aggregate")
                & (pl.col("metric") == "MAE")
            )["value"].to_list()
        )
        assert b2_mae_h1 != b2_mae_h3, (
            f"B2_w5 MAE should differ between h=1 and h=3: h1={b2_mae_h1}, h3={b2_mae_h3}"
        )

        # B3 MAE must also differ.
        b3_mae_h1 = (
            r1.filter(
                (pl.col("baseline") == "B3")
                & (pl.col("direction") == "aggregate")
                & (pl.col("metric") == "MAE")
            )["value"].to_list()
        )
        b3_mae_h3 = (
            r3.filter(
                (pl.col("baseline") == "B3")
                & (pl.col("direction") == "aggregate")
                & (pl.col("metric") == "MAE")
            )["value"].to_list()
        )
        assert b3_mae_h1 != b3_mae_h3, (
            f"B3 MAE should differ between h=1 and h=3: h1={b3_mae_h1}, h3={b3_mae_h3}"
        )

    def test_b0_b4_unchanged_by_horizon(self):
        """B0 and B4_HA values must be identical regardless of horizon.

        B0 is a slot-mean constant; B4_HA is a per-hour mean.
        Neither depends on the prediction horizon.
        """
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result_h1 = evaluate_corridor(df, "E2", horizon=1)
        result_h3 = evaluate_corridor(df, "E2", horizon=3)

        sort_key = ["corridor", "direction", "baseline", "metric"]
        r1 = result_h1.sort(sort_key)
        r3 = result_h3.sort(sort_key)

        for baseline_name in ("B0", "B4_HA"):
            vals_h1 = (
                r1.filter(pl.col("baseline") == baseline_name)["value"].to_list()
            )
            vals_h3 = (
                r3.filter(pl.col("baseline") == baseline_name)["value"].to_list()
            )
            assert vals_h1 == vals_h3, (
                f"{baseline_name} values must be identical across horizons: "
                f"h1={vals_h1}, h3={vals_h3}"
            )

    def test_horizon_call_returns_48_rows(self):
        """evaluate_corridor with horizon=3 must still return 48 rows (with B5_XGB)."""
        from src.baselines.harness import evaluate_corridor

        df = _make_corridor_frame()
        result = evaluate_corridor(df, "E2", horizon=3)
        assert len(result) == 48, f"Expected 48 rows, got {len(result)}"
