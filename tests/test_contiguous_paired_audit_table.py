"""Contracts on the committed paired-audit table.

The table exists to make one claim checkable: under contract C1 the LSTM, the
XGBoost baseline and persistence score the SAME cells, so restricting any of
them to the three-way intersection must not move its metrics. Audit §2.1 found
that restriction moving the old pipeline by 0.28–0.53 min — larger than most of
the margins being claimed on top of it.

These tests pin the measured bias and the retention that produces it, so a
regenerated artifact that drifts back toward separate populations fails here
instead of shipping a comparison nobody can attribute.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_paired_audit import (  # noqa: E402
    FRAMING_TOL_MIN,
    OUT_CSV,
    assert_no_framing_bias,
)
from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402

# The bias the audit measured on the old pipeline, in minutes. The new table has
# to come in far under it for the retraining to have bought anything.
OLD_PIPELINE_FRAMING_BIAS_MIN = 0.28

pytestmark = pytest.mark.skipif(
    not OUT_CSV.exists(),
    reason=f"{OUT_CSV.name} not generated — run src.build_contiguous_paired_audit",
)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    return pl.read_csv(OUT_CSV)


@pytest.fixture(scope="module")
def aggregate(table) -> pl.DataFrame:
    return table.filter(pl.col("direction") == "aggregate")


class TestCoverage:
    def test_every_cell_is_present(self, aggregate):
        cells = set(
            zip(aggregate.get_column("corridor"), aggregate.get_column("horizon"))
        )
        assert cells == {(c, h) for c in CORRIDORS for h in HORIZONS}

    def test_each_cell_reports_both_directions_and_the_aggregate(self, table):
        counts = table.group_by(["corridor", "horizon"]).len()
        assert counts.get_column("len").to_list() == [3] * counts.height
        assert set(table.get_column("direction")) == {"aggregate", "+1", "-1"}

    def test_directions_partition_the_aggregate(self, table):
        per_cell = table.group_by(["corridor", "horizon"]).agg(
            pl.col("n_paired")
            .filter(pl.col("direction") == "aggregate")
            .first()
            .alias("total"),
            pl.col("n_paired")
            .filter(pl.col("direction") != "aggregate")
            .sum()
            .alias("split"),
        )
        assert (
            per_cell.get_column("total") == per_cell.get_column("split")
        ).all()


class TestSharedPopulation:
    """C1 in numbers: the claim §2.1 said the old pipeline could not support."""

    def test_framing_bias_is_within_tolerance(self, table):
        assert_no_framing_bias(table)  # the builder's own gate, re-run on disk

    def test_framing_bias_is_orders_of_magnitude_below_the_old_one(self, table):
        worst = float(
            max(
                table.get_column("framing_delta_lstm").abs().max(),
                table.get_column("framing_delta_xgb").abs().max(),
            )
        )
        assert worst < OLD_PIPELINE_FRAMING_BIAS_MIN / 100

    def test_the_bias_never_exceeds_the_margin_it_would_explain(self, table):
        """§2.1's actual complaint: the framing gap dwarfed 7 of 8 margins."""
        margins = table.get_column("delta_lstm_xgb").abs().to_numpy()
        bias = table.get_column("framing_delta_lstm").abs().to_numpy()
        assert np.all(bias < margins)

    def test_retention_is_effectively_total(self, table):
        """A few LSTM cells lack an XGBoost counterpart at the vector tail; the
        share is small enough that no metric moves, which the framing delta
        confirms independently."""
        for col in ("retained_lstm_pct", "retained_xgb_pct"):
            assert table.get_column(col).min() > 99.85
            assert table.get_column(col).max() <= 100.0

    def test_every_dropped_row_is_explained_by_the_vector_width(self, table):
        """Retention below 100% must trace to the ``max_N`` asymmetry and nothing
        else. The LSTM is dimensioned on the global ``max_N`` while XGBoost stops
        at each direction's own p99, so where the widths agree the join must be
        exact — and where they differ, only the LSTM may hold extra rows.

        The check is per direction on purpose. On an aggregate row the width is
        the max over both directions, so E2's ``13 == 13`` hides that the gap
        lives entirely in dir ``-1`` — the same asymmetry, invisible at that
        level of aggregation.
        """
        per_direction = table.filter(pl.col("direction") != "aggregate")
        same_width = per_direction.filter(
            pl.col("max_pair_rank_lstm") == pl.col("max_pair_rank_xgb")
        )
        assert (same_width.get_column("retained_lstm_pct") == 100.0).all(), (
            "rows dropped without a vector-width difference to explain them"
        )
        assert (
            table.get_column("max_pair_rank_lstm")
            >= table.get_column("max_pair_rank_xgb")
        ).all(), "XGBoost predicted a slot the LSTM never emitted"

    def test_xgboost_is_never_the_side_that_loses_rows(self, table):
        """Its narrower vector makes it a subset, so it must retain everything."""
        assert (table.get_column("retained_xgb_pct") == 100.0).all()


class TestReportedQuantities:
    def test_deltas_reconstruct_from_the_paired_maes(self, table):
        lstm = table.get_column("mae_lstm_paired").to_numpy()
        persist = table.get_column("mae_persist_paired").to_numpy()
        xgb = table.get_column("mae_xgb_paired").to_numpy()
        assert np.allclose(lstm - persist, table.get_column("delta_lstm_persist"))
        assert np.allclose(lstm - xgb, table.get_column("delta_lstm_xgb"))
        assert np.allclose(xgb - persist, table.get_column("delta_xgb_persist"))

    def test_rmse_dominates_mae(self, table):
        """Jensen: a rebuild that swapped the two columns would show up here."""
        for model in ("lstm", "xgb", "persist"):
            assert (
                table.get_column(f"rmse_{model}_paired")
                >= table.get_column(f"mae_{model}_paired")
            ).all()

    def test_framing_delta_is_the_difference_it_claims_to_be(self, table):
        for model in ("lstm", "xgb"):
            paired = table.get_column(f"mae_{model}_paired").to_numpy()
            own = table.get_column(f"mae_{model}_own").to_numpy()
            assert np.allclose(
                paired - own, table.get_column(f"framing_delta_{model}").to_numpy()
            )


class TestTheFinding:
    """The headline, pinned on the population every claim must cite."""

    def test_persistence_wins_at_one_step_in_every_corridor(self, aggregate):
        h1 = aggregate.filter(pl.col("horizon") == 1)
        assert (h1.get_column("delta_lstm_persist") > 0).all()

    def test_the_learner_wins_from_five_steps_in_every_corridor(self, aggregate):
        long = aggregate.filter(pl.col("horizon") >= 5)
        assert (long.get_column("delta_lstm_persist") < 0).all()

    def test_the_advantage_grows_with_the_horizon(self, aggregate):
        for corridor, sub in aggregate.sort("horizon").group_by(
            "corridor", maintain_order=True
        ):
            deltas = sub.get_column("delta_lstm_persist").to_numpy()
            assert np.all(np.diff(deltas) < 0), f"{corridor}: {deltas}"

    def test_h3_is_the_transition_and_not_a_clean_win(self, aggregate):
        """A1 found the mean and the median disagreeing at h=3; here the two
        travel directions disagree in E4 and nearly do in E59."""
        h3 = aggregate.filter(pl.col("horizon") == 3)
        assert (h3.get_column("delta_lstm_persist") < 0).all()
        # ... but the margin is far smaller than at h=5.
        h5 = aggregate.filter(pl.col("horizon") == 5).sort("corridor")
        assert (
            h3.sort("corridor").get_column("delta_lstm_persist").abs()
            < h5.get_column("delta_lstm_persist").abs()
        ).all()

    def test_xgboost_reproduces_the_crossover(self, aggregate):
        """The crossover is a property of the problem, not of deep learning."""
        assert (
            aggregate.filter(pl.col("horizon") == 1).get_column("delta_xgb_persist")
            > 0
        ).all()
        assert (
            aggregate.filter(pl.col("horizon") >= 5).get_column("delta_xgb_persist")
            < 0
        ).all()

    def test_the_lstm_vs_xgboost_verdict_splits_by_corridor(self, aggregate):
        """Not a global winner — and the tuning budget is not level, so this is
        reported, not attributed to model class."""
        at_h10 = {
            corridor: delta
            for corridor, delta in zip(
                aggregate.filter(pl.col("horizon") == 10).get_column("corridor"),
                aggregate.filter(pl.col("horizon") == 10).get_column(
                    "delta_lstm_xgb"
                ),
            )
        }
        assert at_h10["E2"] > 0  # XGBoost ahead in E2
        assert at_h10["E59"] < 0  # LSTM ahead in E59
        assert at_h10["E4"] < 0
