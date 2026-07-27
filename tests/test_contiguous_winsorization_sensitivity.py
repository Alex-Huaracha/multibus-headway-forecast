"""Contracts on the winsorization sensitivity table — audit pending #3.

This is the paper's most attackable seam: the training contract clips the top 1%
of headways, and a reviewer will ask whether the reported advantage is an
artifact of that ceiling. The table answers it by rescoring every verdict against
raw un-clipped targets, and these tests pin the answer so a later rebuild cannot
quietly weaken it.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_contiguous_vector_metrics import MODELS, VECTOR_CSV  # noqa: E402
from src.build_contiguous_winsorization_sensitivity import OUT_CSV  # noqa: E402

pytestmark = pytest.mark.skipif(
    not OUT_CSV.exists(),
    reason=f"{OUT_CSV.name} not generated — run src.build_contiguous_winsorization_sensitivity",
)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    return pl.read_csv(OUT_CSV)


@pytest.fixture(scope="module")
def learners(table) -> pl.DataFrame:
    return table.filter(pl.col("model") != "Persistence")


class TestCoverage:
    def test_one_row_per_model_and_cell(self, table):
        assert set(
            zip(
                table.get_column("model"),
                table.get_column("corridor"),
                table.get_column("horizon"),
            )
        ) == {(name, c, h) for name, _ in MODELS for c in CORRIDORS for h in HORIZONS}

    def test_the_ceiling_is_one_value_per_corridor(self, table):
        """It is a train-only scalar, so it cannot vary by horizon or model."""
        spread = table.group_by("corridor").agg(
            pl.col("p99_threshold").n_unique().alias("distinct")
        )
        assert spread.get_column("distinct").to_list() == [1] * spread.height


class TestClippingFootprint:
    def test_the_ceiling_clips_about_one_percent(self, table):
        pct = table.get_column("pct_clipped_targets")
        assert 0.5 < pct.min()
        assert pct.max() < 1.5

    def test_the_thresholds_match_the_documented_values(self, table):
        """Audit §2.3 recorded these; a drift would mean the split moved."""
        expected = {"E2": 28.467923, "E59": 27.996949, "E4": 29.098441}
        got = dict(
            zip(
                table.get_column("corridor"),
                table.get_column("p99_threshold"),
            )
        )
        for corridor, value in expected.items():
            assert got[corridor] == pytest.approx(value, abs=1e-5)


class TestTheScalarVerdictSurvives:
    """The answer to pending #3."""

    def test_no_sign_flips_when_the_ceiling_comes_off(self, learners):
        clipped = learners.get_column("delta_vs_persist_clipped").to_numpy()
        raw = learners.get_column("delta_vs_persist_raw_fair").to_numpy()
        assert np.all(np.sign(clipped) == np.sign(raw))

    def test_the_margin_barely_moves(self, learners):
        """Stated in minutes, not as a ratio. The relative form is unstable
        exactly where the margin is near zero — E2 h=1's margin is 0.067 min, so
        a 0.004 min shift reads as 6% and says nothing about robustness."""
        clipped = learners.get_column("delta_vs_persist_clipped").to_numpy()
        raw = learners.get_column("delta_vs_persist_raw_fair").to_numpy()
        assert np.abs(raw - clipped).max() < 0.01

    def test_the_relative_shift_is_negligible_where_the_margin_is_real(self, learners):
        """Under 2% on every margin above 0.1 min; the observed worst case is
        1.8%, on E59 h=3, which is the narrowest margin in that set."""
        wide = learners.filter(pl.col("delta_vs_persist_clipped").abs() > 0.1)
        clipped = wide.get_column("delta_vs_persist_clipped").to_numpy()
        raw = wide.get_column("delta_vs_persist_raw_fair").to_numpy()
        assert (np.abs(raw - clipped) / np.abs(clipped)).max() < 0.02

    def test_removing_the_ceiling_does_not_favour_persistence(self, learners):
        """The reviewer's worry is that the ceiling manufactured the advantage.
        It runs the other way: without it every learner's position improves,
        marginally."""
        clipped = learners.get_column("delta_vs_persist_clipped").to_numpy()
        raw = learners.get_column("delta_vs_persist_raw_fair").to_numpy()
        assert np.all(raw <= clipped + 1e-9)

    def test_the_crossover_is_unchanged(self, table):
        lstm = table.filter(pl.col("model") == "LSTM")
        h1 = lstm.filter(pl.col("horizon") == 1)
        assert (h1.get_column("delta_vs_persist_raw_fair") > 0).all()
        long = lstm.filter(pl.col("horizon") >= 5)
        assert (long.get_column("delta_vs_persist_raw_fair") < 0).all()

    def test_the_transition_cells_stay_non_significant(self, table):
        """E2 h=1 and E4 h=3 lost significance under clustering; the raw rescoring
        must not resurrect them, or the ceiling WAS doing work."""
        lstm = table.filter(pl.col("model") == "LSTM")
        for corridor, horizon in (("E2", 1), ("E4", 3)):
            cell = lstm.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            assert cell.get_column("dm_p_raw_fair").item() > 0.05


class TestFairPersistence:
    def test_the_ceiling_helps_persistence_rather_than_handicapping_it(self, table):
        """Counter-intuitive, and it is the thesis arriving from a second door.

        Persistence propagates the last observation. Clipping an extreme 35 min
        down to 28.5 moves that prediction toward the bulk of targets, so MAE
        improves. Winsorization is a shrinkage, and MAE rewards shrinkage — the
        same mechanism that makes the learners flatten the vector, reached here
        through preprocessing instead of through a loss function.
        """
        persistence = table.filter(pl.col("model") == "Persistence")
        assert (
            persistence.get_column("mae_vs_raw_target_fair")
            > persistence.get_column("mae_vs_raw_target_as_produced")
        ).all()

    def test_the_help_is_small_enough_not_to_drive_any_verdict(self, table):
        persistence = table.filter(pl.col("model") == "Persistence")
        shift = (
            persistence.get_column("mae_vs_raw_target_fair").to_numpy()
            - persistence.get_column("mae_vs_raw_target_as_produced").to_numpy()
        )
        assert shift.max() < 0.01

    def test_the_learners_have_no_fair_variant(self, learners):
        """Only persistence's prediction is recomputable without a ceiling; for
        the learners the two columns are the same number by construction."""
        assert np.allclose(
            learners.get_column("mae_vs_raw_target_fair").to_numpy(),
            learners.get_column("mae_vs_raw_target_as_produced").to_numpy(),
        )


class TestTheVectorFindingSurvives:
    """The ceiling clips the GAP tail, not the bunching tail, so it should be
    inert here — and confirming that is what turns an expectation into evidence.
    """

    @pytest.fixture(scope="class")
    def winsorized(self) -> pl.DataFrame:
        if not VECTOR_CSV.exists():
            pytest.skip("vector metrics table not generated")
        return pl.read_csv(VECTOR_CSV)

    def test_bunching_f1_is_unchanged_against_raw_targets(self, table, winsorized):
        merged = table.select(
            ["model", "corridor", "horizon", "raw_bunching_f1"]
        ).join(
            winsorized.select(["model", "corridor", "horizon", "bunching_f1"]),
            on=["model", "corridor", "horizon"],
            how="inner",
        )
        assert merged.height == table.height
        difference = (
            merged.get_column("raw_bunching_f1").to_numpy()
            - merged.get_column("bunching_f1").to_numpy()
        )
        assert np.abs(difference).max() < 0.005

    def test_persistence_still_wins_bunching_everywhere(self, table):
        best = (
            table.sort("raw_bunching_f1", descending=True)
            .group_by(["corridor", "horizon"], maintain_order=True)
            .first()
        )
        assert set(best.get_column("model")) == {"Persistence"}

    def test_learner_recall_still_collapses_with_the_horizon(self, table):
        lstm = table.filter(pl.col("model") == "LSTM").sort("horizon")
        for corridor, sub in lstm.group_by("corridor", maintain_order=True):
            recall = sub.get_column("raw_bunching_recall").to_numpy()
            assert np.all(np.diff(recall) < 0), f"{corridor}: {recall}"
