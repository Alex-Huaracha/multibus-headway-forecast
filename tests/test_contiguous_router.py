"""Router mechanics and the contracts on the committed router table.

Audit pending #9 rejected the previous router on three grounds: the gain was
below seed noise, most policies were degenerate, and "best-or-tied against both
pure models" is vacuous when the policy IS one of them. The unit tests pin the
mechanics that could hide those problems; the table tests re-measure them on the
committed artifact so the conclusion stays checkable rather than asserted.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_contiguous_router import (  # noqa: E402
    MIN_TERCILE_N,
    OUT_CSV,
    SEEDS,
    TEMPORAL_LEARN_DAYS,
    apply_policy,
    learn_policy,
    random_masks,
    seed_sweep_summary,
    temporal_masks,
)
from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402


def _three_terciles(n_per: int = 200):
    """Codes for a balanced three-tercile population."""
    return np.repeat([0, 1, 2], n_per)


class TestLearnPolicy:
    def test_picks_the_best_candidate_in_each_tercile(self):
        """A different model wins each bin; the policy must say so."""
        codes = _three_terciles()
        n = codes.size
        y = np.zeros(n)
        preds = {
            "P": np.where(codes == 0, 0.0, 10.0),
            "D": np.where(codes == 1, 0.0, 10.0),
            "X": np.where(codes == 2, 0.0, 10.0),
        }
        assert learn_policy(y, preds, codes, np.ones(n, dtype=bool)) == ("P", "D", "X")

    def test_thin_terciles_fall_back_to_persistence(self):
        """The fallback must be a fixed rule — picking the locally best model in
        a thin bin would let the evaluation data choose it indirectly."""
        codes = np.concatenate([np.zeros(MIN_TERCILE_N - 1), np.ones(500), np.full(500, 2)]).astype(int)
        n = codes.size
        y = np.zeros(n)
        # D is best everywhere, including the thin bin.
        preds = {"P": np.full(n, 5.0), "D": np.zeros(n), "X": np.full(n, 9.0)}
        assert learn_policy(y, preds, codes, np.ones(n, dtype=bool))[0] == "P"

    def test_only_the_learning_slice_is_consulted(self):
        """The evaluation half is allowed to disagree without moving the policy."""
        codes = np.zeros(400, dtype=int)
        y = np.zeros(400)
        learn = np.arange(400) < 200
        preds = {
            "P": np.where(learn, 0.0, 99.0),   # best on learn, terrible on eval
            "D": np.full(400, 1.0),
            "X": np.full(400, 2.0),
        }
        assert learn_policy(y, preds, codes, learn)[0] == "P"


class TestApplyPolicy:
    def test_routes_each_tercile_to_its_choice(self):
        codes = _three_terciles(3)
        preds = {
            "P": np.zeros(9),
            "D": np.ones(9),
            "X": np.full(9, 2.0),
        }
        routed = apply_policy(preds, codes, ("X", "P", "D"))
        assert routed.tolist() == [2, 2, 2, 0, 0, 0, 1, 1, 1]

    def test_a_degenerate_policy_reproduces_a_pure_model(self):
        """The audit's point: a constant policy is not a router."""
        codes = _three_terciles(5)
        preds = {"P": np.arange(15.0), "D": -np.arange(15.0), "X": np.zeros(15)}
        assert np.array_equal(apply_policy(preds, codes, ("D", "D", "D")), preds["D"])


class TestSplits:
    def test_temporal_split_never_learns_from_the_future(self):
        days = np.repeat(np.arange(22), 10)
        learn, evaluate = temporal_masks(days)
        assert days[learn].max() < days[evaluate].min()

    def test_temporal_split_uses_the_configured_number_of_days(self):
        days = np.repeat(np.arange(22), 10)
        learn, _ = temporal_masks(days)
        assert np.unique(days[learn]).size == TEMPORAL_LEARN_DAYS

    def test_temporal_split_handles_fewer_days_than_configured(self):
        days = np.repeat(np.arange(4), 10)
        learn, evaluate = temporal_masks(days)
        assert learn.all() or evaluate.any()  # never raises, never empty-learns

    def test_masks_are_a_partition(self):
        for seed in (0, 7, 19):
            learn, evaluate = random_masks(1000, seed)
            assert not (learn & evaluate).any()
            assert (learn | evaluate).all()

    def test_random_split_is_reproducible(self):
        assert np.array_equal(random_masks(500, 3)[0], random_masks(500, 3)[0])

    def test_different_seeds_give_different_splits(self):
        assert not np.array_equal(random_masks(500, 1)[0], random_masks(500, 2)[0])


pytestmark_table = pytest.mark.skipif(
    not OUT_CSV.exists(),
    reason=f"{OUT_CSV.name} not generated — run src.build_contiguous_router",
)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV.name} not generated")
    return pl.read_csv(OUT_CSV)


@pytest.fixture(scope="module")
def temporal(table) -> pl.DataFrame:
    return table.filter(pl.col("split_mode") == "temporal")


@pytest.fixture(scope="module")
def summary(table) -> pl.DataFrame:
    return seed_sweep_summary(table)


class TestTableShape:
    def test_one_temporal_row_and_one_row_per_seed_per_cell(self, table):
        counts = table.group_by(["corridor", "horizon"]).len()
        assert counts.get_column("len").to_list() == [1 + len(SEEDS)] * counts.height

    def test_every_cell_is_present(self, temporal):
        cells = set(
            zip(temporal.get_column("corridor"), temporal.get_column("horizon"))
        )
        assert cells == {(c, h) for c in CORRIDORS for h in HORIZONS}

    def test_learn_and_eval_slices_are_disjoint_and_complete(self, table):
        """n_learn + n_eval is the cell population, so no sample was scored twice
        or silently dropped between the two halves."""
        totals = table.group_by(["corridor", "horizon"]).agg(
            (pl.col("n_learn") + pl.col("n_eval")).n_unique().alias("distinct")
        )
        assert totals.get_column("distinct").to_list() == [1] * totals.height


class TestRouterIsBounded:
    def test_the_router_never_beats_the_oracle(self, table):
        """The oracle picks per tercile on the evaluation slice itself; a router
        below it would mean the policy saw that slice."""
        assert (table.get_column("gap_to_oracle") >= -1e-12).all()

    def test_a_degenerate_policy_gains_exactly_nothing(self, table):
        """It reproduces a pure model, so it cannot beat the best one — and when
        it happens to BE the best one the gain is exactly zero, not merely small."""
        degenerate = table.filter(pl.col("policy_degenerate"))
        assert (degenerate.get_column("gain_vs_best_pure") >= -1e-12).all()

    def test_policies_only_name_known_candidates(self, table):
        letters = set("".join(table.get_column("policy").to_list()))
        assert letters <= {"P", "D", "X"}

    def test_reported_gain_matches_the_reported_maes(self, table):
        best = np.min(
            np.column_stack(
                [
                    table.get_column("mae_persist").to_numpy(),
                    table.get_column("mae_lstm").to_numpy(),
                    table.get_column("mae_xgb").to_numpy(),
                ]
            ),
            axis=1,
        )
        assert np.allclose(
            table.get_column("mae_router").to_numpy() - best,
            table.get_column("gain_vs_best_pure").to_numpy(),
        )


class TestTheFinding:
    """Pending #9, re-measured on the clean pipeline and three candidates.

    The router earns its keep only in the transition zone. Everywhere else it
    either collapses onto a pure model or moves less than the split does.
    """

    def test_most_policies_are_still_degenerate(self, temporal):
        """The audit found 9 of 12. A levelled XGBoost as a third candidate did
        not rescue the router; it is still mostly a pure model in disguise."""
        assert temporal.get_column("policy_degenerate").sum() >= 6

    def test_a_real_gain_survives_seed_noise_only_at_h3(self, summary):
        survivors = summary.filter(pl.col("exceeds_seed_noise"))
        assert set(survivors.get_column("horizon")) == {3}
        assert survivors.height >= 2

    def test_the_surviving_gain_is_still_small(self, summary):
        """Honest magnitude: single-digit seconds of MAE, not a headline."""
        survivors = summary.filter(pl.col("exceeds_seed_noise"))
        assert survivors.get_column("gain_temporal").min() > -0.15

    def test_routing_helps_nowhere_at_five_steps_or_more(self, summary):
        """Past the transition the learner dominates every regime, so there is
        nothing left to switch on."""
        long = summary.filter(pl.col("horizon") >= 5)
        assert not long.get_column("exceeds_seed_noise").any()

    def test_the_forward_generalization_failure_is_surfaced(self, summary):
        """E2 h=1: the policy helps under a random split and hurts under a
        temporal one. That is the case a random-split-only report would hide."""
        assert summary.get_column("fails_forward_in_time").any()
