"""The AUC and MCC verdicts need an instrument, and this is its contract.

Section IV-E defines a verdict as "which of the two wins, by how much, and
whether the difference survives its test". The MAE side has Diebold-Mariano; the
ranking side had nothing, so a 0.001 AUC margin counted exactly like a 0.05 one.

These tests pin the behaviour that makes such a margin reportable: the interval
is built by resampling SERVICE DAYS, not rows, because the paper already argues
that samples from one day share weather, incidents and demand.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ranking_significance import (
    RankingSignificanceError,
    clustered_bootstrap_delta,
    fast_auc,
)
from src.evaluation.vector_metrics import ranking_scores


class TestFastAUC:
    """The bootstrap's AUC must be the published AUC, not merely close to it."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_fast_auc_matches_ranking_scores(self, seed):
        rng = np.random.default_rng(seed)
        truth = rng.random(4000) < 0.3
        score = rng.normal(size=4000)
        assert fast_auc(truth, score) == pytest.approx(
            ranking_scores(truth, score)["auc"], rel=0, abs=1e-12
        )

    def test_matches_under_heavy_ties(self):
        """Ties are where a rank implementation goes wrong, so they get their own case."""
        rng = np.random.default_rng(11)
        truth = rng.random(3000) < 0.4
        score = rng.integers(0, 5, size=3000).astype(float)
        assert fast_auc(truth, score) == pytest.approx(
            ranking_scores(truth, score)["auc"], rel=0, abs=1e-12
        )

    def test_constant_score_is_one_half(self):
        truth = np.array([True, False, True, False])
        assert fast_auc(truth, np.ones(4)) == pytest.approx(0.5)

    def test_single_class_has_no_auc(self):
        assert np.isnan(fast_auc(np.ones(5, dtype=bool), np.arange(5.0)))


def _auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC, kept local so the test does not inherit a bug it checks."""
    pos, neg = score[truth], score[~truth]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _synthetic(n_days: int = 20, per_day: int = 30, seed: int = 7):
    """A corpus where `good` ranks the event and `useless` is pure noise."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    cluster = np.repeat(np.arange(n_days), per_day)
    truth = rng.random(n) < 0.3
    good = rng.normal(loc=truth.astype(float), scale=0.5)
    useless = rng.normal(size=n)
    return truth, good, useless, cluster


class TestClusteredBootstrapDelta:
    def test_real_difference_excludes_zero(self):
        truth, good, useless, cluster = _synthetic()
        result = clustered_bootstrap_delta(
            truth, good, useless, cluster, statistic=_auc, n_boot=400, seed=1
        )
        assert result.delta > 0
        assert result.ci_low > 0
        assert not result.crosses_zero

    def test_identical_scores_give_zero_delta_and_cross_zero(self):
        truth, good, _, cluster = _synthetic()
        result = clustered_bootstrap_delta(
            truth, good, good, cluster, statistic=_auc, n_boot=400, seed=1
        )
        assert result.delta == pytest.approx(0.0, abs=1e-12)
        assert result.crosses_zero

    def test_same_seed_reproduces_the_interval(self):
        truth, good, useless, cluster = _synthetic()
        kwargs = dict(statistic=_auc, n_boot=300, seed=99)
        first = clustered_bootstrap_delta(truth, good, useless, cluster, **kwargs)
        second = clustered_bootstrap_delta(truth, good, useless, cluster, **kwargs)
        assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)

    def test_different_seeds_move_the_interval(self):
        truth, good, useless, cluster = _synthetic()
        a = clustered_bootstrap_delta(truth, good, useless, cluster,
                                      statistic=_auc, n_boot=300, seed=1)
        b = clustered_bootstrap_delta(truth, good, useless, cluster,
                                      statistic=_auc, n_boot=300, seed=2)
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)

    def test_resampling_is_over_clusters_not_rows(self):
        """One cluster means every draw is the same corpus, so the interval is a point.

        This is the property that separates a clustered bootstrap from a naive
        one: with a single day there is nothing to resample, and a row-level
        bootstrap would still manufacture a width out of nothing.
        """
        truth, good, useless, _ = _synthetic()
        one_day = np.zeros(truth.size, dtype=int)
        result = clustered_bootstrap_delta(
            truth, good, useless, one_day, statistic=_auc, n_boot=50, seed=1
        )
        assert result.n_clusters == 1
        assert result.ci_low == pytest.approx(result.ci_high)

    def test_more_clusters_narrow_the_interval(self):
        narrow = clustered_bootstrap_delta(
            *_synthetic(n_days=60)[:3], _synthetic(n_days=60)[3],
            statistic=_auc, n_boot=400, seed=5,
        )
        wide = clustered_bootstrap_delta(
            *_synthetic(n_days=8)[:3], _synthetic(n_days=8)[3],
            statistic=_auc, n_boot=400, seed=5,
        )
        assert (narrow.ci_high - narrow.ci_low) < (wide.ci_high - wide.ci_low)

    def test_reports_the_cluster_count_it_actually_used(self):
        truth, good, useless, cluster = _synthetic(n_days=13)
        result = clustered_bootstrap_delta(
            truth, good, useless, cluster, statistic=_auc, n_boot=100, seed=1
        )
        assert result.n_clusters == 13

    def test_level_widens_the_interval(self):
        truth, good, useless, cluster = _synthetic()
        tight = clustered_bootstrap_delta(truth, good, useless, cluster,
                                          statistic=_auc, n_boot=400, seed=3,
                                          level=0.80)
        loose = clustered_bootstrap_delta(truth, good, useless, cluster,
                                          statistic=_auc, n_boot=400, seed=3,
                                          level=0.99)
        assert (loose.ci_high - loose.ci_low) > (tight.ci_high - tight.ci_low)

    def test_mismatched_lengths_raise(self):
        truth, good, useless, cluster = _synthetic()
        with pytest.raises(RankingSignificanceError):
            clustered_bootstrap_delta(truth, good, useless[:-1], cluster,
                                      statistic=_auc, n_boot=10, seed=1)

    def test_no_usable_draw_raises_instead_of_returning_nan(self):
        """A statistic that never evaluates must fail loudly, not report nothing.

        Returning a nan interval here would print as a blank cell next to real
        ones, which reads as "no difference" rather than "no measurement".
        """
        truth, good, useless, cluster = _synthetic()
        with pytest.raises(RankingSignificanceError):
            clustered_bootstrap_delta(
                truth, good, useless, cluster,
                statistic=lambda t, s: float("nan"), n_boot=50, seed=1,
            )
