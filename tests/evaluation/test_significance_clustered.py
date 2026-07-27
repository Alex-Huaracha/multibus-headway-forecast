"""Tests for the corrected significance apparatus.

Each class maps to one audit finding: the clustered variance (#6), the
small-sample DM corrections (#7), and the directional Wilcoxon (#1). The
properties asserted are the ones that make the corrections *matter* — that the
clustered estimator is more conservative, that the lag floor binds, that HLN
shrinks rather than inflates — not merely that the functions return numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.significance_clustered import (
    dm_clustered,
    dm_hac_hln,
    hln_scale,
    wilcoxon_directional,
)

RNG = np.random.default_rng(20240727)


def _correlated_by_day(n_days=40, per_day=200, day_effect=0.4, noise=1.0):
    """Loss differentials with a shared per-day shift — the structure #6 targets."""
    day_shift = RNG.normal(0.0, day_effect, size=n_days)
    d, days = [], []
    for g in range(n_days):
        d.append(RNG.normal(day_shift[g], noise, size=per_day))
        days.append(np.full(per_day, g))
    return np.concatenate(d), np.concatenate(days)


class TestClusteredVarianceIsMoreConservative:
    """#6 — samples inside a service day are correlated; ignoring it inflates significance."""

    def test_clustered_standard_error_exceeds_hac_under_day_effects(self):
        d, days = _correlated_by_day()
        hac = dm_hac_hln(d, horizon=3)
        clu = dm_clustered(d, days, horizon=3)
        assert abs(clu.stat) < abs(hac.stat), (
            "clustering must shrink the statistic when days are correlated"
        )
        assert clu.p_value > hac.p_value

    def test_effective_sample_size_is_the_number_of_days(self):
        d, days = _correlated_by_day(n_days=25, per_day=100)
        clu = dm_clustered(d, days, horizon=3)
        assert clu.n_clusters == 25
        assert clu.dof == 24, "the t reference must use G-1, not n-1"
        assert clu.n == 2500

    def test_without_day_effects_the_two_agree_closely(self):
        """No correlation to capture -> the correction should be nearly inert."""
        d, days = _correlated_by_day(day_effect=0.0)
        hac = dm_hac_hln(d, horizon=3)
        clu = dm_clustered(d, days, horizon=3)
        assert abs(clu.stat) == pytest.approx(abs(hac.stat), rel=0.35)

    def test_accepts_non_integer_cluster_labels(self):
        d, days = _correlated_by_day(n_days=6, per_day=50)
        labels = np.array([f"2024-02-{g + 1:02d}" for g in days])
        clu = dm_clustered(d, labels, horizon=3)
        assert clu.n_clusters == 6

    def test_rejects_a_single_cluster(self):
        d = RNG.normal(size=200)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            dm_clustered(d, np.zeros(200), horizon=3)


class TestSmallSampleApparatus:
    """#7 — HLN correction, lag floor, and a t reference."""

    def test_hln_scale_shrinks_and_never_exceeds_one(self):
        for n in (50, 500, 5000):
            for h in (1, 3, 5, 10):
                s = hln_scale(n, h)
                assert 0.0 < s <= 1.0, (n, h, s)

    def test_hln_shrinks_more_at_longer_horizons(self):
        assert hln_scale(1000, 10) < hln_scale(1000, 1)

    def test_hln_is_negligible_at_large_n(self):
        assert hln_scale(1_000_000, 3) == pytest.approx(1.0, abs=1e-4)

    def test_lag_floor_binds_at_long_horizons(self):
        """n^(1/3) is below h-1 for small n; the floor must win."""
        d = RNG.normal(0.1, 1.0, size=64)  # floor(64^(1/3)) == 4
        assert dm_hac_hln(d, horizon=10).lag == 9
        assert dm_hac_hln(d, horizon=3).lag == 4  # data-driven wins here

    def test_explicit_lag_overrides_the_floor(self):
        d = RNG.normal(0.1, 1.0, size=500)
        assert dm_hac_hln(d, horizon=10, lag=2).lag == 2

    def test_t_reference_is_used(self):
        d = RNG.normal(0.05, 1.0, size=30)
        assert dm_hac_hln(d, horizon=1).dof == 29

    def test_zero_mean_gives_no_significance(self):
        d = np.zeros(100)
        res = dm_hac_hln(d, horizon=1)
        assert res.p_value == 1.0
        assert res.stat == 0.0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            dm_hac_hln(np.array([1.0, 2.0]), horizon=1)


class TestDirectionalWilcoxon:
    """#1 — a model can win the mean and lose the median."""

    def test_detects_mean_median_disagreement(self):
        """Many small losses traded for a few large gains: the audited pattern."""
        d = np.concatenate([np.full(900, 0.10), np.full(100, -1.50)])
        out = wilcoxon_directional(d)
        assert out["mean_diff"] < 0, "the mean says the first model wins"
        assert out["median_diff"] > 0, "the median says it loses"
        assert out["mean_median_disagree"] is True
        assert out["win_rate"] == pytest.approx(0.10)

    def test_agreement_is_reported_as_such(self):
        d = RNG.normal(-0.5, 0.2, size=500)
        out = wilcoxon_directional(d)
        assert out["mean_median_disagree"] is False
        assert out["win_rate"] > 0.9

    def test_one_sided_follows_the_mean(self):
        out = wilcoxon_directional(RNG.normal(-0.5, 0.2, size=300))
        assert out["wilcoxon_direction"] == "less"
        out = wilcoxon_directional(RNG.normal(+0.5, 0.2, size=300))
        assert out["wilcoxon_direction"] == "greater"

    def test_one_sided_is_never_larger_than_two_sided(self):
        out = wilcoxon_directional(RNG.normal(-0.3, 1.0, size=400))
        assert out["wilcoxon_p_one_sided"] <= out["wilcoxon_p_two_sided"] + 1e-12

    def test_win_rate_counts_samples_not_magnitude(self):
        d = np.array([-10.0, 0.1, 0.1, 0.1])
        assert wilcoxon_directional(d)["win_rate"] == pytest.approx(0.25)


class TestCubeRootFloor:
    """The naive `int(n ** (1/3))` is off by one on exact cubes."""

    @pytest.mark.parametrize("n,expected", [(1, 1), (8, 2), (27, 3), (64, 4), (125, 5), (1000, 10)])
    def test_exact_cubes(self, n, expected):
        from src.evaluation.significance_clustered import cube_root_floor

        assert cube_root_floor(n) == expected

    @pytest.mark.parametrize("n", [2, 7, 26, 63, 65, 999, 1001, 90_469, 240_907])
    def test_matches_the_mathematical_floor(self, n):
        from src.evaluation.significance_clustered import cube_root_floor

        r = cube_root_floor(n)
        assert r**3 <= n < (r + 1) ** 3

    def test_the_naive_expression_is_the_one_that_is_wrong(self):
        """Documents the wart rather than trusting it stays fixed elsewhere."""
        assert int(64 ** (1.0 / 3.0)) == 3  # the bug
        from src.evaluation.significance_clustered import cube_root_floor

        assert cube_root_floor(64) == 4  # the fix
