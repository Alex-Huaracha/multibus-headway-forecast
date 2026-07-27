"""Contract tests for the canonical sample index (C1 + C2).

These are the guardians that did not exist. V1 covers temporal contiguity, V2
covers sample uniqueness. Both must fail closed: a regression that reintroduces
positional anchoring or per-``pair_rank`` replication has to turn one of these
red.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.data.sample_index import (
    effective_horizon_minutes,
    make_sample_index,
)

T_IN = 12
BASE = datetime(2024, 1, 8, 8, 0)


def _frame(offsets_minutes, *, empresaid=2, direction=1, pair_ranks=(0,)):
    """Headway frame whose snapshots sit at BASE + each offset, for each pair_rank."""
    rows = []
    for off in offsets_minutes:
        for pr in pair_ranks:
            rows.append(
                {
                    "empresaid": empresaid,
                    "direction": direction,
                    "pair_rank": pr,
                    "t": BASE + timedelta(minutes=int(off)),
                    "delta_t_min": 5.0,
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# V1 — temporal contiguity
# ---------------------------------------------------------------------------

class TestV1Contiguity:
    def test_effective_horizon_always_equals_nominal(self):
        """The realized gap is the declared horizon, for every sample."""
        df = _frame(range(60))
        for horizon in (1, 3, 5, 10):
            idx = make_sample_index(df, horizon=horizon, T_in=T_IN)
            assert idx.height > 0
            eff = effective_horizon_minutes(idx, T_in=T_IN)
            assert eff.unique().to_list() == [horizon], (
                f"h={horizon}: realized gaps {sorted(eff.unique().to_list())}"
            )

    def test_window_spanning_a_gap_is_dropped(self):
        """A discontinuity inside the window disqualifies it — the §3 defect."""
        # 0..19 contiguous, then a 23-hour jump, then 20 more contiguous minutes.
        offsets = list(range(20)) + [20 + 23 * 60 + k for k in range(20)]
        df = _frame(offsets)
        idx = make_sample_index(df, horizon=10, T_in=T_IN)

        # Each contiguous run of 20 yields 20 - (12+10) + 1 = -1 -> zero windows.
        # Nothing may bridge the gap.
        assert idx.height == 0

    def test_only_the_healthy_run_survives(self):
        """With one long run and one short run, only the long one produces samples."""
        offsets = list(range(40)) + [40 + 23 * 60 + k for k in range(5)]
        df = _frame(offsets)
        idx = make_sample_index(df, horizon=3, T_in=T_IN)

        span = T_IN + 3
        assert idx.height == 40 - span + 1
        eff = effective_horizon_minutes(idx, T_in=T_IN)
        assert eff.unique().to_list() == [3]

    def test_positional_anchoring_would_have_passed_but_contiguity_does_not(self):
        """Regression guard for the exact §3 scenario.

        Twelve contiguous minutes followed by a next-day snapshot: positional
        slicing calls that a valid h=1 window, contiguity refuses it.
        """
        offsets = list(range(T_IN)) + [T_IN + 23 * 60]
        df = _frame(offsets)
        assert df.height == T_IN + 1  # positionally, one h=1 window exists

        idx = make_sample_index(df, horizon=1, T_in=T_IN)
        assert idx.height == 0


# ---------------------------------------------------------------------------
# V2 — sample uniqueness (no fleet-density weighting)
# ---------------------------------------------------------------------------

class TestV2Uniqueness:
    def test_each_anchor_appears_exactly_once(self):
        df = _frame(range(60))
        idx = make_sample_index(df, horizon=5, T_in=T_IN)
        keys = idx.select(["empresaid", "direction", "start_ts", "horizon"])
        assert keys.height == keys.unique().height

    def test_pair_rank_multiplicity_does_not_replicate_samples(self):
        """The #13 defect: five slots per snapshot must NOT yield five samples."""
        single = make_sample_index(
            _frame(range(60), pair_ranks=(0,)), horizon=5, T_in=T_IN
        )
        many = make_sample_index(
            _frame(range(60), pair_ranks=(0, 1, 2, 3, 4)), horizon=5, T_in=T_IN
        )
        assert many.height == single.height, (
            "sample count must be independent of fleet density"
        )

    def test_series_are_kept_apart(self):
        """Two directions are two series; neither bridges into the other."""
        df = pl.concat(
            [
                _frame(range(40), direction=1),
                _frame(range(40), direction=-1),
            ]
        )
        idx = make_sample_index(df, horizon=3, T_in=T_IN)
        per_dir = idx.group_by("direction").len().sort("direction")
        counts = per_dir.get_column("len").to_list()
        assert counts == [40 - (T_IN + 3) + 1] * 2


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:
    def test_too_short_series_yields_empty_index_with_schema(self):
        idx = make_sample_index(_frame(range(3)), horizon=10, T_in=T_IN)
        assert idx.height == 0
        assert idx.columns == [
            "empresaid",
            "direction",
            "start_ts",
            "target_ts",
            "horizon",
        ]

    @pytest.mark.parametrize("horizon", [0, -1])
    def test_invalid_horizon_raises(self, horizon):
        with pytest.raises(ValueError, match="horizon must be"):
            make_sample_index(_frame(range(30)), horizon=horizon, T_in=T_IN)

    def test_deterministic_ordering(self):
        df = _frame(range(50))
        a = make_sample_index(df, horizon=3, T_in=T_IN)
        b = make_sample_index(df.sample(fraction=1.0, shuffle=True, seed=7), horizon=3, T_in=T_IN)
        assert a.equals(b)
