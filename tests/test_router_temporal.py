"""Temporal-block router: split contract + committed-CSV honesty checks.

The uniform-permutation split (`build_router.policy_eval_split`) scatters near-twin
overlapping windows across both slices, so the test barely discriminates. The
temporal block split learns the policy on the EARLIER part of the test period and
scores on the LATER part, which is a stricter, more honest evaluation. These tests
lock the split contract and whatever the honest run produced — they do NOT assume
the router still wins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.build_router_temporal import temporal_block_split

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_earlier_block_learns_later_block_scores() -> None:
    """policy = earliest ~frac by timestamp; eval = the rest; disjoint + exhaustive."""
    ts = np.arange(10).astype("datetime64[us]")
    pol, ev = temporal_block_split(ts, 0.6)
    assert set(pol).isdisjoint(set(ev))
    assert sorted([*pol, *ev]) == list(range(10))
    # earliest timestamps are in policy, latest in eval
    assert max(ts[pol]) < min(ts[ev])
    assert len(pol) == 6 and len(ev) == 4


def test_whole_timestamp_groups_do_not_straddle() -> None:
    """Samples sharing a timestamp (multiple buses per snapshot) stay on one side."""
    # 3 buses at each of 4 timestamps → cut must fall on a timestamp boundary.
    ts = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype="datetime64[us]")
    pol, ev = temporal_block_split(ts, 0.6)
    # No timestamp appears on both sides.
    assert set(ts[pol]).isdisjoint(set(ts[ev]))
    assert max(ts[pol]) < min(ts[ev])


def test_is_deterministic_no_rng() -> None:
    ts = np.array([5, 1, 3, 2, 4, 0, 6, 9, 7, 8], dtype="datetime64[us]")
    a = temporal_block_split(ts, 0.6)
    b = temporal_block_split(ts, 0.6)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_degenerate_single_timestamp_raises() -> None:
    ts = np.zeros(10, dtype="datetime64[us]")  # all identical → cannot form 2 blocks
    with pytest.raises(ValueError, match="a block is empty"):
        temporal_block_split(ts, 0.6)


# --- Committed-CSV honesty checks (lock the actual temporal-split result) ---

import polars as pl  # noqa: E402

CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
TEMPORAL_CSV = CSV_DIR / "router_temporal_multihorizon.csv"
UNIFORM_CSV = CSV_DIR / "router_multihorizon.csv"

_TOL = 1e-3
_ALIGN_REGRESSION_BOUND = 1e-4


def _load_temporal() -> pl.DataFrame:
    assert TEMPORAL_CSV.exists(), f"temporal router CSV missing: {TEMPORAL_CSV}"
    return pl.read_csv(TEMPORAL_CSV)


def test_covers_three_corridors_four_horizons_temporal() -> None:
    df = _load_temporal()
    assert df.height == 12
    assert set(df["corridor"].unique()) == {"E2", "E59", "E4"}
    assert set(df["horizon"].unique()) == {1, 3, 5, 10}
    assert (df["split_kind"] == "temporal_block").all()


def test_policy_eval_blocks_are_disjoint_and_roughly_60_40() -> None:
    df = _load_temporal()
    frac = df["n_policy"].to_numpy() / (df["n_policy"].to_numpy() + df["n_eval"].to_numpy())
    # A timestamp-boundary cut lands near POLICY_FRAC but not exactly (whole
    # snapshots stay on one side), so allow a small band.
    assert np.all(np.abs(frac - 0.6) < 0.02)


def test_positional_join_gate_passed_under_temporal_split() -> None:
    df = _load_temporal()
    assert (df["align_tolerance"] == 1e-2).all()
    assert float(df["align_max_abs_diff"].max()) < _ALIGN_REGRESSION_BOUND


def test_router_ties_or_beats_both_models_and_matches_oracle() -> None:
    """Under the stricter temporal split the router still never loses to a pure
    model, and it matches the on-eval oracle in every cell — so the 12/12 oracle
    match is NOT an artifact of the uniform split's overlapping-window twins."""
    df = _load_temporal()
    assert (df["router_vs_dl"] <= _TOL).all(), "router worse than always-DL somewhere"
    assert (df["router_vs_persist"] <= _TOL).all(), "router worse than always-persist"
    assert ((df["mae_router"] - df["mae_oracle"]).abs() < 1e-6).all(), "router != oracle"


def test_policy_is_identical_to_the_uniform_split() -> None:
    """The honest headline: switching from a uniform to a temporal split does NOT
    change the learned per-tercile policy in any cell."""
    t = _load_temporal().sort(["corridor", "horizon"])
    u = pl.read_csv(UNIFORM_CSV).sort(["corridor", "horizon"])
    assert (
        t["policy_low_mid_high"].to_list() == u["policy_low_mid_high"].to_list()
    ), "temporal split changed the policy — report it, do not hide it"


def test_gain_over_trivial_rule_stays_small_and_sparse() -> None:
    """The honest ablation survives the temporal split: the volatility signal adds
    little beyond the trivial horizon rule (persistence at h=1, DL at h>=3)."""
    df = _load_temporal()
    w = df["n_eval"].to_numpy()
    rv_trivial = df["router_vs_trivial"].to_numpy()
    assert (rv_trivial <= _TOL).all(), "router must never lose to the trivial rule"
    # The meaningful beats are a small minority of cells (marginal ones aside).
    assert int(np.sum(rv_trivial < -0.01)) == 2
    assert -0.03 < np.average(rv_trivial, weights=w) < -0.005
