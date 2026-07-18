"""The ex-ante volatility router is best-or-tied against both pure models.

`src/build_router.py` writes `docs/resultados/csv-multihorizon/router_multihorizon.csv`
with a leakage-free evaluation (regime terciles frozen on train+val; switching
policy learned on a held-out test slice and scored on the disjoint remainder).

These tests lock the honest headline claim and the leakage-discipline contract on
the committed CSV, without re-running the (slow) builder.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.build_router import policy_eval_split

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTER_CSV = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "router_multihorizon.csv"

# Negligible in minutes; absorbs float noise while still catching a real regression
# (a policy that flips and makes the router worse than a pure model would exceed it).
_TOL = 1e-3

# Observed alignment diffs are ~2.6e-6 (float32 round-trip); this bound catches a real
# ordering/count regression without tripping on float noise. Far below ALIGN_TOL (1e-2).
_ALIGN_REGRESSION_BOUND = 1e-4


def _load() -> pl.DataFrame:
    assert ROUTER_CSV.exists(), f"router CSV missing: {ROUTER_CSV}"
    return pl.read_csv(ROUTER_CSV)


def test_covers_three_corridors_four_horizons() -> None:
    df = _load()
    assert df.height == 12
    assert set(df["corridor"].unique()) == {"E2", "E59", "E4"}
    assert set(df["horizon"].unique()) == {1, 3, 5, 10}


def test_router_ties_or_beats_both_pure_models() -> None:
    """The core contribution: one policy is best-or-tied vs always-DL AND always-persist."""
    df = _load()
    worse_than_dl = df.filter(pl.col("mae_router") > pl.col("mae_dl") + _TOL)
    worse_than_persist = df.filter(pl.col("mae_router") > pl.col("mae_persist") + _TOL)
    assert worse_than_dl.height == 0, f"router worse than always-DL in: {worse_than_dl}"
    assert worse_than_persist.height == 0, f"router worse than always-persist in: {worse_than_persist}"


def test_router_no_better_than_oracle() -> None:
    """Sanity: the deployable router cannot beat the on-eval oracle upper bound."""
    df = _load()
    impossible = df.filter(pl.col("mae_router") < pl.col("mae_oracle") - _TOL)
    assert impossible.height == 0, f"router beats its own oracle (bug) in: {impossible}"


def test_leakage_discipline_fields_frozen() -> None:
    df = _load()
    assert (df["policy_frac"] == 0.6).all()
    assert (df["seed"] == 42).all()


def test_policy_and_eval_slices_are_disjoint_and_exhaustive() -> None:
    """The leakage-critical property, checked on the split itself.

    The CSV columns alone cannot prove this: the builder writes ``seed``/``policy_frac``
    verbatim, so asserting them is tautological. This exercises the real split function.
    """
    for n in (10, 1_000, 654_303):
        pol, ev = policy_eval_split(n, seed=42, policy_frac=0.6)
        assert set(pol).isdisjoint(set(ev)), f"policy/eval slices overlap at n={n}"
        assert len(pol) + len(ev) == n
        assert set(pol) | set(ev) == set(range(n)), "slices do not partition the samples"
        assert len(pol) == int(0.6 * n)


def test_policy_eval_split_is_deterministic() -> None:
    a_pol, a_ev = policy_eval_split(5_000, seed=42, policy_frac=0.6)
    b_pol, b_ev = policy_eval_split(5_000, seed=42, policy_frac=0.6)
    assert (a_pol == b_pol).all() and (a_ev == b_ev).all()


def test_positional_join_alignment_gate_passed_everywhere() -> None:
    """DL predictions are joined by position; every cell must have cleared the audit.

    Guards the horizon that matters most: h=1 carries the router's entire gain over
    always-DL, and was never covered by the ex-ante builder's audit (h in {3,5,10}).
    """
    df = _load()
    assert (df["align_tolerance"] == 1e-2).all()
    worst = float(df["align_max_abs_diff"].max())
    assert worst < _ALIGN_REGRESSION_BOUND, f"alignment degraded: max abs diff {worst}"
    assert df.filter(pl.col("horizon") == 1).height == 3, "h=1 must be audited too"


def test_gain_over_dl_concentrates_at_short_horizons() -> None:
    """Honest structural fact: at h>=5 the router reduces to always-DL (zero gain)."""
    df = _load()
    long_h = df.filter(pl.col("horizon").is_in([5, 10]))
    # At long horizons DL wins every tercile, so the router adds nothing (within tol).
    assert (long_h["router_vs_dl"].abs() <= _TOL).all()
