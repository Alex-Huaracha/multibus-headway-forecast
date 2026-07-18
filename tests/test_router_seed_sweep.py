"""The router's gain is not an artifact of one lucky policy/eval shuffle.

`src/build_router_seed_sweep.py` re-runs the partition under six seeds and writes
`docs/resultados/csv-multihorizon/router_seed_sweep_multihorizon.csv`.

These tests lock the robustness claim on the committed CSV, without re-running the
(slow) builder.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.build_router_seed_sweep import SEEDS

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "router_seed_sweep_multihorizon.csv"
)

# Same tolerance as tests/test_router.py: negligible in minutes, but a policy that
# flipped and made the router lose to a pure model would blow straight through it.
_TOL = 1e-3


def _load() -> pl.DataFrame:
    assert SWEEP_CSV.exists(), f"sweep CSV missing: {SWEEP_CSV}"
    return pl.read_csv(SWEEP_CSV)


def test_covers_every_cell_under_every_seed() -> None:
    df = _load()
    assert df.height == 12 * len(SEEDS)
    assert set(df["seed"].unique()) == set(SEEDS)
    assert df.group_by(["corridor", "horizon"]).len()["len"].to_list() == [len(SEEDS)] * 12


def test_learned_policy_is_identical_across_seeds() -> None:
    """The headline robustness claim: the shuffle never changes which model wins where."""
    unstable = (
        df := _load()
    ).group_by(["corridor", "horizon"]).agg(
        pl.col("policy_low_mid_high").n_unique().alias("variants")
    ).filter(pl.col("variants") > 1)
    assert unstable.height == 0, f"policy depends on the seed in: {unstable}"
    assert df.height > 0


def test_router_never_loses_to_either_pure_model_under_any_seed() -> None:
    df = _load()
    worse = df.filter(
        (pl.col("router_vs_dl") > _TOL) | (pl.col("router_vs_persist") > _TOL)
    )
    assert worse.height == 0, f"router loses to a pure model in: {worse}"


def test_gain_spread_across_seeds_is_negligible() -> None:
    """Per cell, the seed may jitter the gain slightly but must not change its size."""
    spread = _load().group_by(["corridor", "horizon"]).agg(
        (pl.col("router_vs_dl").max() - pl.col("router_vs_dl").min()).alias("spread")
    )
    worst = float(spread["spread"].max())
    assert worst < 0.02, f"gain varies by {worst:.4f} min across seeds — not robust"
