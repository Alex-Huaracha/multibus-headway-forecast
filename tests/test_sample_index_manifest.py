"""V3 — shared-population gate for the retrained pipeline.

The retrained LSTM, the refitted XGBoost and persistence must consume the SAME
sample population. Historically they did not (audit §2.1), and the mismatch was
patched by a post-hoc join.

The enforcement mechanism is a digest, not a shipped file: same code plus same
input bytes yields the same index, so every consumer recomputes it and asserts
the digest matches ``sample_index_manifest.csv``. These tests are the local half
of that gate.

Tests that need the processed parquets skip when they are absent — the data is
gitignored and lives in Kaggle. The manifest itself is committed, so the
schema and self-consistency checks always run.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from src.build_sample_index import (  # noqa: E402
    CORRIDORS,
    HORIZONS,
    OUT_CSV,
    SPLIT_BOUNDS,
    T_IN,
    index_digest,
    load_corridor,
)
from src.data.sample_index import make_sample_index  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

_missing = [
    f"headways_E{emp}.parquet"
    for emp, _ in CORRIDORS
    if not (DATA_DIR / f"headways_E{emp}.parquet").exists()
]
needs_data = pytest.mark.skipif(
    bool(_missing),
    reason=f"processed parquets absent locally: {_missing}. "
    "uv run kaggle kernels output alexhuaracha/04-preprocessing -p <dir>",
)


@pytest.fixture(scope="module")
def manifest() -> pl.DataFrame:
    if not OUT_CSV.exists():
        pytest.skip(f"{OUT_CSV} not built yet — run: uv run python -m src.build_sample_index")
    return pl.read_csv(OUT_CSV)


class TestManifestShape:
    def test_covers_every_corridor_horizon_split(self, manifest):
        assert manifest.height == len(CORRIDORS) * len(HORIZONS) * len(SPLIT_BOUNDS)

    def test_digests_are_unique_per_cell(self, manifest):
        """A repeated digest would mean two cells share an index — a bug."""
        digests = manifest.get_column("sha256").to_list()
        assert len(digests) == len(set(digests))

    def test_every_cell_has_samples(self, manifest):
        assert manifest.filter(pl.col("n_samples") <= 0).height == 0

    def test_contiguity_cost_stays_within_measured_band(self, manifest):
        """Guard against a silent regression in the contiguity predicate.

        Measured 2026-07-27: 81.9-90.6 % of snapshots survive. A build that
        suddenly keeps ~100 % means the predicate stopped filtering; one that
        keeps far less means it over-filters. Either way, look before trusting.
        """
        pct = manifest.get_column("pct_snapshots_usable")
        assert pct.min() >= 75.0, f"over-filtering: min {pct.min()}%"
        assert pct.max() <= 95.0, f"predicate likely inert: max {pct.max()}%"

    def test_scalar_n_exceeds_sample_n(self, manifest):
        """Each sample predicts a vector, so scalars must outnumber samples."""
        bad = manifest.filter(pl.col("n_scalar_effective") <= pl.col("n_samples"))
        assert bad.height == 0, bad

    def test_paired_n_is_reported_and_bounded_by_target_n(self, manifest):
        """The paired count is what the significance tests run on.

        Reporting only ``n_scalar_effective`` overstates power: a cell whose
        target exists but whose last input snapshot is missing cannot be paired
        against B1 and is dropped from every comparison.
        """
        assert "n_scalar_paired" in manifest.columns
        bad = manifest.filter(
            pl.col("n_scalar_paired") > pl.col("n_scalar_effective")
        )
        assert bad.height == 0, bad

    def test_paired_n_leaves_usable_power(self, manifest):
        """Guard the go/no-go: every cell must keep enough paired samples."""
        worst = manifest.get_column("n_scalar_paired").min()
        assert worst >= 50_000, f"paired n collapsed to {worst}"


@needs_data
class TestSharedPopulation:
    def test_digest_is_reproducible(self):
        """Rebuilding the same cell twice yields the same digest."""
        emp, _ = CORRIDORS[0]
        frame = load_corridor(emp)
        lo, hi = SPLIT_BOUNDS["test"]
        day = pl.col("t").dt.date()
        part = frame.filter((day >= lo) & (day <= hi))

        a = make_sample_index(part, horizon=3, T_in=T_IN)
        b = make_sample_index(part, horizon=3, T_in=T_IN)
        assert index_digest(a) == index_digest(b)

    def test_manifest_matches_a_fresh_recomputation(self, manifest):
        """The gate every Kaggle kernel will run, executed locally.

        A drift here means the committed manifest no longer describes the index
        the code produces — exactly the staleness class that produced the
        obsolete Figure 1 and the stale residual tree.
        """
        day = pl.col("t").dt.date()
        for emp, corridor in CORRIDORS:
            frame = load_corridor(emp)
            for split, (lo, hi) in SPLIT_BOUNDS.items():
                part = frame.filter((day >= lo) & (day <= hi))
                for horizon in HORIZONS:
                    index = make_sample_index(part, horizon=horizon, T_in=T_IN)
                    expected = manifest.filter(
                        (pl.col("corridor") == corridor)
                        & (pl.col("split") == split)
                        & (pl.col("horizon") == horizon)
                    )
                    assert expected.height == 1, f"{corridor}/{split}/h{horizon} missing"
                    row = expected.row(0, named=True)
                    assert index.height == row["n_samples"], (
                        f"{corridor}/{split}/h{horizon}: "
                        f"{index.height} samples vs manifest {row['n_samples']}"
                    )
                    assert index_digest(index) == row["sha256"], (
                        f"{corridor}/{split}/h{horizon}: digest drift"
                    )

    def test_population_is_independent_of_pair_rank_multiplicity(self):
        """C1 on real data: dropping duplicate slots must not change the index."""
        emp, _ = CORRIDORS[0]
        frame = load_corridor(emp)
        lo, hi = SPLIT_BOUNDS["test"]
        day = pl.col("t").dt.date()
        part = frame.filter((day >= lo) & (day <= hi))

        full = make_sample_index(part, horizon=5, T_in=T_IN)
        # One row per snapshot instead of one per (snapshot, pair_rank).
        collapsed = part.unique(subset=["empresaid", "direction", "t"], keep="first")
        thin = make_sample_index(collapsed, horizon=5, T_in=T_IN)

        assert index_digest(full) == index_digest(thin)
