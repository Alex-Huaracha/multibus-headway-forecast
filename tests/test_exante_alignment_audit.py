"""Sample-to-sample alignment audit is persisted and stays within tolerance.

`src/build_exante_volatility.py` reconstructs the target and persistence from the
raw data and checks them against the downloaded residual CSVs. It writes the
per-corridor x horizon max absolute differences to
`docs/resultados/csv-multihorizon/exante_alignment_multihorizon.csv` so the §5
"Δ máx observado" claim in `documento-resultados.md` traces to a CSV instead of a
runtime print. These tests lock that contract.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIGN_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "exante_alignment_multihorizon.csv"
)

# Well above the ~2.7e-6 float-reconstruction noise but far below the 1e-2 hard
# gate: a real misalignment (wrong sample order, shifted horizon) would blow past
# this by orders of magnitude, while legitimate float jitter stays under it.
_REGRESSION_BOUND = 1e-4


def _load() -> pl.DataFrame:
    assert ALIGN_CSV.exists(), f"alignment audit CSV missing: {ALIGN_CSV}"
    return pl.read_csv(ALIGN_CSV)


def test_alignment_audit_covers_nine_corridor_horizon_cells() -> None:
    df = _load()
    assert df.height == 9  # 3 corridors x 3 horizons (h in {3, 5, 10})
    assert set(df["corridor"].unique()) == {"E2", "E59", "E4"}
    assert set(df["horizon"].unique()) == {3, 5, 10}


def test_every_cell_passes_the_hard_gate() -> None:
    df = _load()
    assert df["passed"].all()
    assert (df["tolerance"] == 1e-2).all()


def test_max_alignment_diff_stays_negligible() -> None:
    df = _load()
    global_max = max(
        df["max_abs_diff_target"].max(),
        df["max_abs_diff_persist"].max(),
    )
    assert global_max < _REGRESSION_BOUND, (
        f"alignment drifted to {global_max:.2e} (bound {_REGRESSION_BOUND:.0e}) — "
        "reconstructed target/persistence no longer match the residual CSVs"
    )
