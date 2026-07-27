"""The local residual store must carry WINSORIZED test targets, not raw ones.

Why this test exists
--------------------
The store under ``docs/resultados/residuos-multihorizon/`` went stale without any
alarm: it predated the winsorization fix, so its ``y_true`` column held UNCLIPPED
test targets capped at the raw 30-minute trip-gap ceiling instead of the train
p99. Every local recompute that reads it — significance, volatility, ex-ante,
router, paired audit, and the DL-vs-XGBoost comparison — silently produced
numbers for a population the Kaggle kernels had stopped producing. Nine of nine
model x corridor combinations were affected and nothing failed.

The winsorization CONTRACT itself is guarded upstream
(``tests/test_preprocessing_winsorization_contract.py``,
``tests/test_notebook_integrity_guard.py``). What was missing is a guard on the
DOWNLOADED ARTIFACT: proof that the bytes on this machine came from a kernel that
honoured the contract. That is what this test adds.

Two signatures, both asserted
-----------------------------
1. no ``y_true`` above its corridor's train p99 ceiling (plus ``CEILING_TOL``);
2. no ``y_true`` exactly equal to ``RAW_TRIP_GAP_CEILING`` (30.0).

(2) is the sharper signal — the stale files had thousands of rows sitting exactly
at 30.0 — while (1) catches a ceiling that is wrong without being exactly 30.

``CEILING_TOL`` is 1e-4, NOT 1e-6: the residual CSVs round-trip their targets
through float32, so rows sitting exactly AT the ceiling read back ~1e-6 above it
(E4 ceiling 29.098441, observed CSV max 29.098442). A 1e-6 threshold reports
roughly 900 false positives per E4 file.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESID_DIR = REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon"

# Train-only p99 of delta_t_min per corridor — the winsorization ceiling that the
# fresh Kaggle kernels apply to every split, test ground truth included.
P99_CEILING = {"E2": 28.467923, "E59": 27.996949, "E4": 29.098441}

# float32 round-trip slack of the residual CSV export. See the module docstring:
# 1e-6 is too tight and flags the legitimate at-the-ceiling rows.
CEILING_TOL = 1e-4

# The raw trip-gap cap the STALE (pre-fix) exports were capped at instead.
RAW_TRIP_GAP_CEILING = 30.0


def _residual_files() -> list[Path]:
    return sorted(RESID_DIR.rglob("*_residuals_h*.csv"))


def test_residual_store_is_populated() -> None:
    """A missing store must fail loudly rather than vacuously passing below."""
    files = _residual_files()
    assert files, (
        f"no per-sample residual CSV under {RESID_DIR} — refresh the store from "
        "Kaggle before running the residual-derived report builders"
    )


@pytest.mark.parametrize(
    "path", _residual_files(), ids=lambda p: str(p.relative_to(RESID_DIR))
)
def test_residual_targets_are_winsorized_to_the_train_p99(path: Path) -> None:
    """No target above its corridor ceiling, and none parked at the raw 30.0 cap."""
    frame = pl.read_csv(path, columns=["corridor", "y_true"])
    corridors = sorted(frame.get_column("corridor").unique().to_list())
    assert corridors, f"{path} carries no corridor labels"

    for corridor in corridors:
        assert corridor in P99_CEILING, f"{path}: unknown corridor {corridor!r}"
        y_true = frame.filter(pl.col("corridor") == corridor).get_column("y_true")
        ceiling = P99_CEILING[corridor]

        above = int((y_true > ceiling + CEILING_TOL).sum())
        assert above == 0, (
            f"{path.name} [{corridor}]: {above} rows above the train p99 ceiling "
            f"{ceiling} (max={y_true.max()}) — these residuals are STALE "
            "(pre-winsorization-fix) and every metric derived from them is invalid"
        )

        at_raw_cap = int((y_true == RAW_TRIP_GAP_CEILING).sum())
        assert at_raw_cap == 0, (
            f"{path.name} [{corridor}]: {at_raw_cap} rows sit exactly at the raw "
            f"trip-gap ceiling {RAW_TRIP_GAP_CEILING} — the unmistakable signature "
            "of a pre-winsorization-fix export"
        )
