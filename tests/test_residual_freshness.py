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
1. no ``y_true`` above the train p99 ceiling of its (origin, corridor) pair
   (plus ``CEILING_TOL``);
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

# Train-only p99 of delta_t_min, per ORIGIN and corridor — the winsorization
# ceiling that the fresh Kaggle kernels apply to every split, test ground truth
# included.
#
# The ceiling depends on the origin, not just the corridor: each fold winsorizes
# on ITS OWN train window (`src/evaluation/splits.py`), so r1 and r2 legitimately
# clip at a different value than the published fold. This table was flat while
# only one origin existed; measuring r1's targets against `main`'s ceiling
# reported ~2100 correctly-winsorized E59 rows as stale.
#
# The rolling values are transcribed from the `<corridor>: winsor threshold=`
# lines of each kernel log, printed at four decimals. That rounding is at most
# 5e-5, comfortably inside `CEILING_TOL` even after the float32 round-trip. The
# same lines reproduce `main`'s six-decimal entries (28.4679 / 27.9969 /
# 29.0984), which is what makes the transcription trustworthy.
#
# The ceilings do NOT order with the length of the train window: E4 clips higher
# at r1 (29.1026) than at the published fold (29.098441). Reusing one origin's
# ceiling for another is wrong in both directions, not just conservative.
P99_CEILING = {
    "main": {"E2": 28.467923, "E59": 27.996949, "E4": 29.098441},
    "r1": {"E2": 28.4230, "E59": 28.0000, "E4": 29.1026},
    "r2": {"E2": 28.4252, "E59": 27.9693, "E4": 29.0794},
}


# float32 round-trip slack of the residual CSV export. See the module docstring:
# 1e-6 is too tight and flags the legitimate at-the-ceiling rows.
CEILING_TOL = 1e-4

# The raw trip-gap cap the STALE (pre-fix) exports were capped at instead.
RAW_TRIP_GAP_CEILING = 30.0


def _fold_of(path: Path) -> str:
    """The origin a residual file came from, read off its filename.

    Family 21 tags every non-published origin into the stem
    (`lstm_contig_E4_r1_residuals_h3.csv`); everything else is the published
    fold.
    """
    for fold in ("r1", "r2"):
        if f"_{fold}_" in path.name:
            return fold
    return "main"


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

    fold = _fold_of(path)
    ceilings = P99_CEILING[fold]

    for corridor in corridors:
        # An unrecorded (origin, corridor) pair must stop the suite, not fall
        # back to another origin's ceiling: silently measuring against the wrong
        # ceiling is exactly the failure this table was widened to prevent.
        assert corridor in ceilings, (
            f"{path.name}: no recorded train p99 for corridor {corridor!r} at "
            f"origin {fold!r}. Read the `{corridor}: winsor threshold=` line off "
            f"that kernel's log and add it to P99_CEILING[{fold!r}]."
        )
        y_true = frame.filter(pl.col("corridor") == corridor).get_column("y_true")
        ceiling = ceilings[corridor]

        above = int((y_true > ceiling + CEILING_TOL).sum())
        assert above == 0, (
            f"{path.name} [{corridor}@{fold}]: {above} rows above the train p99 "
            f"ceiling {ceiling} (max={y_true.max()}) — these residuals are STALE "
            "(pre-winsorization-fix) and every metric derived from them is invalid"
        )

        at_raw_cap = int((y_true == RAW_TRIP_GAP_CEILING).sum())
        assert at_raw_cap == 0, (
            f"{path.name} [{corridor}]: {at_raw_cap} rows sit exactly at the raw "
            f"trip-gap ceiling {RAW_TRIP_GAP_CEILING} — the unmistakable signature "
            "of a pre-winsorization-fix export"
        )
