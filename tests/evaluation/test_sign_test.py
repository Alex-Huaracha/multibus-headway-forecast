"""Cell-level sign test for DL-vs-XGBoost — function contract + committed CSV.

The per-sample DM/Wilcoxon test is not applicable to DL-vs-XGBoost (the exports
differ in granularity and share no per-sample key, and the DL residuals overcount
each target ~4.5x via overlapping windows). `sign_test_across_cells` gives a
coarse but valid alternative: one fair-coin trial per (corridor, horizon) cell.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from src.evaluation.significance import SignTestResult, sign_test_across_cells

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIGNTEST_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "xgb_vs_lstm_signtest.csv"
)


def test_all_wins_gives_binomial_pvalue() -> None:
    """8/8 wins → one-sided p = (1/2)^8, two-sided = 2*(1/2)^8."""
    model = [1.0] * 8  # lower loss everywhere → model wins all 8 cells
    reference = [2.0] * 8
    res = sign_test_across_cells(model, reference)
    assert isinstance(res, SignTestResult)
    assert res.n_cells == 8
    assert res.n_model_wins == 8
    assert res.n_ties == 0
    assert math.isclose(res.p_one_sided, 0.5**8, rel_tol=1e-9)
    assert math.isclose(res.p_two_sided, 2 * 0.5**8, rel_tol=1e-9)


def test_minority_wins_is_not_significant() -> None:
    """1 win of 4 (the E4 shape) → the model is NOT favored one-sided."""
    model = [1.0, 3.0, 3.0, 3.0]  # wins only cell 0
    reference = [2.0, 2.0, 2.0, 2.0]
    res = sign_test_across_cells(model, reference)
    assert res.n_model_wins == 1
    assert res.p_one_sided > 0.5  # far from significant


def test_ties_are_dropped_from_the_binomial() -> None:
    """Exact ties do not count as wins or losses; the trial count shrinks."""
    model = [1.0, 1.0, 2.0]  # win, tie, loss
    reference = [2.0, 1.0, 1.0]
    res = sign_test_across_cells(model, reference)
    assert res.n_cells == 3
    assert res.n_model_wins == 1
    assert res.n_ties == 1
    # 1 win of 2 non-tie trials → two-sided p = 1.0
    assert math.isclose(res.p_two_sided, 1.0, rel_tol=1e-9)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        sign_test_across_cells([1.0, 2.0], [1.0])


def test_all_ties_raise() -> None:
    with pytest.raises(ValueError, match="every cell ties"):
        sign_test_across_cells([1.0, 2.0], [1.0, 2.0])


def test_committed_csv_matches_the_reported_claim() -> None:
    """The committed result underwrites the paper's significance statement."""
    assert SIGNTEST_CSV.exists(), f"missing {SIGNTEST_CSV}"
    df = pl.read_csv(SIGNTEST_CSV)
    rows = {r["group"]: r for r in df.iter_rows(named=True)}

    big = rows["E2+E59"]
    assert big["n_cells"] == 8
    assert big["n_lstm_wins"] == 8
    assert big["p_one_sided"] < 0.01  # significant: LSTM beats the leveled XGBoost

    e4 = rows["E4"]
    assert e4["n_cells"] == 4
    assert e4["n_lstm_wins"] == 1  # E4 is the scale caveat: XGBoost wins short/mid
    assert e4["p_one_sided"] > 0.5  # not significant for the LSTM
