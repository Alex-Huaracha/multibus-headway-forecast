"""Cell-level sign test for DL-vs-XGBoost — function contract + committed CSV.

Since NB20 exports per-sample XGBoost predictions keyed on
``(direction, t, pair_rank)``, a per-sample paired DM/Wilcoxon test IS available
and is emitted separately on the de-duplicated population
(``xgb_paired_significance.csv``). The sign test remains as the coarse companion
over the MULTIPLICITY-MATCHED population — the one whose MAE the paper prints —
where a naive per-sample n would treat the ~4.5 overlapping-window replicas of
each target as independent observations. One fair-coin trial per
``(corridor, horizon)`` cell is immune to that replication.

The committed CSV assertions below are the restricted result: they were 8/8 while
the builder compared an LSTM MAE and an XGBoost MAE computed over DIFFERENT
populations, and the true, population-matched count is 6/8.
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
    """The committed result underwrites the paper's significance statement.

    Every count here comes from the population-matched comparison in
    ``xgb_paired_dl_metrics.csv``. The headline is NOT significant: 6/8 for the
    large corridors (p = 37/256) and 7/12 pooled (p = 1586/4096). The paper must
    read as "no reliable difference", not "the LSTM wins".
    """
    assert SIGNTEST_CSV.exists(), f"missing {SIGNTEST_CSV}"
    df = pl.read_csv(SIGNTEST_CSV)
    rows = {r["group"]: r for r in df.iter_rows(named=True)}
    assert all(r["population"] == "multiplicity_matched" for r in rows.values())

    big = rows["E2+E59"]
    assert big["n_cells"] == 8
    assert big["n_lstm_wins"] == 6
    # P(X >= 6 | Binomial(8, 0.5)) = 37/256 — NOT significant at any usual level.
    assert math.isclose(big["p_one_sided"], 37 / 256, rel_tol=1e-9)
    assert big["p_one_sided"] > 0.10

    e4 = rows["E4"]
    assert e4["n_cells"] == 4
    assert e4["n_lstm_wins"] == 1  # E4 is the scale caveat: XGBoost wins short/mid
    assert e4["p_one_sided"] > 0.5  # not significant for the LSTM

    pooled = rows["pooled"]
    assert pooled["n_cells"] == 12
    assert pooled["n_lstm_wins"] == 7
    # P(X >= 7 | Binomial(12, 0.5)) = 1586/4096.
    assert math.isclose(pooled["p_one_sided"], 1586 / 4096, rel_tol=1e-9)
    assert pooled["p_one_sided"] > 0.10


def test_committed_csv_is_reproducible_from_the_paired_metrics() -> None:
    """The committed CSV must be what the rewired builder emits, not a leftover."""
    import src.build_xgb_vs_lstm_signtest as builder

    if not builder.PAIRED_METRICS_CSV.exists():
        pytest.skip(f"{builder.PAIRED_METRICS_CSV.name} not built yet")
    rebuilt = builder.build()
    committed = pl.read_csv(SIGNTEST_CSV)
    assert rebuilt.columns == committed.columns
    assert rebuilt.to_dicts() == committed.to_dicts()


def test_signtest_builder_refuses_the_mismatched_aggregate_fallback(tmp_path) -> None:
    """A missing paired-metrics CSV must fail loudly, never silently fall back."""
    import src.build_xgb_vs_lstm_signtest as builder

    with pytest.raises(ValueError, match="run .*build_xgb_paired_metrics"):
        builder.build(tmp_path / "xgb_paired_dl_metrics.csv")
