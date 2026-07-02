"""Tests for src/evaluation/significance.py — paired significance tests (Fase 6.5).

The multi-horizon residual export (NB11/12/13) leaves one CSV per model/horizon
holding PER-SAMPLE predictions on the paired test windows, with schema
    corridor, direction, horizon, y_true, y_pred_dl, y_pred_persist

This module turns those per-sample errors into paired significance statistics
(Diebold-Mariano + Wilcoxon signed-rank) over the loss differential
    d_i = loss(y_true, y_pred_dl) - loss(y_true, y_pred_persist)
so a NEGATIVE mean differential means the DL model beats persistence.

Because n is huge (~millions of rows), the p-value will be tiny almost no matter
what; the paper argues from the EFFECT SIZE (Δ MAE), and these tests assert both
the sign of the statistic and that the effect-size column is reported.

Acceptance criteria:
    AC-LOAD-1   load_residuals concatenates every CSV in the dir into one frame
    AC-LOAD-2   load_residuals raises on missing dir / no CSVs
    AC-LOAD-3   load_residuals raises if a CSV has the wrong schema
    AC-LOAD-4   load_residuals(pattern=...) loads only matching CSVs, so a
                co-located *_results_*.csv (different schema) is skipped
    AC-LOSS-1   loss_differential (MAE)  = |e_dl| - |e_persist|, sign convention
    AC-LOSS-2   loss_differential (RMSE) = e_dl**2 - e_persist**2
    AC-DM-1     diebold_mariano on a clearly DL-better sample -> stat < 0, p small
    AC-DM-2     diebold_mariano on zero-mean noise -> large p (not significant)
    AC-DM-3     HAC (Newey-West) variance with lag>0 differs from the iid variance
    AC-WX-1     wilcoxon_signed_rank returns a p-value in [0, 1]
    AC-TABLE-1  significance_table: one row per (corridor, horizon)
    AC-TABLE-2  significance_table reports n, delta_loss, delta_mae and p-values
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation.significance import (
    load_residuals,
    loss_differential,
    diebold_mariano,
    wilcoxon_signed_rank,
    significance_table,
)

RESID_SCHEMA = [
    "corridor",
    "direction",
    "horizon",
    "y_true",
    "y_pred_dl",
    "y_pred_persist",
]


def _resid_frame(corridor, horizon, y_true, y_dl, y_persist, direction=None):
    """Build a residual frame. ``direction`` defaults to alternating -1/1 to
    mirror the real per-sample schema (integer directions, no 'aggregate' row)."""
    n = len(y_true)
    if direction is None:
        direction = [(-1 if i % 2 == 0 else 1) for i in range(n)]
    elif not isinstance(direction, list):
        direction = [direction] * n
    return pl.DataFrame(
        {
            "corridor": [corridor] * n,
            "direction": direction,
            "horizon": [horizon] * n,
            "y_true": list(y_true),
            "y_pred_dl": list(y_dl),
            "y_pred_persist": list(y_persist),
        }
    )


@pytest.fixture
def residuals_dir(tmp_path):
    """Two residual CSVs: DL beats persistence on E2/h3, ties on E59/h3."""
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.normal(0, 5, n)
    # E2: DL error half of persistence error -> DL clearly better
    e2 = _resid_frame(
        "E2", 3, y_true, y_true + rng.normal(0, 1, n), y_true + rng.normal(0, 2, n)
    )
    e2.write_csv(tmp_path / "lstm_residuals_h3.csv")
    # E59: identical predictions -> no difference
    same = y_true + rng.normal(0, 1.5, n)
    e59 = _resid_frame("E59", 3, y_true, same, same)
    e59.write_csv(tmp_path / "convlstm_residuals_h3.csv")
    return tmp_path


# --- AC-LOAD ---------------------------------------------------------------
class TestLoadResiduals:
    def test_concatenates_all_csvs(self, residuals_dir):
        """AC-LOAD-1: every row from both CSVs present, schema preserved."""
        df = load_residuals(residuals_dir)
        assert df.columns == RESID_SCHEMA
        assert df.height == 1000
        assert set(df["corridor"].unique()) == {"E2", "E59"}

    def test_missing_dir_raises(self, tmp_path):
        """AC-LOAD-2: empty dir -> ValueError."""
        with pytest.raises(ValueError, match="no CSV"):
            load_residuals(tmp_path)

    def test_wrong_schema_raises(self, tmp_path):
        """AC-LOAD-3: a CSV missing required columns -> ValueError."""
        pl.DataFrame({"foo": [1]}).write_csv(tmp_path / "bad.csv")
        with pytest.raises(ValueError, match="schema"):
            load_residuals(tmp_path)

    def test_pattern_skips_results_csv(self, tmp_path):
        """AC-LOAD-4: a co-located results CSV (Kaggle layout) is skipped by pattern.

        The Kaggle download leaves both ``*_residuals_*.csv`` and the
        aggregate-schema ``*_results_*.csv`` in the same folder. A glob over
        ``*.csv`` would choke on the results file; the pattern selects only the
        per-sample residuals.
        """
        _resid_frame("E2", 3, [10.0], [11.0], [13.0]).write_csv(
            tmp_path / "lstm_residuals_h3.csv"
        )
        # aggregate-schema results file with an incompatible column set
        pl.DataFrame(
            {
                "corridor": ["E2"],
                "direction": ["aggregate"],
                "baseline": ["LSTM"],
                "metric": ["MAE"],
                "value": [5.1],
                "horizon": [3],
            }
        ).write_csv(tmp_path / "lstm_results_h3.csv")

        # default glob would hit the wrong-schema results CSV and raise
        with pytest.raises(ValueError, match="schema"):
            load_residuals(tmp_path)
        # pattern selects only the residuals file
        df = load_residuals(tmp_path, pattern="*_residuals_*.csv")
        assert df.columns == RESID_SCHEMA
        assert df.height == 1


# --- AC-LOSS ---------------------------------------------------------------
class TestLossDifferential:
    def test_mae_differential(self):
        """AC-LOSS-1: d = |e_dl| - |e_persist|; negative when DL is closer."""
        df = _resid_frame("E2", 3, [10.0], [11.0], [13.0])  # |1| - |3| = -2
        d = loss_differential(df, metric="MAE")
        assert d[0] == pytest.approx(-2.0)

    def test_rmse_differential(self):
        """AC-LOSS-2: d = e_dl**2 - e_persist**2."""
        df = _resid_frame("E2", 3, [10.0], [11.0], [13.0])  # 1 - 9 = -8
        d = loss_differential(df, metric="RMSE")
        assert d[0] == pytest.approx(-8.0)


# --- AC-DM -----------------------------------------------------------------
class TestDieboldMariano:
    def test_dl_better_is_significant(self):
        """AC-DM-1: strongly negative differential -> stat < 0 and small p."""
        d = np.full(400, -2.0) + np.random.default_rng(1).normal(0, 0.1, 400)
        res = diebold_mariano(d)
        assert res.stat < 0
        assert res.p_value < 0.01

    def test_zero_mean_not_significant(self):
        """AC-DM-2: zero-mean noise -> large p."""
        d = np.random.default_rng(2).normal(0, 1, 400)
        res = diebold_mariano(d)
        assert res.p_value > 0.05

    def test_hac_lag_changes_variance(self):
        """AC-DM-3: positively autocorrelated d -> HAC stat differs from iid."""
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, 400)
        d = base + np.roll(base, 1)  # strong lag-1 autocorrelation
        iid = diebold_mariano(d, lag=0)
        hac = diebold_mariano(d, lag=5)
        assert iid.stat != pytest.approx(hac.stat)


# --- AC-WX -----------------------------------------------------------------
class TestWilcoxon:
    def test_returns_pvalue(self):
        """AC-WX-1: p-value in [0, 1]."""
        d = np.random.default_rng(4).normal(-1, 1, 200)
        p = wilcoxon_signed_rank(d)
        assert 0.0 <= p <= 1.0


# --- AC-TABLE --------------------------------------------------------------
class TestSignificanceTable:
    def test_one_row_per_group(self, residuals_dir):
        """AC-TABLE-1: one row per (corridor, horizon)."""
        df = load_residuals(residuals_dir)
        table = significance_table(df, metric="MAE")
        assert table.height == 2
        assert set(table["corridor"]) == {"E2", "E59"}

    def test_reports_effect_size_and_pvalues(self, residuals_dir):
        """AC-TABLE-2: n, delta_loss, delta_mae and p-value columns present."""
        df = load_residuals(residuals_dir)
        table = significance_table(df, metric="MAE")
        for col in ("n", "delta_loss", "delta_mae", "dm_stat", "dm_p", "wilcoxon_p"):
            assert col in table.columns
        e2 = table.filter(pl.col("corridor") == "E2")
        assert e2["delta_loss"].item() == pytest.approx(e2["delta_mae"].item())
        assert e2["delta_mae"].item() < 0  # DL better on E2

    def test_rmse_rows_use_metric_loss_for_verdict_but_keep_mae_effect_size(self) -> None:
        """RMSE rows must not reuse the MAE sign for dl_better."""
        df = _resid_frame(
            "E4",
            3,
            y_true=[0.0, 0.0],
            y_dl=[3.0, 3.0],
            y_persist=[0.0, 5.0],
        )
        table = significance_table(df, metric="RMSE")
        row = table.row(0, named=True)

        assert row["delta_mae"] > 0
        assert row["delta_loss"] < 0
        assert row["dl_better"] is True
