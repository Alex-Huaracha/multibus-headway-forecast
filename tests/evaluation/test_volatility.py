"""Tests for src/evaluation/volatility.py — volatility-regime stratification (Fase 7).

The aggregate degradation curve shows DL beating persistence by a margin that
GROWS with the horizon. This module explains the MECHANISM: persistence only
fails when the headway actually changes, so the DL advantage should concentrate
in high-change windows. We stratify the per-sample residuals (same schema as
``significance.py``: corridor, direction, horizon, y_true, y_pred_dl,
y_pred_persist) by the REALIZED headway change

    headway_change = |y_true - y_pred_persist|

because persistence predicts ``y(t+H) = y(t)``, so its error magnitude IS the
realized change. Fixed minute thresholds (not quantiles) are used on purpose:
they tell two stories at once — within a horizon DL wins more on high-change
samples, AND the share of high-change samples grows with the horizon.

Acceptance criteria:
    AC-CHANGE-1   headway_change = |y_true - y_pred_persist|
    AC-REGIME-1   assign_volatility_regime bins by the minute edges
    AC-REGIME-2   assign_volatility_regime adds columns and preserves every row
    AC-TABLE-1    one row per (corridor, horizon, regime) actually present
    AC-TABLE-2    reports n, mean_change, delta_mae, dm_p, wilcoxon_p, dl_better
    AC-TABLE-3    the DL advantage (negative delta_mae) is stronger in the
                  high-change regime than in the stable regime — the paper claim
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation.volatility import (
    headway_change,
    assign_volatility_regime,
    volatility_significance_table,
)


def _resid_frame(corridor, horizon, y_true, y_dl, y_persist, direction=None):
    """Residual frame mirroring the real per-sample schema (integer directions)."""
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


# --- AC-CHANGE -------------------------------------------------------------
class TestHeadwayChange:
    def test_change_is_abs_persist_error(self):
        """AC-CHANGE-1: |y_true - y_pred_persist|, the realized headway change."""
        df = _resid_frame("E2", 3, [10.0, 5.0], [9.0, 6.0], [13.0, 5.5])
        change = headway_change(df)
        assert change[0] == pytest.approx(3.0)  # |10 - 13|
        assert change[1] == pytest.approx(0.5)  # |5 - 5.5|


# --- AC-REGIME -------------------------------------------------------------
class TestAssignRegime:
    def test_bins_by_edges(self):
        """AC-REGIME-1: edges=(1, 3) -> low <1, moderate 1..3, high >3."""
        df = _resid_frame(
            "E2",
            3,
            y_true=[10.0, 10.0, 10.0],
            y_dl=[10.0, 10.0, 10.0],
            y_persist=[10.5, 12.0, 15.0],  # change = 0.5, 2.0, 5.0
        )
        out = assign_volatility_regime(df, edges=(1.0, 3.0))
        assert out["volatility_regime"].to_list() == ["low", "moderate", "high"]

    def test_adds_columns_preserves_rows(self):
        """AC-REGIME-2: headway_change + volatility_regime added, height kept."""
        df = _resid_frame("E2", 3, [10.0] * 5, [10.0] * 5, [11.0] * 5)
        out = assign_volatility_regime(df, edges=(1.0, 3.0))
        assert out.height == df.height
        assert "headway_change" in out.columns
        assert "volatility_regime" in out.columns
        # every original column survives
        for col in df.columns:
            assert col in out.columns


# --- AC-TABLE --------------------------------------------------------------
@pytest.fixture
def stratified_residuals():
    """E2/h3 where the DL advantage lives in the high-change regime.

    - stable samples: headway barely moves (change ~0.3); DL == persistence,
      so delta_mae ~ 0.
    - high-change samples: headway jumps ~5 min; DL tracks it (small error),
      persistence carries the stale value (large error) -> delta_mae very
      negative.
    """
    rng = np.random.default_rng(0)
    n = 400
    y_stable = rng.normal(20, 0.5, n)
    persist_stable = y_stable - 0.3  # change ~0.3 (low regime)
    dl_stable = persist_stable  # DL ties persistence when nothing moves
    y_high = rng.normal(20, 0.5, n)
    persist_high = y_high - 5.0  # change ~5 (high regime)
    dl_high = y_high - 0.5  # DL tracks the jump, persistence does not
    df = pl.concat(
        [
            _resid_frame("E2", 3, y_stable, dl_stable, persist_stable),
            _resid_frame("E2", 3, y_high, dl_high, persist_high),
        ],
        how="vertical",
    )
    return df


class TestVolatilityTable:
    def test_one_row_per_present_regime(self, stratified_residuals):
        """AC-TABLE-1: one row per (corridor, horizon, regime) with samples."""
        table = volatility_significance_table(
            stratified_residuals, metric="MAE", edges=(1.0, 3.0)
        )
        assert set(table["regime"].unique()) <= {"low", "moderate", "high"}
        # both populated regimes (low and high) appear exactly once
        counts = table.group_by("regime").len().sort("regime")
        assert table.filter(pl.col("regime") == "low").height == 1
        assert table.filter(pl.col("regime") == "high").height == 1

    def test_reports_effect_and_pvalues(self, stratified_residuals):
        """AC-TABLE-2: required reporting columns present."""
        table = volatility_significance_table(
            stratified_residuals, metric="MAE", edges=(1.0, 3.0)
        )
        for col in (
            "corridor",
            "horizon",
            "regime",
            "n",
            "mean_change",
            "delta_mae",
            "dm_stat",
            "dm_p",
            "wilcoxon_p",
            "dl_better",
        ):
            assert col in table.columns

    def test_advantage_concentrates_in_high_change(self, stratified_residuals):
        """AC-TABLE-3: DL advantage is stronger (more negative) at high change."""
        table = volatility_significance_table(
            stratified_residuals, metric="MAE", edges=(1.0, 3.0)
        )
        low = table.filter(pl.col("regime") == "low")["delta_mae"].item()
        high = table.filter(pl.col("regime") == "high")["delta_mae"].item()
        assert high < low  # DL wins by more where the headway actually moved
        assert high < 0  # and DL genuinely beats persistence there
        assert low == pytest.approx(0.0, abs=0.05)  # ~tie when nothing moves
