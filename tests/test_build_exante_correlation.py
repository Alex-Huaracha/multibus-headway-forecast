"""Tests for src/build_exante_correlation.py — O1 anti-circularity analysis.

Validates that the ex-ante volatility stratifier (§5.2) is NOT a disguised
version of the retrospective regime (Figure 2). The correlation between
sigma(input window) and |y_real - persistence| is moderate (~0.25 Pearson,
~0.22 Spearman), and the high ex-ante tercile only lifts retrospective-high
incidence by ~1.1–1.3x.

Because the full materialization pipeline is expensive (reads large parquets),
these tests ONLY cover pure functions using small synthetic arrays:
  - classify_retro_regime: fixed-minute-cut labelling
  - compute_exante_terciles: frozen-threshold tercile assignment
  - compute_lift: lift of retrospective-high in the high ex-ante tercile
  - compute_correlation_stats: Pearson r, Spearman rho, r^2
  - build_csv_row: assembles one dict with the expected CSV schema
  - CSV schema: expected columns present in each row

Acceptance criteria:
  AC-REGIME-1  classify_retro_regime bins by fixed minute cuts <1/1-3/>=3
  AC-REGIME-2  classify_retro_regime preserves array length
   AC-TERCILE-1 compute_exante_terciles assigns 0/1/2 by frozen p33/p66 thresholds
  AC-TERCILE-2 compute_exante_terciles handles ties at boundaries correctly
  AC-LIFT-1    compute_lift returns lift = P(retro_high | ex_high) / P(retro_high)
  AC-LIFT-2    compute_lift returns NaN when marginal P(retro_high) == 0
  AC-CORR-1    compute_correlation_stats returns Pearson r, Spearman rho, r^2, n
  AC-CORR-2    compute_correlation_stats matches known analytical values
  AC-ROW-1     build_csv_row produces all required CSV column names
  AC-ROW-2     build_csv_row fractions within high-tercile sum to 1.0
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.build_exante_correlation import (
    classify_retro_regime,
    compute_exante_terciles,
    compute_lift,
    compute_correlation_stats,
    build_csv_row,
)
from src.evaluation.exante_terciles import compute_frozen_thresholds

# ---------------------------------------------------------------------------
# AC-REGIME-1 / AC-REGIME-2 — fixed-minute-cut labelling
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "corridor",
    "horizon",
    "n",
    "pearson_r",
    "spearman_rho",
    "r2",
    "frac_highexante_stable",
    "frac_highexante_moderate",
    "frac_highexante_high",
    "lift_high",
}


class TestClassifyRetroRegime:
    """AC-REGIME-1/2: fixed-minute-cut labelling of |y_real - persistence|."""

    def test_bins_by_minute_cuts(self):
        """AC-REGIME-1: <1 → stable(0), 1-3 → moderate(1), >=3 → high(2)."""
        persist_err = np.array([0.5, 0.0, 1.0, 2.0, 3.0, 4.5])
        labels = classify_retro_regime(persist_err)
        assert list(labels) == [0, 0, 1, 1, 2, 2]

    def test_preserves_array_length(self):
        """AC-REGIME-2: output shape matches input shape."""
        persist_err = np.array([0.1, 1.5, 5.0, 2.9, 0.99])
        labels = classify_retro_regime(persist_err)
        assert labels.shape == persist_err.shape

    def test_boundary_below_one_is_stable(self):
        """Values strictly below 1.0 are stable (0)."""
        labels = classify_retro_regime(np.array([0.999]))
        assert labels[0] == 0

    def test_boundary_exactly_one_is_moderate(self):
        """Exactly 1.0 → moderate (1). Boundary belongs to moderate."""
        labels = classify_retro_regime(np.array([1.0]))
        assert labels[0] == 1

    def test_boundary_exactly_three_is_high(self):
        """Exactly 3.0 → high (2). Boundary belongs to high."""
        labels = classify_retro_regime(np.array([3.0]))
        assert labels[0] == 2


# ---------------------------------------------------------------------------
# AC-TERCILE-1 / AC-TERCILE-2 — frozen-threshold tercile assignment
# ---------------------------------------------------------------------------


class TestComputeExanteTerciles:
    """AC-TERCILE-1/2: ex-ante tercile assignment by frozen p33/p66."""

    def test_assigns_terciles_by_percentiles(self):
        """AC-TERCILE-1: values ≤p33 → 0, (p33, p66] → 1, >p66 → 2."""
        # 9 values: sorted as 1..9; p33~3, p66~6
        ex = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        terciles = compute_exante_terciles(ex, compute_frozen_thresholds(ex))
        assert terciles.shape == ex.shape
        # lowest values should be in tercile 0, highest in tercile 2
        assert terciles[0] == 0
        assert terciles[-1] == 2

    def test_roughly_equal_bin_sizes(self):
        """AC-TERCILE-1: with uniform data each tercile gets ~1/3 of samples."""
        rng = np.random.default_rng(42)
        ex = rng.uniform(0, 10, size=300)
        terciles = compute_exante_terciles(ex, compute_frozen_thresholds(ex))
        for t in range(3):
            frac = float((terciles == t).mean())
            assert 0.28 < frac < 0.38, f"tercile {t} fraction {frac:.3f} out of expected range"

    def test_all_same_value_handled(self):
        """AC-TERCILE-2: degenerate input (all values identical) does not crash."""
        ex = np.ones(30)
        terciles = compute_exante_terciles(ex, compute_frozen_thresholds(ex))
        assert terciles.shape == (30,)


# ---------------------------------------------------------------------------
# AC-LIFT-1 / AC-LIFT-2 — lift computation
# ---------------------------------------------------------------------------


class TestComputeLift:
    """AC-LIFT-1/2: lift = P(retro_high | ex_high) / P(retro_high)."""

    def test_lift_perfect_overlap(self):
        """AC-LIFT-1: if high ex-ante → always retro-high, lift == 1/marginal."""
        # All samples in high ex-ante tercile (t==2) have retro=2
        # Marginal P(retro_high) = 0.5 → lift = 1.0/0.5 = 2.0
        retro = np.array([0, 0, 2, 2])
        ex_terciles = np.array([0, 0, 2, 2])
        lift = compute_lift(retro, ex_terciles)
        assert lift == pytest.approx(2.0)

    def test_lift_no_enrichment(self):
        """AC-LIFT-1: if high ex-ante tercile has same retro-high rate as overall → lift=1."""
        # 50% retro-high everywhere, including in high ex-ante tercile
        retro = np.array([2, 0, 2, 0, 2, 0])
        ex_terciles = np.array([0, 0, 1, 1, 2, 2])
        lift = compute_lift(retro, ex_terciles)
        assert lift == pytest.approx(1.0)

    def test_lift_nan_when_no_marginal_high(self):
        """AC-LIFT-2: if no retro-high samples exist, lift is NaN."""
        retro = np.array([0, 0, 1, 1, 0, 1])
        ex_terciles = np.array([0, 0, 1, 1, 2, 2])
        lift = compute_lift(retro, ex_terciles)
        assert math.isnan(lift)

    def test_lift_nan_when_no_high_exante(self):
        """AC-LIFT-2: if no high ex-ante samples, lift is NaN."""
        retro = np.array([2, 0, 1])
        ex_terciles = np.array([0, 0, 1])  # no tercile-2 samples
        lift = compute_lift(retro, ex_terciles)
        assert math.isnan(lift)


# ---------------------------------------------------------------------------
# AC-CORR-1 / AC-CORR-2 — correlation statistics
# ---------------------------------------------------------------------------


class TestComputeCorrelationStats:
    """AC-CORR-1/2: Pearson r, Spearman rho, r^2, n."""

    def test_returns_required_keys(self):
        """AC-CORR-1: result dict has pearson_r, spearman_rho, r2, n."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_correlation_stats(x, y)
        for key in ("pearson_r", "spearman_rho", "r2", "n"):
            assert key in result, f"Missing key: {key}"

    def test_perfect_positive_correlation(self):
        """AC-CORR-2: perfectly correlated arrays → r=1, rho=1, r2=1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x * 2.0 + 1.0
        result = compute_correlation_stats(x, y)
        assert result["pearson_r"] == pytest.approx(1.0)
        assert result["spearman_rho"] == pytest.approx(1.0)
        assert result["r2"] == pytest.approx(1.0)
        assert result["n"] == 5

    def test_zero_correlation(self):
        """AC-CORR-2: uncorrelated arrays → r ~ 0."""
        rng = np.random.default_rng(0)
        x = np.arange(100, dtype=float)
        y = rng.permutation(x)  # shuffle destroys correlation
        result = compute_correlation_stats(x, y)
        assert abs(result["pearson_r"]) < 0.3
        assert result["n"] == 100

    def test_r2_equals_pearson_squared(self):
        """AC-CORR-2: r^2 == pearson_r^2 always."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(50)
        y = 0.3 * x + rng.standard_normal(50)
        result = compute_correlation_stats(x, y)
        assert result["r2"] == pytest.approx(result["pearson_r"] ** 2, abs=1e-10)


# ---------------------------------------------------------------------------
# AC-ROW-1 / AC-ROW-2 — CSV row assembly
# ---------------------------------------------------------------------------


class TestBuildCsvRow:
    """AC-ROW-1/2: build_csv_row assembles correct schema."""

    def _make_inputs(self):
        """Synthetic arrays for one corridor×horizon cell."""
        rng = np.random.default_rng(99)
        n = 120
        ex_ante = rng.exponential(1.0, n)
        persist_err = rng.exponential(2.0, n)
        return ex_ante, persist_err

    @staticmethod
    def _thresholds(ex_ante):
        return compute_frozen_thresholds(ex_ante[np.isfinite(ex_ante)])

    def test_row_has_all_expected_columns(self):
        """AC-ROW-1: all 10 expected column names present in the returned dict."""
        ex_ante, persist_err = self._make_inputs()
        row = build_csv_row("E2", 3, ex_ante, persist_err, self._thresholds(ex_ante))
        for col in EXPECTED_COLUMNS:
            assert col in row, f"Column '{col}' missing from build_csv_row output"

    def test_row_corridor_and_horizon(self):
        """AC-ROW-1: corridor and horizon are passed through correctly."""
        ex_ante, persist_err = self._make_inputs()
        row = build_csv_row("E59", 10, ex_ante, persist_err, self._thresholds(ex_ante))
        assert row["corridor"] == "E59"
        assert row["horizon"] == 10

    def test_high_tercile_fractions_sum_to_one(self):
        """AC-ROW-2: frac_highexante_{stable,moderate,high} sum to 1.0."""
        ex_ante, persist_err = self._make_inputs()
        row = build_csv_row("E4", 5, ex_ante, persist_err, self._thresholds(ex_ante))
        total = (
            row["frac_highexante_stable"]
            + row["frac_highexante_moderate"]
            + row["frac_highexante_high"]
        )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_n_matches_non_nan_count(self):
        """AC-ROW-1: n is the count of non-NaN ex_ante values."""
        ex_ante, persist_err = self._make_inputs()
        # inject some NaNs
        ex_ante_with_nan = ex_ante.copy()
        ex_ante_with_nan[:5] = np.nan
        row = build_csv_row(
            "E2", 3, ex_ante_with_nan, persist_err, self._thresholds(ex_ante_with_nan)
        )
        assert row["n"] == int((~np.isnan(ex_ante_with_nan)).sum())
