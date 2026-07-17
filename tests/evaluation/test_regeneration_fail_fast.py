"""Regeneration must fail fast when downloaded Kaggle residuals are absent.

Post-Kaggle regeneration is gated on fresh per-sample residual CSVs being
present under ``docs/resultados/residuos-multihorizon/`` (PKR1). If a residual
directory is empty or missing, the residuals-derived report builders MUST raise
a clear "missing residuals" error instead of silently emitting empty/degenerate
CSVs from an empty residual dir.

These tests lock in the fail-fast contract for the two shared discovery
functions the report builders route through:

* ``significance.load_residuals`` — used by ``build_significance_table`` and
  ``build_volatility_table``.
* ``paired_audit.discover_residual_files`` — used by ``build_paired_audit``.
"""
from __future__ import annotations

import pytest

import src.build_significance_table as build_significance_table
import src.build_volatility_table as build_volatility_table
from src.evaluation.paired_audit import build_paired_metrics, discover_residual_files
from src.evaluation.significance import load_residuals


class TestSignificanceLoadFailsFast:
    def test_empty_dir_raises_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="no CSV matching .* found in"):
            load_residuals(tmp_path, pattern="**/*_residuals_*.csv")

    def test_nonexistent_dir_raises_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="no CSV matching .* found in"):
            load_residuals(tmp_path / "does-not-exist", pattern="**/*_residuals_*.csv")


class TestBuildersFailFastOnEmptyResiduals:
    def test_significance_builder_fails_fast(self, tmp_path):
        with pytest.raises(ValueError, match="no CSV matching .* found in"):
            build_significance_table.build(
                resid_dir=tmp_path, out_dir=tmp_path / "out"
            )
        # No degenerate CSV should be written when residuals are missing.
        assert not (tmp_path / "out" / "significance_multihorizon.csv").exists()

    def test_volatility_builder_fails_fast(self, tmp_path):
        with pytest.raises(ValueError, match="no CSV matching .* found in"):
            build_volatility_table.build(
                resid_dir=tmp_path, out_dir=tmp_path / "out"
            )
        assert not (tmp_path / "out" / "volatility_multihorizon.csv").exists()


class TestPairedAuditDiscoveryFailsFast:
    def test_discover_residual_files_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no residual CSVs for horizons"):
            discover_residual_files(tmp_path)

    def test_build_paired_metrics_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no residual CSVs for horizons"):
            build_paired_metrics(tmp_path)
