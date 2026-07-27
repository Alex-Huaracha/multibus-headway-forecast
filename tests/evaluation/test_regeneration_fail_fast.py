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
* ``xgb_paired.validate_inputs`` / ``load_xgb_export`` — used by
  ``build_xgb_paired_metrics``, which additionally needs the per-sample NB20
  XGBoost export, not just the DL residuals.
"""
from __future__ import annotations

import pytest

import src.build_significance_table as build_significance_table
import src.build_volatility_table as build_volatility_table
import src.build_xgb_paired_metrics as build_xgb_paired_metrics
from src.evaluation.paired_audit import build_paired_metrics, discover_residual_files
from src.evaluation.significance import load_residuals
from src.evaluation.xgb_paired import (
    load_xgb_export,
    residual_csv_path,
    validate_inputs,
    xgb_export_path,
)


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


class TestXgbPairedFailsFast:
    def test_validate_inputs_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no per-sample inputs found"):
            validate_inputs(tmp_path)

    def test_validate_inputs_names_the_missing_xgb_export(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(tmp_path)
        assert str(xgb_export_path(tmp_path)) in str(excinfo.value)

    def test_validate_inputs_names_a_missing_residual_csv(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(tmp_path)
        assert str(residual_csv_path(tmp_path, "E4", 10)) in str(excinfo.value)

    def test_load_xgb_export_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no XGB paired export found in"):
            load_xgb_export(xgb_export_path(tmp_path))

    def test_load_xgb_export_rejects_a_wrong_schema(self, tmp_path):
        path = tmp_path / "xgb_paired_persample_test.csv"
        path.write_text("corridor,horizon\nE2,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="has wrong schema — missing"):
            load_xgb_export(path)

    def test_builder_fails_fast_before_writing_anything(self, tmp_path):
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="no per-sample inputs found"):
            build_xgb_paired_metrics.build(resid_dir=tmp_path, out_dir=out_dir)
        # No degenerate CSV, and no output directory, when inputs are missing.
        assert not out_dir.exists()
