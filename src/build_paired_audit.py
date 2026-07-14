"""Build the paired residual metric audit CSVs.

Usage:
    uv run python -m src.build_paired_audit

Inputs:
    docs/resultados/residuos-multihorizon/<model-dir>/h{3,5,10}/*_residuals_h*.csv
    docs/resultados/csv-multihorizon/*_results_h*.csv
    docs/resultados/csv-multihorizon/baselines*_results_multih.csv

Outputs:
    docs/resultados/csv-multihorizon/paired_dl_persistence_metrics.csv
    docs/resultados/csv-multihorizon/paired_vs_reported_audit.csv

This script reads residual files sequentially and does not train models or
regenerate notebooks.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl

from src.evaluation.paired_audit import audit_against_reported, build_paired_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
RESID_DIR = REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"

PAIRED_METRICS_CSV = "paired_dl_persistence_metrics.csv"
AUDIT_CSV = "paired_vs_reported_audit.csv"


def build(resid_dir: Path = RESID_DIR, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """Write paired metrics and paired-vs-reported audit CSVs."""
    paired = build_paired_metrics(resid_dir)
    audit = audit_against_reported(paired, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    paired_path = out_dir / PAIRED_METRICS_CSV
    audit_path = out_dir / AUDIT_CSV
    paired.write_csv(paired_path)
    audit.write_csv(audit_path)
    return paired_path, audit_path


def _print_summary(paired_path: Path, audit_path: Path) -> None:
    paired = pl.read_csv(paired_path)
    audit = pl.read_csv(audit_path)
    max_abs_diff = audit.select(
        pl.max_horizontal("abs_diff_dl", "abs_diff_persist").max()
    ).item()
    mismatches = audit.filter(pl.col("sign_mismatch") == True)

    print(f"Wrote paired metrics: {paired_path} ({paired.height} rows)")
    print(f"Wrote paired audit: {audit_path} ({audit.height} rows)")
    print(f"Max absolute reported-vs-paired difference: {max_abs_diff}")
    print(f"Sign mismatches: {mismatches.height}")
    if mismatches.height:
        with pl.Config(tbl_rows=mismatches.height, tbl_width_chars=200):
            print(mismatches.select(["model", "corridor", "horizon", "metric"]))


if __name__ == "__main__":
    outputs = build()
    _print_summary(*outputs)
