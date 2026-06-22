"""Build the multi-seed confidence-interval table — C2 (NB15).

Reads the per-seed result CSVs (``lstm_multiseed_h{H}.csv``, downloaded from the
four Kaggle kernels and versioned in docs/resultados/csv-multihorizon/) and
reduces the 5 seeds [42, 123, 456, 789, 999] of every experimental cell to
mean ± a Student-t 95% confidence interval.

This is the reproducible entrypoint that closes gap C2: it turns the raw per-seed
samples into the artifact a reviewer asks for — proof that the LSTM result is
stable across initialisation seeds, not a lucky draw. The narrow intervals here
also feed the error bars on the degradation curve (build_degradation_curve.py).

Usage:
    uv run python -m src.build_multiseed_table

Output (versioned — small, paper reproducibility):
    docs/resultados/csv-multihorizon/multiseed_ci_multihorizon.csv
    columns: corridor, direction, baseline, metric, horizon, n_seeds,
             mean, std, ci_half, ci_low, ci_high, cv_pct
    One row per corridor × direction × metric × horizon (48 rows: 2×3×2×4).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.evaluation.multiseed import load_multiseed, multiseed_summary

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
CONFIDENCE = 0.95


def build(results_dir: Path = RESULTS_DIR, confidence: float = CONFIDENCE) -> Path:
    """Summarise the per-seed CSVs and write the consolidated CI table.

    Returns the path to the written CSV.
    """
    df = load_multiseed(results_dir)
    summary = multiseed_summary(df, confidence=confidence)

    out_csv = results_dir / "multiseed_ci_multihorizon.csv"
    summary.write_csv(out_csv)
    return out_csv


if __name__ == "__main__":
    out = build()
    print(f"Tabla de IC multi-seed escrita en {out}")
    summary = pl.read_csv(out)
    print(
        f"\nResumen C2: {summary.height} celdas (corredor×dirección×métrica×horizonte), "
        f"5 seeds c/u."
    )
    print(
        f"CV máximo entre seeds: {summary['cv_pct'].max():.2f}%  "
        f"(IC95 ±{summary['ci_half'].max():.4f} min en el peor caso)"
    )
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=4):
        print(summary.sort(["metric", "corridor", "direction", "horizon"]))
