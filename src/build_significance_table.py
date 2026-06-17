"""Build the paired-significance table — Fase 6.5 (§6.6 del paper).

The degradation curve (``build_degradation_curve.py``) shows DL beating
persistence at long horizons using AGGREGATED MAE. This script answers the
reviewer's question — *is that gap statistically real?* — by running the paired
Diebold-Mariano and Wilcoxon tests (``src.evaluation.significance``) over the
PER-SAMPLE residuals exported by NB11/12/13 and downloaded from Kaggle.

It is the reproducible entrypoint that turns the tested library into the actual
paper artifact: one consolidated CSV with the effect size (Δ MAE) and the two
p-values per model × corridor × horizon.

Usage:
    uv run python -m src.build_significance_table

Inputs (downloaded from Kaggle, NOT versioned — see .gitignore):
    docs/resultados/residuos-multihorizon/<model-dir>/h{3,5,10}/*_residuals_*.csv

Output (versioned — small, paper reproducibility):
    docs/resultados/csv-multihorizon/significance_multihorizon.csv
    columns: model, corridor, horizon, n, delta_mae, dm_stat, dm_p,
             wilcoxon_p, dl_better

Note on n: with millions of paired samples both p-values collapse to ~0; the
paper leads with Δ MAE (effect size) and uses the p-values only to confirm the
sign is not noise. Watch for the honest exception (e.g. E59/h3 Wilcoxon).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.evaluation.significance import load_residuals, significance_table

REPO_ROOT = Path(__file__).resolve().parent.parent
RESID_DIR = REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"

# (display name, Kaggle download subdir). Persistence is the in-kernel reference
# already baked into every residual file as ``y_pred_persist``.
MODELS = [
    ("LSTM", "11-lstm"),
    ("SpatialConvLSTM", "12-spatialconvlstm"),
    ("SpatialTransformer", "13-spatialtransformer"),
]
RESIDUALS_GLOB = "**/*_residuals_*.csv"  # skip the co-located *_results_*.csv


def build(resid_dir: Path = RESID_DIR, out_dir: Path = OUT_DIR) -> Path:
    """Run the paired tests for every model and write the consolidated CSV.

    Returns the path to the written CSV.

    Raises
    ------
    ValueError
        If a model's residual directory holds no matching CSV.
    """
    tables: list[pl.DataFrame] = []
    for name, subdir in MODELS:
        df = load_residuals(resid_dir / subdir, pattern=RESIDUALS_GLOB)
        table = significance_table(df, metric="MAE").with_columns(
            pl.lit(name).alias("model")
        )
        tables.append(table)

    consolidated = pl.concat(tables, how="vertical").select(
        [
            "model",
            "corridor",
            "horizon",
            "n",
            "delta_mae",
            "dm_stat",
            "dm_p",
            "wilcoxon_p",
            "dl_better",
        ]
    ).sort(["model", "corridor", "horizon"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "significance_multihorizon.csv"
    consolidated.write_csv(out_csv)
    return out_csv


if __name__ == "__main__":
    out = build()
    print(f"Tabla de significancia escrita en {out}")
    with pl.Config(tbl_rows=30, tbl_width_chars=200, float_precision=4):
        print(pl.read_csv(out))
