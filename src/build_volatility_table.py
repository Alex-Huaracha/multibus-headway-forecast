"""Build the volatility-regime stratification table — Fase 7.

The paired-significance table (``build_significance_table.py``) proves the
DL-vs-persistence gap is real and grows with the horizon. This script explains
the MECHANISM a reviewer will ask about: persistence is only wrong when the
headway actually moves, so the DL advantage should concentrate in high-change
windows. It stratifies the SAME per-sample residuals by the realized headway
change ``|y_true - y_pred_persist|`` (``src.evaluation.volatility``) and reports
the effect size + significance within each volatility regime.

Usage:
    uv run python -m src.build_volatility_table

Inputs (downloaded from Kaggle, NOT versioned — see .gitignore):
    docs/resultados/residuos-multihorizon/<model-dir>/h{3,5,10}/*_residuals_*.csv

Output (versioned — small, paper reproducibility):
    docs/resultados/csv-multihorizon/volatility_multihorizon.csv
    columns: model, metric, corridor, horizon, regime, regime_order, n,
             mean_change, delta_mae, dm_stat, dm_p, wilcoxon_p, dl_better
    One row per model × metric × corridor × horizon × volatility regime.
    ``delta_mae`` is the headline effect size (always in MAE units); read the
    table by fixing (model, corridor, horizon) and walking the regimes from
    ``low`` to ``high`` — the gap should widen.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl

from src.evaluation.significance import load_residuals
from src.evaluation.volatility import DEFAULT_EDGES, volatility_significance_table

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
METRICS = ["MAE", "RMSE"]

OUTPUT_COLUMNS = [
    "model",
    "metric",
    "corridor",
    "horizon",
    "regime",
    "regime_order",
    "n",
    "mean_change",
    "delta_mae",
    "dm_stat",
    "dm_p",
    "wilcoxon_p",
    "dl_better",
]


def build(
    resid_dir: Path = RESID_DIR,
    out_dir: Path = OUT_DIR,
    edges: tuple[float, ...] = DEFAULT_EDGES,
) -> Path:
    """Stratify every model × metric by volatility regime and write the CSV.

    Returns the path to the written CSV.

    Raises
    ------
    ValueError
        If a model's residual directory holds no matching CSV.
    """
    tables: list[pl.DataFrame] = []
    for name, subdir in MODELS:
        df = load_residuals(resid_dir / subdir, pattern=RESIDUALS_GLOB)
        for metric in METRICS:
            table = volatility_significance_table(
                df, metric=metric, edges=edges
            ).with_columns(
                pl.lit(name).alias("model"), pl.lit(metric).alias("metric")
            )
            tables.append(table)

    consolidated = (
        pl.concat(tables, how="vertical")
        .select(OUTPUT_COLUMNS)
        .sort(["model", "metric", "corridor", "horizon", "regime_order"])
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "volatility_multihorizon.csv"
    consolidated.write_csv(out_csv)
    return out_csv


if __name__ == "__main__":
    out = build()
    print(f"Tabla de volatilidad escrita en {out}")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=4):
        print(pl.read_csv(out))
