"""Build the multi-horizon degradation curve figure — Fase 6.5.

Reads the consolidated multi-horizon results from docs/resultados/csv-multihorizon/,
exports a tidy master CSV, and renders the central paper figure: forecast error
(MAE and RMSE) versus horizon, comparing the persistence baseline against the
three deep models, per corridor.

Usage:
    uv run python -m src.build_degradation_curve

Outputs (written to docs/resultados/):
    consolidated_multihorizon.csv   — single tidy table (all models/horizons)
    curva-degradacion.png           — 2x2 grid: rows MAE/RMSE, cols E2/E59

Note: statistical significance (Diebold-Mariano / Wilcoxon) is intentionally NOT
computed here — it needs per-sample errors, which the result CSVs do not contain.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in WSL/CI
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.degradation import degradation_table, load_results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados"

# Models to plot, in legend order. Persistence (B1) is the reference the deep
# models must beat; B3 is the strongest statistical baseline (see fase-6b doc).
MODELS = [
    ("B1", "Persistencia (B1)", "tab:gray", "o", "--", 2.4),
    ("B3", "Mejor baseline est. (B3)", "tab:olive", "s", ":", 1.6),
    ("LSTM", "LSTM", "tab:blue", "^", "-", 1.8),
    ("SpatialConvLSTM", "ConvLSTM", "tab:green", "D", "-", 1.8),
    ("SpatialTransformer", "Transformer", "tab:red", "v", "-", 1.8),
]
METRICS = ["MAE", "RMSE"]
CORRIDORS = ["E2", "E59"]


def _horizon_axis(table_cols: list[str]) -> list[int]:
    """Extract sorted horizon integers from h{H} column names."""
    return sorted(int(c[1:]) for c in table_cols if c.startswith("h"))


def build(results_dir: Path = RESULTS_DIR, out_dir: Path = OUT_DIR) -> Path:
    """Consolidate results, export the master CSV, render the figure.

    Returns the path to the written figure.
    """
    df = load_results(results_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "consolidated_multihorizon.csv"
    df.sort(["corridor", "metric", "baseline", "horizon"]).write_csv(master_csv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, metric in enumerate(METRICS):
        for col, corridor in enumerate(CORRIDORS):
            ax = axes[row][col]
            table = degradation_table(
                df, metric=metric, direction="aggregate", corridor=corridor
            )
            horizons = _horizon_axis(table.columns)
            present = set(table["baseline"])
            for name, label, color, marker, ls, lw in MODELS:
                if name not in present:
                    continue
                row_df = table.filter(table["baseline"] == name)
                xs = [h for h in horizons if row_df[f"h{h}"].item() is not None]
                ys = [row_df[f"h{h}"].item() for h in xs]
                ax.plot(
                    xs, ys, label=label, color=color, marker=marker,
                    linestyle=ls, linewidth=lw, markersize=6,
                )
            ax.set_title(f"{corridor} — {metric} agregado")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(horizons)
            if row == 1:
                ax.set_xlabel("Horizonte (minutos)")
            if col == 0:
                ax.set_ylabel(f"{metric} (minutos)")

    fig.suptitle(
        "Curva de degradación por horizonte — persistencia vs. modelos profundos",
        y=0.99, fontsize=13,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=len(labels), frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    fig_path = out_dir / "curva-degradacion.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    out = build()
    print(f"Figura escrita en {out}")
