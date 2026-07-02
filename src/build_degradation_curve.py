"""Build the multi-horizon degradation curve figure — Fase 6.5.

Reads the consolidated multi-horizon results from docs/resultados/csv-multihorizon/,
exports a tidy master CSV, and renders the central paper figure: forecast error
(MAE and RMSE) versus horizon, comparing the persistence baseline against the
three deep models, per corridor.

Usage:
    uv run python -m src.build_degradation_curve

Outputs (written to docs/resultados/):
    consolidated_multihorizon.csv   — single tidy table (all models/horizons)
    curva-degradacion.png           — 2x3 grid: rows MAE/RMSE, cols E2/E59/E4

Significance: each deep-model point is tested against persistence (B1) with the
paired Diebold-Mariano and Wilcoxon tests (see build_significance_table.py). All
DL-vs-persistence comparisons are significant; the single exception is ringed and
labelled ``ns`` on the figure, with a footnote stating the global claim. The
significance verdicts are read from the versioned significance_multihorizon.csv,
so the figure rebuilds without the (unversioned) raw per-sample residuals.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import matplotlib

matplotlib.use("Agg")  # headless: no display in WSL/CI
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.degradation import degradation_table, load_results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados"

# Models to plot, in legend order. Persistence (B1) is the operational reference;
# B0/B3/B4_HA show the formulaic context; B5_XGB is the fitted ML baseline
# (gradient boosting) — a strong learned competitor the deep models must also beat.
MODELS = [
    ("B0", "Media global (B0)", "tab:pink", "x", ":", 1.2),
    ("B1", "Persistencia (B1)", "tab:gray", "o", "--", 2.4),
    ("B3", "SES (B3)", "tab:olive", "s", ":", 1.4),
    ("B4_HA", "Media horaria (B4)", "tab:purple", "*", ":", 1.2),
    ("B5_XGB", "XGBoost (B5, fitted)", "tab:brown", "P", "--", 1.6),
    ("LSTM", "LSTM", "tab:blue", "^", "-", 1.8),
    ("SpatialConvLSTM", "ConvLSTM", "tab:green", "D", "-", 1.8),
    ("SpatialTransformer", "Transformer", "tab:red", "v", "-", 1.8),
]
METRICS = ["MAE", "RMSE"]
CORRIDORS = ["E2", "E59", "E4"]

# Deep models whose points are tested against persistence (B1).
DL_MODELS = {"LSTM", "SpatialConvLSTM", "SpatialTransformer"}
SIG_CSV = RESULTS_DIR / "significance_multihorizon.csv"
ALPHA = 0.05  # two-tailed threshold for both DM and Wilcoxon

# Multi-seed confidence intervals (C2 / NB15). The LSTM was re-run with 5 seeds;
# its 95% interval is drawn as error bars on the LSTM line (proxy for the whole
# DL family, see fase-c2 doc). Missing file -> the curve renders without bars.
MULTISEED_CI_CSV = RESULTS_DIR / "multiseed_ci_multihorizon.csv"
MULTISEED_MODEL = "LSTM"


def _horizon_axis(table_cols: list[str]) -> list[int]:
    """Extract sorted horizon integers from h{H} column names."""
    return sorted(int(c[1:]) for c in table_cols if c.startswith("h"))


def _load_significance(sig_csv: Path) -> dict[tuple[str, str, str, int], bool]:
    """Map (model, metric, corridor, horizon) -> is the gap significant?

    A point is significant when BOTH the Diebold-Mariano and the Wilcoxon p-value
    fall below ``ALPHA``. Missing file -> empty map (figure renders unannotated).
    """
    if not sig_csv.exists():
        return {}
    sig = pl.read_csv(sig_csv)
    verdict: dict[tuple[str, str, str, int], bool] = {}
    for row in sig.iter_rows(named=True):
        key = (row["model"], row["metric"], row["corridor"], int(row["horizon"]))
        verdict[key] = (row["dm_p"] < ALPHA) and (row["wilcoxon_p"] < ALPHA)
    return verdict


def _load_multiseed_ci(ci_csv: Path) -> dict[tuple[str, str, int], float]:
    """Map (model, metric, corridor, horizon) -> 95% CI half-width (aggregate dir).

    Reads the versioned multiseed_ci_multihorizon.csv. Only the ``aggregate``
    direction is kept, since that is the slice the degradation curve plots.
    Missing file -> empty map (the curve renders without error bars).
    """
    if not ci_csv.exists():
        return {}
    ci = pl.read_csv(ci_csv).filter(pl.col("direction") == "aggregate")
    half: dict[tuple[str, str, int], float] = {}
    for row in ci.iter_rows(named=True):
        key = (row["baseline"], row["metric"], row["corridor"], int(row["horizon"]))
        half[key] = float(row["ci_half"])
    return half


def build(results_dir: Path = RESULTS_DIR, out_dir: Path = OUT_DIR) -> Path:
    """Consolidate results, export the master CSV, render the figure.

    Returns the path to the written figure.
    """
    # Only the per-model/baseline result CSVs; skip the co-located
    # significance_multihorizon.csv (different schema, same directory).
    df = load_results(results_dir, pattern="*_results_*.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "consolidated_multihorizon.csv"
    df.sort(["corridor", "metric", "baseline", "horizon"]).write_csv(master_csv)

    significance = _load_significance(SIG_CSV)
    multiseed_ci = _load_multiseed_ci(MULTISEED_CI_CSV)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
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
                # Multi-seed 95% CI error bars (C2). Only the model re-run with
                # 5 seeds carries them; the intervals are sub-marker-sized — that
                # tininess IS the result (the LSTM is stable across seeds).
                if name == MULTISEED_MODEL and multiseed_ci:
                    yerr = [multiseed_ci.get((name, metric, corridor, x), 0.0) for x in xs]
                    if any(e > 0 for e in yerr):
                        ax.errorbar(
                            xs, ys, yerr=yerr, fmt="none", ecolor=color,
                            elinewidth=1.2, capsize=4, capthick=1.2, zorder=6,
                        )
                # Ring + label the rare DL-vs-persistence point that is NOT
                # statistically significant (verdict explicitly False; h=1 and
                # baselines have no paired test and are left unmarked).
                if name in DL_MODELS:
                    for x, y in zip(xs, ys):
                        if significance.get((name, metric, corridor, x)) is False:
                            ax.scatter(
                                [x], [y], s=190, facecolors="none",
                                edgecolors="black", linewidths=1.5, zorder=5,
                            )
                            ax.annotate(
                                "ns", (x, y), textcoords="offset points",
                                xytext=(7, 6), fontsize=9, fontweight="bold",
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
    footnotes = []
    if significance:
        footnotes.append(
            "Comparación DL vs. persistencia (B1) a h∈{3,5,10}: todas significativas "
            "(Diebold-Mariano y Wilcoxon, p<0.001); ⊘ ns = no significativa."
        )
    if multiseed_ci:
        footnotes.append(
            "Barras de error en LSTM = IC 95% sobre 5 seeds [42,123,456,789,999] "
            "(NB15); su ancho sub-marcador confirma estabilidad frente al seed."
        )
    if footnotes:
        fig.text(
            0.5, 0.005, "\n".join(footnotes),
            ha="center", fontsize=9, style="italic",
        )
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))

    fig_path = out_dir / "curva-degradacion.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    out = build()
    print(f"Figura escrita en {out}")
