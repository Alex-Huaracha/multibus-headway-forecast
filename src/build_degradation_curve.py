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
paired Diebold-Mariano and Wilcoxon tests (see build_significance_table.py). The
verdicts are read from the versioned significance_multihorizon.csv, so the figure
rebuilds without the (unversioned) raw per-sample residuals.

Three annotation rules, all derived from the data rather than asserted:

* ``B1`` + square ring — persistence is significantly BETTER than the deep model
  (``dl_better`` is false). These must be marked: an earlier version of this
  builder tested only ``dm_p``/``wilcoxon_p`` and ignored ``dl_better``, so a
  significant win *for persistence* rendered as an unannotated point, implying
  the opposite of the truth. ``SpatialTransformer / E4 / h=3 / MAE`` is exactly
  such a cell and it sits inside the h∈{3,5,10} band the footnote describes.
* ``ns`` + circle ring — the gap is not significant at ALPHA in both tests.
* Nothing — the deep model wins significantly.

Framing caveat, stated on the figure itself: these panels plot the AGGREGATE
metric over the full test split, whereas the paper's canonical DL-vs-persistence
claim is the PAIRED one over identical samples. The two framings disagree in
sign for a handful of cells (read from paired_vs_reported_audit.csv and counted
into the footnote), all of them at the crossover where the true margin is
smaller than the framing bias. Every such cell also has ``dl_better`` false, so
the ``B1`` marker already flags it visually; the footnote names the cause.

The significance footnote is COMPUTED from significance_multihorizon.csv, never
hardcoded — the previous text claimed "todas significativas ... p<0.001", which
was false in three separate ways.
"""
from __future__ import annotations

import os

# Byte-identical consolidated CSV across runs (see CLAUDE.md determinism
# contract). Must precede the polars import.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402
import matplotlib  # noqa: E402

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
# Paired-vs-aggregate reconciliation: names the cells where this figure's framing
# disagrees in sign with the paper's canonical paired base.
PAIRED_AUDIT_CSV = RESULTS_DIR / "paired_vs_reported_audit.csv"
ALPHA = 0.05  # two-tailed threshold for both DM and Wilcoxon

# Verdict markers. "B1"/"B1≠" (persistence wins / and the paired framing also
# reverses this panel) are deliberately louder than "ns": a reader skimming the
# panels is far more likely to misread a persistence win as a DL win than to
# over-read a non-significant gap.
MARK_STYLES = {
    "B1":  {"marker": "s", "color": "darkred", "s": 210, "lw": 1.8},
    "B1≠": {"marker": "s", "color": "darkred", "s": 210, "lw": 1.8},
    "ns":  {"marker": "o", "color": "black", "s": 190, "lw": 1.5},
}

# Footnote wrapping width in characters. The figure is 16 in wide at fontsize 9;
# unwrapped captions ran off both edges and were silently clipped.
FOOTNOTE_WRAP = 155

# Multi-seed confidence intervals (C2 / NB15). The LSTM was re-run with 5 seeds;
# its 95% interval is drawn as error bars on the LSTM line (proxy for the whole
# DL family, see fase-c2 doc). Missing file -> the curve renders without bars.
MULTISEED_CI_CSV = RESULTS_DIR / "multiseed_ci_multihorizon.csv"
MULTISEED_MODEL = "LSTM"


def _horizon_axis(table_cols: list[str]) -> list[int]:
    """Extract sorted horizon integers from h{H} column names."""
    return sorted(int(c[1:]) for c in table_cols if c.startswith("h"))


def _load_significance(
    sig_csv: Path,
) -> dict[tuple[str, str, str, int], tuple[bool, bool]]:
    """Map (model, metric, corridor, horizon) -> (is_significant, dl_better).

    A point is significant when BOTH the Diebold-Mariano and the Wilcoxon p-value
    fall below ``ALPHA``. ``dl_better`` carries the DIRECTION of the effect and is
    kept separate on purpose: significance without direction cannot distinguish
    "the deep model wins" from "persistence wins", and conflating them is what
    made an earlier version of this figure assert the reverse of its own table.
    Missing file -> empty map (figure renders unannotated).
    """
    if not sig_csv.exists():
        return {}
    sig = pl.read_csv(sig_csv)
    verdict: dict[tuple[str, str, str, int], tuple[bool, bool]] = {}
    for row in sig.iter_rows(named=True):
        key = (row["model"], row["metric"], row["corridor"], int(row["horizon"]))
        significant = (row["dm_p"] < ALPHA) and (row["wilcoxon_p"] < ALPHA)
        verdict[key] = (significant, bool(row["dl_better"]))
    return verdict


def _significance_footnote(sig_csv: Path, horizons: tuple[int, ...] = (3, 5, 10)) -> str:
    """Build the significance caption FROM the table, for the operational band.

    Counts, over the DL-vs-persistence cells at ``horizons``: how many favour the
    deep model, how many of those clear p<0.001 in both tests, how many only
    clear p<0.05, and how many favour persistence instead. Hardcoding this text
    is how the figure came to overstate its own evidence.
    """
    if not sig_csv.exists():
        return ""
    sig = pl.read_csv(sig_csv).filter(pl.col("horizon").is_in(list(horizons)))
    total = sig.height
    dl_win = sig.filter(pl.col("dl_better"))
    p001 = dl_win.filter((pl.col("dm_p") < 1e-3) & (pl.col("wilcoxon_p") < 1e-3)).height
    p05 = dl_win.filter((pl.col("dm_p") < ALPHA) & (pl.col("wilcoxon_p") < ALPHA)).height
    reversed_cells = total - dl_win.height
    band = "{" + ",".join(str(h) for h in horizons) + "}"
    parts = [
        f"DL vs. persistencia (B1) a h∈{band}: el DL tiene menor error en "
        f"{dl_win.height} de {total} celdas; {p001} a p<0.001 y {p05} a p<0.05 "
        f"en ambos tests (Diebold-Mariano y Wilcoxon)."
    ]
    if reversed_cells:
        parts.append(
            f"En {reversed_cells} celda(s) gana la persistencia (marcada B1 ▪)."
        )
    parts.append("⊘ ns = diferencia no significativa.")
    return " ".join(parts)


def _load_sign_mismatch(audit_csv: Path) -> set[tuple[str, str, str, int]]:
    """Cells where the AGGREGATE framing disagrees in sign with the PAIRED one.

    The figure plots aggregate metrics (the only framing in which the formulaic
    and fitted baselines exist at all), but the paper's canonical claim is paired
    over identical samples. Where the two disagree the figure must say so rather
    than let the reader take the panel as the verdict.
    """
    if not audit_csv.exists():
        return set()
    audit = pl.read_csv(audit_csv).filter(pl.col("sign_mismatch"))
    return {
        (r["model"], r["metric"], r["corridor"], int(r["horizon"]))
        for r in audit.iter_rows(named=True)
    }


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
    sign_mismatch = _load_sign_mismatch(PAIRED_AUDIT_CSV)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for row, metric in enumerate(METRICS):
        for col, corridor in enumerate(CORRIDORS):
            ax = axes[row][col]
            table = degradation_table(
                df, metric=metric, direction="aggregate", corridor=corridor
            )
            horizons = _horizon_axis(table.columns)
            present = set(table["baseline"])
            marks: dict[str, list[tuple[int, float]]] = {}
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
                # Collect the DL points whose verdict is not "the deep model wins
                # significantly". Rings are drawn per model (three stacked rings
                # is itself information), but the text label is emitted ONCE per
                # (horizon, verdict) after the loop — at h=1 the three models
                # land on nearly the same y and three copies of the same label
                # overlap into noise.
                if name in DL_MODELS:
                    for x, y in zip(xs, ys):
                        verdict = significance.get((name, metric, corridor, x))
                        if verdict is None:
                            continue
                        significant, dl_better = verdict
                        if not dl_better:
                            kind = ("B1≠" if (name, metric, corridor, x)
                                    in sign_mismatch else "B1")
                        elif not significant:
                            kind = "ns"
                        else:
                            continue
                        marks.setdefault(kind, []).append((x, y))
            # Draw the collected verdict marks: a ring on every affected model
            # point, one text label per (horizon, verdict) placed above the
            # topmost ring of that group.
            for kind, pts in marks.items():
                style = MARK_STYLES[kind]
                ax.scatter(
                    [p[0] for p in pts], [p[1] for p in pts],
                    s=style["s"], facecolors="none",
                    edgecolors=style["color"], linewidths=style["lw"],
                    marker=style["marker"], zorder=5,
                )
                for x in sorted({p[0] for p in pts}):
                    top = max(p[1] for p in pts if p[0] == x)
                    ax.annotate(
                        kind, (x, top), textcoords="offset points",
                        xytext=(9, 8), fontsize=9, fontweight="bold",
                        color=style["color"],
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
        footnotes.append(_significance_footnote(SIG_CSV))
    # The framing caveat is not optional: without it a reader takes these
    # aggregate panels as the paper's verdict, which at the crossover they are not.
    footnotes.append(
        "Estos paneles son el MAE/RMSE AGREGADO sobre el test completo; la "
        "comparación canónica del trabajo es la PAREADA sobre muestras idénticas "
        "(Sección 3). El encuadre agregado favorece al DL en 0.28–0.53 min, así "
        f"que en {len(sign_mismatch)} celda(s) el signo que se ve acá se invierte "
        "en la base pareada (marcadas B1≠); no leer el cruce desde esta figura."
    )
    if multiseed_ci:
        footnotes.append(
            "Barras de error en LSTM = IC 95% sobre 5 seeds [42,123,456,789,999] "
            "(NB15); su ancho sub-marcador confirma estabilidad frente al seed."
        )
    if footnotes:
        wrapped = "\n".join(
            textwrap.fill(f, width=FOOTNOTE_WRAP) for f in footnotes if f
        )
        fig.text(
            0.5, 0.004, wrapped,
            ha="center", va="bottom", fontsize=9, style="italic",
        )
    # Bottom margin scales with the wrapped caption so it is never clipped.
    n_lines = wrapped.count("\n") + 1 if footnotes else 0
    fig.tight_layout(rect=(0, 0.018 * n_lines, 1, 0.90))

    fig_path = out_dir / "curva-degradacion.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    out = build()
    print(f"Figura escrita en {out}")
