"""Build the volatility-regime crossover figure — Fase 7.

The degradation curve shows the DL-vs-persistence gap widening with the horizon;
the volatility table (``build_volatility_table.py``) explains WHY. This figure
renders that explanation: the per-sample MAE gap (DL − persistence) versus the
realized headway-change regime, per corridor, one line per horizon.

The story it tells in one image:
  - There is a CROSSOVER. In stable windows (small change) the gap is positive
    (persistence wins — DL only adds noise when the headway does not move); in
    high-change windows it goes sharply negative (DL wins).
  - Longer horizons push the crossover deeper and shift mass toward the
    high-change regime, which is why the aggregate curve diverges.

Usage:
    uv run python -m src.build_volatility_curve

Input (versioned — rebuilds without the raw per-sample residuals):
    docs/resultados/csv-multihorizon/volatility_multihorizon.csv

Output (written to docs/resultados/):
    volatilidad-crossover.png   — 1x2 grid: cols E2/E59, x=regime, y=Δ MAE

The figure uses the LSTM (the DL baseline) as the representative deep model; the
two spatial models give a near-identical crossover (within noise), so plotting
all three would only clutter the panels. Change ``MODEL`` to switch.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl
import matplotlib

matplotlib.use("Agg")  # headless: no display in WSL/CI
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados"
VOL_CSV = RESULTS_DIR / "volatility_multihorizon.csv"

MODEL = "LSTM"  # representative deep model; spatial models match within noise
METRIC = "MAE"  # headline effect size (delta_mae)
CORRIDORS = ["E2", "E59", "E4"]

# Regime display order + Spanish labels for the paper x-axis.
REGIME_ORDER = ["low", "moderate", "high"]
REGIME_LABELS = {"low": "estable\n(<1 min)", "moderate": "moderado\n(1–3 min)", "high": "alto\n(>3 min)"}

# One line per horizon, colour-graded so the longer horizon reads as "deeper".
HORIZON_STYLE = {
    3: ("tab:blue", "o", "h = 3 min"),
    5: ("tab:orange", "s", "h = 5 min"),
    10: ("tab:red", "^", "h = 10 min"),
}


def build(vol_csv: Path = VOL_CSV, out_dir: Path = OUT_DIR) -> Path:
    """Render the volatility crossover figure from the versioned table.

    Returns the path to the written figure.

    Raises
    ------
    ValueError
        If the volatility CSV is missing (run build_volatility_table first).
    """
    if not vol_csv.exists():
        raise ValueError(
            f"build_volatility_curve: {vol_csv} not found — run "
            "`uv run python -m src.build_volatility_table` first"
        )
    df = pl.read_csv(vol_csv).filter(
        (pl.col("model") == MODEL) & (pl.col("metric") == METRIC)
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    xs = list(range(len(REGIME_ORDER)))
    for col, corridor in enumerate(CORRIDORS):
        ax = axes[col]
        sub = df.filter(pl.col("corridor") == corridor)
        for horizon, (color, marker, label) in HORIZON_STYLE.items():
            hrow = sub.filter(pl.col("horizon") == horizon).sort("regime_order")
            if hrow.height == 0:
                continue
            # Align y to REGIME_ORDER so missing regimes leave a gap, not a shift.
            ys_by_regime = dict(zip(hrow["regime"], hrow["delta_mae"]))
            ys = [ys_by_regime.get(r) for r in REGIME_ORDER]
            ax.plot(
                xs, ys, label=label, color=color, marker=marker,
                linestyle="-", linewidth=1.9, markersize=7,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(f"{corridor}")
        ax.set_xticks(xs)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
        ax.set_xlabel("Régimen de cambio del headway")
        ax.grid(True, alpha=0.3)
        # Annotate the two half-planes once, on the left panel.
        if col == 0:
            ax.set_ylabel("Δ MAE = MAE(DL) − MAE(persistencia)  [min]")
            ymax = ax.get_ylim()[1]
            ymin = ax.get_ylim()[0]
            ax.text(0.02, 0.96, "persistencia mejor ▲", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, color="dimgray")
            ax.text(0.02, 0.04, "DL mejor ▼", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=9, color="dimgray")

    fig.suptitle(
        f"Crossover de volatilidad — la ventaja del DL se concentra en el alto cambio ({MODEL})",
        y=0.99, fontsize=12.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=len(labels), frameon=False,
    )
    fig.text(
        0.5, 0.005,
        "Δ MAE por régimen de cambio realizado del headway (|y − persistencia|). "
        "Positivo: la persistencia gana; negativo: el DL gana. El cruce se "
        "profundiza con el horizonte.",
        ha="center", fontsize=8.5, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))

    fig_path = out_dir / "volatilidad-crossover.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    out = build()
    print(f"Figura escrita en {out}")
