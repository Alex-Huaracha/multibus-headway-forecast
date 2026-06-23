"""Build the EX-ANTE volatility figure — la versión operativa del crossover.

El crossover de volatilidad (``build_volatility_curve.py``) estratifica por el
cambio *realizado* del headway, conocido solo a posteriori — es descriptivo. Esta
figura repite el corte usando un estratificador **ex-ante**: la volatilidad de la
ventana de entrada (desvío estándar de los últimos 12 min observados), conocida
*antes* de predecir. Muestra que la ventaja del DL no solo sobrevive al corte
ex-ante sino que crece monótonamente con la volatilidad reciente.

La historia en una imagen:
  - La ventaja del DL (Δ MAE < 0) se acentúa de tercil calmo → medio → volátil.
  - En el tercil de alta volatilidad ex-ante el DL gana en los tres corredores y
    los tres horizontes; a h=3 la ventaja queda confinada al régimen volátil
    (la persistencia iguala en el tercil calmo).
  - Como el régimen se asigna con información disponible al predecir, la regla
    "usar DL cuando el servicio viene errático" es ejecutable en vivo.

Usage:
    uv run python -m src.build_exante_curve

Input (versioned):
    docs/resultados/csv-multihorizon/exante_volatility_multihorizon.csv

Output (written to docs/resultados/):
    volatilidad-exante.png   — 1x3 grid: cols E2/E59/E4, x=tercil ex-ante, y=Δ MAE
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import matplotlib

matplotlib.use("Agg")  # headless: no display in WSL/CI
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados"
EXANTE_CSV = RESULTS_DIR / "exante_volatility_multihorizon.csv"

CORRIDORS = ["E2", "E59", "E4"]

# Tercile display order + Spanish labels for the paper x-axis.
TERCILE_ORDER = ["low", "mid", "high"]
TERCILE_LABELS = {
    "low": "poca\n(tercil bajo)",
    "mid": "media\n(tercil medio)",
    "high": "mucha\n(tercil alto)",
}

# One line per horizon, colour-graded so the longer horizon reads as "deeper".
HORIZON_STYLE = {
    3: ("tab:blue", "o", "h = 3 min"),
    5: ("tab:orange", "s", "h = 5 min"),
    10: ("tab:red", "^", "h = 10 min"),
}


def build(exante_csv: Path = EXANTE_CSV, out_dir: Path = OUT_DIR) -> Path:
    """Render the ex-ante volatility figure from the versioned table.

    Returns the path to the written figure.

    Raises
    ------
    ValueError
        If the ex-ante CSV is missing (run build_exante_volatility first).
    """
    if not exante_csv.exists():
        raise ValueError(
            f"build_exante_curve: {exante_csv} not found — run "
            "`uv run python -m src.build_exante_volatility` first"
        )
    df = pl.read_csv(exante_csv)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    xs = list(range(len(TERCILE_ORDER)))
    for col, corridor in enumerate(CORRIDORS):
        ax = axes[col]
        sub = df.filter(pl.col("corridor") == corridor)
        for horizon, (color, marker, label) in HORIZON_STYLE.items():
            hrow = sub.filter(pl.col("horizon") == horizon)
            if hrow.height == 0:
                continue
            ys_by_tercile = dict(zip(hrow["tercile"], hrow["delta_mae"]))
            ys = [ys_by_tercile.get(t) for t in TERCILE_ORDER]
            ax.plot(
                xs, ys, label=label, color=color, marker=marker,
                linestyle="-", linewidth=1.9, markersize=7,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(f"{corridor}")
        ax.set_xticks(xs)
        ax.set_xticklabels([TERCILE_LABELS[t] for t in TERCILE_ORDER])
        ax.set_xlabel("Volatilidad reciente (ex-ante, ventana de entrada)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Δ MAE = MAE(DL) − MAE(persistencia)  [min]")
            ax.text(0.02, 0.96, "persistencia mejor ▲", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, color="dimgray")
            ax.text(0.02, 0.04, "DL mejor ▼", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=9, color="dimgray")

    fig.suptitle(
        "Estratificación ex-ante — la ventaja del DL crece con la volatilidad "
        "reciente (conocida al predecir) (LSTM)",
        y=0.99, fontsize=12.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=len(labels), frameon=False,
    )
    fig.text(
        0.5, 0.005,
        "Δ MAE por tercil de volatilidad de la ventana de entrada (desvío de los "
        "12 min previos, conocido ANTES de predecir). Negativo: el DL gana. La "
        "pendiente descendente = la ventaja del DL se acentúa cuando el servicio venía errático.",
        ha="center", fontsize=8.5, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))

    fig_path = out_dir / "volatilidad-exante.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    out = build()
    print(f"Figura escrita en {out}")
