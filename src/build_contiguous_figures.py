"""Figures for the contiguous pipeline — the three the rewritten document needs.

The existing figures (``curva-degradacion.png``, ``volatilidad-crossover.png``)
render the frozen 11/12/13 families, which carry the framing bias the retraining
removed. They stay as the record of that comparison; these are their successors,
and they are built from the **committed CSVs** rather than from the raw residuals
so a figure can never disagree with the table it illustrates.

    contiguo-degradacion.png   MAE vs horizonte, por corredor. El resultado escalar.
    contiguo-volatilidad.png   Δ MAE por tercil ex-ante. Que la frontera es la
                               volatilidad, no el horizonte.
    contiguo-disociacion.png   Ventaja escalar contra fidelidad vectorial, ambas
                               contra el horizonte. El aporte.

The third is the paper's headline: two quantities moving in opposite directions
as the horizon grows, on one pair of axes, per corridor. Everything else in the
document supports it.

Usage
-----
    uv run python -m src.build_contiguous_figures
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: no display in WSL/CI

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados"

CORRIDORS = ("E2", "E59", "E4")
HORIZONS = (1, 3, 5, 10)
TERCILES = ("low", "mid", "high")
TERCILE_LABELS = {
    "low": "ventana\ncalma",
    "mid": "ventana\nmedia",
    "high": "ventana\nvolátil",
}

MODEL_STYLE = {
    "mae_persist_paired": ("Persistencia", "tab:gray", "o", "--"),
    "mae_xgb_paired": ("XGBoost", "tab:green", "s", "-"),
    "mae_lstm_paired": ("LSTM", "tab:blue", "^", "-"),
}
HORIZON_STYLE = {
    1: ("tab:purple", "o", "h = 1 min"),
    3: ("tab:blue", "s", "h = 3 min"),
    5: ("tab:orange", "^", "h = 5 min"),
    10: ("tab:red", "D", "h = 10 min"),
}

# Figures are deliberately colour-blind safe in ORDER as well as hue: every
# series also differs by marker and linestyle, so a greyscale print stays legible.
CAPTION_SIZE = 8.5
DPI = 150

# Captions are wrapped by hand rather than left to the renderer: matplotlib does
# not wrap `fig.text`, so a long single line is silently clipped at both figure
# edges — which is how the first render lost the start and the end of its own
# caption. Each entry below is one rendered line.
CAPTION_LINE_HEIGHT = 0.026


def _caption(fig, lines: list[str], *, bottom: float = 0.012) -> None:
    """Render a multi-line italic caption anchored at the figure bottom."""
    for index, line in enumerate(reversed(lines)):
        fig.text(
            0.5, bottom + index * CAPTION_LINE_HEIGHT, line,
            ha="center", fontsize=CAPTION_SIZE, style="italic",
        )


def _load(name: str) -> pl.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name} — run its builder before rendering figures"
        )
    return pl.read_csv(path)


def degradation() -> Path:
    """MAE vs horizon, one panel per corridor. The scalar result."""
    audit = _load("contiguous_paired_audit.csv").filter(
        pl.col("direction") == "aggregate"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, corridor in zip(axes, CORRIDORS):
        sub = audit.filter(pl.col("corridor") == corridor).sort("horizon")
        for column, (label, color, marker, style) in MODEL_STYLE.items():
            ax.plot(
                sub.get_column("horizon"), sub.get_column(column),
                label=label, color=color, marker=marker, linestyle=style,
                linewidth=1.9, markersize=7,
            )
        ax.set_title(corridor)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel("Horizonte de predicción [min]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("MAE [min]")

    fig.suptitle(
        "Curva de degradación — pipeline contiguo, muestras idénticas",
        y=0.99, fontsize=12.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=3, frameon=False,
    )
    _caption(fig, [
        "Los tres modelos puntúan las mismas celdas (contrato C1; sesgo de encuadre medido: 0.001 min).",
        "La persistencia gana a h = 1 en los tres corredores; los dos aprendices la superan desde h = 5 con holgura creciente.",
        "El XGBoost reproduce el cruce: no es una propiedad del Deep Learning.",
    ])
    fig.tight_layout(rect=(0, 0.10, 1, 0.88))

    path = OUT_DIR / "contiguo-degradacion.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def volatility() -> Path:
    """Δ MAE by ex-ante tercile. That the frontier is volatility, not horizon."""
    table = _load("contiguous_exante_volatility.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    xs = list(range(len(TERCILES)))
    for ax, corridor in zip(axes, CORRIDORS):
        sub = table.filter(pl.col("corridor") == corridor)
        for horizon, (color, marker, label) in HORIZON_STYLE.items():
            row = sub.filter(pl.col("horizon") == horizon).sort("tercile_order")
            if row.height == 0:
                continue
            by_tercile = dict(
                zip(row.get_column("tercile"), row.get_column("delta_lstm_persist"))
            )
            ax.plot(
                xs, [by_tercile.get(name) for name in TERCILES],
                label=label, color=color, marker=marker,
                linestyle="-", linewidth=1.9, markersize=7,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(corridor)
        ax.set_xticks(xs)
        ax.set_xticklabels([TERCILE_LABELS[name] for name in TERCILES])
        ax.set_xlabel("Volatilidad de la ventana de entrada (ex-ante)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Δ MAE = LSTM − persistencia  [min]")
    axes[0].text(
        0.02, 0.96, "persistencia mejor ▲", transform=axes[0].transAxes,
        va="top", ha="left", fontsize=9, color="dimgray",
    )
    axes[0].text(
        0.02, 0.04, "LSTM mejor ▼", transform=axes[0].transAxes,
        va="bottom", ha="left", fontsize=9, color="dimgray",
    )

    fig.suptitle(
        "La frontera no es el horizonte: es la volatilidad que el horizonte cruza",
        y=0.99, fontsize=12.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=4, frameon=False,
    )
    _caption(fig, [
        "Terciles de dispersión de la ventana de entrada, con umbrales congelados en train+val: información disponible",
        "al predecir, así que el contraste no es circular. Dentro de cada horizonte la ventaja crece del tercil calmo al",
        "volátil en las 12 celdas; alargar el horizonte la empuja hacia los terciles más calmos.",
    ])
    fig.tight_layout(rect=(0, 0.11, 1, 0.88))

    path = OUT_DIR / "contiguo-volatilidad.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def dissociation() -> Path:
    """The headline: scalar advantage and vector fidelity, opposite directions."""
    audit = _load("contiguous_paired_audit.csv").filter(
        pl.col("direction") == "aggregate"
    )
    vector = _load("contiguous_vector_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    # Legend handles are collected while plotting the FIRST panel. Recovering
    # them afterwards would mean indexing into `fig.axes`, whose order mixes the
    # primary and twin axes and silently changes if a panel is added.
    legend_handles: list = []
    twin_axes: list = []

    for ax, corridor in zip(axes, CORRIDORS):
        scalar = audit.filter(pl.col("corridor") == corridor).sort("horizon")
        # Advantage plotted as a POSITIVE gain so "up is better" holds on both
        # axes; the underlying column is negative when the learner wins.
        gain = [-value for value in scalar.get_column("delta_lstm_persist")]
        (scalar_line,) = ax.plot(
            scalar.get_column("horizon"), gain,
            color="tab:blue", marker="^", linewidth=2.2, markersize=8,
            label="Ventaja escalar del LSTM (MAE)",
        )
        if ax is axes[0]:
            legend_handles.append(scalar_line)
        ax.axhline(0.0, color="tab:blue", linewidth=1.0, linestyle=":", alpha=0.6)
        ax.set_title(corridor)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel("Horizonte de predicción [min]")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelcolor="tab:blue")

        twin = ax.twinx()
        twin_axes.append(twin)
        for model, label, color, marker, style in (
            ("Persistence", "F1 de bunching — persistencia", "tab:gray", "o", "--"),
            ("LSTM", "F1 de bunching — LSTM", "tab:red", "s", "-"),
        ):
            row = vector.filter(
                (pl.col("corridor") == corridor) & (pl.col("model") == model)
            ).sort("horizon")
            (line,) = twin.plot(
                row.get_column("horizon"), row.get_column("bunching_f1"),
                color=color, marker=marker, linestyle=style,
                linewidth=2.0, markersize=7, label=label,
            )
            if ax is axes[0]:
                legend_handles.append(line)
        # Shared scale across panels so the collapse is comparable between
        # corridors rather than rescaled away in each one.
        twin.set_ylim(0, 0.75)
        twin.tick_params(axis="y", labelcolor="tab:red")
        # Only the rightmost twin keeps tick labels; the inner ones would sit on
        # top of the next panel's own axis.
        if ax is not axes[-1]:
            twin.set_yticklabels([])

    axes[0].set_ylabel("Ventaja en MAE sobre persistencia [min]", color="tab:blue")
    twin_axes[-1].set_ylabel("F1 de detección de bunching", color="tab:red")

    fig.suptitle(
        "La disociación: el aprendiz gana el error medio y pierde el vector",
        y=0.99, fontsize=12.5,
    )
    fig.legend(
        legend_handles, [line.get_label() for line in legend_handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False,
    )
    _caption(fig, [
        "Eje izquierdo (azul, ↑ mejor): cuánto MAE le gana el LSTM a la persistencia. Eje derecho (rojo y gris, ↑ mejor):",
        "F1 de detección conjunta de bunching, en escala común a los tres paneles. Alargar el horizonte mejora lo primero",
        "y destruye lo segundo, en los tres corredores. Una evaluación escalar solo ve la curva azul.",
    ])
    fig.tight_layout(rect=(0, 0.11, 1, 0.88))

    path = OUT_DIR / "contiguo-disociacion.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> None:
    for renderer in (degradation, volatility, dissociation):
        path = renderer()
        print(f"Figura escrita en {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
