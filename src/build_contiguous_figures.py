"""Figures for the contiguous pipeline — the three the rewritten document needs.

The existing figures (``curva-degradacion.png``, ``volatilidad-crossover.png``)
render the frozen 11/12/13 families, which carry the framing bias the retraining
removed. They stay as the record of that comparison; these are their successors,
and they are built from the **committed CSVs** rather than from the raw residuals
so a figure can never disagree with the table it illustrates.

    contiguo-degradacion.png            MAE vs horizonte, por corredor. El resultado
                                        escalar.
    contiguo-volatilidad.png            Δ MAE por tercil ex-ante. Que la frontera es
                                        la volatilidad, no el horizonte.
    contiguo-artefacto-umbral.png       Tasa de disparo contra tasa real del evento.
                                        El artefacto que se está explicando.
    contiguo-deteccion-sin-umbral.png   Ventaja escalar y AUC de detección, juntas.
                                        El veredicto corregido.
    contiguo-compresion-dispersion.png  CV observado contra CV predicho, a h = 10.
                                        La causa del artefacto.
    contiguo-compresion-vs-horizonte.png  El mismo sesgo contra el horizonte. La
                                        dosis-respuesta.

``contiguo-artefacto-umbral`` and ``contiguo-deteccion-sin-umbral`` are the
paper's headline and they only work as a pair: the first shows a comparison
decided by its own operating point, the second shows what the same data says
once the operating point is removed. Publishing either alone would misrepresent
the result — the earlier ``contiguo-disociacion.png`` did exactly that by
plotting fixed-cut F1 as if it measured the models, and it is gone. The two
compression figures pair the same way: one measures the flattening at a single
horizon, the other shows it deepening with every minute added.

Two variants of each figure
---------------------------
The **chrome** variant is what ``documento-resultados.md`` embeds: a Spanish
suptitle plus the hand-wrapped italic footnote, written into
``docs/resultados/``. The document has no caption machinery of its own, so the
figure has to carry its own argument.

The **clean** variant drops both and lands in ``docs/paper/figuras/`` as
``<stem>.<lang>.png``, because the manuscript numbers and captions its own
figures — a suptitle there would be a second, competing title. Only the four
figures the paper reproduces have a clean variant, and each is emitted in both
languages. The plotting code is written once and parameterised; the two
variants differ only in the chrome, the layout band it needs, and the language
of the labels.

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
PAPER_DIR = REPO_ROOT / "docs" / "paper" / "figuras"

# Panel order matches the tables in the document (E2, E4, E59). It used to be
# E2/E59/E4, which forced the reader to re-map on every table-figure crossing.
CORRIDORS = ("E2", "E4", "E59")
HORIZONS = (1, 3, 5, 10)
TERCILES = ("low", "mid", "high")

# Every user-facing string lives here, keyed by language: the paper needs the
# same axes in Spanish and in English, and duplicating the plotting code to get
# them would let the two drift. Panel titles are corridor codes (E2/E4/E59) and
# stay out of this table — they are language-neutral, and so are the horizon
# legend labels (``h = 1 min``), which are built from a format string.
LANG = {
    "es": {
        "horizon_axis": "Horizonte de predicción [min]",
        "mae_axis": "MAE [min]",
        "model_persistence": "Persistencia",
        "model_xgboost": "XGBoost",
        "model_lstm": "LSTM",
        "tercile_low": "ventana\ncalma",
        "tercile_mid": "ventana\nmedia",
        "tercile_high": "ventana\nvolátil",
        "volatility_axis": "Volatilidad de la ventana de entrada (ex-ante)",
        "delta_axis": "Δ MAE = LSTM − persistencia  [min]",
        "persistence_better": "persistencia mejor ▲",
        "lstm_better": "LSTM mejor ▼",
        "base_rate": "Tasa real de bunching (lo que habría que detectar)",
        "persistence_fires": "Persistencia — dispara a la tasa base",
        "lstm_silenced": "LSTM — el corte fijo lo silencia",
        "fire_rate_axis": "Fracción de posiciones marcadas como bunching",
        "scalar_advantage": "Ventaja escalar del LSTM (MAE)",
        "auc_persistence": "AUC de bunching — persistencia",
        "auc_lstm": "AUC de bunching — LSTM",
        "advantage_axis": "Ventaja en MAE sobre persistencia [min]",
        "auc_axis": "AUC de detección de bunching",
        "observed_bars": "Realidad observada",
        "predicted_bars": "Lo que el modelo predice",
        "cv_axis": "Coeficiente de variación del vector",
        "cv_bias_axis": "Sesgo de dispersión (CV predicho − CV observado)",
        "compression_hint": "predicción más plana ▼",
    },
    "en": {
        "horizon_axis": "Prediction horizon [min]",
        "mae_axis": "MAE [min]",
        "model_persistence": "Persistence",
        "model_xgboost": "XGBoost",
        "model_lstm": "LSTM",
        "tercile_low": "calm\nwindow",
        "tercile_mid": "mid\nwindow",
        "tercile_high": "volatile\nwindow",
        "volatility_axis": "Input-window volatility (ex-ante)",
        "delta_axis": "Δ MAE = LSTM − persistence  [min]",
        "persistence_better": "persistence better ▲",
        "lstm_better": "LSTM better ▼",
        "base_rate": "Observed bunching rate (what should be detected)",
        "persistence_fires": "Persistence — fires at the base rate",
        "lstm_silenced": "LSTM — the fixed cut silences it",
        "fire_rate_axis": "Fraction of positions flagged as bunching",
        "scalar_advantage": "LSTM scalar advantage (MAE)",
        "auc_persistence": "Bunching AUC — persistence",
        "auc_lstm": "Bunching AUC — LSTM",
        "advantage_axis": "MAE advantage over persistence [min]",
        "auc_axis": "Bunching detection AUC",
        "observed_bars": "Observed",
        "predicted_bars": "Predicted",
        "cv_axis": "Vector coefficient of variation",
        "cv_bias_axis": "Dispersion bias (predicted − observed CV)",
        "compression_hint": "flatter forecast ▼",
    },
}

TERCILE_LABEL_KEYS = {"low": "tercile_low", "mid": "tercile_mid", "high": "tercile_high"}

# Styles carry a LANG key rather than a literal label, so a series keeps the same
# colour/marker/linestyle in both languages and cannot be relabelled in one only.
MODEL_STYLE = {
    "mae_persist_paired": ("model_persistence", "tab:gray", "o", "--"),
    "mae_xgb_paired": ("model_xgboost", "tab:green", "s", "-"),
    "mae_lstm_paired": ("model_lstm", "tab:blue", "^", "-"),
}
# The same three series, keyed by the ``model`` column of the vector-metrics
# table instead of by a paired-audit column name.
VECTOR_MODEL_STYLE = {
    "Persistence": ("model_persistence", "tab:gray", "o", "--"),
    "XGBoost": ("model_xgboost", "tab:green", "s", "-"),
    "LSTM": ("model_lstm", "tab:blue", "^", "-"),
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

# Layout for the clean variant. With no suptitle and no footnote the axes take
# the whole canvas except one strip for the legend, which rides at the very top
# edge; reusing the chrome ``rect`` here would leave a wide empty band where the
# title and the caption used to be.
CLEAN_LEGEND_Y = 0.99
CLEAN_RECT = (0.0, 0.0, 1.0, 0.92)

# Results-document filename, and the paper stem when the manuscript reproduces
# the figure. ``None`` means there is no clean variant: the paper does not carry
# that figure, so emitting a bilingual pair for it would be dead output.
#
# Paper stems carry NO figure number. The order in which the paper presents its
# figures is still moving while the remaining sections are written, and a stem
# like ``fig1a-`` that ends up printed as "Fig. 3" is a trap: it silently invites
# a wrong cross-reference every time the order shifts. The number belongs in
# paper.md, which is the only place that knows the order of first appearance.
FIGURE_NAMES = {
    "degradation": ("contiguo-degradacion.png", None),
    "volatility": ("contiguo-volatilidad.png", None),
    "threshold_artifact": (
        "contiguo-artefacto-umbral.png", "artefacto-umbral",
    ),
    "detection_without_threshold": (
        "contiguo-deteccion-sin-umbral.png", "deteccion-sin-umbral",
    ),
    "dispersion_compression": (
        "contiguo-compresion-dispersion.png", "compresion-dispersion",
    ),
    "dispersion_vs_horizon": (
        "contiguo-compresion-vs-horizonte.png", "compresion-vs-horizonte",
    ),
}


def _caption(fig, lines: list[str], *, bottom: float = 0.012) -> None:
    """Render a multi-line italic caption anchored at the figure bottom."""
    for index, line in enumerate(reversed(lines)):
        fig.text(
            0.5, bottom + index * CAPTION_LINE_HEIGHT, line,
            ha="center", fontsize=CAPTION_SIZE, style="italic",
        )


def _resolve(figure: str, lang: str, chrome: bool) -> Path:
    """Where a variant is written, and the invariants that pair the two.

    Checked before any plotting so an unbuildable request fails on the first
    line rather than after a render. The chrome variant is Spanish-only: its
    suptitle and footnote are prose written for ``documento-resultados.md``, so
    asking for it in English would emit a half-translated figure over the
    published filename.
    """
    if lang not in LANG:
        raise ValueError(f"unknown language {lang!r}; expected one of {sorted(LANG)}")
    doc_name, paper_stem = FIGURE_NAMES[figure]
    if chrome:
        if lang != "es":
            raise ValueError(
                f"{figure}: the chrome variant exists only in Spanish — its "
                "suptitle and footnote belong to documento-resultados.md"
            )
        return OUT_DIR / doc_name
    if paper_stem is None:
        raise ValueError(f"{figure}: the paper does not reproduce this figure")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    return PAPER_DIR / f"{paper_stem}.{lang}.png"


def _load(name: str) -> pl.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name} — run its builder before rendering figures"
        )
    return pl.read_csv(path)


def degradation(*, lang: str = "es", chrome: bool = True) -> Path:
    """MAE vs horizon, one panel per corridor. The scalar result."""
    path = _resolve("degradation", lang, chrome)
    words = LANG[lang]
    audit = _load("contiguous_paired_audit.csv").filter(
        pl.col("direction") == "aggregate"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, corridor in zip(axes, CORRIDORS):
        sub = audit.filter(pl.col("corridor") == corridor).sort("horizon")
        for column, (label_key, color, marker, style) in MODEL_STYLE.items():
            ax.plot(
                sub.get_column("horizon"), sub.get_column(column),
                label=words[label_key], color=color, marker=marker, linestyle=style,
                linewidth=1.9, markersize=7,
            )
        ax.set_title(corridor)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel(words["horizon_axis"])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(words["mae_axis"])

    if chrome:
        fig.suptitle(
            "Curva de degradación — pipeline contiguo, muestras idénticas",
            y=0.99, fontsize=12.5,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.935 if chrome else CLEAN_LEGEND_Y),
        ncol=3, frameon=False,
    )
    if chrome:
        _caption(fig, [
            "Los tres modelos puntúan las mismas celdas (contrato C1; sesgo de encuadre medido: 0.001 min).",
            "La persistencia gana a h = 1 en los tres corredores; los dos aprendices la superan desde h = 5 con holgura creciente.",
            "El XGBoost reproduce el cruce: no es una propiedad del Deep Learning.",
        ])
    fig.tight_layout(rect=(0, 0.10, 1, 0.88) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def volatility(*, lang: str = "es", chrome: bool = True) -> Path:
    """Δ MAE by ex-ante tercile. That the frontier is volatility, not horizon."""
    path = _resolve("volatility", lang, chrome)
    words = LANG[lang]
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
        ax.set_xticklabels([words[TERCILE_LABEL_KEYS[name]] for name in TERCILES])
        ax.set_xlabel(words["volatility_axis"])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(words["delta_axis"])
    axes[0].text(
        0.02, 0.96, words["persistence_better"], transform=axes[0].transAxes,
        va="top", ha="left", fontsize=9, color="dimgray",
    )
    axes[0].text(
        0.02, 0.04, words["lstm_better"], transform=axes[0].transAxes,
        va="bottom", ha="left", fontsize=9, color="dimgray",
    )

    if chrome:
        fig.suptitle(
            "La frontera no es el horizonte: es la volatilidad que el horizonte cruza",
            y=0.99, fontsize=12.5,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.935 if chrome else CLEAN_LEGEND_Y),
        ncol=4, frameon=False,
    )
    if chrome:
        _caption(fig, [
            "Terciles de dispersión de la ventana de entrada, con umbrales congelados en train+val: información disponible",
            "al predecir, así que el contraste no es circular. Dentro de cada horizonte la ventaja crece del tercil calmo al",
            "volátil en las 12 celdas; alargar el horizonte la empuja hacia los terciles más calmos.",
        ])
    fig.tight_layout(rect=(0, 0.11, 1, 0.88) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def threshold_artifact(*, lang: str = "es", chrome: bool = True) -> Path:
    """The artifact, in one glance: who fires, and how often the event happens.

    The previously published verdict rested on bunching F1 at a fixed relative
    cut. This figure shows why that comparison was decided by the cut and not by
    the models: persistence fires almost exactly at the base rate — because the
    rule was calibrated in the units it lives in — while the learner's compressed
    vector puts the same relative cut deep in its tail, so it falls silent.
    """
    path = _resolve("threshold_artifact", lang, chrome)
    words = LANG[lang]
    table = _load("contiguous_detection_calibrated.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
    legend_handles: list = []

    for ax, corridor in zip(axes, CORRIDORS):
        cell = table.filter(pl.col("corridor") == corridor)

        base = cell.filter(pl.col("model") == "LSTM").sort("horizon")
        (base_line,) = ax.plot(
            base.get_column("horizon"), base.get_column("base_rate"),
            color="black", marker="", linestyle=":", linewidth=1.8,
            label=words["base_rate"],
        )
        for model, label_key, color, marker, style in (
            ("Persistence", "persistence_fires", "tab:gray", "o", "--"),
            ("LSTM", "lstm_silenced", "tab:red", "s", "-"),
        ):
            row = cell.filter(pl.col("model") == model).sort("horizon")
            (line,) = ax.plot(
                row.get_column("horizon"), row.get_column("fire_rate_fixed"),
                color=color, marker=marker, linestyle=style,
                linewidth=2.0, markersize=7, label=words[label_key],
            )
            if ax is axes[0]:
                legend_handles.append(line)
        if ax is axes[0]:
            legend_handles.append(base_line)

        ax.set_title(corridor)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel(words["horizon_axis"])
        ax.set_ylim(0, 0.35)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(words["fire_rate_axis"])
    if chrome:
        fig.suptitle(
            "El artefacto: con el corte fijo, la persistencia dispara a la tasa base "
            "y el aprendiz enmudece",
            y=0.99, fontsize=12.5,
        )
    fig.legend(
        legend_handles, [line.get_label() for line in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925 if chrome else CLEAN_LEGEND_Y),
        ncol=3, frameon=False,
    )
    if chrome:
        _caption(fig, [
            "La regla marca toda posición por debajo de 0.5x la media de su vector. La persistencia propaga el vector observado, así que",
            "hereda su dispersión y el corte cae donde fue diseñado: dispara casi exactamente tan seguido como ocurre el evento. El",
            "pronóstico puntual emite un vector comprimido (CV 0.16 contra 0.79), así que el mismo corte relativo le queda en la cola.",
        ])
    fig.tight_layout(rect=(0, 0.11, 1, 0.88) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def detection_without_threshold(*, lang: str = "es", chrome: bool = True) -> Path:
    """The corrected verdict: remove the cut and both metrics cross over together.

    Left axis is the scalar advantage, right axis is threshold-free
    discrimination (AUC) for both models. The point is that they AGREE:
    persistence leads at h=1 on both, the learner leads at h=10 on both. The
    "dissociation" reported earlier was the fixed cut, not the models.
    """
    path = _resolve("detection_without_threshold", lang, chrome)
    words = LANG[lang]
    audit = _load("contiguous_paired_audit.csv").filter(
        pl.col("direction") == "aggregate"
    )
    detection = _load("contiguous_detection_calibrated.csv")

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
            label=words["scalar_advantage"],
        )
        if ax is axes[0]:
            legend_handles.append(scalar_line)
        ax.axhline(0.0, color="tab:blue", linewidth=1.0, linestyle=":", alpha=0.6)
        ax.set_title(corridor)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel(words["horizon_axis"])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelcolor="tab:blue")

        twin = ax.twinx()
        twin_axes.append(twin)
        for model, label_key, color, marker, style in (
            ("Persistence", "auc_persistence", "tab:gray", "o", "--"),
            ("LSTM", "auc_lstm", "tab:red", "s", "-"),
        ):
            row = detection.filter(
                (pl.col("corridor") == corridor) & (pl.col("model") == model)
            ).sort("horizon")
            (line,) = twin.plot(
                row.get_column("horizon"), row.get_column("auc"),
                color=color, marker=marker, linestyle=style,
                linewidth=2.0, markersize=7, label=words[label_key],
            )
            if ax is axes[0]:
                legend_handles.append(line)
        # Shared scale across panels so the crossover is comparable between
        # corridors rather than rescaled away in each one. 0.5 is chance, and it
        # is inside the range on purpose: the reader must see how far from blind
        # these curves are.
        twin.set_ylim(0.50, 0.85)
        twin.axhline(0.5, color="tab:red", linewidth=1.0, linestyle=":", alpha=0.6)
        twin.tick_params(axis="y", labelcolor="tab:red")
        # Only the rightmost twin keeps tick labels; the inner ones would sit on
        # top of the next panel's own axis.
        if ax is not axes[-1]:
            twin.set_yticklabels([])

    axes[0].set_ylabel(words["advantage_axis"], color="tab:blue")
    twin_axes[-1].set_ylabel(words["auc_axis"], color="tab:red")

    if chrome:
        fig.suptitle(
            "Sin umbral las dos métricas coinciden: la persistencia manda corto, "
            "el aprendiz manda largo",
            y=0.99, fontsize=12.5,
        )
    fig.legend(
        legend_handles, [line.get_label() for line in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935 if chrome else CLEAN_LEGEND_Y),
        ncol=3, frameon=False,
    )
    if chrome:
        _caption(fig, [
            "Eje izquierdo (azul, ↑ mejor): cuánto MAE le gana el LSTM a la persistencia. Eje derecho (rojo y gris, ↑ mejor): AUC de",
            "detección de bunching, invariante a cualquier reescalado monótono del pronóstico y por lo tanto inmune al artefacto de",
            "umbral. Los dos cruces van en el mismo sentido y en la misma zona; el de detección ocurre igual o algo más tarde que el",
            "escalar. Ninguna serie está cerca del azar (0.5, punteado): el aprendiz no es ciego al evento en ninguna celda.",
        ])
    fig.tight_layout(rect=(0, 0.11, 1, 0.88) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def dispersion_compression(*, lang: str = "es", chrome: bool = True) -> Path:
    """Observed vs predicted coefficient of variation, by model and corridor.

    This is the cause of the artefact plotted by ``threshold_artifact``: a point
    forecast emits a vector flatter than the one it describes, so a cut calibrated
    on observed dispersion lands in its left tail. Persistence is the control —
    it propagates the observed vector, so it inherits the real dispersion and its
    bias is ~0. That both learners compress, and by comparable amounts, is what
    makes this a property of point forecasting rather than of one architecture.
    """
    path = _resolve("dispersion_compression", lang, chrome)
    words = LANG[lang]
    metrics = _load("contiguous_vector_metrics.csv").filter(pl.col("horizon") == 10)

    models = ("Persistence", "LSTM", "XGBoost")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    x = range(len(models))
    bar_w = 0.36

    for ax, corridor in zip(axes, CORRIDORS):
        sub = metrics.filter(pl.col("corridor") == corridor)
        true_cv, pred_cv = [], []
        for model in models:
            row = sub.filter(pl.col("model") == model)
            true_cv.append(float(row.get_column("mean_cv_true")[0]))
            pred_cv.append(float(row.get_column("mean_cv_pred")[0]))

        ax.bar([i - bar_w / 2 for i in x], true_cv, bar_w,
               color="tab:gray", label=words["observed_bars"])
        ax.bar([i + bar_w / 2 for i in x], pred_cv, bar_w,
               color="tab:red", label=words["predicted_bars"])

        for i, (t, pr) in enumerate(zip(true_cv, pred_cv)):
            ax.text(i - bar_w / 2, t + 0.018, f"{t:.2f}", ha="center", fontsize=8.5)
            ax.text(i + bar_w / 2, pr + 0.018, f"{pr:.2f}", ha="center", fontsize=8.5)

        ax.set_title(corridor)
        ax.set_xticks(list(x))
        ax.set_xticklabels([words[VECTOR_MODEL_STYLE[m][0]] for m in models])
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(words["cv_axis"])
    if chrome:
        fig.suptitle(
            "La compresión de dispersión: el pronóstico describe un corredor más parejo del que hay",
            y=0.99, fontsize=12.5,
        )
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center",
               bbox_to_anchor=(0.5, 0.935 if chrome else CLEAN_LEGEND_Y),
               ncol=2, frameon=False)
    if chrome:
        _caption(fig, [
            "Horizonte de 10 minutos. La persistencia propaga el vector observado, así que hereda su dispersión y su barra roja iguala a la gris: es el control.",
            "Los dos aprendices la aplanan, y por márgenes comparables — el efecto es del pronóstico puntual, no de una arquitectura.",
            "Sobre un vector aplanado, un corte calibrado en la dispersión real cae en la cola izquierda y no se dispara nunca.",
        ])
    fig.tight_layout(rect=(0, 0.13, 1, 0.88) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def dispersion_vs_horizon(*, lang: str = "es", chrome: bool = True) -> Path:
    """The same compression, plotted against the horizon: the dose-response.

    ``dispersion_compression`` measures the flattening at one horizon, which
    leaves it open to being read as a fixed quirk of the loss. This one shows it
    is a dose: persistence stays pinned to zero at every horizon — it propagates
    the observed vector, so it inherits the observed dispersion and acts as the
    control — while both learners dive monotonically further negative as the
    horizon grows. The bias tracks how much uncertainty the point forecast has
    to average over, which is what makes it a property of point forecasting.
    """
    path = _resolve("dispersion_vs_horizon", lang, chrome)
    words = LANG[lang]
    metrics = _load("contiguous_vector_metrics.csv")

    # Shared y so the three corridors are read against one scale: E2 compresses
    # roughly twice as hard as E4, and a per-panel scale would hide that.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, corridor in zip(axes, CORRIDORS):
        sub = metrics.filter(pl.col("corridor") == corridor)
        for model, (label_key, color, marker, style) in VECTOR_MODEL_STYLE.items():
            row = sub.filter(pl.col("model") == model).sort("horizon")
            ax.plot(
                row.get_column("horizon"), row.get_column("cv_bias"),
                label=words[label_key], color=color, marker=marker, linestyle=style,
                linewidth=1.9, markersize=7,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(corridor)
        ax.set_xticks(list(HORIZONS))
        ax.set_xlabel(words["horizon_axis"])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(words["cv_bias_axis"])
    axes[0].text(
        0.02, 0.04, words["compression_hint"], transform=axes[0].transAxes,
        va="bottom", ha="left", fontsize=9, color="dimgray",
    )

    if chrome:
        fig.suptitle(
            "La compresión no es un umbral: se profundiza con cada minuto de horizonte",
            y=0.99, fontsize=12.5,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.935 if chrome else CLEAN_LEGEND_Y),
        ncol=3, frameon=False,
    )
    if chrome:
        _caption(fig, [
            "Sesgo de dispersión = CV predicho − CV observado, sobre los mismos vectores que mide la figura anterior. La persistencia propaga el vector",
            "observado, así que hereda su dispersión y se queda pegada al cero: es el control. Los dos aprendices se hunden monótonamente en los cuatro",
            "horizontes y en los tres corredores — la compresión escala con la incertidumbre que hay que promediar, no con la arquitectura que la promedia.",
        ])
    fig.tight_layout(rect=(0, 0.11, 1, 0.88) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


RENDERERS = (
    degradation, volatility, threshold_artifact, detection_without_threshold,
    dispersion_compression, dispersion_vs_horizon,
)


def main() -> None:
    for renderer in RENDERERS:
        path = renderer()
        print(f"Figura escrita en {path.relative_to(REPO_ROOT)}")
    # Clean bilingual variants, for the subset the manuscript reproduces. Driven
    # off FIGURE_NAMES so adding a paper stem is the only edit a new one needs.
    for renderer in RENDERERS:
        if FIGURE_NAMES[renderer.__name__][1] is None:
            continue
        for lang in LANG:
            path = renderer(lang=lang, chrome=False)
            print(f"Figura escrita en {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
