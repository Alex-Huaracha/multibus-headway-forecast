"""Explanatory diagrams for docs/paper/metodologia.md.

These are schematics, not result plots: they carry no measurement and are built
from geometry declared here rather than from the committed CSVs. They exist
because four steps of the pipeline are inherently spatial and cost a reader more
in prose than in a picture — reconstructing the corridor axis, collapsing two
dimensions into one, defining the headway by backward crossing, and laying the
three evaluation origins on a calendar.

    esquema-pipeline.png            The whole flow, ping to verdict.
    esquema-eje-corredor.png        Ping cloud and the axis fitted through it.
    esquema-proyeccion.png          A ping projected onto the axis: (lat, lon) -> s.
    esquema-headway.png             Time-space diagram: the backward crossing.
    esquema-particion-temporal.png  Splits and the three rolling origins.

Two variants of each figure
---------------------------
The **chrome** variant is what ``metodologia.md`` and ``sintesis.md`` embed: the
Spanish title plus the hand-wrapped italic footnote, written under the names
listed above. Those documents have no caption machinery of their own, so the
figure has to carry its own argument.

The **clean** variant drops both and lands beside it as ``<stem>.<lang>.png``,
because the manuscript numbers and captions its own figures — a baked-in title
there would be a second, competing title. Only the two schematics ``paper.md``
reproduces have a clean variant, and each is emitted in both languages. The
plotting code is written once and parameterised; the two variants differ only in
the chrome, the layout band it needs, and the language of the labels.

Unlike ``build_contiguous_figures``, both variants land in the SAME directory:
the chrome schematics were always paper assets, so there is no second document
tree to write into. Hence ``esquema-headway.png`` is the chrome Spanish figure
and ``esquema-headway.es.png`` is the clean Spanish one.

Determinism: the only stochastic element is the illustrative ping cloud, drawn
from a seeded generator, so repeated runs are byte-identical (CLAUDE.md contract).

Usage
-----
    uv run python -m src.build_schematic_figures
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in WSL/CI

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "paper" / "figuras"

CAPTION_SIZE = 8.5
DPI = 150
CAPTION_LINE_HEIGHT = 0.030

PING_SEED = 20240208  # first day of the reported test window; any fixed value works

# Shared palette, kept close to build_contiguous_figures so the document reads as
# one set of figures rather than two.
AXIS_COLOR = "tab:blue"
PING_COLOR = "tab:gray"
OFFROUTE_COLOR = "tab:red"
ACCENT = "tab:orange"

# Every user-facing string lives here, keyed by language: the paper needs the
# same schematics in Spanish and in English, and duplicating the drawing code to
# get them would let the two drift. Nothing that renders as text may be a literal
# below this table — that is the whole point of it. Values that read the same in
# both languages (``ping (lat, lon)``) are still listed twice, so a later edit
# has one obvious place to change and cannot silently half-apply.
#
# Captions are stored as tuples of already-wrapped lines, one entry per rendered
# line, for the reason documented on ``_caption``.
LANG = {
    "es": {
        # --- 1. pipeline ---
        "pipeline_title": "Del ping de GPS al veredicto — el recorrido completo",
        "stage_ping": "Ping de GPS\n(empresa, unidad,\nhora, lat, lon)",
        "stage_cleaning": "Limpieza\n(10 problemas\nmedidos)",
        "stage_axis": "Eje del corredor\n+ proyección\n(2D → 1D)",
        "stage_vector": "Vector de\n*headways*\npor minuto",
        "stage_sample": "Muestra:\nventana 12 min\n→ objetivo",
        "pipeline_band_data": "Partes I y II — fabricar el dato",
        "stage_models": "Modelos:\npersistencia,\nXGBoost, LSTM",
        "stage_forecast": "Predicción del\nvector a 1, 3,\n5 y 10 min",
        "stage_threshold": "Umbral de\ndecisión sobre\nlo predicho",
        "stage_alarm": "Alarma\nsí / no",
        "stage_evaluation": "Evaluación\npareada +\nsignificancia",
        "pipeline_band_experiment": "Partes III, IV y V — experimentar y medir",
        "pipeline_caption": (
            "Las cajas naranjas son el objeto de este trabajo: el umbral de decisión es una capa aparte del modelo, aplicada después,",
            "y es donde apareció el hallazgo. La ejecución corre en Kaggle sobre GPU; el análisis, local.",
        ),
        # --- 2. corridor axis ---
        "corridor_axis_title": "El eje del corredor, reconstruido desde los propios pings",
        "band_300m": "Banda de 300 m — dentro se conserva el ping",
        "moving_pings": "Pings de buses en movimiento",
        "offroute_pings": "Pings fuera de ruta — se descartan",
        "fitted_axis": "Eje ajustado (50 vértices)",
        "parallel_street": "calle paralela",
        "depot": "depósito",
        "corridor_axis_caption": (
            "Esquema ilustrativo, no datos reales. Solo se usan pings con velocidad ≥ 10 km/h: los buses detenidos en terminales",
            "acumulan cientos de puntos en un mismo lugar y deforman el ajuste. En la calibración real el filtro de 300 m descartó el 43.7 % de los pings.",
        ),
        # --- 3. projection (shares the lat/lon axis labels with 2) ---
        "lon_axis": "Longitud →",
        "lat_axis": "Latitud →",
        "projection_title": "De dos dimensiones a una: cuántos metros lleva recorridos el bus",
        "corridor_axis_legend": "Eje del corredor",
        "arc_length_legend": "s — metros recorridos sobre el eje, desde el inicio",
        "ping_label": "ping (lat, lon)",
        "projection_foot": "proyección sobre el eje",
        "corridor_start": "inicio del corredor\n(s = 0)",
        "arc_length_callout": "este largo es s:\nun solo número reemplaza al par (lat, lon)",
        "projection_caption": (
            "Esquema ilustrativo, no datos reales. Con latitud y longitud no se puede decidir sin ambigüedad cuál bus va adelante.",
            "Con s sí: el que tiene el número mayor va más adelantado. Todo el cálculo de intervalos depende de poder ordenar los buses.",
        ),
        # --- 4. headway ---
        "headway_title": "El headway: hace cuánto el bus de adelante pasó por donde está ahora el de atrás",
        "leading_bus": "Bus de adelante",
        "following_bus": "Bus de atrás",
        "same_point": "un mismo punto del corredor",
        "headway_callout": "headway = este tiempo",
        "follower_now": "el de atrás\nestá acá ahora",
        "leader_before": "el de adelante pasó\npor acá antes",
        "time_axis": "Tiempo [min] →",
        "distance_axis": "Distancia recorrida sobre el eje (s) →",
        "headway_caption": (
            "Trayectorias ilustrativas, no datos reales. Es una definición de cruce por posición, no por parada: no necesita una tabla de paradas,",
            "que es exactamente lo que falta en estos datos.",
            "Se calcula para cada par de buses consecutivos en el mismo sentido; con N buses circulando, el vector tiene N − 1 números.",
            "Si el cruce hallado tiene más de 30 minutos de antigüedad se emite «sin dato», para no arrastrar pasos de horas o días antes.",
        ),
        # --- 5. temporal split ---
        "temporal_split_title": "Partición por tiempo, y los tres orígenes de evaluación",
        "split_train": "Entrenamiento",
        "split_val": "Validación",
        "split_test": "Prueba",
        "origin_3": "Origen 3 — el que se reporta",
        "origin_2": "Origen 2 — calibra el umbral",
        "origin_1": "Origen 1 — réplica más antigua",
        "train_days": "  {days} d de entrenamiento",
        "threshold_note": "el umbral se calibra sobre esta ventana\ny se aplica hacia adelante",
        "calendar_axis": "2023-10-01 → 2024-02-29 · 152 días seguidos, sin huecos",
        "temporal_split_caption": (
            "Nunca al azar: un operador solo tiene el pasado. Los tres orígenes arrancan el mismo día y el entrenamiento se alarga —61, 83 y 107 días—,",
            "y sus tres períodos de prueba no se solapan entre sí. Como los entrenamientos están anidados, esto establece estabilidad frente a la",
            "elección del período de prueba, no réplica independiente. Se declara así.",
        ),
    },
    "en": {
        # --- 1. pipeline ---
        "pipeline_title": "From GPS ping to verdict — the complete path",
        "stage_ping": "GPS ping\n(company, vehicle,\ntime, lat, lon)",
        "stage_cleaning": "Cleaning\n(10 measured\ndefects)",
        "stage_axis": "Corridor axis\n+ projection\n(2D → 1D)",
        "stage_vector": "Per-minute\n*headway*\nvector",
        "stage_sample": "Sample:\n12 min window\n→ target",
        "pipeline_band_data": "Parts I and II — building the data",
        "stage_models": "Models:\npersistence,\nXGBoost, LSTM",
        "stage_forecast": "Vector forecast\nat 1, 3,\n5 and 10 min",
        "stage_threshold": "Decision\nthreshold on\nthe forecast",
        "stage_alarm": "Alarm\nyes / no",
        "stage_evaluation": "Paired\nevaluation +\nsignificance",
        "pipeline_band_experiment": "Parts III, IV and V — experiment and measure",
        "pipeline_caption": (
            "The orange boxes are what this work is about: the decision threshold is a layer separate from the model, applied afterwards,",
            "and that is where the finding appeared. Training runs on Kaggle GPUs; the analysis runs locally.",
        ),
        # --- 2. corridor axis ---
        "corridor_axis_title": "The corridor axis, reconstructed from the pings themselves",
        "band_300m": "300 m band — pings inside it are kept",
        "moving_pings": "Pings from moving buses",
        "offroute_pings": "Off-route pings — discarded",
        "fitted_axis": "Fitted axis (50 vertices)",
        "parallel_street": "parallel street",
        "depot": "depot",
        "corridor_axis_caption": (
            "Illustrative schematic, not real data. Only pings with speed ≥ 10 km/h are used: buses standing at terminals",
            "pile up hundreds of points in one spot and distort the fit. In the actual calibration the 300 m filter discarded 43.7 % of the pings.",
        ),
        # --- 3. projection (shares the lat/lon axis labels with 2) ---
        "lon_axis": "Longitude →",
        "lat_axis": "Latitude →",
        "projection_title": "From two dimensions to one: how far along the corridor the bus has travelled",
        "corridor_axis_legend": "Corridor axis",
        "arc_length_legend": "s — metres travelled along the axis, from its start",
        "ping_label": "ping (lat, lon)",
        "projection_foot": "projection onto the axis",
        "corridor_start": "corridor start\n(s = 0)",
        "arc_length_callout": "this length is s:\na single number replaces the (lat, lon) pair",
        "projection_caption": (
            "Illustrative schematic, not real data. Latitude and longitude cannot settle unambiguously which bus is ahead of which.",
            "Arc length s can: the larger value is the one further along. The whole headway computation depends on being able to order the buses.",
        ),
        # --- 4. headway ---
        "headway_title": "The headway: how long ago the leading bus passed the point where the follower is now",
        "leading_bus": "Leading bus",
        "following_bus": "Following bus",
        "same_point": "one and the same point of the corridor",
        "headway_callout": "headway = this elapsed time",
        "follower_now": "the follower\nis here now",
        "leader_before": "the leader passed\nthrough here earlier",
        "time_axis": "Time [min] →",
        "distance_axis": "Distance travelled along the axis (s) →",
        "headway_caption": (
            "Illustrative trajectories, not real data. This is a crossing definition by position, not by stop: it needs no stop inventory,",
            "which is exactly what these data lack.",
            "It is computed for every pair of consecutive buses in the same direction; with N buses in service the vector holds N − 1 values.",
            "If the crossing found is more than 30 minutes old the value is emitted as missing, so passages hours or days earlier are not carried forward.",
        ),
        # --- 5. temporal split ---
        "temporal_split_title": "Splitting by time, and the three evaluation origins",
        "split_train": "Training",
        "split_val": "Validation",
        "split_test": "Test",
        "origin_3": "Origin 3 — the one reported",
        "origin_2": "Origin 2 — calibrates the threshold",
        "origin_1": "Origin 1 — earliest replication",
        "train_days": "  {days} d of training",
        "threshold_note": "the threshold is calibrated on this window\nand applied forward",
        "calendar_axis": "2023-10-01 → 2024-02-29 · 152 consecutive days, no gaps",
        "temporal_split_caption": (
            "Never at random: an operator only ever has the past. The three origins start on the same day and the training span grows —61, 83 and 107 days—,",
            "and their three test periods do not overlap. Because the training spans are nested, this establishes stability against the",
            "choice of test period, not independent replication. It is reported as such.",
        ),
    },
}

# Styles carry a LANG key rather than a literal label, so a split keeps the same
# colour in both languages and cannot be relabelled in one only.
SPLIT_STYLE = {
    "train": ("split_train", "tab:blue"),
    "val": ("split_val", "tab:orange"),
    "test": ("split_test", "tab:red"),
}

# Layout for the clean variant. With no title and no footnote the axes take the
# whole canvas; reusing a chrome ``rect`` here would leave the empty bands where
# the title and the caption used to be.
CLEAN_RECT = (0.0, 0.0, 1.0, 1.0)

# Document filename, and the paper stem when the manuscript reproduces the
# figure. ``None`` means there is no clean variant: ``paper.md`` does not carry
# that schematic, so emitting a bilingual pair for it would be dead output.
#
# Paper stems carry NO figure number, for the same reason as in
# build_contiguous_figures: the order of first appearance still moves while the
# remaining sections are written, and only paper.md knows it.
FIGURE_NAMES = {
    "pipeline": ("esquema-pipeline.png", None),
    "corridor_axis": ("esquema-eje-corredor.png", None),
    "projection": ("esquema-proyeccion.png", None),
    "headway": ("esquema-headway.png", "esquema-headway"),
    "temporal_split": (
        "esquema-particion-temporal.png", "esquema-particion-temporal",
    ),
}


def _caption(fig, lines, *, bottom: float = 0.012) -> None:
    """Render a multi-line italic caption anchored at the figure bottom.

    Hand-wrapped for the same reason as in build_contiguous_figures: matplotlib
    does not wrap ``fig.text``, it clips it.
    """
    for index, line in enumerate(reversed(lines)):
        fig.text(
            0.5, bottom + index * CAPTION_LINE_HEIGHT, line,
            ha="center", fontsize=CAPTION_SIZE, style="italic",
        )


def _resolve(figure: str, lang: str, chrome: bool) -> Path:
    """Where a variant is written, and the invariants that pair the two.

    Checked before any drawing so an unbuildable request fails on the first line
    rather than after a render. The chrome variant is Spanish-only: its title and
    footnote are prose written for the Spanish methodology documents, so asking
    for it in English would emit a half-translated figure over the published
    filename.

    Both variants share ``OUT_DIR`` — these schematics were always paper assets —
    so no directory is created here; ``main`` makes the one directory involved.
    """
    if lang not in LANG:
        raise ValueError(f"unknown language {lang!r}; expected one of {sorted(LANG)}")
    doc_name, paper_stem = FIGURE_NAMES[figure]
    if chrome:
        if lang != "es":
            raise ValueError(
                f"{figure}: the chrome variant exists only in Spanish — its "
                "title and footnote belong to the Spanish methodology documents"
            )
        return OUT_DIR / doc_name
    if paper_stem is None:
        raise ValueError(f"{figure}: the paper does not reproduce this figure")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{paper_stem}.{lang}.png"


def _corridor_path(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """A gently bent corridor centreline, in arbitrary metric units."""
    t = np.linspace(0.0, 1.0, n)
    x = 100.0 + 900.0 * t
    y = 300.0 + 180.0 * np.sin(2.2 * t) - 90.0 * t
    return x, y


def _box(ax, xy, w, h, text, color, *, fontsize=9.0) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.13,
        )
    )
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center", fontsize=fontsize, color="black",
    )


def _arrow(ax, start, end, *, color="black", lw=1.3) -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                        shrinkA=0, shrinkB=0),
    )


# --------------------------------------------------------------------------- #
# 1. The whole pipeline
# --------------------------------------------------------------------------- #
def pipeline(*, lang: str = "es", chrome: bool = True) -> Path:
    """Ping to verdict, in two rows: what happens to the data, and where."""
    path = _resolve("pipeline", lang, chrome)
    words = LANG[lang]

    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    if chrome:
        fig.suptitle(words["pipeline_title"], y=0.965, fontsize=12.5)

    # Row 1 — what the data becomes.
    row1_y, box_h = 62.0, 17.0
    stages = [
        (words["stage_ping"], PING_COLOR),
        (words["stage_cleaning"], PING_COLOR),
        (words["stage_axis"], AXIS_COLOR),
        (words["stage_vector"], AXIS_COLOR),
        (words["stage_sample"], AXIS_COLOR),
    ]
    width, gap = 16.0, 4.0
    for index, (text, color) in enumerate(stages):
        x = 2.0 + index * (width + gap)
        _box(ax, (x, row1_y), width, box_h, text.replace("*", ""), color)
        if index:
            _arrow(ax, (x - gap, row1_y + box_h / 2), (x, row1_y + box_h / 2))

    ax.text(
        1.0, row1_y + box_h + 4.5, words["pipeline_band_data"],
        fontsize=10.5, style="italic", color="dimgray",
    )

    # Row 2 — what is done with it.
    row2_y = 20.0
    stages2 = [
        (words["stage_models"], "tab:green"),
        (words["stage_forecast"], "tab:green"),
        (words["stage_threshold"], ACCENT),
        (words["stage_alarm"], ACCENT),
        (words["stage_evaluation"], "tab:purple"),
    ]
    for index, (text, color) in enumerate(stages2):
        x = 2.0 + index * (width + gap)
        _box(ax, (x, row2_y), width, box_h, text, color)
        if index:
            _arrow(ax, (x - gap, row2_y + box_h / 2), (x, row2_y + box_h / 2))

    ax.text(
        26.0, row2_y + box_h + 4.5, words["pipeline_band_experiment"],
        fontsize=10.5, style="italic", color="dimgray",
    )

    # The elbow that links the two rows. Routed through the empty band between
    # them rather than across the boxes, which an earlier arc did.
    x_end = 2.0 + 4 * (width + gap) + width / 2
    x_start = 2.0 + width / 2
    band_y = 50.0
    ax.plot(
        [x_end, x_end, x_start, x_start],
        [row1_y, band_y, band_y, row2_y + box_h],
        color="dimgray", linewidth=1.3, linestyle=(0, (4, 3)), zorder=1,
    )
    _arrow(ax, (x_start, band_y - 4.0), (x_start, row2_y + box_h),
           color="dimgray")

    if chrome:
        _caption(fig, words["pipeline_caption"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.94) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 2. The corridor axis
# --------------------------------------------------------------------------- #
def corridor_axis(*, lang: str = "es", chrome: bool = True) -> Path:
    """Illustrative ping cloud with the fitted centreline and the 300 m band."""
    path = _resolve("corridor_axis", lang, chrome)
    words = LANG[lang]
    rng = np.random.default_rng(PING_SEED)
    ax_x, ax_y = _corridor_path()

    # On-route pings: scattered around the centreline.
    idx = rng.integers(0, ax_x.size, 340)
    on_x = ax_x[idx] + rng.normal(0.0, 26.0, idx.size)
    on_y = ax_y[idx] + rng.normal(0.0, 26.0, idx.size)

    # Off-route pings: a parallel street and a depot, the reason the filter exists.
    par_idx = rng.integers(0, ax_x.size, 70)
    par_x = ax_x[par_idx] + rng.normal(0.0, 18.0, par_idx.size)
    par_y = ax_y[par_idx] - 215.0 + rng.normal(0.0, 14.0, par_idx.size)
    dep_x = 250.0 + rng.normal(0.0, 26.0, 45)
    dep_y = 60.0 + rng.normal(0.0, 24.0, 45)

    fig, ax = plt.subplots(figsize=(11, 5.6))

    ax.fill_between(
        ax_x, ax_y - 100.0, ax_y + 100.0,
        color=AXIS_COLOR, alpha=0.10,
        label=words["band_300m"],
    )
    ax.scatter(on_x, on_y, s=11, color=PING_COLOR, alpha=0.75,
               label=words["moving_pings"])
    ax.scatter(np.concatenate([par_x, dep_x]), np.concatenate([par_y, dep_y]),
               s=11, color=OFFROUTE_COLOR, alpha=0.75,
               label=words["offroute_pings"])
    ax.plot(ax_x, ax_y, color=AXIS_COLOR, linewidth=2.6,
            label=words["fitted_axis"])

    par_target = float(np.interp(760.0, ax_x, ax_y)) - 215.0
    ax.annotate(
        words["parallel_street"], xy=(760.0, par_target), xytext=(700.0, -40.0),
        fontsize=9, color=OFFROUTE_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=OFFROUTE_COLOR, linewidth=1.1),
    )
    ax.annotate(
        words["depot"], xy=(250.0, 60.0), xytext=(130.0, -40.0),
        fontsize=9, color=OFFROUTE_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=OFFROUTE_COLOR, linewidth=1.1),
    )

    if chrome:
        ax.set_title(words["corridor_axis_title"], fontsize=12.5)
    ax.set_xlabel(words["lon_axis"])
    ax.set_ylabel(words["lat_axis"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    if chrome:
        _caption(fig, words["corridor_axis_caption"])
    fig.tight_layout(rect=(0, 0.09, 1, 1) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 3. Two dimensions into one
# --------------------------------------------------------------------------- #
def projection(*, lang: str = "es", chrome: bool = True) -> Path:
    """A ping, its foot on the axis, and the arc length that replaces it."""
    path = _resolve("projection", lang, chrome)
    words = LANG[lang]
    ax_x, ax_y = _corridor_path()

    foot = 260  # index of the projection foot along the centreline

    # Place the ping on the true normal at `foot`, so that with equal aspect the
    # dotted segment renders perpendicular to the axis — which is the whole point
    # of the diagram. Eyeballing an offset does not, and did not, look square.
    tx = ax_x[foot + 1] - ax_x[foot - 1]
    ty = ax_y[foot + 1] - ax_y[foot - 1]
    norm = float(np.hypot(tx, ty))
    px = ax_x[foot] - ty / norm * 130.0
    py = ax_y[foot] + tx / norm * 130.0

    fig, ax = plt.subplots(figsize=(11, 6.0))

    ax.plot(ax_x, ax_y, color=AXIS_COLOR, linewidth=2.6,
            label=words["corridor_axis_legend"])
    ax.plot(ax_x[:foot + 1], ax_y[:foot + 1], color=ACCENT, linewidth=6.0,
            alpha=0.50, solid_capstyle="butt",
            label=words["arc_length_legend"])

    ax.scatter([px], [py], s=95, color="black", zorder=5)
    ax.text(px + 14.0, py + 12.0, words["ping_label"], fontsize=10, va="bottom")

    ax.plot([px, ax_x[foot]], [py, ax_y[foot]], linestyle=":", color="black",
            linewidth=1.6)
    ax.scatter([ax_x[foot]], [ax_y[foot]], s=90, marker="o",
               facecolor="white", edgecolor=ACCENT, linewidth=2.2, zorder=6)
    ax.annotate(
        words["projection_foot"], xy=(ax_x[foot], ax_y[foot]),
        xytext=(ax_x[foot] + 60.0, ax_y[foot] - 130.0),
        fontsize=9.5, color=ACCENT,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=1.2),
    )

    ax.scatter([ax_x[0]], [ax_y[0]], s=80, marker="s", color=AXIS_COLOR, zorder=5)
    ax.text(ax_x[0] + 8.0, ax_y[0] - 60.0, words["corridor_start"],
            fontsize=9, ha="left", va="top")

    ax.annotate(
        words["arc_length_callout"],
        xy=(ax_x[150], ax_y[150]), xytext=(ax_x[60], ax_y[60] + 150.0),
        fontsize=10, color=ACCENT, ha="left",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=1.3),
    )

    if chrome:
        ax.set_title(words["projection_title"], fontsize=12.5)
    ax.set_xlabel(words["lon_axis"])
    ax.set_ylabel(words["lat_axis"])
    ax.set_xticks([])
    ax.set_yticks([])
    # Equal aspect: a perpendicular projection has to look perpendicular.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(40.0, 1080.0)
    ax.set_ylim(60.0, 640.0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    if chrome:
        _caption(fig, words["projection_caption"])
    fig.tight_layout(rect=(0, 0.09, 1, 1) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 4. The headway definition
# --------------------------------------------------------------------------- #
def headway(*, lang: str = "es", chrome: bool = True) -> Path:
    """Time-space diagram: the headway as a backward crossing at fixed position."""
    path = _resolve("headway", lang, chrome)
    words = LANG[lang]
    t = np.linspace(0.0, 30.0, 601)

    # Two trajectories along the corridor, both strictly increasing in time so
    # the backward crossing is well defined. The leader is ahead in space; the
    # sinusoid is a mild speed variation, kept small enough not to break
    # monotonicity (|d/dt| of the sine term stays under the base speed).
    def _run(offset: float, phase: float) -> np.ndarray:
        return 14.0 * t + offset + 22.0 * np.sin((t - phase) / 7.0)

    lead = _run(offset=190.0, phase=0.0)
    follow = _run(offset=0.0, phase=5.0)

    t_now = 23.0
    s_now = float(np.interp(t_now, t, follow))
    t_cross = float(np.interp(s_now, lead, t))  # lead is monotone in t

    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.plot(t, lead, color=AXIS_COLOR, linewidth=2.4, label=words["leading_bus"])
    ax.plot(t, follow, color=OFFROUTE_COLOR, linewidth=2.4,
            label=words["following_bus"])

    ax.axhline(s_now, color="dimgray", linestyle=":", linewidth=1.3)
    ax.text(0.4, s_now + 6.0, words["same_point"],
            fontsize=9, color="dimgray")

    ax.scatter([t_now], [s_now], s=95, color=OFFROUTE_COLOR, zorder=5)
    ax.scatter([t_cross], [s_now], s=95, color=AXIS_COLOR, zorder=5)

    ax.annotate(
        "", xy=(t_now, s_now), xytext=(t_cross, s_now),
        arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, linewidth=2.4),
    )
    ax.text(
        (t_now + t_cross) / 2.0, s_now + 13.0,
        words["headway_callout"],
        ha="center", fontsize=11.5, color=ACCENT, fontweight="bold",
    )

    ax.annotate(
        words["follower_now"], xy=(t_now, s_now),
        xytext=(t_now + 1.0, s_now - 145.0), fontsize=9.5,
        color=OFFROUTE_COLOR, ha="left",
        arrowprops=dict(arrowstyle="-|>", color=OFFROUTE_COLOR, linewidth=1.1),
    )
    ax.annotate(
        words["leader_before"], xy=(t_cross, s_now),
        xytext=(t_cross - 1.0, s_now - 145.0), fontsize=9.5,
        color=AXIS_COLOR, ha="right",
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, linewidth=1.1),
    )
    ax.set_xlim(t[0] - 1.0, t[-1] + 1.0)

    if chrome:
        ax.set_title(words["headway_title"], fontsize=12.5)
    ax.set_xlabel(words["time_axis"])
    ax.set_ylabel(words["distance_axis"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    if chrome:
        _caption(fig, words["headway_caption"])
    fig.tight_layout(rect=(0, 0.12, 1, 1) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 5. Splits and rolling origins
# --------------------------------------------------------------------------- #
def temporal_split(*, lang: str = "es", chrome: bool = True) -> Path:
    """The reported split on top, the three rolling origins underneath."""
    path = _resolve("temporal_split", lang, chrome)
    words = LANG[lang]
    # Row labels are LANG keys rather than literals, for the same reason as
    # SPLIT_STYLE: an origin must not end up named in one language only.
    origins = [
        ("origin_3",
         date(2023, 10, 1), date(2024, 1, 15),
         date(2024, 1, 16), date(2024, 2, 7),
         date(2024, 2, 8), date(2024, 2, 29)),
        ("origin_2",
         date(2023, 10, 1), date(2023, 12, 22),
         date(2023, 12, 23), date(2024, 1, 13),
         date(2024, 1, 14), date(2024, 2, 4)),
        ("origin_1",
         date(2023, 10, 1), date(2023, 11, 30),
         date(2023, 12, 1), date(2023, 12, 22),
         date(2023, 12, 23), date(2024, 1, 13)),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    height = 0.52
    # Rows are pitched wider than their bars so the calibration note has a
    # lane of its own instead of sitting on top of the row above it.
    ROW_PITCH = 1.45

    for row, (label_key, tr0, tr1, va0, va1, te0, te1) in enumerate(origins):
        y = (len(origins) - 1 - row) * ROW_PITCH
        for (start, end), key in (
            ((tr0, tr1), "train"), ((va0, va1), "val"), ((te0, te1), "test")
        ):
            name_key, color = SPLIT_STYLE[key]
            ax.barh(
                y, (end - start).days + 1, left=start, height=height,
                color=color, alpha=0.80, edgecolor="white", linewidth=0.8,
                label=words[name_key] if row == 0 else None,
            )
        ax.text(
            tr0, y + height / 2 + 0.10, words[label_key],
            fontsize=9.5, va="bottom", color="black",
        )
        ax.text(
            tr0, y, words["train_days"].format(days=(tr1 - tr0).days + 1),
            fontsize=8.5, va="center", ha="left", color="white",
            fontweight="bold",
        )

    # The forward calibration: origin 2's test window feeds origin 3's threshold.
    # Routed under the bars so it never crosses one.
    ax.annotate(
        "", xy=(date(2024, 2, 18), 2 - height / 2 - 0.03),
        xytext=(date(2024, 1, 24), 1 + height / 2 + 0.03),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=2.0,
                        connectionstyle="arc3,rad=0.30"),
    )
    ax.text(
        date(2023, 12, 26), 2 * ROW_PITCH - height / 2 - 0.12,
        words["threshold_note"],
        fontsize=9, color=ACCENT, ha="left", va="top",
    )

    ax.set_yticks([])
    ax.set_ylim(-0.55, (len(origins) - 1) * ROW_PITCH + 0.55)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel(words["calendar_axis"])
    ax.grid(True, axis="x", alpha=0.3)
    if chrome:
        ax.set_title(words["temporal_split_title"], fontsize=12.5)
    ax.legend(loc="lower right", fontsize=9, ncol=3, framealpha=0.9)

    if chrome:
        _caption(fig, words["temporal_split_caption"])
    fig.tight_layout(rect=(0, 0.15, 1, 1) if chrome else CLEAN_RECT)

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


RENDERERS = (pipeline, corridor_axis, projection, headway, temporal_split)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for renderer in RENDERERS:
        path = renderer()
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    # Clean bilingual variants, for the subset the manuscript reproduces. Driven
    # off FIGURE_NAMES so adding a paper stem is the only edit a new one needs.
    for renderer in RENDERERS:
        if FIGURE_NAMES[renderer.__name__][1] is None:
            continue
        for lang in LANG:
            path = renderer(lang=lang, chrome=False)
            print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
