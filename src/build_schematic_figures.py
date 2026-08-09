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

SPLIT_STYLE = {
    "train": ("Entrenamiento", "tab:blue"),
    "val": ("Validación", "tab:orange"),
    "test": ("Prueba", "tab:red"),
}


def _caption(fig, lines: list[str], *, bottom: float = 0.012) -> None:
    """Render a multi-line italic caption anchored at the figure bottom.

    Hand-wrapped for the same reason as in build_contiguous_figures: matplotlib
    does not wrap ``fig.text``, it clips it.
    """
    for index, line in enumerate(reversed(lines)):
        fig.text(
            0.5, bottom + index * CAPTION_LINE_HEIGHT, line,
            ha="center", fontsize=CAPTION_SIZE, style="italic",
        )


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
def pipeline() -> Path:
    """Ping to verdict, in two rows: what happens to the data, and where."""
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.suptitle(
        "Del ping de GPS al veredicto — el recorrido completo",
        y=0.965, fontsize=12.5,
    )

    # Row 1 — what the data becomes.
    row1_y, box_h = 62.0, 17.0
    stages = [
        ("Ping de GPS\n(empresa, unidad,\nhora, lat, lon)", PING_COLOR),
        ("Limpieza\n(10 problemas\nmedidos)", PING_COLOR),
        ("Eje del corredor\n+ proyección\n(2D → 1D)", AXIS_COLOR),
        ("Vector de\n*headways*\npor minuto", AXIS_COLOR),
        ("Muestra:\nventana 12 min\n→ objetivo", AXIS_COLOR),
    ]
    width, gap = 16.0, 4.0
    for index, (text, color) in enumerate(stages):
        x = 2.0 + index * (width + gap)
        _box(ax, (x, row1_y), width, box_h, text.replace("*", ""), color)
        if index:
            _arrow(ax, (x - gap, row1_y + box_h / 2), (x, row1_y + box_h / 2))

    ax.text(
        1.0, row1_y + box_h + 4.5, "Partes I y II — fabricar el dato",
        fontsize=10.5, style="italic", color="dimgray",
    )

    # Row 2 — what is done with it.
    row2_y = 20.0
    stages2 = [
        ("Modelos:\npersistencia,\nXGBoost, LSTM", "tab:green"),
        ("Predicción del\nvector a 1, 3,\n5 y 10 min", "tab:green"),
        ("Umbral de\ndecisión sobre\nlo predicho", ACCENT),
        ("Alarma\nsí / no", ACCENT),
        ("Evaluación\npareada +\nsignificancia", "tab:purple"),
    ]
    for index, (text, color) in enumerate(stages2):
        x = 2.0 + index * (width + gap)
        _box(ax, (x, row2_y), width, box_h, text, color)
        if index:
            _arrow(ax, (x - gap, row2_y + box_h / 2), (x, row2_y + box_h / 2))

    ax.text(
        26.0, row2_y + box_h + 4.5, "Partes III, IV y V — experimentar y medir",
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

    _caption(fig, [
        "Las cajas naranjas son el objeto de este trabajo: el umbral de decisión es una capa aparte del modelo, aplicada después,",
        "y es donde apareció el hallazgo. La ejecución corre en Kaggle sobre GPU; el análisis, local.",
    ])
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    path = OUT_DIR / "esquema-pipeline.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 2. The corridor axis
# --------------------------------------------------------------------------- #
def corridor_axis() -> Path:
    """Illustrative ping cloud with the fitted centreline and the 300 m band."""
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
        label="Banda de 300 m — dentro se conserva el ping",
    )
    ax.scatter(on_x, on_y, s=11, color=PING_COLOR, alpha=0.75,
               label="Pings de buses en movimiento")
    ax.scatter(np.concatenate([par_x, dep_x]), np.concatenate([par_y, dep_y]),
               s=11, color=OFFROUTE_COLOR, alpha=0.75,
               label="Pings fuera de ruta — se descartan")
    ax.plot(ax_x, ax_y, color=AXIS_COLOR, linewidth=2.6,
            label="Eje ajustado (50 vértices)")

    par_target = float(np.interp(760.0, ax_x, ax_y)) - 215.0
    ax.annotate(
        "calle paralela", xy=(760.0, par_target), xytext=(700.0, -40.0),
        fontsize=9, color=OFFROUTE_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=OFFROUTE_COLOR, linewidth=1.1),
    )
    ax.annotate(
        "depósito", xy=(250.0, 60.0), xytext=(130.0, -40.0),
        fontsize=9, color=OFFROUTE_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=OFFROUTE_COLOR, linewidth=1.1),
    )

    ax.set_title("El eje del corredor, reconstruido desde los propios pings",
                 fontsize=12.5)
    ax.set_xlabel("Longitud →")
    ax.set_ylabel("Latitud →")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    _caption(fig, [
        "Esquema ilustrativo, no datos reales. Solo se usan pings con velocidad ≥ 10 km/h: los buses detenidos en terminales",
        "acumulan cientos de puntos en un mismo lugar y deforman el ajuste. En la calibración real el filtro de 300 m descartó el 43.7 % de los pings.",
    ])
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    path = OUT_DIR / "esquema-eje-corredor.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 3. Two dimensions into one
# --------------------------------------------------------------------------- #
def projection() -> Path:
    """A ping, its foot on the axis, and the arc length that replaces it."""
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

    ax.plot(ax_x, ax_y, color=AXIS_COLOR, linewidth=2.6, label="Eje del corredor")
    ax.plot(ax_x[:foot + 1], ax_y[:foot + 1], color=ACCENT, linewidth=6.0,
            alpha=0.50, solid_capstyle="butt",
            label="s — metros recorridos sobre el eje, desde el inicio")

    ax.scatter([px], [py], s=95, color="black", zorder=5)
    ax.text(px + 14.0, py + 12.0, "ping (lat, lon)", fontsize=10, va="bottom")

    ax.plot([px, ax_x[foot]], [py, ax_y[foot]], linestyle=":", color="black",
            linewidth=1.6)
    ax.scatter([ax_x[foot]], [ax_y[foot]], s=90, marker="o",
               facecolor="white", edgecolor=ACCENT, linewidth=2.2, zorder=6)
    ax.annotate(
        "proyección sobre el eje", xy=(ax_x[foot], ax_y[foot]),
        xytext=(ax_x[foot] + 60.0, ax_y[foot] - 130.0),
        fontsize=9.5, color=ACCENT,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=1.2),
    )

    ax.scatter([ax_x[0]], [ax_y[0]], s=80, marker="s", color=AXIS_COLOR, zorder=5)
    ax.text(ax_x[0] + 8.0, ax_y[0] - 60.0, "inicio del corredor\n(s = 0)",
            fontsize=9, ha="left", va="top")

    ax.annotate(
        "este largo es s:\nun solo número reemplaza al par (lat, lon)",
        xy=(ax_x[150], ax_y[150]), xytext=(ax_x[60], ax_y[60] + 150.0),
        fontsize=10, color=ACCENT, ha="left",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=1.3),
    )

    ax.set_title("De dos dimensiones a una: cuántos metros lleva recorridos el bus",
                 fontsize=12.5)
    ax.set_xlabel("Longitud →")
    ax.set_ylabel("Latitud →")
    ax.set_xticks([])
    ax.set_yticks([])
    # Equal aspect: a perpendicular projection has to look perpendicular.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(40.0, 1080.0)
    ax.set_ylim(60.0, 640.0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    _caption(fig, [
        "Esquema ilustrativo, no datos reales. Con latitud y longitud no se puede decidir sin ambigüedad cuál bus va adelante.",
        "Con s sí: el que tiene el número mayor va más adelantado. Todo el cálculo de intervalos depende de poder ordenar los buses.",
    ])
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    path = OUT_DIR / "esquema-proyeccion.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 4. The headway definition
# --------------------------------------------------------------------------- #
def headway() -> Path:
    """Time-space diagram: the headway as a backward crossing at fixed position."""
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

    ax.plot(t, lead, color=AXIS_COLOR, linewidth=2.4, label="Bus de adelante")
    ax.plot(t, follow, color=OFFROUTE_COLOR, linewidth=2.4, label="Bus de atrás")

    ax.axhline(s_now, color="dimgray", linestyle=":", linewidth=1.3)
    ax.text(0.4, s_now + 6.0, "un mismo punto del corredor",
            fontsize=9, color="dimgray")

    ax.scatter([t_now], [s_now], s=95, color=OFFROUTE_COLOR, zorder=5)
    ax.scatter([t_cross], [s_now], s=95, color=AXIS_COLOR, zorder=5)

    ax.annotate(
        "", xy=(t_now, s_now), xytext=(t_cross, s_now),
        arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, linewidth=2.4),
    )
    ax.text(
        (t_now + t_cross) / 2.0, s_now + 13.0,
        "headway = este tiempo",
        ha="center", fontsize=11.5, color=ACCENT, fontweight="bold",
    )

    ax.annotate(
        "el de atrás\nestá acá ahora", xy=(t_now, s_now),
        xytext=(t_now + 1.0, s_now - 145.0), fontsize=9.5,
        color=OFFROUTE_COLOR, ha="left",
        arrowprops=dict(arrowstyle="-|>", color=OFFROUTE_COLOR, linewidth=1.1),
    )
    ax.annotate(
        "el de adelante pasó\npor acá antes", xy=(t_cross, s_now),
        xytext=(t_cross - 1.0, s_now - 145.0), fontsize=9.5,
        color=AXIS_COLOR, ha="right",
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, linewidth=1.1),
    )
    ax.set_xlim(t[0] - 1.0, t[-1] + 1.0)

    ax.set_title(
        "El headway: hace cuánto el bus de adelante pasó por donde está ahora el de atrás",
        fontsize=12.5,
    )
    ax.set_xlabel("Tiempo [min] →")
    ax.set_ylabel("Distancia recorrida sobre el eje (s) →")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    _caption(fig, [
        "Trayectorias ilustrativas, no datos reales. Es una definición de cruce por posición, no por parada: no necesita una tabla de paradas,",
        "que es exactamente lo que falta en estos datos.",
        "Se calcula para cada par de buses consecutivos en el mismo sentido; con N buses circulando, el vector tiene N − 1 números.",
        "Si el cruce hallado tiene más de 30 minutos de antigüedad se emite «sin dato», para no arrastrar pasos de horas o días antes.",
    ])
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    path = OUT_DIR / "esquema-headway.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 5. Splits and rolling origins
# --------------------------------------------------------------------------- #
def temporal_split() -> Path:
    """The reported split on top, the three rolling origins underneath."""
    origins = [
        ("Origen 3 — el que se reporta",
         date(2023, 10, 1), date(2024, 1, 15),
         date(2024, 1, 16), date(2024, 2, 7),
         date(2024, 2, 8), date(2024, 2, 29)),
        ("Origen 2 — calibra el umbral",
         date(2023, 10, 1), date(2023, 12, 22),
         date(2023, 12, 23), date(2024, 1, 13),
         date(2024, 1, 14), date(2024, 2, 4)),
        ("Origen 1 — réplica más antigua",
         date(2023, 10, 1), date(2023, 11, 30),
         date(2023, 12, 1), date(2023, 12, 22),
         date(2023, 12, 23), date(2024, 1, 13)),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    height = 0.52
    # Rows are pitched wider than their bars so the calibration note has a
    # lane of its own instead of sitting on top of the row above it.
    ROW_PITCH = 1.45

    for row, (label, tr0, tr1, va0, va1, te0, te1) in enumerate(origins):
        y = (len(origins) - 1 - row) * ROW_PITCH
        for (start, end), key in (
            ((tr0, tr1), "train"), ((va0, va1), "val"), ((te0, te1), "test")
        ):
            name, color = SPLIT_STYLE[key]
            ax.barh(
                y, (end - start).days + 1, left=start, height=height,
                color=color, alpha=0.80, edgecolor="white", linewidth=0.8,
                label=name if row == 0 else None,
            )
        ax.text(
            tr0, y + height / 2 + 0.10, label,
            fontsize=9.5, va="bottom", color="black",
        )
        ax.text(
            tr0, y, f"  {(tr1 - tr0).days + 1} d de entrenamiento",
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
        "el umbral se calibra sobre esta ventana\ny se aplica hacia adelante",
        fontsize=9, color=ACCENT, ha="left", va="top",
    )

    ax.set_yticks([])
    ax.set_ylim(-0.55, (len(origins) - 1) * ROW_PITCH + 0.55)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("2023-10-01 → 2024-02-29 · 152 días seguidos, sin huecos")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(
        "Partición por tiempo, y los tres orígenes de evaluación",
        fontsize=12.5,
    )
    ax.legend(loc="lower right", fontsize=9, ncol=3, framealpha=0.9)

    _caption(fig, [
        "Nunca al azar: un operador solo tiene el pasado. Los tres orígenes arrancan el mismo día y el entrenamiento se alarga —61, 83 y 107 días—,",
        "y sus tres períodos de prueba no se solapan entre sí. Como los entrenamientos están anidados, esto establece estabilidad frente a la",
        "elección del período de prueba, no réplica independiente. Se declara así.",
    ])
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    path = OUT_DIR / "esquema-particion-temporal.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for builder in (pipeline, corridor_axis, projection, headway, temporal_split):
        print(f"wrote {builder().relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
