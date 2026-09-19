"""The bunching-rule schematic: the same headway under two thresholds.

The manuscript's Figure 1 used to be two hand-drawn Excalidraw strips exported
at 609x71 px, which print pixelated in a two-column layout and narrate the
threshold in the caption instead of drawing it. This builder replaces them with
a two-panel bar chart in the same matplotlib style as the data figures: one bar
per vector position, the dashed line of the relative threshold tau = mean/2 in
each panel, and the 2.0-minute bar present in both. The reader sees the bar
keep its height while the line moves — which is the whole mechanism of
Section II-C — instead of computing it from the caption.

The vectors are the illustrative examples the caption already prints, not data:
(a) irregular corridor [9.5, 1.2, 11.0, 2.0], (b) regular corridor
[3.5, 2.0, 4.0, 3.0]. Lines are drawn at the exact computed mean and threshold;
labels round to one decimal, matching the caption.

Outputs
-------
``docs/paper/figuras/bunching-umbral.es.png``
``docs/paper/figuras/bunching-umbral.en.png``

Usage
-----
    uv run python -m src.build_bunching_panel
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in WSL/CI

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "docs" / "paper" / "figuras"

DPI = 150
RHO = 0.5

# The two illustrative vectors of the manuscript's Figure 1, front position last.
PANELS: tuple[tuple[str, list[float]], ...] = (
    ("irregular", [9.5, 1.2, 11.0, 2.0]),
    ("regular", [3.5, 2.0, 4.0, 3.0]),
)
SHARED_HEADWAY = 2.0  # the value both panels carry, classified in opposite ways

LANG = {
    "es": {
        "irregular": "(a) Corredor irregular",
        "regular": "(b) Corredor regular",
        "headway_axis": "Headway [min]",
        "position_axis": "Posición del vector",
        "mean": "promedio",
        "tau": "umbral τ = promedio/2",
        "bunching": "bunching: headway < τ",
        "no_bunching": "headway ≥ τ",
        "shared": "el mismo\nheadway, 2.0",
    },
    "en": {
        "irregular": "(a) Irregular corridor",
        "regular": "(b) Regular corridor",
        "headway_axis": "Headway [min]",
        "position_axis": "Vector position",
        "mean": "mean",
        "tau": "threshold τ = mean/2",
        "bunching": "bunching: headway < τ",
        "no_bunching": "headway ≥ τ",
        "shared": "the same\nheadway, 2.0",
    },
}

COLOR_BUNCHING = "tab:red"
COLOR_NORMAL = "0.75"


def build(lang: str) -> "plt.Figure":
    labels = LANG[lang]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)

    for ax, (key, vector) in zip(axes, PANELS):
        mean = sum(vector) / len(vector)
        tau = RHO * mean
        positions = list(range(1, len(vector) + 1))

        colors = [COLOR_BUNCHING if h < tau else COLOR_NORMAL for h in vector]
        edges = ["black" if h == SHARED_HEADWAY else "none" for h in vector]
        widths = [1.6 if h == SHARED_HEADWAY else 0.0 for h in vector]
        ax.bar(positions, vector, color=colors, edgecolor=edges, linewidth=widths)

        ax.axhline(mean, color="0.45", linestyle=":", linewidth=1.2)
        ax.axhline(tau, color="black", linestyle="--", linewidth=1.4)
        # Only the numeric level goes next to each line; its name lives in the
        # legend, where it cannot collide with a bar.
        ax.text(4.82, mean + 0.14, f"{mean:.1f}", ha="right", fontsize=8,
                color="0.35")
        ax.text(4.82, tau + 0.14, f"{tau:.1f}", ha="right", fontsize=8)

        for x, h in zip(positions, vector):
            if h != SHARED_HEADWAY:
                ax.text(x, h + 0.15, f"{h:.1f}", ha="center", fontsize=8)
        shared_x = positions[vector.index(SHARED_HEADWAY)]
        ax.annotate(labels["shared"], xy=(shared_x, SHARED_HEADWAY),
                    xytext=(shared_x + 0.05, SHARED_HEADWAY + 4.6),
                    fontsize=8.5, ha="center",
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.9})

        ax.set_title(labels[key], fontsize=10)
        ax.set_xlabel(labels["position_axis"], fontsize=9)
        ax.set_xticks(positions)
        ax.set_xlim(0.4, 4.9)
        ax.set_ylim(0, 12.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)

    axes[0].set_ylabel(labels["headway_axis"], fontsize=9)
    axes[1].legend(
        handles=[
            Line2D([], [], color="black", linestyle="--", linewidth=1.4,
                   label=labels["tau"]),
            Line2D([], [], color="0.45", linestyle=":", linewidth=1.2,
                   label=labels["mean"]),
            Patch(facecolor=COLOR_BUNCHING, label=labels["bunching"]),
            Patch(facecolor=COLOR_NORMAL, label=labels["no_bunching"]),
        ],
        loc="upper right", fontsize=8, frameon=False,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    for lang in LANG:
        fig = build(lang)
        path = PAPER_DIR / f"bunching-umbral.{lang}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
