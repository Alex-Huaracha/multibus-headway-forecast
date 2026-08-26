"""The paper's three result tables, rendered as Markdown from the committed CSVs.

Same contract as ``build_contiguous_figures``: every cell traces to a CSV under
``docs/resultados/csv-multihorizon/``, so a table can never disagree with the
figure it sits next to, and no number in the paper is ever typed by hand.

    tabla-1-deteccion-corte-trasplantado.md   The artifact. Detection scored with
                                              the cut carried over from the
                                              observations, next to the floor of
                                              the do-nothing detector.
    tabla-2-veredicto-sin-umbral.md           The repair. The same predictions
                                              scored without a cut, and with the
                                              cut refitted out of sample.
    tabla-3-robustez.md                       Does it survive the month, and does
                                              it survive the field's own event
                                              rule instead of ours.

The output is pasted into ``docs/paper/paper.md``. Re-run this and re-paste when
an upstream CSV changes; do not edit a number in the manuscript directly.

Usage
-----
    uv run python -m src.build_paper_tables
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "paper" / "tablas"

CORRIDORS = ("E2", "E4", "E59")
HORIZONS = (1, 3, 5, 10)

# The learner the paper carries. XGBoost appears in the figures as the
# architecture control, but the tables compare the published pair.
LEARNER = "LSTM"
RIVAL = "Persistence"

# Spanish decimal comma: the draft is written in Spanish and translated at the
# end. `_num` is the single place that has to change for the English pass.
DECIMAL_SEP = ","


def _load(name: str) -> pl.DataFrame:
    path = CSV_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name} — run its builder before rendering tables"
        )
    return pl.read_csv(path)


def _num(value: float | None, places: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}".replace(".", DECIMAL_SEP)


def _factor(ratio: float | None) -> str:
    """A ratio that spans 1.5 to 253 needs two formats, not one.

    Rounding everything to an integer turns the 1.5x cell into "1x", which reads
    as "no difference" — the opposite of what it says.
    """
    if ratio is None:
        return "—"
    if ratio < 10:
        return _num(ratio, 1) + "×"
    return f"{ratio:.0f}×"


def _render(headers: list[str], rows: list[list[str]], *, aligns: str) -> str:
    """Markdown table. `aligns` is one char per column: l, c or r."""
    if len(aligns) != len(headers):
        raise ValueError("aligns must have one entry per column")
    rule = {"l": ":---", "c": ":---:", "r": "---:"}
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(rule[a] for a in aligns) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _cell(df: pl.DataFrame, column: str, **keys) -> float | None:
    sub = df
    for key, value in keys.items():
        sub = sub.filter(pl.col(key) == value)
    if sub.height != 1:
        return None
    return sub[column][0]


def tabla_1() -> str:
    """Detection under the transplanted cut, against the trivial floor.

    The dagger is the point of the table: it marks the cells where marking
    *every* cell — a rule with no content — beats the declared winner. A ranking
    that an empty rule can win is not ranking anything.
    """
    det = _load("contiguous_detection_calibrated.csv")

    rows: list[list[str]] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            keys = {"corridor": corridor, "horizon": horizon}
            base = _cell(det, "base_rate", model=LEARNER, **keys)
            floor = _cell(det, "trivial_f1", model=LEARNER, **keys)
            f1_rival = _cell(det, "f1_fixed", model=RIVAL, **keys)
            f1_learner = _cell(det, "f1_fixed", model=LEARNER, **keys)
            ratio = f1_rival / f1_learner if f1_learner else None
            beaten = floor is not None and f1_rival is not None and f1_rival < floor
            rows.append([
                corridor, str(horizon), _num(base), _num(floor),
                _num(f1_rival) + ("&nbsp;†" if beaten else ""),
                _num(f1_learner),
                _factor(ratio),
            ])

    table = _render(
        ["Corredor", "h", "Tasa base", "Piso trivial", "F1 persistencia",
         "F1 aprendiz", "Factor"],
        rows,
        aligns="lrrrrrr",
    )
    note = (
        "\n\n† La regla vacía —marcar todas las celdas— supera al ganador "
        "declarado en estas celdas."
    )
    return table + note


def tabla_2() -> str:
    """The same predictions scored without a cut, and with the cut refitted.

    Two scorings sit side by side on purpose. The area under the curve needs no
    operating point at all; the refitted Matthews correlation keeps one but
    fixes it on a window the scoring never sees. They agree on the direction,
    which is what makes the repair a repair and not a second artifact.
    """
    det = _load("contiguous_detection_calibrated.csv")

    rows: list[list[str]] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            keys = {"corridor": corridor, "horizon": horizon}
            auc_l = _cell(det, "auc", model=LEARNER, **keys)
            auc_r = _cell(det, "auc", model=RIVAL, **keys)
            mcc_l = _cell(det, "mcc_calibrated", model=LEARNER, **keys)
            mcc_r = _cell(det, "mcc_calibrated", model=RIVAL, **keys)
            winner = "aprendiz" if auc_l > auc_r else "persistencia"
            rows.append([
                corridor, str(horizon),
                f"**{_num(auc_l)}**" if auc_l > auc_r else _num(auc_l),
                f"**{_num(auc_r)}**" if auc_r > auc_l else _num(auc_r),
                f"**{_num(mcc_l)}**" if mcc_l > mcc_r else _num(mcc_l),
                f"**{_num(mcc_r)}**" if mcc_r > mcc_l else _num(mcc_r),
                winner,
            ])

    return _render(
        ["Corredor", "h", "AUC aprendiz", "AUC persist.",
         "MCC recal. aprendiz", "MCC recal. persist.", "Gana AUC"],
        rows,
        aligns="lrrrrrl",
    )


def tabla_3() -> str:
    """Robustness on two axes: the month, and whose event rule.

    The last column is the paper attacking itself. It re-scores the learner
    under the field's dominant convention — an absolute cut at a quarter of the
    reference — rather than under the relative rule this work proposes. The
    finding survives, and the one cell where it does not is visible here rather
    than buried in prose.
    """
    roll = _load("rolling_origin_dissociation_agreement.csv")
    absolute = _load("threshold_absolute_comparison.csv").filter(
        (pl.col("model") == LEARNER) & (pl.col("absolute_ratio") == 0.25)
    )

    label = {"lstm": "aprendiz", "persist": "persist.", "persistence": "persist."}

    rows: list[list[str]] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            keys = {"corridor": corridor, "horizon": horizon}
            winners = [
                str(_cell(roll, f"winner_auc_{origin}", **keys)).lower()
                for origin in ("r1", "r2", "main")
            ]
            agrees = _cell(roll, "agrees_auc", **keys)
            auc_abs = _cell(absolute, "auc_absolute", **keys)
            chance = auc_abs is not None and auc_abs < 0.52
            rows.append([
                corridor, str(horizon),
                *[label.get(w, w) for w in winners],
                "sí" if agrees else "**no**",
                _num(auc_abs) + ("&nbsp;‡" if chance else ""),
            ])

    table = _render(
        ["Corredor", "h", "Ventana 1", "Ventana 2", "Ventana 3", "Coinciden",
         "AUC, corte absoluto"],
        rows,
        aligns="lrlllcr",
    )
    note = (
        "\n\n‡ Indistinguible del azar. Es el único punto donde la afirmación "
        "no se sostiene bajo la convención del campo, y se declara como tal."
    )
    return table + note


TABLES = {
    "tabla-1-deteccion-corte-trasplantado.md": tabla_1,
    "tabla-2-veredicto-sin-umbral.md": tabla_2,
    "tabla-3-robustez.md": tabla_3,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in TABLES.items():
        path = OUT_DIR / filename
        path.write_text(builder() + "\n", encoding="utf-8")
        print(f"Tabla escrita en {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
