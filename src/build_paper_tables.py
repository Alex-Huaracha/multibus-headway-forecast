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
    tabla-4-cobertura-headway.md              How much of the corpus the headway
                                              construction answered, per
                                              corridor.
    tabla-5-formulaciones-headway.md          The four candidate headway
                                              definitions the viability probe
                                              compared, on the two dimensions
                                              that decided the discards.

Tables 1 to 3 are pasted into ``docs/paper/paper.md`` as numbered tables. Tables
4 and 5 are not: Sections IV-A and III-A quote their figures in prose, and the
files exist so those figures have a regenerable source instead of living only in
the manuscript. Re-run this and re-paste when an upstream CSV changes; do not
edit a number in the manuscript directly.

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

from src.build_headway_coverage import aggregate_coverage  # noqa: E402

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


def _int(value: int) -> str:
    """Thousands grouped the Spanish way, with a space that cannot wrap."""
    return f"{int(value):,}".replace(",", "&nbsp;")


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
    *every position* — a rule with no content — beats the declared winner. A
    ranking that an empty rule can win is not ranking anything.

    "Position", not "cell": a cell of this table is one corridor x horizon, and
    the trivial detector flags every position of every vector inside it. The
    manuscript uses "celda" for the former throughout, so the footnote has to
    name the latter.
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
         f"F1 {LEARNER}", "Factor"],
        rows,
        aligns="lrrrrrr",
    )
    note = (
        "\n\n† La regla vacía —marcar toda posición— supera al ganador "
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
            winner = LEARNER if auc_l > auc_r else "persistencia"
            rows.append([
                corridor, str(horizon),
                f"**{_num(auc_l)}**" if auc_l > auc_r else _num(auc_l),
                f"**{_num(auc_r)}**" if auc_r > auc_l else _num(auc_r),
                f"**{_num(mcc_l)}**" if mcc_l > mcc_r else _num(mcc_l),
                f"**{_num(mcc_r)}**" if mcc_r > mcc_l else _num(mcc_r),
                winner,
            ])

    return _render(
        ["Corredor", "h", f"AUC {LEARNER}", "AUC persist.",
         f"MCC recal. {LEARNER}", "MCC recal. persist.", "Gana AUC"],
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

    label = {"lstm": LEARNER, "persist": "persist.", "persistence": "persist."}

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


def tabla_4() -> str:
    """How much of the corpus the headway construction actually answered.

    Section IV-A quotes the percentages; the counts are here so the reader can
    see the denominator. Both directions are summed because the paper compares
    corridors, not directions — the directional split stays in the CSV.
    """
    counts = _load("headway_coverage.csv")
    aggregate = aggregate_coverage(counts)

    rows = [
        [
            row["corridor"],
            _int(row["valid_pairs"]),
            _int(row["total_pairs"]),
            _num(row["coverage_pct"], 1) + " %",
        ]
        for row in aggregate.to_dicts()
    ]
    return _render(
        ["Corredor", "Pares con headway", "Pares evaluados", "Cobertura"],
        rows,
        aligns="lrrr",
    )


class FormulationError(ValueError):
    """The probe matrix cannot answer for the candidates being compared."""


# The probe's ids, in the order the table lists them, with the names Section
# III-A uses. The adopted one is marked here and not in the prose, so the table
# cannot disagree with the manuscript about which definition won.
FORMULATIONS = (
    ("A", "Puntos virtuales del eje"),
    ("B", "Distancia en metros"),
    ("C1", "Proyectado hacia adelante"),
    ("C2", "Cruce hacia atrás (adoptada)"),
)

# The probe reports one row per corridor; the paper compares definitions, so the
# two corridors are pivoted onto one row each.
PROBE_CORRIDORS = (2, 59)

PROBE_COLUMNS = ("autocorr_5min", "mi_bits", "pass_count_total")


def formulation_rows(
    frame: pl.DataFrame, formulations: tuple[str, ...]
) -> list[dict]:
    """Pivot the probe matrix to one row per candidate definition.

    Fails closed on an absent candidate, an absent corridor or a pass count the
    two corridors disagree on. Any of the three would let the table report a
    comparison the probe never made.
    """
    for column in ("formulation", "empresa", *PROBE_COLUMNS):
        if column not in frame.columns:
            raise FormulationError(f"probe matrix has no {column} column")

    rows: list[dict] = []
    for formulation in formulations:
        sub = frame.filter(pl.col("formulation") == formulation)
        if sub.is_empty():
            raise FormulationError(f"probe matrix has no rows for {formulation}")

        cells: dict[int, dict] = {}
        for corridor in PROBE_CORRIDORS:
            match = sub.filter(pl.col("empresa") == corridor)
            if match.height != 1:
                raise FormulationError(
                    f"{formulation} needs exactly one row for corridor {corridor}"
                )
            cells[corridor] = match.to_dicts()[0]

        passed = {cells[c]["pass_count_total"] for c in PROBE_CORRIDORS}
        if len(passed) != 1:
            raise FormulationError(
                f"{formulation} has pass_count_total disagreeing across corridors"
            )

        rows.append({
            "formulation": formulation,
            "autocorr_e2": cells[2]["autocorr_5min"],
            "autocorr_e59": cells[59]["autocorr_5min"],
            "mi_e2": cells[2]["mi_bits"],
            "mi_e59": cells[59]["mi_bits"],
            "passed": int(passed.pop()),
        })
    return rows


def tabla_5() -> str:
    """The four candidate headway definitions, and what separated them.

    The probe scored seven dimensions; two decided the discards and are the two
    Section III-A quotes. The rest stay in the CSV. The last column is the
    probe's own verdict count, kept so the reader can see that the metric
    definition and the adopted one were adjudicated as equals.
    """
    probe = _load("headway_formulations.csv")
    rows = formulation_rows(probe, tuple(key for key, _ in FORMULATIONS))
    labels = dict(FORMULATIONS)

    return _render(
        ["Formulación", "Autocorr. 5 min E2", "Autocorr. 5 min E59",
         "Info. mutua E2", "Info. mutua E59", "Dimensiones pasadas"],
        [
            [
                labels[row["formulation"]],
                _num(row["autocorr_e2"]),
                _num(row["autocorr_e59"]),
                _num(row["mi_e2"]),
                _num(row["mi_e59"]),
                f"{row['passed']} de 7",
            ]
            for row in rows
        ],
        aligns="lrrrrr",
    )


#: (display name, filename stem) of the three architectures the frozen
#: generation ran over one corpus. The order is the one the table prints.
ARCHITECTURES: tuple[tuple[str, str], ...] = (
    ("LSTM", "lstm"),
    ("SpatialConvLSTM", "spatial_conv_lstm"),
    ("SpatialTransformer", "spatial_transformer"),
)


class AblationError(RuntimeError):
    """The frozen results do not support the architecture comparison."""


def _frozen_name(stem: str, corridor: str, horizon: int) -> str:
    """E4 trained on its own kernels, so its results carry the corridor tag."""
    tag = "_E4" if corridor == "E4" else ""
    return f"{stem}{tag}_results_h{horizon}.csv"


def load_frozen_results(
    corridors: tuple[str, ...] = CORRIDORS,
    horizons: tuple[int, ...] = HORIZONS,
) -> pl.DataFrame:
    """Every frozen architecture result, in one long frame.

    E2 and E59 share a results file per horizon, so the same file is read for
    both and filtered by corridor afterwards.
    """
    seen: set[str] = set()
    frames: list[pl.DataFrame] = []
    for _, stem in ARCHITECTURES:
        for corridor in corridors:
            for horizon in horizons:
                name = _frozen_name(stem, corridor, horizon)
                if name in seen:
                    continue
                seen.add(name)
                frames.append(_load(name))
    return pl.concat(frames)


def ablation_rows(
    results: pl.DataFrame,
    corridors: tuple[str, ...] = CORRIDORS,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[list[str]]:
    """One row per corridor and horizon, one column per architecture.

    Reads the ``aggregate`` row and not the per-direction ones. A results file
    carries both, and averaging all three counts the corridor twice; falling
    back to a single direction would change the unit without saying so. Either
    absence raises instead.
    """
    pooled = results.filter(
        (pl.col("direction") == "aggregate") & (pl.col("metric") == "MAE")
    )

    rows: list[list[str]] = []
    for corridor in corridors:
        for horizon in horizons:
            maes: list[float] = []
            for name, _ in ARCHITECTURES:
                match = pooled.filter(
                    (pl.col("corridor") == corridor)
                    & (pl.col("horizon") == horizon)
                    & (pl.col("baseline") == name)
                )
                if match.height != 1:
                    raise AblationError(
                        f"{corridor} h{horizon} needs exactly one aggregate MAE "
                        f"for {name}, found {match.height}"
                    )
                maes.append(float(match["value"][0]))
            rows.append(
                [corridor, str(horizon)]
                + [_num(mae) for mae in maes]
                + [_num(max(maes) - min(maes))]
            )
    return rows


def tabla_6() -> str:
    """Does modelling the relation between neighbouring positions pay?

    The three architectures ran over one frozen corpus, so they are comparable
    to each other and to nothing else in this paper. The last column is the
    widest gap of the row, which is what carries the answer: the spatial
    component moves the scalar error by less than the rounding of a second.
    """
    rows = ablation_rows(load_frozen_results())
    return _render(
        ["Corredor", "h", *(name for name, _ in ARCHITECTURES), "Rango"],
        rows,
        aligns="lr" + "r" * (len(ARCHITECTURES) + 1),
    )


TABLES = {
    "tabla-1-deteccion-corte-trasplantado.md": tabla_1,
    "tabla-2-veredicto-sin-umbral.md": tabla_2,
    "tabla-3-robustez.md": tabla_3,
    "tabla-4-cobertura-headway.md": tabla_4,
    "tabla-5-formulaciones-headway.md": tabla_5,
    "tabla-6-ablacion-arquitectura.md": tabla_6,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in TABLES.items():
        path = OUT_DIR / filename
        path.write_text(builder() + "\n", encoding="utf-8")
        print(f"Tabla escrita en {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
