"""Intervals for the ranking and operating-point verdicts, per corridor x horizon.

Section IV-E defines a verdict as "which of the two wins, by how much, and
whether the difference survives its test", and then declares that this work
emits verdicts on the MAE, on the detection quantities, on the MCC and on the
AUC. Two of those four had an instrument: Diebold-Mariano for the MAE and
Clopper-Pearson for the precision. The MCC and the AUC had none, so the headline
count — the LSTM ordering better than persistence in nine of nine combinations
at ten minutes — rested on point estimates alone, with margins as thin as 0.001
counted the same as margins of 0.05.

This builder supplies the missing instrument. For every cell it bounds three
differences, LSTM minus persistence:

``delta_auc``
    The threshold-free verdict. This is the one the headline count is made of.
``delta_mcc_fixed``
    The operating point of Equation (7), the observed threshold transplanted
    onto the predicted vector.
``delta_mcc_calibrated``
    The operating point refitted on origin ``r2`` and applied forward, which is
    the repair Section V-D reports. Only the published origin has one, because
    only it is scored against an earlier fit.

The interval comes from resampling SERVICE DAYS with replacement, not rows —
``src.evaluation.ranking_significance`` carries the argument for why. The
effective sample is 22 days, which is what makes a 0.001 margin worth bounding
in the first place.

The population is ``build_detection_calibrated.prepared``, reused rather than
rebuilt, so an interval here can never bound a number that differs from the one
Table 2 prints.

Output is ``docs/resultados/csv-multihorizon/detection_ranking_ci.csv``.

Usage
-----
    uv run python -m src.build_detection_ranking_ci
"""

from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_detection_calibrated import (  # noqa: E402
    CALIBRATION_ORIGIN,
    SCORING_ORIGIN,
    prepared,
)
from src.evaluation.ranking_significance import (  # noqa: E402
    RankingSignificanceError,
    clustered_bootstrap_delta,
    fast_auc,
)
from src.evaluation.vector_metrics import (  # noqa: E402
    best_threshold,
    matthews_corrcoef,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "detection_ranking_ci.csv"

ORIGINS = ("r1", "r2", "main")
N_BOOT = 2000
SEED = 42
LEVEL = 0.95


_auc = fast_auc


def _mcc(truth: np.ndarray, flag: np.ndarray) -> float:
    return float(matthews_corrcoef(truth, flag.astype(bool)))


def _bound(truth, a, b, day, statistic) -> dict[str, float | None]:
    """One bounded difference, or nulls when the cell is too thin to bound.

    A cell with a single class present has no AUC and no MCC to compare. That is
    an absent measurement, not a zero difference, so it is written as null.
    """
    try:
        result = clustered_bootstrap_delta(
            truth, a, b, day, statistic=statistic,
            n_boot=N_BOOT, seed=SEED, level=LEVEL,
        )
    except RankingSignificanceError:
        return {"delta": None, "ci_low": None, "ci_high": None, "survives": None}
    return {
        "delta": result.delta,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "survives": not result.crosses_zero,
    }


def build() -> pl.DataFrame:
    """Bound every ranking and operating-point difference the paper reports."""
    fit = prepared(CALIBRATION_ORIGIN)

    rows: list[dict] = []
    for origin in ORIGINS:
        score = prepared(origin)
        for corridor in CORRIDORS:
            for horizon in HORIZONS:
                where = (
                    (pl.col("corridor") == corridor)
                    & (pl.col("horizon") == horizon)
                )
                cell = score.filter(where)
                if cell.height == 0:
                    continue

                truth = cell.get_column("_bunch_y_true").to_numpy()
                day = cell.get_column("target_ts").dt.date().to_numpy()
                s_model = cell.get_column("_score_y_pred_model").to_numpy()
                s_persist = cell.get_column("_score_y_pred_persist").to_numpy()
                f_model = cell.get_column("_bunch_y_pred_model").to_numpy()
                f_persist = cell.get_column("_bunch_y_pred_persist").to_numpy()

                auc = _bound(truth, s_model, s_persist, day, _auc)
                mcc_fixed = _bound(truth, f_model, f_persist, day, _mcc)

                # The calibrated operating point exists only where an earlier
                # origin supplied it, which is the published origin alone.
                if origin == SCORING_ORIGIN:
                    fit_cell = fit.filter(where)
                    if fit_cell.height:
                        ft = fit_cell.get_column("_bunch_y_true").to_numpy()
                        cal_model = (
                            s_model >= best_threshold(
                                ft,
                                fit_cell.get_column("_score_y_pred_model").to_numpy(),
                                objective="mcc",
                            )
                        )
                        cal_persist = (
                            s_persist >= best_threshold(
                                ft,
                                fit_cell.get_column("_score_y_pred_persist").to_numpy(),
                                objective="mcc",
                            )
                        )
                        mcc_cal = _bound(truth, cal_model, cal_persist, day, _mcc)
                    else:
                        mcc_cal = {"delta": None, "ci_low": None,
                                   "ci_high": None, "survives": None}
                else:
                    mcc_cal = {"delta": None, "ci_low": None,
                               "ci_high": None, "survives": None}

                rows.append({
                    "origin": origin,
                    "corridor": corridor,
                    "horizon": int(horizon),
                    "n": int(cell.height),
                    "n_service_days": int(np.unique(day).size),
                    "base_rate": float(truth.mean()),
                    "delta_auc": auc["delta"],
                    "auc_ci_low": auc["ci_low"],
                    "auc_ci_high": auc["ci_high"],
                    "auc_survives": auc["survives"],
                    "delta_mcc_fixed": mcc_fixed["delta"],
                    "mcc_fixed_ci_low": mcc_fixed["ci_low"],
                    "mcc_fixed_ci_high": mcc_fixed["ci_high"],
                    "mcc_fixed_survives": mcc_fixed["survives"],
                    "delta_mcc_calibrated": mcc_cal["delta"],
                    "mcc_calibrated_ci_low": mcc_cal["ci_low"],
                    "mcc_calibrated_ci_high": mcc_cal["ci_high"],
                    "mcc_calibrated_survives": mcc_cal["survives"],
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "level": LEVEL,
                })

    if not rows:
        raise RankingSignificanceError("no cells to bound; download the residuals first")

    origin_order = {name: index for index, name in enumerate(ORIGINS)}
    corridor_order = {name: index for index, name in enumerate(CORRIDORS)}
    return (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("origin").replace_strict(origin_order).alias("_o"),
            pl.col("corridor").replace_strict(corridor_order).alias("_c"),
        ])
        .sort("_o", "_c", "horizon")
        .drop("_o", "_c")
    )


def render(frame: pl.DataFrame) -> str:
    """The CSV text, so determinism can be asserted without touching the disk."""
    return frame.write_csv()


def main() -> None:
    frame = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text(render(frame), encoding="utf-8", newline="")
    print(f"Intervalos escritos en {OUT_CSV.relative_to(REPO_ROOT)}")
    # Plain ASCII, not the polars repr: its box-drawing characters cannot be
    # encoded by the cp1252 console this project is run from.
    print(f"{'origen':<6} {'corr':<4} {'h':>3}  {'dAUC':>7} "
          f"{'[IC 95%]':^18} {'sobrevive':>10}")
    for row in frame.iter_rows(named=True):
        if row["delta_auc"] is None:
            print(f"{row['origin']:<6} {row['corridor']:<4} {row['horizon']:>3}"
                  f"  {'sin medida':>7}")
            continue
        band = f"[{row['auc_ci_low']:+.4f}, {row['auc_ci_high']:+.4f}]"
        print(f"{row['origin']:<6} {row['corridor']:<4} {row['horizon']:>3}  "
              f"{row['delta_auc']:+7.4f} {band:^18} "
              f"{'si' if row['auc_survives'] else 'NO':>10}")


if __name__ == "__main__":
    main()
