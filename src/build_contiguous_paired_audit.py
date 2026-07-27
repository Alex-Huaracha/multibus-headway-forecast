"""Canonical paired audit for the retrained pipeline.

Audit §2.1: «el LSTM le gana al XGBoost en las 8 celdas de E2 y E59 (p = 0.004)»
compared MAEs computed over **different sample sets** — XGBoost over every test
row with a non-null prediction, the LSTM over the window population. The framing
bias that gap introduced was measured at 0.28–0.53 min, larger than 7 of the 8
margins claimed.

``evaluation/paired_audit.py`` reconciled that after the fact, by joining the
paired residuals back to the reported aggregates. It stays as the record of the
frozen 11/12/13 comparison.

This builder answers the same question for the retrained pipeline, and it is
deliberately not a reconciliation. Contract C1 makes the three models consume one
population by construction, so the audit's job is to **measure that the framing
bias is now zero** rather than to correct for it. For every model it computes the
headline metrics twice — over that model's own exported rows, and over the
three-way paired intersection — and reports the difference. A non-zero
``framing_delta`` means C1 is not holding and the build fails closed.

Direction is reported alongside the aggregate. E4's population is ~60/40 across
directions, which is a property of the source parquet rather than an inference
defect, but a per-direction split is what makes that visible instead of assumed.

Why retention is not exactly 100%
---------------------------------
The two exports truncate the vector at different widths. ``train.py`` dimensions
one network per corridor, so notebook 21 feeds it the **global** ``max_N`` —
``max`` over both directions — while ``baselines/contiguous_features`` takes
``max_N_by_direction`` and stops at each direction's own train-p99. Where one
direction runs a smaller fleet the LSTM therefore predicts a few tail slots the
XGBoost baseline never emits: in E4 dir ``+1`` the LSTM reaches ``pair_rank`` 10
against XGBoost's 7, which costs 51 of 99 880 rows (0.05%).

Those rows are outside the intersection and outside every reported verdict. The
audit keeps them visible through ``max_pair_rank_*`` rather than rounding the
retention up, and the framing delta confirms independently that dropping them
moves no metric — 0.00067 min in that cell.


Outputs
-------
``docs/resultados/csv-multihorizon/contiguous_paired_audit.csv``
    One row per corridor x horizon x direction (``aggregate``, ``-1``, ``+1``).

Usage
-----
    uv run python -m src.build_contiguous_paired_audit
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_significance import (  # noqa: E402
    CORRIDORS,
    HORIZONS,
    XGB_CSV,
    load_lstm,
)
from src.evaluation.residual_export import RESIDUAL_KEY_COLUMNS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "contiguous_paired_audit.csv"
)

# Minutes. The framing bias the audit measured on the old pipeline was 0.28–0.53;
# anything above this floor means the models are again scoring different
# populations and no "A beats B" claim built on the table would be attributable.
FRAMING_TOL_MIN = 0.01

DIRECTIONS: tuple[int | None, ...] = (None, -1, 1)


def _errors(frame: pl.DataFrame, pred_col: str) -> tuple[float, float]:
    """(MAE, RMSE) of one prediction column over ``frame``."""
    err = frame.get_column("y_true").to_numpy() - frame.get_column(pred_col).to_numpy()
    return float(np.abs(err).mean()), float(np.sqrt((err**2).mean()))


def audit_cell(
    lstm: pl.DataFrame,
    xgb: pl.DataFrame,
    paired: pl.DataFrame,
    *,
    corridor: str,
    horizon: int,
    direction: int | None,
) -> dict:
    """One row: each model scored on its own rows and on the shared population."""
    if direction is not None:
        where = pl.col("direction") == direction
        lstm, xgb, paired = lstm.filter(where), xgb.filter(where), paired.filter(where)

    mae_lstm_own, _ = _errors(lstm, "y_pred_model")
    mae_xgb_own, _ = _errors(xgb, "y_pred_model")

    mae_lstm, rmse_lstm = _errors(paired, "y_pred_model")
    mae_xgb, rmse_xgb = _errors(paired, "y_pred_xgb")
    mae_persist, rmse_persist = _errors(paired, "y_pred_persist")

    return {
        "corridor": corridor,
        "horizon": horizon,
        "direction": "aggregate" if direction is None else f"{direction:+d}",
        "n_lstm_own": lstm.height,
        "n_xgb_own": xgb.height,
        "n_paired": paired.height,
        "retained_lstm_pct": round(100.0 * paired.height / lstm.height, 6),
        "retained_xgb_pct": round(100.0 * paired.height / xgb.height, 6),
        # The vector width each export actually reached. When these differ the
        # gap explains the retention exactly — see the module docstring.
        "max_pair_rank_lstm": int(lstm.get_column("pair_rank").max()),
        "max_pair_rank_xgb": int(xgb.get_column("pair_rank").max()),
        "mae_lstm_own": mae_lstm_own,
        "mae_lstm_paired": mae_lstm,
        "framing_delta_lstm": mae_lstm - mae_lstm_own,
        "mae_xgb_own": mae_xgb_own,
        "mae_xgb_paired": mae_xgb,
        "framing_delta_xgb": mae_xgb - mae_xgb_own,
        "mae_persist_paired": mae_persist,
        "rmse_lstm_paired": rmse_lstm,
        "rmse_xgb_paired": rmse_xgb,
        "rmse_persist_paired": rmse_persist,
        "delta_lstm_persist": mae_lstm - mae_persist,
        "delta_lstm_xgb": mae_lstm - mae_xgb,
        "delta_xgb_persist": mae_xgb - mae_persist,
    }


def build() -> pl.DataFrame:
    lstm_all = load_lstm()
    xgb_all = pl.read_csv(XGB_CSV, try_parse_dates=True)

    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            where = (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            lstm = lstm_all.filter(where)
            xgb = xgb_all.filter(where)
            if lstm.height == 0 or xgb.height == 0:
                continue

            paired = lstm.join(
                xgb.select(RESIDUAL_KEY_COLUMNS + ["y_pred_model"]).rename(
                    {"y_pred_model": "y_pred_xgb"}
                ),
                on=RESIDUAL_KEY_COLUMNS,
                how="inner",
            )
            for direction in DIRECTIONS:
                rows.append(
                    audit_cell(
                        lstm, xgb, paired,
                        corridor=corridor, horizon=horizon, direction=direction,
                    )
                )

    return pl.DataFrame(rows).sort(["corridor", "horizon", "direction"])


def assert_no_framing_bias(table: pl.DataFrame) -> None:
    """Fail closed when a model scores differently on its own rows than on the shared ones.

    This is the assertion the old pipeline could not make. Under contract C1 the
    three models predict the same ``(corridor, direction, horizon, split,
    start_ts, target_ts, pair_rank)`` cells, so restricting to the intersection
    must be a no-op on every metric.
    """
    offenders = table.filter(
        (pl.col("framing_delta_lstm").abs() > FRAMING_TOL_MIN)
        | (pl.col("framing_delta_xgb").abs() > FRAMING_TOL_MIN)
    )
    if offenders.height > 0:
        raise ValueError(
            "shared-population contract C1 violated: "
            f"{offenders.height} cells shift by more than {FRAMING_TOL_MIN} min "
            "when restricted to the paired intersection:\n"
            f"{offenders.select(['corridor', 'horizon', 'direction', 'framing_delta_lstm', 'framing_delta_xgb'])}"
        )


def main() -> None:
    table = build()
    assert_no_framing_bias(table)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    aggregate = table.filter(pl.col("direction") == "aggregate")
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200):
        print(
            aggregate.select(
                ["corridor", "horizon", "n_paired", "retained_lstm_pct",
                 "retained_xgb_pct", "framing_delta_lstm", "framing_delta_xgb",
                 "mae_persist_paired", "mae_lstm_paired", "mae_xgb_paired",
                 "delta_lstm_persist", "delta_lstm_xgb"]
            )
        )
        print("\nPer direction (paired population only):")
        print(
            table.filter(pl.col("direction") != "aggregate").select(
                ["corridor", "horizon", "direction", "n_paired",
                 "mae_persist_paired", "mae_lstm_paired", "mae_xgb_paired",
                 "delta_lstm_persist"]
            )
        )

    worst = float(
        max(
            table.get_column("framing_delta_lstm").abs().max(),
            table.get_column("framing_delta_xgb").abs().max(),
        )
    )
    print(
        f"\nFraming bias: max |delta| = {worst:.6g} min over {table.height} rows "
        f"(tolerance {FRAMING_TOL_MIN})"
    )
    print(f"Wrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
