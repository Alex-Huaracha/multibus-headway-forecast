"""Historical average vs the learner, split by input-window volatility.

``build_ha_paired_audit`` establishes that the hourly average beats the LSTM in
exactly one of twelve cells: E2 at ten minutes, by 0.07 min. This script asks the
follow-up that decides how to read that crossing.

The average is blind to the input window — it answers "what is normal at this
hour" and nothing else. The learner's advantage over persistence is known to grow
with the dispersion of the window it observed (Figure 2). So if the E2 crossing is
a property of calm windows, where there is little for a learner to use, then the
learner still earns its place in the regime that matters operationally. If the
average wins in volatile windows too, it does not.

Thresholds are the frozen train+val terciles of the existing pipeline, reused
rather than recomputed, so this stratification is the same one the document
already reports.

Usage
-----
    uv run python -m src.build_ha_volatility
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_volatility import (  # noqa: E402
    CORRIDOR_IDS,
    TERCILE_NAMES,
    assign_regime,
    calibrate,
    corridor_max_N,
    dispersion_frame,
    prepare,
)
from src.build_ha_paired_audit import HA_KEY, RESID_DIR, RESIDUALS, _hourly_average  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "contiguous_ha_volatility.csv"

HORIZONS = (5, 10)  # the horizons where the crossing is in play
CORRIDORS = ("E2", "E4", "E59")


def cell_rows(corridor: str, horizon: int) -> list[dict]:
    df = prepare(CORRIDOR_IDS[corridor])
    max_N = corridor_max_N(df)
    thresholds = calibrate(df, corridor=corridor, horizon=horizon, max_N=max_N)
    test_std = dispersion_frame(
        df, corridor=corridor, split="test", horizon=horizon, max_N=max_N
    )

    residuals = (
        pl.read_csv(RESID_DIR / RESIDUALS[corridor].format(h=horizon), try_parse_dates=True)
        .filter((pl.col("corridor") == corridor) & (pl.col("split") == "test"))
    )

    joined = (
        residuals
        .join(test_std, on=["corridor", "direction", "start_ts", "pair_rank"], how="inner")
        .with_columns(pl.col("target_ts").dt.hour().alias("_hour"))
        .join(_hourly_average(corridor), on=HA_KEY, how="inner")
    )

    work, _std, codes = assign_regime(joined, thresholds)
    if work.height == 0:
        return []

    y = work.get_column("y_true").to_numpy()
    lstm = work.get_column("y_pred_model").to_numpy()
    ha = work.get_column("y_pred_ha").to_numpy()
    pers = work.get_column("y_pred_persist").to_numpy()

    rows: list[dict] = []
    for code, name in enumerate(TERCILE_NAMES):
        sel = codes == code
        n = int(sel.sum())
        if n == 0:
            continue
        mae_lstm = float(np.abs(y[sel] - lstm[sel]).mean())
        mae_ha = float(np.abs(y[sel] - ha[sel]).mean())
        mae_pers = float(np.abs(y[sel] - pers[sel]).mean())
        rows.append(
            {
                "corridor": corridor,
                "horizon": horizon,
                "tercile": name,
                "n": n,
                "mae_lstm": round(mae_lstm, 4),
                "mae_ha": round(mae_ha, 4),
                "mae_persist": round(mae_pers, 4),
                "delta_lstm_ha": round(mae_lstm - mae_ha, 4),
                "lstm_beats_ha": mae_lstm < mae_ha,
            }
        )
    return rows


def main() -> None:
    records: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            print(f"  {corridor} h={horizon} …")
            records.extend(cell_rows(corridor, horizon))

    table = pl.DataFrame(records)
    table.write_csv(OUT_CSV)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)
    print(f"\nescrito en {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
