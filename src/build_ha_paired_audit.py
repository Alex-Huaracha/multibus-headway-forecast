"""Historical-average baseline, scored on the paired population.

The hourly historical average (``B4_HA`` in ``src/baselines/statistical.py``) has
only ever been reported in the aggregate framing, where every model scores its own
rows. That framing favours the learner by 0.28-0.53 min, which is wider than the
LSTM's measured margin over the average in E2 — so the aggregate number cannot
settle whether the network beats an almanac there.

This script closes that. It recomputes the average under the same split and
winsorization contract, then scores it on **exactly the rows the LSTM and
persistence were scored on**, taken from the contiguous per-sample residuals. No
retraining and no GPU: the average is a group mean, and the paired rows already
exist on disk.

Why the comparison matters at all: Section V.4 measures that the forecast is
nearly flat (CV 0.16 against 0.79 observed). Having measured that, the obvious
question is whether the flat forecast is just the historical mean. This answers it.

Usage
-----
    uv run python -m src.build_ha_paired_audit
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from src.evaluation.splits import (  # noqa: E402
    MAIN_FOLD,
    split_temporal,
    winsorize_train_p99,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
RESID_DIR = REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon" / "21-lstm-contiguous"
OUT_CSV = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "contiguous_ha_paired.csv"

HORIZONS = (1, 3, 5, 10)

# The residual exports split E4 into its own file; E2 and E59 share one.
RESIDUALS = {
    "E2": "lstm_contig_residuals_h{h}.csv",
    "E59": "lstm_contig_residuals_h{h}.csv",
    "E4": "lstm_contig_E4_residuals_h{h}.csv",
}
# Grouping key of B4_HA. empresaid is constant inside a per-corridor parquet, so
# it drops out of the key here; direction and pair_rank are what remain of
# _SLOT_COLS, plus the hour of day.
HA_KEY = ["direction", "pair_rank", "_hour"]


def _hourly_average(corridor: str) -> pl.DataFrame:
    """Train-only mean of delta_t_min by (direction, pair_rank, hour of day)."""
    headways = pl.read_parquet(DATA_DIR / f"headways_{corridor}.parquet")
    headways = split_temporal(headways, MAIN_FOLD)
    headways, _threshold = winsorize_train_p99(headways)

    return (
        headways
        .with_columns(pl.col("t").dt.hour().alias("_hour"))
        .filter((pl.col("split") == "train") & pl.col("delta_t_min").is_not_null())
        .group_by(HA_KEY)
        .agg(pl.col("delta_t_min").mean().alias("y_pred_ha"))
    )


def _paired_rows(corridor: str, horizon: int) -> pl.DataFrame:
    path = RESID_DIR / RESIDUALS[corridor].format(h=horizon)
    return (
        pl.read_csv(path, try_parse_dates=True)
        .filter((pl.col("corridor") == corridor) & (pl.col("split") == "test"))
        .with_columns(pl.col("target_ts").dt.hour().alias("_hour"))
    )


def audit_cell(corridor: str, horizon: int) -> dict:
    rows = _paired_rows(corridor, horizon)
    n_total = rows.height

    joined = rows.join(_hourly_average(corridor), on=HA_KEY, how="left")

    # Score all three on identical rows. An hour never seen in train yields a null
    # average; those rows leave the comparison for every model, not just for HA.
    scored = joined.filter(
        pl.col("y_true").is_not_null()
        & pl.col("y_pred_model").is_not_null()
        & pl.col("y_pred_persist").is_not_null()
        & pl.col("y_pred_ha").is_not_null()
    )

    mae = lambda col: float(  # noqa: E731
        (scored.get_column("y_true") - scored.get_column(col)).abs().mean()
    )
    mae_lstm, mae_pers, mae_ha = mae("y_pred_model"), mae("y_pred_persist"), mae("y_pred_ha")

    return {
        "corridor": corridor,
        "horizon": horizon,
        "n_paired": scored.height,
        "coverage_pct": round(100.0 * scored.height / n_total, 3) if n_total else 0.0,
        "mae_lstm": round(mae_lstm, 4),
        "mae_persist": round(mae_pers, 4),
        "mae_ha": round(mae_ha, 4),
        "delta_lstm_ha": round(mae_lstm - mae_ha, 4),
        "delta_lstm_persist": round(mae_lstm - mae_pers, 4),
        "delta_ha_persist": round(mae_ha - mae_pers, 4),
        "lstm_beats_ha": mae_lstm < mae_ha,
    }


def main() -> None:
    records = [
        audit_cell(corridor, horizon)
        for corridor in ("E2", "E4", "E59")
        for horizon in HORIZONS
    ]
    table = pl.DataFrame(records).sort(["corridor", "horizon"])
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)
    print(f"\nescrito en {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
