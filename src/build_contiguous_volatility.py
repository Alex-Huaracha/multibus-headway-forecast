"""Ex-ante volatility stratification on the retrained pipeline.

Rebuilds the volatility story on the contiguous-pipeline residuals, with two
corrections relative to ``build_exante_volatility.py`` (which stays as the record
of the frozen 11/12/13 comparison):

1. **The stratifier is ex-ante, so the subgroup tests are legitimate.** Windows
   are binned by the dispersion of their own input window
   (``evaluation/exante_volatility``), not by the realized change, which is
   persistence's error. The p-values that had to be deleted from the retrospective
   table (pending #2) are reportable here.
2. **The join is on the full key, not on row position.** The old script
   reconstructed windows locally and matched the residual CSVs positionally,
   gated by a tolerance check. The new exports carry
   ``(corridor, direction, horizon, split, start_ts, target_ts, pair_rank)``, so
   the merge is an inner join that cannot silently misalign.

Thresholds are frozen on **train+val** and applied to test, per corridor and
horizon — never calibrated on the split being reported (CLAUDE.md ex-ante
contract).

Verdicts use the clustered variance from ``significance_clustered``: within a
tercile the effective sample size is still the number of service days, not the
number of rows.

Outputs
-------
``docs/resultados/csv-multihorizon/contiguous_exante_volatility.csv``
    One row per corridor x horizon x tercile: mass, per-model MAE, and the
    paired LSTM-vs-persistence and LSTM-vs-XGBoost verdicts inside the bin.

Usage
-----
    uv run python -m src.build_contiguous_volatility
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_significance import (  # noqa: E402
    CORRIDORS as SIGNIFICANCE_CORRIDORS,
    HORIZONS,
    XGB_CSV,
    load_lstm,
)
from src.build_sample_index import SPLIT_BOUNDS, T_IN, load_corridor  # noqa: E402
from src.data.contiguous_dataset import materialize_arrays  # noqa: E402
from src.data.sample_index import make_sample_index  # noqa: E402
from src.data.windowing import compute_max_N  # noqa: E402
from src.evaluation.exante_terciles import (  # noqa: E402
    TercileThresholds,
    compute_frozen_thresholds,
)
from src.evaluation.exante_volatility import window_dispersion  # noqa: E402
from src.evaluation.residual_export import RESIDUAL_KEY_COLUMNS  # noqa: E402
from src.evaluation.significance_clustered import dm_clustered  # noqa: E402
from src.evaluation.splits import split_temporal, winsorize_train_p99  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "contiguous_exante_volatility.csv"
)

# (empresaid, corridor label), in the order the significance package reports.
CORRIDOR_IDS: dict[str, int] = {"E2": 2, "E59": 59, "E4": 4}
TERCILE_NAMES = ("low", "mid", "high")

# Raw winsorized minutes, not the z-scored column: the thresholds are meant to be
# read as "windows that wobbled more than X minutes".
VALUE_COL = "delta_t_min"

# Every paired residual must find the window it came from. The floor is not 100%
# only because `max_N` truncation is recomputed here and can differ by a cell or
# two at the tail (AC-MAXN-2).
MIN_JOIN_COVERAGE_PCT = 99.0


def prepare(empresaid: int) -> pl.DataFrame:
    """Split-tagged, train-p99-winsorized frame — the notebooks' preprocessing.

    Z-scoring is deliberately skipped: the stratifier is reported in minutes and
    the normalization stats would only cancel out of a standard deviation up to
    a per-direction scale factor, which would make the frozen thresholds
    incomparable across directions.
    """
    df = split_temporal(load_corridor(empresaid))
    df, _threshold = winsorize_train_p99(df)
    return df


def dispersion_frame(
    df: pl.DataFrame, *, corridor: str, split: str, horizon: int, max_N: int
) -> pl.DataFrame:
    """Ex-ante dispersion per ``(direction, start_ts, pair_rank)`` for one split.

    Returns a frame joinable to the residual exports on the subset of the
    canonical key that identifies a window cell.
    """
    empty = pl.DataFrame(
        schema={
            "corridor": pl.Utf8,
            "direction": pl.Int64,
            "start_ts": pl.Datetime("us"),
            "pair_rank": pl.Int64,
            "exante_std": pl.Float64,
        }
    )

    lo, hi = SPLIT_BOUNDS[split]
    day = pl.col("t").dt.date()
    part = df.filter((day >= lo) & (day <= hi))
    index = make_sample_index(part, horizon=horizon, T_in=T_IN)
    if index.height == 0:
        return empty

    frames: list[pl.DataFrame] = []
    for direction in (-1, 1):
        sub_idx = index.filter(pl.col("direction") == direction)
        if sub_idx.height == 0:
            continue
        arrays = materialize_arrays(
            part.filter(pl.col("direction") == direction),
            sub_idx,
            max_N=max_N,
            T_in=T_IN,
            horizon=horizon,
            value_col=VALUE_COL,
            context_cols=(),
        )
        std = window_dispersion(arrays["input"], arrays["input_mask"])
        n_rows = sub_idx.height
        frames.append(
            pl.DataFrame(
                {
                    "corridor": np.full(n_rows * max_N, corridor),
                    # Integer, matching the residual CSVs: the exports write the
                    # signed label ("-1"/"+1") but polars reads both back as i64.
                    "direction": np.full(
                        n_rows * max_N, direction, dtype=np.int64
                    ),
                    "start_ts": np.repeat(
                        sub_idx.get_column("start_ts").to_numpy(), max_N
                    ),
                    "pair_rank": np.tile(np.arange(max_N, dtype=np.int64), n_rows),
                    "exante_std": std.ravel(),
                }
            )
        )

    return pl.concat(frames) if frames else empty


def calibrate(
    df: pl.DataFrame, *, corridor: str, horizon: int, max_N: int
) -> TercileThresholds:
    """Freeze p33/p66 on train+val only — never on the split being reported."""
    calib = pl.concat(
        [
            dispersion_frame(
                df, corridor=corridor, split=split, horizon=horizon, max_N=max_N
            )
            for split in ("train", "val")
        ]
    )
    return compute_frozen_thresholds(calib.get_column("exante_std").to_numpy())


def tercile_rows(
    joined: pl.DataFrame,
    thresholds: TercileThresholds,
    *,
    corridor: str,
    horizon: int,
) -> list[dict]:
    """Per-tercile mass, errors and paired verdicts for one corridor x horizon."""
    work, std, codes = assign_regime(joined, thresholds)
    if work.height == 0:
        return []

    y_true = work.get_column("y_true").to_numpy()
    dl = work.get_column("y_pred_model").to_numpy()
    xg = work.get_column("y_pred_xgb").to_numpy()
    pe = work.get_column("y_pred_persist").to_numpy()
    day = work.get_column("target_ts").dt.date().to_numpy()

    rows: list[dict] = []
    for code, name in enumerate(TERCILE_NAMES):
        sel = codes == code
        n = int(sel.sum())
        if n == 0:
            continue
        ae_dl = np.abs(y_true[sel] - dl[sel])
        ae_xgb = np.abs(y_true[sel] - xg[sel])
        ae_pe = np.abs(y_true[sel] - pe[sel])

        vs_persist = dm_clustered(ae_dl - ae_pe, day[sel], horizon=horizon)
        vs_xgb = dm_clustered(ae_dl - ae_xgb, day[sel], horizon=horizon)

        rows.append(
            {
                "corridor": corridor,
                "horizon": horizon,
                "tercile": name,
                "tercile_order": code,
                "n": n,
                "share": n / work.height,
                "mean_exante_std": float(std[sel].mean()),
                "mae_persist": float(ae_pe.mean()),
                "mae_lstm": float(ae_dl.mean()),
                "mae_xgb": float(ae_xgb.mean()),
                "delta_lstm_persist": float(ae_dl.mean() - ae_pe.mean()),
                "delta_lstm_xgb": float(ae_dl.mean() - ae_xgb.mean()),
                "dm_p_lstm_persist": vs_persist.p_value,
                "dm_p_lstm_xgb": vs_xgb.p_value,
                "n_service_days": vs_persist.n_clusters,
                "p33_threshold": thresholds.p33,
                "p66_threshold": thresholds.p66,
                "calib_split": thresholds.calib_split,
                "calib_n": thresholds.calib_n,
            }
        )
    return rows


def corridor_max_N(df: pl.DataFrame) -> int:
    """Global train-p99 vector width, as ``train.py`` dimensions the network."""
    return max(
        compute_max_N(df.filter(pl.col("split") == "train"), quantile=0.99).values()
    )


def paired_cell(
    lstm: pl.DataFrame,
    xgb: pl.DataFrame,
    df: pl.DataFrame,
    *,
    corridor: str,
    horizon: int,
    max_N: int,
    verbose: bool = True,
) -> tuple[pl.DataFrame, TercileThresholds] | None:
    """Three-way paired residuals for one cell, carrying the ex-ante regime.

    Shared by this builder and the router: both need the same population, the
    same frozen thresholds and the same fail-closed join, and having two copies
    of that is how the two analyses would drift apart.

    Returns ``None`` when either model has no rows for the cell.
    """
    where = (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
    left = lstm.filter(where)
    right = xgb.filter(where)
    if left.height == 0 or right.height == 0:
        return None

    thresholds = calibrate(df, corridor=corridor, horizon=horizon, max_N=max_N)
    test_std = dispersion_frame(
        df, corridor=corridor, split="test", horizon=horizon, max_N=max_N
    )

    joined = left.join(
        right.select(RESIDUAL_KEY_COLUMNS + ["y_pred_model"]).rename(
            {"y_pred_model": "y_pred_xgb"}
        ),
        on=RESIDUAL_KEY_COLUMNS,
        how="inner",
    ).join(
        test_std,
        on=["corridor", "direction", "start_ts", "pair_rank"],
        how="inner",
    )

    # Fail closed on a bad join. The dispersion frame is rebuilt from the same
    # parquet and the same sample index the kernels consumed, so every paired
    # residual must find its window. A silent drop here would re-stratify a
    # different population than the one being reported — the exact class of
    # defect this pipeline was rebuilt to prevent.
    paired = left.join(
        right.select(RESIDUAL_KEY_COLUMNS), on=RESIDUAL_KEY_COLUMNS, how="inner"
    ).height
    coverage = 100.0 * joined.height / paired if paired else 0.0
    if coverage < MIN_JOIN_COVERAGE_PCT:
        raise ValueError(
            f"{corridor} h={horizon}: ex-ante dispersion covers only "
            f"{coverage:.4f}% of the {paired:,} paired residuals "
            f"(floor {MIN_JOIN_COVERAGE_PCT}%)"
        )
    if verbose:
        print(
            f"  {corridor} h={horizon}: {joined.height:,} rows, "
            f"join coverage {coverage:.4f}%"
        )
    return joined, thresholds


def assign_regime(joined: pl.DataFrame, thresholds: TercileThresholds):
    """Finite-dispersion subset plus its frozen tercile code, in row order.

    Cells whose window held fewer than two observations have no dispersion and
    are dropped from the analysis rather than folded into a bin.
    """
    std = joined.get_column("exante_std").to_numpy()
    finite = np.isfinite(std)
    work = joined.filter(pl.Series(finite))
    std = std[finite]
    codes = np.where(std <= thresholds.p33, 0, np.where(std <= thresholds.p66, 1, 2))
    return work, std, codes


def build() -> pl.DataFrame:
    lstm = load_lstm()
    xgb = pl.read_csv(XGB_CSV, try_parse_dates=True)

    rows: list[dict] = []
    for corridor in SIGNIFICANCE_CORRIDORS:
        df = prepare(CORRIDOR_IDS[corridor])
        max_N = corridor_max_N(df)

        for horizon in HORIZONS:
            cell = paired_cell(
                lstm, xgb, df, corridor=corridor, horizon=horizon, max_N=max_N
            )
            if cell is None:
                continue
            joined, thresholds = cell
            rows.extend(
                tercile_rows(joined, thresholds, corridor=corridor, horizon=horizon)
            )

    return pl.DataFrame(rows).sort(["corridor", "horizon", "tercile_order"])


def main() -> None:
    table = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=220):
        print(
            table.select(
                ["corridor", "horizon", "tercile", "n", "share", "mean_exante_std",
                 "mae_persist", "mae_lstm", "mae_xgb",
                 "delta_lstm_persist", "dm_p_lstm_persist",
                 "delta_lstm_xgb", "dm_p_lstm_xgb"]
            )
        )
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
