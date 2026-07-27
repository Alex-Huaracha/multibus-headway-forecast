"""Paired DL-vs-XGBoost comparison over the DL model's own sample population.

Why this module exists
----------------------
``build_xgb_vs_lstm_signtest`` used to compare an LSTM MAE and an XGBoost MAE
that were computed over DIFFERENT populations:

* ``baselines_results_multih.csv`` aggregates over EVERY test row with a
  non-null prediction (one row per ``(direction, t, pair_rank)``);
* the DL metrics aggregate over the DL WINDOW population — cold-start rows are
  dropped and every target is replicated once per anchoring window slot
  (roughly 4.5x).

The project's own measured aggregate-vs-paired framing bias for persistence is
0.28-0.53 min, larger than most of the claimed XGB margins, so no "A beats B"
verdict could be defended from those two numbers. This module removes the
framing difference: NB20 (``src.baselines.paired_export``) exports per-sample
XGBoost TEST predictions carrying the full unique key of the headways frame
``(direction, t, pair_rank)``, so XGBoost can be RE-SCORED over exactly the rows
the DL model was scored on.

The two populations, and why both are emitted
---------------------------------------------
``multiplicity_matched``
    One XGB row per LSTM row (LEFT JOIN from the LSTM population). Each distinct
    target is counted as many times as the DL export counted it. This is the ONLY
    construction whose MAE is literally comparable to the reported LSTM MAE, so
    it is the population the headline deltas and the cell-level sign test use.
    It is NOT a valid basis for a per-sample significance test: the replicas of
    one target are not independent observations.

``distinct_target``
    One row per ``(direction, t, pair_rank)`` — the first replica in temporal
    order. Each test target appears exactly once, which is the sound basis for a
    paired Diebold-Mariano / Wilcoxon test. Its MAE is NOT the reported LSTM MAE
    (a different subset of window slots), so it must not be quoted as such.

Sign convention
---------------
Every delta in this module is ``XGB - DL``, so POSITIVE means the DL model has
the lower loss (the DL model wins). This matches the sign test's
"does the LSTM beat the leveled XGBoost?" framing.

Gates (all fail closed — nothing is emitted for a cell that fails)
------------------------------------------------------------------
1. reconstructed population row count == residual CSV row count, and
   ``max|Delta y_true|`` / ``max|Delta y_pred_persist|`` below
   ``build_exante_volatility.ALIGN_TOL`` — delegated to ``verify_alignment``;
2. every LSTM row finds its XGB counterpart on ``(direction, t, pair_rank)``
   (100% join coverage);
3. the XGB export key is unique within a ``(corridor, horizon)`` cell.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.evaluation.significance import (
    diebold_mariano,
    loss_differential,
    wilcoxon_signed_rank,
)

# Schema of the NB20 per-sample XGBoost TEST export.
XGB_EXPORT_COLUMNS = [
    "corridor",
    "empresaid",
    "direction",
    "horizon",
    "t",
    "pair_rank",
    "y_true",
    "y_pred_xgb",
    "y_pred_persist",
]

# The unique key of the headways frame within one (corridor, horizon) cell.
JOIN_KEYS = ["direction", "t", "pair_rank"]

# Helper column stamped on the DL population in export order, so the choice of
# "first replica" in distinct_population is a total order rather than a tie-break.
REPLICA_INDEX = "_dl_row_index"

CORRIDOR_EMPRESAID = {"E2": 2, "E59": 59, "E4": 4}
HORIZONS = (1, 3, 5, 10)

# Wilcoxon/DM alternative for the directional claim. H1: median(loss_xgb -
# loss_dl) > 0, i.e. the DL model has the lower loss. Recorded in the emitted CSV
# so no reader can mistake a two-sided p-value for support of a signed claim.
DL_BETTER_ALTERNATIVE = "greater"

PAIRED_METRIC_COLUMNS = [
    "corridor",
    "horizon",
    "n_dl_rows",
    "n_xgb_matched",
    "coverage_pct",
    "n_distinct_targets",
    "align_max_abs_diff_target",
    "align_max_abs_diff_persist",
    "align_tol",
    "mae_persist_matched",
    "mae_dl_matched",
    "mae_xgb_matched",
    "delta_xgb_minus_dl_matched",
    "dl_better_matched",
    "mae_dl_distinct",
    "mae_xgb_distinct",
    "delta_xgb_minus_dl_distinct",
    "dl_better_distinct",
]

AUDIT_COLUMNS = [
    "corridor",
    "horizon",
    "metric",
    "n_dl_rows",
    "restricted_xgb",
    "reported_xgb",
    "abs_diff",
    "restricted_dl",
    "reported_dl",
    "restricted_delta",
    "reported_delta",
    "restricted_dl_better",
    "reported_dl_better",
    "sign_flip",
]

SIGNIFICANCE_COLUMNS = [
    "corridor",
    "horizon",
    "population",
    "metric",
    "n",
    "delta_loss",
    "delta_mae",
    "median_delta",
    "win_rate",
    "dm_stat",
    "dm_lag",
    "dm_p_two_sided",
    "dm_p_one_sided",
    "wilcoxon_alternative",
    "wilcoxon_p_one_sided",
    "wilcoxon_p_two_sided",
    "dl_better",
]


# ---------------------------------------------------------------------------
# Input discovery and validation (fail fast, before any expensive work)
# ---------------------------------------------------------------------------

def residual_csv_path(resid_dir: str | Path, corridor: str, horizon: int) -> Path:
    """Path of the LSTM per-sample residual CSV holding ``corridor`` at ``horizon``.

    E2 and E59 share one export (NB11); E4 has its own (NB17) downloaded into the
    same ``11-lstm/h{H}`` directory.
    """
    if corridor not in CORRIDOR_EMPRESAID:
        raise ValueError(f"residual_csv_path: unknown corridor {corridor!r}")
    stem = "lstm_E4_residuals" if corridor == "E4" else "lstm_residuals"
    return Path(resid_dir) / "11-lstm" / f"h{horizon}" / f"{stem}_h{horizon}.csv"


def xgb_export_path(resid_dir: str | Path) -> Path:
    """Path of the NB20 per-sample XGBoost TEST export inside the residual store."""
    return Path(resid_dir) / "20-xgb-paired" / "xgb_paired_persample_test.csv"


def validate_inputs(
    resid_dir: str | Path,
    corridors: tuple[str, ...] = ("E2", "E59", "E4"),
    horizons: tuple[int, ...] = HORIZONS,
) -> None:
    """Fail fast when any required per-sample input is missing (PKR1 contract).

    Raises
    ------
    ValueError
        Naming every missing residual CSV / XGB export, before the builder does
        any parquet loading or materialization work.
    """
    missing: list[str] = []
    export = xgb_export_path(resid_dir)
    if not export.is_file():
        missing.append(str(export))
    for corridor in corridors:
        for horizon in horizons:
            path = residual_csv_path(resid_dir, corridor, horizon)
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise ValueError(
            "xgb_paired: no per-sample inputs found — missing "
            f"{len(missing)} required file(s): {missing}"
        )


def load_xgb_export(path: str | Path) -> pl.DataFrame:
    """Read and validate the NB20 per-sample XGBoost export.

    Raises
    ------
    ValueError
        If the file is absent, empty, or lacks a required column.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"load_xgb_export: no XGB paired export found in {path}")
    frame = pl.read_csv(path, try_parse_dates=True)
    missing = [c for c in XGB_EXPORT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"load_xgb_export: {path.name} has wrong schema — missing {missing}"
        )
    if frame.height == 0:
        raise ValueError(f"load_xgb_export: {path.name} is empty")
    return frame.select(XGB_EXPORT_COLUMNS)


def xgb_cell(export: pl.DataFrame, corridor: str, horizon: int) -> pl.DataFrame:
    """Join-ready XGB slice for one cell: ``JOIN_KEYS`` plus ``y_pred_xgb``.

    Raises
    ------
    ValueError
        If the cell is absent, or if ``JOIN_KEYS`` is not unique within it (the
        join would then silently fan out and corrupt the multiplicity match).
    """
    cell = export.filter(
        (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
    )
    if cell.height == 0:
        raise ValueError(f"xgb_cell: no XGB rows for corridor={corridor} h={horizon}")
    duplicates = cell.group_by(JOIN_KEYS).len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError(
            f"xgb_cell: {corridor} h={horizon} has {duplicates.height} duplicated "
            f"{JOIN_KEYS} keys — the export is not per-sample unique"
        )
    return cell.select(
        [
            pl.col("direction"),
            pl.col("t"),
            pl.col("pair_rank").cast(pl.Int64),
            pl.col("y_pred_xgb"),
        ]
    )


# ---------------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------------

def direction_labels(directions: np.ndarray) -> np.ndarray:
    """Map integer directions to the ``"-1"`` / ``"+1"`` labels the exports use."""
    return np.where(np.asarray(directions) < 0, "-1", "+1")


def dl_population(
    residuals: pl.DataFrame,
    timestamps: np.ndarray,
    slots: np.ndarray,
    directions: np.ndarray,
) -> pl.DataFrame:
    """Attach the reconstructed ``(direction, t, pair_rank)`` key to the DL rows.

    ``residuals`` must already be filtered to the cell and left in export order;
    the reconstructed arrays come from ``materialize_corridor(...,
    return_timestamps=True, return_slots=True)`` and are positionally aligned to
    it — a contract the caller MUST have gated with ``verify_alignment`` first.
    """
    if not (residuals.height == len(timestamps) == len(slots) == len(directions)):
        raise ValueError(
            "dl_population: residual rows and reconstructed keys disagree in length "
            f"({residuals.height} vs {len(timestamps)}/{len(slots)}/{len(directions)})"
        )
    return pl.DataFrame(
        {
            "direction": direction_labels(directions),
            "t": timestamps,
            "pair_rank": np.asarray(slots).astype(np.int64),
            "y_true": residuals.get_column("y_true").to_numpy(),
            "y_pred_dl": residuals.get_column("y_pred_dl").to_numpy(),
            "y_pred_persist": residuals.get_column("y_pred_persist").to_numpy(),
        }
    )


def join_xgb(dl: pl.DataFrame, xgb: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """LEFT JOIN the XGB predictions onto the DL population; require 100% coverage.

    Returns ``(joined, coverage_pct)``. Because ``JOIN_KEYS`` is unique on the XGB
    side, the join emits exactly one row per DL row, so the multiplicity of the DL
    population is preserved exactly.

    Raises
    ------
    ValueError
        If any DL row lacks an XGB counterpart.
    """
    # The index is stamped on the LEFT frame, in DL export order, before the join
    # can reorder anything — see distinct_population for why it must be exact.
    joined = dl.with_row_index(REPLICA_INDEX).join(xgb, on=JOIN_KEYS, how="left")
    if joined.height != dl.height:
        raise ValueError(
            "join_xgb: the join changed the row count "
            f"({dl.height} -> {joined.height}); the XGB key is not unique"
        )
    n_missing = int(joined.get_column("y_pred_xgb").is_null().sum())
    coverage_pct = 100.0 * (1.0 - n_missing / joined.height)
    if n_missing:
        raise ValueError(
            f"join_xgb: {n_missing} of {joined.height} DL rows have no XGB "
            f"counterpart on {JOIN_KEYS} (coverage {coverage_pct:.3f}%)"
        )
    return joined, coverage_pct


def distinct_population(joined: pl.DataFrame) -> pl.DataFrame:
    """One row per ``(direction, t, pair_rank)``: the FIRST replica, in temporal order.

    Determinism matters more here than anywhere else in this module. The replicas
    of one target share ``y_true`` and ``y_pred_xgb`` but each carries a DIFFERENT
    ``y_pred_dl`` (a different anchoring window predicted it), so *which* replica
    survives changes ``mae_dl_distinct``. Leaving that to polars' sort tie-break
    (``maintain_order`` defaults to False) or to ``unique``'s hashing would make the
    emitted CSV non-reproducible.

    So the choice is made explicit: "first replica" means the lowest row index in
    the DL export order, which is window-major and therefore the EARLIEST
    anchoring window for that target. An explicit index column joins the sort key,
    making the ordering a total order with no ties at all.

    The frame is then left in ``(direction, t, pair_rank)`` order, which also makes
    the Newey-West HAC lag in :func:`significance_row` meaningful: consecutive rows
    are consecutive minutes, exactly the autocorrelation the HAC kernel absorbs.
    """
    if REPLICA_INDEX not in joined.columns:
        raise ValueError(
            f"distinct_population: {REPLICA_INDEX!r} is missing — pass the frame "
            "returned by join_xgb, which stamps it in DL export order"
        )
    return (
        joined.sort([*JOIN_KEYS, REPLICA_INDEX], maintain_order=True)
        .unique(subset=JOIN_KEYS, keep="first", maintain_order=True)
        .drop(REPLICA_INDEX)
    )


# ---------------------------------------------------------------------------
# Metrics and significance
# ---------------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())


def population_mae(frame: pl.DataFrame, prediction: str) -> float:
    """MAE of ``prediction`` against ``y_true`` over every row of ``frame``."""
    return _mae(
        frame.get_column("y_true").to_numpy(), frame.get_column(prediction).to_numpy()
    )


def xgb_minus_dl_differential(joined: pl.DataFrame, metric: str = "MAE") -> np.ndarray:
    """Per-sample ``d_i = loss(XGB_i) - loss(DL_i)``; positive => the DL model wins.

    Delegates the loss algebra to :func:`significance.loss_differential` by
    mapping XGB onto its ``y_pred_dl`` slot and the DL model onto its
    ``y_pred_persist`` slot, so MAE/RMSE per-sample losses have exactly ONE
    implementation in the repo. The remap only flips which model is the reference;
    the resulting sign convention is the one documented above.
    """
    remapped = joined.select(
        [
            pl.col("y_true"),
            pl.col("y_pred_xgb").alias("y_pred_dl"),
            pl.col("y_pred_dl").alias("y_pred_persist"),
        ]
    )
    return loss_differential(remapped, metric)


def significance_row(
    corridor: str,
    horizon: int,
    frame: pl.DataFrame,
    metric: str = "MAE",
    population: str = "distinct_target",
) -> dict:
    """Paired DM + one-sided Wilcoxon for one cell of a de-duplicated population.

    The one-sided tests use ``alternative="greater"`` on ``d = loss_xgb -
    loss_dl``, i.e. H1 is "the DL model has the lower loss". The alternative is
    written into the row so a directional claim can never be underwritten by a
    two-sided p-value (the mistake the existing significance table makes).

    ``dm_p_one_sided`` is folded from the two-sided DM p-value, exact under the
    test's symmetric N(0, 1) null, so the HAC variance keeps its single
    implementation in ``significance.diebold_mariano``.
    """
    d_metric = xgb_minus_dl_differential(frame, metric)
    d_mae = xgb_minus_dl_differential(frame, "MAE")
    dm = diebold_mariano(d_metric)
    dm_p_one = (
        dm.p_value / 2.0 if dm.stat > 0 else 1.0 - dm.p_value / 2.0
    )
    return {
        "corridor": corridor,
        "horizon": int(horizon),
        "population": population,
        "metric": metric,
        "n": int(frame.height),
        "delta_loss": float(np.mean(d_metric)),
        "delta_mae": float(np.mean(d_mae)),
        "median_delta": float(np.median(d_metric)),
        "win_rate": float(np.mean(d_metric > 0.0)),
        "dm_stat": dm.stat,
        "dm_lag": dm.lag,
        "dm_p_two_sided": dm.p_value,
        "dm_p_one_sided": float(dm_p_one),
        "wilcoxon_alternative": DL_BETTER_ALTERNATIVE,
        "wilcoxon_p_one_sided": wilcoxon_signed_rank(
            d_metric, alternative=DL_BETTER_ALTERNATIVE
        ),
        "wilcoxon_p_two_sided": wilcoxon_signed_rank(d_metric),
        "dl_better": bool(np.mean(d_metric) > 0.0),
    }


def cell_metrics(
    corridor: str,
    horizon: int,
    joined: pl.DataFrame,
    distinct: pl.DataFrame,
    coverage_pct: float,
    align_diff_target: float,
    align_diff_persist: float,
    align_tol: float,
) -> dict:
    """One ``PAIRED_METRIC_COLUMNS`` row for a fully gated cell."""
    mae_dl = population_mae(joined, "y_pred_dl")
    mae_xgb = population_mae(joined, "y_pred_xgb")
    mae_persist = population_mae(joined, "y_pred_persist")
    mae_dl_distinct = population_mae(distinct, "y_pred_dl")
    mae_xgb_distinct = population_mae(distinct, "y_pred_xgb")
    return {
        "corridor": corridor,
        "horizon": int(horizon),
        "n_dl_rows": int(joined.height),
        "n_xgb_matched": int(joined.height),
        "coverage_pct": float(coverage_pct),
        "n_distinct_targets": int(distinct.height),
        "align_max_abs_diff_target": float(align_diff_target),
        "align_max_abs_diff_persist": float(align_diff_persist),
        "align_tol": float(align_tol),
        "mae_persist_matched": mae_persist,
        "mae_dl_matched": mae_dl,
        "mae_xgb_matched": mae_xgb,
        "delta_xgb_minus_dl_matched": mae_xgb - mae_dl,
        "dl_better_matched": bool(mae_xgb > mae_dl),
        "mae_dl_distinct": mae_dl_distinct,
        "mae_xgb_distinct": mae_xgb_distinct,
        "delta_xgb_minus_dl_distinct": mae_xgb_distinct - mae_dl_distinct,
        "dl_better_distinct": bool(mae_xgb_distinct > mae_dl_distinct),
    }


# ---------------------------------------------------------------------------
# Reconciliation against the committed aggregate baselines CSV
# ---------------------------------------------------------------------------

def load_reported_xgb_mae(results_dir: str | Path) -> pl.DataFrame:
    """Committed aggregate-population B5_XGB MAE per ``(corridor, horizon)``."""
    frames = []
    for name in ("baselines_results_multih.csv", "baselines_E4_results_multih.csv"):
        path = Path(results_dir) / name
        if not path.is_file():
            raise ValueError(f"load_reported_xgb_mae: missing {path}")
        frames.append(pl.read_csv(path))
    return (
        pl.concat(frames, how="vertical")
        .filter(
            (pl.col("direction") == "aggregate")
            & (pl.col("metric") == "MAE")
            & (pl.col("baseline") == "B5_XGB")
        )
        .select(
            [
                "corridor",
                pl.col("horizon").cast(pl.Int64),
                pl.col("value").alias("reported_xgb"),
            ]
        )
    )


def load_reported_dl_mae(
    results_dir: str | Path, horizons: tuple[int, ...] = HORIZONS
) -> pl.DataFrame:
    """Committed aggregate-direction LSTM MAE per ``(corridor, horizon)``.

    The ``baseline == "LSTM"`` filter is explicit even though today's DL results
    CSVs hold only LSTM rows: these files have grown extra baselines before, and
    a second row per key would silently fan the audit join out.
    """
    frames = []
    for horizon in horizons:
        for name in (f"lstm_results_h{horizon}.csv", f"lstm_E4_results_h{horizon}.csv"):
            path = Path(results_dir) / name
            if not path.is_file():
                raise ValueError(f"load_reported_dl_mae: missing {path}")
            frames.append(pl.read_csv(path))
    return (
        pl.concat(frames, how="vertical")
        .filter(
            (pl.col("direction") == "aggregate")
            & (pl.col("metric") == "MAE")
            & (pl.col("baseline") == "LSTM")
        )
        .select(
            [
                "corridor",
                pl.col("horizon").cast(pl.Int64),
                pl.col("value").alias("reported_dl"),
            ]
        )
    )


def _validate_reported(paired: pl.DataFrame, reported: pl.DataFrame, label: str) -> None:
    """The reported map must be one row per cell AND cover every paired cell."""
    keys = ["corridor", "horizon"]
    duplicates = reported.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError(
            f"audit_against_reported: duplicate reported {label} keys: "
            f"{duplicates.to_dicts()}"
        )
    missing = paired.select(keys).unique().join(reported, on=keys, how="anti").sort(keys)
    if missing.height > 0:
        raise ValueError(
            f"audit_against_reported: missing reported {label} MAE for "
            f"{missing.to_dicts()}"
        )


def audit_against_reported(
    paired: pl.DataFrame, results_dir: str | Path
) -> pl.DataFrame:
    """Reconcile the restricted MAEs against the committed aggregate values.

    ``abs_diff`` is the framing bias for XGBoost: how far the aggregate-population
    MAE sits from the same model re-scored over the DL's rows. ``sign_flip`` marks
    the cells where that bias alone reverses the DL-vs-XGB verdict — the reason
    the aggregate comparison could not be trusted.
    """
    reported_xgb = load_reported_xgb_mae(results_dir)
    reported_dl = load_reported_dl_mae(results_dir)
    _validate_reported(paired, reported_xgb, "B5_XGB")
    _validate_reported(paired, reported_dl, "LSTM")
    return (
        paired.select(
            [
                "corridor",
                "horizon",
                pl.lit("MAE").alias("metric"),
                "n_dl_rows",
                pl.col("mae_xgb_matched").alias("restricted_xgb"),
                pl.col("mae_dl_matched").alias("restricted_dl"),
                pl.col("delta_xgb_minus_dl_matched").alias("restricted_delta"),
                pl.col("dl_better_matched").alias("restricted_dl_better"),
            ]
        )
        .join(reported_xgb, on=["corridor", "horizon"], how="left")
        .join(reported_dl, on=["corridor", "horizon"], how="left")
        .with_columns(
            [
                (pl.col("restricted_xgb") - pl.col("reported_xgb"))
                .abs()
                .alias("abs_diff"),
                (pl.col("reported_xgb") - pl.col("reported_dl")).alias(
                    "reported_delta"
                ),
            ]
        )
        .with_columns((pl.col("reported_delta") > 0).alias("reported_dl_better"))
        .with_columns(
            (pl.col("restricted_dl_better") != pl.col("reported_dl_better")).alias(
                "sign_flip"
            )
        )
        .select(AUDIT_COLUMNS)
        .sort(["corridor", "horizon"])
    )
