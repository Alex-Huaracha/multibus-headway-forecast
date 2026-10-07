"""Does the paper's finding depend on the twelve-minute input window?

Why this exists
---------------
``DEFAULT_T_IN = 12`` was fixed when the windowing module was written and never
tuned: no notebook trains with another window, and nothing in the repo records
why twelve. A reviewer asked where the number comes from. The honest answer is
that it is arbitrary, and an arbitrary choice is only harmless if the result
survives changing it. This builder measures that instead of arguing it.

What it does
------------
It refits the XGBoost of notebook 22 locally with input windows of 5, 10, 12 and
20 minutes, on the main origin, and scores each fit with the same functions the
paper's tables use: the error, the dispersion ratio of Section V-B, the
unadjusted-threshold detection of Section V-C and the threshold-free AUC.

Two choices keep the comparison about the window and nothing else:

1. **One population for every window.** A longer window needs a longer run of
   contiguous minutes, so it admits fewer samples. Scoring each window on its own
   population would mix the window's effect with a change of sample. Every window
   is therefore scored on the targets the longest window admits, keyed by
   ``(corridor, direction, horizon, target_ts, pair_rank)``.
2. **The published hyperparameters.** Each ``(corridor, horizon)`` reuses the
   configuration notebook 22 selected on validation, and only the number of lags
   changes. Re-running the 24-configuration search per window would let the
   search absorb part of the window's effect.

The 12-minute row is the reproduction check: on the full population its MAE must
match ``xgb_contig_results.csv``, or this local refit is not the published model.

The LSTM is not refit: it trains on Kaggle GPU. The XGBoost shares its
squared-error objective, which is what Section III-B's argument rests on.

Requires ``data/processed/headways_E{2,4,59}.parquet`` (gitignored; see
``docs/dataset-manifest.md``).

Usage
-----
    uv run python -m src.build_window_sensitivity
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.baselines import contiguous_features  # noqa: E402
from src.baselines.contiguous_features import build_contiguous_features  # noqa: E402
from src.data.sample_index import make_sample_index  # noqa: E402
from src.data.windowing import compute_max_N  # noqa: E402
from src.evaluation.splits import split_temporal, winsorize_train_p99  # noqa: E402
from src.evaluation.vector_metrics import (  # noqa: E402
    MIN_VECTOR_LEN,
    VECTOR_KEY,
    bunching_flags,
    bunching_score,
    detection_scores,
    matthews_corrcoef,
    ranking_scores,
    regularity_error,
    vector_frame,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
SEARCH_CSV = CSV_DIR / "xgb_contig_search_config.csv"
PUBLISHED_CSV = CSV_DIR / "xgb_contig_results.csv"
OUT_CSV = CSV_DIR / "window_sensitivity.csv"

CORRIDORS = [("E2", 2), ("E59", 59), ("E4", 4)]
HORIZONS = [1, 3, 5, 10]
WINDOWS = [5, 10, 12, 20]
PUBLISHED_WINDOW = 12

TARGET_KEY = ["corridor", "direction", "horizon", "target_ts", "pair_rank"]
MODELS = (("XGBoost", "y_pred_model"), ("Persistence", "y_pred_persist"))


def published_configs() -> dict[tuple[str, int], tuple[dict, int]]:
    """The validation-selected configuration and stopping round of notebook 22."""
    search = pl.read_csv(SEARCH_CSV)
    out = {}
    for row in search.iter_rows(named=True):
        params = {k.removeprefix("param_"): v for k, v in row.items() if k.startswith("param_")}
        out[(row["corridor"], int(row["horizon"]))] = (params, int(row["best_iteration"]))
    return out


def prepare(name: str, emp: int, path: Path | None = None) -> tuple[pl.DataFrame, dict]:
    path = path or DATA_DIR / f"headways_{name}.parquet"
    raw = pl.read_parquet(path).with_columns(
        pl.lit(emp, dtype=pl.Int64).alias("empresaid")
    )
    frame, _threshold = winsorize_train_p99(split_temporal(raw))
    max_n = compute_max_N(frame.filter(pl.col("split") == "train"), quantile=0.99)
    return frame, max_n


def features(frame, max_n, split, horizon, window):
    part = frame.filter(pl.col("split") == split)
    index = make_sample_index(part, horizon=horizon, T_in=window)
    # The lag count is a module constant tied to the published window; it is
    # rebound here so the model reads exactly ``window`` lags.
    contiguous_features.N_LAGS = window
    try:
        return build_contiguous_features(
            part, index, horizon=horizon, T_in=window, max_N_by_direction=max_n
        )
    finally:
        contiguous_features.N_LAGS = PUBLISHED_WINDOW


def fit_predict(frame, max_n, name, horizon, window, config) -> pl.DataFrame:
    """Test residuals of one window, under the published configuration."""
    params, _published_round = config
    names = [f"lag_{k}" for k in range(1, window + 1)] + [
        "hour", "weekday", "direction", "pair_rank"
    ]
    Xtr, ytr, _ = features(frame, max_n, "train", horizon, window)
    Xva, yva, _ = features(frame, max_n, "val", horizon, window)
    Xte, yte, kte = features(frame, max_n, "test", horizon, window)

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=names)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=names)
    booster = xgb.train(
        {"objective": "reg:squarederror", "seed": 42, "nthread": 4, **params},
        dtr, num_boost_round=800, evals=[(dva, "val")],
        early_stopping_rounds=40, verbose_eval=False,
    )
    pred = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))

    return pl.DataFrame({
        "corridor": [name] * len(yte),
        "direction": kte.get_column("direction"),
        "horizon": kte.get_column("horizon"),
        "start_ts": kte.get_column("start_ts"),
        "target_ts": kte.get_column("target_ts"),
        "pair_rank": kte.get_column("pair_rank"),
        "y_true": yte.astype("float64"),
        "y_pred_model": pred.astype("float64"),
        "y_pred_persist": kte.get_column("y_pred_persist").cast(pl.Float64),
        "best_iteration": [booster.best_iteration] * len(yte),
    })


def score(cell: pl.DataFrame) -> list[dict]:
    """The paper's measurements for one window and cell, per model."""
    lengths = cell.group_by(VECTOR_KEY).agg(pl.len().alias("vector_len"))
    cell = cell.join(lengths, on=VECTOR_KEY, how="inner").filter(
        pl.col("vector_len") >= MIN_VECTOR_LEN
    )
    columns = ["y_true"] + [c for _, c in MODELS]
    cell = cell.with_columns(
        [bunching_flags(cell, c).alias(f"_bunch_{c}") for c in columns]
        + [bunching_score(cell, c).alias(f"_score_{c}") for c in columns]
    )
    vectors = vector_frame(cell, columns)
    truth = cell.get_column("_bunch_y_true").to_numpy()

    variances = (
        cell.group_by(VECTOR_KEY, maintain_order=True)
        .agg([pl.col(c).var(ddof=0).alias(f"v_{c}") for c in columns])
        .drop_nulls()
    )
    v_true = float(variances.get_column("v_y_true").mean())

    rows = []
    for name, column in MODELS:
        flags = cell.get_column(f"_bunch_{column}").to_numpy()
        detection = detection_scores(truth, flags)
        ranking = ranking_scores(truth, cell.get_column(f"_score_{column}").to_numpy())
        rows.append({
            "model": name,
            "n_rows": cell.height,
            "mae": float((cell.get_column("y_true") - cell.get_column(column)).abs().mean()),
            "cv_bias": regularity_error(vectors, column)["cv_bias"],
            "dispersion_ratio": float(variances.get_column(f"v_{column}").mean()) / v_true,
            "alarm_rate": detection.pred_rate,
            "event_rate": detection.true_rate,
            "f1_unadjusted": detection.f1,
            "mcc_unadjusted": matthews_corrcoef(truth, flags),
            "auc": ranking["auc"],
        })
    return rows


def main() -> None:
    configs = published_configs()
    published = pl.read_csv(PUBLISHED_CSV).filter(
        (pl.col("baseline") == "B5_XGB_CONTIG") & (pl.col("metric") == "MAE")
    )

    rows: list[dict] = []
    for name, emp in CORRIDORS:
        frame, max_n = prepare(name, emp)
        for horizon in HORIZONS:
            residuals = {}
            for window in WINDOWS:
                t0 = time.time()
                residuals[window] = fit_predict(
                    frame, max_n, name, horizon, window, configs[(name, horizon)]
                )
                print(f"{name} h{horizon} L{window}: {residuals[window].height:,} rows "
                      f"({time.time() - t0:.0f}s)", flush=True)

            # Reproduction check, on the window's own (full) population.
            full = residuals[PUBLISHED_WINDOW]
            mae_full = float((full["y_true"] - full["y_pred_model"]).abs().mean())
            mae_pub = published.filter(
                (pl.col("corridor") == name) & (pl.col("horizon") == horizon)
            ).get_column("value").item()

            common = residuals[max(WINDOWS)].select(TARGET_KEY)
            for window in WINDOWS[:-1]:
                common = common.join(residuals[window].select(TARGET_KEY), on=TARGET_KEY)

            for window in WINDOWS:
                cell = residuals[window].join(common, on=TARGET_KEY, how="inner")
                for row in score(cell):
                    rows.append({
                        "corridor": name, "horizon": horizon, "window": window,
                        **row,
                        "best_iteration": int(cell.get_column("best_iteration")[0]),
                        "mae_full_population_L12": mae_full,
                        "mae_published_L12": mae_pub,
                    })

    table = pl.DataFrame(rows).sort(["corridor", "horizon", "window", "model"])
    table.write_csv(OUT_CSV)
    print(f"\nWritten: {OUT_CSV} ({table.height} rows)")


if __name__ == "__main__":
    sys.exit(main())
