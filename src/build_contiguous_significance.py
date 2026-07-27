"""Significance package for the retrained pipeline.

Recomputes the DM / Wilcoxon verdicts on the contiguous-pipeline residuals, with
the corrections the audit left open folded in (see
``evaluation/significance_clustered``): a variance clustered on service day
instead of a mis-ordered HAC (#6), the HLN small-sample apparatus (#7), and a
Wilcoxon reported with its direction rather than two-sided beside a mean-derived
verdict (#1).

Three comparisons per cell, all over **identical samples** joined on the full
key — no aggregate stands in for a paired test:

    LSTM vs persistence     the headline crossover claim
    LSTM vs XGBoost         does the network beat a levelled learner
    XGBoost vs persistence  is the crossover a property of deep learning at all

Both variance estimators are reported side by side. The clustered one is the
verdict; the HAC one is kept so the size of the correction is visible rather
than asserted.

Usage
-----
    uv run python -m src.build_contiguous_significance
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.evaluation.residual_export import RESIDUAL_KEY_COLUMNS  # noqa: E402
from src.evaluation.significance_clustered import (  # noqa: E402
    dm_clustered,
    dm_hac_hln,
    wilcoxon_directional,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LSTM_DIR = REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon" / "21-lstm-contiguous"
XGB_CSV = (
    REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon"
    / "22-xgb-contiguous" / "xgb_contig_residuals.csv"
)
OUT_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "contiguous_significance.csv"
)

CORRIDORS = ("E2", "E59", "E4")
HORIZONS = (1, 3, 5, 10)
METRICS = ("MAE", "RMSE")

# `alpha` is not applied here — verdicts are reported with their p-values so the
# multiple-comparison family can be chosen at analysis time, not baked in.


def load_lstm() -> pl.DataFrame:
    """All LSTM residuals, both corridor groups, every horizon."""
    frames = []
    for horizon in HORIZONS:
        for stem in (f"lstm_contig_residuals_h{horizon}",
                     f"lstm_contig_E4_residuals_h{horizon}"):
            path = LSTM_DIR / f"{stem}.csv"
            if path.exists():
                frames.append(pl.read_csv(path, try_parse_dates=True))
    if not frames:
        raise FileNotFoundError(
            f"no LSTM residuals under {LSTM_DIR}. Download the kernel outputs first."
        )
    return pl.concat(frames)


def per_sample_loss(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> np.ndarray:
    """Per-sample loss whose mean is the reported metric."""
    if metric == "MAE":
        return np.abs(y_true - y_pred)
    if metric == "RMSE":
        # Squared error: its mean is the MSE behind RMSE.
        return (y_true - y_pred) ** 2
    raise ValueError(f"unknown metric: {metric}")


def verdict(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    service_day: np.ndarray,
    *,
    metric: str,
    horizon: int,
) -> dict:
    """One paired verdict: model A against model B over identical samples.

    ``d = loss(A) - loss(B)``, so a negative mean means A wins.
    """
    d = per_sample_loss(y_true, pred_a, metric) - per_sample_loss(y_true, pred_b, metric)

    hac = dm_hac_hln(d, horizon=horizon)
    clu = dm_clustered(d, service_day, horizon=horizon)
    wil = wilcoxon_directional(d)

    # Effect size is always reported in MAE minutes, whatever the test metric.
    delta_mae = float(
        (np.abs(y_true - pred_a) - np.abs(y_true - pred_b)).mean()
    )

    return {
        "n": int(d.size),
        "delta_loss": float(d.mean()),
        "delta_mae": delta_mae,
        "a_better": bool(d.mean() < 0),
        "dm_stat_hac": hac.stat,
        "dm_p_hac": hac.p_value,
        "dm_lag": hac.lag,
        "dm_stat_clustered": clu.stat,
        "dm_p_clustered": clu.p_value,
        "n_service_days": clu.n_clusters,
        "wilcoxon_p_two_sided": wil["wilcoxon_p_two_sided"],
        "wilcoxon_p_one_sided": wil["wilcoxon_p_one_sided"],
        "median_diff": wil["median_diff"],
        "win_rate": wil["win_rate"],
        "mean_median_disagree": wil["mean_median_disagree"],
    }


def build() -> pl.DataFrame:
    lstm = load_lstm()
    xgb = pl.read_csv(XGB_CSV, try_parse_dates=True)

    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            left = lstm.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            right = xgb.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            if left.height == 0 or right.height == 0:
                continue

            joined = left.join(
                right.select(RESIDUAL_KEY_COLUMNS + ["y_pred_model"]).rename(
                    {"y_pred_model": "y_pred_xgb"}
                ),
                on=RESIDUAL_KEY_COLUMNS,
                how="inner",
            )
            coverage = 100.0 * joined.height / left.height

            y_true = joined.get_column("y_true").to_numpy()
            dl = joined.get_column("y_pred_model").to_numpy()
            xg = joined.get_column("y_pred_xgb").to_numpy()
            pe = joined.get_column("y_pred_persist").to_numpy()
            # Service day of the TARGET: the day whose conditions the sample lives in.
            day = joined.get_column("target_ts").dt.date().to_numpy()

            comparisons = (
                ("LSTM_vs_PERSIST", dl, pe),
                ("LSTM_vs_XGB", dl, xg),
                ("XGB_vs_PERSIST", xg, pe),
            )
            for metric in METRICS:
                for name, pred_a, pred_b in comparisons:
                    rows.append(
                        {
                            "corridor": corridor,
                            "horizon": horizon,
                            "metric": metric,
                            "comparison": name,
                            "join_coverage_pct": round(coverage, 4),
                            **verdict(
                                y_true, pred_a, pred_b, day,
                                metric=metric, horizon=horizon,
                            ),
                        }
                    )

    return pl.DataFrame(rows).sort(["comparison", "metric", "corridor", "horizon"])


def main() -> None:
    table = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=100, tbl_cols=14, tbl_width_chars=220):
        print(
            table.filter(pl.col("metric") == "MAE").select(
                ["comparison", "corridor", "horizon", "n", "n_service_days",
                 "delta_mae", "dm_p_hac", "dm_p_clustered",
                 "wilcoxon_p_one_sided", "win_rate", "mean_median_disagree"]
            )
        )
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
