"""Paired residual metric audit for DL-vs-persistence aggregates.

The paper's reported multi-horizon CSVs contain aggregate metrics, while the
significance tests operate on paired per-sample residuals. This module computes
the same headline metrics directly from the paired residual samples and joins
them back to the reported aggregate CSVs, without retraining or regenerating
notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl

from src.evaluation.degradation import load_results

RESIDUAL_COLUMNS = [
    "corridor",
    "direction",
    "horizon",
    "y_true",
    "y_pred_dl",
    "y_pred_persist",
]

MODEL_DIRS = {
    "11-lstm": "LSTM",
    "12-spatialconvlstm": "SpatialConvLSTM",
    "13-spatialtransformer": "SpatialTransformer",
}
MODEL_SLUGS = {
    "lstm": "LSTM",
    "spatial_conv_lstm": "SpatialConvLSTM",
    "spatial_transformer": "SpatialTransformer",
}

HORIZONS = (3, 5, 10)
METRICS = ("MAE", "RMSE")

PAIRED_METRIC_COLUMNS = [
    "model",
    "corridor",
    "horizon",
    "n",
    "mae_dl",
    "rmse_dl",
    "mae_persist",
    "rmse_persist",
    "delta_mae",
    "delta_rmse",
]

AUDIT_COLUMNS = [
    "model",
    "corridor",
    "horizon",
    "metric",
    "n",
    "paired_dl",
    "reported_dl",
    "abs_diff_dl",
    "paired_persist",
    "reported_persist",
    "abs_diff_persist",
    "paired_delta",
    "reported_delta",
    "paired_dl_better",
    "reported_dl_better",
    "sign_mismatch",
]

_SLUG_PATTERN = "|".join(
    re.escape(slug) for slug in sorted(MODEL_SLUGS, key=len, reverse=True)
)
_RESIDUAL_NAME_RE = re.compile(
    rf"^(?P<slug>{_SLUG_PATTERN})(?:_E4)?_residuals_h(?P<horizon>\d+)\.csv$"
)
_HORIZON_DIR_RE = re.compile(r"^h(?P<horizon>\d+)$")


@dataclass(frozen=True)
class ResidualFile:
    """A discovered residual CSV with model and horizon parsed from context."""

    path: Path
    model: str
    horizon: int


def parse_residual_file(path: str | Path) -> ResidualFile:
    """Parse model and horizon metadata from a residual CSV path.

    The model is resolved from the canonical residual directory when present
    (``11-lstm``, ``12-spatialconvlstm``, ``13-spatialtransformer``), and cross-
    checked against the filename slug. The horizon is parsed from the filename
    and cross-checked against the parent ``h{H}`` directory when present.
    """
    path = Path(path)
    match = _RESIDUAL_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"parse_residual_file: not a residual CSV name: {path.name}")

    file_model = MODEL_SLUGS[match.group("slug")]
    file_horizon = int(match.group("horizon"))

    dir_model = next(
        (MODEL_DIRS[part] for part in path.parts if part in MODEL_DIRS), None
    )
    if dir_model is not None and dir_model != file_model:
        raise ValueError(
            "parse_residual_file: model mismatch between directory "
            f"{dir_model!r} and filename {file_model!r} for {path}"
        )

    dir_horizon = None
    parent_match = _HORIZON_DIR_RE.match(path.parent.name)
    if parent_match is not None:
        dir_horizon = int(parent_match.group("horizon"))
    if dir_horizon is not None and dir_horizon != file_horizon:
        raise ValueError(
            "parse_residual_file: horizon mismatch between directory "
            f"h{dir_horizon} and filename h{file_horizon} for {path}"
        )

    return ResidualFile(path=path, model=dir_model or file_model, horizon=file_horizon)


def discover_residual_files(
    residuals_root: str | Path, horizons: tuple[int, ...] = HORIZONS
) -> list[ResidualFile]:
    """Discover residual CSVs for the requested horizons.

    Only filenames matching ``*_residuals_h*.csv`` are considered, so co-located
    ``*_results_h*.csv`` aggregate files are not accidentally included.
    """
    residuals_root = Path(residuals_root)
    wanted = set(horizons)
    discovered: list[ResidualFile] = []

    for path in sorted(residuals_root.rglob("*_residuals_h*.csv")):
        meta = parse_residual_file(path)
        if meta.horizon in wanted:
            discovered.append(meta)

    if not discovered:
        raise ValueError(
            f"discover_residual_files: no residual CSVs for horizons {horizons} "
            f"under {residuals_root}"
        )

    return sorted(discovered, key=lambda item: (item.model, item.horizon, str(item.path)))


def _format_path_label(path: Path | None) -> str:
    return f"{path.name}: " if path is not None else ""


def _validate_residual_frame(
    df: pl.DataFrame, path: Path | None = None, expected_horizon: int | None = None
) -> None:
    missing = [column for column in RESIDUAL_COLUMNS if column not in df.columns]
    if missing:
        label = _format_path_label(path)
        raise ValueError(f"paired_metrics: {label}missing residual columns {missing}")

    if expected_horizon is None:
        return

    label = _format_path_label(path)
    try:
        horizons = (
            df.select(pl.col("horizon").cast(pl.Int64, strict=True).unique().sort())
            .get_column("horizon")
            .to_list()
        )
    except Exception as exc:
        raise ValueError(
            f"paired_metrics: {label}residual horizon values must be integer-compatible "
            f"with parsed/discovered horizon h{expected_horizon}"
        ) from exc

    if horizons != [expected_horizon]:
        raise ValueError(
            f"paired_metrics: {label}residual horizon values {horizons} do not match "
            f"parsed/discovered horizon h{expected_horizon}"
        )


def paired_metrics_table(df: pl.DataFrame, model: str) -> pl.DataFrame:
    """Compute aggregate paired metrics per ``(model, corridor, horizon)``."""
    _validate_residual_frame(df)

    table = (
        df.select(RESIDUAL_COLUMNS)
        .with_columns(
            [
                (pl.col("y_true") - pl.col("y_pred_dl")).abs().alias("_ae_dl"),
                (pl.col("y_true") - pl.col("y_pred_persist"))
                .abs()
                .alias("_ae_persist"),
                ((pl.col("y_true") - pl.col("y_pred_dl")) ** 2).alias("_se_dl"),
                ((pl.col("y_true") - pl.col("y_pred_persist")) ** 2).alias(
                    "_se_persist"
                ),
            ]
        )
        .group_by(["corridor", "horizon"])
        .agg(
            [
                pl.len().alias("n"),
                pl.col("_ae_dl").mean().alias("mae_dl"),
                pl.col("_se_dl").mean().sqrt().alias("rmse_dl"),
                pl.col("_ae_persist").mean().alias("mae_persist"),
                pl.col("_se_persist").mean().sqrt().alias("rmse_persist"),
            ]
        )
        .with_columns(
            [
                pl.lit(model).alias("model"),
                pl.col("horizon").cast(pl.Int64),
                (pl.col("mae_dl") - pl.col("mae_persist")).alias("delta_mae"),
                (pl.col("rmse_dl") - pl.col("rmse_persist")).alias("delta_rmse"),
            ]
        )
        .select(PAIRED_METRIC_COLUMNS)
        .sort(["model", "corridor", "horizon"])
    )
    return table


def build_paired_metrics(
    residuals_root: str | Path, horizons: tuple[int, ...] = HORIZONS
) -> pl.DataFrame:
    """Compute paired metrics by reading residual files sequentially."""
    tables: list[pl.DataFrame] = []
    for residual_file in discover_residual_files(residuals_root, horizons=horizons):
        frame = pl.read_csv(residual_file.path)
        _validate_residual_frame(
            frame, residual_file.path, expected_horizon=residual_file.horizon
        )
        tables.append(paired_metrics_table(frame, model=residual_file.model))

    consolidated = pl.concat(tables, how="vertical").sort(
        ["model", "corridor", "horizon"]
    )
    duplicates = (
        consolidated.group_by(["model", "corridor", "horizon"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicates.height > 0:
        raise ValueError(
            "build_paired_metrics: duplicate model/corridor/horizon groups found: "
            f"{duplicates.to_dicts()}"
        )
    return consolidated.select(PAIRED_METRIC_COLUMNS)


def paired_metrics_long(paired: pl.DataFrame) -> pl.DataFrame:
    """Convert wide paired metrics into one row per metric for auditing."""
    mae = paired.select(
        [
            "model",
            "corridor",
            "horizon",
            "n",
            pl.lit("MAE").alias("metric"),
            pl.col("mae_dl").alias("paired_dl"),
            pl.col("mae_persist").alias("paired_persist"),
            pl.col("delta_mae").alias("paired_delta"),
        ]
    )
    rmse = paired.select(
        [
            "model",
            "corridor",
            "horizon",
            "n",
            pl.lit("RMSE").alias("metric"),
            pl.col("rmse_dl").alias("paired_dl"),
            pl.col("rmse_persist").alias("paired_persist"),
            pl.col("delta_rmse").alias("paired_delta"),
        ]
    )
    return pl.concat([mae, rmse], how="vertical").select(
        [
            "model",
            "corridor",
            "horizon",
            "metric",
            "n",
            "paired_dl",
            "paired_persist",
            "paired_delta",
        ]
    )


def load_reported_dl_metrics(
    results_dir: str | Path, horizons: tuple[int, ...] = HORIZONS
) -> pl.DataFrame:
    """Load reported aggregate DL metrics from model ``*_results_h*.csv`` files."""
    return (
        load_results(results_dir, pattern="*_results_h*.csv")
        .filter(
            (pl.col("direction") == "aggregate")
            & pl.col("baseline").is_in(list(MODEL_DIRS.values()))
            & pl.col("metric").is_in(list(METRICS))
            & pl.col("horizon").is_in(list(horizons))
        )
        .select(
            [
                pl.col("baseline").alias("model"),
                "corridor",
                "horizon",
                "metric",
                pl.col("value").alias("reported_dl"),
            ]
        )
    )


def load_reported_persistence_metrics(
    results_dir: str | Path, horizons: tuple[int, ...] = HORIZONS
) -> pl.DataFrame:
    """Load reported aggregate B1 persistence metrics from baseline CSVs."""
    return (
        load_results(results_dir, pattern="baselines*_results_multih.csv")
        .filter(
            (pl.col("direction") == "aggregate")
            & (pl.col("baseline") == "B1")
            & pl.col("metric").is_in(list(METRICS))
            & pl.col("horizon").is_in(list(horizons))
        )
        .select(
            [
                "corridor",
                "horizon",
                "metric",
                pl.col("value").alias("reported_persist"),
            ]
        )
    )


def _validate_unique_metric_keys(
    df: pl.DataFrame, keys: list[str], label: str
) -> None:
    duplicates = df.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError(
            f"audit_against_reported: duplicate {label} metric keys found: "
            f"{duplicates.to_dicts()}"
        )


def _validate_reported_metric_coverage(
    paired_long: pl.DataFrame,
    reported: pl.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    needed = paired_long.select(keys).unique()
    available = reported.select(keys).unique()
    missing = needed.join(available, on=keys, how="anti").sort(keys)
    if missing.height > 0:
        raise ValueError(
            f"audit_against_reported: missing reported {label} metric keys: "
            f"{missing.to_dicts()}"
        )


def audit_against_reported(
    paired: pl.DataFrame,
    results_dir: str | Path,
    horizons: tuple[int, ...] = HORIZONS,
) -> pl.DataFrame:
    """Join paired residual metrics to reported aggregates and flag sign changes."""
    paired_long = paired_metrics_long(paired)
    reported_dl = load_reported_dl_metrics(results_dir, horizons=horizons)
    reported_persist = load_reported_persistence_metrics(results_dir, horizons=horizons)

    dl_keys = ["model", "corridor", "horizon", "metric"]
    persist_keys = ["corridor", "horizon", "metric"]
    _validate_unique_metric_keys(reported_dl, dl_keys, "DL")
    _validate_unique_metric_keys(reported_persist, persist_keys, "persistence")
    _validate_reported_metric_coverage(paired_long, reported_dl, dl_keys, "DL")
    _validate_reported_metric_coverage(
        paired_long, reported_persist, persist_keys, "persistence"
    )

    audit = (
        paired_long.join(
            reported_dl,
            on=["model", "corridor", "horizon", "metric"],
            how="left",
        )
        .join(reported_persist, on=["corridor", "horizon", "metric"], how="left")
        .with_columns(
            [
                (pl.col("paired_dl") - pl.col("reported_dl"))
                .abs()
                .alias("abs_diff_dl"),
                (pl.col("paired_persist") - pl.col("reported_persist"))
                .abs()
                .alias("abs_diff_persist"),
                (pl.col("reported_dl") - pl.col("reported_persist")).alias(
                    "reported_delta"
                ),
                (pl.col("paired_delta") < 0).alias("paired_dl_better"),
            ]
        )
        .with_columns((pl.col("reported_delta") < 0).alias("reported_dl_better"))
        .with_columns(
            (pl.col("paired_dl_better") != pl.col("reported_dl_better")).alias(
                "sign_mismatch"
            )
        )
        .select(AUDIT_COLUMNS)
        .sort(["model", "corridor", "horizon", "metric"])
    )
    return audit
