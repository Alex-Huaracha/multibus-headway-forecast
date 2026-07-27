"""Build the paired DL-vs-XGBoost metric, audit and significance CSVs.

Usage:
    uv run python -m src.build_xgb_paired_metrics

Inputs (downloaded from Kaggle, NOT versioned — see .gitignore):
    docs/resultados/residuos-multihorizon/11-lstm/h{1,3,5,10}/lstm_residuals_h*.csv
    docs/resultados/residuos-multihorizon/11-lstm/h{1,3,5,10}/lstm_E4_residuals_h*.csv
    docs/resultados/residuos-multihorizon/20-xgb-paired/xgb_paired_persample_test.csv
    data/processed/headways_E{2,4,59}.parquet   (to reconstruct the DL population key)

Inputs (versioned):
    docs/resultados/csv-multihorizon/baselines_results_multih.csv
    docs/resultados/csv-multihorizon/baselines_E4_results_multih.csv
    docs/resultados/csv-multihorizon/lstm_results_h*.csv
    docs/resultados/csv-multihorizon/lstm_E4_results_h*.csv

Outputs (versioned — small, paper reproducibility):
    docs/resultados/csv-multihorizon/xgb_paired_dl_metrics.csv        (12 rows)
    docs/resultados/csv-multihorizon/xgb_paired_vs_reported_audit.csv (12 rows)
    docs/resultados/csv-multihorizon/xgb_paired_significance.csv      (12 rows)

What this fixes
---------------
The paper's DL-vs-XGBoost verdict used to compare an LSTM MAE over the DL WINDOW
population against an XGBoost MAE over the FULL test population. This builder
re-scores XGBoost over exactly the rows the LSTM was scored on, using the
per-sample NB20 export keyed on ``(direction, t, pair_rank)``. See
``src.evaluation.xgb_paired`` for the two populations emitted and why both exist.

Every cell passes three hard gates before a single number is emitted for it —
positional alignment against the residual CSV (``verify_alignment``), 100% join
coverage, and XGB key uniqueness. A failing gate raises; nothing is skipped
silently and no partial CSV is written.

Output-naming constraint (enforced by tests/test_xgb_paired_metrics.py): the
filenames must not contain ``_results_``, ``_residuals_h`` or ``_multiseed_``.
Those substrings are globbed by ``build_degradation_curve``,
``build_significance_table`` and ``evaluation.multiseed`` respectively, which
would pick these incompatible schemas up and crash or silently contaminate the
degradation figure.

This script reads residual files sequentially. It does not train models, does not
regenerate notebooks, and does not touch any existing CSV.
"""
from __future__ import annotations

import os
from pathlib import Path

# Determinism contract (see tests/test_report_builder_determinism.py): pin the
# polars thread count BEFORE importing polars so outputs are byte-identical.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402

from src.build_exante_volatility import (  # noqa: E402
    ALIGN_TOL,
    materialize_corridor,
    prepare_df,
    verify_alignment,
)
from src.evaluation.xgb_paired import (  # noqa: E402
    AUDIT_COLUMNS,
    CORRIDOR_EMPRESAID,
    HORIZONS,
    PAIRED_METRIC_COLUMNS,
    SIGNIFICANCE_COLUMNS,
    audit_against_reported,
    cell_metrics,
    distinct_population,
    dl_population,
    join_xgb,
    load_xgb_export,
    residual_csv_path,
    significance_row,
    validate_inputs,
    xgb_cell,
    xgb_export_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESID_DIR = REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon"
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"

PAIRED_METRICS_CSV = "xgb_paired_dl_metrics.csv"
AUDIT_CSV = "xgb_paired_vs_reported_audit.csv"
SIGNIFICANCE_CSV = "xgb_paired_significance.csv"

CORRIDORS = ("E2", "E59", "E4")


def _corridor_cells(
    corridor: str,
    resid_dir: Path,
    export: pl.DataFrame,
    horizons: tuple[int, ...],
) -> tuple[list[dict], list[dict]]:
    """Metric + significance rows for every horizon of one corridor.

    ``prepare_df`` is paid once per corridor (it loads and winsorizes the whole
    parquet), then reused across horizons.
    """
    empresaid = CORRIDOR_EMPRESAID[corridor]
    # flush=True: a full run takes ~13 minutes, and a redirected stdout is
    # block-buffered, so without this the log stays empty until the very end and
    # a stalled run is indistinguishable from a slow one.
    print(f"\n=== {corridor} (empresaid={empresaid}) ===", flush=True)
    df, stats = prepare_df(empresaid)

    metric_rows: list[dict] = []
    significance_rows: list[dict] = []
    for horizon in horizons:
        csv_path = residual_csv_path(resid_dir, corridor, horizon)
        targets, persist, _ex_ante, timestamps, slots, directions = (
            materialize_corridor(
                df,
                stats,
                empresaid,
                horizon,
                splits=("test",),
                return_timestamps=True,
                return_slots=True,
            )
        )

        # GATE 1 — the reconstructed key arrays are only positionally joinable to
        # the residual CSV if the reconstruction reproduces its targets exactly.
        passed, diff_target, diff_persist, _n_rec = verify_alignment(
            corridor, horizon, targets, persist, csv_path,
            csv_corridor_filter=corridor,
        )
        if not passed:
            raise ValueError(
                f"build_xgb_paired_metrics: alignment gate FAILED for {corridor} "
                f"h={horizon} (max|Δtarget|={diff_target}, "
                f"max|Δpersist|={diff_persist}, tol={ALIGN_TOL}) — refusing to "
                "emit metrics for a population that is not provably the DL's"
            )

        residuals = pl.read_csv(csv_path).filter(pl.col("corridor") == corridor)
        dl = dl_population(residuals, timestamps, slots, directions)

        # GATES 2 and 3 — key uniqueness on the XGB side, 100% join coverage.
        joined, coverage_pct = join_xgb(dl, xgb_cell(export, corridor, horizon))
        distinct = distinct_population(joined)

        row = cell_metrics(
            corridor, horizon, joined, distinct, coverage_pct,
            diff_target, diff_persist, ALIGN_TOL,
        )
        metric_rows.append(row)
        significance_rows.append(significance_row(corridor, horizon, distinct))

        print(
            f"  h={horizon:>2}: n={row['n_dl_rows']:>9,} "
            f"cov={coverage_pct:7.3f}% distinct={row['n_distinct_targets']:>8,} "
            f"DL={row['mae_dl_matched']:.4f} XGB={row['mae_xgb_matched']:.4f} "
            f"Δ={row['delta_xgb_minus_dl_matched']:+.4f} "
            f"{'DL' if row['dl_better_matched'] else 'XGB'} wins",
            flush=True,
        )

    return metric_rows, significance_rows


def build(
    resid_dir: Path = RESID_DIR,
    out_dir: Path = OUT_DIR,
    results_dir: Path | None = None,
    corridors: tuple[str, ...] = CORRIDORS,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[Path, Path, Path]:
    """Write the paired metrics, reported-vs-restricted audit and significance CSVs.

    ``results_dir`` defaults to ``out_dir`` (the committed aggregate CSVs live
    beside the outputs, as in ``build_paired_audit``).
    """
    results_dir = out_dir if results_dir is None else results_dir

    # Fail fast BEFORE any parquet loading, so a missing Kaggle download costs a
    # clear error instead of minutes of materialization (PKR1).
    validate_inputs(resid_dir, corridors=corridors, horizons=horizons)
    export = load_xgb_export(xgb_export_path(resid_dir))

    metric_rows: list[dict] = []
    significance_rows: list[dict] = []
    for corridor in corridors:
        cells, sig = _corridor_cells(corridor, Path(resid_dir), export, horizons)
        metric_rows.extend(cells)
        significance_rows.extend(sig)

    paired = (
        pl.DataFrame(metric_rows)
        .select(PAIRED_METRIC_COLUMNS)
        .sort(["corridor", "horizon"])
    )
    significance = (
        pl.DataFrame(significance_rows)
        .select(SIGNIFICANCE_COLUMNS)
        .sort(["corridor", "horizon"])
    )
    audit = audit_against_reported(paired, results_dir).select(AUDIT_COLUMNS)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_path = out_dir / PAIRED_METRICS_CSV
    audit_path = out_dir / AUDIT_CSV
    significance_path = out_dir / SIGNIFICANCE_CSV
    paired.write_csv(paired_path)
    audit.write_csv(audit_path)
    significance.write_csv(significance_path)
    return paired_path, audit_path, significance_path


def _print_summary(paired_path: Path, audit_path: Path, significance_path: Path) -> None:
    paired = pl.read_csv(paired_path)
    audit = pl.read_csv(audit_path)
    significance = pl.read_csv(significance_path)

    print(f"\nWrote paired metrics: {paired_path} ({paired.height} rows)")
    print(f"Wrote reported audit: {audit_path} ({audit.height} rows)")
    print(f"Wrote significance:   {significance_path} ({significance.height} rows)")

    with pl.Config(tbl_rows=40, tbl_width_chars=220, float_precision=4):
        print("\n=== MULTIPLICITY-MATCHED (comparable to the reported LSTM MAE) ===")
        print(
            paired.select(
                [
                    "corridor", "horizon", "n_dl_rows", "coverage_pct",
                    "mae_persist_matched", "mae_dl_matched", "mae_xgb_matched",
                    "delta_xgb_minus_dl_matched", "dl_better_matched",
                ]
            )
        )
        print("\n=== DISTINCT-TARGET (basis for the paired significance test) ===")
        print(
            paired.select(
                [
                    "corridor", "horizon", "n_distinct_targets",
                    "mae_dl_distinct", "mae_xgb_distinct",
                    "delta_xgb_minus_dl_distinct", "dl_better_distinct",
                ]
            )
        )
        print("\n=== RESTRICTED vs REPORTED (XGB framing bias) ===")
        print(
            audit.select(
                [
                    "corridor", "horizon", "restricted_xgb", "reported_xgb",
                    "abs_diff", "restricted_dl_better", "reported_dl_better",
                    "sign_flip",
                ]
            )
        )
        print("\n=== PAIRED SIGNIFICANCE (distinct-target, one-sided) ===")
        print(
            significance.select(
                [
                    "corridor", "horizon", "n", "delta_mae", "median_delta",
                    "win_rate", "dm_p_one_sided", "wilcoxon_alternative",
                    "wilcoxon_p_one_sided", "dl_better",
                ]
            )
        )

    n_flips = int(audit.get_column("sign_flip").sum())
    print(f"\nCells whose verdict flips between reported and restricted: {n_flips}/12")
    n_dl_wins = int(paired.get_column("dl_better_matched").sum())
    print(f"Cells where the LSTM wins on the restricted population: {n_dl_wins}/12")


if __name__ == "__main__":
    _print_summary(*build())
