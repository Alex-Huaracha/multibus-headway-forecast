"""Cell-level sign test: does the LSTM beat the leveled XGBoost across cells?

Why a sign test and not the per-sample DM/Wilcoxon test the paper uses for the
DL-vs-persistence claim: the DL and XGBoost residual exports cannot be paired
per sample. The DL export emits one row per overlapping window x bus (each test
target is overcounted ~4.5x) and carries no per-sample key; the baseline export
emits one row per test target. Realigning or de-duplicating them would require
re-running the DL kernels. So the strong-competitor comparison is tested at the
CELL level instead: each (corridor, horizon) cell is one fair-coin trial under
H0, which is immune to the overlapping-window overcounting (see
`SignTestResult`).

Reads the committed aggregate-MAE CSVs, runs the binomial sign test separately
for the large corridors (E2+E59) and the small one (E4), and writes a tidy,
auditable result CSV. Deterministic: single-threaded polars, no randomness.

Run: `uv run python -m src.build_xgb_vs_lstm_signtest`
"""
from __future__ import annotations

import os
from pathlib import Path

# Determinism contract (see tests/test_report_builder_determinism.py): pin the
# polars thread count BEFORE importing polars so outputs are byte-identical.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402

from src.evaluation.significance import sign_test_across_cells  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = CSV_DIR / "xgb_vs_lstm_signtest.csv"

HORIZONS = (1, 3, 5, 10)


def _aggregate_mae(path: Path, baseline: str) -> dict[tuple[str, int], float]:
    """Map (corridor, horizon) -> aggregate-direction MAE for one baseline."""
    df = pl.read_csv(path).filter(
        (pl.col("direction") == "aggregate")
        & (pl.col("metric") == "MAE")
        & (pl.col("baseline") == baseline)
    )
    return {(r["corridor"], r["horizon"]): r["value"] for r in df.iter_rows(named=True)}


def _lstm_aggregate_mae(per_horizon: dict[int, Path]) -> dict[tuple[str, int], float]:
    """Map (corridor, horizon) -> aggregate-direction LSTM MAE, across horizons."""
    out: dict[tuple[str, int], float] = {}
    for h, path in per_horizon.items():
        df = pl.read_csv(path).filter(
            (pl.col("direction") == "aggregate") & (pl.col("metric") == "MAE")
        )
        for r in df.iter_rows(named=True):
            out[(r["corridor"], r["horizon"])] = r["value"]
    return out


def _signtest_row(group: str, corridors: list[str], xgb, lstm) -> dict:
    cells = [(c, h) for c in corridors for h in HORIZONS]
    lstm_losses = [lstm[k] for k in cells]
    xgb_losses = [xgb[k] for k in cells]
    res = sign_test_across_cells(lstm_losses, xgb_losses)
    return {
        "group": group,
        "n_cells": res.n_cells,
        "n_lstm_wins": res.n_model_wins,
        "n_ties": res.n_ties,
        "p_one_sided": res.p_one_sided,
        "p_two_sided": res.p_two_sided,
    }


def build() -> pl.DataFrame:
    xgb = _aggregate_mae(CSV_DIR / "baselines_results_multih.csv", "B5_XGB")
    lstm = _lstm_aggregate_mae(
        {h: CSV_DIR / f"lstm_results_h{h}.csv" for h in HORIZONS}
    )
    xgb_e4 = _aggregate_mae(CSV_DIR / "baselines_E4_results_multih.csv", "B5_XGB")
    lstm_e4 = _lstm_aggregate_mae(
        {h: CSV_DIR / f"lstm_E4_results_h{h}.csv" for h in HORIZONS}
    )

    rows = [
        _signtest_row("E2+E59", ["E2", "E59"], xgb, lstm),
        _signtest_row("E4", ["E4"], xgb_e4, lstm_e4),
    ]
    return pl.DataFrame(rows)


def main() -> None:
    table = build()
    table.write_csv(OUT_CSV)
    print(f"Wrote {OUT_CSV}")
    print(table)


if __name__ == "__main__":
    main()
