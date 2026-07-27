"""Cell-level sign test: does the LSTM beat the leveled XGBoost across cells?

What changed, and why it matters
--------------------------------
This script used to read the LSTM MAE from the DL results CSVs and the XGBoost
MAE from the aggregate baselines CSV. Those two numbers describe DIFFERENT sample
populations: the DL metrics live on the DL WINDOW population (cold-start rows
dropped, every target replicated once per anchoring window slot), while the
baseline aggregates cover every test row with a non-null prediction. The measured
framing bias between the two is 0.03-0.25 min per cell — larger than most of the
margins the sign test was counting — so the old 8/8 result was an artifact of the
mismatch, not a finding.

It now reads the RESTRICTED MAEs from ``xgb_paired_dl_metrics.csv``
(``src/build_xgb_paired_metrics.py``), where XGBoost has been re-scored over
exactly the rows the LSTM was scored on via the per-sample NB20 export keyed on
``(direction, t, pair_rank)``, at 100% join coverage. Both halves of every trial
now come from one population, so the coin flip is fair.

Why a sign test at all, now that per-sample pairing IS available
---------------------------------------------------------------
The per-sample paired DM/Wilcoxon test is emitted separately, on the
de-duplicated ``distinct_target`` population
(``xgb_paired_significance.csv``). The sign test survives because it answers a
different, coarser question over the population whose MAE the paper actually
prints: each ``(corridor, horizon)`` cell is one Bernoulli trial, immune to the
~4.5x overlapping-window replication that would make a naive per-sample n on the
matched population wildly anti-conservative.

Three groups are reported: the large corridors (E2+E59), the small one (E4), and
the pooled 12 cells. Pooling is reported because splitting first and pooling only
when convenient is how a null result gets talked into a positive one.

Deterministic: single-threaded polars, no randomness.

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
from src.evaluation.xgb_paired import HORIZONS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
PAIRED_METRICS_CSV = CSV_DIR / "xgb_paired_dl_metrics.csv"
OUT_CSV = CSV_DIR / "xgb_vs_lstm_signtest.csv"

# The population the trials live on: one XGB row per LSTM row, so each cell's MAE
# is directly comparable to the LSTM MAE the paper reports.
POPULATION = "multiplicity_matched"

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("E2+E59", ("E2", "E59")),
    ("E4", ("E4",)),
    ("pooled", ("E2", "E59", "E4")),
)

SIGNTEST_COLUMNS = [
    "group",
    "population",
    "n_cells",
    "n_lstm_wins",
    "n_ties",
    "p_one_sided",
    "p_two_sided",
]


def load_restricted_mae(
    path: Path = PAIRED_METRICS_CSV,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    """Map ``(corridor, horizon)`` to the restricted LSTM and XGBoost MAEs.

    Raises
    ------
    ValueError
        If the paired-metrics CSV is missing (run
        ``src.build_xgb_paired_metrics`` first) or lacks a restricted MAE column.
    """
    if not Path(path).is_file():
        raise ValueError(
            f"load_restricted_mae: missing {path} — run "
            "`uv run python -m src.build_xgb_paired_metrics` first; the sign test "
            "must not fall back to the mismatched aggregate MAEs"
        )
    frame = pl.read_csv(path)
    required = ["corridor", "horizon", "mae_dl_matched", "mae_xgb_matched"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"load_restricted_mae: {Path(path).name} is missing {missing}")

    lstm: dict[tuple[str, int], float] = {}
    xgb: dict[tuple[str, int], float] = {}
    for row in frame.iter_rows(named=True):
        key = (row["corridor"], int(row["horizon"]))
        lstm[key] = float(row["mae_dl_matched"])
        xgb[key] = float(row["mae_xgb_matched"])
    return lstm, xgb


def _signtest_row(
    group: str,
    corridors: tuple[str, ...],
    lstm: dict[tuple[str, int], float],
    xgb: dict[tuple[str, int], float],
) -> dict:
    cells = [(c, h) for c in corridors for h in HORIZONS]
    missing = [cell for cell in cells if cell not in lstm or cell not in xgb]
    if missing:
        raise ValueError(f"_signtest_row: {group} has no restricted MAE for {missing}")
    res = sign_test_across_cells([lstm[k] for k in cells], [xgb[k] for k in cells])
    return {
        "group": group,
        "population": POPULATION,
        "n_cells": res.n_cells,
        "n_lstm_wins": res.n_model_wins,
        "n_ties": res.n_ties,
        "p_one_sided": res.p_one_sided,
        "p_two_sided": res.p_two_sided,
    }


def build(paired_metrics_csv: Path = PAIRED_METRICS_CSV) -> pl.DataFrame:
    """Run the sign test for every group over the restricted per-cell MAEs."""
    lstm, xgb = load_restricted_mae(paired_metrics_csv)
    return pl.DataFrame(
        [_signtest_row(group, corridors, lstm, xgb) for group, corridors in GROUPS]
    ).select(SIGNTEST_COLUMNS)


def main() -> None:
    table = build()
    table.write_csv(OUT_CSV)
    print(f"Wrote {OUT_CSV}")
    print(table)


if __name__ == "__main__":
    main()
