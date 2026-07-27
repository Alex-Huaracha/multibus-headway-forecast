"""Window-contiguity audit: is the reported horizon the realized horizon?

``make_window_index`` (``src/data/windowing.py``) slices each slot's rows by
POSITION and never checks that consecutive positions are consecutive minutes. A
slot's timestamp list is discontinuous at every day boundary, every trip cut
(``GAP_CUT_SECONDS = 30*60``) and every fleet-size dip, so the nominal horizon is
a ROW OFFSET, not a TIME OFFSET. Whenever a window straddles such a gap, its
realized horizon is larger — sometimes by hours.

This matters because the paper is organised around the horizon axis. Part of the
observed degradation is increasing window contamination rather than increasing
forecast difficulty, and the two are not separable from the aggregate numbers.

This builder splits every (corridor, horizon) cell into

  * NOMINAL      — realized horizon == nominal horizon; the label is true
  * NON-NOMINAL  — the window straddles a gap; the realized horizon is larger

and reports persistence-vs-LSTM MAE on each, so the horizon claim can be read off
the subset where the axis means what it says. It does NOT change the windowing
code and requires no retraining: fixing the root cause would mean regenerating
and re-running the six DL notebook families on GPU.

Usage:
    uv run python -m src.build_contiguity_audit

Output (docs/resultados/csv-multihorizon/):
    contiguity_audit_multihorizon.csv   — 12 rows, one per (corridor, horizon)

The filename deliberately avoids the substrings ``_results_``, ``_residuals_h``
and ``_multiseed_``, which are globbed by build_degradation_curve,
build_significance_table and evaluation/multiseed respectively.

Every cell is gated on ``verify_alignment`` (row count + max|Δ| < ALIGN_TOL)
against the committed DL residual CSV before any number is emitted; a cell that
fails the gate raises rather than being silently skipped.
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_exante_volatility import (  # noqa: E402
    ALIGN_TOL,
    materialize_corridor,
    prepare_df,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESID_DIR = REPO_ROOT / "docs" / "resultados" / "residuos-multihorizon" / "11-lstm"
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "contiguity_audit_multihorizon.csv"

CORRIDORS: dict[str, int] = {"E2": 2, "E59": 59, "E4": 4}
HORIZONS: tuple[int, ...] = (1, 3, 5, 10)
# A window is nominal when its realized horizon equals the nominal one. The
# tolerance is numerical only: timestamps are on a 1-minute grid.
NOMINAL_ATOL = 1e-6


def residual_path(corridor: str, horizon: int) -> Path:
    """Committed LSTM residual CSV for one cell (E4 lives in its own file)."""
    stem = "lstm_E4_residuals" if corridor == "E4" else "lstm_residuals"
    return RESID_DIR / f"h{horizon}" / f"{stem}_h{horizon}.csv"


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def audit_cell(corridor: str, empresaid: int, horizon: int,
               df: pl.DataFrame, stats) -> dict:
    """One (corridor, horizon) row. Raises when the alignment gate fails."""
    targets, persist, _ex, effective = materialize_corridor(
        df, stats, empresaid, horizon, splits=("test",),
        return_effective_horizon=True,
    )

    path = residual_path(corridor, horizon)
    if not path.exists():
        raise FileNotFoundError(
            f"DL residuals missing for {corridor} h={horizon}: {path}. "
            "Download them from Kaggle before running this audit."
        )
    res = pl.read_csv(path).filter(
        (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
    )

    # --- alignment gate ------------------------------------------------------
    if res.height != len(targets):
        raise ValueError(
            f"{corridor} h={horizon}: row count mismatch — "
            f"CSV={res.height}, reconstructed={len(targets)}"
        )
    y_true = res.get_column("y_true").to_numpy()
    y_persist = res.get_column("y_pred_persist").to_numpy()
    d_target = float(np.abs(y_true - targets).max())
    d_persist = float(np.abs(y_persist - persist).max())
    if max(d_target, d_persist) >= ALIGN_TOL:
        raise ValueError(
            f"{corridor} h={horizon}: alignment gate FAILED — "
            f"max|Δtarget|={d_target:.3e}, max|Δpersist|={d_persist:.3e}, "
            f"tolerance={ALIGN_TOL}"
        )
    y_dl = res.get_column("y_pred_dl").to_numpy()

    nominal = np.isclose(effective, float(horizon), atol=NOMINAL_ATOL)
    non_nominal = ~nominal
    n = int(len(effective))
    n_nominal = int(nominal.sum())

    row = {
        "corridor": corridor,
        "horizon": horizon,
        "n": n,
        "n_nominal": n_nominal,
        "n_non_nominal": n - n_nominal,
        "pct_non_nominal": round(100.0 * (n - n_nominal) / n, 4),
        "effective_horizon_mean": round(float(effective.mean()), 4),
        "effective_horizon_p99": round(float(np.percentile(effective, 99)), 4),
        "effective_horizon_max": round(float(effective.max()), 4),
        "align_max_abs_diff_target": d_target,
        "align_max_abs_diff_persist": d_persist,
        "align_tol": ALIGN_TOL,
        "mae_persist_all": round(_mae(y_true, y_persist), 6),
        "mae_dl_all": round(_mae(y_true, y_dl), 6),
    }
    row["delta_all"] = round(row["mae_dl_all"] - row["mae_persist_all"], 6)
    row["dl_better_all"] = bool(row["delta_all"] < 0)

    for label, mask in (("nominal", nominal), ("non_nominal", non_nominal)):
        if not mask.any():
            row |= {f"mae_persist_{label}": None, f"mae_dl_{label}": None,
                    f"delta_{label}": None, f"dl_better_{label}": None}
            continue
        mp = _mae(y_true[mask], y_persist[mask])
        md = _mae(y_true[mask], y_dl[mask])
        row |= {
            f"mae_persist_{label}": round(mp, 6),
            f"mae_dl_{label}": round(md, 6),
            f"delta_{label}": round(md - mp, 6),
            f"dl_better_{label}": bool(md - mp < 0),
        }
    # The headline consequence: does restricting to honest windows flip the cell?
    row["verdict_flips_on_nominal"] = bool(
        row["dl_better_all"] is not row["dl_better_nominal"]
    )
    return row


def build(out_csv: Path = OUT_CSV) -> Path:
    """Audit all 12 cells and write the tidy table. Returns the output path."""
    rows: list[dict] = []
    for corridor, empresaid in CORRIDORS.items():
        df, stats = prepare_df(empresaid)
        for horizon in HORIZONS:
            row = audit_cell(corridor, empresaid, horizon, df, stats)
            rows.append(row)
            print(
                f"{corridor:>4} h={horizon:>2}: n={row['n']:>9,} "
                f"non-nominal={row['pct_non_nominal']:6.2f}% "
                f"eff_h_mean={row['effective_horizon_mean']:7.2f} "
                f"| delta_all={row['delta_all']:+.4f} "
                f"delta_nominal={row['delta_nominal']:+.4f}"
                f"{'  <-- VERDICT FLIPS' if row['verdict_flips_on_nominal'] else ''}",
                flush=True,
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).sort(["corridor", "horizon"]).write_csv(out_csv)
    return out_csv


if __name__ == "__main__":
    out = build()
    print(f"\nAuditoría de contigüidad escrita en {out}")
