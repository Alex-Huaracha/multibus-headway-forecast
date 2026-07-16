"""Ex-ante vs retrospective correlation analysis — O1 anti-circularity artifact.

Tests whether the ex-ante volatility stratifier of §5.2 (sigma of the 12-step
input window) is merely a disguised version of the retrospective regime of
Figure 2, via volatility clustering.

Method:
  - For each corridor (E2=2, E59=59, E4=4) × horizon (3, 5, 10):
      1. Materialize the test set using the same pipeline as build_exante_volatility.
      2. Compute Pearson r and Spearman rho between sigma(input window) and
         |y_real - persistence| (the variable that DEFINES the retrospective regime).
      3. Classify each sample into ex-ante terciles (percentiles 33/66) and into
         retrospective regimes (fixed minute cuts: <1 stable, 1-3 moderate, >=3 high).
      4. Compute the composition of the HIGH ex-ante tercile by retrospective regime,
         plus the lift of the retrospective-high category.

Output:
  docs/resultados/csv-multihorizon/exante_correlation_multihorizon.csv
  9 rows (3 corridors × 3 horizons), columns:
    corridor, horizon, n, pearson_r, spearman_rho, r2,
    frac_highexante_stable, frac_highexante_moderate, frac_highexante_high, lift_high

Reuses prepare_df and materialize_corridor from src.build_exante_volatility.

Usage:
    uv run python src/build_exante_correlation.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.build_exante_volatility import prepare_df, materialize_corridor
from src.evaluation.exante_terciles import TercileThresholds, assign_terciles, compute_frozen_thresholds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "docs" / "resultados" / "recertificado" / "csv-multihorizon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORRIDORS = [("E2", 2), ("E59", 59), ("E4", 4)]
HORIZONS = [3, 5, 10]


# ---------------------------------------------------------------------------
# Pure functions (testable without parquet loading)
# ---------------------------------------------------------------------------


def classify_retro_regime(persist_err: np.ndarray) -> np.ndarray:
    """Classify persistence-error magnitudes into retrospective regime codes.

    Fixed minute cuts (same as Figure 2 / Sección 5.1):
      <1  → 0 (stable)
      1–3 → 1 (moderate)
      >=3 → 2 (high)

    Args:
        persist_err: 1-D array of |y_real - y_persistence| in minutes.

    Returns:
        Integer array of same shape as persist_err with values in {0, 1, 2}.
    """
    labels = np.ones(len(persist_err), dtype=int)  # default: moderate
    labels[persist_err < 1.0] = 0
    labels[persist_err >= 3.0] = 2
    return labels


def compute_exante_terciles(
    ex_ante: np.ndarray, thresholds: TercileThresholds
) -> np.ndarray:
    """Assign ex-ante tercile codes with frozen train+val thresholds.

    Values <= p33 → 0, (p33, p66] → 1, > p66 → 2.

    Args:
        ex_ante: 1-D array of finite ex-ante sigma values.
        thresholds: Frozen thresholds calibrated from train+val values.

    Returns:
        Integer array of same shape with values in {0, 1, 2}.
    """
    return assign_terciles(ex_ante, thresholds)


def compute_lift(retro: np.ndarray, ex_terciles: np.ndarray) -> float:
    """Compute lift of retrospective-high in the high ex-ante tercile.

    lift = P(retro_high | ex_ante_high) / P(retro_high)

    Returns NaN when the marginal P(retro_high) == 0 or when there are no
    high ex-ante samples.
    """
    marginal_high = float(np.mean(retro == 2))
    if marginal_high == 0.0:
        return math.nan
    high_mask = ex_terciles == 2
    if not high_mask.any():
        return math.nan
    conditional_high = float(np.mean(retro[high_mask] == 2))
    return conditional_high / marginal_high


def compute_correlation_stats(
    ex_ante: np.ndarray, persist_err: np.ndarray
) -> dict[str, float | int]:
    """Compute Pearson r, Spearman rho, r^2 and sample count.

    Args:
        ex_ante:     1-D array, ex-ante sigma values.
        persist_err: 1-D array, |y_real - y_persistence| values (same length).

    Returns:
        Dict with keys: pearson_r, spearman_rho, r2, n.
    """
    r, _ = sp_stats.pearsonr(ex_ante, persist_err)
    rho, _ = sp_stats.spearmanr(ex_ante, persist_err)
    return {
        "pearson_r": float(r),
        "spearman_rho": float(rho),
        "r2": float(r * r),
        "n": int(len(ex_ante)),
    }


def build_csv_row(
    corridor: str,
    horizon: int,
    ex_ante_full: np.ndarray,
    persist_err_full: np.ndarray,
    thresholds: TercileThresholds,
) -> dict:
    """Build one CSV row for a corridor × horizon cell.

    Handles non-finite filtering internally before frozen classification.

    Returns a dict with all 10 CSV columns:
      corridor, horizon, n, pearson_r, spearman_rho, r2,
      frac_highexante_stable, frac_highexante_moderate, frac_highexante_high,
      lift_high.
    """
    valid = np.isfinite(ex_ante_full)
    ex = ex_ante_full[valid]
    err = persist_err_full[valid]

    corr = compute_correlation_stats(ex, err)
    ex_terc = compute_exante_terciles(ex, thresholds)
    retro = classify_retro_regime(err)

    # Composition of HIGH ex-ante tercile by retrospective regime
    high_mask = ex_terc == 2
    high_retro = retro[high_mask]
    n_high = len(high_retro)

    if n_high > 0:
        frac_stable = float(np.mean(high_retro == 0))
        frac_moderate = float(np.mean(high_retro == 1))
        frac_high = float(np.mean(high_retro == 2))
    else:
        frac_stable = frac_moderate = frac_high = math.nan

    lift = compute_lift(retro, ex_terc)

    return {
        "corridor": corridor,
        "horizon": horizon,
        "n": corr["n"],
        "pearson_r": corr["pearson_r"],
        "spearman_rho": corr["spearman_rho"],
        "r2": corr["r2"],
        "frac_highexante_stable": frac_stable,
        "frac_highexante_moderate": frac_moderate,
        "frac_highexante_high": frac_high,
        "lift_high": lift,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    import polars as pl

    print("=" * 70)
    print("Ex-ante Correlation Analysis (O1 anti-circularity)")
    print("=" * 70)

    rows = []
    for corridor, empresaid in CORRIDORS:
        print(f"\n=== {corridor} (empresaid={empresaid}) ===")
        df, stats = prepare_df(empresaid)
        for h in HORIZONS:
            print(f"  h={h} ...", end=" ", flush=True)
            _, _, calibration_ex_ante = materialize_corridor(
                df, stats, empresaid, h, splits=("train", "val")
            )
            thresholds = compute_frozen_thresholds(calibration_ex_ante)
            targets, persist, ex_ante = materialize_corridor(df, stats, empresaid, h)
            persist_err = np.abs(targets - persist)
            row = build_csv_row(corridor, h, ex_ante, persist_err, thresholds)
            rows.append(row)
            print(
                f"n={row['n']:,}  "
                f"r={row['pearson_r']:.4f}  "
                f"rho={row['spearman_rho']:.4f}  "
                f"r2={row['r2']:.4f}  "
                f"lift={row['lift_high']:.2f}"
            )

    result_df = pl.DataFrame(rows)
    out_path = OUT_DIR / "exante_correlation_multihorizon.csv"
    result_df.write_csv(out_path)
    print(f"\nResults written to: {out_path}")

    print("\n" + "=" * 70)
    print("CORRELATION TABLE (all 9 corridor×horizon cells)")
    print("=" * 70)
    with pl.Config(tbl_rows=20, tbl_width_chars=120, float_precision=4):
        print(result_df.sort(["corridor", "horizon"]))


if __name__ == "__main__":
    main()
