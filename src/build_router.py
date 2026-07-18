"""Ex-ante volatility router: a single regime-aware switching policy.

Motivation
----------
No single pure predictor is best across all horizons: persistence (B1) wins at
h=1, the LSTM wins at h>=3. This builder demonstrates that a single policy that
switches between them **using only information available before predicting** —
the volatility of the input window (its standard deviation) — is best-or-tied
against BOTH pure models at every horizon, recovering persistence's short-horizon
edge without sacrificing the LSTM's long-horizon edge.

Leakage discipline (read before trusting the numbers)
-----------------------------------------------------
- The regime is defined by ex-ante input-window volatility, binned into terciles
  whose p33/p66 cutoffs are **frozen on train+val** (never on test), via
  ``compute_frozen_thresholds`` — identical contract to the ex-ante stratification.
- The **policy** (which model to trust in each tercile) is learned on a held-out
  ``POLICY_FRAC`` slice of the test set and the router is scored on the DISJOINT
  remainder. The reported MAE never informed the policy.
- Honest limitation: the policy is learned on a test sub-portion, not on train+val,
  because the Kaggle kernels exported per-sample predictions for the TEST split
  only (no train+val DL predictions are available locally). The evaluation slice is
  therefore ~``1 - POLICY_FRAC`` of the test set, so router MAE levels are not
  directly comparable to the full-test MAEs reported elsewhere in the document.
- DL predictions are joined POSITIONALLY against locally reconstructed targets, so
  every corridor x horizon passes ``verify_alignment`` (same-sample check against the
  residual CSV) as a hard gate before scoring; the observed max abs diff is persisted
  per row so the gate is auditable from the output CSV alone.
- ``mae_oracle`` (per-tercile best chosen ON the eval slice) is reported as an
  upper bound only — it is NOT deployable; the gap router-vs-oracle measures how
  much the held-out policy leaves on the table (here: essentially zero).

Determinism: ``POLARS_MAX_THREADS=1`` and a fixed RNG seed make the split — and
therefore the output CSV — byte-reproducible.

Output: ``docs/resultados/csv-multihorizon/router_multihorizon.csv``
(12 rows: 3 corridors x 4 horizons).
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.build_exante_volatility import (
    ALIGN_TOL,
    RESID_DIR,
    materialize_corridor,
    prepare_df,
    verify_alignment,
)
from src.evaluation.exante_terciles import assign_terciles, compute_frozen_thresholds

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "resultados" / "csv-multihorizon"

HORIZONS = [1, 3, 5, 10]
POLICY_FRAC = 0.6          # share of test used to LEARN the policy (rest is held-out eval)
SEED = 42                  # frozen RNG seed for the leakage-free split
MIN_TERCILE_N = 50         # min policy-slice samples in a tercile; else fall back to DL

# (display name, empresaid, residual-CSV corridor filter, residual path fn)
CORRIDORS = [
    ("E2", 2, "E2", lambda h: RESID_DIR / f"h{h}" / f"lstm_residuals_h{h}.csv"),
    ("E59", 59, "E59", lambda h: RESID_DIR / f"h{h}" / f"lstm_residuals_h{h}.csv"),
    ("E4", 4, None, lambda h: RESID_DIR / f"h{h}" / f"lstm_E4_residuals_h{h}.csv"),
]


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def policy_eval_split(n: int, seed: int, policy_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Split ``n`` sample indices into a policy-learning slice and a disjoint eval slice.

    Extracted so the leakage-critical property — the two slices partition the samples
    with no overlap — is unit-testable without re-running the (slow) builder.
    """
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(policy_frac * n)
    return perm[:cut], perm[cut:]


def evaluate(corridor, empresaid, csv_filter, resid_path_fn, horizon, df, stats) -> dict:
    """Build and score the router for one corridor x horizon (leakage-free)."""
    _, _, calib = materialize_corridor(df, stats, empresaid, horizon, splits=("train", "val"))
    thr = compute_frozen_thresholds(calib)

    y_true, y_persist, ex_ante = materialize_corridor(df, stats, empresaid, horizon)
    csv_path = resid_path_fn(horizon)

    # Hard gate: y_dl is joined POSITIONALLY against the locally reconstructed
    # y_true/y_persist, so a divergence in row order or count would silently pair the
    # wrong DL prediction with the wrong observation. Same audit as the ex-ante builder.
    passed, max_diff_target, max_diff_persist, _ = verify_alignment(
        corridor, horizon, y_true, y_persist, csv_path, csv_corridor_filter=csv_filter,
    )
    if not passed:
        print(f"  HARD GATE FAILED for {corridor} h={horizon} — stopping.")
        sys.exit(1)

    csv_df = pl.read_csv(csv_path)
    if csv_filter is not None:
        csv_df = csv_df.filter(pl.col("corridor") == csv_filter)
    y_dl = csv_df["y_pred_dl"].to_numpy()

    mask = np.isfinite(ex_ante)
    y_true, y_persist, y_dl, ex_ante = y_true[mask], y_persist[mask], y_dl[mask], ex_ante[mask]
    codes = assign_terciles(ex_ante, thr)  # 0=low, 1=mid, 2=high

    pol_idx, ev_idx = policy_eval_split(len(y_true), SEED, POLICY_FRAC)

    # Learn per-tercile policy on the policy slice only.
    policy: dict[int, str] = {}
    for t in range(3):
        sel = pol_idx[codes[pol_idx] == t]
        if len(sel) < MIN_TERCILE_N:
            policy[t] = "dl"  # safe default: DL wins overall at operational horizons
            continue
        use_dl = _mae(y_true[sel], y_dl[sel]) <= _mae(y_true[sel], y_persist[sel])
        policy[t] = "dl" if use_dl else "persist"

    # Score everything on the disjoint eval slice.
    yt, yd, yp, ce = y_true[ev_idx], y_dl[ev_idx], y_persist[ev_idx], codes[ev_idx]
    router_pred = np.where(np.array([policy[c] == "dl" for c in ce]), yd, yp)
    oracle_pred = yp.copy()
    for t in range(3):
        tm = ce == t
        if tm.any() and _mae(yt[tm], yd[tm]) <= _mae(yt[tm], yp[tm]):
            oracle_pred[tm] = yd[tm]

    mae_persist, mae_dl = _mae(yt, yp), _mae(yt, yd)
    mae_router = _mae(yt, router_pred)
    return {
        "corridor": corridor,
        "horizon": horizon,
        "n_eval": int(len(ev_idx)),
        "policy_low_mid_high": "".join({"dl": "D", "persist": "P"}[policy[t]] for t in range(3)),
        "mae_persist": mae_persist,
        "mae_dl": mae_dl,
        "mae_router": mae_router,
        "mae_oracle": _mae(yt, oracle_pred),
        "router_vs_dl": mae_router - mae_dl,          # <= 0: router at least ties always-DL
        "router_vs_persist": mae_router - mae_persist,  # <= 0: router at least ties always-persist
        "policy_frac": POLICY_FRAC,
        "seed": SEED,
        # Persisted so the positional-join gate is auditable from the CSV alone,
        # rather than being a check that merely ran once at build time.
        "align_max_abs_diff": max(max_diff_target, max_diff_persist),
        "align_tolerance": ALIGN_TOL,
    }


def main() -> None:
    print("=" * 70)
    print("Ex-ante Volatility Router")
    print("=" * 70)

    rows: list[dict] = []
    for corridor, empresaid, csv_filter, fn in CORRIDORS:
        print(f"\n=== {corridor} (empresaid={empresaid}) ===")
        df, stats = prepare_df(empresaid)
        for horizon in HORIZONS:
            row = evaluate(corridor, empresaid, csv_filter, fn, horizon, df, stats)
            rows.append(row)
            print(
                f"  h={horizon:<2} policy={row['policy_low_mid_high']} "
                f"persist={row['mae_persist']:.4f} dl={row['mae_dl']:.4f} "
                f"router={row['mae_router']:.4f} (vs DL {row['router_vs_dl']:+.4f})"
            )

    result = pl.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "router_multihorizon.csv"
    result.write_csv(out_path)
    print(f"\nResults written to: {out_path}")

    wins = int((result["router_vs_dl"] <= 1e-9).sum())
    w = result["n_eval"].to_numpy()
    print(f"\nRouter <= always-DL in {wins}/{result.height} cells")
    print(f"Weighted mean router - always-DL:      {np.average(result['router_vs_dl'].to_numpy(), weights=w):+.4f} min")
    print(f"Weighted mean router - always-persist: {np.average(result['router_vs_persist'].to_numpy(), weights=w):+.4f} min")


if __name__ == "__main__":
    main()
