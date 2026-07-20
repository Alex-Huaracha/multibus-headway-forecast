"""Ex-ante volatility router — TEMPORAL block split (honest stress test).

Why this builder exists
-----------------------
`build_router.py` splits the test set into policy/eval slices with a UNIFORM
permutation. But the samples are overlapping windows (stride 1, sharing 11/12
input steps), so near-twin windows land on BOTH sides of a uniform split. The
slices are disjoint by index but NOT independent, which is why the deployable
router matches the on-eval oracle in 12/12 cells with Δ ≈ 0: the test barely
discriminates.

This builder re-runs the SAME router under a stricter, more honest split: the
EARLIEST ~60 % of the test period learns the policy, the LATEST ~40 % scores it
(`temporal_block_split`). Near-twin windows no longer straddle the boundary
except within a thin seam (~T_in+horizon steps) around the single cut.

INTELLECTUAL-HONESTY CONTRACT (do not violate)
----------------------------------------------
The temporal split MAY make the router look worse: the per-tercile policy can
flip, the router can stop matching the oracle, and the gain can shrink or vanish.
That is an ACCEPTABLE and EXPECTED outcome. This builder reports whatever the
first honest run produces. Do NOT tune, reshuffle, add gaps, or adjust
`POLICY_FRAC` to make the numbers look good. If it degrades, the number stands
and the document says so.

What it adds over `build_router.py`
-----------------------------------
- A temporal (not random) policy/eval split — no RNG, so no seed.
- Per cell, the gain over the TRIVIAL horizon rule (persistence at h=1, DL at
  h>=3), which is the honest benchmark — not just vs always-DL.
- Writes a SEPARATE CSV; it never overwrites `router_multihorizon.csv`, so the
  two bases (uniform vs temporal) can be reported side by side.

Output: ``docs/resultados/csv-multihorizon/router_temporal_multihorizon.csv``
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
    materialize_corridor,
    prepare_df,
    verify_alignment,
)
from src.build_router import (
    CORRIDORS,
    HORIZONS,
    MIN_TERCILE_N,
    POLICY_FRAC,
    _mae,
)
from src.evaluation.exante_terciles import assign_terciles, compute_frozen_thresholds

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "resultados" / "csv-multihorizon"
UNIFORM_CSV = OUT_DIR / "router_multihorizon.csv"


def temporal_block_split(
    ts: np.ndarray, policy_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into an EARLIER policy block and a LATER eval block.

    The cut falls on a timestamp boundary: all samples whose target timestamp is
    strictly before the ~``policy_frac`` cutoff learn the policy, the rest are
    scored. Whole snapshots stay on one side, so no single timestamp straddles
    the seam. Deterministic (no RNG) — the split depends only on the timestamps.

    Raises
    ------
    ValueError
        If either block would be empty (degenerate — a cell with too few distinct
        timestamps to form two temporal blocks).
    """
    n = len(ts)
    order = np.argsort(ts, kind="stable")
    cut = min(max(int(policy_frac * n), 1), n - 1)
    cutoff = ts[order][cut]
    policy_idx = np.nonzero(ts < cutoff)[0]
    eval_idx = np.nonzero(ts >= cutoff)[0]
    if len(policy_idx) == 0 or len(eval_idx) == 0:
        raise ValueError(
            "temporal_block_split: a block is empty — too few distinct timestamps "
            f"(n={n}, cutoff={cutoff}, |policy|={len(policy_idx)}, |eval|={len(eval_idx)})"
        )
    return policy_idx, eval_idx


def evaluate_temporal(
    corridor, empresaid, csv_filter, resid_path_fn, horizon, df, stats
) -> dict:
    """Build and score the router for one corridor x horizon under the temporal split."""
    _, _, calib = materialize_corridor(df, stats, empresaid, horizon, splits=("train", "val"))
    thr = compute_frozen_thresholds(calib)

    y_true, y_persist, ex_ante, ts = materialize_corridor(
        df, stats, empresaid, horizon, return_timestamps=True
    )
    csv_path = resid_path_fn(horizon)

    # Same hard positional-join gate as the uniform builder: y_dl is joined by
    # position, so a row-order/count divergence would silently mispair. The
    # timestamps ride along the SAME arrays, so once the gate passes they are
    # aligned with y_true/y_persist/y_dl too.
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
    y_true, y_persist, y_dl, ex_ante, ts = (
        y_true[mask], y_persist[mask], y_dl[mask], ex_ante[mask], ts[mask],
    )
    codes = assign_terciles(ex_ante, thr)  # 0=low, 1=mid, 2=high

    pol_idx, ev_idx = temporal_block_split(ts, POLICY_FRAC)

    # Learn per-tercile policy on the EARLIER block only.
    policy: dict[int, str] = {}
    for t in range(3):
        sel = pol_idx[codes[pol_idx] == t]
        if len(sel) < MIN_TERCILE_N:
            policy[t] = "dl"  # safe default: DL wins overall at operational horizons
            continue
        use_dl = _mae(y_true[sel], y_dl[sel]) <= _mae(y_true[sel], y_persist[sel])
        policy[t] = "dl" if use_dl else "persist"

    # Score everything on the LATER (disjoint) block.
    yt, yd, yp, ce = y_true[ev_idx], y_dl[ev_idx], y_persist[ev_idx], codes[ev_idx]
    router_pred = np.where(np.array([policy[c] == "dl" for c in ce]), yd, yp)
    oracle_pred = yp.copy()
    for t in range(3):
        tm = ce == t
        if tm.any() and _mae(yt[tm], yd[tm]) <= _mae(yt[tm], yp[tm]):
            oracle_pred[tm] = yd[tm]

    mae_persist, mae_dl = _mae(yt, yp), _mae(yt, yd)
    mae_router = _mae(yt, router_pred)
    # The honest benchmark: the trivial horizon rule (persistence at h=1, DL at
    # h>=3), scored on the SAME eval block. router_vs_trivial <= 0 means the
    # volatility signal adds value beyond simply knowing the horizon.
    mae_trivial = mae_persist if horizon == 1 else mae_dl
    return {
        "corridor": corridor,
        "horizon": horizon,
        "split_kind": "temporal_block",
        "n_policy": int(len(pol_idx)),
        "n_eval": int(len(ev_idx)),
        "policy_low_mid_high": "".join({"dl": "D", "persist": "P"}[policy[t]] for t in range(3)),
        "mae_persist": mae_persist,
        "mae_dl": mae_dl,
        "mae_router": mae_router,
        "mae_oracle": _mae(yt, oracle_pred),
        "mae_trivial": mae_trivial,
        "router_vs_dl": mae_router - mae_dl,
        "router_vs_persist": mae_router - mae_persist,
        "router_vs_trivial": mae_router - mae_trivial,
        "policy_frac": POLICY_FRAC,
        "align_max_abs_diff": max(max_diff_target, max_diff_persist),
        "align_tolerance": ALIGN_TOL,
    }


def main() -> None:
    print("=" * 70)
    print("Ex-ante Volatility Router — TEMPORAL block split")
    print("=" * 70)

    rows: list[dict] = []
    for corridor, empresaid, csv_filter, fn in CORRIDORS:
        print(f"\n=== {corridor} (empresaid={empresaid}) ===")
        df, stats = prepare_df(empresaid)
        for horizon in HORIZONS:
            row = evaluate_temporal(corridor, empresaid, csv_filter, fn, horizon, df, stats)
            rows.append(row)
            print(
                f"  h={horizon:<2} policy={row['policy_low_mid_high']} "
                f"router={row['mae_router']:.4f} (vs DL {row['router_vs_dl']:+.4f}, "
                f"vs trivial {row['router_vs_trivial']:+.4f})"
            )

    result = pl.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "router_temporal_multihorizon.csv"
    result.write_csv(out_path)
    print(f"\nResults written to: {out_path}")

    w = result["n_eval"].to_numpy()
    ties_dl = int((result["router_vs_dl"] <= 1e-9).sum())
    beats_trivial = int((result["router_vs_trivial"] < -1e-9).sum())
    print(f"\nTemporal split — router <= always-DL in {ties_dl}/{result.height} cells")
    print(f"Temporal split — router BEATS the trivial rule in {beats_trivial}/{result.height} cells")
    print(f"Weighted mean router - always-DL:   {np.average(result['router_vs_dl'].to_numpy(), weights=w):+.4f} min")
    print(f"Weighted mean router - trivial rule:{np.average(result['router_vs_trivial'].to_numpy(), weights=w):+.4f} min")

    # Side-by-side with the uniform base (for reporting only — never overwritten).
    if UNIFORM_CSV.exists():
        uni = pl.read_csv(UNIFORM_CSV)
        uni_gain = np.average(uni["router_vs_dl"].to_numpy(), weights=uni["n_eval"].to_numpy())
        print(f"\n[reference] uniform-split router - always-DL: {uni_gain:+.4f} min "
              f"(router_multihorizon.csv, {uni.height} cells)")


if __name__ == "__main__":
    main()
