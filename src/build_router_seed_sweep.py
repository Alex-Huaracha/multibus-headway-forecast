"""Robustness of the ex-ante router to the policy/eval partition.

``src/build_router.py`` learns the switching policy on a ``POLICY_FRAC`` slice of the
test set and scores it on the disjoint remainder. That partition is a random shuffle,
so the obvious objection is that the reported gain is an artifact of one lucky draw.

This builder re-runs the partition under several seeds — reusing a single
materialization per corridor x horizon, which is the expensive part — and records the
learned policy and the gain for every (corridor, horizon, seed). A stable result means
the policy is a property of the volatility regimes, not of the shuffle.

Everything else (frozen train+val terciles, disjoint policy/eval slices, positional-join
alignment gate) is inherited from ``build_router``; see its docstring for the leakage
discipline and the honest limitations, which apply here unchanged.

Output: ``docs/resultados/csv-multihorizon/router_seed_sweep_multihorizon.csv``
(72 rows: 3 corridors x 4 horizons x 6 seeds).
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.build_exante_volatility import materialize_corridor, prepare_df, verify_alignment
from src.build_router import (
    CORRIDORS,
    HORIZONS,
    MIN_TERCILE_N,
    POLICY_FRAC,
    _mae,
    policy_eval_split,
)
from src.evaluation.exante_terciles import assign_terciles, compute_frozen_thresholds

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "resultados" / "csv-multihorizon"

# The frozen seed of build_router plus five arbitrary others; fixed in source so the
# sweep is reproducible and cannot be quietly re-rolled until it looks favourable.
SEEDS = [42, 1, 7, 123, 2024, 31337]


def sweep_cell(corridor, empresaid, csv_filter, path_fn, horizon, df, stats) -> list[dict]:
    """Materialize one corridor x horizon once, then re-partition it under every seed."""
    _, _, calib = materialize_corridor(df, stats, empresaid, horizon, splits=("train", "val"))
    thr = compute_frozen_thresholds(calib)

    y_true, y_persist, ex_ante = materialize_corridor(df, stats, empresaid, horizon)
    csv_path = path_fn(horizon)

    passed, _, _, _ = verify_alignment(
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
    codes = assign_terciles(ex_ante, thr)

    rows = []
    for seed in SEEDS:
        pol_idx, ev_idx = policy_eval_split(len(y_true), seed, POLICY_FRAC)
        policy: dict[int, str] = {}
        for t in range(3):
            sel = pol_idx[codes[pol_idx] == t]
            if len(sel) < MIN_TERCILE_N:
                policy[t] = "dl"
                continue
            use_dl = _mae(y_true[sel], y_dl[sel]) <= _mae(y_true[sel], y_persist[sel])
            policy[t] = "dl" if use_dl else "persist"

        yt, yd, yp, ce = y_true[ev_idx], y_dl[ev_idx], y_persist[ev_idx], codes[ev_idx]
        router_pred = np.where(np.array([policy[c] == "dl" for c in ce]), yd, yp)
        mae_router = _mae(yt, router_pred)
        rows.append({
            "corridor": corridor,
            "horizon": horizon,
            "seed": seed,
            "policy_low_mid_high": "".join(
                {"dl": "D", "persist": "P"}[policy[t]] for t in range(3)
            ),
            "n_eval": int(len(ev_idx)),
            "mae_router": mae_router,
            "router_vs_dl": mae_router - _mae(yt, yd),
            "router_vs_persist": mae_router - _mae(yt, yp),
        })
    return rows


def main() -> None:
    print("=" * 70)
    print(f"Router seed sweep — seeds {SEEDS}")
    print("=" * 70)

    rows: list[dict] = []
    for corridor, empresaid, csv_filter, path_fn in CORRIDORS:
        print(f"\n=== {corridor} (empresaid={empresaid}) ===")
        df, stats = prepare_df(empresaid)
        for horizon in HORIZONS:
            cell = sweep_cell(corridor, empresaid, csv_filter, path_fn, horizon, df, stats)
            rows.extend(cell)
            policies = {r["policy_low_mid_high"] for r in cell}
            gains = [r["router_vs_dl"] for r in cell]
            verdict = "STABLE" if len(policies) == 1 else f"VARIES {sorted(policies)}"
            print(
                f"  h={horizon:<2} policy={sorted(policies)[0]} {verdict:22} "
                f"router_vs_dl [{min(gains):+.4f}, {max(gains):+.4f}]"
            )

    result = pl.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "router_seed_sweep_multihorizon.csv"
    result.write_csv(out_path)
    print(f"\nResults written to: {out_path}")

    print("\n=== weighted aggregate per seed ===")
    for seed in SEEDS:
        s = result.filter(pl.col("seed") == seed)
        w = s["n_eval"].to_numpy()
        print(
            f"seed {seed:<6} vs DL {np.average(s['router_vs_dl'].to_numpy(), weights=w):+.4f}  "
            f"vs persist {np.average(s['router_vs_persist'].to_numpy(), weights=w):+.4f}  "
            f"cells router<=DL {int((s['router_vs_dl'] <= 1e-9).sum())}/{s.height}"
        )

    n_unstable = sum(
        1
        for (_, _), g in result.group_by(["corridor", "horizon"], maintain_order=True)
        if g["policy_low_mid_high"].n_unique() > 1
    )
    print(f"\nCells with a seed-dependent policy: {n_unstable}/12")
    print(f"Worst single cell (router - DL) across all seeds: {result['router_vs_dl'].max():+.6f}")


if __name__ == "__main__":
    main()
