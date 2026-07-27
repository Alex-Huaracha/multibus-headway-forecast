"""Ex-ante volatility router on the retrained pipeline.

The question the router answers is operational rather than architectural: given
that no pure predictor wins everywhere, can a dispatcher pick one **per sample**
using only what it knows before predicting?

Audit pending #9 judged the previous router oversized: it bought -0.018 min
(about one second) over the trivial rule, in 3 of 12 cells, below the spread
between random seeds, and 9 of its 12 learned policies were degenerate — a
constant policy is not a router, it *is* one of the pure models. That verdict was
reached on the old pipeline and against two candidates.

Two things changed. The ex-ante stratification (``build_contiguous_volatility``)
showed the volatility tercile separating the winners far more sharply than the
horizon does — in E4 h=3 persistence wins the calm tercile by +0.370 and loses
the volatile one by -0.451, a 0.82 min spread inside a cell whose aggregate is
-0.064. And XGBoost is now a levelled third candidate. So the router is rebuilt
and re-measured rather than assumed to still be marginal.

Nothing here is tuned to make it look good. The gain is reported against the best
pure model **chosen on the same evaluation slice**, which is the hardest baseline
available; policies are flagged when degenerate; and the seed sweep exists so the
gain can be read against its own noise instead of beside it.

Leakage discipline
------------------
- The regime is the ex-ante input-window dispersion, binned with p33/p66 frozen
  on **train+val** (``build_contiguous_volatility.calibrate``).
- The **policy** — which model to trust in each tercile — is learned on one slice
  of test and scored on the DISJOINT remainder. The reported MAE never informed
  the policy that produced it.
- Honest limitation, unchanged from the previous router: the policy is learned on
  a test sub-portion rather than on train+val, because the kernels export
  per-sample predictions for the test split only. Router MAEs are therefore over a
  fraction of test and are not comparable to the full-test MAEs elsewhere.

Split modes
-----------
``temporal``
    Learn on the first ``TEMPORAL_LEARN_DAYS`` service days of test, score on the
    rest. This is the only mode that mirrors deployment, where the policy can
    only have been fitted on the past. It is the mode to quote.
``random``
    A seeded random partition, repeated over ``SEEDS``. Its purpose is to
    quantify how much the gain moves with the split — pending #9's actual
    objection — not to produce a headline.

Output
------
``docs/resultados/csv-multihorizon/contiguous_router.csv``
    One row per corridor x horizon x split mode (temporal plus one per seed).

Usage
-----
    uv run python -m src.build_contiguous_router
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_significance import (  # noqa: E402
    CORRIDORS,
    HORIZONS,
    XGB_CSV,
    load_lstm,
)
from src.build_contiguous_volatility import (  # noqa: E402
    CORRIDOR_IDS,
    TERCILE_NAMES,
    assign_regime,
    corridor_max_N,
    paired_cell,
    prepare,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon" / "contiguous_router.csv"
)

# Candidate predictors, in the order their initial appears in a policy string.
CANDIDATES: tuple[tuple[str, str], ...] = (
    ("P", "y_pred_persist"),
    ("D", "y_pred_model"),
    ("X", "y_pred_xgb"),
)

POLICY_FRAC = 0.6         # share of test used to LEARN the policy in random mode
TEMPORAL_LEARN_DAYS = 13  # first 13 of the test window's 22 service days (~0.6)
SEEDS: tuple[int, ...] = tuple(range(20))
MIN_TERCILE_N = 50        # below this the tercile has no reliable winner


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).mean())


def learn_policy(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    codes: np.ndarray,
    learn: np.ndarray,
) -> tuple[str, ...]:
    """Best candidate per tercile, measured on the learning slice only.

    A tercile with too few learning samples falls back to persistence: the
    fallback has to be a fixed rule rather than the locally best model, or the
    thin bins would be decided by the evaluation data through the back door.
    """
    policy: list[str] = []
    for code in range(len(TERCILE_NAMES)):
        sel = learn & (codes == code)
        if int(sel.sum()) < MIN_TERCILE_N:
            policy.append("P")
            continue
        errors = {name: _mae(y_true[sel], preds[name][sel]) for name, _ in CANDIDATES}
        policy.append(min(errors, key=errors.get))
    return tuple(policy)


def apply_policy(
    preds: dict[str, np.ndarray], codes: np.ndarray, policy: tuple[str, ...]
) -> np.ndarray:
    """Route each sample to the model its tercile's policy selected."""
    routed = np.empty_like(preds["P"])
    for code, choice in enumerate(policy):
        sel = codes == code
        routed[sel] = preds[choice][sel]
    return routed


def temporal_masks(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Learn on the earliest service days, evaluate on the later ones."""
    ordered = np.unique(days)
    cutoff = ordered[min(TEMPORAL_LEARN_DAYS, ordered.size) - 1]
    learn = days <= cutoff
    return learn, ~learn


def random_masks(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    learn = np.zeros(n, dtype=bool)
    learn[rng.permutation(n)[: int(round(n * POLICY_FRAC))]] = True
    return learn, ~learn


def score(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    codes: np.ndarray,
    learn: np.ndarray,
    evaluate: np.ndarray,
    *,
    corridor: str,
    horizon: int,
    split_mode: str,
    seed: int | None,
) -> dict:
    """One router row: the policy, what it scored, and what it beat."""
    policy = learn_policy(y_true, preds, codes, learn)
    y_eval = y_true[evaluate]
    routed = apply_policy(preds, codes, policy)[evaluate]

    pure = {
        name: _mae(y_eval, preds[name][evaluate]) for name, _ in CANDIDATES
    }
    best_pure_name = min(pure, key=pure.get)
    mae_router = _mae(y_eval, routed)

    # Upper bound: the per-tercile winner chosen ON the evaluation slice. Not
    # deployable — it is reported so the gap to the honest router is visible.
    oracle = np.empty_like(y_eval)
    eval_codes = codes[evaluate]
    for code in range(len(TERCILE_NAMES)):
        sel = eval_codes == code
        if not sel.any():
            continue
        errors = {
            name: _mae(y_eval[sel], preds[name][evaluate][sel])
            for name, _ in CANDIDATES
        }
        oracle[sel] = preds[min(errors, key=errors.get)][evaluate][sel]

    return {
        "corridor": corridor,
        "horizon": horizon,
        "split_mode": split_mode,
        "seed": seed,
        "n_learn": int(learn.sum()),
        "n_eval": int(evaluate.sum()),
        "policy": "".join(policy),
        "policy_degenerate": len(set(policy)) == 1,
        "mae_router": mae_router,
        "mae_persist": pure["P"],
        "mae_lstm": pure["D"],
        "mae_xgb": pure["X"],
        "mae_oracle": _mae(y_eval, oracle),
        "best_pure": best_pure_name,
        # Negative = the router beats the best pure model on the same slice.
        "gain_vs_best_pure": mae_router - pure[best_pure_name],
        "gap_to_oracle": mae_router - _mae(y_eval, oracle),
    }


def build() -> pl.DataFrame:
    lstm = load_lstm()
    xgb = pl.read_csv(XGB_CSV, try_parse_dates=True)

    rows: list[dict] = []
    for corridor in CORRIDORS:
        df = prepare(CORRIDOR_IDS[corridor])
        max_N = corridor_max_N(df)

        for horizon in HORIZONS:
            cell = paired_cell(
                lstm, xgb, df, corridor=corridor, horizon=horizon, max_N=max_N
            )
            if cell is None:
                continue
            joined, thresholds = cell
            work, _std, codes = assign_regime(joined, thresholds)
            if work.height == 0:
                continue

            y_true = work.get_column("y_true").to_numpy()
            preds = {
                name: work.get_column(column).to_numpy()
                for name, column in CANDIDATES
            }
            days = work.get_column("target_ts").dt.date().to_numpy()

            learn, evaluate = temporal_masks(days)
            rows.append(
                score(
                    y_true, preds, codes, learn, evaluate,
                    corridor=corridor, horizon=horizon,
                    split_mode="temporal", seed=None,
                )
            )
            for seed in SEEDS:
                learn, evaluate = random_masks(work.height, seed)
                rows.append(
                    score(
                        y_true, preds, codes, learn, evaluate,
                        corridor=corridor, horizon=horizon,
                        split_mode="random", seed=seed,
                    )
                )

    return pl.DataFrame(rows).sort(["corridor", "horizon", "split_mode", "seed"])


def seed_sweep_summary(table: pl.DataFrame) -> pl.DataFrame:
    """Per cell: the temporal gain beside the spread the random splits produce.

    Pending #9's objection in one table — a gain smaller than its own seed spread
    is not a finding, whatever its sign.
    """
    temporal = table.filter(pl.col("split_mode") == "temporal").select(
        ["corridor", "horizon",
         pl.col("policy").alias("policy_temporal"),
         pl.col("gain_vs_best_pure").alias("gain_temporal")]
    )
    random_stats = (
        table.filter(pl.col("split_mode") == "random")
        .group_by(["corridor", "horizon"])
        .agg(
            pl.col("gain_vs_best_pure").median().alias("gain_random_median"),
            pl.col("gain_vs_best_pure").min().alias("gain_random_min"),
            pl.col("gain_vs_best_pure").max().alias("gain_random_max"),
            pl.col("policy").n_unique().alias("n_distinct_policies"),
            pl.col("policy_degenerate").mean().alias("degenerate_frac"),
        )
        .with_columns(
            (pl.col("gain_random_max") - pl.col("gain_random_min")).alias(
                "gain_random_spread"
            )
        )
    )
    return (
        temporal.join(random_stats, on=["corridor", "horizon"], how="inner")
        .with_columns(
            (pl.col("gain_temporal").abs() > pl.col("gain_random_spread")).alias(
                "exceeds_seed_noise"
            ),
            # A policy that helps under a random split and hurts under a temporal
            # one has not generalized forward in time — the only direction that
            # matters for deployment. Worth surfacing separately from magnitude.
            (
                (pl.col("gain_temporal") > 0) & (pl.col("gain_random_median") < 0)
            ).alias("fails_forward_in_time"),
        )
        .sort(["corridor", "horizon"])
    )


def main() -> None:
    table = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=200):
        print("\nTemporal split (the deployable one):")
        print(
            table.filter(pl.col("split_mode") == "temporal").select(
                ["corridor", "horizon", "n_eval", "policy", "policy_degenerate",
                 "mae_persist", "mae_lstm", "mae_xgb", "mae_router", "mae_oracle",
                 "best_pure", "gain_vs_best_pure"]
            )
        )
        print("\nGain against seed noise:")
        print(seed_sweep_summary(table))

    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
