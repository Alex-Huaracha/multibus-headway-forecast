"""Point values and intervals for every threshold rule in Table 3.

Table 3 counts the cells where a thresholded rule names the same winner as the
comparison with no threshold. A difference whose interval contains zero names no
winner, so the count is taken only over decided cells, and that needs an
interval for each rule. ``detection_ranking_ci.csv`` bounds the threshold-free
AUC and the optimized threshold; this builder bounds the other four rules, LSTM
minus persistence in MCC, with the same day-clustered bootstrap.

The rules are the three of ``build_threshold_denominators`` plus the threshold
in minutes of ``build_threshold_robustness``: a quarter of the median observed
headway of each corridor and direction, taken on the earlier origin. That rule
has no row in ``threshold_denominators.csv``, so the point values Table 3 prints
for every rule (A/E, MCC, overlap with the event of Eq. 4) are carried here too.

The population is the one both source builders share: the residuals of the
scored origin, restricted to vectors of at least three positions.

Output is ``docs/resultados/csv-multihorizon/threshold_rules_ci.csv``.

Usage
-----
    uv run python -m src.build_threshold_rules_ci
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.build_contiguous_significance import CORRIDORS, HORIZONS  # noqa: E402
from src.build_detection_ranking_ci import LEVEL, N_BOOT, SEED  # noqa: E402
from src.build_threshold_denominators import (  # noqa: E402
    CALIBRATION_ORIGIN,
    MODELS,
    RULES,
    SCORING_ORIGIN,
    TRUTH,
    _arm,
    _jaccard,
    _published_event,
    calibration_rates,
    prepared,
)
from src.build_threshold_robustness import absolute_cuts  # noqa: E402
from src.evaluation.ranking_significance import (  # noqa: E402
    clustered_bootstrap_delta,
)
from src.evaluation.vector_metrics import matthews_corrcoef  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "threshold_rules_ci.csv"

# The field's convention, a quarter of a reference headway (Section II-A), with
# the median observed headway standing in for the schedule these corridors lack.
ABSOLUTE_RATIO = 0.25

RULES_CI: tuple[str, ...] = (*RULES, "absolute")


def _mcc(truth: np.ndarray, flag: np.ndarray) -> float:
    return float(matthews_corrcoef(truth, flag.astype(bool)))


def _with_absolute_cut(score: pl.DataFrame) -> pl.DataFrame:
    """Attach the per-(corridor, direction) cut in minutes, fitted on r2."""
    cuts = absolute_cuts(CALIBRATION_ORIGIN, ABSOLUTE_RATIO)
    frame = pl.DataFrame(
        [{"corridor": c, "direction": d, "_cut": k} for (c, d), k in sorted(cuts.items())]
    )
    tagged = score.join(frame, on=["corridor", "direction"], how="left")
    if tagged.get_column("_cut").null_count():
        raise ValueError("a scored direction has no cut in minutes on r2")
    return tagged


def _flags(
    cell: pl.DataFrame, rule: str, value_col: str, rate: float
) -> tuple[pl.Series, pl.Series]:
    """One rule's (event, alarm) flags for one method."""
    if rule == "absolute":
        flags = cell.select(
            (pl.col(TRUTH) < pl.col("_cut")).alias("t"),
            (pl.col(value_col) < pl.col("_cut")).alias("a"),
        )
        return flags.get_column("t"), flags.get_column("a")
    truth, alarm, _, _ = _arm(cell, rule, value_col, rate)
    return truth, alarm


def build() -> pl.DataFrame:
    score = _with_absolute_cut(prepared(SCORING_ORIGIN))
    quota_rates = calibration_rates()

    rows: list[dict] = []
    for rule in RULES_CI:
        for corridor in CORRIDORS:
            for horizon in HORIZONS:
                cell = score.filter(
                    (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
                )
                if cell.height == 0:
                    continue
                rate = quota_rates[(corridor, horizon)]
                flags = {
                    name: _flags(cell, rule, column, rate) for name, column in MODELS
                }
                (learner, (truth, alarm_learner)), (rival, (truth_rival, alarm_rival)) = (
                    flags.items()
                )
                # Every rule builds the event from the observed vector alone, so
                # both methods are scored against the same flags.
                if not truth.equals(truth_rival):
                    raise ValueError(f"{rule} {corridor} h{horizon}: events differ")

                truth_np = truth.to_numpy()
                result = clustered_bootstrap_delta(
                    truth_np,
                    alarm_learner.to_numpy(),
                    alarm_rival.to_numpy(),
                    cell.get_column("target_ts").dt.date().to_numpy(),
                    statistic=_mcc,
                    n_boot=N_BOOT,
                    seed=SEED,
                    level=LEVEL,
                )
                base = float(truth_np.mean())
                row = {"rule": rule, "corridor": corridor, "horizon": int(horizon)}
                for name, (_, alarm) in flags.items():
                    alarm_np = alarm.to_numpy()
                    row[f"ae_{name}"] = float(alarm_np.mean()) / base if base else float("nan")
                    row[f"mcc_{name}"] = _mcc(truth_np, alarm_np)
                row["jaccard"] = _jaccard(truth_np, _published_event(cell).to_numpy())
                row.update({
                    "delta_mcc": result.delta,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "survives": not result.crosses_zero,
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "level": LEVEL,
                })
                rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    frame = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.write_text(frame.write_csv(), encoding="utf-8", newline="")
    print(f"Intervalos escritos en {OUT_CSV.relative_to(REPO_ROOT)}")
    for row in frame.iter_rows(named=True):
        print(f"{row['rule']:<10} {row['corridor']:<4} {row['horizon']:>3}  "
              f"{row['delta_mcc']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
              f"  {'si' if row['survives'] else 'NO'}")


if __name__ == "__main__":
    main()
