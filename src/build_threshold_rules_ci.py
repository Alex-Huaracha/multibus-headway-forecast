"""Intervals for the verdict of every threshold rule in Table 3.

Table 3 counts the cells where a thresholded rule names the same winner as the
comparison with no threshold. A difference whose interval contains zero names no
winner, so the count is taken only over decided cells, and that needs an
interval for each rule. ``detection_ranking_ci.csv`` bounds the threshold-free
AUC and the optimized threshold; this builder bounds the three rules of
``build_threshold_denominators``, LSTM minus persistence in MCC, with the same
day-clustered bootstrap.

The population and the flags are that builder's own (``prepared`` and
``_arm``), so an interval here bounds exactly the MCC Table 3 prints.

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
    MODELS,
    RULES,
    SCORING_ORIGIN,
    _arm,
    calibration_rates,
    prepared,
)
from src.evaluation.ranking_significance import (  # noqa: E402
    clustered_bootstrap_delta,
)
from src.evaluation.vector_metrics import matthews_corrcoef  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "threshold_rules_ci.csv"


def _mcc(truth: np.ndarray, flag: np.ndarray) -> float:
    return float(matthews_corrcoef(truth, flag.astype(bool)))


def build() -> pl.DataFrame:
    score = prepared(SCORING_ORIGIN)
    quota_rates = calibration_rates()
    (_, learner_col), (_, rival_col) = MODELS

    rows: list[dict] = []
    for rule in RULES:
        for corridor in CORRIDORS:
            for horizon in HORIZONS:
                cell = score.filter(
                    (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
                )
                if cell.height == 0:
                    continue
                rate = quota_rates[(corridor, horizon)]
                truth, alarm_learner, _, _ = _arm(cell, rule, learner_col, rate)
                truth_rival, alarm_rival, _, _ = _arm(cell, rule, rival_col, rate)
                # Every rule builds the event from the observed vector alone, so
                # both methods are scored against the same flags.
                if not truth.equals(truth_rival):
                    raise ValueError(f"{rule} {corridor} h{horizon}: events differ")

                result = clustered_bootstrap_delta(
                    truth.to_numpy(),
                    alarm_learner.to_numpy(),
                    alarm_rival.to_numpy(),
                    cell.get_column("target_ts").dt.date().to_numpy(),
                    statistic=_mcc,
                    n_boot=N_BOOT,
                    seed=SEED,
                    level=LEVEL,
                )
                rows.append({
                    "rule": rule,
                    "corridor": corridor,
                    "horizon": int(horizon),
                    "delta_mcc": result.delta,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "survives": not result.crosses_zero,
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "level": LEVEL,
                })
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
