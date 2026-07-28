"""Does the LSTM-vs-persistence finding survive outside the published window?

Every published number rests on ONE test window of 22 days (February 2024). The
obvious objection is that the finding is a property of that month. The rolling
origins answer it by re-running the whole protocol — winsorization, population
gate, training, export — at two earlier origins, and asking whether the same
comparison lands the same way.

    r1    test 2023-12-23 .. 2024-01-13
    r2    test 2024-01-14 .. 2024-02-04
    main  test 2024-02-08 .. 2024-02-29   (the published window)

Only LSTM vs persistence
------------------------
The XGBoost baseline is NOT re-run at the rolling origins, so the two
comparisons that need it are unavailable here. That is not a gap: persistence
travels inside the LSTM's own residual export (``y_pred_persist``, same rows,
same key), so this comparison is paired by construction at every origin — no
join, no coverage loss, nothing to reconcile. And it is the headline claim.

Why `main` here differs slightly from the published table
---------------------------------------------------------
``build_contiguous_significance`` inner-joins the LSTM residuals against the
XGBoost export first, so ALL of its comparisons — including LSTM vs persistence
— are scored on the LSTM∩XGB population (coverage 99.95–99.998%). This builder
has no XGBoost to join against at the rolling origins, so it scores the FULL
LSTM population at every origin, `main` included.

That is the right trade: comparability ACROSS origins is what this table exists
for, and it would be lost if one origin were restricted and two were not. The
cost is that `main` here is not byte-comparable with the published table — e.g.
E2·h=1 reads n=90480, p=0.0638 instead of n=90469, p=0.0619, an 11-row
difference. Quote the published table for the published window; quote this one
only for the comparison between windows.

What the table answers
----------------------
Per (corridor, horizon) it reports the sign of ``delta_mae`` at each origin,
where negative means the LSTM wins. ``agrees`` is the question of this whole
exercise: whether the three origins put the win on the same side. A cell that
flips is not necessarily a defect — it is the honest width of the claim, and it
belongs in threats-to-validity rather than being smoothed away.

Usage
-----
    uv run python -m src.build_rolling_origin_significance
"""
from __future__ import annotations

import os

# Byte-identical output across runs (CLAUDE.md determinism contract).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from src.build_contiguous_significance import (  # noqa: E402
    CORRIDORS,
    HORIZONS,
    METRICS,
    load_lstm,
    verdict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "rolling_origin_significance.csv"
)
OUT_SUMMARY_CSV = (
    REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
    / "rolling_origin_agreement.csv"
)

# Chronological, published origin last: the rolling ones are earlier windows.
ORIGINS = ("r1", "r2", "main")


def build() -> pl.DataFrame:
    """One row per (origin, corridor, horizon, metric)."""
    rows: list[dict] = []
    for origin in ORIGINS:
        lstm = load_lstm(origin)
        for corridor in CORRIDORS:
            for horizon in HORIZONS:
                cell = lstm.filter(
                    (pl.col("corridor") == corridor)
                    & (pl.col("horizon") == horizon)
                )
                if cell.height == 0:
                    continue

                y_true = cell.get_column("y_true").to_numpy()
                dl = cell.get_column("y_pred_model").to_numpy()
                pe = cell.get_column("y_pred_persist").to_numpy()
                # Service day of the TARGET, matching the published builder.
                day = cell.get_column("target_ts").dt.date().to_numpy()

                for metric in METRICS:
                    rows.append(
                        {
                            "origin": origin,
                            "corridor": corridor,
                            "horizon": horizon,
                            "metric": metric,
                            "comparison": "LSTM_vs_PERSIST",
                            **verdict(
                                y_true, dl, pe, day,
                                metric=metric, horizon=horizon,
                            ),
                        }
                    )

    return pl.DataFrame(rows).sort(["metric", "corridor", "horizon", "origin"])


def agreement(table: pl.DataFrame) -> pl.DataFrame:
    """Per (corridor, horizon): does every origin put the win on the same side?

    Reported on MAE, the effect size the paper quotes.
    """
    mae = table.filter(pl.col("metric") == "MAE")
    rows: list[dict] = []
    for corridor in CORRIDORS:
        for horizon in HORIZONS:
            cell = mae.filter(
                (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
            )
            if cell.height == 0:
                continue
            by_origin = {r["origin"]: r for r in cell.iter_rows(named=True)}
            winners = {o: ("LSTM" if r["delta_mae"] < 0 else "PERSIST")
                       for o, r in by_origin.items()}
            row = {"corridor": corridor, "horizon": horizon}
            for origin in ORIGINS:
                r = by_origin.get(origin)
                row[f"delta_mae_{origin}"] = None if r is None else r["delta_mae"]
                row[f"winner_{origin}"] = winners.get(origin)
                row[f"dm_p_{origin}"] = None if r is None else r["dm_p_clustered"]
            row["agrees"] = len(set(winners.values())) == 1
            row["winner"] = (
                next(iter(set(winners.values()))) if row["agrees"] else "SPLIT"
            )
            rows.append(row)
    return pl.DataFrame(rows).sort(["corridor", "horizon"])


def main() -> None:
    table = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    summary = agreement(table)
    summary.write_csv(OUT_SUMMARY_CSV)

    with pl.Config(tbl_rows=100, tbl_cols=14, tbl_width_chars=220):
        print(summary.select(
            ["corridor", "horizon", "winner_r1", "winner_r2", "winner_main",
             "delta_mae_r1", "delta_mae_r2", "delta_mae_main", "agrees"]
        ))

    n_agree = int(summary.get_column("agrees").sum())
    print(f"\n{n_agree}/{summary.height} celdas coinciden en los tres origenes")


if __name__ == "__main__":
    main()
