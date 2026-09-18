"""What the compression of Section V-B is made of.

Section V-B measures that the predicted vector carries less dispersion than the
observed one, in the twelve cells and the three origins. That measurement alone
does not say where the compression comes from, and a reviewer has a ready
alternative: this corpus has no published geometry, the direction of travel is
inferred from the traces, and every measurement error enlarges the prediction
error. A noisier target compresses more for reasons that concern the corpus and
not forecasting.

The variance decomposition separates those two readings. Taken across the
positions of one vector, the observed spread splits into the surviving spread,
the error, and their covariance. Where the forecast behaves like a conditional
mean the covariance vanishes, and the compression ratio becomes a function of one
measurable quantity: how large the error is relative to the observed spread. This
builder computes the three terms per cell and reports how far the ratio sits from
that function.

What it finds is the relation holding at r = 0.99 over ratios that span from 0.05
to 0.55. A property of this corpus has no reason to track the error term across
that range, so the compression is the decomposition operating rather than
preprocessing noise. The same columns carry the other reading of the measurement,
which the manuscript owes its reader: the share of the within-vector spread the
forecast reproduces, which falls to a few percent at the longest horizon.

Closing that reading opens a second one: that compressing is what a recurrent
network does, rather than what a forecast fitted by squared error does. The
corollary is a statement about conditional means and says nothing about
architectures, so measuring a single forecaster leaves the general claim resting
on the theorem alone. This builder therefore runs the decomposition over both
forecasters the paper publishes — the recurrent network and the boosted-tree
ensemble — which share a squared-error objective and no inductive bias. They are
scored over the intersection of their residual keys, so the two ratios of a cell
are averages over the same vectors and the comparison is a paired one, as the
manuscript's own contract for A-beats-B claims requires.

The axis is the one Section V-B uses — across the buses of a corridor at one
instant — and not the temporal axis of Patton and Timmermann's corollary. The
population is the same one Section IV-C fixes, so these columns are comparable
with the rest of the table rather than with a separate corpus.
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
from src.evaluation.residual_export import RESIDUAL_KEY_COLUMNS  # noqa: E402
from src.evaluation.vector_metrics import MIN_VECTOR_LEN, VECTOR_KEY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "dispersion_identity.csv"

SCORING_ORIGIN = "main"

# The two forecasters of the recertified line. Persistence is absent on purpose:
# it copies an observed vector forward, so it compresses nothing and would answer
# a different question than the one this table asks.
MODELS = ("lstm", "xgb")

# The one Section V features. The manuscript's range of explained shares is read
# off this model; the other is there to show the range is not its doing.
PUBLISHED_MODEL = "lstm"

TRUTH = "y_true"
FORECAST = "y_pred_model"


def paired_residuals() -> dict[str, pl.DataFrame]:
    """Both forecasters over the vectors they have in common.

    The two exports are written by different kernels and do not agree row for
    row: the boosted-tree one is a few hundred rows shorter out of 1.6 million.
    Left free, each model would then average its ratio over a slightly different
    set of vectors, and a difference between the two could be read as a property
    of the corpus split rather than of the forecast.

    Restricting both to the intersection of the canonical residual key removes
    that reading: for any cell the two rows are built from the same vectors, so
    ``v_true`` is the same observed spread and only the forecast column changes.
    """
    frames = {
        "lstm": load_lstm(SCORING_ORIGIN),
        "xgb": pl.read_csv(XGB_CSV, try_parse_dates=True),
    }
    if set(frames) != set(MODELS):
        raise ValueError(f"loaders {sorted(frames)} do not cover {sorted(MODELS)}")

    shared = frames[MODELS[0]].select(RESIDUAL_KEY_COLUMNS)
    for name in MODELS[1:]:
        shared = shared.join(
            frames[name].select(RESIDUAL_KEY_COLUMNS),
            on=RESIDUAL_KEY_COLUMNS,
            how="inner",
        )
    if shared.height == 0:
        raise ValueError(
            "the two residual exports share no rows; check that both were "
            f"downloaded for origin {SCORING_ORIGIN!r}"
        )

    return {
        name: frame.join(shared, on=RESIDUAL_KEY_COLUMNS, how="inner")
        for name, frame in frames.items()
    }


def variance_terms(frame: pl.DataFrame) -> pl.DataFrame:
    """The three terms of the decomposition, one row per vector.

    The variance is taken over the positions of a single vector and with the
    population divisor, which is the quantity Section V-B's coefficient of
    variation is built on. Vectors shorter than the minimum carry no dispersion
    worth decomposing and are dropped, as they are there.
    """
    lengths = frame.group_by(VECTOR_KEY).agg(pl.len().alias("vector_len"))
    frame = frame.join(lengths, on=VECTOR_KEY, how="inner").filter(
        pl.col("vector_len") >= MIN_VECTOR_LEN
    )
    frame = frame.with_columns((pl.col(TRUTH) - pl.col(FORECAST)).alias("_e"))

    # The means below are sums in floating point, so a different row order is a
    # different last digit. Grouping and sorting deterministically is what makes
    # the emitted CSV byte-identical across runs.
    return (
        frame.group_by(VECTOR_KEY, maintain_order=True)
        .agg(
            pl.col(TRUTH).var(ddof=0).alias("v_true"),
            pl.col(FORECAST).var(ddof=0).alias("v_pred"),
            pl.col("_e").var(ddof=0).alias("v_err"),
            pl.cov(pl.col(FORECAST), pl.col("_e"), ddof=0).alias("c_cross"),
        )
        .drop_nulls()
        .sort(VECTOR_KEY)
    )


def build() -> pl.DataFrame:
    paired = paired_residuals()

    rows: list[dict] = []
    for model in MODELS:
        for corridor in CORRIDORS:
            residuals = paired[model].filter(pl.col("corridor") == corridor)
            for horizon in HORIZONS:
                cell = residuals.filter(pl.col("horizon") == horizon)
                if cell.height == 0:
                    continue

                terms = variance_terms(cell)
                v_true = float(terms.get_column("v_true").mean())
                v_pred = float(terms.get_column("v_pred").mean())
                v_err = float(terms.get_column("v_err").mean())
                c_cross = float(terms.get_column("c_cross").mean())

                measured = v_pred / v_true
                # What the ratio would be if the forecast were a conditional
                # mean, i.e. if its error carried no covariance with it.
                explained = 1.0 - v_err / v_true

                rows.append(
                    {
                        "model": model,
                        "corridor": corridor,
                        "horizon": horizon,
                        "n_vectors": terms.height,
                        "v_true": v_true,
                        "v_pred": v_pred,
                        "v_err": v_err,
                        "c_cross": c_cross,
                        "ratio_measured": measured,
                        "explained": explained,
                        "gap_no_cov": measured - explained,
                    }
                )

    table = pl.DataFrame(rows).sort(["model", "corridor", "horizon"])

    # The manuscript quotes these correlations, so they travel in the table
    # rather than only in this script's console output. One per model: a
    # correlation pooled over both would hide either one failing.
    correlations = {
        model: float(
            np.corrcoef(
                cell.get_column("ratio_measured").to_numpy(),
                cell.get_column("explained").to_numpy(),
            )[0, 1]
        )
        for model in MODELS
        if (cell := table.filter(pl.col("model") == model)).height
    }
    return table.with_columns(
        pl.col("model").replace_strict(correlations).alias("r_identity")
    )


def main() -> None:
    table = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    # Plain ASCII, not the polars repr: its box-drawing characters cannot be
    # encoded by the cp1252 console this project is run from.
    print(f"Variance decomposition on origin {SCORING_ORIGIN!r}\n")
    print(f"{'model':<6}{'corr':<5}{'h':>3}{'n_vec':>8}"
          f"{'v_true':>10}{'v_pred':>9}{'v_err':>10}"
          f"{'ratio':>8}{'expl':>8}{'gap':>8}")
    for row in table.iter_rows(named=True):
        print(f"{row['model']:<6}{row['corridor']:<5}{row['horizon']:>3}"
              f"{row['n_vectors']:>8}"
              f"{row['v_true']:>10.4f}{row['v_pred']:>9.4f}{row['v_err']:>10.4f}"
              f"{row['ratio_measured']:>8.4f}{row['explained']:>8.4f}"
              f"{row['gap_no_cov']:>8.4f}")

    print()
    for model in MODELS:
        cell = table.filter(pl.col("model") == model)
        print(
            f"[{model}] correlacion entre la razon medida y la que predice el "
            f"termino de error: {cell.get_column('r_identity')[0]:.4f}"
        )
        worst = cell.sort("explained").row(0, named=True)
        best = cell.sort("explained", descending=True).row(0, named=True)
        print(
            f"[{model}] dispersion del vector reproducida: "
            f"{100.0 * best['explained']:.1f} % en {best['corridor']} "
            f"h={best['horizon']}, "
            f"{100.0 * worst['explained']:.1f} % en {worst['corridor']} "
            f"h={worst['horizon']}"
        )

    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
