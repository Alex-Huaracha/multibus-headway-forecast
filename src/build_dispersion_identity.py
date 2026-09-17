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

What it finds is the relation holding at r = 0,99 over ratios that span from 0,05
to 0,55. A property of this corpus has no reason to track the error term across
that range, so the compression is the decomposition operating rather than
preprocessing noise. The same columns carry the other reading of the measurement,
which the manuscript owes its reader: the share of the within-vector spread the
forecast reproduces, which falls to a few percent at the longest horizon.

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
    load_lstm,
)
from src.evaluation.vector_metrics import MIN_VECTOR_LEN, VECTOR_KEY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "resultados" / "csv-multihorizon"
OUT_CSV = OUT_DIR / "dispersion_identity.csv"

SCORING_ORIGIN = "main"

TRUTH = "y_true"
FORECAST = "y_pred_model"


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
    rows: list[dict] = []
    for corridor in CORRIDORS:
        residuals = load_lstm(SCORING_ORIGIN).filter(pl.col("corridor") == corridor)
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
            # What the ratio would be if the forecast were a conditional mean,
            # i.e. if its error carried no covariance with it.
            explained = 1.0 - v_err / v_true

            rows.append(
                {
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

    table = pl.DataFrame(rows).sort(["corridor", "horizon"])

    # The manuscript quotes this correlation, so it travels in the table rather
    # than only in this script's console output.
    correlation = float(
        np.corrcoef(
            table.get_column("ratio_measured").to_numpy(),
            table.get_column("explained").to_numpy(),
        )[0, 1]
    )
    return table.with_columns(pl.lit(correlation).alias("r_identity"))


def main() -> None:
    table = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)

    with pl.Config(tbl_rows=20, tbl_cols=10, tbl_width_chars=170, float_precision=4):
        print(f"Variance decomposition on origin {SCORING_ORIGIN!r}\n")
        print(
            table.select(
                "corridor", "horizon", "n_vectors",
                "v_true", "v_pred", "v_err",
                "ratio_measured", "explained", "gap_no_cov",
            )
        )

    print(
        "\nCorrelacion entre la razon medida y la que predice el termino de error: "
        f"{table.get_column('r_identity')[0]:.4f}"
    )
    worst = table.sort("explained").row(0, named=True)
    best = table.sort("explained", descending=True).row(0, named=True)
    print(
        "Dispersion del vector reproducida por el modelo: "
        f"{100.0 * best['explained']:.1f} % en {best['corridor']} h={best['horizon']}, "
        f"{100.0 * worst['explained']:.1f} % en {worst['corridor']} h={worst['horizon']}"
    )
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)} ({table.height} rows)")


if __name__ == "__main__":
    main()
