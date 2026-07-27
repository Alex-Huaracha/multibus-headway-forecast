"""Volatility-regime stratification of the DL-vs-persistence gap — Fase 7.

The degradation curve (``degradation.py``) and the paired significance tests
(``significance.py``) show that DL beats persistence by a margin that GROWS with
the horizon. This module describes the mechanism: persistence predicts
``y(t+H) = y(t)``, so it is only wrong when the headway actually moves, and the
DL advantage concentrates in the windows where the headway changed a lot.

We measure the realized change per paired sample as

    headway_change = |y_true - y_pred_persist|

(``y_pred_persist`` is the last observed headway, so its error magnitude IS the
realized change) and bin it into volatility regimes with FIXED minute
thresholds.

Fixed minute edges (not quantiles) are deliberate. They tell two stories at
once: (1) within a horizon, the DL advantage is larger in high-change regimes;
(2) the SHARE of samples in the high-change regime grows with the horizon —
which is exactly why the aggregate persistence error degrades as the horizon
lengthens. Quantile bins would force equal mass per regime and hide (2).

**No inference is reported here, deliberately (audit pending #2).** The regime
is ``|y_true - y_pred_persist|`` — persistence's own error, i.e. one of the two
quantities the loss differential is built from. Conditioning on it and then
testing that differential conditions on the dependent variable, so the verdict
is forced arithmetically: the "high" bin is *defined* as where persistence is
wrong. The 432 p-values this module used to emit were all of that kind and are
gone. What survives is descriptive and honest — the per-regime MAEs and the
mass shift across regimes — and it is the part that carries the mechanism.

For an inferential answer to the same question, stratify by a variable
observable BEFORE the target: ``evaluation/exante_volatility.py`` uses the
dispersion of the input window, which is independent of the outcome, so the
paired tests it feeds are legitimate.

Public API:
    headway_change(df) -> np.ndarray
    assign_volatility_regime(df, edges, labels) -> pl.DataFrame
    volatility_effect_table(df, direction, edges) -> pl.DataFrame
"""
from __future__ import annotations

import numpy as np
import polars as pl

# Default regime labels for 3 bins separated by 2 edges, ordered ascending.
DEFAULT_LABELS = ("low", "moderate", "high")
# Default minute thresholds: <1 min ~ stable, 1-3 min moderate, >3 min high.
# 3 min is roughly the aggregate MAE scale, so ">3" marks a genuinely large move.
DEFAULT_EDGES = (1.0, 3.0)


def headway_change(df: pl.DataFrame) -> np.ndarray:
    """Realized headway change per sample: ``|y_true - y_pred_persist|``.

    Persistence predicts the last observed headway, so the magnitude of its
    error equals how much the headway actually moved over the horizon. This is
    the stratification variable, not a model output.

    Parameters
    ----------
    df:
        Residual frame with ``y_true`` and ``y_pred_persist`` columns.

    Returns
    -------
    np.ndarray of shape ``(n,)`` — non-negative change magnitude in minutes.
    """
    return np.abs(df["y_true"].to_numpy() - df["y_pred_persist"].to_numpy())


def assign_volatility_regime(
    df: pl.DataFrame,
    edges: tuple[float, ...] = DEFAULT_EDGES,
    labels: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Add ``headway_change`` and ``volatility_regime`` columns.

    The regime is assigned by digitizing :func:`headway_change` against
    ``edges``: a sample falls in bin ``k`` when it is ``>= edges[k-1]`` and
    ``< edges[k]`` (the last bin is the open right tail). With the default
    ``edges=(1.0, 3.0)`` a change ``c`` maps to ``low`` (``c < 1``),
    ``moderate`` (``1 <= c < 3``) or ``high`` (``c >= 3``).

    Parameters
    ----------
    df:
        Residual frame (schema of ``significance.load_residuals``).
    edges:
        Ascending minute thresholds; ``len(labels) == len(edges) + 1``.
    labels:
        Regime names, ascending. Defaults to :data:`DEFAULT_LABELS` when
        ``edges`` has length 2, otherwise generated as ``r0, r1, ...``.

    Returns
    -------
    pl.DataFrame — ``df`` with the two columns appended, all rows preserved.

    Raises
    ------
    ValueError
        If ``edges`` is not strictly ascending, or ``labels`` length mismatches.
    """
    if list(edges) != sorted(edges) or len(set(edges)) != len(edges):
        raise ValueError(f"assign_volatility_regime: edges must be strictly ascending, got {edges}")

    if labels is None:
        labels = (
            DEFAULT_LABELS
            if len(edges) == 2
            else tuple(f"r{i}" for i in range(len(edges) + 1))
        )
    if len(labels) != len(edges) + 1:
        raise ValueError(
            f"assign_volatility_regime: need {len(edges) + 1} labels for "
            f"{len(edges)} edges, got {len(labels)}"
        )

    change = headway_change(df)
    # np.digitize with the default right=False: bin index k means edges[k-1] <= c < edges[k].
    bin_idx = np.digitize(change, edges, right=False)
    regime = np.asarray(labels)[bin_idx]

    return df.with_columns(
        pl.Series("headway_change", change),
        pl.Series("volatility_regime", regime),
    )


def volatility_effect_table(
    df: pl.DataFrame,
    direction: str = "aggregate",
    edges: tuple[float, ...] = DEFAULT_EDGES,
    labels: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Per (corridor, horizon, regime) descriptive error summary.

    Each row answers "within this corridor/horizon, on the samples whose headway
    moved by THIS much, what does each model's MAE look like?". Only regimes
    that actually contain samples produce a row.

    Columns: ``corridor, horizon, regime, regime_order, n, share, mean_change,
    mae_persist, mae_dl, delta_mae, dl_better``. ``delta_mae`` is
    ``mae_dl - mae_persist`` (negative => DL better); ``share`` is the regime's
    fraction of the corridor/horizon's samples, which is what makes the mass
    shift across horizons readable. ``regime_order`` is the ascending bin index,
    for stable sorting/plotting independent of the label strings.

    No p-value is returned and none should be computed downstream: the regime is
    a function of persistence's error, so any test of the loss differential
    within it is circular (see the module docstring). ``dl_better`` is the sign
    of a descriptive difference, not a verdict.

    There is no ``metric`` parameter. It only ever selected the loss for the
    removed tests — the effect size was always reported in MAE — so with the
    tests gone the ``"MAE"`` and ``"RMSE"`` variants of a row were byte-identical
    duplicates.

    Parameters
    ----------
    df:
        Long residual frame (``significance.load_residuals``).
    direction:
        ``"aggregate"`` pools both directions; pass an integer (``-1``/``1``) to
        restrict to one travel direction.
    edges, labels:
        Forwarded to :func:`assign_volatility_regime`.

    Returns
    -------
    pl.DataFrame, one row per populated ``(corridor, horizon, regime)``, sorted
    by ``corridor, horizon, regime_order``.

    Raises
    ------
    ValueError
        If no rows match ``direction``.
    """
    if direction == "aggregate":
        work = df
    else:
        work = df.filter(pl.col("direction") == direction)
    if work.height == 0:
        raise ValueError(
            f"volatility_effect_table: no rows for direction={direction!r}"
        )

    work = assign_volatility_regime(work, edges=edges, labels=labels)
    if labels is None:
        labels = (
            DEFAULT_LABELS
            if len(edges) == 2
            else tuple(f"r{i}" for i in range(len(edges) + 1))
        )
    order_of = {label: i for i, label in enumerate(labels)}

    groups = (
        work.select(["corridor", "horizon", "volatility_regime"])
        .unique()
        .sort(["corridor", "horizon", "volatility_regime"])
    )

    # Denominator of `share`: the corridor/horizon population the regimes partition.
    cell_totals = {
        (corridor, int(horizon)): int(height)
        for corridor, horizon, height in work.group_by(["corridor", "horizon"])
        .len()
        .iter_rows()
    }

    rows: list[dict] = []
    for corridor, horizon, regime in groups.iter_rows():
        sub = work.filter(
            (pl.col("corridor") == corridor)
            & (pl.col("horizon") == horizon)
            & (pl.col("volatility_regime") == regime)
        )
        y_true = sub["y_true"].to_numpy()
        mae_persist = float(np.mean(np.abs(y_true - sub["y_pred_persist"].to_numpy())))
        mae_dl = float(np.mean(np.abs(y_true - sub["y_pred_dl"].to_numpy())))
        delta_mae = mae_dl - mae_persist
        rows.append(
            {
                "corridor": corridor,
                "horizon": int(horizon),
                "regime": regime,
                "regime_order": order_of[regime],
                "n": int(sub.height),
                "share": sub.height / cell_totals[(corridor, int(horizon))],
                "mean_change": float(sub["headway_change"].mean()),
                "mae_persist": mae_persist,
                "mae_dl": mae_dl,
                "delta_mae": delta_mae,
                "dl_better": delta_mae < 0,
            }
        )

    return pl.DataFrame(rows).sort(["corridor", "horizon", "regime_order"])
