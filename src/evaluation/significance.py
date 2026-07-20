"""Paired statistical significance for the DL-vs-persistence claim — Fase 6.5.

The degradation curve (``degradation.py``) shows DL beating persistence at long
horizons using AGGREGATED MAE/RMSE. A reviewer will ask: is that gap real, or
sampling noise? This module answers that with paired tests over the PER-SAMPLE
errors exported by NB11/12/13 (schema
``corridor, direction, horizon, y_true, y_pred_dl, y_pred_persist``).

Both models predict the SAME target on the SAME paired window, so the test is on
the per-sample loss differential
    d_i = loss(y_true_i, y_pred_dl_i) - loss(y_true_i, y_pred_persist_i)
A negative mean differential => DL has lower loss => DL beats persistence.

Two complementary tests:
  - Diebold-Mariano: t-like statistic on mean(d), but with a HAC / Newey-West
    long-run variance so serial autocorrelation in the differenced series does
    not inflate significance (forecast errors on consecutive minutes correlate).
  - Wilcoxon signed-rank: distribution-free paired test, robust to the heavy
    tails of headway errors; a sanity check on the parametric DM result.

CAVEAT — with n ~ millions both p-values collapse to ~0 regardless of how small
the gap is. The paper argues from the EFFECT SIZE (``delta_mae``), with the
p-values only confirming the sign is not noise. ``significance_table`` reports
the effect size alongside the p-values for exactly this reason.

Public API:
    load_residuals(residuals_dir) -> pl.DataFrame
    loss_differential(df, metric) -> np.ndarray
    diebold_mariano(d, lag) -> DMResult
    wilcoxon_signed_rank(d) -> float
    significance_table(df, metric, direction) -> pl.DataFrame
    sign_test_across_cells(model_losses, reference_losses) -> SignTestResult
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

REQUIRED_COLUMNS = [
    "corridor",
    "direction",
    "horizon",
    "y_true",
    "y_pred_dl",
    "y_pred_persist",
]


@dataclass(frozen=True)
class DMResult:
    """Outcome of a Diebold-Mariano test.

    Attributes
    ----------
    stat:
        The DM statistic ``mean(d) / sqrt(HAC_var(mean(d)))``. Negative when the
        DL model has the lower loss.
    p_value:
        Two-sided p-value under the asymptotic N(0, 1) null.
    mean_diff:
        Mean loss differential ``mean(d)`` — the raw (signed) effect.
    lag:
        Newey-West truncation lag used for the HAC variance (0 == iid).
    """

    stat: float
    p_value: float
    mean_diff: float
    lag: int


def load_residuals(
    residuals_dir: str | Path, pattern: str = "*.csv"
) -> pl.DataFrame:
    """Concatenate every matching per-sample residual CSV into one validated frame.

    Parameters
    ----------
    residuals_dir:
        Directory holding the residual CSVs downloaded from Kaggle (NB11/12/13).
    pattern:
        Glob selecting which CSVs to load. Defaults to ``"*.csv"``. The Kaggle
        download leaves both ``*_residuals_*.csv`` and the aggregate-schema
        ``*_results_*.csv`` in the same folder, so pass ``"*_residuals_*.csv"``
        to load only the per-sample files and skip the incompatible results CSV.

    Returns
    -------
    pl.DataFrame with columns ``REQUIRED_COLUMNS``.

    Raises
    ------
    ValueError
        If no CSV matches ``pattern``, or any matching CSV lacks a required column.
    """
    residuals_dir = Path(residuals_dir)
    csv_paths = sorted(residuals_dir.glob(pattern))
    if not csv_paths:
        raise ValueError(
            f"load_residuals: no CSV matching {pattern!r} found in {residuals_dir}"
        )

    frames: list[pl.DataFrame] = []
    for path in csv_paths:
        frame = pl.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"load_residuals: {path.name} has wrong schema — missing {missing}"
            )
        frames.append(frame.select(REQUIRED_COLUMNS))

    return pl.concat(frames, how="vertical")


def loss_differential(df: pl.DataFrame, metric: str = "MAE") -> np.ndarray:
    """Per-sample loss differential ``d_i = loss_dl_i - loss_persist_i``.

    ``metric="MAE"`` uses absolute error ``|y_true - y_pred|``; ``metric="RMSE"``
    uses squared error ``(y_true - y_pred)**2`` (the per-sample loss whose mean is
    the MSE behind RMSE). Negative entries mark samples where DL beats persistence.

    Parameters
    ----------
    df:
        Frame with at least ``y_true``, ``y_pred_dl``, ``y_pred_persist``.
    metric:
        ``"MAE"`` or ``"RMSE"``.

    Returns
    -------
    np.ndarray of shape ``(n,)`` — the loss differential per row.

    Raises
    ------
    ValueError
        On an unknown ``metric``.
    """
    y_true = df["y_true"].to_numpy()
    e_dl = y_true - df["y_pred_dl"].to_numpy()
    e_persist = y_true - df["y_pred_persist"].to_numpy()

    if metric == "MAE":
        return np.abs(e_dl) - np.abs(e_persist)
    if metric == "RMSE":
        return e_dl**2 - e_persist**2
    raise ValueError(f"loss_differential: unknown metric {metric!r} (use MAE/RMSE)")


def diebold_mariano(d: np.ndarray, lag: int | None = None) -> DMResult:
    """Diebold-Mariano test on a loss-differential series ``d``.

    The variance of ``mean(d)`` is estimated with a Newey-West (HAC) kernel so
    serial autocorrelation does not understate it. When ``lag is None`` a default
    truncation ``floor(n**(1/3))`` is used; ``lag=0`` collapses to the iid
    (simple t) variance.

    Parameters
    ----------
    d:
        1-D loss differential from :func:`loss_differential`.
    lag:
        Newey-West truncation lag. ``None`` -> data-driven default; ``0`` -> iid.

    Returns
    -------
    DMResult

    Raises
    ------
    ValueError
        If ``d`` has fewer than 2 finite samples.
    """
    from scipy import stats

    d = np.asarray(d, dtype="float64")
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        raise ValueError("diebold_mariano: need at least 2 finite samples")

    if lag is None:
        lag = int(np.floor(n ** (1.0 / 3.0)))
    lag = max(0, min(lag, n - 1))

    mean_diff = float(d.mean())
    centered = d - mean_diff
    # gamma_0 plus Bartlett-weighted autocovariances (Newey-West HAC).
    gamma0 = float(np.dot(centered, centered) / n)
    s = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / n)
        s += 2.0 * weight * gamma_k

    # Var(mean) = S / n; guard against a non-positive HAC estimate.
    var_mean = s / n
    if var_mean <= 0:
        stat = 0.0 if mean_diff == 0 else np.inf * np.sign(mean_diff)
        p_value = 1.0 if mean_diff == 0 else 0.0
    else:
        stat = mean_diff / np.sqrt(var_mean)
        p_value = float(2.0 * stats.norm.sf(abs(stat)))

    return DMResult(stat=float(stat), p_value=p_value, mean_diff=mean_diff, lag=lag)


def wilcoxon_signed_rank(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on the loss differential ``d``.

    Tests the null that the median differential is zero. Distribution-free, so it
    is robust to the heavy-tailed headway errors that can distort the DM test.

    Parameters
    ----------
    d:
        1-D loss differential.

    Returns
    -------
    float p-value in ``[0, 1]``. Returns ``1.0`` if every differential is zero
    (no evidence against the null).
    """
    from scipy import stats

    d = np.asarray(d, dtype="float64")
    d = d[np.isfinite(d)]
    if d.size == 0 or np.allclose(d, 0.0):
        return 1.0
    # zero_method="zsplit" keeps ties; "auto" picks exact/normal by sample size.
    res = stats.wilcoxon(d, zero_method="zsplit", alternative="two-sided")
    return float(res.pvalue)


def significance_table(
    df: pl.DataFrame, metric: str = "MAE", direction: str = "aggregate"
) -> pl.DataFrame:
    """Per (corridor, horizon) significance + effect-size summary.

    For each ``(corridor, horizon)`` group in ``df`` (filtered to ``direction``),
    computes the loss differential and runs both tests, returning one row with:
    ``corridor, horizon, n, delta_loss, delta_mae, dm_stat, dm_p, wilcoxon_p,
    dl_better``. ``delta_loss`` is the mean differential for the row's metric
    (MAE absolute-error units; RMSE squared-error units because the paired test
    runs on squared errors). ``delta_mae`` is the mean MAE differential (the
    headline effect size the paper leads with). ``dl_better`` follows
    ``delta_loss < 0`` so RMSE rows are internally consistent.

    Parameters
    ----------
    df:
        Long residual frame from :func:`load_residuals`.
    metric:
        Loss for the DM/Wilcoxon differential (``"MAE"`` or ``"RMSE"``).
    direction:
        ``"aggregate"`` (default) pools BOTH travel directions per
        ``(corridor, horizon)`` — the per-sample residuals carry only the
        integer directions ``-1`` / ``1`` (there is no stored 'aggregate' row),
        so the aggregate slice is the union of all paired samples. Pass an
        integer (``-1`` or ``1``) to restrict the test to one direction.

    Returns
    -------
    pl.DataFrame, one row per ``(corridor, horizon)``, sorted ascending.

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
        raise ValueError(f"significance_table: no rows for direction={direction!r}")

    groups = (
        work.select(["corridor", "horizon"]).unique().sort(["corridor", "horizon"])
    )

    rows: list[dict] = []
    for corridor, horizon in groups.iter_rows():
        sub = work.filter(
            (pl.col("corridor") == corridor) & (pl.col("horizon") == horizon)
        )
        d_metric = loss_differential(sub, metric)
        d_mae = loss_differential(sub, "MAE")  # effect size always in MAE units
        dm = diebold_mariano(d_metric)
        delta_loss = float(np.mean(d_metric))
        delta_mae = float(np.mean(d_mae))
        rows.append(
            {
                "corridor": corridor,
                "horizon": int(horizon),
                "n": int(sub.height),
                "delta_loss": delta_loss,
                "delta_mae": delta_mae,
                "dm_stat": dm.stat,
                "dm_p": dm.p_value,
                "wilcoxon_p": wilcoxon_signed_rank(d_metric),
                "dl_better": delta_loss < 0,
            }
        )

    return pl.DataFrame(rows)


@dataclass(frozen=True)
class SignTestResult:
    """Cell-level binomial sign test for "model beats reference across cells".

    A coarse companion to the per-sample DM/Wilcoxon test, for comparisons where
    per-sample pairing is NOT available — notably DL vs XGBoost, whose residual
    exports differ in granularity (the DL export emits one row per overlapping
    window x bus, overcounting each target ~4.5x, while the baseline emits one
    row per test target) and share no per-sample key to realign or de-duplicate.

    Each (corridor, horizon) cell contributes ONE Bernoulli trial: does ``model``
    have the strictly lower aggregate loss? Under H0 (models equal) each trial is
    a fair coin, so the win count is Binomial(n_trials, 0.5). This deliberately
    sidesteps the overlapping-window overcounting that would inflate a naive
    per-sample n and understate the p-value.

    Attributes
    ----------
    n_cells:
        Number of (corridor, horizon) cells compared.
    n_model_wins:
        Cells where ``model`` loss is strictly below ``reference`` loss.
    n_ties:
        Cells with exactly equal loss (dropped from the binomial, per the
        standard sign-test convention).
    p_one_sided:
        P(>= n_model_wins wins | fair coin) — evidence that ``model`` is better.
    p_two_sided:
        Two-sided binomial p-value (no directional assumption).
    """

    n_cells: int
    n_model_wins: int
    n_ties: int
    p_one_sided: float
    p_two_sided: float


def sign_test_across_cells(
    model_losses: Sequence[float], reference_losses: Sequence[float]
) -> SignTestResult:
    """Binomial sign test over per-cell losses (see :class:`SignTestResult`).

    Parameters
    ----------
    model_losses, reference_losses:
        Aligned per-cell aggregate losses (e.g. MAE per (corridor, horizon)).
        Element ``i`` of both must describe the SAME cell. Lower is better.

    Ties (exactly equal loss) are dropped before the binomial, so the trial
    count is ``n_cells - n_ties``.

    Raises
    ------
    ValueError
        If the inputs differ in length, or if every non-tie cell is absent
        (nothing to test).
    """
    from scipy import stats

    model = np.asarray(model_losses, dtype=float)
    reference = np.asarray(reference_losses, dtype=float)
    if model.shape != reference.shape:
        raise ValueError(
            "sign_test_across_cells: model_losses and reference_losses must have "
            f"the same length ({model.size} vs {reference.size})"
        )
    n_cells = int(model.size)
    wins = int(np.sum(model < reference))
    losses = int(np.sum(model > reference))
    n_trials = wins + losses
    if n_trials == 0:
        raise ValueError(
            "sign_test_across_cells: every cell ties — no signal to test"
        )
    p_one = float(stats.binomtest(wins, n_trials, 0.5, alternative="greater").pvalue)
    p_two = float(stats.binomtest(wins, n_trials, 0.5, alternative="two-sided").pvalue)
    return SignTestResult(
        n_cells=n_cells,
        n_model_wins=wins,
        n_ties=n_cells - n_trials,
        p_one_sided=p_one,
        p_two_sided=p_two,
    )
