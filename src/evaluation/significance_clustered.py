"""Significance testing for the retrained pipeline — with the audited corrections.

``evaluation/significance.py`` stays as it is: it produced the published numbers
and the frozen artifacts still have to be reproducible from it. This module is
its corrected successor, applied to the contiguous-pipeline residuals, and it
folds in three findings the audit left open.

**#6 — the long-run variance was computed over the wrong axis.** Newey-West was
applied to residuals ordered *slot-major*, so exact replicas of one target sat
tens of thousands of positions apart, far outside the truncation lag. Contract
C1 removes the replication itself, but samples from the same service day are
still correlated — an incident at 08:00 shapes the whole morning. The right
estimator groups by **service day**, which is what ``dm_clustered`` does.

**#7 — the DM test was missing its small-sample apparatus.** Three gaps: no
Harvey-Leybourne-Newbold correction, no ``lag >= h-1`` floor (an h-step-ahead
forecast error is MA(h-1) by construction, so a data-driven ``n^(1/3)`` that
comes out smaller understates the variance), and a normal reference where the
HLN statistic calls for Student's t. None of them flips a verdict at these
sample sizes, but the text presented ``n^(1/3)`` as the DM standard, and it is
not.

**#1 — the Wilcoxon was reported two-sided next to a mean-based direction.**
``dl_better`` was set from the *mean* differential while the two-sided Wilcoxon
p-value sat beside it looking like corroboration. It is not: a model can win the
mean and lose the median. ``wilcoxon_directional`` reports the median, the win
rate and a one-sided p-value in the direction the mean claims, so the two
statistics can be read against each other instead of conflated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class DMResult:
    """One Diebold-Mariano verdict, with the apparatus that produced it."""

    stat: float
    p_value: float
    mean_diff: float
    n: int
    variance_kind: str        # "hac-hln" or "cluster-day"
    lag: int | None           # truncation lag; None for the clustered estimator
    n_clusters: int | None    # cluster count; None for HAC
    dof: int                  # degrees of freedom of the t reference


def _t_sf(stat: float, dof: int) -> float:
    """Two-sided p-value under Student's t with ``dof`` degrees of freedom."""
    from scipy import stats

    return float(2.0 * stats.t.sf(abs(stat), df=dof))


def _degenerate(mean_diff: float, n: int, kind: str, lag, n_clusters, dof) -> DMResult:
    """Verdict when the variance estimate is non-positive.

    A non-positive long-run variance means the estimator has nothing to work
    with; reporting p=0 for a non-zero mean would be an artifact, so the result
    is flagged through an infinite statistic rather than smuggled in as
    significance.
    """
    stat = 0.0 if mean_diff == 0 else math.inf * (1 if mean_diff > 0 else -1)
    p = 1.0 if mean_diff == 0 else 0.0
    return DMResult(stat, p, mean_diff, n, kind, lag, n_clusters, dof)


def cube_root_floor(n: int) -> int:
    """``floor(n ** (1/3))`` without floating-point drift.

    ``int(64 ** (1/3))`` is 3, not 4: the power evaluates to 3.9999999999999996.
    ``significance.py`` computes the truncation lag that way, so on exact cubes
    it silently uses one lag fewer than the rule states. Harmless in practice,
    wrong as written — and this module is the one that gets to be right.
    """
    r = int(round(n ** (1.0 / 3.0)))
    while r > 0 and r * r * r > n:
        r -= 1
    while (r + 1) ** 3 <= n:
        r += 1
    return r


def hln_scale(n: int, horizon: int) -> float:
    """Harvey-Leybourne-Newbold small-sample scaling factor.

    ``DM* = DM * sqrt( [n + 1 - 2h + h(h-1)/n] / n )``, compared against
    ``t_{n-1}``. The factor is below 1, so it *shrinks* the statistic — omitting
    it makes results look more significant than they are.
    """
    num = n + 1 - 2 * horizon + horizon * (horizon - 1) / n
    return math.sqrt(max(num, 0.0) / n)


def dm_hac_hln(
    d: np.ndarray, *, horizon: int, lag: int | None = None
) -> DMResult:
    """Diebold-Mariano with a Bartlett HAC variance, HLN correction and t reference.

    The truncation lag is ``max(floor(n^(1/3)), horizon - 1)``. The floor matters:
    an h-step-ahead forecast error follows an MA(h-1) process, so any lag below
    ``h-1`` leaves real autocovariance out of the variance.
    """
    d = np.asarray(d, dtype="float64")
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        raise ValueError("dm_hac_hln: need at least 3 finite samples")

    if lag is None:
        lag = max(cube_root_floor(n), horizon - 1)
    lag = max(0, min(lag, n - 1))

    mean_diff = float(d.mean())
    centered = d - mean_diff
    s = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        s += 2.0 * weight * float(np.dot(centered[k:], centered[:-k]) / n)

    var_mean = s / n
    dof = n - 1
    if var_mean <= 0:
        return _degenerate(mean_diff, n, "hac-hln", lag, None, dof)

    stat = (mean_diff / math.sqrt(var_mean)) * hln_scale(n, horizon)
    return DMResult(stat, _t_sf(stat, dof), mean_diff, n, "hac-hln", lag, None, dof)


def dm_clustered(
    d: np.ndarray, cluster_ids: np.ndarray, *, horizon: int
) -> DMResult:
    """Diebold-Mariano with a variance clustered on ``cluster_ids``.

    Samples inside one service day share weather, incidents and demand, so their
    loss differentials are correlated in a way no lag structure over a flattened
    array can capture. The cluster-robust variance of the mean is

        Var(d̄) = (1 / n²) · Σ_g ( Σ_{i∈g} (d_i - d̄) )²

    and the reference is ``t_{G-1}``, with ``G`` the number of clusters — the
    effective sample size is the number of *days*, not the number of rows. This
    is the estimator the audit identified as correct; it is strictly more
    conservative than the HAC variant.
    """
    d = np.asarray(d, dtype="float64")
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(d)
    d, cluster_ids = d[finite], cluster_ids[finite]
    n = d.size
    if n < 3:
        raise ValueError("dm_clustered: need at least 3 finite samples")

    # Dense rank maps arbitrary cluster labels (dates, strings) to 0..G-1 so
    # bincount can aggregate over them.
    codes = pl.Series(cluster_ids).rank("dense").to_numpy().astype(np.int64) - 1
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise ValueError("dm_clustered: need at least 2 clusters")

    mean_diff = float(d.mean())
    centered = d - mean_diff
    per_cluster = np.bincount(codes, weights=centered, minlength=n_clusters)

    var_mean = float(np.dot(per_cluster, per_cluster)) / (n * n)
    dof = n_clusters - 1
    if var_mean <= 0:
        return _degenerate(mean_diff, n, "cluster-day", None, n_clusters, dof)

    stat = (mean_diff / math.sqrt(var_mean)) * hln_scale(n, horizon)
    return DMResult(
        stat, _t_sf(stat, dof), mean_diff, n, "cluster-day", None, n_clusters, dof
    )


def wilcoxon_directional(d: np.ndarray) -> dict:
    """Wilcoxon signed-rank reported with its direction.

    Returns the two-sided p-value, the median differential, the fraction of
    samples the first model wins, and a one-sided p-value testing the direction
    the *mean* claims. Reporting only the two-sided value beside a mean-derived
    verdict hides the case the audit flagged: a model that wins the average by
    trading many small losses for a few large gains loses the median outright.
    """
    from scipy import stats

    d = np.asarray(d, dtype="float64")
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        raise ValueError("wilcoxon_directional: need at least 2 finite samples")

    mean_diff = float(d.mean())
    median_diff = float(np.median(d))
    # d < 0 means the first model has the smaller loss on that sample.
    win_rate = float((d < 0).mean())

    two_sided = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    # One-sided in the direction the mean asserts.
    alt = "less" if mean_diff < 0 else "greater"
    one_sided = float(stats.wilcoxon(d, alternative=alt).pvalue)

    return {
        "wilcoxon_p_two_sided": two_sided,
        "wilcoxon_p_one_sided": one_sided,
        "wilcoxon_direction": alt,
        "median_diff": median_diff,
        "win_rate": win_rate,
        "mean_diff": mean_diff,
        # True when the mean and the median disagree on who wins — the exact
        # pattern the audit found unreported at h=3 in E59 and E4.
        "mean_median_disagree": bool((mean_diff < 0) != (median_diff < 0)),
    }
