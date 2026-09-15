"""Intervals for the ranking verdicts — the instrument Section IV-E promised.

The paper defines a verdict as three parts: which of the two methods wins, by
how much, and whether the difference survives its test. It then emits verdicts
on the MAE (Diebold-Mariano, ``significance_clustered``) and on the precision
(Clopper-Pearson, ``vector_metrics.precision_interval``) — but the headline
count is over AUC, and the AUC had no test at all. A win of 0.001 and a win of
0.05 were reported the same way.

Why a clustered bootstrap and not DeLong
----------------------------------------
DeLong is the standard test for two correlated AUCs, and it assumes independent
observations. This corpus is not independent: ``significance_clustered`` already
argues that samples from one service day share weather, incidents and demand,
and the MAE variance is estimated by clustering on exactly that. Using DeLong
here would contradict the paper's own stated position about its own data.

Resampling service days with replacement keeps that position, and it costs
nothing in generality: the same routine bounds the MCC, which has no closed-form
test of its own worth importing.

The effective sample size is the number of service days — 22 in the published
window — not the hundreds of thousands of rows. That is the point. An interval
built on the row count would be narrow for the same reason the unclustered MAE
test was too generous.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def fast_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """The same AUC as ``vector_metrics.ranking_scores``, without its Python loop.

    ``ranking_scores`` resolves ties by walking every element in Python. That is
    fine once per cell and ruinous inside a bootstrap, which asks for the AUC
    four thousand times over hundreds of thousands of rows. This computes the
    identical average ranks with a cumulative sum over tie-group boundaries.

    ``ranking_scores`` is deliberately left alone: it produced every published
    number, and a rewrite there would put those bytes at risk for no gain.
    ``test_fast_auc_matches_ranking_scores`` pins the two together.
    """
    truth = np.asarray(truth, dtype=bool)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    truth, score = truth[ok], score[ok]

    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ordered = score[order]
    # First sorted position of each tie group, then the average 1-based rank the
    # whole group shares: start + (count - 1) / 2 + 1.
    opens = np.empty(ordered.size, dtype=bool)
    opens[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=opens[1:])
    starts = np.flatnonzero(opens)
    counts = np.diff(np.append(starts, ordered.size))
    group_rank = starts + (counts - 1) / 2.0 + 1.0

    ranks = np.empty(ordered.size, dtype=float)
    ranks[order] = np.repeat(group_rank, counts)
    return float((ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class RankingSignificanceError(RuntimeError):
    """The interval could not be computed, so no interval is returned.

    A nan printed beside real bounds reads as "no difference" when it means "no
    measurement". Every failure here is raised rather than encoded.
    """


@dataclass(frozen=True)
class BootstrapDelta:
    """One method's advantage over another, with its resampled interval."""

    delta: float
    ci_low: float
    ci_high: float
    n_clusters: int
    n_boot: int
    n_usable: int
    level: float

    @property
    def crosses_zero(self) -> bool:
        """True when the interval admits no advantage in either direction."""
        return self.ci_low <= 0.0 <= self.ci_high


def clustered_bootstrap_delta(
    truth: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cluster: np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
    level: float = 0.95,
) -> BootstrapDelta:
    """Bound ``statistic(truth, a) - statistic(truth, b)`` by resampling clusters.

    Args:
        truth: boolean outcome per row.
        a, b: the two scores being compared, aligned with ``truth``.
        cluster: the resampling unit per row — the service day, here.
        statistic: (truth, score) -> float. ``ranking_scores``' AUC and
            ``matthews_corrcoef`` both fit, so the same routine serves the
            threshold-free and the operating-point verdicts.
        n_boot: number of resamples.
        seed: fixed, because the published interval has to be regenerable.
        level: two-sided coverage of the percentile interval.

    Returns:
        BootstrapDelta. ``delta`` is computed once on the full corpus, not as
        the mean of the resamples: the resamples bound it, they do not define it.

    Raises:
        RankingSignificanceError: on ragged input, on a degenerate level, when
        the point estimate itself does not evaluate, or when no resample does.
    """
    truth = np.asarray(truth, dtype=bool)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cluster = np.asarray(cluster)

    sizes = {truth.size, a.size, b.size, cluster.size}
    if len(sizes) != 1:
        raise RankingSignificanceError(
            f"truth, a, b and cluster must align; got sizes {sorted(sizes)}"
        )
    if truth.size == 0:
        raise RankingSignificanceError("no rows to bound")
    if not 0.0 < level < 1.0:
        raise RankingSignificanceError(f"level must lie in (0, 1); got {level}")
    if n_boot < 1:
        raise RankingSignificanceError(f"n_boot must be positive; got {n_boot}")

    delta = _delta(truth, a, b, statistic)
    if not np.isfinite(delta):
        raise RankingSignificanceError(
            "the statistic does not evaluate on the full corpus; "
            "a cell with one class present cannot be bounded"
        )

    # Row indices grouped by cluster, so a draw concatenates whole days.
    order = np.argsort(cluster, kind="mergesort")
    keys, starts = np.unique(cluster[order], return_index=True)
    groups = np.split(order, starts[1:])
    n_clusters = len(keys)

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[k] for k in picked])
        value = _delta(truth[idx], a[idx], b[idx], statistic)
        if np.isfinite(value):
            deltas.append(value)

    if not deltas:
        raise RankingSignificanceError(
            f"no resample of {n_boot} produced a finite statistic; "
            "the cell is too thin to bound"
        )

    tail = (1.0 - level) / 2.0
    low, high = np.quantile(deltas, [tail, 1.0 - tail])
    return BootstrapDelta(
        delta=float(delta),
        ci_low=float(low),
        ci_high=float(high),
        n_clusters=int(n_clusters),
        n_boot=int(n_boot),
        n_usable=len(deltas),
        level=float(level),
    )


def _delta(
    truth: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """The advantage of ``a`` over ``b`` on one corpus, nan if either side fails."""
    first = statistic(truth, a)
    second = statistic(truth, b)
    if not (np.isfinite(first) and np.isfinite(second)):
        return float("nan")
    return float(first - second)
