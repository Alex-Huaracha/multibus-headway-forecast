"""Ex-ante volatility: the non-circular way to stratify the DL-vs-persistence gap.

``evaluation/volatility.py`` bins samples by ``|y_true - y_pred_persist|`` — the
realized change, which is also persistence's error. That bin is informative to
LOOK at and impossible to TEST inside: it conditions on one of the two terms of
the loss differential, so any p-value it produces is arithmetic rather than
evidence (audit pending #2). Its inferential columns are gone.

This module supplies the stratifier that can be tested. The dispersion of the
**input window** is known at prediction time — it uses only the ``T_in``
snapshots the model itself consumed, and nothing at or after the target — so it
is statistically independent of the outcome in the way the regime variable was
not. Conditioning on it and then testing the loss differential is a legitimate
subgroup analysis.

Definition
----------
For each ``(sample, pair_rank)`` cell, the sample standard deviation (``ddof=1``)
of the observed values in that cell's input window. Cells with fewer than
``min_obs`` observed timesteps get ``NaN`` — they carry no dispersion
information, and ``exante_terciles.compute_frozen_thresholds`` already excludes
non-finite values from calibration.

The unit is minutes of headway, matching the residual exports, so a threshold
can be read directly as "windows whose headway wobbled more than X minutes".
"""
from __future__ import annotations

import numpy as np


def window_dispersion(
    values: np.ndarray, mask: np.ndarray, *, min_obs: int = 2
) -> np.ndarray:
    """Per-cell standard deviation of the observed input window.

    Parameters
    ----------
    values:
        ``(n_samples, T_in, max_N)`` window values in the units to report.
        Positions where ``mask`` is False are ignored whatever they hold — the
        grid convention writes 0.0 there, and that 0.0 must not enter the mean.
    mask:
        ``(n_samples, T_in, max_N)`` boolean, True = VALID (INV-5).
    min_obs:
        Minimum observed timesteps for a defined dispersion. Below 2 the sample
        variance is undefined, so values under 2 are rejected rather than
        silently producing zeros.

    Returns
    -------
    ``(n_samples, max_N)`` float64. ``NaN`` where the cell has fewer than
    ``min_obs`` observations.

    Raises
    ------
    ValueError
        If the shapes disagree or ``min_obs`` is below 2.
    """
    if min_obs < 2:
        raise ValueError(f"window_dispersion: min_obs must be >= 2, got {min_obs}")
    if values.shape != mask.shape:
        raise ValueError(
            f"window_dispersion: shape mismatch, values {values.shape} vs "
            f"mask {mask.shape}"
        )
    if values.ndim != 3:
        raise ValueError(
            f"window_dispersion: expected (n, T_in, max_N), got {values.shape}"
        )

    valid = mask.astype(np.float64)
    obs = np.asarray(values, dtype=np.float64) * valid

    counts = valid.sum(axis=1)
    # Guard the division; cells below min_obs are overwritten with NaN below.
    safe = np.where(counts > 0, counts, 1.0)
    means = obs.sum(axis=1) / safe

    deviations = (np.asarray(values, dtype=np.float64) - means[:, None, :]) * valid
    sum_sq = (deviations**2).sum(axis=1)

    ddof_counts = np.where(counts > 1, counts - 1, 1.0)
    dispersion = np.sqrt(sum_sq / ddof_counts)

    return np.where(counts >= min_obs, dispersion, np.nan)
