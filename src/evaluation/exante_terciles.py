"""Frozen ex-ante volatility tercile calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TercileThresholds:
    """Frozen percentile thresholds derived from one calibration split."""

    p33: float
    p66: float
    calib_split: str
    calib_n: int


def compute_frozen_thresholds(calib_values: np.ndarray) -> TercileThresholds:
    """Compute frozen p33/p66 thresholds from finite train+val values only.

    At least three finite values are required to define the three operational
    regimes. NaN ex-ante values are expected for windows without enough input
    timesteps; NaN and positive or negative infinity are deliberately excluded
    from calibration.
    """
    values = np.asarray(calib_values)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 3:
        raise ValueError(
            "compute_frozen_thresholds requires at least three finite calibration values"
        )

    return TercileThresholds(
        p33=float(np.percentile(finite_values, 100 / 3)),
        p66=float(np.percentile(finite_values, 200 / 3)),
        calib_split="train+val",
        calib_n=int(finite_values.size),
    )


def assign_terciles(values: np.ndarray, thresholds: TercileThresholds) -> np.ndarray:
    """Classify finite values using frozen thresholds; exclude all non-finite rows."""
    values = np.asarray(values)
    finite_values = values[np.isfinite(values)]
    codes = np.where(
        finite_values <= thresholds.p33,
        0,
        np.where(finite_values <= thresholds.p66, 1, 2),
    )
    return codes.astype(int)
