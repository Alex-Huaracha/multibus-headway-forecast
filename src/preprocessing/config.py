"""Configuration and frozen parameters for the Fase 2 preprocessing pipeline.

All productive parameter values are locked to docs/decisiones-headway-fase2.md §3.
Changing any value requires updating that document first (versioned decision),
then updating the literal here. The freeze-assertion test in
tests/preprocessing/test_config.py encodes this contract as executable checks.
"""
import math
from dataclasses import dataclass
from typing import Literal, Mapping

# ---------------------------------------------------------------------------
# Coordinate constants — local flat-Earth at Arequipa (-16.4°)
# ---------------------------------------------------------------------------

LAT_DEG_M: float = 111_000.0
LON_DEG_M: float = 111_000.0 * math.cos(math.radians(-16.4))

# ---------------------------------------------------------------------------
# Quality thresholds (decisiones-limpieza-fase2 §2 rows 4-5)
# ---------------------------------------------------------------------------

MAX_PLAUSIBLE_SPEED_KMH: float = 80.0
MAX_PLAUSIBLE_JUMP_M: float = 500.0

# ---------------------------------------------------------------------------
# Trip segmentation (decisión §3.3 of decisiones-limpieza-fase2)
# ---------------------------------------------------------------------------

GAP_CUT_SECONDS: int = 30 * 60         # 30-minute gap between consecutive pings
TERMINAL_BAND_M: float = 200.0         # within X m of s_min / s_max → terminal candidate
TERMINAL_DWELL_SECONDS: int = 5 * 60   # stopped > 5 min near a terminal → cut
TERMINAL_MAX_SPEED_KMH: float = 5.0    # stopped threshold for terminal-dwell detection


# ---------------------------------------------------------------------------
# Frozen productive parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductiveParams:
    """Frozen contract — every field mirrors docs/decisiones-headway-fase2.md §3.

    Changing a value requires updating that document FIRST (versioned), then
    this file. The freeze-assertion test in test_config.py turns this into
    executable code.
    """

    grid_seconds: int = 60
    min_speed_for_centerline_kmh: float = 10.0
    centerline_latlon_quantile_lo: float = 0.005
    centerline_latlon_quantile_hi: float = 0.995
    centerline_n_bins: int = 50
    centerline_trim_pct: float = 0.025
    centerline_smooth_win: int = 5
    lateral_offset_threshold_m: float = 300.0
    direction_smooth_win: int = 5
    min_buses_per_snapshot: int = 2
    # Max staleness in minutes for a historical crossing to count as a real
    # trailing pair. Older crossings → emit delta_t_min = NULL. Bound exists
    # because multi-filar corridors (e.g. E2 in Arequipa) project unrelated
    # buses to the same s; without this bound, np.searchsorted finds ancient
    # crossings and reports them as valid headways. See decisiones-headway-fase2 §3.
    max_interpolation_lookback_minutes: float = 30.0
    # Lateral distance threshold (meters) between bus_front and bus_back to
    # consider them on the same track. Pairs with |lateral_m_front -
    # lateral_m_back| > threshold are filtered out as cross-street pairs.
    #
    # DEFAULT: float('inf') — filter is OFF by default (no-op).
    # Rationale: Kaggle 04b v4 Figure 7 (2026-05-21) showed a monotonically-
    # decreasing |lateral_delta| distribution for E2/E59 with no bimodal valley.
    # A calibration threshold cannot be meaningfully chosen from this shape.
    # Root cause is upstream (centerline + projection per direction), addressed
    # by Option D SDD (multi-filar-direction-balanced-centerline).
    #
    # Opt-in: set EmpresaConfig.lateral_pair_threshold_m_override to a finite
    # value for any empresa where the filter should be active.
    # See decisiones-headway-fase2 §7.0b.1.
    #
    # Per-empresa override available via EmpresaConfig.lateral_pair_threshold_m_override.
    lateral_pair_threshold_m: float = float('inf')
    # Centerline strategy: "single" (default, single-pass PCA over all pings)
    # or "two-pass" (per-direction PCA for multi-filar corridors).
    # Overridden per empresa via EmpresaConfig.centerline_strategy_override.
    # See decisiones-headway-fase2 §8, R-CFG1.
    centerline_strategy: Literal["single", "two-pass"] = "single"
    # Minimum ping count per direction subset to attempt pass-2 PCA.
    # Below this threshold, build_centerline_per_direction falls back to the
    # single-pass centerline. See R-CL1, R-CFG1.
    centerline_min_pings_per_direction: int = 1_000


PRODUCTIVE_PARAMS = ProductiveParams()


# ---------------------------------------------------------------------------
# Per-empresa configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmpresaConfig:
    """Per-empresa settings.

    has_heading: E2/E4 report a `direccion` field usable as cross-check;
                 E58/E59 do not.
    centerline_sample_cap: maximum pings used to build the centerline.
    lateral_offset_threshold_m_override: when set, overrides
        PRODUCTIVE_PARAMS.lateral_offset_threshold_m for this empresa.
        Used for Caveat 3 monitoring (see decisiones-headway-fase2 §4).
    """

    empresaid: int
    has_heading: bool
    centerline_sample_cap: int = 50_000
    lateral_offset_threshold_m_override: float | None = None
    # Per-empresa override for the lateral pair filter threshold (meters).
    # When set, overrides PRODUCTIVE_PARAMS.lateral_pair_threshold_m for this
    # empresa. Used after Kaggle calibration of the |lateral_delta| histogram.
    lateral_pair_threshold_m_override: float | None = None
    # Per-empresa override for the centerline strategy.
    # When set, overrides PRODUCTIVE_PARAMS.centerline_strategy for this empresa.
    # E2 and E59 are set to "two-pass" (multi-filar corridor, Option D SDD).
    centerline_strategy_override: Literal["single", "two-pass"] | None = None


EMPRESA_CONFIG: Mapping[int, EmpresaConfig] = {
    2:  EmpresaConfig(empresaid=2,  has_heading=True,  centerline_strategy_override="two-pass"),
    59: EmpresaConfig(empresaid=59, has_heading=False, centerline_strategy_override="two-pass"),
    # E4 — third (validation) corridor for external validity (added 2026-06-22).
    # has_heading=True: E4 reports a `direccion` field (like E2).
    # centerline_strategy_override left None (→ "single", global default) PROVISIONALLY:
    # E2/E59 were set to "two-pass" only after Kaggle NB04 calibration evidence
    # (multi-filar projection). E4 has no such evidence yet — run NB04 for E4,
    # inspect the stale-crossing / headway sanity outputs, and flip to "two-pass"
    # here (1-line change) if it shows the same multi-filar behaviour.
    4:  EmpresaConfig(empresaid=4,  has_heading=True),
}


# ---------------------------------------------------------------------------
# Per-direction sort key calibration (SDD dir1-pair-ordering-h7)
# ---------------------------------------------------------------------------
# Empirically calibrated via observational evidence from Kaggle NB04 v7
# bucket analysis (obs #126). See sdd/dir1-pair-ordering-h7/apply-progress
# for full calibration evidence and cross-references.
#
# Set to +1 because the dir=+1 per-direction centerline has `s` inverse to
# physical direction of motion for empresas E2 and E59 (multi-filar corridor,
# two-pass PCA). Evidence: dir+1 yields 83.5%/92.6% stale-crossing rate;
# dir-1 yields ~70% success rate.
#
# CALIBRATED_INVERTED_DIRECTION: the direction value whose sort key is -s
# (negated arc-length) so that ascending sort places the physically-front
# bus first. For the other direction, sort key == s (canonical ascending).
#
# The consequence, spelled out because it is the pipeline's least obvious
# invariant: "first" is the slot shift(1) reads, so the physically-FRONT bus
# ends up in the `bus_back` / `s_back` columns and the trailing bus in
# `bus_front` / `s_front`. The pair labels are the mirror image of motion in
# BOTH directions. Do not rename the columns — see the module docstring of
# src/preprocessing/headways.py and docs/decisiones-headway-fase2.md §2.1.
CALIBRATED_INVERTED_DIRECTION: Literal[1, -1] = 1


def centerline_strategy_for(empresaid: int) -> str:
    """Return the effective centerline strategy for a given empresa.

    Checks EmpresaConfig.centerline_strategy_override first; falls back to
    PRODUCTIVE_PARAMS.centerline_strategy. Returns the global default for
    empresas not in EMPRESA_CONFIG (graceful missing-key handling).

    R-CFG1: E2 and E59 return "two-pass"; all others return "single".
    """
    cfg = EMPRESA_CONFIG.get(empresaid)
    if cfg is not None and cfg.centerline_strategy_override is not None:
        return cfg.centerline_strategy_override
    return PRODUCTIVE_PARAMS.centerline_strategy


def lateral_threshold_for(empresaid: int) -> float:
    """Return the effective lateral offset threshold for a given empresa.

    Checks EmpresaConfig.lateral_offset_threshold_m_override first; falls
    back to PRODUCTIVE_PARAMS.lateral_offset_threshold_m (Caveat 3 hook).
    Returns the global default for empresas not in EMPRESA_CONFIG (graceful
    missing-key handling, mirroring the other resolvers).
    """
    cfg = EMPRESA_CONFIG.get(empresaid)
    if cfg is not None and cfg.lateral_offset_threshold_m_override is not None:
        return cfg.lateral_offset_threshold_m_override
    return PRODUCTIVE_PARAMS.lateral_offset_threshold_m


def lateral_pair_threshold_for(empresaid: int) -> float:
    """Return the effective lateral pair filter threshold for a given empresa.

    Checks EmpresaConfig.lateral_pair_threshold_m_override first; falls back
    to PRODUCTIVE_PARAMS.lateral_pair_threshold_m. Returns the global default
    for empresas not in EMPRESA_CONFIG (graceful missing-key handling).

    Used by compute_pairs to decide which (front, back) pairs are cross-street
    contamination and should be filtered out.
    """
    cfg = EMPRESA_CONFIG.get(empresaid)
    if cfg is not None and cfg.lateral_pair_threshold_m_override is not None:
        return cfg.lateral_pair_threshold_m_override
    return PRODUCTIVE_PARAMS.lateral_pair_threshold_m
