"""Freeze-assertion tests for src/preprocessing/config.py.

These tests encode the contract from docs/decisiones-headway-fase2.md §3.
If any value drifts without updating the decision document, the test fails —
making the freeze contract executable rather than only documentary.
"""
import math

from src.preprocessing.config import (
    EMPRESA_CONFIG,
    EmpresaConfig,
    GAP_CUT_SECONDS,
    LAT_DEG_M,
    LON_DEG_M,
    PRODUCTIVE_PARAMS,
    TERMINAL_BAND_M,
    TERMINAL_DWELL_SECONDS,
    lateral_threshold_for,
    lateral_pair_threshold_for,
)


class TestProductiveParamsFreeze:
    """Assert every ProductiveParams field matches decisiones-headway-fase2 §3."""

    def test_grid_seconds(self):
        assert PRODUCTIVE_PARAMS.grid_seconds == 60

    def test_min_speed_for_centerline_kmh(self):
        assert PRODUCTIVE_PARAMS.min_speed_for_centerline_kmh == 10.0

    def test_centerline_latlon_quantile_lo(self):
        assert PRODUCTIVE_PARAMS.centerline_latlon_quantile_lo == 0.005

    def test_centerline_latlon_quantile_hi(self):
        assert PRODUCTIVE_PARAMS.centerline_latlon_quantile_hi == 0.995

    def test_centerline_n_bins(self):
        assert PRODUCTIVE_PARAMS.centerline_n_bins == 50

    def test_lateral_offset_threshold_m(self):
        assert PRODUCTIVE_PARAMS.lateral_offset_threshold_m == 300.0

    def test_direction_smooth_win(self):
        assert PRODUCTIVE_PARAMS.direction_smooth_win == 5

    def test_min_buses_per_snapshot(self):
        assert PRODUCTIVE_PARAMS.min_buses_per_snapshot == 2

    def test_max_interpolation_lookback_minutes(self):
        assert PRODUCTIVE_PARAMS.max_interpolation_lookback_minutes == 30.0

    def test_frozen(self):
        """ProductiveParams must raise on attempted mutation."""
        import dataclasses
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            PRODUCTIVE_PARAMS.grid_seconds = 30  # type: ignore[misc]


class TestEmpresaConfig:
    """Assert EMPRESA_CONFIG has exactly {2, 59, 4} with correct has_heading.

    E4 added 2026-06-22 as the third (validation) corridor for external validity.
    E4 reports a `direccion` field (has_heading=True, like E2).
    """

    def test_keys(self):
        assert set(EMPRESA_CONFIG.keys()) == {2, 59, 4}

    def test_e2_has_heading(self):
        assert EMPRESA_CONFIG[2].has_heading is True

    def test_e59_no_heading(self):
        assert EMPRESA_CONFIG[59].has_heading is False

    def test_e4_has_heading(self):
        assert EMPRESA_CONFIG[4].has_heading is True


class TestModuleConstants:
    """Assert module-level constants are exported and have expected types."""

    def test_lat_deg_m(self):
        assert LAT_DEG_M == 111_000.0

    def test_lon_deg_m(self):
        expected = 111_000.0 * math.cos(math.radians(-16.4))
        assert abs(LON_DEG_M - expected) < 1e-6

    def test_gap_cut_seconds(self):
        assert GAP_CUT_SECONDS == 1800

    def test_terminal_band_m(self):
        assert TERMINAL_BAND_M == 200.0

    def test_terminal_dwell_seconds(self):
        assert TERMINAL_DWELL_SECONDS == 300


class TestLateralThreshold:
    """Assert lateral_threshold_for returns the expected value per empresa."""

    def test_default_no_override(self):
        assert lateral_threshold_for(2) == 300.0
        assert lateral_threshold_for(59) == 300.0
        assert lateral_threshold_for(4) == 300.0

    def test_missing_empresa_returns_global_default(self):
        """Resolver must not KeyError on an empresa absent from EMPRESA_CONFIG.

        Regression: lateral_threshold_for used EMPRESA_CONFIG[empresaid] (direct
        index) and would raise KeyError for any new corridor. It must mirror the
        other resolvers and fall back to the global default via .get().
        """
        assert lateral_threshold_for(999) == PRODUCTIVE_PARAMS.lateral_offset_threshold_m


class TestProductiveParamsFreezeLateralPair:
    """AC-C3: lateral_pair_threshold_m default = float('inf') (filter OFF by default).

    Changed 2026-05-21: default switched from 50.0 to float('inf') after Kaggle
    04b v4 Figure 7 showed a monotonically-decreasing |lateral_delta| distribution
    with no bimodal valley — calibration is impossible. See decisiones-headway-fase2
    §7.0b.1. The filter remains available as opt-in via EmpresaConfig override.
    """

    def test_lateral_pair_threshold_default(self):
        """Default must be float('inf') so the filter is a no-op unless overridden."""
        assert math.isinf(PRODUCTIVE_PARAMS.lateral_pair_threshold_m), (
            f"Expected float('inf') (filter OFF); got {PRODUCTIVE_PARAMS.lateral_pair_threshold_m}"
        )


class TestLateralPairThreshold:
    """AC-C4: lateral_pair_threshold_for resolver."""

    def test_default_no_override(self):
        """Resolver must return global default (float('inf')) for empresas without override."""
        assert math.isinf(lateral_pair_threshold_for(2)), (
            f"Expected float('inf') for E2 (no override); got {lateral_pair_threshold_for(2)}"
        )
        assert math.isinf(lateral_pair_threshold_for(59)), (
            f"Expected float('inf') for E59 (no override); got {lateral_pair_threshold_for(59)}"
        )

    def test_override_used_when_set(self):
        """Resolver must return per-empresa override when set."""
        import src.preprocessing.config as config_module
        original_config = config_module.EMPRESA_CONFIG
        try:
            # Temporarily override EMPRESA_CONFIG to include an override for empresa 2.
            config_module.EMPRESA_CONFIG = {
                2: EmpresaConfig(
                    empresaid=2,
                    has_heading=True,
                    lateral_pair_threshold_m_override=30.0,
                ),
                59: EmpresaConfig(empresaid=59, has_heading=False),
            }
            result = lateral_pair_threshold_for(2)
            assert result == 30.0, f"Expected 30.0 with override; got {result}"
        finally:
            config_module.EMPRESA_CONFIG = original_config

    def test_missing_empresa_returns_global_default(self):
        """Resolver must return global default for empresa not in EMPRESA_CONFIG."""
        result = lateral_pair_threshold_for(999)
        assert result == PRODUCTIVE_PARAMS.lateral_pair_threshold_m


# ---------------------------------------------------------------------------
# T1.1 RED: TestCenterlineStrategy — global default + min pings threshold
# ---------------------------------------------------------------------------


class TestCenterlineStrategy:
    """D2-CONFIG — centerline_strategy resolver and min pings per direction.

    R-CFG1: PRODUCTIVE_PARAMS.centerline_strategy == "single" (global default).
    R-CFG1: PRODUCTIVE_PARAMS.centerline_min_pings_per_direction == 1_000.
    R-CFG1: centerline_strategy_for(2) == "two-pass".
    R-CFG1: centerline_strategy_for(59) == "two-pass".
    R-CFG1: centerline_strategy_for(4) == "single" (E4 in config, override=None →
            global default; provisional pending NB04 calibration evidence).
    R-CFG1: centerline_strategy_for(999) == "single" (empresa absent → fallback).
    """

    def test_centerline_strategy_global_default(self):
        """Global default must be 'single' for backward compatibility (R-CFG1)."""
        from src.preprocessing.config import PRODUCTIVE_PARAMS
        assert PRODUCTIVE_PARAMS.centerline_strategy == "single"
        assert PRODUCTIVE_PARAMS.centerline_min_pings_per_direction == 1_000

    def test_centerline_strategy_override_E2_E59(self):
        """E2 and E59 must resolve to 'two-pass'; E4 resolves to 'single' (no override yet)."""
        from src.preprocessing.config import centerline_strategy_for
        assert centerline_strategy_for(2) == "two-pass"
        assert centerline_strategy_for(59) == "two-pass"
        assert centerline_strategy_for(4) == "single"

    def test_centerline_strategy_missing_empresa(self):
        """An empresa absent from EMPRESA_CONFIG falls back to the global default."""
        from src.preprocessing.config import centerline_strategy_for
        assert centerline_strategy_for(999) == "single"
