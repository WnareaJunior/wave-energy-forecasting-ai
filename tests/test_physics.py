"""Analytic checks on the wave physics.

These are the formulas every downstream number depends on, so they are pinned
against textbook values rather than against previous outputs.
"""

import numpy as np
import pytest

from src.processing.wave_power_flux import (
    ENERGY_PERIOD_FACTORS,
    PERIOD_VARIABLE_PRIORITY,
    calculate_consistency_metrics,
    calculate_wave_power_flux,
    create_revenue_surface_map,
    group_velocity_deep_water,
    steepness,
    wavelength_deep_water,
)


class TestPowerFlux:
    def test_textbook_value(self):
        """Hm0 = 2 m, Te = 8 s is the standard ~15.7 kW/m reference case."""
        p = calculate_wave_power_flux(2.0, 8.0)
        assert p == pytest.approx(15_700, rel=0.01)

    def test_coefficient(self):
        """The rho*g^2/(64*pi) prefactor is ~490 for seawater."""
        assert calculate_wave_power_flux(1.0, 1.0) == pytest.approx(490.6, rel=0.001)

    def test_scales_with_height_squared(self):
        assert calculate_wave_power_flux(4.0, 8.0) == pytest.approx(
            4 * calculate_wave_power_flux(2.0, 8.0)
        )

    def test_scales_linearly_with_period(self):
        assert calculate_wave_power_flux(2.0, 16.0) == pytest.approx(
            2 * calculate_wave_power_flux(2.0, 8.0)
        )

    def test_zero_height_gives_zero_power(self):
        assert calculate_wave_power_flux(0.0, 10.0) == 0.0

    def test_vectorised(self):
        heights = np.array([1.0, 2.0, 3.0])
        result = calculate_wave_power_flux(heights, 10.0)
        assert result.shape == (3,)
        assert np.all(np.diff(result) > 0)


class TestPeriodConversion:
    def test_vtm10_is_energy_period(self):
        """VTM10 is Te by definition, so it needs no conversion."""
        assert ENERGY_PERIOD_FACTORS["VTM10"] == 1.0

    def test_priority_prefers_vtm10(self):
        assert PERIOD_VARIABLE_PRIORITY[0] == "VTM10"

    def test_all_priority_variables_have_a_factor(self):
        assert set(PERIOD_VARIABLE_PRIORITY) == set(ENERGY_PERIOD_FACTORS)

    def test_peak_period_converts_downward(self):
        """Te < Tp, and Te > Tm02, for a realistic wind-sea spectrum."""
        assert ENERGY_PERIOD_FACTORS["VTPK"] < 1.0
        assert ENERGY_PERIOD_FACTORS["VTM02"] > 1.0


class TestWaveKinematics:
    def test_group_velocity(self):
        """cg = g*T/(4*pi); a 10 s deep-water wave group travels ~7.8 m/s."""
        assert group_velocity_deep_water(10.0) == pytest.approx(7.806, rel=0.001)

    def test_wavelength(self):
        """L = g*T^2/(2*pi); a 10 s deep-water wave is ~156 m long."""
        assert wavelength_deep_water(10.0) == pytest.approx(156.13, rel=0.001)

    def test_group_velocity_is_half_phase_velocity(self):
        """Deep-water group velocity is exactly half the phase velocity."""
        period = 12.0
        phase_velocity = wavelength_deep_water(period) / period
        assert group_velocity_deep_water(period) == pytest.approx(phase_velocity / 2)

    def test_steepness_below_breaking_limit(self):
        """A 2 m, 10 s swell is far from the 1/7 breaking limit."""
        assert steepness(2.0, 10.0) < 1 / 7

    def test_steepness_flags_breaking_waves(self):
        """A 20 m wave on a 6 s period is beyond the physical limit."""
        assert steepness(20.0, 6.0) > 1 / 7


class TestConsistencyMetrics:
    def test_documented_keys_are_returned(self):
        result = calculate_consistency_metrics(np.array([10.0, 20.0, 30.0, 40.0]))
        assert set(result) == {
            "std_deviation",
            "coefficient_of_variation",
            "percent_above_threshold",
            "threshold_value",
            "mean_power",
            "median_power",
        }

    def test_nans_are_ignored(self):
        with_nans = np.array([10.0, np.nan, 20.0, np.nan, 30.0])
        without = np.array([10.0, 20.0, 30.0])
        assert calculate_consistency_metrics(with_nans)["mean_power"] == pytest.approx(
            calculate_consistency_metrics(without)["mean_power"]
        )

    def test_all_nan_input_returns_nan_not_a_crash(self):
        result = calculate_consistency_metrics(np.array([np.nan, np.nan]))
        assert all(np.isnan(v) for v in result.values())

    def test_empty_input_returns_nan(self):
        result = calculate_consistency_metrics(np.array([]))
        assert all(np.isnan(v) for v in result.values())

    def test_coefficient_of_variation(self):
        constant = calculate_consistency_metrics(np.full(10, 25.0))
        assert constant["coefficient_of_variation"] == pytest.approx(0.0)


class TestRevenue:
    def test_default_is_per_metre_of_crest(self):
        """30 kW/m at $50/MWh -> 30 kW * 8760 h = 262.8 MWh -> $13,140/m/yr."""
        assert create_revenue_surface_map(30_000.0) == pytest.approx(13_140.0)

    def test_device_parameters_scale_revenue(self):
        """A 10 m wide device at 35% efficiency captures 3.5 m-equivalents."""
        per_metre = create_revenue_surface_map(30_000.0)
        device = create_revenue_surface_map(
            30_000.0, capture_width_m=10, efficiency=0.35
        )
        assert device == pytest.approx(per_metre * 3.5)

    def test_availability_reduces_revenue(self):
        full = create_revenue_surface_map(30_000.0)
        assert create_revenue_surface_map(30_000.0, availability=0.9) == pytest.approx(
            full * 0.9
        )
