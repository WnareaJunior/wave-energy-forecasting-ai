"""Tests for the deployment, accessibility and revenue models.

The claim this whole module set exists to support is that **ranking sites on
wave power alone is misleading**, because access cost rises with the same seas
that carry the energy. Several tests below assert exactly that relationship
rather than just checking arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from src.economics.accessibility import (
    accessible_fraction,
    expected_waiting_hours,
    find_windows,
    seasonal_accessibility,
    window_statistics,
)
from src.economics.deployment import (
    AHTS,
    ANNUAL_OPERATIONS,
    CTV,
    DEFAULT_STANDBY_CAP_HOURS,
    SiteLogistics,
    false_start_multiplier,
    mooring_capex,
    operation_cost,
    transit_hours,
    window_hours_required,
)
from src.economics.revenue import (
    DeviceSpec,
    MarketSpec,
    annual_energy_mwh,
    annual_om_cost,
    downtime_fraction,
    evaluate_site,
)


def make_hs(mean=2.5, amplitude=1.2, years=3, seed=0, freq="1h"):
    """Synthetic Hs with a winter peak and storm-scale persistence."""
    index = pd.date_range("2016-01-01", periods=int(years * 8766), freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    doy = index.dayofyear.to_numpy()
    seasonal = mean + amplitude * np.cos(2 * np.pi * (doy - 5) / 365.25)
    # AR(1) so calm and rough conditions come in runs, as they do at sea.
    noise = rng.normal(0, 0.3, len(index))
    ar = np.zeros(len(index))
    for i in range(1, len(index)):
        ar[i] = 0.98 * ar[i - 1] + noise[i]
    return pd.Series(np.clip(seasonal + ar, 0.2, 15.0), index=index, name="WVHT")


class TestTransitAndWindows:
    def test_transit_is_round_trip(self):
        """A 100 km site at 11 kn is about 10 hours there and back."""
        assert transit_hours(100.0, AHTS) == pytest.approx(9.82, rel=0.02)

    def test_faster_vessel_transits_quicker(self):
        assert transit_hours(100.0, CTV) < transit_hours(100.0, AHTS)

    def test_window_includes_transit_and_contingency(self):
        operation = ANNUAL_OPERATIONS[0]
        required = window_hours_required(operation, 100.0)
        assert required > operation.on_site_hours
        assert required > transit_hours(100.0, operation.vessel)

    def test_distance_lengthens_the_required_window(self):
        """The mechanism by which distance penalises a site twice."""
        operation = ANNUAL_OPERATIONS[0]
        assert window_hours_required(operation, 200.0) > window_hours_required(
            operation, 20.0
        )


class TestFindWindows:
    def test_finds_a_calm_stretch(self):
        index = pd.date_range("2016-01-01", periods=100, freq="1h", tz="UTC")
        hs = pd.Series(np.full(100, 3.0), index=index)
        hs.iloc[20:50] = 1.0
        windows = find_windows(hs, threshold_m=1.5, min_hours=10)
        assert len(windows) == 1
        assert windows.iloc[0]["duration_hours"] == pytest.approx(30, abs=1)

    def test_ignores_windows_that_are_too_short(self):
        index = pd.date_range("2016-01-01", periods=100, freq="1h", tz="UTC")
        hs = pd.Series(np.full(100, 3.0), index=index)
        hs.iloc[20:25] = 1.0
        assert find_windows(hs, 1.5, min_hours=10).empty

    def test_a_gap_breaks_a_window(self):
        """Missing data cannot be asserted to have been calm."""
        index = pd.date_range("2016-01-01", periods=60, freq="1h", tz="UTC")
        hs = pd.Series(np.full(60, 1.0), index=index)
        hs.iloc[30] = np.nan
        windows = find_windows(hs, 1.5, min_hours=5)
        assert len(windows) == 2

    def test_longer_jobs_find_fewer_windows(self):
        """The non-linearity that makes distant sites disproportionately hard."""
        hs = make_hs()
        short = len(find_windows(hs, 1.5, min_hours=6))
        long = len(find_windows(hs, 1.5, min_hours=48))
        assert long < short

    def test_empty_input(self):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        assert find_windows(empty, 1.5, 6).empty

    def test_works_on_a_read_only_backed_series(self):
        """Regression: pandas 3 copy-on-write returns read-only arrays.

        `(hs <= threshold).to_numpy()` hands back a read-only view under
        pandas 3, so the in-place `&=` that follows raised "output array is
        read-only". It passed locally on pandas 2.3 and failed in CI on 3.0 -
        the notebook run was the first thing to catch it.
        """
        values = np.full(200, 1.0)
        values[50:100] = 3.0
        values.flags.writeable = False
        index = pd.date_range("2016-01-01", periods=200, freq="1h", tz="UTC")
        hs = pd.Series(values, index=index)

        windows = find_windows(hs, threshold_m=1.5, min_hours=10)
        assert not windows.empty


class TestAccessibility:
    def test_fraction_below_threshold(self):
        index = pd.date_range("2016-01-01", periods=100, freq="1h", tz="UTC")
        hs = pd.Series(np.concatenate([np.full(40, 1.0), np.full(60, 3.0)]), index=index)
        assert accessible_fraction(hs, 1.5) == pytest.approx(0.4)

    def test_higher_threshold_is_more_accessible(self):
        hs = make_hs()
        assert accessible_fraction(hs, 2.5) > accessible_fraction(hs, 1.5)

    def test_waiting_time_grows_with_job_length(self):
        hs = make_hs()
        short = expected_waiting_hours(hs, 2.0, 12)
        long = expected_waiting_hours(hs, 2.0, 72)
        assert long > short

    def test_waiting_time_grows_as_the_limit_tightens(self):
        hs = make_hs()
        assert expected_waiting_hours(hs, 1.5, 24) > expected_waiting_hours(hs, 2.5, 24)

    def test_rougher_site_waits_longer(self):
        """The core claim: energetic sites cost more to reach."""
        calm = make_hs(mean=1.5, seed=1)
        rough = make_hs(mean=3.5, seed=1)
        assert expected_waiting_hours(rough, 2.0, 24) > expected_waiting_hours(
            calm, 2.0, 24
        )

    def test_no_adequate_window_gives_nan(self):
        index = pd.date_range("2016-01-01", periods=200, freq="1h", tz="UTC")
        hs = pd.Series(np.full(200, 6.0), index=index)
        assert np.isnan(expected_waiting_hours(hs, 1.5, 24))

    def test_statistics_keys(self):
        stats = window_statistics(make_hs(), 2.0, 24)
        assert {"accessible_fraction", "windows_per_year", "expected_wait_hours"} <= set(
            stats
        )

    def test_winter_is_less_accessible_than_summer(self):
        """On this coast the energetic season is the inaccessible season."""
        table = seasonal_accessibility(make_hs(), 2.0, 24)
        assert (
            table.loc["winter (DJF)", "accessible_fraction"]
            < table.loc["summer (JJA)", "accessible_fraction"]
        )


class TestEnergy:
    def test_rated_power_clips_output(self):
        index = pd.date_range("2016-01-01", periods=8766, freq="1h", tz="UTC")
        flux = pd.Series(np.full(8766, 1_000_000.0), index=index)  # 1000 kW/m
        device = DeviceSpec(capture_width_m=10, efficiency=0.35, rated_power_kw=500)
        result = annual_energy_mwh(flux, device)
        # Raw would be 3500 kW; the generator caps at 500.
        assert result["annual_energy_mwh"] == pytest.approx(500 * 8766 / 1000, rel=0.01)
        assert result["clipped_fraction"] > 0.8

    def test_capacity_factor_is_bounded(self):
        hs = make_hs()
        flux = 490.0 * hs**2 * 8.0
        result = annual_energy_mwh(flux, DeviceSpec(), hs)
        assert 0.0 <= result["capacity_factor"] <= 1.0

    def test_survival_shutdown_removes_energy(self):
        hs = make_hs(mean=5.0, amplitude=3.0)
        flux = 490.0 * hs**2 * 8.0
        with_cutout = annual_energy_mwh(flux, DeviceSpec(survival_hs_m=6.0), hs)
        without = annual_energy_mwh(flux, DeviceSpec(survival_hs_m=99.0), hs)
        assert with_cutout["annual_energy_mwh"] < without["annual_energy_mwh"]
        assert with_cutout["survival_fraction"] > 0

    def test_empty_input(self):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        assert np.isnan(annual_energy_mwh(empty, DeviceSpec())["annual_energy_mwh"])


class TestCosts:
    def test_standby_is_charged(self):
        operation = ANNUAL_OPERATIONS[0]
        dry = operation_cost(operation, 50.0, waiting_hours=0.0)
        waiting = operation_cost(operation, 50.0, waiting_hours=48.0)
        assert waiting["total_usd"] > dry["total_usd"]

    def test_mooring_cost_rises_with_depth(self):
        shallow = mooring_capex(SiteLogistics(distance_km=50, water_depth_m=50))
        deep = mooring_capex(SiteLogistics(distance_km=50, water_depth_m=200))
        assert deep > shallow

    def test_om_cost_rises_with_distance(self):
        hs = make_hs()
        near = annual_om_cost(hs, SiteLogistics(distance_km=20))
        far = annual_om_cost(hs, SiteLogistics(distance_km=200))
        assert far["total_annual_usd"] > near["total_annual_usd"]

    def test_sea_state_penalty_lands_on_downtime_not_vessel_hire(self):
        """Two sites the same distance out, different wave climates.

        With the standby cap in place both sites hit the same hire ceiling, so
        vessel cost barely separates them. The rough site is penalised through
        downtime instead - the device waits, the vessel does not. Asserting the
        old relationship would re-enshrine the bug the cap fixed.
        """
        site = SiteLogistics(distance_km=60)
        calm_hs = make_hs(mean=1.5, seed=2)
        rough_hs = make_hs(mean=3.5, seed=2)

        assert annual_om_cost(rough_hs, site)["total_annual_usd"] >= annual_om_cost(
            calm_hs, site
        )["total_annual_usd"]
        assert downtime_fraction(rough_hs, site) > downtime_fraction(calm_hs, site)

    def test_standby_is_capped(self):
        """A vessel is not kept on hire for a thousand hours."""
        operation = ANNUAL_OPERATIONS[0]
        modest = operation_cost(operation, 50.0, waiting_hours=40.0)
        absurd = operation_cost(operation, 50.0, waiting_hours=4000.0)
        assert absurd["charged_wait_hours"] == DEFAULT_STANDBY_CAP_HOURS
        assert absurd["uncharged_wait_hours"] == pytest.approx(4000.0 - DEFAULT_STANDBY_CAP_HOURS)
        assert absurd["total_usd"] < modest["total_usd"] * 2

    def test_uncapped_standby_would_be_absurd(self):
        """Guards the reason the cap exists.

        Without it, one operation at an exposed site costs more than the
        device earns in a year.
        """
        operation = ANNUAL_OPERATIONS[2]
        uncapped = operation_cost(
            operation, 150.0, waiting_hours=2500.0, standby_cap_hours=1e9
        )
        assert uncapped["total_usd"] > 1_000_000


class TestForecastValue:
    """Forecast skill enters the economics through aborted sailings."""

    def test_perfect_forecast_costs_nothing_extra(self):
        assert false_start_multiplier(0.0, 2.0) == 1.0

    def test_worse_forecast_means_more_sailings(self):
        assert false_start_multiplier(0.8, 2.0) >= false_start_multiplier(0.2, 2.0)

    def test_multiplier_is_at_least_one(self):
        for rmse in (0.0, 0.1, 0.5, 2.0):
            assert false_start_multiplier(rmse, 2.0) >= 1.0

    def test_forecast_error_raises_om_cost(self):
        """The dollar value of the forecast work, made explicit."""
        hs = make_hs()
        site = SiteLogistics(distance_km=80)
        perfect = annual_om_cost(hs, site, forecast_rmse_m=0.0)
        gefs = annual_om_cost(hs, site, forecast_rmse_m=0.41)
        assert gefs["total_annual_usd"] > perfect["total_annual_usd"]

    def test_downtime_rises_with_distance(self):
        hs = make_hs()
        assert downtime_fraction(hs, SiteLogistics(distance_km=200)) > downtime_fraction(
            hs, SiteLogistics(distance_km=20)
        )

    def test_downtime_is_a_fraction(self):
        hs = make_hs(mean=4.0)
        assert 0.0 <= downtime_fraction(hs, SiteLogistics(distance_km=150)) <= 1.0


class TestSiteEvaluation:
    def test_returns_the_headline_quantities(self):
        hs = make_hs()
        flux = 490.0 * hs**2 * 8.0
        result = evaluate_site(hs, flux, SiteLogistics(distance_km=50))
        assert {"net_annual_usd", "gross_annual_usd", "om_annual_usd"} <= set(result)

    def test_net_is_below_gross(self):
        """The gap between them is the entire point of this module."""
        hs = make_hs()
        flux = 490.0 * hs**2 * 8.0
        result = evaluate_site(hs, flux, SiteLogistics(distance_km=50))
        assert result["net_annual_usd"] < result["gross_annual_usd"]

    def test_distance_reduces_net_revenue_at_equal_resource(self):
        """Identical wave climate, different distance: net must fall."""
        hs = make_hs()
        flux = 490.0 * hs**2 * 8.0
        near = evaluate_site(hs, flux, SiteLogistics(distance_km=20))
        far = evaluate_site(hs, flux, SiteLogistics(distance_km=250))
        assert near["gross_annual_usd"] == pytest.approx(far["gross_annual_usd"])
        assert near["net_annual_usd"] > far["net_annual_usd"]

    def test_energetic_sites_pay_a_penalty_a_resource_ranking_misses(self):
        """The claim that justifies the whole module.

        A more energetic site earns more gross revenue *and* is harder to
        reach. Gross ranking sees only the first; downtime carries the second,
        so the net gap is always narrower than the gross gap.
        """
        calm = make_hs(mean=2.0, seed=3)
        rough = make_hs(mean=3.2, seed=3)
        site = SiteLogistics(distance_km=220)

        calm_result = evaluate_site(calm, 490.0 * calm**2 * 8.0, site)
        rough_result = evaluate_site(rough, 490.0 * rough**2 * 8.0, site)

        assert rough_result["gross_annual_usd"] > calm_result["gross_annual_usd"]
        assert rough_result["downtime_fraction"] > calm_result["downtime_fraction"]

        gross_gap = rough_result["gross_annual_usd"] - calm_result["gross_annual_usd"]
        net_gap = rough_result["net_annual_usd"] - calm_result["net_annual_usd"]
        assert net_gap < gross_gap

    def test_price_scales_revenue(self):
        hs = make_hs()
        flux = 490.0 * hs**2 * 8.0
        site = SiteLogistics(distance_km=50)
        cheap = evaluate_site(hs, flux, site, market=MarketSpec(energy_price_usd_per_mwh=60))
        dear = evaluate_site(hs, flux, site, market=MarketSpec(energy_price_usd_per_mwh=240))
        assert dear["gross_annual_usd"] == pytest.approx(
            4 * cheap["gross_annual_usd"], rel=0.01
        )
