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
from src.config import (
    NDBC_STATIONS,
    PORT_CLASS_HEAVY,
    PORT_CLASS_LIGHT,
    PORT_CLASS_WORKBOAT,
    Port,
    great_circle_km,
    nearest_port,
)
from src.economics.deployment import (
    AHTS,
    ANNUAL_OPERATIONS,
    CTV,
    DEFAULT_STANDBY_CAP_HOURS,
    SiteLogistics,
    site_logistics,
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

    def test_seasonal_output_omits_the_meaningless_wait(self):
        """Subsetting by month makes expected_wait_hours nonsense.

        The same season across years is concatenated, so the wait runs from
        February into the following December and counts nine months of summer
        as waiting. Real output showed 5,307 hours for a 90-day winter. The
        column is dropped rather than reported wrong.
        """
        table = seasonal_accessibility(make_hs(), 2.0, 24)
        assert "expected_wait_hours" not in table.columns
        assert "accessible_fraction" in table.columns
        assert "windows_per_year" in table.columns


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

    def test_multiplier_actually_varies_with_rmse(self):
        """Regression: it used to be constant.

        The margin was set equal to the RMSE, so the RMSE cancelled out of
        z = margin / (rmse * sqrt(2)) and every forecast quality returned
        1.19. Only visible once several RMSE values were printed side by side
        and every row was identical.
        """
        values = [false_start_multiplier(r, 1.75) for r in (0.2, 0.41, 0.6, 1.0)]
        assert len(set(round(v, 4) for v in values)) == len(values)
        assert values == sorted(values), "a worse forecast must cost more"
        assert values[-1] > values[0] * 1.2


class TestSensitivityPlumbing:
    """Operations must be passable, not patched onto a module global."""

    def test_scaled_day_rates_change_the_cost(self):
        from dataclasses import replace

        import src.economics.deployment as dep

        hs = make_hs()
        site = SiteLogistics(distance_km=80)
        dearer = tuple(
            replace(op, vessel=replace(op.vessel, day_rate_usd=op.vessel.day_rate_usd * 2))
            for op in dep.ANNUAL_OPERATIONS
        )
        base = annual_om_cost(hs, site)["total_annual_usd"]
        scaled = annual_om_cost(hs, site, operations=dearer)["total_annual_usd"]
        assert scaled > base * 1.5

    def test_failure_rate_changes_downtime(self):
        from dataclasses import replace

        import src.economics.deployment as dep

        hs = make_hs()
        site = SiteLogistics(distance_km=80)
        frequent = tuple(
            replace(op, per_year=op.per_year * 3 if "Unscheduled" in op.name else op.per_year)
            for op in dep.ANNUAL_OPERATIONS
        )
        assert downtime_fraction(hs, site, frequent) > downtime_fraction(hs, site)

    def test_evaluate_site_threads_operations_through(self):
        """The bug: rebinding the module global left every scenario identical."""
        from dataclasses import replace

        import src.economics.deployment as dep

        hs = make_hs()
        flux = 490.0 * hs**2 * 8.0
        site = SiteLogistics(distance_km=80)
        dearer = tuple(
            replace(op, vessel=replace(op.vessel, day_rate_usd=op.vessel.day_rate_usd * 3))
            for op in dep.ANNUAL_OPERATIONS
        )
        base = evaluate_site(hs, flux, site)["net_annual_usd"]
        scaled = evaluate_site(hs, flux, site, operations=dearer)["net_annual_usd"]
        assert scaled < base

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


class TestPortAssignment:
    """The support port is a property of (site, vessel), not of the site.

    The model originally costed every vessel at every site from a single base
    at Grays Harbor. That charged the Neah Bay buoy a 183 km transit for a
    routine inspection when the town of Neah Bay is 16 km away, and it did so
    at the one site where the forecast postprocessing showed real skill. These
    tests pin the corrected behaviour, including the constraint that made the
    original assumption tempting: a harbour that can take a crew boat cannot
    necessarily take an anchor handler.
    """

    def test_port_class_ordering_is_a_ceiling_not_a_list(self):
        small = Port("small", 48.0, -124.0, PORT_CLASS_LIGHT)
        big = Port("big", 48.0, -124.0, PORT_CLASS_HEAVY)

        assert small.can_host(PORT_CLASS_LIGHT)
        assert not small.can_host(PORT_CLASS_WORKBOAT)
        assert not small.can_host(PORT_CLASS_HEAVY)
        # A heavy port takes everything smaller too.
        assert all(
            big.can_host(c)
            for c in (PORT_CLASS_LIGHT, PORT_CLASS_WORKBOAT, PORT_CLASS_HEAVY)
        )

    def test_rejects_unknown_class(self):
        with pytest.raises(ValueError):
            Port("bad", 48.0, -124.0, "supertanker")
        with pytest.raises(ValueError):
            Port("ok", 48.0, -124.0).can_host("supertanker")

    def test_nearest_port_refuses_rather_than_downgrading(self):
        """No capable port must raise, not silently return an unusable one.

        Falling back to the closest port regardless of class would reintroduce
        precisely the error this module exists to correct, and it would do it
        invisibly.
        """
        light_only = (Port("tiny", 48.0, -124.0, PORT_CLASS_LIGHT),)
        with pytest.raises(ValueError):
            nearest_port(48.5, -124.7, PORT_CLASS_HEAVY, light_only)

    def test_anchor_handler_is_not_based_at_neah_bay(self):
        """The regression that motivated the whole change, from the other side.

        Neah Bay is by far the closest harbour to buoy 46087, so a
        class-blind nearest-port rule would put the tow-out spread there. It
        has no quay or laydown for a 61 m spar.
        """
        station = NDBC_STATIONS["46087"]
        site = site_logistics(station.latitude, station.longitude)

        assert "Neah Bay" not in site.port_for(AHTS)
        assert "Neah Bay" in site.port_for(CTV)

    def test_service_and_towout_distances_differ_by_an_order_of_magnitude(self):
        station = NDBC_STATIONS["46087"]
        site = site_logistics(station.latitude, station.longitude)

        assert site.distance_for(CTV) < 25.0
        assert site.distance_for(AHTS) > 90.0
        assert site.distance_for(AHTS) > 4 * site.distance_for(CTV)

    def test_single_distance_construction_still_works(self):
        """The old one-distance form must be unchanged for every vessel.

        Most of this file builds sites that way, and the fallback is what lets
        the two coexist.
        """
        site = SiteLogistics(distance_km=75.0)
        assert site.distance_for(AHTS) == 75.0
        assert site.distance_for(CTV) == 75.0
        assert site.port_for(AHTS) == site.port_name

    def test_correct_basing_shortens_the_required_window(self):
        """The point of the fix: a shorter transit needs a shorter calm spell.

        Long windows are disproportionately rarer than short ones, so this is
        the channel through which the port assumption reached downtime.
        """
        station = NDBC_STATIONS["46087"]
        correct = site_logistics(station.latitude, station.longitude)
        single_port = SiteLogistics(distance_km=correct.distance_for(AHTS))

        operation = [op for op in ANNUAL_OPERATIONS if "Unscheduled" in op.name][0]
        assert window_hours_required(
            operation, correct.distance_for(operation.vessel)
        ) < window_hours_required(operation, single_port.distance_km)

    def test_downtime_falls_when_vessels_are_based_correctly(self):
        hs = make_hs(mean=2.2, seed=11)
        station = NDBC_STATIONS["46087"]
        correct = site_logistics(station.latitude, station.longitude)
        single_port = SiteLogistics(distance_km=183.4, water_depth_m=correct.water_depth_m)

        assert downtime_fraction(hs, correct) < downtime_fraction(hs, single_port)

    def test_evaluate_site_reports_both_distances(self):
        hs = make_hs(seed=5)
        flux = 490.0 * hs**2 * 8.0
        station = NDBC_STATIONS["46087"]
        site = site_logistics(station.latitude, station.longitude)

        result = evaluate_site(hs, flux, site)
        assert result["distance_km"] == pytest.approx(site.distance_for(AHTS))
        assert result["service_distance_km"] < result["distance_km"]

    def test_great_circle_matches_known_separation(self):
        # One degree of latitude is ~111.2 km anywhere on the globe.
        assert great_circle_km(47.0, -124.0, 48.0, -124.0) == pytest.approx(111.2, abs=0.5)
        assert great_circle_km(47.0, -124.0, 47.0, -124.0) == 0.0


class TestUnservicableSites:
    """A site too rough to service must be the worst case, not the best.

    ``expected_waiting_hours`` returns NaN when the record holds no window long
    enough to do the job at all. Every consumer used to coerce that to a zero
    wait, so the least accessible possible site scored as perfectly
    accessible. The bug was invisible on the real buoy records - windows exist
    at all three - and only surfaced when a shorter required window made the
    comparison site cross the threshold in the opposite direction.
    """

    @staticmethod
    def _brutal_hs():
        # Never below any vessel's working limit, so no window ever opens.
        index = pd.date_range("2016-01-01", periods=24 * 365 * 2, freq="1h")
        return pd.Series(np.full(len(index), 6.0), index=index)

    def test_no_window_means_total_downtime(self):
        assert downtime_fraction(self._brutal_hs(), SiteLogistics(distance_km=60)) == 1.0

    def test_unservicable_is_worse_than_merely_difficult(self):
        """The ordering that the NaN-to-zero coercion inverted."""
        difficult = make_hs(mean=3.0, amplitude=1.5, seed=7)
        impossible = self._brutal_hs()
        site = SiteLogistics(distance_km=60)

        assert downtime_fraction(impossible, site) >= downtime_fraction(difficult, site)

    def test_unservicable_site_is_not_cheaper(self):
        difficult = make_hs(mean=3.0, amplitude=1.5, seed=7)
        site = SiteLogistics(distance_km=60)

        impossible_cost = annual_om_cost(self._brutal_hs(), site)["total_annual_usd"]
        difficult_cost = annual_om_cost(difficult, site)["total_annual_usd"]
        assert impossible_cost >= difficult_cost

    def test_charged_wait_is_capped_not_infinite(self):
        """Finite cost, so the model stays comparable across sites."""
        detail = annual_om_cost(self._brutal_hs(), SiteLogistics(distance_km=60))["detail"]
        assert (detail["charged_wait_hours"] <= DEFAULT_STANDBY_CAP_HOURS).all()
        assert np.isfinite(detail["annual_usd"]).all()

    def test_evaluate_site_survives_an_unservicable_record(self):
        hs = self._brutal_hs()
        result = evaluate_site(hs, 490.0 * hs**2 * 8.0, SiteLogistics(distance_km=60))
        assert result["downtime_fraction"] == 1.0
        assert result["delivered_annual_usd"] == pytest.approx(0.0)
        assert np.isfinite(result["net_annual_usd"])
