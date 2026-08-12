"""Site economics: energy, availability, cost, and what is left over.

Combines three things the rest of the codebase produces separately:

  * **resource** - wave power flux from :mod:`src.processing.wave_power_flux`
  * **access** - weather windows from :mod:`src.economics.accessibility`
  * **logistics** - vessel costs from :mod:`src.economics.deployment`

The point of combining them is that ranking on resource alone is misleading.
Gross energy revenue rises with wave power; so does the cost of servicing the
device, and so does the time it spends broken and unreachable. Net revenue is
the difference of two quantities that move together, and which site wins is not
obvious in advance.

Every economic parameter is a documented planning assumption. They are here to
support *relative* comparison between sites, which is robust to getting the
absolute level wrong, and not to produce a bankable number.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.economics.accessibility import window_statistics
from src.economics.deployment import (
    ANNUAL_OPERATIONS,
    DEFAULT_STANDBY_CAP_HOURS,
    DEPLOYMENT_OPERATIONS,
    RETRIEVAL_OPERATIONS,
    SiteLogistics,
    false_start_multiplier,
    mooring_capex,
    operation_cost,
    scale_operations,
    window_hours_required,
)

HOURS_PER_YEAR = 8766.0


def _wait_hours(stats: dict) -> tuple:
    """Expected wait, plus whether the job is impossible at this site.

    :func:`~src.economics.accessibility.expected_waiting_hours` returns NaN
    when the record contains no window long enough to do the job *at all*.
    Every call site here used to read that NaN as a zero wait, which scored
    the least accessible possible site as perfectly accessible - and it did so
    silently, because NaN-to-zero looks like ordinary missing-data hygiene.

    The failure is not hypothetical: it inverts the ranking exactly where the
    model is supposed to be discriminating. A site whose required window is
    just barely achievable reports a long wait and a large penalty, while a
    site whose window is fractionally too long reports no wait and none.

    Returns:
        ``(wait_hours, unservicable)``. ``wait_hours`` is NaN when
        unservicable; callers decide what that costs.
    """
    # Keyed on the window count rather than on the wait being NaN. The two are
    # equivalent - every window contains at least its own start timestep, and
    # timesteps inside a window are assigned a zero wait, so a non-empty window
    # set always yields a finite mean - but the count says what is actually
    # meant. Censoring waits at the end of each run of observation looked like
    # it might make NaN reachable with windows present; it cannot, and a
    # fallback for that case would have been unreachable code.
    if not stats["n_windows"]:
        return float("nan"), True
    return float(stats["expected_wait_hours"]), False


@dataclass(frozen=True)
class DeviceSpec:
    """The energy converter being sited.

    Args:
        name: Label.
        capture_width_m: Metres of wave crest the device intercepts. For a
            point absorber this can exceed the physical beam, but a
            conservative value is used here.
        efficiency: Wave-to-wire conversion efficiency, all losses included.
            35% is optimistic-but-defensible for a well-developed WEC; many
            prototypes achieve considerably less.
        rated_power_kw: Generator ceiling. Real devices saturate in big seas
            rather than scaling indefinitely with H^2, and ignoring that
            overstates revenue at exactly the energetic sites a naive study
            already favours.
        survival_hs_m: Sea state above which the device enters survival mode and
            stops generating.
        capex_usd: Device capital cost, excluding moorings.
        design_life_years: Deployment life used to annualise capital.
    """

    name: str = "Panthalassa 200 ft buoy"
    capture_width_m: float = 20.0
    efficiency: float = 0.35
    rated_power_kw: float = 1_000.0
    survival_hs_m: float = 8.0
    capex_usd: float = 12_000_000.0
    design_life_years: float = 20.0


@dataclass(frozen=True)
class MarketSpec:
    """Price and financing assumptions.

    Args:
        energy_price_usd_per_mwh: Flat offtake price. Wholesale PNW power sits
            well below this; the figure assumes a renewable premium or a
            demonstration PPA.
        discount_rate: Used only to annualise capital.
    """

    energy_price_usd_per_mwh: float = 120.0
    discount_rate: float = 0.08


def annual_energy_mwh(
    power_flux_w_per_m: pd.Series,
    device: DeviceSpec,
    hs: pd.Series | None = None,
) -> dict:
    """Annual energy production from a power-flux time series.

    Applies two limits a flat ``flux x width x efficiency`` calculation misses,
    both of which bite hardest at energetic sites:

    * **rated power** - the generator saturates
    * **survival shutdown** - the device stops generating in extreme seas

    Args:
        power_flux_w_per_m: Wave power flux time series, W/m.
        device: Device specification.
        hs: Significant wave height, for the survival cut-out. Omit to skip it.

    Returns:
        dict with annual energy, capacity factor, and the share of theoretical
        energy lost to each limit.
    """
    flux = power_flux_w_per_m.dropna()
    if flux.empty:
        return {
            "annual_energy_mwh": float("nan"),
            "capacity_factor": float("nan"),
            "clipped_fraction": float("nan"),
            "survival_fraction": float("nan"),
        }

    raw_kw = flux * device.capture_width_m * device.efficiency / 1000.0
    theoretical = raw_kw.copy()

    clipped = raw_kw.clip(upper=device.rated_power_kw)
    clipped_loss = (theoretical - clipped).sum()

    if hs is not None:
        shutdown = hs.reindex(clipped.index) > device.survival_hs_m
        survival_loss = clipped[shutdown.fillna(False)].sum()
        clipped = clipped.mask(shutdown.fillna(False), 0.0)
    else:
        survival_loss = 0.0

    step_hours = _median_step_hours(clipped.index)
    record_hours = len(clipped) * step_hours
    years = record_hours / HOURS_PER_YEAR

    energy_mwh = clipped.sum() * step_hours / 1000.0
    annual = energy_mwh / years if years else float("nan")

    theoretical_total = theoretical.sum()
    return {
        "annual_energy_mwh": annual,
        "capacity_factor": (
            annual * 1000.0 / (device.rated_power_kw * HOURS_PER_YEAR)
            if device.rated_power_kw
            else float("nan")
        ),
        "clipped_fraction": (
            float(clipped_loss / theoretical_total) if theoretical_total else 0.0
        ),
        "survival_fraction": (
            float(survival_loss / theoretical_total) if theoretical_total else 0.0
        ),
    }


def _median_step_hours(index) -> float:
    import numpy as np

    if len(index) < 2:
        return 1.0
    steps = np.diff(index.to_numpy()).astype("timedelta64[s]").astype(float) / 3600.0
    return float(np.median(steps))


def annual_om_cost(
    hs: pd.Series,
    site: SiteLogistics,
    forecast_rmse_m: float = 0.0,
    operations=None,
) -> dict:
    """Recurring vessel cost per year.

    Each operation's required window is computed from its own duration plus
    transit to this site, then the wave record is asked how long a vessel would
    typically wait for a window that long.

    Only standby up to the cap is charged as hire. Beyond it the wait is real
    but is borne as **downtime**, not cost, because a vessel mobilised against
    a forecast does not sit on hire for months. That split matters: with the
    cap in place, a rougher sea state raises cost only weakly, and shows up
    instead in :func:`downtime_fraction`. Before the cap existed this function
    returned annual O&M an order of magnitude above the device's gross revenue.

    Mobilisation is charged once per operation *type* per year rather than per
    occurrence, on the assumption that repeat visits within a year share a
    campaign.

    Args:
        hs: Significant wave height record for the site.
        site: Distance and depth context.
        forecast_rmse_m: Forecast RMSE at the mobilisation lead time. Non-zero
            values add the cost of sailings that abort because the forecast was
            wrong - the one place forecast skill carries a dollar value.
    """
    # Taken as a parameter rather than read from the module global: a caller
    # rebinding src.economics.deployment.ANNUAL_OPERATIONS cannot affect the
    # name this module bound at import time. A sensitivity analysis that tried
    # exactly that silently produced baseline numbers for every scenario.
    operations = ANNUAL_OPERATIONS if operations is None else operations

    rows = []
    seen_vessels = set()

    for operation in operations:
        # Distance is a property of (site, vessel), not of the site: a crew
        # boat and an anchor handler sail from different ports.
        distance = site.distance_for(operation.vessel)
        required = window_hours_required(operation, distance)
        stats = window_statistics(hs, operation.vessel.max_hs_operate_m, required)
        wait, unservicable = _wait_hours(stats)
        # No window ever long enough: the vessel still mobilises, sits out the
        # standby cap and demobilises. Charging the cap keeps the cost finite
        # while making the site the most expensive rather than the cheapest.
        wait = DEFAULT_STANDBY_CAP_HOURS if unservicable else wait

        first_of_vessel = operation.vessel.name not in seen_vessels
        seen_vessels.add(operation.vessel.name)

        cost = operation_cost(
            operation,
            distance,
            waiting_hours=wait,
            include_mobilisation=first_of_vessel,
        )
        cost["port"] = site.port_for(operation.vessel)
        cost["distance_km"] = distance
        # Mobilising against an imperfect forecast means some sailings abort
        # and are paid for anyway.
        multiplier = false_start_multiplier(
            forecast_rmse_m, operation.vessel.max_hs_operate_m
        )
        cost["false_start_multiplier"] = multiplier
        cost["per_year"] = operation.per_year
        cost["annual_usd"] = (
            cost["vessel_cost_usd"] * operation.per_year * multiplier
            + cost["mobilisation_usd"]
        )
        cost["accessible_fraction"] = stats["accessible_fraction"]
        cost["windows_per_year"] = stats["windows_per_year"]
        rows.append(cost)

    detail = pd.DataFrame(rows)
    return {
        "total_annual_usd": float(detail["annual_usd"].sum()),
        "total_waiting_hours": float(
            (detail["waiting_hours"] * detail["per_year"]).sum()
        ),
        "detail": detail,
    }


def campaign_cost(hs: pd.Series, site: SiteLogistics, operations) -> float:
    """One-off cost of a deployment or retrieval campaign, with standby.

    ``per_year`` is read as a trip count here, not an annual rate, so an array
    that needs one tow-out per device is charged for each of them. Mobilisation
    is still charged once per vessel, which is where an array saves.
    """
    total = 0.0
    seen_vessels = set()
    for operation in operations:
        distance = site.distance_for(operation.vessel)
        required = window_hours_required(operation, distance)
        stats = window_statistics(hs, operation.vessel.max_hs_operate_m, required)
        wait, unservicable = _wait_hours(stats)
        wait = DEFAULT_STANDBY_CAP_HOURS if unservicable else wait
        first_of_vessel = operation.vessel.name not in seen_vessels
        seen_vessels.add(operation.vessel.name)
        cost = operation_cost(
            operation, distance, wait, include_mobilisation=first_of_vessel
        )
        total += cost["vessel_cost_usd"] * operation.per_year + cost["mobilisation_usd"]
    return total


def downtime_fraction(
    hs: pd.Series, site: SiteLogistics, operations=None
) -> float:
    """Share of the year the device is expected to be down awaiting repair.

    A failure does not end when a vessel is dispatched; it ends when the
    weather lets one work. Expected downtime is therefore the unscheduled
    failure rate multiplied by the expected wait for a suitable window - a
    quantity that depends entirely on the local wave climate and the distance
    to port, and that a resource-only ranking ignores completely.
    """
    operations = ANNUAL_OPERATIONS if operations is None else operations
    unscheduled = [op for op in operations if "Unscheduled" in op.name]
    if not unscheduled:
        return 0.0

    total_down = 0.0
    for operation in unscheduled:
        required = window_hours_required(operation, site.distance_for(operation.vessel))
        stats = window_statistics(hs, operation.vessel.max_hs_operate_m, required)
        wait, unservicable = _wait_hours(stats)
        if unservicable:
            # A repair that can never be attempted means the device is down
            # for good. Full downtime, not zero.
            return 1.0
        total_down += (wait + required) * operation.per_year

    return min(total_down / HOURS_PER_YEAR, 1.0)


def evaluate_site(
    hs: pd.Series,
    power_flux_w_per_m: pd.Series,
    site: SiteLogistics,
    device: DeviceSpec | None = None,
    market: MarketSpec | None = None,
    operations=None,
    forecast_rmse_m: float = 0.0,
    n_devices: int = 1,
) -> dict:
    """Full economic assessment of one candidate site.

    Args:
        n_devices: Devices in the array. Costs that are shared across an array
            are shared here: mobilisation is charged per campaign rather than
            per device, and campaign operations service several devices per
            sailing. Costs that are not shared are not: breakdowns scale with
            the device count, as do device capex, moorings and tow-out.

    Returns:
        dict of resource, access, cost and net-revenue quantities. The headline
        is ``net_annual_usd`` for the whole array, with ``net_per_device_usd``
        alongside it; ``gross_annual_usd`` is what a resource-only study would
        have reported, and the gap between them is the point of this module.
    """
    device = device or DeviceSpec()
    market = market or MarketSpec()

    base_operations = ANNUAL_OPERATIONS if operations is None else operations
    fleet_operations = scale_operations(base_operations, n_devices)

    energy = annual_energy_mwh(power_flux_w_per_m, device, hs)
    array_energy_mwh = energy["annual_energy_mwh"] * n_devices

    # Downtime is a *per-device* quantity and uses the unscaled operations: one
    # device's expected outage does not depend on how many neighbours it has.
    # Scaling the failure rate here instead would multiply each device's own
    # downtime by the size of the array. (Vessel contention between
    # simultaneous failures is not modelled.)
    downtime = downtime_fraction(hs, site, base_operations)
    delivered_mwh = array_energy_mwh * (1.0 - downtime)

    gross = array_energy_mwh * market.energy_price_usd_per_mwh
    delivered = delivered_mwh * market.energy_price_usd_per_mwh

    om = annual_om_cost(hs, site, forecast_rmse_m, fleet_operations)
    # One tow-out per device: you move a 61 m spar one at a time, so these
    # scale in trips rather than in on-site hours. Mobilisation is still
    # charged once.
    deploy = campaign_cost(
        hs, site, scale_operations(DEPLOYMENT_OPERATIONS, n_devices, 1)
    )
    retrieve = campaign_cost(
        hs, site, scale_operations(RETRIEVAL_OPERATIONS, n_devices, 1)
    )
    moorings = mooring_capex(site) * n_devices

    capex_total = device.capex_usd * n_devices + moorings + deploy
    annualised_capex = (
        capex_total / device.design_life_years
        + retrieve / device.design_life_years
    )

    # The tow-out distance and the service distance are different numbers and
    # answer different questions: the first sets capex, the second sets O&M and
    # downtime. Reporting only the first is what made Neah Bay look unservicable.
    service_vessels = {op.vessel for op in base_operations}
    service_distance = (
        max(site.distance_for(v) for v in service_vessels)
        if service_vessels
        else site.distance_km
    )

    net = delivered - om["total_annual_usd"] - annualised_capex

    return {
        "n_devices": n_devices,
        "distance_km": site.distance_km,
        "service_distance_km": service_distance,
        "port_name": site.port_name,
        "mean_flux_kw_per_m": float(power_flux_w_per_m.mean() / 1000.0),
        "annual_energy_mwh": array_energy_mwh,
        "capacity_factor": energy["capacity_factor"],
        "clipped_fraction": energy["clipped_fraction"],
        "survival_fraction": energy["survival_fraction"],
        "downtime_fraction": downtime,
        "gross_annual_usd": gross,
        "delivered_annual_usd": delivered,
        "om_annual_usd": om["total_annual_usd"],
        "annualised_capex_usd": annualised_capex,
        "net_annual_usd": net,
        "net_per_device_usd": net / n_devices,
        "om_per_device_usd": om["total_annual_usd"] / n_devices,
        "deployment_campaign_usd": deploy,
        "retrieval_campaign_usd": retrieve,
        "mooring_capex_usd": moorings,
        "standby_hours_per_year": om["total_waiting_hours"],
    }
