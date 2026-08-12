"""Inverting the siting question: what would have to be true to break even?

The forward model answers "does this site make money" and the answer is no, by
$690k a year at the best of three. That is where a study usually stops, and it
is the least useful form of the result: it says the gap exists without saying
how wide it is.

This module asks the inverse. Holding the wave climate and the logistics
fixed - the parts grounded in data - what capex, price, capture width or array
size would put net revenue at zero? The answer converts "loses money" into a
factor, and a factor is something you can compare against how much a technology
plausibly improves.

Two properties make this cheap and exact rather than a search:

* Net revenue is **linear in price** (gross scales with it) and **linear in
  capex** (it only enters through annualisation). Both break-evens solve in
  closed form.
* The expensive part of the model - weather windows, waiting times, downtime -
  depends only on the wave record, the site and the operation set. It does
  *not* depend on price, capex or capture width. So it is computed once per
  (site, array size) and reused across an entire sweep.

That second point is what makes a two-dimensional sweep affordable: window
statistics over a nine-year hourly record are the cost of this whole model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.economics.deployment import (
    ANNUAL_OPERATIONS,
    DEPLOYMENT_OPERATIONS,
    RETRIEVAL_OPERATIONS,
    SiteLogistics,
    mooring_capex,
    scale_operations,
)
from src.economics.revenue import (
    DeviceSpec,
    MarketSpec,
    annual_energy_mwh,
    annual_om_cost,
    campaign_cost,
    downtime_fraction,
)


@dataclass(frozen=True)
class WeatherProfile:
    """The weather-dependent part of a site's economics, computed once.

    Everything here is invariant to price, capex and capture width, which is
    why a sweep over those costs nothing beyond arithmetic.
    """

    site: SiteLogistics
    n_devices: int
    downtime_fraction: float
    om_annual_usd: float
    deployment_usd: float
    retrieval_usd: float
    mooring_usd: float

    @property
    def fixed_capex_usd(self) -> float:
        """Capex that does not depend on the device's unit price."""
        return self.mooring_usd + self.deployment_usd


def weather_profile(
    hs: pd.Series,
    site: SiteLogistics,
    n_devices: int = 1,
    operations=None,
    forecast_rmse_m: float = 0.0,
) -> WeatherProfile:
    """Compute the expensive, price-independent half of the model."""
    base_operations = ANNUAL_OPERATIONS if operations is None else operations
    fleet_operations = scale_operations(base_operations, n_devices)

    return WeatherProfile(
        site=site,
        n_devices=n_devices,
        # Per-device, from the unscaled operations - see evaluate_site.
        downtime_fraction=downtime_fraction(hs, site, base_operations),
        om_annual_usd=annual_om_cost(
            hs, site, forecast_rmse_m, fleet_operations
        )["total_annual_usd"],
        deployment_usd=campaign_cost(
            hs, site, scale_operations(DEPLOYMENT_OPERATIONS, n_devices, 1)
        ),
        retrieval_usd=campaign_cost(
            hs, site, scale_operations(RETRIEVAL_OPERATIONS, n_devices, 1)
        ),
        mooring_usd=mooring_capex(site) * n_devices,
    )


def net_annual_usd(
    profile: WeatherProfile,
    power_flux_w_per_m: pd.Series,
    hs: pd.Series,
    device: DeviceSpec,
    market: MarketSpec,
) -> dict:
    """Net revenue for one configuration, reusing a precomputed profile.

    Agrees with :func:`~src.economics.revenue.evaluate_site` by construction -
    it is the same arithmetic with the weather half lifted out. A test asserts
    the two match, because a fast path that quietly disagrees with the slow one
    is worse than no fast path.
    """
    n = profile.n_devices
    energy = annual_energy_mwh(power_flux_w_per_m, device, hs)
    array_mwh = energy["annual_energy_mwh"] * n
    delivered_mwh = array_mwh * (1.0 - profile.downtime_fraction)

    capex_total = device.capex_usd * n + profile.fixed_capex_usd
    annualised_capex = (
        capex_total + profile.retrieval_usd
    ) / device.design_life_years

    delivered_usd = delivered_mwh * market.energy_price_usd_per_mwh
    net = delivered_usd - profile.om_annual_usd - annualised_capex

    return {
        "annual_energy_mwh": array_mwh,
        "delivered_mwh": delivered_mwh,
        "capacity_factor": energy["capacity_factor"],
        "delivered_annual_usd": delivered_usd,
        "om_annual_usd": profile.om_annual_usd,
        "annualised_capex_usd": annualised_capex,
        "net_annual_usd": net,
        "net_per_device_usd": net / n,
    }


def breakeven_price_usd_per_mwh(
    profile: WeatherProfile,
    power_flux_w_per_m: pd.Series,
    hs: pd.Series,
    device: DeviceSpec,
) -> float:
    """Offtake price at which net revenue is zero.

    Exact, not searched: revenue is proportional to price and nothing else in
    the model depends on it, so this is (annual cost) / (delivered MWh).
    """
    n = profile.n_devices
    energy = annual_energy_mwh(power_flux_w_per_m, device, hs)
    delivered_mwh = (
        energy["annual_energy_mwh"] * n * (1.0 - profile.downtime_fraction)
    )
    if not delivered_mwh or np.isnan(delivered_mwh):
        return float("nan")

    annualised_capex = (
        device.capex_usd * n + profile.fixed_capex_usd + profile.retrieval_usd
    ) / device.design_life_years
    return (profile.om_annual_usd + annualised_capex) / delivered_mwh


def breakeven_device_capex_usd(
    profile: WeatherProfile,
    power_flux_w_per_m: pd.Series,
    hs: pd.Series,
    device: DeviceSpec,
    market: MarketSpec,
) -> float:
    """Per-device capital cost at which net revenue is zero.

    May come back **negative**, and that is a real answer rather than an error:
    it means the device would have to be free *and* come with a subsidy, since
    operating cost alone exceeds revenue. Callers should report the sign rather
    than clip it.
    """
    n = profile.n_devices
    energy = annual_energy_mwh(power_flux_w_per_m, device, hs)
    delivered_usd = (
        energy["annual_energy_mwh"]
        * n
        * (1.0 - profile.downtime_fraction)
        * market.energy_price_usd_per_mwh
    )
    surplus = delivered_usd - profile.om_annual_usd
    capex_budget = surplus * device.design_life_years
    return (
        capex_budget - profile.fixed_capex_usd - profile.retrieval_usd
    ) / n


def array_sweep(
    hs: pd.Series,
    power_flux_w_per_m: pd.Series,
    site: SiteLogistics,
    device_counts=(1, 2, 5, 10, 20),
    device: DeviceSpec | None = None,
    market: MarketSpec | None = None,
    forecast_rmse_m: float = 0.0,
) -> pd.DataFrame:
    """How the economics change with array size.

    One weather profile per array size - the expensive part - then arithmetic.
    """
    device = device or DeviceSpec()
    market = market or MarketSpec()

    rows = []
    for n in device_counts:
        profile = weather_profile(hs, site, n, forecast_rmse_m=forecast_rmse_m)
        result = net_annual_usd(profile, power_flux_w_per_m, hs, device, market)
        rows.append(
            {
                "n_devices": n,
                "om_per_device_$k": profile.om_annual_usd / n / 1e3,
                "capex_per_device_$k": (
                    device.capex_usd + profile.fixed_capex_usd / n
                )
                / 1e3,
                "downtime_%": 100 * profile.downtime_fraction,
                "net_$k": result["net_annual_usd"] / 1e3,
                "net_per_device_$k": result["net_per_device_usd"] / 1e3,
                "breakeven_price_$/MWh": breakeven_price_usd_per_mwh(
                    profile, power_flux_w_per_m, hs, device
                ),
                "breakeven_capex_$M": breakeven_device_capex_usd(
                    profile, power_flux_w_per_m, hs, device, market
                )
                / 1e6,
            }
        )
    return pd.DataFrame(rows).set_index("n_devices")


def breakeven_surface(
    hs: pd.Series,
    power_flux_w_per_m: pd.Series,
    site: SiteLogistics,
    capture_widths=(10, 20, 35, 50),
    prices=(60, 120, 250, 400),
    n_devices: int = 1,
    device: DeviceSpec | None = None,
    forecast_rmse_m: float = 0.0,
) -> pd.DataFrame:
    """Break-even device capex over a grid of capture width and price.

    Reads as: "at this capture width and this price, the device may cost at
    most this much and still break even." Compare the cells against the $12M
    baseline to see which combinations are even in the right universe.
    """
    device = device or DeviceSpec()
    profile = weather_profile(
        hs, site, n_devices, forecast_rmse_m=forecast_rmse_m
    )

    from dataclasses import replace

    grid = {}
    for width in capture_widths:
        wide = replace(device, capture_width_m=width)
        grid[width] = {
            price: breakeven_device_capex_usd(
                profile,
                power_flux_w_per_m,
                hs,
                wide,
                MarketSpec(energy_price_usd_per_mwh=price),
            )
            / 1e6
            for price in prices
        }

    frame = pd.DataFrame(grid).T
    frame.index.name = "capture_width_m"
    frame.columns = [f"${p}/MWh" for p in prices]
    return frame
