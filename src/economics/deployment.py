"""Deployment, retrieval and maintenance logistics for an offshore WEC.

Every number in this module is a **planning assumption**, not a vendor quote.
Each carries the reasoning behind it, and each is a named parameter you can
override. They are meant to be right to within a factor of about two - enough
to rank sites against each other, which is what a siting study needs - and not
to be quoted as a budget.

Why logistics belongs in a wave-resource study
----------------------------------------------
The obvious siting rule is "put it where the waves are biggest". That rule is
wrong, and this module is why.

Marine operations need a **weather window**: a stretch of calm long enough to
transit out, do the work, and get back. Wave height sets whether a vessel can
work at all. So the same seas that make a site energetic make it inaccessible,
and the cost of reaching it rises exactly where the revenue does.

Distance compounds it. A site 100 km out needs a window several hours longer
than one 20 km out for the same job, and long windows are disproportionately
rarer than short ones. Two sites with identical mean wave power can differ
substantially in delivered revenue once access is priced in.

The device
----------
A 200 ft (61 m) energy-harvesting buoy. At that length it is a spar or
articulated structure, not something a crew-transfer vessel handles: it needs
an anchor-handling tug for tow-out and mooring hookup. Routine inspection can
be done from a smaller vessel; a component failure needs the tug back.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Nautical mile in kilometres.
KM_PER_NM = 1.852


@dataclass(frozen=True)
class Vessel:
    """A vessel class with the properties that drive cost and accessibility.

    Args:
        name: Label.
        day_rate_usd: All-in chartered day rate including crew and fuel.
        transit_speed_kn: Service speed in knots.
        max_hs_operate_m: Significant wave height above which work stops. This
            is the number that couples the cost model to the wave climate.
        max_hs_transit_m: Wave height above which the vessel will not sail.
            Higher than the working limit - transiting is easier than lifting.
        mobilisation_usd: One-off cost to bring the vessel to the region and
            release it, independent of the job length.
    """

    name: str
    day_rate_usd: float
    transit_speed_kn: float
    max_hs_operate_m: float
    max_hs_transit_m: float
    mobilisation_usd: float = 0.0


#: Anchor-handling tug / supply vessel. Tow-out, mooring pre-lay and hookup, and
#: recovery of a 61 m structure. Day rates for AHTS tonnage in a non-peak market
#: sit in the low tens of thousands; 25k is a mid-range planning figure and the
#: single biggest cost driver in the model.
AHTS = Vessel(
    name="AHTS (anchor handler)",
    day_rate_usd=25_000,
    transit_speed_kn=11.0,
    max_hs_operate_m=2.0,
    max_hs_transit_m=3.5,
    mobilisation_usd=250_000,
)

#: Multi-purpose workboat for mooring inspection and light intervention.
MULTICAT = Vessel(
    name="Multicat workboat",
    day_rate_usd=12_000,
    transit_speed_kn=9.0,
    max_hs_operate_m=1.75,
    max_hs_transit_m=3.0,
    mobilisation_usd=75_000,
)

#: Crew transfer vessel for inspection and sensor work. Fast and cheap, but the
#: personnel-transfer limit is low, which is what makes an energetic site
#: expensive to visit.
CTV = Vessel(
    name="Crew transfer vessel",
    day_rate_usd=3_500,
    transit_speed_kn=22.0,
    max_hs_operate_m=1.5,
    max_hs_transit_m=2.5,
    mobilisation_usd=0.0,
)


@dataclass(frozen=True)
class Operation:
    """A marine operation with an on-site duration and a vessel requirement.

    Args:
        name: Label.
        vessel: Which vessel class performs it.
        on_site_hours: Working hours on station, excluding transit.
        per_year: Expected occurrences per year. Fractional values represent
            operations expected less often than annually.
        contingency: Multiplier on the total window requirement, covering
            set-up, standby and the fact that marine jobs overrun.
    """

    name: str
    vessel: Vessel
    on_site_hours: float
    per_year: float = 1.0
    contingency: float = 1.3


#: Tow-out, mooring hookup and commissioning of the buoy. Assumes moorings are
#: pre-laid on a separate mobilisation, which is why it appears twice.
DEPLOYMENT_OPERATIONS = (
    Operation("Mooring pre-lay", AHTS, on_site_hours=36, per_year=1.0),
    Operation("Tow-out and hookup", AHTS, on_site_hours=24, per_year=1.0),
    Operation("Commissioning", MULTICAT, on_site_hours=12, per_year=1.0),
)

#: Recovery at end of life or for major overhaul: disconnect, tow in, recover
#: moorings.
RETRIEVAL_OPERATIONS = (
    Operation("Disconnect and tow-in", AHTS, on_site_hours=24, per_year=1.0),
    Operation("Mooring recovery", AHTS, on_site_hours=30, per_year=1.0),
)

#: Recurring operations across a deployment year. The unscheduled rate is the
#: soft number here: it is the expected count of failures needing a vessel, and
#: for a prototype marine device it is optimistic rather than conservative.
ANNUAL_OPERATIONS = (
    Operation("Scheduled inspection", CTV, on_site_hours=6, per_year=4.0),
    Operation("Planned maintenance", MULTICAT, on_site_hours=16, per_year=1.0),
    Operation("Unscheduled intervention", MULTICAT, on_site_hours=20, per_year=1.5),
)


@dataclass(frozen=True)
class SiteLogistics:
    """Distance and port context for a candidate site.

    Args:
        distance_km: Great-circle distance from the support port.
        port_name: Label for the assumed base of operations.
        water_depth_m: Drives mooring cost and pre-lay duration.
    """

    distance_km: float
    port_name: str = "Grays Harbor, WA"
    water_depth_m: float = 100.0


def transit_hours(distance_km: float, vessel: Vessel) -> float:
    """Round-trip transit time in hours."""
    speed_kmh = vessel.transit_speed_kn * KM_PER_NM
    return 2 * distance_km / speed_kmh


def window_hours_required(operation: Operation, distance_km: float) -> float:
    """Continuous weather window an operation needs, including transit.

    This is the quantity that couples logistics to the wave climate: a longer
    required window is exponentially harder to find, so distance penalises a
    site twice - once through vessel time, and again through how rarely the
    weather permits the job at all.
    """
    return (
        transit_hours(distance_km, operation.vessel) + operation.on_site_hours
    ) * operation.contingency


#: Hours a vessel will sit on hire waiting for a window before demobilising.
#:
#: This cap is the difference between a plausible cost model and a nonsensical
#: one. Expected waiting time at an exposed site can run to hundreds of hours -
#: in winter, thousands - and charging a day rate across all of it produced
#: annual O&M an order of magnitude above the device's gross revenue.
#:
#: Real operations do not work that way. You mobilise **against a forecast**:
#: the vessel is called out when a window is predicted, works, and goes home.
#: The long wait is real, but it is borne as device downtime, not as vessel
#: hire. Two days of on-hire standby covers the forecast horizon over which a
#: window can be committed to with confidence.
DEFAULT_STANDBY_CAP_HOURS = 48.0


def operation_cost(
    operation: Operation,
    distance_km: float,
    waiting_hours: float = 0.0,
    include_mobilisation: bool = True,
    standby_cap_hours: float = DEFAULT_STANDBY_CAP_HOURS,
) -> dict:
    """Cost of one occurrence of an operation.

    Args:
        operation: What is being done.
        distance_km: Distance from port.
        waiting_hours: Expected wait for a usable window. Only the portion up
            to ``standby_cap_hours`` is charged as vessel hire; the remainder
            is real but is borne as device downtime, because a vessel called
            out on a forecast does not sit idle for months.
        include_mobilisation: Whether to charge mobilisation.
        standby_cap_hours: On-hire standby ceiling. See
            :data:`DEFAULT_STANDBY_CAP_HOURS`.

    Returns:
        dict of hours and cost components in USD. ``uncharged_wait_hours``
        carries the part of the wait that becomes downtime instead of cost.
    """
    transit = transit_hours(distance_km, operation.vessel)
    working = window_hours_required(operation, distance_km)
    charged_wait = min(waiting_hours, standby_cap_hours)
    chargeable = working + charged_wait
    vessel_cost = (chargeable / 24.0) * operation.vessel.day_rate_usd
    mobilisation = operation.vessel.mobilisation_usd if include_mobilisation else 0.0

    return {
        "operation": operation.name,
        "vessel": operation.vessel.name,
        "transit_hours": transit,
        "window_hours": working,
        "waiting_hours": waiting_hours,
        "charged_wait_hours": charged_wait,
        "uncharged_wait_hours": max(0.0, waiting_hours - charged_wait),
        "chargeable_hours": chargeable,
        "vessel_cost_usd": vessel_cost,
        "mobilisation_usd": mobilisation,
        "total_usd": vessel_cost + mobilisation,
    }


def false_start_multiplier(forecast_rmse_m: float, threshold_m: float) -> float:
    """Cost inflation from mobilising against an imperfect forecast.

    This is where forecast quality enters the economics, and it is the only
    place in this project where forecast skill has a dollar value.

    A vessel is called out when the forecast says a window is open. If the
    forecast is wrong the vessel sails and aborts, and the day is paid for
    anyway. The more uncertain the forecast relative to the working limit, the
    more often that happens.

    Modelled as the probability that the true sea state exceeds the limit when
    the forecast sat at the limit - a normal error of the given RMSE, so a
    coin-flip at zero margin, falling as skill improves. The multiplier is
    ``1 / (1 - p_abort)``: the expected number of sailings per successful job.

    Args:
        forecast_rmse_m: Forecast RMSE at the relevant lead time, metres. GEFS
            measures 0.41 m at +24 h at NDBC 46041.
        threshold_m: The vessel's working limit.

    Returns:
        Multiplier on vessel cost, at least 1.0.
    """
    from math import erf, sqrt

    if forecast_rmse_m <= 0:
        return 1.0

    # Operators do not sail at the limit; they keep a margin. Assume they aim
    # one RMSE below it, so the abort probability is P(error > 1 RMSE).
    margin = forecast_rmse_m
    z = margin / (forecast_rmse_m * sqrt(2.0))
    p_abort = 0.5 * (1.0 - erf(z))
    p_abort = min(p_abort, 0.9)
    return 1.0 / (1.0 - p_abort)


def mooring_capex(site: SiteLogistics, n_lines: int = 3) -> float:
    """Mooring system capital cost, scaled by water depth.

    Chain and synthetic line cost rises roughly linearly with depth, and deeper
    sites need larger anchors and more scope. A flat per-metre-per-line figure
    plus a fixed anchor cost captures the shape well enough to rank sites.
    """
    cost_per_m_per_line = 350.0
    anchor_cost_each = 85_000.0
    scope_factor = 2.5  # line length as a multiple of water depth
    line_cost = n_lines * site.water_depth_m * scope_factor * cost_per_m_per_line
    return line_cost + n_lines * anchor_cost_each
