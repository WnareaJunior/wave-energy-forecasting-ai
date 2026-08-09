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

from dataclasses import dataclass, field

from src.config import (
    PORT_CLASS_HEAVY,
    PORT_CLASS_LIGHT,
    PORT_CLASS_WORKBOAT,
    SUPPORT_PORTS,
    nearest_port,
)

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
        port_class: Smallest class of port that can host this vessel. Decides
            which support base it sails from, and therefore its transit.
    """

    name: str
    day_rate_usd: float
    transit_speed_kn: float
    max_hs_operate_m: float
    max_hs_transit_m: float
    mobilisation_usd: float = 0.0
    port_class: str = PORT_CLASS_HEAVY


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
    port_class=PORT_CLASS_HEAVY,
)

#: Multi-purpose workboat for mooring inspection and light intervention.
MULTICAT = Vessel(
    name="Multicat workboat",
    day_rate_usd=12_000,
    transit_speed_kn=9.0,
    max_hs_operate_m=1.75,
    max_hs_transit_m=3.0,
    mobilisation_usd=75_000,
    port_class=PORT_CLASS_WORKBOAT,
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
    port_class=PORT_CLASS_LIGHT,
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
class VesselBase:
    """Where one vessel class sails from, and how far it has to go."""

    vessel_name: str
    port_name: str
    distance_km: float


@dataclass(frozen=True)
class SiteLogistics:
    """Distance and port context for a candidate site.

    Args:
        distance_km: Distance from the primary support port - the one the
            heavy tow-out spread uses. Also the fallback for any vessel with
            no explicit base, which keeps the single-distance construction
            ``SiteLogistics(distance_km=50)`` working as before.
        port_name: Label for the primary base of operations.
        water_depth_m: Drives mooring cost and pre-lay duration.
        bases: Per-vessel bases, from :func:`site_logistics`. A tuple rather
            than a dict so the dataclass stays hashable.
    """

    distance_km: float
    port_name: str = "Grays Harbor, WA"
    water_depth_m: float = 100.0
    bases: tuple = field(default=())

    def distance_for(self, vessel: Vessel) -> float:
        """Transit distance for one vessel class.

        Falls back to :attr:`distance_km` when the vessel has no assigned
        base, so a site built the old way behaves exactly as it used to.
        """
        for base in self.bases:
            if base.vessel_name == vessel.name:
                return base.distance_km
        return self.distance_km

    def port_for(self, vessel: Vessel) -> str:
        """Support port for one vessel class."""
        for base in self.bases:
            if base.vessel_name == vessel.name:
                return base.port_name
        return self.port_name


def site_logistics(
    latitude: float,
    longitude: float,
    water_depth_m: float = 100.0,
    vessels: tuple = (AHTS, MULTICAT, CTV),
    ports: tuple = SUPPORT_PORTS,
) -> SiteLogistics:
    """Build site logistics by basing each vessel at its nearest capable port.

    This is the correction to the model's original single-port assumption.
    Because an anchor handler and a crew boat cannot use the same harbours,
    the transit distance is a property of the *pair* (site, vessel), not of
    the site alone - and for a site tucked near a small harbour the two can
    differ by an order of magnitude.

    The primary distance and port are taken from the heaviest vessel supplied,
    since that is the tow-out passage and the closest thing to a meaningful
    "how far offshore is it" figure.

    Args:
        latitude: Site latitude.
        longitude: Site longitude, degrees east.
        water_depth_m: Water depth at the site.
        vessels: Vessel classes needing a base.
        ports: Candidate support ports.
    """
    bases = []
    for vessel in vessels:
        port, distance = nearest_port(latitude, longitude, vessel.port_class, ports)
        bases.append(VesselBase(vessel.name, port.name, distance))

    heaviest = max(
        vessels, key=lambda v: (v.port_class == PORT_CLASS_HEAVY, v.day_rate_usd)
    )
    primary = next(b for b in bases if b.vessel_name == heaviest.name)

    return SiteLogistics(
        distance_km=primary.distance_km,
        port_name=primary.port_name,
        water_depth_m=water_depth_m,
        bases=tuple(bases),
    )


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


#: Wave-height margin an operator keeps below the working limit when deciding
#: to sail. A fixed margin, not one scaled to forecast error - the operator
#: does not know the forecast's RMSE, they apply a habitual buffer.
DEFAULT_SAIL_MARGIN_M = 0.3


def false_start_multiplier(
    forecast_rmse_m: float,
    threshold_m: float,
    margin_m: float = DEFAULT_SAIL_MARGIN_M,
) -> float:
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
        margin_m: How far below the limit the operator aims.

    Returns:
        Multiplier on vessel cost, at least 1.0.

    Note:
        An earlier version set the margin equal to the RMSE, which made the
        RMSE cancel out of ``z = margin / (rmse * sqrt(2))`` - the multiplier
        came out at 1.19 for every forecast quality from 0.2 m to 1.0 m. It was
        only visible once the table was printed with several RMSE values side
        by side and every row was identical.
    """
    from math import erf, sqrt

    if forecast_rmse_m <= 0:
        return 1.0

    z = margin_m / (forecast_rmse_m * sqrt(2.0))
    p_abort = min(0.5 * (1.0 - erf(z)), 0.9)
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
