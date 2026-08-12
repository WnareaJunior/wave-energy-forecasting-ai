"""Wave power flux and resource-assessment metrics.

Unit convention
---------------
Every function in this module works in **W/m** (watts per metre of wave crest).
Divide by 1000 for the kW/m figures used in the notebooks and figures.

Period convention
-----------------
The deep-water energy flux formula

    P = (rho * g^2 / (64 * pi)) * Hm0^2 * Te

is defined in terms of the **energy period** Te = m_{-1} / m_0, not the peak
period and not the zero-crossing period. In the Copernicus variable set:

    VTM10  IS Te by definition (mean period from the inverse frequency moment)
    VTM02  is the zero-crossing period; Te ~ 1.2 * VTM02 for a JONSWAP spectrum
    VTPK   is the peak period;         Te ~ 0.86 * VTPK for JONSWAP (gamma=3.3)

Use VTM10 where it is available. The conversion factors below are spectrum-shape
assumptions, so results derived from VTM02 or VTPK carry extra uncertainty; the
factor actually applied is recorded in the output attributes.
"""

from datetime import datetime

import numpy as np
import pandas as pd

#: Multiply the named Copernicus period variable by this to approximate Te.
#: Source: JONSWAP spectrum relations, gamma = 3.3.
ENERGY_PERIOD_FACTORS = {
    "VTM10": 1.00,  # already Te
    "VTM02": 1.20,
    "VTPK": 0.86,
}

#: Preference order when several period variables are present.
PERIOD_VARIABLE_PRIORITY = ("VTM10", "VTM02", "VTPK")

SEAWATER_DENSITY = 1025.0  # kg/m^3
GRAVITY = 9.81  # m/s^2


def calculate_wave_power_flux(
    wave_height,
    energy_period,
    water_density=SEAWATER_DENSITY,
    gravity=GRAVITY,
):
    """Deep-water wave power flux.

    Formula: P = (rho * g^2 / (64 * pi)) * Hm0^2 * Te

    Args:
        wave_height: Significant wave height Hm0, in metres.
        energy_period: Energy period Te, in seconds. See the module docstring —
            convert VTM02 or VTPK with ``ENERGY_PERIOD_FACTORS`` before calling.
        water_density: Seawater density in kg/m^3.
        gravity: Gravitational acceleration in m/s^2.

    Returns:
        Wave power flux in **W/m**. For Hm0 = 2 m and Te = 8 s this is
        ~15.7 kW/m.
    """
    coefficient = (water_density * gravity**2) / (64 * np.pi)
    return coefficient * (wave_height**2) * energy_period


def group_velocity_deep_water(period, gravity=GRAVITY):
    """Deep-water group velocity, cg = g * T / (4 * pi), in m/s.

    Needed by the physics-constrained models: cg is what transports wave energy,
    so it appears in the energy-balance residual.
    """
    return gravity * np.asarray(period) / (4 * np.pi)


def wavelength_deep_water(period, gravity=GRAVITY):
    """Deep-water wavelength, L = g * T^2 / (2 * pi), in metres."""
    return gravity * np.asarray(period) ** 2 / (2 * np.pi)


def steepness(wave_height, period, gravity=GRAVITY):
    """Wave steepness Hm0 / L (dimensionless).

    Physically bounded below ~1/7; waves break above that. Useful as a QC filter
    and as a soft constraint on model output.
    """
    return np.asarray(wave_height) / wavelength_deep_water(period, gravity)


def select_period_variable(ds):
    """Pick the best available period variable and its Te conversion factor.

    Args:
        ds: xarray Dataset holding Copernicus wave variables.

    Returns:
        (variable_name, factor) — multiply the variable by ``factor`` to get Te.

    Raises:
        ValueError: if none of the known period variables are present.
    """
    for name in PERIOD_VARIABLE_PRIORITY:
        if name in ds.variables:
            return name, ENERGY_PERIOD_FACTORS[name]
    raise ValueError(
        f"No wave period variable found. Expected one of {PERIOD_VARIABLE_PRIORITY}, "
        f"got {sorted(ds.variables)}"
    )


def calculate_monthly_aggregations(power_flux_data, time_coord):
    """Monthly mean/std/max/min of wave power flux.

    Args:
        power_flux_data: xarray object with a time dimension.
        time_coord: Name of the time coordinate.

    Returns:
        dict of month-grouped statistics.
    """
    monthly_grouped = power_flux_data.groupby(f"{time_coord}.month")
    return {
        "mean": monthly_grouped.mean(),
        "std": monthly_grouped.std(),
        "max": monthly_grouped.max(),
        "min": monthly_grouped.min(),
    }


def calculate_consistency_metrics(power_flux_data, threshold_percentile=75):
    """Variability and reliability metrics for a power-flux time series.

    Args:
        power_flux_data: 1-D array of wave power flux in W/m.
        threshold_percentile: Percentile defining the "good conditions" threshold.

    Returns:
        dict with keys ``std_deviation``, ``coefficient_of_variation``,
        ``percent_above_threshold``, ``threshold_value``, ``mean_power`` and
        ``median_power``. Seasonality is a separate calculation — see
        :func:`calculate_seasonal_ratio`, which needs the time coordinate.
    """
    data = np.asarray(power_flux_data, dtype=float)
    clean_data = data[~np.isnan(data)]

    keys = (
        "std_deviation",
        "coefficient_of_variation",
        "percent_above_threshold",
        "threshold_value",
        "mean_power",
        "median_power",
    )
    if len(clean_data) == 0:
        return dict.fromkeys(keys, np.nan)

    threshold = np.percentile(clean_data, threshold_percentile)
    mean_power = np.mean(clean_data)

    return {
        "std_deviation": np.std(clean_data),
        "coefficient_of_variation": (
            np.std(clean_data) / mean_power if mean_power > 0 else np.nan
        ),
        "percent_above_threshold": (np.sum(clean_data > threshold) / len(clean_data)) * 100,
        "threshold_value": threshold,
        "mean_power": mean_power,
        "median_power": np.median(clean_data),
    }


def calculate_seasonal_ratio(power_flux_data, time_coord):
    """Winter/summer mean power ratio (Northern Hemisphere seasonality).

    Args:
        power_flux_data: xarray object with a time coordinate.
        time_coord: Name of the time coordinate.

    Returns:
        float: DJF mean divided by JJA mean, or NaN if the summer mean is zero.
    """
    winter_months = [12, 1, 2]
    summer_months = [6, 7, 8]

    winter_mean = power_flux_data.where(
        power_flux_data[time_coord].dt.month.isin(winter_months)
    ).mean()
    summer_mean = power_flux_data.where(
        power_flux_data[time_coord].dt.month.isin(summer_months)
    ).mean()

    if float(summer_mean) <= 0:
        return np.nan
    return float(winter_mean / summer_mean)


def create_revenue_surface_map(
    power_flux_data,
    energy_price_per_mwh=50,
    capture_width_m=1.0,
    efficiency=1.0,
    availability=1.0,
):
    """Annual revenue potential from mean wave power flux.

    With the defaults this returns the theoretical resource value in $/m/year —
    revenue per metre of wave crest at 100% capture. To reproduce the
    device-level figures in the notebooks, pass the buoy's capture width and
    conversion efficiency (e.g. ``capture_width_m=10, efficiency=0.35``), which
    returns $/device/year.

    Args:
        power_flux_data: Mean wave power flux in **W/m**.
        energy_price_per_mwh: Energy price in $/MWh.
        capture_width_m: Metres of wave crest intercepted by the device.
        efficiency: Wave-to-wire conversion efficiency, 0-1.
        availability: Fraction of the year the device is operational, 0-1.

    Returns:
        Revenue in $/year, per metre of crest by default.
    """
    hours_per_year = 8760
    watts_to_megawatts = 1e-6

    annual_energy_mwh = (
        power_flux_data
        * capture_width_m
        * efficiency
        * availability
        * hours_per_year
        * watts_to_megawatts
    )
    return annual_energy_mwh * energy_price_per_mwh


def rank_sites_by_potential(
    power_flux_data, lat_coord="latitude", lon_coord="longitude", top_n=10
):
    """Rank grid cells by mean wave power potential.

    Args:
        power_flux_data: Time-averaged power flux with lat/lon dimensions, W/m.
        lat_coord: Latitude coordinate name.
        lon_coord: Longitude coordinate name.
        top_n: Number of sites to return.

    Returns:
        pandas.DataFrame with rank, coordinates, and power flux.
    """
    flattened = power_flux_data.stack(location=(lat_coord, lon_coord))
    valid_data = flattened.dropna("location")
    sorted_data = valid_data.sortby(valid_data, ascending=False)
    top_sites = sorted_data.isel(location=slice(0, top_n))

    results = []
    for i in range(len(top_sites)):
        lat, lon = top_sites.location.values[i]
        results.append(
            {
                "rank": i + 1,
                "latitude": lat,
                "longitude": lon,
                "wave_power_flux_w_per_m": float(top_sites.values[i]),
                "location_description": f"Site {i + 1}",
            }
        )

    return pd.DataFrame(results)


def process_copernicus_wave_data(ds):
    """Add a ``wave_power_flux`` variable to a Copernicus wave dataset.

    Selects the best available period variable (see the module docstring),
    converts it to an energy period, and records which variable and conversion
    factor were used in the output attributes.

    Args:
        ds: xarray Dataset containing ``VHM0`` and at least one period variable.

    Returns:
        The dataset with ``wave_power_flux`` (W/m) added.
    """
    if "VHM0" not in ds.variables:
        raise ValueError(
            f"Significant wave height 'VHM0' not found. Got {sorted(ds.variables)}"
        )

    period_var, factor = select_period_variable(ds)
    energy_period = ds[period_var] * factor

    ds["wave_power_flux"] = calculate_wave_power_flux(ds["VHM0"], energy_period)
    ds["wave_power_flux"].attrs = {
        "units": "W m-1",
        "long_name": "Wave Power Flux",
        "description": "Deep-water wave power per unit width of wave crest",
        "period_variable": period_var,
        "energy_period_factor": factor,
        "formula": "P = (rho * g^2 / (64 * pi)) * Hm0^2 * Te",
    }
    return ds


def generate_processing_summary(ds, region_name="Pacific Northwest"):
    """Summary statistics for a processed wave dataset.

    Args:
        ds: Dataset containing ``wave_power_flux``.
        region_name: Label for the region.

    Returns:
        dict of coverage and power statistics.
    """
    if "wave_power_flux" not in ds.variables:
        raise ValueError("Dataset must contain wave_power_flux variable")

    power_data = ds["wave_power_flux"]

    return {
        "region": region_name,
        "processing_date": datetime.now().isoformat(),
        "data_coverage": {
            "start_date": str(ds.time.min().values)[:10],
            "end_date": str(ds.time.max().values)[:10],
            "total_timesteps": len(ds.time),
        },
        "spatial_coverage": {
            "lat_min": float(ds.latitude.min()),
            "lat_max": float(ds.latitude.max()),
            "lon_min": float(ds.longitude.min()),
            "lon_max": float(ds.longitude.max()),
            "grid_points": len(ds.latitude) * len(ds.longitude),
        },
        "wave_power_statistics": {
            "mean_power_w_per_m": float(power_data.mean()),
            "max_power_w_per_m": float(power_data.max()),
            "min_power_w_per_m": float(power_data.min()),
            "std_power_w_per_m": float(power_data.std()),
        },
        "consistency_metrics": calculate_consistency_metrics(power_data.values.flatten()),
    }
