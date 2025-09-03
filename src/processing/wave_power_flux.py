import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

def calculate_wave_power_flux(wave_height, wave_period, water_density=1025, gravity=9.81):
    """
    Calculate wave power flux from significant wave height and period
    
    Formula: P = (ρ * g² / 64π) * H² * T
    Where:
    - P = wave power flux (W/m)
    - ρ = seawater density (kg/m³, default 1025)
    - g = gravitational acceleration (m/s², default 9.81)
    - H = significant wave height (m)
    - T = wave period (s)
    
    Args:
        wave_height (array): Significant wave height in meters
        wave_period (array): Wave period in seconds
        water_density (float): Seawater density in kg/m³
        gravity (float): Gravitational acceleration in m/s²
    
    Returns:
        array: Wave power flux in W/m
    """
    # Calculate wave power flux coefficient
    coefficient = (water_density * gravity**2) / (64 * np.pi)
    
    # Calculate power flux: P = coefficient * H² * T
    power_flux = coefficient * (wave_height**2) * wave_period
    
    return power_flux

def calculate_monthly_aggregations(power_flux_data, time_coord):
    """
    Calculate monthly statistics for wave power flux
    
    Args:
        power_flux_data (xarray.Dataset): Power flux data with time dimension
        time_coord (str): Name of time coordinate
    
    Returns:
        dict: Monthly statistics
    """
    monthly_stats = {}
    
    # Group by month and calculate statistics
    monthly_grouped = power_flux_data.groupby(f'{time_coord}.month')
    
    monthly_stats['mean'] = monthly_grouped.mean()
    monthly_stats['std'] = monthly_grouped.std()
    monthly_stats['max'] = monthly_grouped.max()
    monthly_stats['min'] = monthly_grouped.min()
    
    return monthly_stats

def calculate_consistency_metrics(power_flux_data, threshold_percentile=75):
    """
    Calculate consistency metrics for wave power assessment
    
    Args:
        power_flux_data (array): Wave power flux time series
        threshold_percentile (float): Percentile for threshold calculation
    
    Returns:
        dict: Consistency metrics
    """
    # Remove NaN values
    clean_data = power_flux_data[~np.isnan(power_flux_data)]
    
    if len(clean_data) == 0:
        return {
            'std_deviation': np.nan,
            'coefficient_of_variation': np.nan,
            'percent_above_threshold': np.nan,
            'threshold_value': np.nan,
            'seasonal_ratio': np.nan
        }
    
    # Calculate threshold value
    threshold = np.percentile(clean_data, threshold_percentile)
    
    # Calculate metrics
    metrics = {
        'std_deviation': np.std(clean_data),
        'coefficient_of_variation': np.std(clean_data) / np.mean(clean_data) if np.mean(clean_data) > 0 else np.nan,
        'percent_above_threshold': (np.sum(clean_data > threshold) / len(clean_data)) * 100,
        'threshold_value': threshold,
        'mean_power': np.mean(clean_data),
        'median_power': np.median(clean_data)
    }
    
    return metrics

def calculate_seasonal_ratio(power_flux_data, time_coord):
    """
    Calculate ratio of winter to summer wave power (seasonality indicator)
    
    Args:
        power_flux_data (xarray.Dataset): Power flux data
        time_coord (str): Name of time coordinate
        
    Returns:
        float: Winter/summer power ratio
    """
    # Define seasons (Northern Hemisphere)
    winter_months = [12, 1, 2]
    summer_months = [6, 7, 8]
    
    # Extract seasonal data
    winter_data = power_flux_data.where(
        power_flux_data[time_coord].dt.month.isin(winter_months)
    ).mean()
    
    summer_data = power_flux_data.where(
        power_flux_data[time_coord].dt.month.isin(summer_months)
    ).mean()
    
    # Calculate ratio
    if summer_data > 0:
        seasonal_ratio = winter_data / summer_data
    else:
        seasonal_ratio = np.nan
        
    return float(seasonal_ratio)

def create_revenue_surface_map(power_flux_data, energy_price_per_mwh=50):
    """
    Create revenue potential surface map from wave power data
    
    Args:
        power_flux_data (xarray.Dataset): Wave power flux data (W/m)
        energy_price_per_mwh (float): Energy price in $/MWh
        
    Returns:
        xarray.Dataset: Revenue potential in $/m/year
    """
    # Convert W/m to MWh/m/year
    # Assuming continuous operation: 8760 hours/year
    hours_per_year = 8760
    watts_to_megawatts = 1e-6
    
    # Calculate annual energy production per meter of wave front
    annual_energy_mwh_per_m = power_flux_data * hours_per_year * watts_to_megawatts
    
    # Calculate revenue potential
    revenue_potential = annual_energy_mwh_per_m * energy_price_per_mwh
    
    return revenue_potential

def rank_sites_by_potential(power_flux_data, lat_coord='latitude', lon_coord='longitude', top_n=10):
    """
    Rank sites by wave power potential
    
    Args:
        power_flux_data (xarray.Dataset): Mean power flux data
        lat_coord (str): Latitude coordinate name
        lon_coord (str): Longitude coordinate name
        top_n (int): Number of top sites to return
        
    Returns:
        pandas.DataFrame: Ranked sites with coordinates and power values
    """
    # Flatten spatial data
    flattened = power_flux_data.stack(location=(lat_coord, lon_coord))
    
    # Remove NaN values and sort
    valid_data = flattened.dropna('location')
    sorted_data = valid_data.sortby(valid_data, ascending=False)
    
    # Extract top sites
    top_sites = sorted_data.isel(location=slice(0, top_n))
    
    # Create results dataframe
    results = []
    for i in range(len(top_sites)):
        lat, lon = top_sites.location.values[i]
        power = float(top_sites.values[i])
        
        results.append({
            'rank': i + 1,
            'latitude': lat,
            'longitude': lon,
            'wave_power_flux_w_per_m': power,
            'location_description': f"Site {i+1}"
        })
    
    return pd.DataFrame(results)

def process_copernicus_wave_data(ds):
    """
    Process Copernicus marine dataset to calculate wave power flux
    
    Args:
        ds (xarray.Dataset): Copernicus wave dataset
        
    Returns:
        xarray.Dataset: Processed dataset with power flux calculations
    """
    # Extract wave variables (adjust variable names based on actual Copernicus data)
    if 'VHM0' in ds.variables and 'VTPK' in ds.variables:
        wave_height = ds['VHM0']  # Significant wave height
        wave_period = ds['VTPK']   # Peak wave period
    elif 'swh' in ds.variables and 'mwp' in ds.variables:
        wave_height = ds['swh']    # Significant wave height
        wave_period = ds['mwp']    # Mean wave period
    else:
        raise ValueError("Could not find wave height and period variables in dataset")
    
    # Calculate wave power flux
    power_flux = calculate_wave_power_flux(wave_height, wave_period)
    
    # Add to dataset
    ds['wave_power_flux'] = power_flux
    ds['wave_power_flux'].attrs = {
        'units': 'W/m',
        'long_name': 'Wave Power Flux',
        'description': 'Wave power per unit width of wave front'
    }
    
    return ds

def generate_processing_summary(ds, region_name="Pacific Northwest"):
    """
    Generate summary statistics for processed wave data
    
    Args:
        ds (xarray.Dataset): Processed wave dataset
        region_name (str): Name of the region
        
    Returns:
        dict: Processing summary
    """
    if 'wave_power_flux' not in ds.variables:
        raise ValueError("Dataset must contain wave_power_flux variable")
    
    power_data = ds['wave_power_flux']
    
    summary = {
        'region': region_name,
        'processing_date': datetime.now().isoformat(),
        'data_coverage': {
            'start_date': str(ds.time.min().values)[:10],
            'end_date': str(ds.time.max().values)[:10],
            'total_timesteps': len(ds.time),
        },
        'spatial_coverage': {
            'lat_min': float(ds.latitude.min()),
            'lat_max': float(ds.latitude.max()),
            'lon_min': float(ds.longitude.min()),
            'lon_max': float(ds.longitude.max()),
            'grid_points': len(ds.latitude) * len(ds.longitude)
        },
        'wave_power_statistics': {
            'mean_power_w_per_m': float(power_data.mean()),
            'max_power_w_per_m': float(power_data.max()),
            'min_power_w_per_m': float(power_data.min()),
            'std_power_w_per_m': float(power_data.std())
        },
        'consistency_metrics': calculate_consistency_metrics(power_data.values.flatten())
    }
    
    return summary