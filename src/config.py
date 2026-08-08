"""Single source of truth for study region, datasets, and storage locations.

Before this module existed, three different geographic boxes were hardcoded
across the repository (the Pacific Northwest box in the notebooks and docs, a
Baja California box in the NOAA downloader, and a Western Pacific box in the
Copernicus downloader). Every script now imports from here so they cannot drift
apart again.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Study region
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A lon/lat bounding box. Longitudes are degrees east, negative for west."""

    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        if self.lon_min >= self.lon_max:
            raise ValueError(f"{self.name}: lon_min must be < lon_max")
        if self.lat_min >= self.lat_max:
            raise ValueError(f"{self.name}: lat_min must be < lat_max")
        if not -180 <= self.lon_min <= 180 or not -180 <= self.lon_max <= 180:
            raise ValueError(f"{self.name}: longitudes must be in [-180, 180]")
        if not -90 <= self.lat_min <= 90 or not -90 <= self.lat_max <= 90:
            raise ValueError(f"{self.name}: latitudes must be in [-90, 90]")

    def grid_shape(self, resolution_deg: float) -> tuple[int, int]:
        """(n_lon, n_lat) grid points at a given resolution, endpoints included."""
        n_lon = int(round((self.lon_max - self.lon_min) / resolution_deg)) + 1
        n_lat = int(round((self.lat_max - self.lat_min) / resolution_deg)) + 1
        return n_lon, n_lat


#: The project's study region: Pacific Northwest offshore (Washington / BC).
#: This is the box used by the notebooks, the published figures, and all docs.
PACIFIC_NORTHWEST = Region(
    name="Pacific Northwest offshore",
    lon_min=-130.0,
    lon_max=-124.0,
    lat_min=46.0,
    lat_max=50.5,
)

#: Active region for all acquisition and processing.
STUDY_REGION = PACIFIC_NORTHWEST


# --------------------------------------------------------------------------
# NDBC buoy stations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Station:
    """An NDBC moored buoy.

    Positions are the nominal published deployment positions. Buoys drift within
    a watch circle and are occasionally relocated between deployments, so treat
    these as approximate (they are used to pick the nearest model grid cell,
    where sub-kilometre accuracy does not matter).
    """

    station_id: str
    name: str
    latitude: float
    longitude: float
    depth_m: float | None = None

    @property
    def in_study_region(self) -> bool:
        r = STUDY_REGION
        return (
            r.lon_min <= self.longitude <= r.lon_max
            and r.lat_min <= self.latitude <= r.lat_max
        )


#: Buoys off the Washington coast, ordered by relevance to the deployment sites.
#: 46041 and 46087 are the primary pair: both are exposed Washington-coast
#: buoys with long, near-continuous records.
NDBC_STATIONS = {
    "46041": Station("46041", "Cape Elizabeth, WA", 47.353, -124.731, depth_m=133.0),
    "46087": Station("46087", "Neah Bay, WA", 48.494, -124.728),
    "46029": Station("46029", "Columbia River Bar, OR/WA", 46.159, -124.514, depth_m=135.0),
    # Deep-water reference ~370 km offshore. Sits west of the study region's
    # -130 boundary, so it is excluded from region-restricted runs by default.
    "46005": Station("46005", "West Washington (offshore)", 46.134, -131.079, depth_m=2780.0),
}

#: Default station set for the pilot experiment.
PILOT_STATIONS = ("46041", "46087", "46029")


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

#: Copernicus Marine global wave reanalysis. Treated as observational truth.
COPERNICUS_DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
COPERNICUS_RESOLUTION_DEG = 0.2
COPERNICUS_START = "2020-01-01T00:00:00"
COPERNICUS_END = "2023-04-30T21:00:00"

#: NOAA GEFSv12 wave ensemble reforecast (WAVEWATCH III v7.12 forced by GEFSv12
#: winds). Public S3 bucket, unsigned access. This is the *forecast* side of the
#: pairing — the physics model whose errors a postprocessing model learns to
#: correct.
#:
#: Layout and characteristics below are from the archive's own
#: Description_of_reforecast_data.pdf and a listing of the bucket, not guessed.
NOAA_SOURCE_BUCKET = "noaa-nws-gefswaves-reforecast-pds"
NOAA_SOURCE_PREFIX_TEMPLATE = "GEFSv12/reforecast/{year}/"
NOAA_RESOLUTION_DEG = 0.25

#: One cycle per day at 03Z, 3-hourly output, 16-day range (35 days on
#: Wednesdays).
NOAA_CYCLE_HOUR = 3
NOAA_OUTPUT_INTERVAL_HOURS = 3
NOAA_FORECAST_RANGE_HOURS = 16 * 24

#: Five members: control plus four perturbed. Expanded to eleven on Wednesdays,
#: so member count varies by day of week and code must not assume it is fixed.
NOAA_MEMBERS = ("c00", "p01", "p02", "p03", "p04")

#: Reforecast coverage. This is the binding constraint on the paired dataset:
#: overlapped with an NDBC record starting in 2015, only 2015-2019 can be used.
NOAA_FIRST_YEAR = 2000
NOAA_LAST_YEAR = 2019

#: Per-cycle key layout:
#:   GEFSv12/reforecast/{year}/{yyyymmdd}/gridded/  ~1.75 GB GRIB2 per member
#:   GEFSv12/reforecast/{year}/{yyyymmdd}/station/  NetCDF point output
#:
#: The station directory holds time series of significant wave height, period
#: and direction at 658 buoy positions. That is the right source for this
#: project: the gridded files bundle the entire 16-day global run to deliver a
#: few numbers per buoy.
NOAA_GRIDDED_TEMPLATE = (
    "GEFSv12/reforecast/{year}/{date}/gridded/"
    "gefs.wave.{date}.{member}.global.0p25.grib2"
)
NOAA_STATION_PREFIX_TEMPLATE = "GEFSv12/reforecast/{year}/{date}/station/"

#: Point-output table: 8.7 MB, the file this project actually needs.
NOAA_STATION_TAB_TEMPLATE = (
    "GEFSv12/reforecast/{year}/{date}/station/gefs.wave.{date}.{member}.tab.nc"
)

#: Full 2D directional spectra, ~516 MB per member per cycle. Not needed here.
NOAA_STATION_SPEC_TEMPLATE = (
    "GEFSv12/reforecast/{year}/{date}/station/gefs.wave.{date}.{member}.spec.nc"
)

#: Schema of a tab.nc file, confirmed by opening one rather than assumed.
#: dims: station=658, time=382, string40=40.
#:
#: `time` holds absolute VALID times, hourly, beginning at the 03Z cycle hour —
#: note that the point output is hourly even though the gridded fields are
#: 3-hourly, and that lead time is therefore not stored: it must be computed as
#: valid_time minus cycle init. Getting that subtraction backwards is the one
#: way to leak the future into a postprocessing dataset.
#:
#: `latitude` and `longitude` are data variables with dims (time, station), not
#: coordinates, so they must be read at a timestep rather than off the index.
NOAA_STATION_N_POINTS = 658
NOAA_STATION_HS_VAR = "hs"  # spectral significant wave height, metres
NOAA_STATION_VARS = {
    "hs": "significant wave height [m]",
    "lm": "mean wave length [m]",
    "tr": "mean period normalised by relative frequency [s]",
    "fp": "peak frequency, Tp = 1/fp [s-1]",
    "th1p": "mean wave direction at spectral peak [degree]",
    "sth1p": "directional spread at spectral peak [degree]",
    "th1m": "mean wave direction from spectral moments [degree]",
    "sth1m": "directional spread from spectral moments [degree]",
}


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

RAW_BUCKET = "panthalassa-ocean-raw-data"
PROCESSED_BUCKET = "panthalassa-ocean-processed"
RESULTS_BUCKET = "panthalassa-ocean-results"
