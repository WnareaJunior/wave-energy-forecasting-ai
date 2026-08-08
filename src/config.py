"""Single source of truth for study region, datasets, and storage locations.

Before this module existed, three different geographic boxes were hardcoded
across the repository (the Pacific Northwest box in the notebooks and docs, a
Baja California box in the NOAA downloader, and a Western Pacific box in the
Copernicus downloader). Every script now imports from here so they cannot drift
apart again.
"""

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
# Datasets
# --------------------------------------------------------------------------

#: Copernicus Marine global wave reanalysis. Treated as observational truth.
COPERNICUS_DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
COPERNICUS_RESOLUTION_DEG = 0.2
COPERNICUS_START = "2020-01-01T00:00:00"
COPERNICUS_END = "2023-04-30T21:00:00"

#: NOAA GEFSv12 wave ensemble reforecast (WAVEWATCH III). Public S3 bucket.
#: This is the *forecast* side of the pairing — the physics model whose errors
#: a postprocessing model learns to correct.
NOAA_SOURCE_BUCKET = "noaa-nws-gefswaves-reforecast-pds"
NOAA_SOURCE_PREFIX_TEMPLATE = "GEFSv12/reforecast/{year}/"
NOAA_RESOLUTION_DEG = 0.25


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

RAW_BUCKET = "panthalassa-ocean-raw-data"
PROCESSED_BUCKET = "panthalassa-ocean-processed"
RESULTS_BUCKET = "panthalassa-ocean-results"
