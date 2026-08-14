"""Build the static feature layer for the PNW study region and upload to S3.

Produces one small NetCDF on the Copernicus 0.2 deg wave-hindcast grid with
the time-invariant features the training table joins on:

  - depth              mean depth of the ETOPO1 ocean pixels inside each
                       0.2 deg cell (m, positive down; NaN where the cell
                       is entirely land)
  - ocean_fraction     fraction of the cell's ETOPO1 pixels below sea level
  - distance_to_coast  great-circle distance from the cell centre to the
                       nearest ETOPO1 land cell (km; 0 for all-land cells)
  - land_mask          1 where the cell contains no ocean pixels at all

Depth is a cell average rather than a point interpolation so that mixed
land/water coastal cells — which the 0.2 deg wave model treats as ocean —
still get a depth instead of falling on the land side of the coastline.

ETOPO1 is fetched from the NOAA CoastWatch ERDDAP over a box padded 1 deg
beyond the study region so distance-to-coast near the grid edges still sees
the nearest land even when it lies outside the region.

Credentials: AWS default boto3 chain. No Copernicus account needed.

Usage:
  python static_layer.py            # build + upload
  python static_layer.py --dry-run  # build + report, skip upload
"""

import argparse
import logging
import os

import boto3
import numpy as np
import requests
import xarray as xr

# ---- CONFIG ----
BUCKET_NAME = "panthalassa-ocean-processed"
OUTPUT_KEY = "static/pnw_static_layer.nc"
LOCAL_TEMP_DIR = "./temp_downloads"

# Copernicus cmems_mod_glo_wav_my_0.2deg grid over the PNW study region;
# must match the grid of the files under copernicus-data-pnw/.
GRID_LAT = np.round(np.arange(46.0, 50.4 + 1e-6, 0.2), 1)
GRID_LON = np.round(np.arange(-130.0, -124.0 + 1e-6, 0.2), 1)

# Study region padded so nearest-land searches near the edges are correct.
PAD_DEG = 1.0
ETOPO_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc"
    "?altitude%5B({lat0}):({lat1})%5D%5B({lon0}):({lon1})%5D"
)

EARTH_RADIUS_KM = 6371.0

log = logging.getLogger("static_layer")


def fetch_etopo(path):
    """Download the padded ETOPO1 subset from ERDDAP (once; ~0.5 MB)."""
    if os.path.exists(path):
        log.info("Using cached ETOPO subset at %s", path)
        return
    url = ETOPO_URL.format(
        lat0=GRID_LAT[0] - PAD_DEG, lat1=GRID_LAT[-1] + PAD_DEG,
        lon0=GRID_LON[0] - PAD_DEG, lon1=GRID_LON[-1] + PAD_DEG,
    )
    log.info("Fetching ETOPO1 subset from ERDDAP")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)


def distance_to_coast_km(lat_pts, lon_pts, land_lat, land_lon):
    """Min haversine distance (km) from each (lat, lon) point to any land cell.

    Brute force over all ETOPO land cells — the target grid is only a few
    hundred points, so exactness is cheap and beats an EDT approximation
    whose x-spacing would otherwise need a single-latitude cosine fudge.
    """
    lat_r = np.radians(lat_pts)[:, None]
    lon_r = np.radians(lon_pts)[:, None]
    land_lat_r = np.radians(land_lat)[None, :]
    land_lon_r = np.radians(land_lon)[None, :]
    a = (np.sin((land_lat_r - lat_r) / 2) ** 2
         + np.cos(lat_r) * np.cos(land_lat_r)
         * np.sin((land_lon_r - lon_r) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)).min(axis=1)


def build_layer(etopo_path):
    etopo = xr.open_dataset(etopo_path)
    elev_fine = etopo.altitude
    half = 0.1  # half-width of a 0.2 deg grid cell

    shape = (GRID_LAT.size, GRID_LON.size)
    depth = np.full(shape, np.nan, dtype=np.float32)
    ocean_frac = np.zeros(shape, dtype=np.float32)
    for i, lat in enumerate(GRID_LAT):
        for j, lon in enumerate(GRID_LON):
            cell = elev_fine.sel(latitude=slice(lat - half, lat + half),
                                 longitude=slice(lon - half, lon + half)).values
            wet = cell < 0
            ocean_frac[i, j] = wet.mean()
            if wet.any():
                depth[i, j] = -cell[wet].mean()
    land_mask = (ocean_frac == 0).astype(np.int8)

    land = elev_fine.values >= 0
    fine_lon, fine_lat = np.meshgrid(etopo.longitude.values,
                                     etopo.latitude.values)
    land_lat, land_lon = fine_lat[land], fine_lon[land]
    log.info("Computing distance to coast against %d land cells",
             land_lat.size)

    grid_lon, grid_lat = np.meshgrid(GRID_LON, GRID_LAT)
    dist = distance_to_coast_km(grid_lat.ravel(), grid_lon.ravel(),
                                land_lat, land_lon).reshape(grid_lat.shape)
    dist[land_mask == 1] = 0.0

    ds = xr.Dataset(
        {
            "depth": (("latitude", "longitude"), depth,
                      {"units": "m", "long_name": "cell-mean water depth, positive down; NaN where cell is all land"}),
            "ocean_fraction": (("latitude", "longitude"), ocean_frac,
                               {"long_name": "fraction of ETOPO1 pixels in cell below sea level"}),
            "distance_to_coast": (("latitude", "longitude"), dist.astype(np.float32),
                                  {"units": "km", "long_name": "great-circle distance from cell centre to nearest ETOPO1 land cell; 0 for all-land cells"}),
            "land_mask": (("latitude", "longitude"), land_mask,
                          {"long_name": "1 = cell contains no ocean pixels"}),
        },
        coords={"latitude": GRID_LAT, "longitude": GRID_LON},
        attrs={
            "title": "PNW static feature layer",
            "source": "ETOPO1 via NOAA CoastWatch ERDDAP (etopo180)",
            "grid": "Copernicus cmems_mod_glo_wav_my_0.2deg PNW subset",
        },
    )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="build and report but do not upload to S3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    etopo_path = os.path.join(LOCAL_TEMP_DIR, "etopo_pnw_padded.nc")
    out_path = os.path.join(LOCAL_TEMP_DIR, os.path.basename(OUTPUT_KEY))

    fetch_etopo(etopo_path)
    ds = build_layer(etopo_path)

    ocean = ds.depth.values[~np.isnan(ds.depth.values)]
    log.info("Grid %dx%d, %d ocean cells; depth %d-%d m; "
             "distance to coast up to %.0f km",
             ds.latitude.size, ds.longitude.size, ocean.size,
             ocean.min(), ocean.max(),
             np.nanmax(ds.distance_to_coast.values))

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    log.info("Wrote %s (%d bytes)", out_path, os.path.getsize(out_path))

    if args.dry_run:
        log.info("Dry run: skipping upload of s3://%s/%s",
                 BUCKET_NAME, OUTPUT_KEY)
        return
    # WAVE_S3_ENDPOINT → MinIO on the devbox for local runs; unset → AWS.
    boto3.client("s3", endpoint_url=os.environ.get("WAVE_S3_ENDPOINT")).upload_file(out_path, BUCKET_NAME, OUTPUT_KEY)
    log.info("Uploaded s3://%s/%s", BUCKET_NAME, OUTPUT_KEY)


if __name__ == "__main__":
    main()
