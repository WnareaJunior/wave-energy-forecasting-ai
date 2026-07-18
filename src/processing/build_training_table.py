"""Join the PNW data sources into a per-year training table (Parquet on S3).

For each calendar year, joins onto the Copernicus 0.2 deg / 3-hourly wave
hindcast grid:

  - the 17 wave variables from copernicus-data-pnw/ monthly chunks
  - GLORYS surface currents + SST (glorys-data-pnw/, daily, 1/12 deg),
    bilinearly regridded to the wave grid with nearest-neighbour fill at
    the coast, then forward-filled from daily to 3-hourly (2000-2019 only;
    NaN outside GLORYS coverage — XGBoost handles missing natively)
  - the static layer (depth, ocean_fraction, distance_to_coast)
  - derived: power_flux_kw (deep-water P from Hs=VHM0 and Te=VTM10),
    current_speed, month

Rows are (time, latitude, longitude) tuples for wave-ocean cells that have
a depth (drops land and the one ETOPO-blind fjord cell). Monthly chunks are
stamped 21:00 -> 18:00, so each year also pulls the previous December chunk
and slices to the exact calendar year; chunk-boundary duplicate timestamps
from early downloads are dropped.

Resume state is S3-derived like the downloaders: a year whose Parquet
object already exists is skipped unless --force.

Usage:
  python build_training_table.py                     # all years, 1980-2023
  python build_training_table.py --years 2000 2001
  python build_training_table.py --years 2000 --local-only
"""

import argparse
import logging
import os
import sys

import boto3
import botocore
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.wave_power_flux import calculate_wave_power_flux

# ---- CONFIG ----
RAW_BUCKET = "panthalassa-ocean-raw-data"
OUT_BUCKET = "panthalassa-ocean-processed"
WAVE_PREFIX = "copernicus-data-pnw"
GLORYS_PREFIX = "glorys-data-pnw"
STATIC_KEY = "static/pnw_static_layer.nc"
OUT_PREFIX = "training-table"
LOCAL_TEMP_DIR = "./temp_downloads"

FIRST_YEAR, LAST_YEAR = 1980, 2023
GLORYS_YEARS = range(2000, 2020)

log = logging.getLogger("build_training_table")


def output_key(year):
    return f"{OUT_PREFIX}/year={year}/part-0.parquet"


def s3_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def fetch(s3, bucket, key, path):
    if not os.path.exists(path):
        s3.download_file(bucket, key, path)
    return path


def load_wave_year(s3, year):
    """Open the year's monthly chunks (plus previous December for the
    21:00-offset chunk boundary), dedupe, and slice to the calendar year."""
    keys = [obj["Key"] for obj in s3.list_objects_v2(
        Bucket=RAW_BUCKET, Prefix=f"{WAVE_PREFIX}/{year}/")["Contents"]]
    prev_dec = f"{WAVE_PREFIX}/{year - 1}/{year - 1}1201_{year}0101.nc"
    if s3_exists(s3, RAW_BUCKET, prev_dec):
        keys.append(prev_dec)
    paths = [fetch(s3, RAW_BUCKET, k,
                   os.path.join(LOCAL_TEMP_DIR, k.replace("/", "_")))
             for k in sorted(keys)]
    chunks = []
    for p in paths:
        with xr.open_dataset(p) as chunk:
            chunks.append(chunk.load())
    ds = xr.concat(chunks, dim="time").sortby("time")
    ds = ds.sel(time=~ds.get_index("time").duplicated())
    return ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))


def load_glorys_year(s3, year, wave):
    """Regrid GLORYS to the wave grid (linear, nearest fill at the coast)
    and forward-fill daily means onto the 3-hourly wave timebase."""
    if year not in GLORYS_YEARS:
        return None
    key = f"{GLORYS_PREFIX}/glorys_pnw_{year}.nc"
    path = fetch(s3, RAW_BUCKET, key,
                 os.path.join(LOCAL_TEMP_DIR, os.path.basename(key)))
    g = xr.open_dataset(path).squeeze("depth", drop=True)
    lin = g.interp(latitude=wave.latitude, longitude=wave.longitude,
                   method="linear")
    near = g.interp(latitude=wave.latitude, longitude=wave.longitude,
                    method="nearest")
    g = lin.fillna(near)
    return g.reindex(time=wave.time, method="ffill")


def build_year(s3, static, year):
    wave = load_wave_year(s3, year)
    log.info("%d: %d timesteps", year, wave.time.size)

    ds = wave
    glorys = load_glorys_year(s3, year, wave)
    if glorys is not None:
        ds = xr.merge([ds, glorys])
    else:
        for v in ("uo", "vo", "thetao"):
            ds[v] = xr.full_like(ds.VHM0, np.nan)
    # The wave grid carries float32 coordinate noise (e.g. 50.400001525);
    # snap the static layer onto it so the merge aligns exactly.
    static = static.reindex(latitude=ds.latitude, longitude=ds.longitude,
                            method="nearest", tolerance=0.01)
    ds = xr.merge([ds, static], join="exact")

    ds["power_flux_kw"] = calculate_wave_power_flux(ds.VHM0, ds.VTM10) / 1e3
    ds["current_speed"] = np.hypot(ds.uo, ds.vo)

    df = ds.drop_vars("land_mask").to_dataframe().reset_index()
    df = df[df.VHM0.notna() & df.depth.notna()]
    df["month"] = df.time.dt.month.astype(np.int8)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(FIRST_YEAR, LAST_YEAR + 1)))
    parser.add_argument("--force", action="store_true",
                        help="rebuild years whose Parquet already exists")
    parser.add_argument("--local-only", action="store_true",
                        help="write Parquet locally, skip upload")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    s3 = boto3.client("s3")
    static = xr.open_dataset(fetch(
        s3, OUT_BUCKET, STATIC_KEY,
        os.path.join(LOCAL_TEMP_DIR, os.path.basename(STATIC_KEY))))

    for year in args.years:
        key = output_key(year)
        if not args.force and not args.local_only and s3_exists(s3, OUT_BUCKET, key):
            log.info("%d: exists, skipping", year)
            continue
        df = build_year(s3, static, year)
        out_path = os.path.join(LOCAL_TEMP_DIR, f"training_{year}.parquet")
        df.to_parquet(out_path, index=False)
        log.info("%d: %d rows, %.1f MB", year, len(df),
                 os.path.getsize(out_path) / 1e6)
        if not args.local_only:
            s3.upload_file(out_path, OUT_BUCKET, key)
            os.remove(out_path)
            log.info("%d: uploaded s3://%s/%s", year, OUT_BUCKET, key)


if __name__ == "__main__":
    main()
