"""Normalize GEFS reforecast files and pair them with hindcast truth.

Produces the forecast-correction training pairs: for each reforecast
init day and member under noaa-data/, every (lead time, grid cell) row
carries the forecast's wave/wind variables plus the Copernicus hindcast
truth (obs_hs, obs_te, obs_power_kw) valid at init + lead.

Normalization applied to the raw GEFS NetCDFs:
  - longitude 230..236 east -> -130..-124 to match the wave grid
  - bilinear regrid from the 0.25 deg GEFS grid to the 0.2 deg wave grid
    (nearest-neighbour fill for coastal cells the bilinear stencil loses)
  - step -> integer lead_hours (0..381, 3-hourly)

Truth comes from the hindcast training table (training-table/year=YYYY/),
so this job depends on build_training_table.py having run. Leads that
cross into January also read the next year's truth partition. Rows whose
valid time has no truth (e.g. beyond the hindcast end) are dropped.

Output: one Parquet per init year and member at
  s3://panthalassa-ocean-processed/forecast-table/year=YYYY/member=MMM/part-0.parquet
A year/member whose Parquet exists is skipped unless --force; partial
years (backfill still running) are rebuilt when rerun with --force.

Sweep mode (--sweep) is for the scheduled EC2 job that runs while the
NOAA backfill is in flight: for every year/member whose raw archive is
complete (day count matches the public source bucket) and whose Parquet
is missing, it builds and uploads, then prints SWEEP_COMPLETE once every
year/member is done — the sweep instance's user data watches for that
marker to tear the schedule down.

Usage:
  python build_forecast_table.py --years 2000 2001
  python build_forecast_table.py --years 2000 --members c00 --local-only
  python build_forecast_table.py --sweep --members c00 p01 p02 p03 p04
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
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- CONFIG ----
SOURCE_BUCKET = "noaa-nws-gefswaves-reforecast-pds"
SOURCE_ROOT = "GEFSv12/reforecast"
RAW_BUCKET = "panthalassa-ocean-raw-data"
OUT_BUCKET = "panthalassa-ocean-processed"
NOAA_PREFIX = "noaa-data"
TRUTH_PREFIX = "training-table"
OUT_PREFIX = "forecast-table"
LOCAL_TEMP_DIR = "./temp_downloads"

FIRST_YEAR, LAST_YEAR = 2000, 2019
MEMBERS = ["c00"]

# GEFS name -> table column. shts/shts_2/shts_3 are the swell partitions
# produced by the downloader's level-suffix rename.
FORECAST_VARS = {
    "swh": "fc_hs", "perpw": "fc_tp", "mp1": "fc_tm", "dirpw": "fc_dir",
    "shww": "fc_hs_windsea", "shts": "fc_hs_swell1", "shts_2": "fc_hs_swell2",
    "shts_3": "fc_hs_swell3", "u": "fc_wind_u", "v": "fc_wind_v",
}
TRUTH_COLS = {"VHM0": "obs_hs", "VTM10": "obs_te",
              "power_flux_kw": "obs_power_kw"}

log = logging.getLogger("build_forecast_table")


def output_key(year, member):
    return f"{OUT_PREFIX}/year={year}/member={member}/part-0.parquet"


def s3_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def snap(values):
    """Round coordinates to the 0.2 deg grid so float noise can't break
    the truth join (wave grid latitudes carry float32 noise)."""
    return np.round(np.asarray(values, dtype=np.float64), 1)


def load_truth(s3, years):
    """Load truth partitions for the given years as a (time, lat, lon)
    indexed frame of the label columns."""
    frames = []
    for year in sorted(set(years)):
        key = f"{TRUTH_PREFIX}/year={year}/part-0.parquet"
        if not s3_exists(s3, OUT_BUCKET, key):
            log.warning("No truth partition for %d; leads into it are dropped", year)
            continue
        path = os.path.join(LOCAL_TEMP_DIR, f"truth_{year}.parquet")
        if not os.path.exists(path):
            s3.download_file(OUT_BUCKET, key, path)
        df = pd.read_parquet(path, columns=["time", "latitude", "longitude",
                                            *TRUTH_COLS])
        frames.append(df)
    truth = pd.concat(frames, ignore_index=True).rename(columns=TRUTH_COLS)
    truth["latitude"] = snap(truth.latitude)
    truth["longitude"] = snap(truth.longitude)
    return truth.set_index(["time", "latitude", "longitude"]).sort_index()


def normalize_forecast(path, grid_lat, grid_lon):
    """One GEFS member-day file -> long dataframe on the wave grid."""
    ds = xr.open_dataset(path)
    ds = ds.assign_coords(longitude=ds.longitude - 360.0)
    keep = [v for v in FORECAST_VARS if v in ds.data_vars]
    ds = ds[keep]
    lin = ds.interp(latitude=grid_lat, longitude=grid_lon, method="linear")
    near = ds.interp(latitude=grid_lat, longitude=grid_lon, method="nearest")
    ds = lin.fillna(near)

    df = ds.to_dataframe().reset_index()
    df["valid_time"] = df.time + df.step
    df["lead_hours"] = (df.step / pd.Timedelta(hours=1)).astype(np.int16)
    df = df.rename(columns={"time": "init_time", **FORECAST_VARS})
    df["latitude"] = snap(df.latitude)
    df["longitude"] = snap(df.longitude)
    cols = (["init_time", "valid_time", "lead_hours", "latitude", "longitude"]
            + [FORECAST_VARS[v] for v in keep])
    df = df[cols]
    # Downcast per file so a year's concat peaks at ~2.5 GB, not ~5 GB —
    # keeps the sweep viable on a mid-size EC2 instance.
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def build_year_member(s3, year, member):
    resp = s3.list_objects_v2(Bucket=RAW_BUCKET,
                              Prefix=f"{NOAA_PREFIX}/{year}/")
    keys = sorted(k["Key"] for k in resp.get("Contents", [])
                  if k["Key"].endswith(f".{member}.pnw.nc") and k["Size"] > 0)
    if not keys:
        log.info("%d/%s: no source files yet", year, member)
        return None
    truth = load_truth(s3, [year, year + 1])
    grid = truth.index
    grid_lat = np.unique(grid.get_level_values("latitude"))
    grid_lon = np.unique(grid.get_level_values("longitude"))

    frames = []
    for key in keys:
        path = os.path.join(LOCAL_TEMP_DIR, os.path.basename(key))
        s3.download_file(RAW_BUCKET, key, path)
        try:
            frames.append(normalize_forecast(path, grid_lat, grid_lon))
        finally:
            os.remove(path)
    fc = pd.concat(frames, ignore_index=True)

    joined = fc.join(truth, on=["valid_time", "latitude", "longitude"],
                     how="inner")
    joined["member"] = member
    for c in joined.columns:
        if joined[c].dtype == np.float64:
            joined[c] = joined[c].astype(np.float32)
    log.info("%d/%s: %d init days, %d rows (%.0f%% of forecast rows had truth)",
             year, member, len(keys), len(joined), 100 * len(joined) / len(fc))
    return joined


def source_day_count(year):
    """Days available in NOAA's public reforecast bucket for a year —
    the definition of 'raw archive complete' for the sweep."""
    src = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    n = 0
    paginator = src.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=SOURCE_BUCKET,
                                   Prefix=f"{SOURCE_ROOT}/{year}/",
                                   Delimiter="/"):
        n += len(page.get("CommonPrefixes", []))
    return n


def upload_year_member(s3, year, member, local_only):
    df = build_year_member(s3, year, member)
    if df is None:
        return
    out_path = os.path.join(LOCAL_TEMP_DIR, f"forecast_{year}_{member}.parquet")
    df.to_parquet(out_path, index=False)
    log.info("%d/%s: %.1f MB", year, member, os.path.getsize(out_path) / 1e6)
    if not local_only:
        s3.upload_file(out_path, OUT_BUCKET, output_key(year, member))
        os.remove(out_path)
        log.info("%d/%s: uploaded s3://%s/%s", year, member,
                 OUT_BUCKET, output_key(year, member))


def sweep(s3, members):
    """Build every year/member whose raw archive is complete and whose
    output Parquet is missing. Prints SWEEP_COMPLETE when nothing is
    left to wait for."""
    done = True
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        expected = None
        raw = s3.list_objects_v2(Bucket=RAW_BUCKET,
                                 Prefix=f"{NOAA_PREFIX}/{year}/")
        raw_names = [o["Key"].rsplit("/", 1)[-1]
                     for o in raw.get("Contents", [])if o["Size"] > 0]
        for member in members:
            if s3_exists(s3, OUT_BUCKET, output_key(year, member)):
                continue
            if expected is None:
                expected = source_day_count(year)
            have = sum(1 for n in raw_names if f".{member}." in n)
            if have < expected:
                log.info("%d/%s: raw %d/%d days — waiting", year, member,
                         have, expected)
                done = False
                continue
            upload_year_member(s3, year, member, local_only=False)
    if done:
        log.info("SWEEP_COMPLETE: all %d-%d year/members built",
                 FIRST_YEAR, LAST_YEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(FIRST_YEAR, LAST_YEAR + 1)))
    parser.add_argument("--members", nargs="+", default=MEMBERS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="build only year/members whose raw archive is "
                             "complete; print SWEEP_COMPLETE when all done")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    s3 = boto3.client("s3")

    if args.sweep:
        sweep(s3, args.members)
        return

    for year in args.years:
        for member in args.members:
            if not args.force and not args.local_only and s3_exists(
                    s3, OUT_BUCKET, output_key(year, member)):
                log.info("%d/%s: exists, skipping", year, member)
                continue
            upload_year_member(s3, year, member, args.local_only)


if __name__ == "__main__":
    main()
