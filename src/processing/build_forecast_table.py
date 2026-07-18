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

Usage:
  python build_forecast_table.py --years 2000 2001
  python build_forecast_table.py --years 2000 --members c00 --local-only
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

# ---- CONFIG ----
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
    return df[cols]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(FIRST_YEAR, LAST_YEAR + 1)))
    parser.add_argument("--members", nargs="+", default=MEMBERS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    s3 = boto3.client("s3")

    for year in args.years:
        for member in args.members:
            key = output_key(year, member)
            if not args.force and not args.local_only and s3_exists(s3, OUT_BUCKET, key):
                log.info("%d/%s: exists, skipping", year, member)
                continue
            df = build_year_member(s3, year, member)
            if df is None:
                continue
            out_path = os.path.join(LOCAL_TEMP_DIR,
                                    f"forecast_{year}_{member}.parquet")
            df.to_parquet(out_path, index=False)
            log.info("%d/%s: %.1f MB", year, member,
                     os.path.getsize(out_path) / 1e6)
            if not args.local_only:
                s3.upload_file(out_path, OUT_BUCKET, key)
                os.remove(out_path)
                log.info("%d/%s: uploaded s3://%s/%s", year, member,
                         OUT_BUCKET, key)


if __name__ == "__main__":
    main()
