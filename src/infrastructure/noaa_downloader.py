"""Subset NOAA GEFSv12 wave reforecast GRIB2 files to the study region.

Reads from the public NOAA reforecast bucket, clips to the study region, and
writes NetCDF to the project's raw bucket. Resumable via a local checkpoint file.

Two bugs were fixed here relative to the original script:
  * the region was hardcoded to 25-30N / -130 to -120E (off Baja California),
    which does not overlap the Pacific Northwest region the analysis uses;
    bounds now come from src.config
  * list_objects_v2 was called without a paginator, silently capping the file
    list at 1000 keys and truncating any full year of data
"""

import argparse
import json
import os
import sys
from io import BytesIO

import boto3
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (  # noqa: E402
    NOAA_SOURCE_BUCKET,
    NOAA_SOURCE_PREFIX_TEMPLATE,
    RAW_BUCKET,
    STUDY_REGION,
)

LOCAL_TMP = "/tmp"
CHECKPOINT_FILE = "progress_checkpoint.json"


def list_grib_keys(s3, bucket, prefix):
    """All .grib2 keys under a prefix, paginated (no 1000-key truncation)."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".grib2"):
                keys.append(obj["Key"])
    return keys


def subset_to_region(ds, region):
    """Clip a dataset to the region, handling either lon convention.

    GRIB files commonly use 0-360 longitudes while the study region is written
    in -180..180, so convert before slicing. Latitude may be stored ascending or
    descending, so the slice is ordered to match.
    """
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"

    lon_min, lon_max = region.lon_min, region.lon_max
    if float(ds[lon_name].min()) >= 0:
        lon_min, lon_max = lon_min % 360, lon_max % 360

    lat_descending = float(ds[lat_name][0]) > float(ds[lat_name][-1])
    lat_slice = (
        slice(region.lat_max, region.lat_min)
        if lat_descending
        else slice(region.lat_min, region.lat_max)
    )

    return ds.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(path, completed):
    with open(path, "w") as f:
        json.dump(sorted(completed), f)


def main(year, destination_bucket=RAW_BUCKET, checkpoint_file=CHECKPOINT_FILE):
    s3 = boto3.client("s3")
    source_prefix = NOAA_SOURCE_PREFIX_TEMPLATE.format(year=year)
    destination_prefix = f"noaa-data/{year}/"

    completed_files = load_checkpoint(checkpoint_file)
    grib_files = list_grib_keys(s3, NOAA_SOURCE_BUCKET, source_prefix)

    print(f"Region: {STUDY_REGION.name} "
          f"({STUDY_REGION.lon_min} to {STUDY_REGION.lon_max} E, "
          f"{STUDY_REGION.lat_min} to {STUDY_REGION.lat_max} N)")
    print(f"Found {len(grib_files)} GRIB2 files under {source_prefix}")

    for key in grib_files:
        filename = key.split("/")[-1]

        if filename in completed_files:
            print(f"Skipping {filename} (already done).")
            continue

        try:
            print(f"Processing {filename} ...")

            obj = s3.get_object(Bucket=NOAA_SOURCE_BUCKET, Key=key)
            file_stream = BytesIO(obj["Body"].read())
            ds = xr.open_dataset(file_stream, engine="cfgrib")

            subset = subset_to_region(ds, STUDY_REGION)
            if 0 in subset.sizes.values():
                print(f"  Skipping {filename}: no grid points inside the region.")
                completed_files.add(filename)
                save_checkpoint(checkpoint_file, completed_files)
                continue

            netcdf_name = filename.replace(".grib2", ".nc")
            local_nc_path = os.path.join(LOCAL_TMP, netcdf_name)
            # Derive the date folder from the key rather than a fixed index, so
            # this survives changes in the bucket's prefix depth.
            parts = key.split("/")
            date_folder = next(
                (p for p in parts if len(p) == 8 and p.isdigit()), str(year)
            )
            dest_key = f"{destination_prefix}{date_folder}/{netcdf_name}"

            subset.to_netcdf(local_nc_path)
            s3.upload_file(local_nc_path, destination_bucket, dest_key)
            os.remove(local_nc_path)

            completed_files.add(filename)
            save_checkpoint(checkpoint_file, completed_files)
            print(f"Completed {filename}.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--destination-bucket", default=RAW_BUCKET)
    parser.add_argument("--checkpoint-file", default=CHECKPOINT_FILE)
    args = parser.parse_args()

    main(args.year, args.destination_bucket, args.checkpoint_file)
