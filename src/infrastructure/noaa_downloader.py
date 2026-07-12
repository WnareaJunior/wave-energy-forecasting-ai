"""Download the NOAA GEFSv12 wave reforecast (2000-2019) for the Pacific
Northwest and upload spatially-subset NetCDF files to S3.

The source GRIB2 files are global 0.25-degree grids (~1.75 GB per member
per day). Rather than copying them wholesale, this script uses each file's
.idx sidecar to byte-range-fetch only the selected variables (~470 MB per
member-day), decodes them with cfgrib, subsets to the PNW study region,
and stores the result (~1-2 MB per member-day).

Resume state is derived from the destination S3 prefix, exactly like
copernicus_downloader.py: rerunning skips any member-day already uploaded.

Credentials: source bucket is public (unsigned requests); destination uses
the default boto3 credential chain (IAM instance role on EC2).

Usage:
  python noaa_downloader.py --dry-run     # show resume state only
  python noaa_downloader.py               # download remaining member-days
  python noaa_downloader.py --limit 1     # process one member-day (testing)
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# ---- CONFIG ----
SOURCE_BUCKET = "noaa-nws-gefswaves-reforecast-pds"
SOURCE_ROOT = "GEFSv12/reforecast"
DEST_BUCKET = "panthalassa-ocean-raw-data"
DEST_PREFIX = "noaa-data"

YEARS = range(2000, 2020)
MEMBERS = ["c00"]  # control only; perturbed members (p01-p04) can be
                   # backfilled later — resume logic won't re-fetch c00.

# Wave-energy core set: significant height, peak period/direction,
# integrated mean wave frequency (for mean period), and 10 m wind.
VARIABLES = {"HTSGW", "PERPW", "DIRPW", "IMWF", "UGRD", "VGRD"}

# Pacific Northwest study region (Washington coast and approaches).
# Source grid longitude runs 0..360 east, latitude descends 90..-90.
MIN_LON, MAX_LON = 230.0, 236.0   # -130 to -124 in +/-180 convention
MIN_LAT, MAX_LAT = 46.0, 50.5

LOCAL_TEMP_DIR = "./temp_downloads"

log = logging.getLogger("noaa_downloader")


def source_key(day, member):
    year = day[:4]
    return (f"{SOURCE_ROOT}/{year}/{day}/gridded/"
            f"gefs.wave.{day}.{member}.global.0p25.grib2")


def output_name(day, member):
    return f"gefs.wave.{day}.{member}.pnw.nc"


def dest_key(day, member):
    return f"{DEST_PREFIX}/{day[:4]}/{output_name(day, member)}"


def list_source_days(src):
    """Enumerate available day folders (some days may be missing)."""
    days = []
    paginator = src.get_paginator("list_objects_v2")
    for year in YEARS:
        for page in paginator.paginate(Bucket=SOURCE_BUCKET,
                                       Prefix=f"{SOURCE_ROOT}/{year}/",
                                       Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                days.append(cp["Prefix"].rstrip("/").rsplit("/", 1)[-1])
    return sorted(days)


def completed_from_s3(dst):
    completed = set()
    paginator = dst.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=DEST_BUCKET, Prefix=f"{DEST_PREFIX}/"):
        for obj in page.get("Contents", []):
            basename = obj["Key"].rsplit("/", 1)[-1]
            if basename.endswith(".nc") and obj["Size"] > 0:
                completed.add(basename)
    return completed


def retry(func, *args, max_attempts=3, base_delay=15, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("Attempt %d failed (%s); retrying in %ds", attempt, e, delay)
            time.sleep(delay)


def wanted_byte_ranges(src, grib_key):
    """Parse the .idx sidecar into (start, end) ranges for VARIABLES.

    idx line format: msg_num:byte_offset:d=YYYYMMDDHH:VAR:level:forecast:
    A message ends where the next one starts; the last runs to EOF (None).
    Contiguous selected ranges are merged to cut request count.
    """
    idx_key = grib_key[: -len(".grib2")] + ".idx"
    idx_text = retry(src.get_object, Bucket=SOURCE_BUCKET,
                     Key=idx_key)["Body"].read().decode()
    lines = [line.split(":") for line in idx_text.strip().split("\n")]
    ranges = []
    for i, parts in enumerate(lines):
        if parts[3] not in VARIABLES:
            continue
        start = int(parts[1])
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        if ranges and ranges[-1][1] is not None and ranges[-1][1] + 1 == start:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    return ranges


def fetch_messages(src, grib_key, ranges, temp_grib):
    with open(temp_grib, "wb") as f:
        for start, end in ranges:
            spec = f"bytes={start}-{'' if end is None else end}"
            body = retry(src.get_object, Bucket=SOURCE_BUCKET,
                         Key=grib_key, Range=spec)["Body"]
            for chunk in body.iter_chunks(1024 * 1024):
                f.write(chunk)


def subset_to_netcdf(temp_grib, temp_nc):
    import cfgrib
    import xarray as xr

    # indexpath="" stops cfgrib writing .idx sidecars next to the temp file
    datasets = cfgrib.open_datasets(temp_grib, indexpath="")
    merged = xr.merge(datasets, compat="override")
    subset = merged.sel(latitude=slice(MAX_LAT, MIN_LAT),
                        longitude=slice(MIN_LON, MAX_LON))
    encoding = {v: {"zlib": True, "complevel": 4} for v in subset.data_vars}
    subset.to_netcdf(temp_nc, encoding=encoding)


def process_member_day(src, dst, day, member):
    name = output_name(day, member)
    grib_key = source_key(day, member)
    temp_grib = os.path.join(LOCAL_TEMP_DIR, f"{day}.{member}.grib2")
    temp_nc = os.path.join(LOCAL_TEMP_DIR, name)
    try:
        ranges = wanted_byte_ranges(src, grib_key)
        log.info("Fetching %s (%d ranges)", name, len(ranges))
        fetch_messages(src, grib_key, ranges, temp_grib)
        subset_to_netcdf(temp_grib, temp_nc)
        key = dest_key(day, member)
        log.info("Uploading %s to s3://%s/%s", name, DEST_BUCKET, key)
        retry(dst.upload_file, temp_nc, DEST_BUCKET, key)
        log.info("%s complete", name)
    finally:
        for path in (temp_grib, temp_nc):
            if os.path.exists(path):
                os.remove(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="report resume state without downloading")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel member-day downloads (default 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N member-days (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

    src = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    dst = boto3.client("s3")

    days = list_source_days(src)
    completed = completed_from_s3(dst)
    pending = [(d, m) for d in days for m in MEMBERS
               if output_name(d, m) not in completed]

    total = len(days) * len(MEMBERS)
    log.info("%d member-days total, %d already in S3, %d pending",
             total, total - len(pending), len(pending))
    if pending:
        log.info("Resume point: %s %s", *pending[0])
    if args.dry_run or not pending:
        return
    if args.limit:
        pending = pending[: args.limit]
        log.info("Limiting this run to %d member-day(s)", len(pending))

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_member_day, src, dst, d, m): (d, m)
                   for d, m in pending}
        for future in as_completed(futures):
            d, m = futures[future]
            try:
                future.result()
            except Exception as e:
                failures.append(f"{d}.{m}")
                log.error("%s.%s failed permanently: %s", d, m, e)

    if failures:
        log.error("%d member-days failed: %s — rerun to retry them",
                  len(failures), ", ".join(sorted(failures)[:20]))
        raise SystemExit(1)
    log.info("All member-days downloaded and uploaded successfully.")


if __name__ == "__main__":
    main()
