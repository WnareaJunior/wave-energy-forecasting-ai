"""Download surface ocean currents and sea surface temperature from the
Copernicus GLORYS12 global physics reanalysis for the Pacific Northwest
study region, in yearly chunks, and upload each chunk to S3.

Covers 2000-2019 to match the NOAA GEFSv12 wave reforecast window. Daily
means, surface level only (~0.5 m), variables uo/vo (currents) and thetao
(temperature). Each yearly file is ~15-20 MB.

Resume state is derived from the destination S3 prefix, exactly like
copernicus_downloader.py. Credentials work the same way too:
COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD or a stored CLI login,
plus the default boto3 chain for AWS.

Usage:
  python glorys_downloader.py --dry-run
  python glorys_downloader.py --workers 2
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3

# ---- CONFIG ----
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ["uo", "vo", "thetao"]
BUCKET_NAME = "panthalassa-ocean-raw-data"
S3_PREFIX = "glorys-data-pnw"
LOCAL_TEMP_DIR = "./temp_downloads"

# Pacific Northwest / Washington coast, matching the other PNW datasets.
MIN_LON, MAX_LON = -130.0, -124.0
MIN_LAT, MAX_LAT = 46.0, 50.5

# Surface level only (top model layer sits at ~0.49 m)
MIN_DEPTH, MAX_DEPTH = 0.0, 1.0

YEARS = range(2000, 2020)

log = logging.getLogger("glorys_downloader")


def output_name(year):
    return f"glorys_pnw_{year}.nc"


def s3_key(year):
    return f"{S3_PREFIX}/{output_name(year)}"


def completed_from_s3(s3):
    completed = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=f"{S3_PREFIX}/"):
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


def download_year(year):
    # Imported here so --dry-run works on machines without copernicusmarine.
    import copernicusmarine

    name = output_name(year)
    # UTC-aware datetimes: copernicusmarine v2 treats naive ones as local
    # time and silently shifts the window.
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=VARIABLES,
        start_datetime=datetime(year, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(year, 12, 31, tzinfo=timezone.utc),
        minimum_longitude=MIN_LON,
        maximum_longitude=MAX_LON,
        minimum_latitude=MIN_LAT,
        maximum_latitude=MAX_LAT,
        minimum_depth=MIN_DEPTH,
        maximum_depth=MAX_DEPTH,
        output_directory=LOCAL_TEMP_DIR,
        output_filename=name,
        overwrite=True,
        disable_progress_bar=True,
    )
    return os.path.join(LOCAL_TEMP_DIR, name)


def process_year(s3, year):
    name = output_name(year)
    temp_file = None
    try:
        log.info("Downloading %s", name)
        temp_file = retry(download_year, year)
        key = s3_key(year)
        log.info("Uploading %s to s3://%s/%s", name, BUCKET_NAME, key)
        retry(s3.upload_file, temp_file, BUCKET_NAME, key)
        log.info("%s complete", name)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="report resume state without downloading")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel yearly downloads (default 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N years (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

    # WAVE_S3_ENDPOINT → MinIO on the devbox for local runs; unset → AWS.
    s3 = boto3.client("s3", endpoint_url=os.environ.get("WAVE_S3_ENDPOINT"))
    completed = completed_from_s3(s3)
    pending = [y for y in YEARS if output_name(y) not in completed]

    log.info("%d years total, %d already in S3, %d pending",
             len(list(YEARS)), len(list(YEARS)) - len(pending), len(pending))
    if pending:
        log.info("Resume point: %d", pending[0])
    if args.dry_run or not pending:
        return
    if args.limit:
        pending = pending[: args.limit]
        log.info("Limiting this run to %d year(s)", len(pending))

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_year, s3, y): y for y in pending}
        for future in as_completed(futures):
            year = futures[future]
            try:
                future.result()
            except Exception as e:
                failures.append(str(year))
                log.error("Year %d failed permanently: %s", year, e)

    if failures:
        log.error("%d years failed: %s — rerun to retry them",
                  len(failures), ", ".join(sorted(failures)))
        raise SystemExit(1)
    log.info("All years downloaded and uploaded successfully.")


if __name__ == "__main__":
    main()
