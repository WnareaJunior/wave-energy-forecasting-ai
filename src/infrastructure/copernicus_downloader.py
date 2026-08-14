"""Download the Copernicus Marine global wave hindcast for a study region
in monthly chunks and upload each chunk to S3.

Resume state is derived from the S3 bucket itself (no local log file): any
chunk whose .nc object already exists under the destination prefix is
skipped, so the script can be killed and restarted — or moved to a fresh
EC2 instance — at any point without losing progress.

Credentials:
  - Copernicus: COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD env vars,
    or a prior `copernicusmarine login` on this machine.
  - AWS: default boto3 credential chain (IAM instance role on EC2).

Usage:
  python copernicus_downloader.py --region pnw --dry-run
  python copernicus_downloader.py --region japan --workers 3
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import boto3

# ---- CONFIG ----
DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
BUCKET_NAME = "panthalassa-ocean-raw-data"
LOCAL_TEMP_DIR = "./temp_downloads"

REGIONS = {
    # Original study region; prefix layout predates multi-region support.
    "japan": {
        "lon": (124.52, 144.6),
        "lat": (16.745, 48.185),
        "prefix": "copernicus-data",
    },
    # Pacific Northwest / Washington coast — the long-term study region.
    "pnw": {
        "lon": (-130.0, -124.0),
        "lat": (46.0, 50.5),
        "prefix": "copernicus-data-pnw",
    },
}

# Full temporal range of the multi-year (reprocessed) product
START_DATE = datetime(1980, 1, 1, 21)
END_DATE = datetime(2023, 4, 30, 21)

# The dataset is 3-hourly; trimming this off each chunk's end avoids
# duplicating the boundary timestep that also starts the next chunk.
TIME_STEP = timedelta(hours=3)

log = logging.getLogger("copernicus_downloader")


def generate_chunks(start, end):
    """Yield (chunk_start, chunk_end) monthly pairs.

    Boundary arithmetic must stay identical to the original 2025 run so
    that generated chunk names match the .nc files already in S3.
    """
    current = start
    while current < end:
        month = current.month % 12 + 1
        year = current.year + (current.month // 12)
        next_chunk = datetime(year, month, min(current.day, 28), current.hour)
        if next_chunk > end:
            next_chunk = end
        yield current, next_chunk
        current = next_chunk


def chunk_name(chunk_start, chunk_end):
    return f"{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}"


def s3_key(region, chunk_start, name):
    return f"{region['prefix']}/{chunk_start.year}/{name}.nc"


def completed_chunks_from_s3(s3, region):
    """Return the set of chunk names that already exist in the bucket."""
    completed = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME,
                                   Prefix=f"{region['prefix']}/"):
        for obj in page.get("Contents", []):
            basename = obj["Key"].rsplit("/", 1)[-1]
            if basename.endswith(".nc") and obj["Size"] > 0:
                completed.add(basename[: -len(".nc")])
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


def download_chunk(region, name, chunk_start, request_end):
    # Imported here so --dry-run works on machines without copernicusmarine.
    import copernicusmarine

    # Naive datetimes get interpreted as *local* time by copernicusmarine
    # v2 and silently shift the window; the dataset's time axis is UTC.
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        start_datetime=chunk_start.replace(tzinfo=timezone.utc),
        end_datetime=request_end.replace(tzinfo=timezone.utc),
        minimum_longitude=region["lon"][0],
        maximum_longitude=region["lon"][1],
        minimum_latitude=region["lat"][0],
        maximum_latitude=region["lat"][1],
        output_directory=LOCAL_TEMP_DIR,
        output_filename=f"{name}.nc",
        overwrite=True,
        disable_progress_bar=True,
    )
    return os.path.join(LOCAL_TEMP_DIR, f"{name}.nc")


def process_chunk(s3, region, chunk_start, chunk_end, is_last):
    name = chunk_name(chunk_start, chunk_end)
    # Trim one timestep except on the final chunk, whose end is the real
    # end of the record rather than the start of a following chunk.
    request_end = chunk_end if is_last else chunk_end - TIME_STEP
    key = s3_key(region, chunk_start, name)
    temp_file = None
    try:
        log.info("Downloading %s (%s -> %s)", name, chunk_start, request_end)
        temp_file = retry(download_chunk, region, name, chunk_start, request_end)
        log.info("Uploading %s to s3://%s/%s", name, BUCKET_NAME, key)
        retry(s3.upload_file, temp_file, BUCKET_NAME, key)
        log.info("Chunk %s complete", name)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--region", choices=sorted(REGIONS), default="japan",
                        help="study region preset (default japan)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report resume state without downloading")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel chunk downloads (default 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N pending chunks (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

    region = REGIONS[args.region]
    # WAVE_S3_ENDPOINT → MinIO on the devbox for local runs; unset → AWS.
    s3 = boto3.client("s3", endpoint_url=os.environ.get("WAVE_S3_ENDPOINT"))
    completed = completed_chunks_from_s3(s3, region)

    all_chunks = list(generate_chunks(START_DATE, END_DATE))
    pending = [(cs, ce) for cs, ce in all_chunks
               if chunk_name(cs, ce) not in completed]

    log.info("[%s] %d chunks total, %d already in S3, %d pending",
             args.region, len(all_chunks), len(all_chunks) - len(pending),
             len(pending))
    if pending:
        log.info("Resume point: %s", pending[0][0])
    if args.dry_run or not pending:
        return
    if args.limit:
        pending = pending[: args.limit]
        log.info("Limiting this run to %d chunk(s)", len(pending))

    last_start = all_chunks[-1][0]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_chunk, s3, region, cs, ce, cs == last_start):
                (cs, ce)
            for cs, ce in pending
        }
        for future in as_completed(futures):
            cs, ce = futures[future]
            try:
                future.result()
            except Exception as e:
                name = chunk_name(cs, ce)
                failures.append(name)
                log.error("Chunk %s failed permanently: %s", name, e)

    if failures:
        log.error("%d chunks failed: %s — rerun to retry them",
                  len(failures), ", ".join(sorted(failures)))
        raise SystemExit(1)
    log.info("All chunks downloaded and uploaded successfully.")


if __name__ == "__main__":
    main()
