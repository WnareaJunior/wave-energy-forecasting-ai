"""Download Copernicus Marine wave reanalysis for the study region into S3.

Downloads in monthly chunks, uploads each to the raw bucket, and records
progress in a local log so an interrupted run resumes where it stopped.

Region and dataset now come from src.config. The original script was hardcoded
to +124.5 to +144.6 E / 16.7 to 48.2 N — the Western Pacific off Japan — which
does not overlap the Pacific Northwest region the rest of the project uses.

Run ``copernicusmarine login`` once on the host before using this.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import boto3
import copernicusmarine
from dask.diagnostics import ProgressBar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (  # noqa: E402
    COPERNICUS_DATASET_ID,
    COPERNICUS_END,
    COPERNICUS_START,
    RAW_BUCKET,
    STUDY_REGION,
)

LOCAL_TEMP_DIR = "./temp_downloads"
LOG_FILE = "./download_log.json"


def load_log(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_log(path, completed_chunks):
    with open(path, "w") as f:
        json.dump(completed_chunks, f, indent=2)


def generate_chunks(start, end):
    """Yield (start, end) pairs covering [start, end) in ~1-month steps."""
    current = start
    while current < end:
        month = current.month % 12 + 1
        year = current.year + (current.month // 12)
        next_chunk = datetime(year, month, min(current.day, 28), current.hour)
        if next_chunk > end:
            next_chunk = end
        yield current, next_chunk
        current = next_chunk


def retry(func, *args, max_attempts=3, delay=10, **kwargs):
    """Call func with retries and a fixed delay between attempts."""
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise


def main(bucket_name=RAW_BUCKET, temp_dir=LOCAL_TEMP_DIR, log_file=LOG_FILE):
    os.makedirs(temp_dir, exist_ok=True)
    s3 = boto3.client("s3")
    completed_chunks = load_log(log_file)

    start_date = datetime.fromisoformat(COPERNICUS_START)
    end_date = datetime.fromisoformat(COPERNICUS_END)

    print(f"Dataset: {COPERNICUS_DATASET_ID}")
    print(f"Region:  {STUDY_REGION.name} "
          f"({STUDY_REGION.lon_min} to {STUDY_REGION.lon_max} E, "
          f"{STUDY_REGION.lat_min} to {STUDY_REGION.lat_max} N)")
    print(f"Period:  {start_date} to {end_date}")

    for chunk_start, chunk_end in generate_chunks(start_date, end_date):
        chunk_name = f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"
        if chunk_name in completed_chunks:
            print(f"Skipping already completed chunk {chunk_name}")
            continue

        print(f"Processing chunk: {chunk_start} -> {chunk_end}")
        temp_file = os.path.join(temp_dir, f"{chunk_name}.nc")

        ds = retry(
            copernicusmarine.open_dataset,
            dataset_id=COPERNICUS_DATASET_ID,
            start_datetime=chunk_start,
            end_datetime=chunk_end,
            minimum_longitude=STUDY_REGION.lon_min,
            maximum_longitude=STUDY_REGION.lon_max,
            minimum_latitude=STUDY_REGION.lat_min,
            maximum_latitude=STUDY_REGION.lat_max,
            chunk_size_limit=1000,
        )

        print(f"Saving chunk to {temp_file} using dask...")
        with ProgressBar():
            retry(ds.chunk({"time": 50}).to_netcdf, path=temp_file)

        s3_key = f"copernicus_raw/{chunk_start.year}/{chunk_name}.nc"
        print(f"Uploading {temp_file} to s3://{bucket_name}/{s3_key}...")
        retry(s3.upload_file, Filename=temp_file, Bucket=bucket_name, Key=s3_key)

        os.remove(temp_file)
        completed_chunks.append(chunk_name)
        save_log(log_file, completed_chunks)
        print(f"Chunk {chunk_name} completed and uploaded.\n")

    print("All chunks downloaded and uploaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=RAW_BUCKET)
    parser.add_argument("--temp-dir", default=LOCAL_TEMP_DIR)
    parser.add_argument("--log-file", default=LOG_FILE)
    args = parser.parse_args()

    main(args.bucket, args.temp_dir, args.log_file)
