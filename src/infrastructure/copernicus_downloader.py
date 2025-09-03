import os
import json
import time
from datetime import datetime, timedelta
import copernicusmarine
import boto3
from dask.diagnostics import ProgressBar

# ---- CONFIG ----
dataset_id = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
bucket_name = "panthalassa-ocean-raw-data"
local_temp_dir = "./temp_downloads"
log_file = "./download_log.json"
os.makedirs(local_temp_dir, exist_ok=True)

# AWS S3 client
s3 = boto3.client('s3')

# Spatial bounds
min_lon, max_lon = 124.52, 144.6
min_lat, max_lat = 16.745, 48.185

# Full temporal range
start_date = datetime(1980, 1, 1, 21)
end_date   = datetime(2023, 4, 30, 21)

# ---- Logging ----
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        completed_chunks = json.load(f)
else:
    completed_chunks = []

def save_log():
    with open(log_file, "w") as f:
        json.dump(completed_chunks, f, indent=2)

# ---- Helper: generate 1-month chunks ----
def generate_chunks(start, end):
    current = start
    while current < end:
        month = current.month % 12 + 1
        year = current.year + (current.month // 12)
        next_chunk = datetime(year, month, min(current.day, 28), current.hour)
        if next_chunk > end:
            next_chunk = end
        yield current, next_chunk
        current = next_chunk

# ---- Retry helper ----
def retry(func, max_attempts=3, delay=10, *args, **kwargs):
    for attempt in range(1, max_attempts+1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise

# ---- Download & upload loop ----
for chunk_start, chunk_end in generate_chunks(start_date, end_date):
    chunk_name = f"{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}"
    if chunk_name in completed_chunks:
        print(f"Skipping already completed chunk {chunk_name}")
        continue

    print(f"Processing chunk: {chunk_start} → {chunk_end}")

    temp_file = os.path.join(local_temp_dir, f"{chunk_name}.nc")

    # Download dataset chunk lazily
    ds = retry(
        copernicusmarine.open_dataset,
        dataset_id=dataset_id,
        start_datetime=chunk_start,
        end_datetime=chunk_end,
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        chunk_size_limit=1000
    )

    print(f"Saving chunk to {temp_file} using dask...")
    with ProgressBar():
        retry(ds.chunk({'time': 50}).to_netcdf, path=temp_file)

    # Organize S3 keys by year
    s3_key = f"{chunk_start.year}/{chunk_name}.nc"
    print(f"Uploading {temp_file} to s3://{bucket_name}/{s3_key}...")
    retry(s3.upload_file, Filename=temp_file, Bucket=bucket_name, Key=s3_key)

    # Remove local file
    os.remove(temp_file)

    # Log completed chunk
    completed_chunks.append(chunk_name)
    save_log()
    print(f"Chunk {chunk_name} completed and uploaded.\n")

print("All chunks downloaded and uploaded successfully.")

