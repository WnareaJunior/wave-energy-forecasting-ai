import boto3
import xarray as xr
from io import BytesIO
import os
import json

# --- CONFIG ---
source_bucket = 'noaa-nws-gefswaves-reforecast-pds'
source_prefix = 'GEFSv12/reforecast/2019/'
destination_bucket = 'panthalassa-ocean-raw-data'
destination_prefix = 'noaa-data/2019/'

lat_min, lat_max = 25, 30       # your latitude bounds
lon_min, lon_max = -130, -120   # your longitude bounds

local_tmp = '/tmp'  # small temporary folder for NetCDFs
checkpoint_file = 'progress_checkpoint.json'

# --- S3 CLIENT ---
s3 = boto3.client('s3')

# --- LOAD OR INIT CHECKPOINT ---
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        completed_files = set(json.load(f))
else:
    completed_files = set()

# --- LIST 2019 FILES ---
objects = s3.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix)
grib_files = [obj['Key'] for obj in objects.get('Contents', []) if obj['Key'].endswith('.grib2')]

print(f"Found {len(grib_files)} files in 2019.")

# --- PROCESS FILES ---
for key in grib_files:
    filename = key.split('/')[-1]

    if filename in completed_files:
        print(f"Skipping {filename} (already done).")
        continue

    try:
        print(f"Processing {filename} ...")

        # Stream from S3
        obj = s3.get_object(Bucket=source_bucket, Key=key)
        file_stream = BytesIO(obj['Body'].read())

        # Open GRIB2 from memory
        ds = xr.open_dataset(file_stream, engine='cfgrib')

        # Subset coordinates
        subset = ds.sel(latitude=slice(lat_min, lat_max),
                        longitude=slice(lon_min, lon_max))

        # Prepare local and destination paths
        date_folder = key.split('/')[4]  # e.g., '20190101'
        netcdf_name = filename.replace('.grib2', '.nc')
        local_nc_path = os.path.join(local_tmp, netcdf_name)
        dest_key = f"{destination_prefix}{date_folder}/{netcdf_name}"

        # Save NetCDF locally
        subset.to_netcdf(local_nc_path)

        # Upload to your bucket
        s3.upload_file(local_nc_path, destination_bucket, dest_key)

        # Cleanup local file
        os.remove(local_nc_path)

        # Update checkpoint
        completed_files.add(filename)
        with open(checkpoint_file, 'w') as f:
            json.dump(list(completed_files), f)

        print(f"Completed {filename}.")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

