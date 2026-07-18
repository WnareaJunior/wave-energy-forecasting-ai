#!/bin/bash
# Bootstrap for the daily forecast-table sweep instance (Amazon Linux 2023).
# Launched on a schedule by EventBridge Scheduler (see
# setup_sweep_schedule.sh); builds any year/member whose raw NOAA archive
# completed since the last sweep, then shuts down (instance terminates on
# shutdown). When the sweep reports everything built, it deletes its own
# schedule so no further instances launch.
set -uxo pipefail
exec > /var/log/panthalassa-bootstrap.log 2>&1
trap 'shutdown -h now' EXIT

export AWS_DEFAULT_REGION=us-east-1

dnf install -y python3.11 python3.11-pip
mkdir -p /opt/panthalassa
python3.11 -m venv /opt/panthalassa/venv
/opt/panthalassa/venv/bin/pip install --no-cache-dir \
    boto3 xarray pandas pyarrow netcdf4

aws s3 cp s3://panthalassa-ocean-raw-data/scripts/build_forecast_table.py \
    /opt/panthalassa/build_forecast_table.py

cd /opt/panthalassa
venv/bin/python build_forecast_table.py --sweep \
    --members c00 p01 p02 p03 p04 2>&1 | tee /var/log/panthalassa-sweep.log

if grep -q SWEEP_COMPLETE /var/log/panthalassa-sweep.log; then
    echo "Forecast table complete - removing sweep schedule"
    aws scheduler delete-schedule --name panthalassa-forecast-sweep
fi
