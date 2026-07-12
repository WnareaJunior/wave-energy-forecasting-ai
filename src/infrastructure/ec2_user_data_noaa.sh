#!/bin/bash
# Bootstrap for the NOAA GEFSv12 reforecast instance (Amazon Linux 2023).
# Same pattern as ec2_user_data.sh: pull the downloader from S3, run it
# under systemd, self-terminate on success. No SSM credentials needed —
# the source bucket is public.
set -euxo pipefail
exec > /var/log/panthalassa-bootstrap.log 2>&1

dnf install -y python3.11 python3.11-pip
mkdir -p /opt/panthalassa
python3.11 -m venv /opt/panthalassa/venv
/opt/panthalassa/venv/bin/pip install --no-cache-dir boto3 xarray netcdf4 cfgrib eccodes
/opt/panthalassa/venv/bin/python -c "import cfgrib" || echo "WARNING: cfgrib import failed"

aws s3 cp s3://panthalassa-ocean-raw-data/scripts/noaa_downloader.py \
    /opt/panthalassa/noaa_downloader.py --region us-east-1

cat > /opt/panthalassa/run.sh <<'EOF'
#!/bin/bash
set -uo pipefail
export AWS_DEFAULT_REGION=us-east-1
cd /opt/panthalassa
venv/bin/python noaa_downloader.py --workers 3
code=$?
if [ $code -eq 0 ]; then
    echo "NOAA backfill complete - shutting down"
    shutdown -h now
fi
exit $code
EOF
chmod +x /opt/panthalassa/run.sh

cat > /etc/systemd/system/panthalassa-noaa-downloader.service <<'EOF'
[Unit]
Description=Panthalassa NOAA GEFSv12 reforecast backfill
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/opt/panthalassa/run.sh
Restart=on-failure
RestartSec=120
StandardOutput=append:/var/log/panthalassa-downloader.log
StandardError=append:/var/log/panthalassa-downloader.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now panthalassa-noaa-downloader.service
