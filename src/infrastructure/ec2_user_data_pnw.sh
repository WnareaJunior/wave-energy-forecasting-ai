#!/bin/bash
# Bootstrap for the PNW Copernicus instance (Amazon Linux 2023).
# Runs the PNW wave hindcast backfill, then the GLORYS currents/SST
# download, then self-terminates. Same pattern as ec2_user_data.sh.
set -euxo pipefail
exec > /var/log/panthalassa-bootstrap.log 2>&1

dnf install -y python3.11 python3.11-pip
mkdir -p /opt/panthalassa
python3.11 -m venv /opt/panthalassa/venv
/opt/panthalassa/venv/bin/pip install --no-cache-dir copernicusmarine==2.4.1 boto3

aws s3 cp s3://panthalassa-ocean-raw-data/scripts/copernicus_downloader.py \
    /opt/panthalassa/copernicus_downloader.py --region us-east-1
aws s3 cp s3://panthalassa-ocean-raw-data/scripts/glorys_downloader.py \
    /opt/panthalassa/glorys_downloader.py --region us-east-1

cat > /opt/panthalassa/run.sh <<'EOF'
#!/bin/bash
set -uo pipefail
export AWS_DEFAULT_REGION=us-east-1
export COPERNICUSMARINE_SERVICE_USERNAME=$(aws ssm get-parameter \
    --name /panthalassa/copernicus/username --with-decryption \
    --query Parameter.Value --output text)
export COPERNICUSMARINE_SERVICE_PASSWORD=$(aws ssm get-parameter \
    --name /panthalassa/copernicus/password --with-decryption \
    --query Parameter.Value --output text)
cd /opt/panthalassa
venv/bin/python copernicus_downloader.py --region pnw --workers 3 \
    && venv/bin/python glorys_downloader.py --workers 2
code=$?
if [ $code -eq 0 ]; then
    echo "PNW backfill (waves + GLORYS) complete - shutting down"
    shutdown -h now
fi
exit $code
EOF
chmod +x /opt/panthalassa/run.sh

cat > /etc/systemd/system/panthalassa-pnw-downloader.service <<'EOF'
[Unit]
Description=Panthalassa PNW backfill (Copernicus waves + GLORYS)
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
systemctl enable --now panthalassa-pnw-downloader.service
