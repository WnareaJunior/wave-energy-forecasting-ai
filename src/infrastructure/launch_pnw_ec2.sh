#!/usr/bin/env bash
# Provision the EC2 instance that runs the PNW backfills: Copernicus wave
# hindcast (1980-2023) followed by GLORYS surface currents/SST (2000-2019).
#
# Reuses the IAM role/instance profile and SSM credentials created by
# launch_copernicus_ec2.sh — run that first if they don't exist.
# Idempotent; the instance self-terminates when both backfills finish.
set -euo pipefail

REGION=us-east-1
BUCKET=panthalassa-ocean-raw-data
ROLE=panthalassa-copernicus-ec2
INSTANCE_NAME=panthalassa-pnw-downloader

cd "$(dirname "$0")"

echo "==> Verifying IAM instance profile and SSM credentials exist"
aws iam get-instance-profile --instance-profile-name "$ROLE" \
    --query InstanceProfile.InstanceProfileName --output text
for p in username password; do
    aws ssm get-parameter --name /panthalassa/copernicus/$p \
        --region "$REGION" --query Parameter.Name --output text
done

echo "==> Uploading downloader scripts to S3"
aws s3 cp copernicus_downloader.py "s3://$BUCKET/scripts/copernicus_downloader.py" --region "$REGION"
aws s3 cp glorys_downloader.py "s3://$BUCKET/scripts/glorys_downloader.py" --region "$REGION"

echo "==> Checking for an existing PNW downloader instance"
EXISTING=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
              "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$EXISTING" ]; then
    echo "Instance already running: $EXISTING — nothing to do"
    exit 0
fi

AMI=$(aws ssm get-parameter --region "$REGION" \
    --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query Parameter.Value --output text)

echo "==> Launching instance (AMI $AMI)"
aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" \
    --instance-type t3.medium \
    --iam-instance-profile "Name=$ROLE" \
    --user-data file://ec2_user_data_pnw.sh \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$INSTANCE_NAME},
        {Key=Project,Value=\"Panthalassa Ocean Forecasting\"},
        {Key=Purpose,Value=\"PNW Copernicus waves + GLORYS backfill\"}]" \
    --query 'Instances[0].{InstanceId:InstanceId,AZ:Placement.AvailabilityZone}' \
    --output table
