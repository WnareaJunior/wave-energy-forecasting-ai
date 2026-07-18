#!/usr/bin/env bash
# Provision an EC2 instance that runs the NOAA GEFSv12 reforecast backfill
# for one ensemble member (default c00; pass e.g. p01 to fetch a perturbed
# member on its own instance — one instance per member parallelizes the
# backfill without S3-resume collisions).
#
# Reuses the IAM role/instance profile created by launch_copernicus_ec2.sh
# (panthalassa-copernicus-ec2) — run that script first if it doesn't exist.
# Idempotent: skips the launch if this member's instance is already
# pending/running. The instance self-terminates when the backfill finishes.
set -euo pipefail

MEMBER=${1:-c00}

REGION=us-east-1
BUCKET=panthalassa-ocean-raw-data
ROLE=panthalassa-copernicus-ec2
INSTANCE_NAME=panthalassa-noaa-downloader
if [ "$MEMBER" != "c00" ]; then
    INSTANCE_NAME=$INSTANCE_NAME-$MEMBER
fi
SCRIPT_S3_KEY=scripts/noaa_downloader.py

cd "$(dirname "$0")"

echo "==> Verifying IAM instance profile exists"
aws iam get-instance-profile --instance-profile-name "$ROLE" \
    --query InstanceProfile.InstanceProfileName --output text

echo "==> Uploading downloader script to S3"
aws s3 cp noaa_downloader.py "s3://$BUCKET/$SCRIPT_S3_KEY" --region "$REGION"

echo "==> Checking for an existing $INSTANCE_NAME instance"
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

USER_DATA=$(mktemp)
trap 'rm -f "$USER_DATA"' EXIT
sed "s/__MEMBERS__/$MEMBER/" ec2_user_data_noaa.sh > "$USER_DATA"

echo "==> Launching $INSTANCE_NAME (AMI $AMI, member $MEMBER)"
aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" \
    --instance-type t3.medium \
    --iam-instance-profile "Name=$ROLE" \
    --user-data "file://$USER_DATA" \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$INSTANCE_NAME},
        {Key=Project,Value=\"Panthalassa Ocean Forecasting\"},
        {Key=Purpose,Value=\"NOAA GEFSv12 reforecast backfill\"}]" \
    --query 'Instances[0].{InstanceId:InstanceId,AZ:Placement.AvailabilityZone}' \
    --output table
