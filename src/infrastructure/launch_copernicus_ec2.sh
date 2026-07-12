#!/usr/bin/env bash
# Provision the EC2 instance that runs the Copernicus backfill.
#
# Idempotent: safe to rerun. Skips the launch if a downloader instance is
# already pending/running. Expects Copernicus credentials to already exist
# in SSM Parameter Store as SecureStrings:
#   /panthalassa/copernicus/username
#   /panthalassa/copernicus/password
#
# The instance self-terminates when the backfill finishes (see
# ec2_user_data.sh), so there is nothing to tear down on success.
set -euo pipefail

REGION=us-east-1
BUCKET=panthalassa-ocean-raw-data
ROLE=panthalassa-copernicus-ec2
POLICY=panthalassa-copernicus-ec2-policy
INSTANCE_NAME=panthalassa-copernicus-downloader
SCRIPT_S3_KEY=scripts/copernicus_downloader.py
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

cd "$(dirname "$0")"

echo "==> Verifying SSM credentials exist"
for p in username password; do
    aws ssm get-parameter --name /panthalassa/copernicus/$p \
        --region "$REGION" --query Parameter.Name --output text
done

echo "==> Uploading downloader script to S3"
aws s3 cp copernicus_downloader.py "s3://$BUCKET/$SCRIPT_S3_KEY" --region "$REGION"

echo "==> Ensuring IAM role and instance profile"
CREATED_ROLE=false
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' >/dev/null
    CREATED_ROLE=true
fi

aws iam put-role-policy --role-name "$ROLE" --policy-name "$POLICY" \
    --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
        {
            \"Effect\": \"Allow\",
            \"Action\": \"s3:ListBucket\",
            \"Resource\": \"arn:aws:s3:::$BUCKET\"
        },
        {
            \"Effect\": \"Allow\",
            \"Action\": [\"s3:GetObject\", \"s3:PutObject\"],
            \"Resource\": \"arn:aws:s3:::$BUCKET/*\"
        },
        {
            \"Effect\": \"Allow\",
            \"Action\": \"ssm:GetParameter\",
            \"Resource\": \"arn:aws:ssm:$REGION:$ACCOUNT_ID:parameter/panthalassa/copernicus/*\"
        }
    ]
}"
aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" \
        --role-name "$ROLE"
fi

if [ "$CREATED_ROLE" = true ]; then
    echo "==> Waiting for IAM propagation"
    sleep 15
fi

echo "==> Checking for an existing downloader instance"
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
    --user-data file://ec2_user_data.sh \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$INSTANCE_NAME},
        {Key=Project,Value=\"Panthalassa Ocean Forecasting\"},
        {Key=Purpose,Value=\"Copernicus Marine backfill\"}]" \
    --query 'Instances[0].{InstanceId:InstanceId,AZ:Placement.AvailabilityZone}' \
    --output table
