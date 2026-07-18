#!/usr/bin/env bash
# Set up the daily forecast-table sweep: an EventBridge Scheduler rule that
# launches a self-terminating EC2 instance (from a launch template) once a
# day at 12:00 UTC. The instance runs build_forecast_table.py --sweep,
# which builds the forecast-table Parquet for any year/member whose raw
# NOAA backfill completed since the last run, and deletes the schedule
# itself once every year/member is built.
#
# Idempotent: safe to rerun; updates the role policy, script copy, launch
# template version, and schedule in place.
set -euo pipefail

REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
RAW_BUCKET=panthalassa-ocean-raw-data
PROCESSED_BUCKET=panthalassa-ocean-processed
INSTANCE_ROLE=panthalassa-copernicus-ec2
SCHEDULER_ROLE=panthalassa-scheduler
TEMPLATE_NAME=panthalassa-forecast-sweep
SCHEDULE_NAME=panthalassa-forecast-sweep

cd "$(dirname "$0")"

echo "==> Extending $INSTANCE_ROLE policy (processed bucket + schedule self-delete)"
aws iam put-role-policy --role-name "$INSTANCE_ROLE" \
    --policy-name "$INSTANCE_ROLE-policy" \
    --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": \"s3:ListBucket\",
     \"Resource\": [\"arn:aws:s3:::$RAW_BUCKET\", \"arn:aws:s3:::$PROCESSED_BUCKET\"]},
    {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:PutObject\"],
     \"Resource\": [\"arn:aws:s3:::$RAW_BUCKET/*\", \"arn:aws:s3:::$PROCESSED_BUCKET/*\"]},
    {\"Effect\": \"Allow\", \"Action\": \"ssm:GetParameter\",
     \"Resource\": \"arn:aws:ssm:$REGION:$ACCOUNT:parameter/panthalassa/copernicus/*\"},
    {\"Effect\": \"Allow\", \"Action\": \"scheduler:DeleteSchedule\",
     \"Resource\": \"arn:aws:scheduler:$REGION:$ACCOUNT:schedule/default/$SCHEDULE_NAME\"}
  ]
}"

echo "==> Uploading build_forecast_table.py to S3"
aws s3 cp ../processing/build_forecast_table.py \
    "s3://$RAW_BUCKET/scripts/build_forecast_table.py" --region "$REGION"

echo "==> Creating/updating launch template $TEMPLATE_NAME"
AMI=$(aws ssm get-parameter --region "$REGION" \
    --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query Parameter.Value --output text)
LT_DATA=$(python3 - "$AMI" <<'EOF'
import base64, json, sys
user_data = base64.b64encode(open("ec2_user_data_sweep.sh", "rb").read()).decode()
print(json.dumps({
    "ImageId": sys.argv[1],
    "InstanceType": "t3.xlarge",
    "IamInstanceProfile": {"Name": "panthalassa-copernicus-ec2"},
    "UserData": user_data,
    "InstanceInitiatedShutdownBehavior": "terminate",
    "MetadataOptions": {"HttpTokens": "required"},
    "BlockDeviceMappings": [{"DeviceName": "/dev/xvda",
        "Ebs": {"VolumeSize": 30, "VolumeType": "gp3",
                "DeleteOnTermination": True}}],
    "TagSpecifications": [{"ResourceType": "instance", "Tags": [
        {"Key": "Name", "Value": "panthalassa-forecast-sweep"},
        {"Key": "Project", "Value": "Panthalassa Ocean Forecasting"},
        {"Key": "Purpose", "Value": "Daily forecast-table sweep"}]}],
}))
EOF
)
if aws ec2 describe-launch-templates --region "$REGION" \
        --launch-template-names "$TEMPLATE_NAME" >/dev/null 2>&1; then
    aws ec2 create-launch-template-version --region "$REGION" \
        --launch-template-name "$TEMPLATE_NAME" \
        --launch-template-data "$LT_DATA" --query \
        'LaunchTemplateVersion.VersionNumber' --output text | while read -r v; do
            aws ec2 modify-launch-template --region "$REGION" \
                --launch-template-name "$TEMPLATE_NAME" --default-version "$v"
        done
else
    aws ec2 create-launch-template --region "$REGION" \
        --launch-template-name "$TEMPLATE_NAME" \
        --launch-template-data "$LT_DATA" >/dev/null
fi

echo "==> Creating/updating scheduler execution role $SCHEDULER_ROLE"
aws iam create-role --role-name "$SCHEDULER_ROLE" \
    --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow",
    "Principal": {"Service": "scheduler.amazonaws.com"},
    "Action": "sts:AssumeRole"}]
}' 2>/dev/null || echo "role exists"
aws iam put-role-policy --role-name "$SCHEDULER_ROLE" \
    --policy-name "$SCHEDULER_ROLE-policy" \
    --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": [\"ec2:RunInstances\", \"ec2:CreateTags\"],
     \"Resource\": \"*\"},
    {\"Effect\": \"Allow\", \"Action\": \"iam:PassRole\",
     \"Resource\": \"arn:aws:iam::$ACCOUNT:role/$INSTANCE_ROLE\",
     \"Condition\": {\"StringEquals\": {\"iam:PassedToService\": \"ec2.amazonaws.com\"}}}
  ]
}"

echo "==> Creating/updating schedule $SCHEDULE_NAME (daily 12:00 UTC)"
# Target JSON goes through a file: inline --target strings get mangled by
# CLI/shell quoting (the role ARN's ":r" was eaten when passed inline).
TARGET_FILE=$(mktemp)
trap 'rm -f "$TARGET_FILE"' EXIT
cat > "$TARGET_FILE" <<EOF
{
  "Arn": "arn:aws:scheduler:::aws-sdk:ec2:runInstances",
  "RoleArn": "arn:aws:iam::$ACCOUNT:role/$SCHEDULER_ROLE",
  "Input": "{\"LaunchTemplate\": {\"LaunchTemplateName\": \"$TEMPLATE_NAME\", \"Version\": \"\$Latest\"}, \"MinCount\": 1, \"MaxCount\": 1}"
}
EOF
if aws scheduler get-schedule --region "$REGION" --name "$SCHEDULE_NAME" \
        >/dev/null 2>&1; then
    aws scheduler update-schedule --region "$REGION" --name "$SCHEDULE_NAME" \
        --schedule-expression "cron(0 12 * * ? *)" \
        --flexible-time-window Mode=OFF \
        --target "file://$TARGET_FILE" >/dev/null
else
    aws scheduler create-schedule --region "$REGION" --name "$SCHEDULE_NAME" \
        --schedule-expression "cron(0 12 * * ? *)" \
        --flexible-time-window Mode=OFF \
        --target "file://$TARGET_FILE" >/dev/null
fi

echo "==> Done. Next sweep: 12:00 UTC daily; schedule self-deletes when the"
echo "    forecast table is complete."
