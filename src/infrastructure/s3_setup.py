import boto3
import json
from botocore.exceptions import ClientError

def create_panthalassa_buckets():
    """
    Create S3 buckets for Panthalassa ocean forecasting project
    """
    s3 = boto3.client('s3', region_name='us-east-2')
    
    bucket_configs = [
        {
            'name': 'panthalassa-ocean-raw-data',
            'description': 'Raw Copernicus Marine and NOAA wave data',
            'folders': ['copernicus_raw/', 'noaa_wave_data/', 'api_tests/', 'connectivity_tests/']
        },
        {
            'name': 'panthalassa-ocean-processed', 
            'description': 'Cleaned and filtered ocean datasets',
            'folders': ['wave_power_flux/', 'monthly_aggregations/', 'spatial_subsets/']
        },
        {
            'name': 'panthalassa-ocean-results',
            'description': 'Model outputs, forecasts, and analysis results', 
            'folders': ['ml_models/', 'forecasts/', 'visualizations/', 'reports/']
        }
    ]
    
    for config in bucket_configs:
        try:
            # Create bucket with correct region configuration
            if s3.meta.region_name == 'us-east-1':
                s3.create_bucket(Bucket=config['name'])
            else:
                s3.create_bucket(
                    Bucket=config['name'],
                    CreateBucketConfiguration={'LocationConstraint': s3.meta.region_name}
                )
            
            print(f"Created bucket: {config['name']}")
            
            # Create folder structure
            for folder in config['folders']:
                s3.put_object(
                    Bucket=config['name'],
                    Key=folder,
                    Body=''
                )
                print(f"  Created folder: {folder}")
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"Bucket {config['name']} already exists")
            else:
                print(f"Error creating bucket {config['name']}: {e}")

def configure_bucket_policies():
    """
    Set up bucket policies for data access
    """
    s3 = boto3.client('s3')
    
    # Example policy for allowing EC2 instances access
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowEC2Access",
                "Effect": "Allow", 
                "Principal": {"AWS": "arn:aws:iam::318865045196:root"},
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": "arn:aws:s3:::panthalassa-ocean-*/*"
            }
        ]
    }
    
    return bucket_policy

def list_bucket_contents():
    """
    Utility function to check bucket contents
    """
    s3 = boto3.client('s3')
    
    buckets = [
        'panthalassa-ocean-raw-data',
        'panthalassa-ocean-processed', 
        'panthalassa-ocean-results'
    ]
    
    for bucket_name in buckets:
        try:
            response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
            print(f"\n{bucket_name}:")
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    size_mb = obj['Size'] / (1024 * 1024)
                    print(f"  {obj['Key']} ({size_mb:.2f} MB)")
            else:
                print("  (empty)")
                
        except ClientError as e:
            print(f"Error accessing {bucket_name}: {e}")

if __name__ == "__main__":
    print("Setting up Panthalassa S3 infrastructure...")
    create_panthalassa_buckets()
    print("\nBucket setup complete!")
    
    # List current contents
    list_bucket_contents()