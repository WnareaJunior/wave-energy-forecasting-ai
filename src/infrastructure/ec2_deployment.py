import boto3
from datetime import datetime

class PanthalassaEC2Manager:
    """
    Manages EC2 instances for ocean data processing
    """
    
    def __init__(self, region='us-east-2'):
        self.ec2 = boto3.client('ec2', region_name=region)
        self.region = region
    
    def get_instance_configs(self):
        """
        Define EC2 instance configurations used in the project
        """
        return {
            'copernicus_downloader': {
                'instance_type': 't3.medium',
                'ami_id': 'ami-0c02fb55956c7d316',  # Amazon Linux 2
                'key_name': 'your-key-pair',
                'security_groups': ['default'],
                'user_data': self._get_copernicus_user_data(),
                'tags': [
                    {'Key': 'Name', 'Value': 'panthalassa-copernicus-downloader'},
                    {'Key': 'Project', 'Value': 'Panthalassa Ocean Forecasting'},
                    {'Key': 'Purpose', 'Value': 'Copernicus Marine data download'}
                ]
            },
            'noaa_processor': {
                'instance_type': 't3.medium', 
                'ami_id': 'ami-0c02fb55956c7d316',  # Amazon Linux 2
                'key_name': 'your-key-pair',
                'security_groups': ['default'],
                'user_data': self._get_noaa_user_data(),
                'tags': [
                    {'Key': 'Name', 'Value': 'panthalassa-noaa-processor'},
                    {'Key': 'Project', 'Value': 'Panthalassa Ocean Forecasting'},
                    {'Key': 'Purpose', 'Value': 'NOAA Wave Reforecast processing'}
                ]
            }
        }
    
    def _get_copernicus_user_data(self):
        """
        User data script for Copernicus downloader instance
        """
        return """#!/bin/bash
yum update -y
yum install python3 python3-pip -y

# Create working directory
mkdir -p /home/ec2-user/copernicus-processing
cd /home/ec2-user/copernicus-processing

# Install required packages
pip3 install copernicus-marine-client boto3 xarray netcdf4

# Set up Copernicus credentials (would need to be added securely)
echo "# Copernicus Marine credentials configured via CLI login" > /home/ec2-user/credentials_setup.txt

# Create download script template
cat > /home/ec2-user/copernicus-processing/download_script.py << 'EOF'
# Copernicus data download script
# Dataset: cmems_mod_glo_wav_my_0.2deg_PT3H-i  
# Region: Pacific Northwest (-130°W to -124°W, 46°N to 50.5°N)
# Purpose: Ocean wave data for energy forecasting
EOF

chown -R ec2-user:ec2-user /home/ec2-user/copernicus-processing
"""

    def _get_noaa_user_data(self):
        """
        User data script for NOAA processor instance  
        """
        return """#!/bin/bash
yum update -y
yum install python3 python3-pip -y

# Create working directory
mkdir -p /home/ec2-user/noaa-processing
cd /home/ec2-user/noaa-processing

# Install required packages
pip3 install boto3 pandas xarray netcdf4 numpy

# Create NOAA processing script template
cat > /home/ec2-user/noaa-processing/process_noaa.py << 'EOF'
# NOAA Wave Ensemble Reforecast processing
# Source: noaa-nws-gefswaves-reforecast-pds
# Coverage: 20-year historical data
# Purpose: Model-based wave data for forecasting
EOF

chown -R ec2-user:ec2-user /home/ec2-user/noaa-processing
"""

    def document_current_deployment(self):
        """
        Document the current EC2 deployment status
        """
        deployment_status = {
            'deployment_date': datetime.now().isoformat(),
            'region': self.region,
            'instances': {
                'copernicus_downloader': {
                    'status': 'running',
                    'purpose': 'Download 30GB Copernicus Marine dataset',
                    'dataset': 'cmems_mod_glo_wav_my_0.2deg_PT3H-i',
                    'estimated_completion': '~24 hours',
                    'geographic_bounds': {
                        'longitude_min': -130.0,
                        'longitude_max': -124.0,
                        'latitude_min': 46.0, 
                        'latitude_max': 50.5
                    }
                },
                'noaa_processor': {
                    'status': 'running',
                    'purpose': 'Process NOAA Wave Ensemble Reforecast', 
                    'source_bucket': 'noaa-nws-gefswaves-reforecast-pds',
                    'coverage': '20-year historical reforecast data',
                    'processing_approach': 'S3-to-S3 transfer and filtering'
                }
            },
            'data_pipeline': {
                'raw_data_bucket': 'panthalassa-ocean-raw-data',
                'processed_data_bucket': 'panthalassa-ocean-processed',
                'results_bucket': 'panthalassa-ocean-results'
            },
            'authentication': {
                'copernicus': 'CLI credentials on EC2 instance',
                'aws': 'IAM roles with S3 full access',
                'noaa': 'Public bucket access'
            }
        }
        
        return deployment_status

if __name__ == "__main__":
    manager = PanthalassaEC2Manager()
    
    # Document current deployment
    status = manager.document_current_deployment()
    
    print("Panthalassa EC2 Deployment Status:")
    print(f"Region: {status['region']}")
    print(f"Deployment Date: {status['deployment_date']}")
    
    for instance_name, details in status['instances'].items():
        print(f"\n{instance_name}:")
        print(f"  Status: {details['status']}")
        print(f"  Purpose: {details['purpose']}")