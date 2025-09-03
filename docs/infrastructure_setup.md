# Panthalassa Ocean Forecasting Infrastructure

## AWS Architecture Overview

### S3 Buckets
- `panthalassa-ocean-raw-data` - Raw Copernicus and NOAA data
- `panthalassa-ocean-processed` - Cleaned and filtered datasets  
- `panthalassa-ocean-results` - Model outputs and analysis results

### EC2 Instances

**Copernicus Data Downloader**
- Instance Type: t3.medium
- Region: us-east-2
- Purpose: Download 30GB Copernicus Marine dataset
- Dataset: cmems_mod_glo_wav_my_0.2deg_PT3H-i
- Geographic bounds: -130°W to -124°W, 46°N to 50.5°N
- Status: Active download (~1 day completion)

**NOAA Data Processor** 
- Instance Type: t3.medium
- Region: us-east-2
- Purpose: Process NOAA Wave Ensemble Reforecast data
- Source: noaa-nws-gefswaves-reforecast-pds bucket
- Coverage: 20-year historical reforecast data
- Status: Active processing

### ECS Fargate (Attempted)
- Cluster: panthalassa-ocean-processing
- Task Definition: panthalassa-copernicus-task
- Container: copernicusmarine/copernicusmarine:latest
- Environment Variables: COPERNICUSMARINE_SERVICE_USERNAME, COPERNICUSMARINE_SERVICE_PASSWORD
- Status: Working but moved to EC2 for better performance

## Data Pipeline Flow
1. Raw data sources → S3 raw-data bucket
2. EC2 processing → S3 processed bucket  
3. Analysis/modeling → S3 results bucket

## Authentication
- Copernicus: CLI credentials stored on EC2
- AWS: IAM roles with S3 full access
- NOAA: Public bucket, no authentication required

## Technical Decisions Made
- **EC2 over Lambda**: 30GB+ datasets exceed Lambda limitations
- **EC2 over ECS**: Simpler deployment, faster iteration for data downloads
- **Parallel processing**: Separate instances for Copernicus vs NOAA data
- **S3 as central storage**: Enables multiple processing approaches

## Next Steps
- Implement wave power flux calculations
- Set up automated data cleaning pipeline
- Deploy ML model training infrastructure