# Panthalassa Buoy AI 🌊⚡

**AI-powered ocean wave forecasting for renewable energy**
Panthalassa Buoy AI uses advanced machine learning trained on Copernicus Marine and NOAA datasets to predict wave activity and identify optimal deployment sites for the **200-ft Panthalassa energy-harvesting buoy**.

This system provides **wave power flux analysis, spatial heatmaps, and forecasting models** to maximize ocean energy capture potential in the Pacific Northwest.

---

## 🚧 Project Status

This repository is **in active development** with cloud infrastructure deployed and data acquisition in progress.

### ✅ Current Progress

**Infrastructure & Data Pipeline**
- Deployed **AWS cloud infrastructure** (EC2, S3, ECS) for large-scale ocean data processing
- Configured **dual data acquisition pipeline**: Copernicus Marine observations + NOAA 20-year reforecast
- Implemented **production S3 data architecture** with raw/processed/results bucket structure
- Active **parallel data downloads**: ~30GB Copernicus + ~30GB NOAA datasets

**Analysis Framework**
- Built **wave power flux calculation engine** with validated oceanographic formulas
- Developed **spatial filtering** for Pacific Northwest region (-130°W to -124°W, 46°N to 50.5°N)
- Created **data validation schemas** for both Copernicus and NOAA datasets
- Implemented **quality control pipelines** and consistency metrics

**Processing Architecture**
- **EC2-based processing cluster**: Separate instances for Copernicus vs NOAA data streams  
- **Container orchestration** attempted (ECS Fargate) - migrated to EC2 for performance
- **Automated data cleaning** and temporal aggregation functions
- **Monthly/seasonal statistics** calculation framework

**Forecasting Framework**
- **Pluggable model interface** (`src/models/base.py`) — baselines, ridge and
  LightGBM implemented; Transformer and PINN rungs plug in at the same point
- **NDBC buoy pipeline** for Washington-coast stations (46041, 46087, 46029)
- **Leakage-safe temporal splits** with enforced gaps and rolling-origin backtesting
- **Per-horizon verification**: RMSE, bias, storm-conditioned error, and skill
  against a configurable reference
- 163 tests, 85% coverage

### 🔜 Next Steps
- Run the [buoy pilot](docs/pilot_experiment.md) on real NDBC data and publish
  the per-horizon skill table
- Build the paired GEFSv12 forecast/observation dataset for NWP postprocessing
- Complete the gridded Copernicus + NOAA pipeline
- Add the deep-learning rungs once LightGBM has set the bar

See [`docs/forecasting_architecture.md`](docs/forecasting_architecture.md) for
the full plan and [`docs/pilot_experiment.md`](docs/pilot_experiment.md) for how
to run the pilot.

---

## 🏗️ Technical Architecture

### **Cloud Infrastructure**
- **AWS Region**: us-east-2 (Ohio)
- **Data Storage**: 3-tier S3 architecture (raw → processed → results)
- **Compute**: EC2 instances for data processing, ECS for containerized workloads
- **Authentication**: Copernicus Marine CLI integration, AWS IAM roles

### **Data Sources**
**Primary Datasets (In Acquisition)**
- **Copernicus Marine**: `cmems_mod_glo_wav_my_0.2deg_PT3H-i` - Satellite-observed wave data (2020-2023, 3-hourly, 0.2° resolution)
- **NOAA Wave Ensemble Reforecast**: `noaa-nws-gefswaves-reforecast-pds` - 20-year WAVEWATCH III model hindcast (3-hourly, 0.25° resolution)

**Data Characteristics**
- **Geographic Focus**: Pacific Northwest offshore (Washington/BC high wave energy zone)
- **Combined Coverage**: Observational truth (Copernicus) + long-term climatology (NOAA)  
- **Processing Scale**: 60GB+ raw data → processed wave power analysis

### **Processing Pipeline**
```
Raw Data Sources → S3 Raw → EC2 Processing → S3 Processed → ML Training → S3 Results
```

---

## 🔬 Scientific Approach

### **Wave Power Physics**
- **Formula**: P = (ρ × g² / 64π) × H² × T
- **Variables**: Significant wave height (H), peak period (T), seawater density (ρ)
- **Output**: Wave power flux in W/m for energy resource assessment

### **Validation Strategy**
- **Copernicus Data**: Satellite observations provide ground truth for model validation
- **NOAA Reforecast**: 20-year model climatology enables seasonal/interannual analysis  
- **Cross-Validation**: Temporal splits preserve time series structure for forecasting

### **Target Performance Metrics**
- **Phase 1**: 5-10% RMSE improvement over persistence baseline
- **Phase 2**: 15-20% improvement with ensemble methods
- **Phase 3**: 25-30% improvement with advanced ML architectures

---

## 📁 Repository Structure

```
├── src/
│   ├── infrastructure/     # AWS deployment and configuration
│   ├── processing/         # Wave power calculations and data cleaning
│   └── data/              # Schemas and configuration files
├── notebooks/             # Jupyter analysis and prototyping
├── docs/                  # Technical documentation and progress logs
├── models/                # ML model implementations
└── tests/                 # Unit tests and validation
```

---

## 🎯 Development Roadmap

**Phase 1: Data Foundation (Weeks 1-2) - IN PROGRESS**
- ✅ Cloud infrastructure deployment
- 🔄 Large-scale data acquisition (Copernicus + NOAA)  
- 🔄 Wave power flux analysis pipeline
- 🔜 Baseline forecasting models

**Phase 2: ML Development (Weeks 3-6)**
- Advanced feature engineering (lags, seasonality, spatial patterns)
- Random Forest and gradient boosting models  
- LSTM neural networks for time series forecasting
- Model validation and hyperparameter tuning

**Phase 3: Production System (Weeks 7-10)**
- Real-time forecasting API development
- Interactive dashboard with spatial visualizations
- Model deployment and monitoring infrastructure
- Performance optimization and scaling

**Phase 4: Advanced Analytics (Weeks 11-16)**
- Physics-informed neural networks
- Ensemble forecasting with uncertainty quantification
- Extreme event detection and handling
- Multi-horizon prediction capabilities

---

## 🛠️ Technical Stack

**Infrastructure**: AWS (EC2, S3, ECS), Docker
**Data Processing**: Python, xarray, pandas, numpy  
**Machine Learning**: scikit-learn, TensorFlow/PyTorch
**Oceanographic Data**: Copernicus Marine Toolbox, NOAA APIs
**Visualization**: matplotlib, plotly, folium

---

## 📈 Current Metrics

- **Data Pipeline**: 60GB+ multi-source ocean datasets
- **Geographic Coverage**: Pacific Northwest energy resource zone
- **Temporal Scope**: 3+ years observations + 20-year climatology  
- **Processing Infrastructure**: Multi-instance AWS deployment
- **Development Status**: Data acquisition phase, analysis framework complete

---

## 🤝 Contributing

Technical collaboration welcome in:
- Oceanographic modeling and validation
- Machine learning for time series forecasting  
- Renewable energy resource assessment
- Cloud infrastructure and scalability

---

## 📜 License

MIT License

---

## 📬 Contact

**Developer**: Wilson Narea  
**Email**: [wilsondev27@outlook.com](mailto:wilsondev27@outlook.com)  
**Project**: Ocean Energy Forecasting Research
