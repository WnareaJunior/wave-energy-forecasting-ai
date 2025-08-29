# Panthalassa Buoy AI 🌊⚡

**AI-powered ocean wave forecasting for renewable energy**
Panthalassa Buoy AI will use advanced machine learning trained on NOAA & Copernicus information systems to predict wave activity and identify optimal deployment sites for the **200-ft Panthalassa energy-harvesting buoy**.

This system aims to provide **real-time forecasts, spatial heatmaps, and historical trend analysis** to maximize ocean energy capture potential.

---

## 🚧 Project Status

This repository is **in early development**.
The codebase will be built in stages following the AI roadmap below, with a future interactive dashboard to display predictions.

## Current Data Stack
**Historical (hindcast/reanalysis for training & seasonality)**

* Global waves + winds: Copernicus WAVERYS (hourly, ~0.2°), ECMWF ERA5 (hourly, 0.36°).

* Currents/SST: Copernicus/ERA5 ancillary layers.

* Bathymetry: GEBCO/EMODnet (for nearshore wave transformation risk & mooring feasibility).

**Live (monitoring & short-term prediction)**

* Global wave forecasts: Copernicus global waves (3-hourly), NOAA WW3 (if needed).

* Lightweight JSON for ops/API: Open-Meteo Marine; (optionally) StormGlass.

* Truth for validation: NDBC/IOOS buoys (coastal), altimetry SWH (Jason/Sentinel) gridded daily; any partner/own buoys.
---

## 🎯 Planned Features

### **1. Interactive Map Visualization**

* **Heatmap Display:** Color gradients showing a “usable wave activity index.”
* **Region Selection:** Zoom into specific ocean regions.
* **Tooltip Data:** Hover or click to see live and predicted values.
* **Multi-Buoy View:** Show buoy locations and statuses.

### **2. Prediction Dashboard**

* **Real-Time Readout:** Live wave activity metrics.
* **Forecast Chart:** Next 7 days of predicted activity.
* **Historical Trends:** Compare past and present performance.

### **3. Device Simulation & Comparison**

* **Buoy Type Selector:** Simulate different buoy designs.
* **Performance Comparison:** Side-by-side charts for selected types.

### **4. Data Filtering & Customization**

* **Date Range Picker:** Select timeframes for analysis.
* **Parameter Filters:** Adjust thresholds and sensitivity.
* **Location Search:** By coordinates or predefined sites.

### **5. User Interaction**

* **Save Locations:** Bookmark favorite areas.
* **Custom Alerts:** Threshold-based visual notifications.
* **Export Data:** CSV or image formats.

### **6. Responsive Modern UI**

* Mobile-friendly layout
* Dark/Light mode
* Smooth animations

---

## 📈 AI Development Roadmap

**Phase 1: Foundation (Weeks 1-4)**

* NOAA buoy data pipeline
* Baseline LSTM model
* Frequency decomposition (wind waves vs swells)
* Simple ensemble prediction

**Phase 2: Enhanced Accuracy (Weeks 5-8)**

* Variational Mode Decomposition (VMD)
* Multi-feature inputs (wind speed, direction, pressure)
* LSTM with attention / CNN-LSTM hybrids
* Uncertainty quantification

**Phase 3: Advanced Modeling (Weeks 9-12)**

* Physics-informed neural networks
* Multi-buoy spatial models (Graph Neural Networks)
* Extreme event handling
* Advanced ensembles

**Phase 4: Production (Weeks 13-16)**

* Real-time pipeline
* Multi-horizon forecasting
* Benchmarking against NOAA
* API + dashboard deployment

**Phase 5: Research Extensions (Weeks 17-20)**

* Wavelet & EMD decomposition
* Transformer architectures
* Satellite & climate data integration
* Publication-ready research paper

---

## 🗺 Data Sources (Planned)

* [NOAA National Data Buoy Center](https://www.ndbc.noaa.gov/)
* Satellite altimetry & oceanography datasets
* Regional wave monitoring stations

---

## 📊 Success Metrics

* Phase 1: **5–10% RMSE improvement** over baseline
* Phase 2: **15–20% improvement** + uncertainty intervals
* Phase 3: **25–30% improvement** + extreme event handling
* Phase 4: Production-ready with <2% downtime
* Phase 5: Publishable research contributions

---

## 🤝 Contributing

We welcome collaboration from those experienced in:

* Oceanography 🌊
* Machine Learning 🤖
* Renewable Energy ⚡
* Data Visualization 📊

---

## 📜 License

MIT.

---

## 📬 Contact

Email: **\[[wilsondev27@outlook.com](mailto:wilsondev27@outlook.com)]**
Website: **[www.panthalassa.com](http://www.panthalassa.com)**

