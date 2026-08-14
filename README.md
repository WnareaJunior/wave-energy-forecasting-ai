# Panthalassa Buoy AI 🌊⚡

**AI-powered ocean wave forecasting for renewable energy**
Panthalassa Buoy AI trains machine learning models on Copernicus Marine and NOAA datasets to predict wave activity and identify optimal deployment sites for the **200-ft Panthalassa energy-harvesting buoy** in the Pacific Northwest.

The system pairs NOAA's 20-year GEFS-Wave ensemble reforecast with the Copernicus wave hindcast as ground truth, so a model can learn to **correct and downscale forecasts site-by-site** — and eventually retrain on recent data on a schedule, publishing updated predictions to the project dashboard.

---

## 🗺️ Study Region

Washington / British Columbia offshore: **130–124°W, 46–50.5°N**, one of the highest wave-energy zones in North America (~38 kW/m mean power flux). All sources are aligned to the Copernicus 0.2° grid, 3-hourly, 598 ocean cells.

---

## 📊 Data Pipeline (complete)

```
NOAA GEFS reforecast ─┐                        ┌─ training-table/   (hindcast, 44 yrs)
Copernicus hindcast  ─┼─ S3 raw ─ normalize ───┤
GLORYS currents/SST  ─┤                        └─ forecast-table/   (forecast⇄truth pairs)
ETOPO1 bathymetry   ──┘
```

| Source | Content | Coverage | Status |
|---|---|---|---|
| Copernicus `cmems_..._wav_my_0.2deg` | 17 wave variables (Hs, Te, Tp, direction, swell/wind-sea partitions) | 1980 – Apr 2023, 3-hourly | ✅ complete |
| GLORYS12 | surface currents (uo, vo), SST | 2000 – 2019, daily | ✅ complete |
| ETOPO1 static layer | depth, ocean fraction, distance-to-coast | static | ✅ complete |
| NOAA GEFSv12 wave reforecast | 10 forecast variables, leads 0–16 d (35 d Wednesdays) | 2000 – 2019, 5 members | 🔄 backfilling (5 parallel EC2 instances, ETA ~Aug 2026) |

**Analysis-ready tables** (Parquet, `s3://panthalassa-ocean-processed/`):
- `training-table/year=YYYY/` — hindcast rows (time × cell): waves + currents + SST + static features + derived power flux (P = ρg²/64π · Hs² · Te, using Te = VTM10). ~77M rows, 2.2 GB.
- `forecast-table/year=YYYY/member=MMM/` — forecast-correction pairs: normalized GEFS forecast variables joined to hindcast truth at valid time, per lead hour. ~32M rows/year/member; fills automatically as the backfill lands.

A daily **EventBridge-scheduled sweep** (12:00 UTC) launches a self-terminating EC2 instance that builds any year/member whose raw backfill completed since the last run, and deletes its own schedule when the table is complete.

### Key scripts

| Script | Purpose |
|---|---|
| `src/infrastructure/copernicus_downloader.py` | Wave hindcast → S3, monthly chunks, S3-derived resume |
| `src/infrastructure/glorys_downloader.py` | GLORYS currents/SST → S3, yearly |
| `src/infrastructure/noaa_downloader.py` | GEFS reforecast → S3 via .idx byte-range subsetting, per-member |
| `src/infrastructure/static_layer.py` | ETOPO1 → depth/distance-to-coast on the wave grid |
| `src/infrastructure/launch_*.sh`, `setup_sweep_schedule.sh` | Self-terminating EC2 backfills + scheduled sweep (IaC) |
| `src/processing/build_training_table.py` | Hindcast sources → training table |
| `src/processing/build_forecast_table.py` | GEFS + truth → forecast table (`--sweep` for the scheduled job) |
| `src/processing/wave_power_flux.py` | Power-flux physics and site-ranking utilities |

---

## 🚧 In Development

**Modeling (next up)**
- **Baselines**: persistence, site×month climatology, raw uncorrected GEFS — the skill reference every model must beat
- **Evaluation harness**: temporal walk-forward splits (spatial-holdout ablation), per-lead-time and per-site skill metrics
- **XGBoost forecast-correction model**: GEFS + static + ocean-state features → corrected Hs / Te / power flux; quantile outputs (q10/q50/q90) for uncertainty
- **Ensemble features**: mean/spread across the 5 members per init/lead/cell (blocked on p01–p04 backfill, ETA ~Aug 2026)

**Product**
- **Reward layer**: tunable "power − fatigue − risk − maintenance" scoring on top of the model's physical predictions (fatigue from swell/wind-sea partition + direction spread; risk/cost from depth + distance-to-coast)
- **Operational loop**: scheduled retraining on recent data; same normalization pipeline pointed at the operational GEFS-Wave feed
- **Live dashboard**: predictions published to S3 and rendered on the project site (`index.html` / `dashboard/`)

---

## 🏗️ Infrastructure

- **AWS us-east-1** — S3 (raw → processed → results), self-terminating t3 EC2 workers, EventBridge Scheduler, IAM instance roles (no long-lived keys)
- All infrastructure is scripted in `src/infrastructure/` (boto3 + AWS CLI); every backfill derives resume state from S3, so any job can be killed and relaunched idempotently
- Stack: Python, xarray, cfgrib, pandas, PyArrow; XGBoost for modeling

### Local mode — the devbox instead of AWS

Set `WAVE_S3_ENDPOINT` (see `.env.example`) and every *destination* S3 write
goes to MinIO on the home server, where the three `panthalassa-ocean-*`
buckets already exist — no AWS bill for experiments. Source reads (the public
NOAA/GEFS buckets) always hit real AWS regardless. The devbox also provides:

- **GPU training** — RTX 4060 Ti via JupyterLab (`http://devbox:8890`) instead of EC2
- **Experiment tracking** — MLflow at `http://devbox:5002`; scripts pick it up
  through `MLFLOW_TRACKING_URI` (`mlflow` is in `requirements.txt`)
- Processing jobs can run on the devbox directly (`ssh devbox`, 18 cores / 48 GB)

---

## 📜 License

MIT License

---

## 📬 Contact

**Developer**: Wilson Narea
**Email**: [wilsondev27@outlook.com](mailto:wilsondev27@outlook.com)
**Project**: Ocean Energy Forecasting Research
