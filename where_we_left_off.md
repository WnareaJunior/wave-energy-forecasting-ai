# Where we left off — 2026-07-18

## Done this session

**Hindcast side is 100% complete.** All normalization code written, run, and validated:

- **Static layer** (`src/infrastructure/static_layer.py`) → `s3://panthalassa-ocean-processed/static/pnw_static_layer.nc`. Cell-mean depth, ocean_fraction, distance-to-coast, land mask on the 0.2° wave grid from ETOPO1. Cell-mean (not point) depth keeps mixed coastal cells; one known gap: the fjord cell at (50.0, −127.0) is wave-ocean but ETOPO-land — drop it via land_mask.
- **Training table** (`src/processing/build_training_table.py`) → `training-table/year=YYYY/part-0.parquet`, 44 years (1980–Apr 2023), 2.2 GB, ~77M rows = 598 ocean cells × 3-hourly. Waves (17 vars) + GLORYS currents/SST (2000–2019, NaN outside) + static + power_flux_kw (Hs²·Te, Te=VTM10). Validated: 38 kW/m mean power flux, exact cell/timestep counts.
- **Forecast table** (`src/processing/build_forecast_table.py`) → `forecast-table/year=YYYY/member=MMM/part-0.parquet`. GEFS forecast variables regridded to the wave grid, joined to hindcast truth (obs_hs, obs_te, obs_power_kw) per lead hour. 2000 + 2001 c00 built (~65M pairs). Validated: skill decay corr 0.98→0.44 over 3h→14d leads; Wednesday inits reach 35-day leads.

**Ensemble backfill launched.** GEFSv12 reforecast has 5 daily members (c00+p01–p04; p05–p10 Wednesdays only — deliberately skipped as inhomogeneous). Five t3.medium instances now running in parallel, one per member (`launch_noaa_ec2.sh <member>`); each does ~7,300 member-days at ~15/hr → **all finish ~2026-08-07**. Cost ~$80 total.

**Daily sweep automation live and verified.** EventBridge schedule `panthalassa-forecast-sweep` (12:00 UTC) launches a self-terminating t3.xlarge that runs `build_forecast_table.py --sweep`: builds any year/member whose raw archive matches the source bucket's day count, uploads its run log to `s3://panthalassa-ocean-raw-data/scripts/logs/`, and deletes its own schedule when all 100 year/members are built. IaC: `setup_sweep_schedule.sh` + `ec2_user_data_sweep.sh` (idempotent). Verified end-to-end with a manual template launch.

**README rewritten** to current reality, with an explicit In-Development list.

## Decisions made

- Model = **forecast correction**: GEFS features → Copernicus truth, XGBoost predicts physical quantities (Hs, Te, power flux); reward function ("power − fatigue − risk − maintenance") stays a separate tunable layer.
- End goal: scheduled retraining on recent data, predictions published to the site (`index.html`/`dashboard/`).
- Open modeling decisions (discussed, not yet fixed): quantile vs point outputs (leaning quantile q10/50/90), lead-time-as-feature vs per-lead models (leaning feature), 2000–2019-only vs all-years training (leaning 2000–2019), predict Hs/Te vs log-power directly (try both).

## Next steps (nothing blocks starting now)

1. **Baselines**: persistence, site×month climatology, raw uncorrected GEFS — on the existing 65M c00 pairs.
2. **Evaluation harness**: temporal walk-forward splits (hold out 2018–2019 final test), per-lead and per-site skill; spatial-holdout ablation.
3. **First XGBoost correction model** on c00 pairs.
4. **Ensemble features** (member mean/spread) — blocked on p01–p04 backfill (~Aug 7).
5. Reward layer, retrain loop, dashboard — after model skill is established.

## Things running unattended

- 5 × NOAA backfill EC2 instances (self-terminating, S3-resume, safe to relaunch via `launch_noaa_ec2.sh [member]`).
- Daily forecast-table sweep (self-dismantling when complete; logs in `scripts/logs/`).
- Gotcha: backfill instances pulled their downloader copy at boot — downloader code changes need instance relaunch to take effect.
- AWS root sessions expire fast; reauth with `aws login`.

All work on branch `copernicus-backfill-v2`, pushed through commit `916cf5c`.
