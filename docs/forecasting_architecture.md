# Forecasting Model Architecture

How to get from the current repository to a framework where persistence, gradient
boosting, LSTM, Transformer and PINN forecasters are interchangeable behind one
interface and compared on one evaluation protocol.

---

## 1. Where the repo actually is

### What exists and works

| Area | Files | Assessment |
|---|---|---|
| Wave physics | `src/processing/wave_power_flux.py` (263 lines) | Only real library code. Power flux, monthly aggregation, consistency metrics, seasonal ratio, revenue map, site ranking. Pure functions, importable, reusable. |
| Data acquisition | `src/infrastructure/copernicus_downloader.py`, `noaa_downloader.py` | Top-level scripts, not modules. Checkpoint/resume logic is sound. Config is hardcoded. |
| AWS provisioning | `src/infrastructure/s3_setup.py`, `ec2_deployment.py` | Bucket layout + instance definitions. `ec2_deployment.py` is largely a documentation object, not a provisioner. |
| Exploration | `notebooks/001_visualize_small_data.ipynb`, `001_wave_power_flux_analysis.ipynb.ipynb` | Load → QC → clean → power flux → siting → revenue. Where the real analysis lives. |
| Results | `images/*.png` (10 figures), `index.html` | Site rankings, revenue surface, best/worst comparison. Presentation layer. |
| Docs | `README.md`, `docs/infrastructure_setup.md`, `deployment_guide.md` | README describes a roadmap well beyond what is committed. `deployment_guide.md` is 0 bytes. |

### What is empty

`models/` (README 0 bytes) · `dashboard/` (README 0 bytes) · `tests/test_models.py`
(0 bytes) · `tests/test_data_pipeline.py` (0 bytes) · `src/data/` (referenced in the
README's structure diagram, does not exist).

**There is no forecasting model in the repository.** The only prediction code is
inside notebook cell 2 of `001_wave_power_flux_analysis`: a `LinearRegression` on a
single lag of *monthly-mean* power flux at one grid cell. Over 2020-01 → 2023-04 that
is ~40 samples with a 70/30 split — roughly 28 training points. It is a demo of the
idea, not a baseline anyone can be measured against.

### Blockers that will bite immediately

1. **`requirements.txt` is UTF-16-LE with a BOM** (`file` reports
   `Unicode text, UTF-16, little-endian`). `pip install -r requirements.txt` fails
   with a `UnicodeDecodeError` on any non-Windows host — which means `DockerFile`
   cannot build today. Re-encode to UTF-8.
2. **No deep-learning dependency.** No `torch`, no `tensorflow`, no `lightgbm`. The
   pinned set is xarray/dask/sklearn/matplotlib only. Every model past Random Forest
   needs a dependency decision first.
3. **CI is broken.** `.github/workflows/pr-build.yml` runs `npm install && npm run
   build`; there is no `package.json` in the repo. Every PR build fails. There is no
   Python CI at all.
4. **Three mutually inconsistent study regions.**
   - README / notebooks / `ec2_deployment.py`: **-130 to -124 °E, 46 to 50.5 °N** (Pacific Northwest)
   - `noaa_downloader.py`: **-130 to -120 °E, 25 to 30 °N** (off Baja California)
   - `copernicus_downloader.py`: **+124.5 to +144.6 °E, 16.7 to 48.2 °N** (Western Pacific / Japan)

   Whatever the NOAA and Copernicus downloaders produced, it does not overlap the
   region the analysis was run on. This must be resolved before any dataset build.
5. **Two power-flux conventions.** `wave_power_flux.py` uses
   `ρg²/64π ≈ 490` and returns **W/m** with `VTPK` (peak period); the notebook uses
   `0.49 · H² · T` returning **kW/m** with `VTM02`. Same physics, different units and
   different period variable.

   Neither period variable is right. The formula is defined on the **energy period**
   `Te = m₋₁/m₀`, which is exactly what Copernicus publishes as **`VTM10`** — a
   variable both code paths ignore. `VTM02` is the zero-crossing period
   (`Te ≈ 1.2·VTM02`) and `VTPK` is the peak period (`Te ≈ 0.86·VTPK`). Using `VTPK`
   raw overstates flux by ~16 %; using `VTM02` raw understates it by ~17 %. Resolved
   in Phase 0: `VTM10` preferred, documented conversion factors otherwise.
6. **No reproducible data path.** Notebooks read `WAVE_DATA_DIR` from `.env`; `data/`,
   `*.nc` and `*.zarr` are gitignored. Nobody but the author can re-run anything.
7. **No GPU.** `docs/infrastructure_setup.md` specifies `t3.large`. Transformers and
   PINNs need `g4dn`/`g5` or SageMaker training jobs.

### Smaller correctness issues worth fixing when the code is touched

- `noaa_downloader.py`: `list_objects_v2` without a paginator caps at 1000 keys, so a
  full year is silently truncated. `key.split('/')[4]` also assumes a path depth the
  `GEFSv12/reforecast/2019/` prefix may not have.
- `wave_power_flux.py::calculate_consistency_metrics` documents a `seasonal_ratio`
  key it never returns, and returns `mean_power`/`median_power` it never documents.
- Notebook `pd.Grouper(freq="M")` is deprecated in pandas ≥ 2.2 (`"ME"`).
- `df_rank` is filtered at `Pflux > 30` but the "worst site" is then taken as
  `df_rank.iloc[-1]` — the worst site is the worst *above threshold*, not the worst
  site. The plots labelled "best vs worst" mean something narrower than they say.

---

## 2. Define the task before writing a model

Nothing else can be specified until these five choices are pinned. The notebook
implicitly answers them in a way that makes the problem trivially easy, which is why
lag-1 linear regression appeared to work.

| Decision | Recommendation | Why |
|---|---|---|
| **Target** | Predict `VHM0` (Hs) and `VTM02` (Tm02) jointly; derive Pflux from the predictions | Pflux ∝ H²T is heavy-tailed and hard to regress directly. Predicting the two physical drivers keeps errors interpretable and lets the physics layer stay exact. |
| **Resolution** | Native 3-hourly, not monthly means | Monthly aggregation destroys the forecasting problem — ~40 samples and near-perfect autocorrelation. 3-hourly gives ~9,700 timesteps over 2020-01 → 2023-04. |
| **Horizon** | Direct multi-horizon, +3 h to +72 h (24 steps) | Matches operational WEC dispatch. Report per-horizon skill, never one aggregate number. |
| **Geometry** | Phase 1 per-site (top-N candidate cells); Phase 2 full field | Point models are cheap and let you rank architectures fast. Spatial models (and the PINN) come after. |
| **Output** | Probabilistic — quantiles or a parametric distribution, not a point | Siting and dispatch decisions need uncertainty. GEFSv12 is an ensemble product, so the labels for this already exist. |

**Data volume check.** PNW box at 0.2° = 31 lon × 23 lat = 713 cells, ~60 % ocean per
the notebook's mask ⇒ ~430 ocean cells × ~9,700 timesteps ≈ **4.2 M point-samples**.
Ample for LSTMs and point Transformers. Not ample for a field-level Transformer,
which sees only ~9,700 field snapshots — that is what the **NOAA GEFSv12 20-year
reforecast** is for, and it is the reason to fix the NOAA downloader's bounds.

---

## 3. Target repository layout

```
src/
  data/
    catalog.py          # dataset registry: id -> S3/zarr URI, variables, bounds
    sources/
      copernicus.py     # refactor of copernicus_downloader.py into a function
      noaa_gefs.py      # refactor of noaa_downloader.py, with pagination
      era5_wind.py      # NEW - u10/v10, required by the PINN source term
    build_dataset.py    # raw netCDF/GRIB -> analysis-ready zarr (chunked on time)
    windows.py          # (context, horizon) sliding-window sample generation
    splits.py           # temporal splits; the single place leakage can be prevented
  features/
    physics.py          # Pflux, group velocity, dispersion, steepness  <- absorbs wave_power_flux.py
    calendar.py         # sin/cos harmonics of day-of-year and hour
    lags.py             # lag / rolling-window construction
    scaling.py          # fit on train only, serialize alongside the model
  models/
    base.py             # Forecaster protocol - THE key file
    registry.py         # "patchtst" -> PatchTSTForecaster
    baselines.py        # persistence, climatology, seasonal-naive, lag-k ridge
    trees.py            # LightGBM, one model per horizon
    recurrent.py        # LSTM / GRU / TCN encoder-decoder
    transformer.py      # PatchTST / iTransformer
    pinn.py             # physics-residual model
    losses.py           # pinball, CRPS, physics-residual penalties
  eval/
    metrics.py          # RMSE, MAE, bias, skill score vs persistence, CRPS, PIT
    backtest.py         # rolling-origin evaluation
    report.py           # per-horizon tables + figures
  training/
    train.py            # single entrypoint: --model X --config Y
    tune.py
configs/
  data/pnw_copernicus.yaml
  model/{persistence,ridge,lgbm,lstm,patchtst,pinn}.yaml
tests/
  test_physics.py       # analytic checks on the flux/dispersion formulas
  test_windows.py       # no-leakage assertions on split boundaries
  test_models.py        # every registered model round-trips fit/predict/save/load
```

### The contract that makes models swappable

Everything below hangs off this. Write it first.

```python
# src/models/base.py
class Forecaster(Protocol):
    name: str
    def fit(self, train: WaveDataset, val: WaveDataset | None) -> None: ...
    def predict(self, batch: WaveDataset) -> Forecast: ...   # (n_samples, n_horizons, n_targets)
    def predict_quantiles(self, batch, q: Sequence[float]) -> Forecast: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Forecaster": ...
```

`WaveDataset` is a fixed tensor contract shared by every model:

```
x_hist   (batch, context_len, n_features)   past Hs, Tm02, Tpk, direction sin/cos, wind
x_static (batch, n_static)                  lat, lon, depth, distance-to-coast
x_future (batch, horizon, n_known)          calendar harmonics, NWP wind if available
y        (batch, horizon, n_targets)        Hs, Tm02
```

Once this holds, adding a model is one file and one registry line. No model
touches data loading, splitting, scaling, or metrics.

---

## 4. Model ladder

Build in this order. Each rung is only worth climbing if it beats the one below it on
the same protocol.

### Rung 0 — Baselines (mandatory)

Persistence, monthly climatology, seasonal-naive, and lag-k ridge. **Persistence is
brutally strong at short horizons** for Hs (r > 0.95 at +3 h) and weakens with lead
time — so the headline metric must be *skill score versus persistence, per horizon*.
The README's "5-10 % RMSE improvement over persistence" target is meaningless
without saying at which horizon.

### Rung 1 — Gradient boosting (LightGBM)

One model per horizon on lag/rolling features. Cheap, CPU-only, and typically the
hardest thing to beat on tabular point forecasting. This is the real bar. Quantile
objective gives probabilistic output for free.

### Rung 2 — Recurrent / convolutional sequence models

LSTM or GRU encoder–decoder, plus a dilated TCN as a cheaper alternative. Context
~168 steps (21 days at 3 h), horizon 24. First rung that needs `torch` and benefits
from a GPU. Trains in minutes on 4 M point-samples with a `g4dn.xlarge`.

### Rung 3 — Transformers

- **PatchTST** — patch the time series, channel-independent attention. Strong default for long-context multivariate forecasting, and robust when training data is limited.
- **iTransformer** — attention across variables rather than time. Good when Hs/Tm02/Tpk/wind interactions matter more than long-range temporal structure.
- **Informer / Autoformer** — only if context length grows past ~1000 steps; the sparse-attention machinery is not worth it below that.

Start with PatchTST. Its channel-independence is a good match for the ~430 ocean
cells: one model, cells as the batch dimension, static coordinates as covariates.

**Precondition:** this rung is where the 3.3-year Copernicus record becomes the
binding constraint. Get the GEFSv12 20-year reforecast landed on the correct bounds
first, or the Transformer will overfit and lose to LightGBM.

### Rung 4 — Physics-Informed Neural Network

The physics to inform with is the **wave action / energy balance equation** that
WAVEWATCH III and SWAN solve:

```
∂E/∂t + ∇·(c_g E) = S_in + S_nl + S_ds
```

Full spectral form is out of scope. Three tractable levels, in increasing order of
ambition:

**Level A — hard physical constraints (do this first, it is nearly free).**
Bolt onto *any* of the models above:
- `Hs ≥ 0`, `Tm02 > 0` via output activation, not a penalty
- deep-water dispersion: `c_g = gT/(4π)`, `ω² = gk`
- steepness limit `Hs / L ≤ 1/7` as a penalty on violations
- consistency between the Hs head, the Tm02 head, and derived Pflux

**Level B — soft residual on bulk energy propagation.** Model `E ∝ Hs²`, compute
`c_g` from predicted `Tm02`, and penalise the residual of
`∂E/∂t + ∇·(c_g E) − S` on the 2D grid, with `S` a small learned network fed 10 m
wind. Autograd for the time derivative, finite differences on the lat/lon grid (which
is regular at 0.2°, so this is straightforward).

**Level C — collocation PINN.** Coordinate network `(x, y, t) → (Hs, Tm02)` trained
on a data loss at observation points plus a residual loss at sampled collocation
points. Elegant, and the honest expectation is that it *loses to PatchTST on RMSE*
while producing more physically coherent fields and extrapolating better into
data-sparse regions. Frame it as a physical-consistency and spatial-gap-filling tool,
not an accuracy play.

**Hard dependency:** Levels B and C need **10 m wind (u10, v10)** for the `S_in`
source term. The current Copernicus wave product does not contain it. Add ERA5 or a
Copernicus wind product to the acquisition pipeline — this is a prerequisite, not a
detail.

---

## 5. Evaluation protocol

Fix this **before** training anything, or the model comparison is worthless.

- **Splits.** Train 2020-01 → 2021-12, validate 2022, test 2023-01 → 2023-04.
  Caveat: the test window is 4 months of winter/spring only, so it is
  seasonally biased toward high-energy conditions. Prefer **rolling-origin
  backtesting** over a single holdout, and once GEFSv12 lands, block-CV by year.
- **Gap between splits.** Insert a gap of at least the context length so no test
  sample's history overlaps training data.
- **Scaling.** Fit normalisation on train only, serialize with the model. This is the
  most common leakage route in this class of problem.
- **Metrics, always per horizon:** RMSE, MAE, bias, skill score vs persistence, plus
  RMSE conditioned on the top decile of Hs (storm performance is the whole point for
  a WEC), CRPS and PIT-histogram calibration for probabilistic models.
- **Downstream metric.** Error in predicted revenue / capacity factor, computed
  through the existing `wave_power_flux.py` functions — the metric the project
  actually cares about.
- **Seeds.** Three seeds minimum for every neural model. Differences smaller than
  seed variance are not differences.

---

## 6. Sequenced work plan

**Phase 0 — Unblock — DONE**
1. ✅ `requirements.txt` re-encoded to UTF-8 and split into `requirements.txt` /
   `requirements-ml.txt` (torch, lightgbm, optuna) / `requirements-dev.txt`
   (pytest, ruff). Dropped `pywin32` (Windows-only, uninstallable on Linux), added
   the missing `cfgrib`/`eccodes` that `noaa_downloader` imports at runtime.
2. ✅ Dead npm workflow removed (`index.html` loads React from a CDN — there was
   never a build step). Replaced with `ci.yml`: ruff + pytest + a guard that fails
   the build if any requirements file stops being UTF-8.
3. ✅ Region unified in `src/config.py`; both downloaders import from it.
4. ✅ One flux convention, W/m throughout, `VTM10` preferred with documented
   conversion factors for `VTM02`/`VTPK`.
5. ✅ 32 tests covering the physics and the region config.

**Phase 1a — NDBC buoy pilot — BUILT, not yet run on real data**

Inserted ahead of the gridded pipeline: a few megabytes of real buoy
observations instead of 60 GB, answering the "is there an edge" question before
the heavy infrastructure is committed to. See
[`pilot_experiment.md`](pilot_experiment.md).

- ✅ `src/data/ndbc.py` — NDBC fetch/parse with header-driven column resolution,
  per-column sentinel handling, physical range filtering, disk cache
- ✅ `src/data/splits.py` — temporal and rolling-origin splits with an enforced gap
- ✅ `src/features/build.py` — causal lag, rolling, physics, calendar and
  circular-direction features
- ✅ `src/models/` — the `Forecaster` contract, registry, and the first five
  rungs (mean, persistence, seasonal-naive, climatology, ridge, LightGBM), plus
  the `raw_nwp` and `nwp_debiased` postprocessing references
- ✅ `src/eval/` — per-horizon metrics including storm-conditioned RMSE, skill
  against a configurable reference, and the backtest loop
- ✅ `experiments/pilot_ndbc.py` — the runnable entrypoint, both tracks
- ✅ 163 tests, 85 % coverage, including direct leakage tests
- ⛔ **Not run on real data.** The session that built this had no network egress
  to `ndbc.noaa.gov` or the NOAA S3 bucket. Verified end to end on synthetic
  data only; synthetic numbers are not results.

**Phase 1b — Track B, NWP postprocessing — BUILT, first run in flight**

The pilot showed buoy history is exhausted past ~48 h. Beating that needs
information from outside the buoy, which is what a physics forecast supplies.

- ✅ Archive layout established from the bucket's own documentation rather than
  guessed: `GEFSv12/reforecast/{year}/{yyyymmdd}/{gridded,station}/`, one 03Z
  cycle per day, five members, 2000–2019.
- ✅ **Point output found**: `*.tab.nc`, 8.7 MB, holding hourly series at 658
  buoy positions — 200× smaller than the gridded GRIB2 and needing no
  interpolation.
- ✅ **All three pilot buoys matched by NDBC id** (46041 idx 107, 46087 idx 108,
  46029 idx 106), positions agreeing to 1–3 km. The forecast/observation join
  is exact.
- ✅ `src/data/gefs.py` extraction, with a half-degree cap on position fallback
  and download-reduce-delete so peak disk stays flat.
- ✅ `align_forecast_to_issue_time()` isolated and tested — the one operation
  that can leak the future into a postprocessing set.
- ✅ `required_columns` threaded through the backtest, so the raw-NWP reference
  is scored on the same rows as the models it is compared against.
- ✅ `experiments/track_b_postprocess.py`, scoring against raw GEFS with a
  bias-corrected NWP baseline alongside.
- 🔄 First run at stride 3.

**Known constraint:** the reforecast ends in 2019 and the NDBC records start in
2015, so the paired window is 2015–2019. Cycles are daily, so a fixed lead time
yields ~1,800 samples at stride 1 and ~600 at stride 3. Enough for bias
correction; thin for anything more ambitious.

**Phase 1 — Data foundation (~1 week)**
5. Refactor the two downloader scripts into `src/data/sources/` functions with config
   objects; fix the NOAA paginator.
6. `build_dataset.py`: raw → analysis-ready **zarr**, chunked on time, land-masked,
   QC thresholds lifted out of the notebook into config.
7. Add ERA5 wind acquisition.
8. Commit a small (<50 MB) cached benchmark subset — a handful of sites, full time
   range — so CI and contributors can actually run the models.
9. `windows.py` + `splits.py`, with leakage tests.

**Phase 2 — Framework and baselines (~1 week)**
10. `models/base.py`, `registry.py`, `training/train.py`, `eval/metrics.py`, `eval/backtest.py`.
11. Rung 0 baselines + Rung 1 LightGBM. Publish the per-horizon skill table. This is
    the first genuinely defensible result the project will have.

**Phase 3 — Deep models (~2 weeks)**
12. Add torch; LSTM/TCN (Rung 2), then PatchTST (Rung 3).
13. Provision a GPU instance or SageMaker training jobs.
14. Extend to the GEFSv12 20-year corpus.

**Phase 4 — Physics (~2 weeks)**
15. Level A constraints retro-fitted to all existing models — cheap, and likely to
    improve every rung at once.
16. Level B residual PINN. Level C only if B shows the residual term is informative.

**Phase 5 — Serving**
17. Forecast API + `dashboard/`, model registry in
    `s3://panthalassa-ocean-results/ml_models/`, scheduled retraining, drift monitoring.

---

## 7. Dependencies to add

```
# requirements-ml.txt
torch>=2.4              # or keep sklearn-only through Phase 2
lightgbm>=4.5
optuna>=4.0             # hyperparameter search
properscoring>=0.1      # CRPS
zarr>=3.1               # already pinned
hydra-core>=1.3         # or pydantic-settings, for configs

# requirements-dev.txt
pytest>=8.0
ruff>=0.6
```

Keep the deep-learning stack in a separate file so the data pipeline and baselines
stay installable without a 2 GB torch download.

---

## 8. The three decisions that block everything else

1. **Which region?** Three are currently encoded in the repo and they do not overlap.
2. **Which corpus for the deep models?** 3.3 years of Copernicus is not enough for a
   Transformer to beat LightGBM. The GEFSv12 20-year reforecast is the answer, and
   getting it requires fixing the NOAA downloader first.
3. **Torch or not?** Everything through Rung 1 is sklearn/LightGBM. Rungs 2-4 all
   require it. Deciding now determines whether Phase 0 splits the requirements files.
