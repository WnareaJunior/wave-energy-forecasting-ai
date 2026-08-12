# Pilot experiment: is there an edge at a Washington-coast buoy?

A small, cheap experiment that runs before the 60 GB gridded pipeline and
answers one question: **at what lead time does a statistical model beat the
traditional alternative, and by how much?**

Everything heavier — the Transformer, the PINN, the full spatial grid — should
be justified against the table this produces.

---

## The two tracks

### Track A — buoy-only forecasting

Forecast wave height at an NDBC buoy using only that buoy's own history.
Reference: **persistence**.

```bash
python experiments/pilot_ndbc.py --station 46041 --years 2015-2023
```

Runs the ladder — mean, persistence, seasonal-naive, climatology, ridge,
LightGBM — at +1, 3, 6, 12, 24, 48 and 72 hours.

This track needs nothing but NDBC. A station-year is a few hundred kilobytes,
cached on disk after the first run.

### Track B — NWP postprocessing

Same buoy, but the model also sees a physics-model forecast and learns to
correct it. Reference: **the raw physics forecast**.

```bash
python experiments/pilot_ndbc.py --station 46041 --years 2015-2023 \
    --nwp-csv data/raw/nwp/46041_gefs.csv
```

This is the track that tests the real claim. Beating persistence is easy;
beating WAVEWATCH III at a specific site is the result worth having, and it is
how ML actually earns its place in operational weather forecasting — not by
replacing the physics, but by learning its site-specific errors.

### Pipeline check without network

```bash
python experiments/pilot_ndbc.py --synthetic --years 2018-2023
python experiments/pilot_ndbc.py --synthetic --years 2018-2023 --nwp-csv synthetic
```

Synthetic numbers are **not results**. The generator in `src/data/synthetic.py`
is deliberately easier to forecast than the real ocean, and its NWP imitation
has a constant bias that makes debiasing look better than it will be. The
synthetic runs exist to prove the plumbing works.

---

## Stations

Defined in `src/config.py`. All positions are nominal deployment positions.

| Station | Name | Position | Note |
|---|---|---|---|
| **46041** | Cape Elizabeth, WA | 47.35 N, 124.73 W | Primary. Exposed WA coast, long record, ~133 m depth |
| **46087** | Neah Bay, WA | 48.49 N, 124.73 W | Northern WA coast, near the Strait entrance |
| **46029** | Columbia River Bar | 46.16 N, 124.51 W | Southern end of the WA shelf |
| 46005 | West Washington | 46.13 N, 131.08 W | Deep-water reference, ~370 km offshore. **Outside** the -130 study boundary |

Start with 46041. It is the most representative of an exposed Washington-coast
deployment site and has the cleanest long record.

---

## Reading the output

Four tables per run, all broken out per horizon. Per-horizon reporting is not
cosmetic — a single aggregate RMSE is dominated by the easy +1 h cases and will
hide a model that is useless at +48 h.

**RMSE by horizon** — absolute error in metres. Useful for scale, not for
comparison.

**Skill vs reference** — the number that matters. `1 - RMSE_model /
RMSE_reference`. Zero means no better than the reference; `+0.10` means 10 % of
the error removed; negative means worse. Expect persistence to be near-unbeatable
at +1 h and to fall apart past +24 h.

**Storm RMSE (top 10 %)** — error restricted to the highest observed sea states.
For a wave energy converter this is the operationally important number: most of
the annual energy arrives in a handful of winter events, and survival loading is
set by the extremes. A model can look excellent overall by nailing calm summer
conditions and still be useless.

**Bias** — mean error, positive meaning the forecast runs high. A systematic
bias is the easiest thing to correct and the most common way a global physics
model fails at one specific site. If the raw NWP shows a consistent bias here,
Track B has something to work with.

---

## The NWP CSV contract

Track B reads forecasts from a CSV rather than fetching them, so the experiment
is not coupled to any one forecast source. Required shape:

```csv
time,h1,h3,h6,h12,h24,h48,h72
2020-01-01T00:00:00Z,2.31,2.44,2.61,2.98,3.40,2.87,2.20
2020-01-01T01:00:00Z,2.35,2.48,2.66,3.01,3.35,2.81,2.18
```

* `time` — the **valid time** in UTC: the time the forecast is *about*.
* `h{N}` — significant wave height in metres, forecast for that valid time from
  an issue time N hours earlier.

The runner shifts each column back onto issue time, so at issue time `t` the
model sees the forecast for `t + N` and nothing later. Getting this backwards is
the one way to leak the future into Track B, which is why the transformation
lives in one place (`run_backtest_with_nwp`) rather than in the CSV.

**Critically: the values must come from a genuine forecast, not a reanalysis.**
A reanalysis has assimilated the observations you are trying to predict. Using
one here would produce spectacular scores and mean nothing.

### Producing the CSV from GEFSv12

The NOAA GEFSv12 wave reforecast lives in the public S3 bucket
`noaa-nws-gefswaves-reforecast-pds` as GRIB2. To build the CSV:

1. List the bucket to establish the current key layout — do this rather than
   assuming a path template, the layout is not stable across the archive:
   `aws s3 ls --no-sign-request s3://noaa-nws-gefswaves-reforecast-pds/GEFSv12/reforecast/`
2. For each forecast cycle, open the GRIB2 with `xarray`/`cfgrib` and select the
   significant-wave-height field.
3. Interpolate to the buoy position (`Station.latitude/longitude` in
   `src/config.py`), nearest-neighbour is fine at 0.25°.
4. For each lead time, write the value against its valid time.
5. Use the ensemble mean, or the control member — record which. Ensemble spread
   is a useful extra feature later, for the probabilistic rung.

This step was not run here: the session that built this pipeline had no egress
to `noaa-nws-gefswaves-reforecast-pds.s3.amazonaws.com`. The CSV contract above
is the interface it needs to satisfy.

---

## What a result looks like

Track A, expected shape based on how wave forecasting generally behaves — **not
a measurement**:

* +1 to +3 h — persistence wins or ties. Nothing to gain.
* +6 to +12 h — ML pulls ahead by a few percent. Real but marginal.
* +24 to +72 h — the interesting region. Persistence collapses, climatology
  becomes competitive, and a model that uses both current conditions and
  seasonality should show double-digit skill.

Track B is the harder bar and the honest test. Published NWP postprocessing
studies typically report single-digit to low-double-digit RMSE reductions over
the raw model. If this pilot lands in that range at a Washington buoy, the
approach is working and the heavier rungs are justified. If it lands at zero,
that is a genuine finding delivered for a few days of work rather than after the
full gridded pipeline is built.

---

## Decision gate

Run Track A, then Track B, then decide:

| Outcome | Next step |
|---|---|
| Track B shows a consistent edge past +12 h | Scale up: more stations, GEFSv12 20-year corpus, then the deep rungs |
| Track B ≈ 0 but Track A shows skill | The buoy history has signal the NWP lacks — worth pursuing, different framing |
| Both ≈ 0 | Re-examine the target. Power flux or a longer horizon may be the better problem than raw Hs |

Do not add torch until this table exists. LightGBM is the honest bar, and if a
Transformer cannot beat it, that is cheap to learn now and expensive to learn
later.
