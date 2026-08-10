# Where we left off

Written so this project can be picked up cold, in a new session, without
re-deriving anything. Last updated at commit `0956281`.

Everything lives on branch **`claude/forecasting-model-architecture-s5180n`**.

---

## 1. What this project is trying to do

Site a 200 ft ("Panthalassa") wave-energy-harvesting buoy off the Washington
coast, and — the actual point — **find a small edge over a traditional
alternative in this specific niche** by tweaking a model and learning from the
data. The siting study is the domain; the ML edge is the goal.

Two threads run in parallel:

- **Forecasting (Track A/B)** — can a learned model beat the physics model
  (GEFS) at predicting wave conditions?
- **Economics** — what is a forecast improvement actually *worth*, and where
  should the device go?

---

## 2. The headline conclusions

### 2.1 The siting question is answered, and the answer is negative

| | resource kW/m | capacity factor | downtime % | gross $k/yr | O&M $k/yr | annualised capex $k/yr | **net $k/yr** |
|---|---|---|---|---|---|---|---|
| 46041 Cape Elizabeth | 31.5 | 0.207 | 6.9 | 218 | 213 | 680 | **−691** |
| 46087 Neah Bay | 19.5 | 0.133 | 3.1 | 140 | 201 | 679 | **−745** |
| 46029 Columbia River Bar | 29.4 | 0.191 | 5.5 | 201 | 210 | 680 | **−700** |

All three lose money by a wide margin. Robust across all seven sensitivity
scenarios; best case is **−$472k** at $250/MWh.

**The loss is not the resource's fault.** 31.5 kW/m × 20 m capture × 35%
efficiency ≈ 220 kW mean, a 0.21 capacity factor — genuinely respectable for
wave. The number that kills it is **$12M capex for a 1 MW unit**, which
annualises to ~$600k against ~$218k of gross revenue.

Ranking: resource rank == gross rank == net rank (46041 → 46029 → 46087) in
the baseline. Under "3× failure rate" 46029 wins by $3k — the ranking *does*
flip under stress, but by a margin too small to lean on.

### 2.2 The forecasting edge is real, small, and in an unexpected place

- Buoy-only models peak at **+24 h (~20% over the best baseline)**, replicated
  across three buoys, and are exhausted by +72 h.
- **GEFS is ~2× more accurate** than anything built from buoy history alone.
- Postprocessing GEFS is a **negative result** at the two open-coast buoys and
  **works only at 46087 Neah Bay: +14.5% ± 1.4% at +12 h**.

The plausible reason for 46087: a 0.25° grid cannot resolve the entrance to
the Strait of Juan de Fuca, so there is structure left for a model to learn.
**The edge exists where the physics model is geometrically blind, not where
the seas are biggest.**

### 2.3 Why forecast skill currently has almost no dollar value

Forecast quality enters the economics through exactly one channel:
`false_start_multiplier` — vessels that sail against a forecast and abort.

| forecast RMSE (m) | sailings/job | 46041 O&M $k | 46087 | 46029 |
|---|---|---|---|---|
| 0.00 | 1.000 | 213 | 201 | 210 |
| 0.20 | 1.072 | 223 | 210 | 219 |
| **0.41** (GEFS @ +24 h) | 1.302 | 255 | 239 | 250 |
| 0.60 | 1.446 | 275 | 258 | 270 |
| 1.00 | 1.618 | 299 | 279 | 293 |

Going from 1.00 m to 0.20 m RMSE saves ~$76k/yr at 46041 — real, but a
second-order term on a first-order $691k loss. **This is why the next phase
matters: the frame has to change before forecast skill can be worth anything.**

---

## 3. Environment constraints (important — these shaped everything)

- **No direct internet for data.** `www.ndbc.noaa.gov`, the GEFS S3 bucket, and
  `*.blob.core.windows.net` all return **403 from the agent proxy**. Per
  `/root/.ccr/README.md`: never disable TLS verification, never unset
  `HTTPS_PROXY`, do not retry policy denials — report them.
- **Consequence: GitHub Actions is the compute substrate.** Every real-data
  result in this repo comes from a CI run. Each round trip is ~9 minutes.
- **Artifact download is blocked** (blob storage). This is why
  `scripts/execute_notebooks.py` has `--print-outputs`: results land in the
  readable job log, which *is* the accessible record.
- **Committed notebooks store no outputs.** The job log is the source of truth.
- The container is ephemeral and starts with **no dependencies installed**.
  Recreate with:
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements-pilot.txt -r requirements-dev.txt pytest
  .venv/bin/pip install xarray netCDF4 scikit-learn lightgbm
  .venv/bin/python -m pytest tests/          # 342 tests, ~95 s
  ```

### How to run the notebooks against real data

```
Workflow: .github/workflows/notebooks.yml   (workflow_dispatch, or push to src/** or notebooks/**)
Read results: get_job_logs with return_content=true, tail_lines≈140-180
```
Other workflows: `ci.yml` (test matrix over pinned/loose deps),
`pilot.yml`, `explore-gefs.yml`, `track-b.yml`.

---

## 4. Repo map

```
src/config.py                    Single source of truth. Region, NDBC_STATIONS,
                                 GEFS archive constants, SUPPORT_PORTS,
                                 Port/nearest_port/great_circle_km.
src/data/ndbc.py                 Header-driven column resolution, per-column
                                 sentinels, load_station (resamples to "1h").
src/data/gefs.py                 GEFS reforecast: download_cycle (threaded) /
                                 read_cycle (serial), align_forecast_to_issue_time.
src/features/build.py            make_supervised, required_columns handling.
src/processing/wave_power_flux.py  P = rho g^2/(64 pi) Hm0^2 Te. Energy period.
src/economics/deployment.py      Vessel/Operation/SiteLogistics, port classes,
                                 site_logistics(), false_start_multiplier.
src/economics/accessibility.py   Weather windows, waiting times, censoring.
src/economics/revenue.py         DeviceSpec/MarketSpec, evaluate_site.
notebooks/001_data_acquisition_and_qc.ipynb
notebooks/002_wave_power_resource.ipynb
notebooks/003_site_economics.ipynb    ← includes a censoring diagnostic cell
scripts/execute_notebooks.py     Notebook runner with --print-outputs.
tests/                           342 tests.
```

### Key physics detail that is easy to get wrong

Wave power flux uses the **energy period** `Te = m₋₁/m₀`, not peak (`VTPK`) or
zero-crossing (`VTM02`) period. JONSWAP conversions live in
`ENERGY_PERIOD_FACTORS`: VTM10 ×1.00, VTM02 ×1.20, VTPK ×0.86. NDBC gives APD
(≈VTM02), so the notebooks multiply by 1.20. Using APD raw understates by
16.7%; using DPD overstates by 18.8–26.0%.

### Economics assumptions (all documented planning figures, not quotes)

- Device: 20 m capture width, 35% efficiency, 1000 kW rated, 8.0 m survival
  cut-out, $12M capex, 20 yr life.
- Market: $120/MWh flat.
- Vessels: AHTS $25k/day (Hs 2.0 m limit, $250k mob), Multicat $12k/day
  (1.75 m, $75k), CTV $3.5k/day (1.5 m, $0).
- `DEFAULT_STANDBY_CAP_HOURS = 48.0` — only 48 h of waiting is charged as
  vessel hire; the rest becomes downtime. **This cap is why cost is nearly
  insensitive to the wave climate and why downtime carries the penalty.**
- Ports carry the largest vessel class they can host; each vessel is based at
  the nearest port that can take it.

---

## 5. Bugs found, and the lesson from each

This list is the most valuable part of this document. Every one of these was
caught by **an output being implausible**, not by a failing test.

| # | Bug | Lesson |
|---|---|---|
| 1 | `requirements.txt` was UTF-16; pip could not parse it | Check encodings, not just contents |
| 2 | NDBC `PTDY` column omitted from a hardcoded list, shifting every later column | Resolve columns from the header |
| 3 | `data/` in `.gitignore` silently swallowed `src/data/` | Anchor gitignore patterns (`/data/`) |
| 4 | Workflow `inputs` context is empty on `push` | Use workflow-level `env` with `${{ inputs.x \|\| 'default' }}` |
| 5 | `dropna()` across 113 features left ~1 row in 65,000 | Synthetic data at 3% missing hid a real 13.4%; match real missingness |
| 6 | Claimed ridge > LightGBM from a single split | Rolling-origin folds showed it was inside fold variance |
| 7 | `explore_gefs.py` exited 0 having learned nothing | A green check is not a result |
| 8 | Track B segfault: netCDF4/HDF5 opened from 8 threads | Split threaded download from serial read; structural test asserts it |
| 9 | `args.member` vs `args.members` killed a run in 3 s | Test now parses each script's real namespace |
| 10 | `python -m nbclient` — package has no `__main__`; all notebooks "passed" in <1 s | Wrote a real runner |
| 11 | pandas 3 copy-on-write made `.to_numpy()` read-only | Added a CI dependency matrix (pinned/loose) |
| 12 | Standby hours of 3,547/yr | Added the 48 h cap |
| 13 | Sensitivity analysis did nothing — rebinding `ANNUAL_OPERATIONS` had no effect because `revenue.py` bound the name at import | **Pass configuration as parameters, never rebind module globals** |
| 14 | `false_start_multiplier` constant at 1.19 — margin set equal to RMSE, so RMSE cancelled | A table with one row hides what a table with five rows reveals |
| 15 | Seasonal `expected_wait_hours` = 5,307 h for a 90-day winter | Subsetting by month concatenates across years |
| 16 | Single support port for every vessel — 183 km transit for a 16 km inspection | Distance is a property of (site, vessel), not of the site |
| 17 | NaN wait read as zero wait — least serviceable site scored as most accessible | NaN-to-zero looks like hygiene, reads as a lie |
| 18 | Sensor outages counted as waiting for weather | 46087's missing 2021 priced as a year of bad seas |
| 19 | Over-strict censoring — discarded the long waits the mean is made of | Fixing #18 too hard |
| 20 | **1000× units error**: `.asi8` returns the index's *own* unit; pandas 3 parses text to µs, `date_range` gives ns | **The bug was invisible to the whole suite by construction** — every fixture was built the one way that made it right. Tests are now parametrised over ns/µs/ms |

### The meta-lesson

Bugs 13, 14, 17, 18, 19 and 20 all produced **plausible single numbers**. They
were found by:
- printing a table with several rows side by side (13, 14),
- checking one output against three others that contradicted it (18),
- and asking whether a magnitude was physically possible at all (12, 19, 20).

`003_site_economics.ipynb` contains a **censoring diagnostic cell** that exists
solely to report these quantities directly rather than leave them inferred.
Keep it.

Also worth knowing: while fixing #19 I reintroduced #13 in a new place —
`max_gap_hours` is a default argument, so it binds at definition time and
monkeypatching the module global does nothing. There is now a test
(`test_tolerance_is_actually_reachable`) that fails if the parameter stops
being threaded.

---

## 6. What is next (the plan being executed now)

### Phase 1 — array economics + break-even inversion  ← DONE, and it settles the question

**Array scaling at 46041** (real data, run 31344776138):

| N | O&M per device $k | capex per device $k | net per device $k | break-even price $/MWh | break-even capex $M |
|---|---|---|---|---|---|
| 1 | 213.3 | 13,166 | −691.1 | 529 | −1.8 |
| 2 | 145.9 | 13,004 | −609.3 | 481 | −0.2 |
| 5 | 126.4 | 12,906 | −581.2 | 464 | +0.4 |
| 10 | 107.0 | 12,874 | −558.9 | 451 | +0.8 |
| 20 | 97.3 | 12,857 | −547.7 | 444 | +1.0 |

**I was wrong about the array being the biggest cost lever.** It halves O&M per
device (213 → 97 $k, −54%) and moves net per device by only −21%. The reason is
arithmetic: annualised capex is ~86% of per-device annual cost, and device
capex does not amortise across an array at all — only mobilisation, moorings
and campaign sharing do. Diminishing returns are steep: 1→2 devices saves
$82k/device, 10→20 saves $11k.

**Break-even, all three sites:**

| site | N | break-even price $/MWh | × vs $120 | break-even capex $M | × vs $12M |
|---|---|---|---|---|---|
| 46041 | 1 | 529 | 4.4 | −1.82 | −0.15 |
| 46041 | 10 | 451 | 3.8 | +0.82 | 0.07 |
| 46087 | 1 | 778 | 6.5 | −2.90 | −0.24 |
| 46087 | 10 | 665 | 5.5 | **−0.33** | −0.03 |
| 46029 | 1 | 562 | 4.7 | −2.00 | −0.17 |
| 46029 | 10 | 479 | 4.0 | +0.62 | 0.05 |

**46087 stays negative even as a ten-device array**: operating cost alone
exceeds revenue, so the device would have to be free *and* subsidised. Note
that this is the site where the ML postprocessing edge lives.

**Break-even device capex ($M) at 46041, 10-device array** — compare to $12M:

| capture width | $60/MWh | $120 | $250 | $400 |
|---|---|---|---|---|
| 10 m | −2.17 | −1.11 | 1.18 | 3.82 |
| 20 m | −1.20 | **0.82** | 5.21 | 10.28 |
| 35 m | −0.05 | 3.13 | 10.01 | 17.95 |
| 50 m | 0.80 | 4.84 | **13.57** | 23.66 |
| 75 m | 1.83 | 6.88 | 17.84 | **30.48** |

Only two cells clear $12M: **$250/MWh with 50 m capture**, or **$400/MWh with
35 m**. At the baseline 20 m / $120 the device may cost $0.82M — capex must
fall **93%**.

**The verdict.** Closing the gap needs roughly a **5× improvement in the
product of price × capture width**, or a 93% capex reduction. That is a
technology-generation change, not a siting or operations change. No
arrangement of these three sites, at any array size in the model, reaches
zero at plausible parameters.

### Phase 1 (original motivation, kept for the record)

**Why.** One device is the wrong unit of analysis. Mobilisation is $250k *per
campaign*, not per device. CTV visits, standby, and weather-window waits all
amortise across an array — the window penalty is paid once per campaign
regardless of how many units get serviced. Expect per-device O&M to fall by
most of its value.

Paired with it: invert the question. Instead of "does this site make money",
sweep capex / capture width / price / rated power and find the **break-even
surface**. That converts "loses money" into "needs a 4× revenue improvement or
a 75% cost cut", which is the number that decides whether to continue.

### Phase 2 — move forecast value to market settlement (reframed by Phase 1)

Currently forecast skill only touches vessel false-starts (~$76k across the
plausible RMSE range). The frame where it has real value is **day-ahead market
settlement**: commit MWh a day ahead, get penalised for deviating. Forecast
error becomes a cost proportional to *production* rather than to vessel days,
and the +14.5% postprocessing edge at 46087 finally gets a price tag.

**Phase 1 changes what this phase can claim.** Settlement penalties are a few
percent of revenue; the gap to break-even is 3.8×. So Phase 2 **cannot** rescue
the business case and should not be sold as trying to. What it can still do —
and this is the project's actual stated goal — is put an honest dollar value on
a forecasting edge in the frame where forecasting genuinely pays. That is a
transferable method result: the number survives even though this particular
deployment does not.

Note the awkward pairing: the ML edge is at 46087, which is the *worst* site
economically and the only one still negative as a ten-device array.

### Deliberately not done

- Downloading the full gridded GEFS archive (1.75 GB/cycle) — the 8.7 MB point
  output `tab.nc` has everything needed.
- Chasing the downtime term further. **It cannot change the investment
  conclusion**: downtime moves delivered revenue by tens of thousands against a
  ~$690k deficit, and the 48 h standby cap keeps vessel hire nearly insensitive
  to it. Five commits went into this term before that was properly weighed.

---

## 7. Caveats a reader should carry

- Straight-line distances understate routed passages, most of all from **Port
  Angeles**, where the run to the open coast is down the Strait of Juan de Fuca
  rather than across the peninsula. 46087's tow-out cost is optimistic.
- The **Columbia River bar** is itself a weather-limited crossing and is not
  charged for. 46029's costs are optimistic.
- Port class assignments are planning judgements, not port-authority
  statements. Each carries its reasoning in `src/config.py`.
- The unscheduled-failure rate (1.5/yr) is the softest number in the model and
  is optimistic for a prototype marine device.
- `seasonal_accessibility` deliberately omits `expected_wait_hours`: within a
  90-day block almost every wait is censored (winter reaches 1.00), so the
  quantity is not estimable a season at a time.
- Postprocessing skill at 46087 is a single-site result with a physical story
  attached. It has not been validated out of sample at another sheltered site.
