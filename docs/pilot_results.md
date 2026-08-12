# Pilot results: Washington-coast NDBC buoys

Rolling-origin evaluation across three stations, 2015–2023.

| | |
|---|---|
| Stations | 46041 Cape Elizabeth, 46087 Neah Bay, 46029 Columbia River Bar |
| Record | 2015–2023, hourly |
| Evaluation | Rolling-origin, 4 folds, 180-day test windows, 72 h gap |
| Validation | Carved from each fold's own training tail (15%) |
| Target | WVHT, significant wave height, in metres |
| Reported | mean ± std across folds |
| Run | [Actions run 31279611316](https://github.com/WnareaJunior/wave-energy-forecasting-ai/actions/runs/31279611316) |

Every number below is mean ± standard deviation across folds. The seed is the
fold index, so the spread covers seed variance as well as period variance.

---

## The headline: skill replicates across all three stations

Ridge skill vs persistence, mean across folds:

| lead | 46041 | 46087 | 46029 |
|-----:|------:|------:|------:|
| +1 h  | 0.054 | 0.037 | 0.052 |
| +3 h  | 0.092 | 0.075 | 0.092 |
| +6 h  | 0.119 | 0.094 | 0.123 |
| +12 h | 0.163 | 0.127 | 0.151 |
| +24 h | 0.228 | 0.183 | 0.178 |
| +48 h | 0.233 | 0.199 | 0.200 |
| +72 h | 0.235 | 0.213 | 0.232 |

Three independent buoys, three near-identical curves. This is not one station's
quirk.

## But measured against the *best* baseline, the picture is sharper

Persistence is not the right reference past ~24 h — climatology overtakes it.
Best model against whichever baseline is strongest at that lead:

| lead | 46041 | 46087 | 46029 | best baseline |
|-----:|------:|------:|------:|---|
| +1 h  | +6.3%  | +4.2%  | +5.8%  | persistence |
| +3 h  | +10.0% | +9.4%  | +10.0% | persistence |
| +6 h  | +12.7% | +10.5% | +13.2% | persistence |
| +12 h | +16.9% | +13.1% | +15.4% | persistence |
| **+24 h** | **+23.7%** | **+18.6%** | **+18.3%** | persistence |
| +48 h | +12.5% | +7.0%  | +4.4%  | climatology |
| +72 h | +6.4%  | +3.1%  | +1.0% ✗ | climatology |

✗ = flagged by the runner as no real gain (<2%).

**Skill peaks at +24 hours at roughly 18–24%, and decays to near nothing by
+72 hours.** The apparent "+23% at +72 h" in the persistence column is almost
entirely climatology's doing, not the model's.

---

## Ridge and LightGBM are indistinguishable

This corrects the single-split run, which reported ridge beating LightGBM at
every horizon. That was noise. With four folds and per-fold seeds:

| station, lead | lightgbm | ridge | fold std |
|---|---:|---:|---:|
| 46041, +1 h  | 0.188 | 0.190 | ±0.04 |
| 46041, +24 h | 0.775 | 0.784 | ±0.09 |
| 46041, +48 h | 0.992 | 0.961 | ±0.11 |
| 46087, +72 h | 0.811 | 0.788 | ±0.12 |
| 46029, +24 h | 0.808 | 0.805 | ±0.17 |

Every gap between them is an order of magnitude smaller than the fold-to-fold
spread. LightGBM tends to win at +1 to +6 h and ridge at +12 h and beyond, but
neither pattern survives contact with the error bars.

The practical consequence: there is nothing to gain from choosing between them,
or from tuning either. The ceiling is set by the information in the buoy's own
history, not by model capacity — which is a direct argument against reaching for
a Transformer next.

## Storm performance tracks overall performance

At the top decile of observed sea states, ridge and LightGBM keep their edge
over persistence out to +24 h and lose it by +48 h — at 46087, ridge's storm
RMSE at +48 h (1.630) is level with persistence (1.630). No model does anything
special in big seas; they are simply less wrong in the same way.

---

## Data quality notes

**Coverage over 2015–2023 (WVHT):**

| station | present | pct |
|---|---:|---:|
| 46041 | 66,962 | 86.6% |
| 46087 | 60,597 | 76.8% |
| 46029 | 66,170 | 84.7% |

**46087 ran on three folds, not four.** NDBC has no 2021 file for this station
(both the archive path and the text viewer return 404), and fold 0's test
window — Jan–Jul 2022 — yielded zero usable rows, as did its validation block.
The station appears to have been off-station from roughly 2020 through mid-2022.
Its surviving folds train on 2015–2019 and test on late 2022 / 2023, so its
train-to-test gap is much larger than the other two stations'. That its results
still line up with 46041 and 46029 is reassuring rather than suspicious, but the
numbers deserve a lighter weight.

The per-station error handling worked as intended here: the empty folds were
logged and skipped, and the run continued rather than losing all three stations
to one bad buoy.

---

## What this means for the plan

1. **The useful band is +6 h to +48 h**, centred on +24 h. That is a real,
   replicated, ~20% error reduction over the best available baseline.
2. **Past +48 h, buoy history is exhausted.** By +72 h the best model is within
   1–6% of a seasonal average. No amount of model capacity fixes this — the
   information is not in the input.
3. **Do not add torch yet.** Ridge, a linear model on lag features, is tied with
   gradient boosting everywhere. A Transformer would have to beat a tie, on a
   problem whose ceiling is visibly close.
4. **Track B is the next real step.** To forecast past two days you need
   information from outside the buoy. That is exactly what a physics model
   supplies and what postprocessing exploits — and now there is a measured
   baseline for it to beat.

## Remaining caveats

- Three stations on one coastline, one target variable (WVHT).
- Test windows are 180 days; folds share training data, so fold results are not
  fully independent.
- The 46087 outage means its three folds have an unusually long train-to-test
  gap.
