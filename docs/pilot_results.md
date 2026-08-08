# Pilot results: NDBC 46041 (Cape Elizabeth, WA)

First real-data run of the buoy pilot.

| | |
|---|---|
| Station | 46041, Cape Elizabeth WA (47.35 N, 124.73 W, ~133 m depth) |
| Record | 2015–2023, hourly, 66,962 observations, 86.6% coverage |
| Train | 2015-03-05 → 2021-12-31 (55,675 rows at +72 h) |
| Validation | 2022 |
| Test | 2023 (8,629 rows at +72 h) |
| Target | WVHT, significant wave height, in metres |
| Reference | persistence |
| Run | [Actions run 31278698042](https://github.com/WnareaJunior/wave-energy-forecasting-ai/actions/runs/31278698042) |

---

## RMSE by horizon (metres, lower is better)

| lead | mean | climatology | seasonal_naive | persistence | lightgbm | **ridge** |
|-----:|-----:|------------:|---------------:|------------:|---------:|----------:|
| +1 h  | 1.206 | 0.999 | 1.048 | 0.168 | 0.162 | **0.160** |
| +3 h  | 1.206 | 1.000 | 1.078 | 0.304 | 0.274 | **0.270** |
| +6 h  | 1.206 | 1.000 | 1.116 | 0.480 | 0.417 | **0.412** |
| +12 h | 1.205 | 1.000 | 1.183 | 0.704 | 0.589 | **0.579** |
| +24 h | 1.205 | 1.000 | 1.240 | 1.030 | 0.808 | **0.801** |
| +48 h | 1.205 | 1.001 | 1.313 | 1.241 | 0.997 | **0.968** |
| +72 h | 1.202 | 0.999 | 1.386 | 1.312 | 1.064 | **0.997** |

## Skill vs persistence (fraction of RMSE removed)

| lead | climatology | lightgbm | ridge |
|-----:|------------:|---------:|------:|
| +1 h  | −4.95 | +3.7% | **+4.8%** |
| +3 h  | −2.29 | +9.7% | **+11.2%** |
| +6 h  | −1.08 | +13.2% | **+14.2%** |
| +12 h | −0.42 | +16.4% | **+17.8%** |
| +24 h | +3.0% | +21.6% | **+22.3%** |
| +48 h | +19.3% | +19.7% | **+22.0%** |
| +72 h | +23.8% | +18.9% | **+24.0%** |

## Storm RMSE, top 10% of observed sea states (metres)

| lead | persistence | lightgbm | ridge |
|-----:|------------:|---------:|------:|
| +1 h  | 0.339 | 0.340 | **0.329** |
| +6 h  | 0.984 | 0.889 | **0.859** |
| +24 h | 1.952 | 1.656 | **1.620** |
| +48 h | 2.273 | 2.227 | **2.027** |
| +72 h | 2.318 | 2.318 | **2.057** |

Bias is negligible for every model (|bias| ≤ 0.06 m at all horizons), so these are
variance differences, not calibration differences.

---

## What this actually says

**There is real skill, and it peaks in the middle of the range.** Ridge beats
persistence at every lead time, by 4.8% at +1 h rising to ~22% by +24 h.

**But the headline "+24% at +72 h" is misleading, and the climatology column is
why.** At +72 h, climatology — which ignores current conditions entirely and
just predicts the seasonal average — scores +23.8%. Ridge scores +24.0%. They
are tied. Ridge's RMSE at +72 h is 0.997 m against climatology's 0.999 m.

So the honest reading, by band:

| band | what is happening |
|---|---|
| **+1 to +6 h** | Modest gains (5–14%) that come from smoothing measurement noise, not from forecasting. Persistence uses one noisy reading; ridge averages recent history. |
| **+12 to +48 h** | The genuinely valuable band. At +24 h ridge is 20% better than climatology *and* 22% better than persistence — it is using current conditions to say something the seasonal average cannot. |
| **+72 h and beyond** | The current-conditions signal is exhausted. Ridge has converged to climatology. Nothing in this buoy's own history carries information that far ahead. |

This is exactly what the climatology baseline was included to expose, and it
would have been invisible from a persistence-only comparison.

**Ridge beats LightGBM at every horizon.** Not by much at short leads, but
consistently, and by more at +48 h and +72 h (0.968 vs 0.997; 0.997 vs 1.064).
Two readings, probably both true: the autoregressive structure of wave height is
close to linear, so trees have little nonlinearity to exploit; and LightGBM's
early stopping ran against a thin validation block (see below), which likely
stopped it short.

---

## Caveats and follow-ups

1. **The 2022 validation block is mostly empty.** At +72 h it yielded 1,716
   usable rows against 8,629 for the 2023 test block, so 46041 appears to have
   a long outage in 2022. That is the set LightGBM early-stops on, so its
   comparison with ridge is not yet clean. Worth re-running with a different
   validation year, or with rolling-origin backtesting instead of a fixed split.
2. **`VIS` and `TIDE` are entirely absent at this station**, and are being
   passed to the models as all-NaN columns — sklearn warns and skips them.
   Harmless but should be dropped during feature construction.
3. **Single test year.** 2023 covers all seasons, which is better than a
   partial-year window, but one year is one draw. Rolling-origin evaluation
   would give an error bar on these numbers.
4. **One station.** 46087 (Neah Bay) and 46029 (Columbia River Bar) have not
   been run. Skill that does not replicate across the three is not a property
   of the Washington coast.
5. **No seed repeats.** LightGBM is stochastic; the ridge/LightGBM gap at short
   leads (0.160 vs 0.162 at +1 h) is well inside what seed variance could
   explain. The +48/+72 h gap is larger and more likely real.

---

## What it means for the plan

The +12 to +48 h band is where a model earns its keep on buoy history alone, and
past ~48 h nothing in the buoy's own past helps. That is a direct argument for
**Track B**: to forecast beyond two days you need information from outside this
buoy — which is what a physics model provides, and what postprocessing it would
exploit.

It is also an argument against reaching for the deep-learning rungs yet. Ridge —
a linear model with lag features — is currently the best thing here, beating
gradient boosting at every horizon. A Transformer has to beat *ridge* to justify
itself, and the gap between ridge and climatology at long leads suggests the
ceiling on buoy-only forecasting is close.
