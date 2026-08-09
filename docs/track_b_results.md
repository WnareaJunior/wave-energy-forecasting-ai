# Track B results: can we improve on the physics forecast?

Postprocessing the GEFSv12 wave reforecast at NDBC 46041.

| | |
|---|---|
| Station | 46041, Cape Elizabeth WA |
| Forecast | GEFSv12 reforecast, control member, point output |
| Truth | NDBC 46041 |
| Window | 2015–2019, all 1,826 daily cycles read |
| Samples | ~1,615 usable rows per horizon; folds of ~1,000 train / ~200 test |
| Evaluation | Rolling-origin, 3 folds, mean ± std |
| Reference | **raw GEFS**, not persistence |
| Run | [Actions run 31287166524](https://github.com/WnareaJunior/wave-energy-forecasting-ai/actions/runs/31287166524) |

---

## Finding 1: GEFS is roughly twice as accurate as anything we built

| lead | GEFS RMSE | best buoy-only model (pilot) | ratio |
|-----:|----------:|------------------------------:|------:|
| +24 h | **0.410** | 0.801 | 2.0× |
| +48 h | **0.458** | 0.961 | 2.1× |
| +72 h | **0.505** | 1.031 | 2.0× |

This is the single most useful number the project has produced. Nine years of
buoy history, a full feature pipeline and a model ladder get to 0.80 m at
+24 h. WAVEWATCH III, forced by GEFS winds, gets to 0.41 m — and its error is
almost flat with lead time where ours doubled.

Any operational forecast for these sites should take the physics model as its
input. It is not a baseline to beat; it is the best available signal.

## Finding 2: its bias is real but small, and that caps what correction can do

| lead | n | RMSE | bias | bias² as % of MSE | best possible gain from debiasing |
|-----:|----:|-----:|------:|---:|---:|
| +6 h | 1636 | 0.402 | +0.120 | 8.4% | +4.3% |
| +12 h | 1630 | 0.382 | +0.102 | 7.5% | +3.8% |
| +24 h | 1630 | 0.410 | +0.123 | 10.3% | +5.3% |
| +48 h | 1630 | 0.458 | +0.119 | 5.6% | +2.9% |
| +72 h | 1630 | 0.505 | +0.122 | 5.4% | +2.7% |

GEFS runs about 0.12 m high at this buoy, consistently across all lead times.
That is a genuine systematic error — but squared, it is only 5–10% of mean
squared error. Removing it *perfectly* buys under 5%. Everything else is
variance, which no amount of bias correction touches.

This arithmetic was done before the models ran, and it predicted the result.

## Finding 3: postprocessing delivers roughly what the arithmetic allows

Skill vs raw GEFS, mean ± std across folds:

| lead | nwp_debiased | ridge_postproc | lgbm_postproc |
|-----:|-------------:|---------------:|--------------:|
| +6 h  | −0.2% ± 4.0% | **+7.2% ± 8.0%** | −2.5% ± 22.6% |
| +12 h | −1.2% ± 2.4% | +3.8% ± 4.8% | −15.8% ± 33.1% |
| +24 h | +0.7% ± 2.7% | +1.2% ± 4.9% | −8.4% ± 3.0% |
| +48 h | +0.2% ± 3.1% | −0.8% ± 6.7% | −11.3% ± 4.6% |
| +72 h | +1.2% ± 1.8% | +1.8% ± 2.8% | −6.9% ± 7.6% |

**Read the error bars.** Ridge's +7.2% at +6 h carries a ±8.0% fold spread — it
straddles zero. Past +12 h everything is inside noise. LightGBM is negative
everywhere even after being resized for the sample count.

Storm performance says the same: at the top decile of sea states, raw GEFS is
the best or tied-best model at every horizon.

**Track B is a negative result.** GEFS is well-calibrated at this site, and a
statistical layer on top of it buys nothing that survives its own error bars.

---

## Why this is worth having

The first attempt at this comparison produced ridge at −20% and LightGBM at
−81% — apparently catastrophic. That was an artifact: 113 buoy features against
~300 training rows per fold. Re-running with ten features chosen for a reason,
LightGBM parameters sized for hundreds of rows rather than tens of thousands,
and three times the cycles moved ridge from −20% to +1%.

The conclusion did not change, but it is now a measurement rather than an
overfitting accident. A negative result is only worth stating if the test was
fair.

## What was NOT tested

1. **Ensemble members.** Only the control run (`c00`) was used. The ensemble
   mean is usually more accurate than the control, and ensemble spread is a
   genuine uncertainty predictor. This would likely *improve the baseline*, so
   it narrows the postprocessing case further — but it is the honest next
   comparison, and the spread feature is valuable in its own right.
2. **Other stations.** 46041 only. 46087 and 46029 are wired and would cost one
   more run.
3. **Other targets.** Only significant wave height. Period and direction errors
   were not examined, and power flux depends on both.
4. **Extremes specifically.** Storm-decile RMSE was reported, but no model was
   trained to target extremes, where a WEC's survival loading is set.

## What it means for the project

The ML-forecasting thread has reached its natural end, and the answer is
useful: **use the physics model**. GEFS-Wave is free, public, twice as accurate
as what we could build from buoy history, and near enough unbiased that
correcting it is not worth the machinery.

That redirects effort rather than ending it. The project's original question —
where to site the Panthalassa buoy and what it would earn — is a *resource
assessment* problem, not a forecasting one, and it is where the remaining value
sits. See the roadmap for what that needs.
