# Track B results: can we improve on the physics forecast?

Postprocessing the GEFSv12 wave reforecast at three Washington-coast buoys.

| | |
|---|---|
| Stations | 46041 Cape Elizabeth, 46087 Neah Bay, 46029 Columbia River Bar |
| Forecast | GEFSv12 reforecast, **5-member ensemble mean**, point output |
| Truth | NDBC observations |
| Window | 2015–2019, 913 cycles per station at stride 2, all read, 5/5 members |
| Evaluation | Rolling-origin, 3 folds, mean ± std across folds |
| Reference | **raw GEFS ensemble mean**, not persistence |
| Run | [Actions run 31288635826](https://github.com/WnareaJunior/wave-energy-forecasting-ai/actions/runs/31288635826) |

---

## Finding 1: GEFS is roughly twice as accurate as anything we built

| lead | GEFS (46041) | best buoy-only model (pilot) | ratio |
|-----:|-------------:|------------------------------:|------:|
| +24 h | **0.420** | 0.801 | 1.9× |
| +48 h | **0.474** | 0.961 | 2.0× |
| +72 h | **0.513** | 1.031 | 2.0× |

Nine years of buoy history, a full feature pipeline and a model ladder reach
0.80 m at +24 h. WAVEWATCH III forced by GEFS winds reaches 0.42 m, and its
error is nearly flat with lead time where ours doubled. **The physics model is
the best available signal — treat it as an input, not a baseline to beat.**

## Finding 2: the three sites behave completely differently

This is what the single-station run missed.

| station | mean bias | RMSE +6 h → +72 h | growth |
|---|---:|---|---:|
| 46041 Cape Elizabeth | **+0.156 m** | 0.415 → 0.513 | +24% |
| 46087 Neah Bay | **−0.015 m** | 0.376 → 0.411 | **+9%** |
| 46029 Columbia River Bar | +0.107 m | 0.389 → 0.448 | +15% |

The two open-coast buoys carry a consistent positive bias of 0.11–0.16 m. Neah
Bay carries essentially none, and its error barely grows with lead time.

## Finding 3: postprocessing works at one station out of three

Ridge skill against raw GEFS, mean ± std across folds, with the ratio that
matters — a mean smaller than its own spread is not separable from zero:

| lead | 46041 | 46087 | 46029 |
|-----:|------:|------:|------:|
| +6 h  | +8.3% ± 10.6% (0.8×) | **+17.8% ± 8.3% (2.1×)** | +6.5% ± 12.3% (0.5×) |
| +12 h | −6.6% ± 12.7% | **+14.5% ± 1.4% (10.4×)** | −1.3% ± 12.0% |
| +24 h | +4.0% ± 4.3% (0.9×) | **+12.4% ± 6.4% (1.9×)** | +1.1% ± 5.1% (0.2×) |
| +48 h | +1.2% ± 12.6% | +5.9% ± 4.0% (1.5×) | +9.1% ± 5.9% (1.5×) |
| +72 h | +1.8% ± 2.2% (0.8×) | +5.6% ± 8.5% (0.7×) | −1.7% ± 1.4% |

At **46041 and 46029 nothing is separable from zero** — every ratio is below 1,
and the earlier single-station conclusion holds there.

At **46087, +12 h and +24 h are real**: +14.5% with a ±1.4% spread is ten times
its own noise, across three independent folds. That is not an artifact.

## Why Neah Bay is different, and why it makes sense

46087 sits at the entrance to the Strait of Juan de Fuca. The other two are open
coast.

A 0.25° global model has grid cells about 25 km across. It cannot resolve a
strait entrance: tidal currents, sheltering by Vancouver Island, and refraction
around the headland all operate below its resolution. So its error there is not
a constant offset — the bias is near zero — but a **state-dependent** error that
depends on conditions the model cannot see.

That is precisely the situation a local statistical correction is for. It has
local observations the global model lacks, and it can learn the conditional
structure of the error. At the open-coast sites there is no unresolved local
physics to exploit, only a constant offset that debiasing already removes, and
sure enough neither ridge nor LightGBM adds anything there.

The flat error growth at 46087 supports the same reading: if error were
dominated by forecast divergence it would grow with lead time as it does at the
other two. It does not, which points at a persistent local effect rather than a
forecasting failure.

**The generalisable claim: postprocessing pays where the physics model cannot
resolve local effects — and only there.** Sheltered, complex or nearshore sites
are candidates. Exposed open-coast sites are not.

---

## What this changes

The single-station conclusion was "GEFS is well-calibrated, postprocessing buys
nothing." That was right about 46041 and wrong as a general statement. Running
the other two stations cost one workflow run and overturned the generalisation
while confirming the specific case.

Worth noting how nearly this was missed. Had the pilot stopped at 46041 — the
buoy chosen because it had the longest cleanest record — the answer would have
been a confident negative.

## Caveats

1. **Three folds.** +12 h at 46087 is convincing; +6 h and +24 h have ratios near
   2, which is suggestive rather than settled. More folds or more cycles would
   tighten them.
2. **Stride 2**, so ~900 cycles rather than 1,826. At a fixed lead that is ~900
   samples.
3. **One target.** Significant wave height only; period and direction untested.
4. **LightGBM still loses** almost everywhere, including at 46087 except at
   +72 h. With a few hundred rows per fold, trees have nothing to work with that
   ridge does not already capture.
5. **Ensemble spread was available as a feature** and did not rescue the two
   open-coast sites, which is itself informative: the ensemble knows when it is
   uncertain, but that does not tell a model how to correct a bias that is not
   there.

## What to do with it

- For 46041 and 46029: **use raw GEFS**. No statistical layer.
- For 46087 or any similarly sheltered site: a **ridge postprocessor at +12–24 h
  is worth about 12–15%**, and ridge specifically — not gradient boosting.
- Before deploying near a strait, headland or shoal, expect the global model to
  be locally wrong in a learnable way, and budget for a local correction.
