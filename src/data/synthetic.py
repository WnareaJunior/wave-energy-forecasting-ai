"""Synthetic buoy records for testing the pipeline without network access.

This exists so the full experiment can be exercised end to end in CI and on a
machine that cannot reach NDBC. It is a crude imitation of a wave record - an
annual cycle, storms arriving as a Poisson process, red-noise variability - and
it is deliberately easier to forecast than the real ocean.

Numbers produced from synthetic data are NOT results. Nothing generated here
should appear in a report. The point is to prove the plumbing works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_buoy_record(
    start: str = "2018-01-01",
    end: str = "2024-01-01",
    freq: str = "1h",
    seed: int = 0,
    missing_fraction: float = 0.13,
    outage_count: int = 6,
) -> pd.DataFrame:
    """Generate a synthetic NDBC-like hourly record.

    Reproduces the structural features that break naive pipelines: a strong
    winter/summer cycle, autocorrelated storms, scattered dropouts, and
    multi-day sensor outages.

    Args:
        start: First timestamp.
        end: Last timestamp (exclusive).
        freq: Sampling frequency.
        seed: Random seed.
        missing_fraction: Fraction of scattered isolated missing values. The
            default matches NDBC 46041's measured gappiness (~13% of hours
            absent over 2015-2023). An earlier default of 3% was optimistic
            enough to hide a bug that only appeared on real data.
        outage_count: Number of multi-day gaps to punch in the record.

    Returns:
        DataFrame with WVHT, DPD, APD, MWD, WSPD, WDIR columns, indexed by UTC
        timestamp - the same shape :func:`src.data.ndbc.load_station` returns.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    n = len(index)

    day_of_year = index.dayofyear.to_numpy()
    # Annual cycle: peaks in January, minimum in July.
    seasonal = 2.6 + 1.5 * np.cos(2 * np.pi * (day_of_year - 5) / 365.25)

    # Red noise via an AR(1) process - wave height is smooth over hours.
    phi = 0.985
    noise = rng.normal(0, 0.35, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + noise[i]

    # Storms: occasional multi-day surges on top of the background.
    storms = np.zeros(n)
    n_storms = max(1, int(outage_count * 8))
    for _ in range(n_storms):
        centre = rng.integers(0, n)
        width = rng.integers(24, 96)
        amplitude = rng.gamma(2.0, 1.2)
        lo, hi = max(0, centre - width), min(n, centre + width)
        window = np.arange(lo, hi)
        storms[lo:hi] += amplitude * np.exp(-(((window - centre) / (width / 2)) ** 2))

    wvht = np.clip(seasonal + ar + storms, 0.15, 20.0)

    # Period correlates with height but not perfectly.
    dpd = np.clip(6.0 + 1.5 * np.sqrt(wvht) + rng.normal(0, 1.0, n), 2.0, 25.0)
    apd = np.clip(dpd * 0.72 + rng.normal(0, 0.4, n), 1.5, 20.0)

    # Direction: mostly westerly swell, wrapping correctly.
    mwd = (265 + 25 * np.sin(2 * np.pi * day_of_year / 365.25) + rng.normal(0, 18, n)) % 360
    wspd = np.clip(3.0 + 2.2 * wvht + rng.normal(0, 1.8, n), 0.0, 40.0)
    wdir = (mwd + rng.normal(0, 30, n)) % 360

    df = pd.DataFrame(
        {"WVHT": wvht, "DPD": dpd, "APD": apd, "MWD": mwd, "WSPD": wspd, "WDIR": wdir},
        index=index,
    )
    df.index.name = "time"

    # Scattered dropouts.
    if missing_fraction > 0:
        for column in df.columns:
            mask = rng.random(n) < missing_fraction
            df.loc[mask, column] = np.nan

    # Multi-day outages, which is how buoy sensors actually fail.
    for _ in range(outage_count):
        start_i = rng.integers(0, max(1, n - 400))
        length = rng.integers(48, 400)
        df.iloc[start_i : start_i + length] = np.nan

    return df


def generate_nwp_forecast(
    truth: pd.DataFrame,
    horizon: int,
    seed: int = 0,
    bias_m: float = 0.25,
    skill_decay_hours: float = 60.0,
) -> pd.Series:
    """Fake a physics-model forecast of WVHT at a given lead time.

    Imitates the error structure real NWP wave forecasts have, which is what
    makes postprocessing worth doing:

    * a **systematic site bias** (constant offset) - the easiest thing for a
      statistical model to remove
    * an **amplitude error** that grows with sea state, so storms are
      under-predicted
    * **random error growing with lead time**

    Args:
        truth: Observed record containing WVHT.
        horizon: Lead time in hours.
        seed: Random seed.
        bias_m: Constant bias in metres.
        skill_decay_hours: E-folding time for random error growth.

    Returns:
        Forecast series valid at each timestamp, indexed like ``truth``.
    """
    rng = np.random.default_rng(seed + horizon)
    actual = truth["WVHT"]
    n = len(actual)

    error_scale = 0.20 + 0.9 * (1 - np.exp(-horizon / skill_decay_hours))
    random_error = rng.normal(0, error_scale, n)

    # Under-predict big seas by 8%: a real and well-documented NWP failure mode.
    amplitude_error = -0.08 * np.clip(actual - 3.0, 0, None)

    forecast = actual + bias_m + amplitude_error + random_error
    return pd.Series(np.clip(forecast, 0.1, None), index=truth.index, name="nwp_wvht")
