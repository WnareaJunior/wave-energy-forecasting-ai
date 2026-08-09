"""Weather-window statistics: how often can a vessel actually reach the site?

This is the module that makes the siting question interesting.

Mean wave power tells you what a site could earn. It says nothing about whether
you can get to it, and those two are *positively correlated* - the seas that
carry energy are the seas that stop vessels working. A ranking on mean power
alone systematically favours sites that are expensive to service and often
unreachable when they break.

What matters operationally is not the fraction of hours below a wave-height
limit, but the availability of **continuous windows** long enough to transit
out, do a job, and return. Long windows are disproportionately rarer than short
ones, because they require a run of calm rather than a single calm hour. That
non-linearity is why a site 100 km offshore can be far more than "twice as
awkward" as one 50 km out.

Everything here operates on an observed or modelled Hs time series, so it uses
exactly the data the resource assessment already loads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Hours per year, for annualising counts.
HOURS_PER_YEAR = 8766.0


def accessible_fraction(hs: pd.Series, threshold_m: float) -> float:
    """Fraction of observed hours with Hs at or below the limit.

    The naive accessibility number. Useful as a headline, but it overstates
    real access because it ignores whether the calm hours are contiguous.
    """
    valid = hs.dropna()
    if valid.empty:
        return float("nan")
    return float((valid <= threshold_m).mean())


def find_windows(
    hs: pd.Series,
    threshold_m: float,
    min_hours: float,
) -> pd.DataFrame:
    """Locate every continuous stretch of workable weather.

    A gap in the record breaks a window: an unobserved period cannot be
    asserted to be calm. That is deliberately conservative - the alternative is
    to assume missing data was workable, which flatters exactly the exposed
    sites where sensors fail most.

    Args:
        hs: Significant wave height series on a regular time index.
        threshold_m: Working limit.
        min_hours: Minimum usable window length.

    Returns:
        DataFrame with start, end and duration_hours, one row per window.
    """
    if hs.empty:
        return pd.DataFrame(columns=["start", "end", "duration_hours"])

    index = hs.index
    step_hours = _median_step_hours(index)
    workable = (hs <= threshold_m).to_numpy()
    workable &= hs.notna().to_numpy()

    # A run breaks either when conditions exceed the limit or when the index
    # skips - both mean the window cannot be asserted to continue.
    gaps = np.diff(index.to_numpy()).astype("timedelta64[s]").astype(float) / 3600.0
    contiguous = np.concatenate([[True], gaps <= step_hours * 1.5])

    windows = []
    start_i = None
    for i, ok in enumerate(workable):
        if ok and (start_i is None or not contiguous[i]):
            if start_i is not None:
                windows.append((start_i, i - 1))
            start_i = i
        elif not ok and start_i is not None:
            windows.append((start_i, i - 1))
            start_i = None
    if start_i is not None:
        windows.append((start_i, len(workable) - 1))

    rows = []
    for lo, hi in windows:
        duration = (index[hi] - index[lo]).total_seconds() / 3600.0 + step_hours
        if duration >= min_hours:
            rows.append(
                {"start": index[lo], "end": index[hi], "duration_hours": duration}
            )

    return pd.DataFrame(rows, columns=["start", "end", "duration_hours"])


def _median_step_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    steps = np.diff(index.to_numpy()).astype("timedelta64[s]").astype(float) / 3600.0
    return float(np.median(steps))


def expected_waiting_hours(
    hs: pd.Series,
    threshold_m: float,
    required_hours: float,
) -> float:
    """Mean wait from an arbitrary moment until a usable window opens.

    This is the number that turns a wave climate into a cost. A vessel that
    arrives ready to work but cannot start is usually still on hire, so waiting
    time is charged at the full day rate. At an exposed site in winter it can
    exceed the cost of the work itself.

    Computed by walking the record: for every timestep, how long until the
    start of the next window long enough to do the job? Averaged over all
    timesteps, so it reflects a randomly-timed need for a vessel rather than
    one scheduled for a calm month.

    Returns:
        Mean waiting hours, or NaN if no adequate window exists in the record.
    """
    windows = find_windows(hs, threshold_m, required_hours)
    if windows.empty:
        return float("nan")

    index = hs.index
    starts = windows["start"].to_numpy()
    times = index.to_numpy()

    # For each timestep, the next window start at or after it.
    positions = np.searchsorted(starts, times, side="left")
    waits = np.full(len(times), np.nan)
    found = positions < len(starts)
    waits[found] = (
        starts[positions[found]] - times[found]
    ).astype("timedelta64[s]").astype(float) / 3600.0

    # Timesteps already inside a window wait zero.
    for _, window in windows.iterrows():
        inside = np.asarray((index >= window["start"]) & (index <= window["end"]))
        waits[inside] = 0.0

    return float(np.nanmean(waits))


def window_statistics(
    hs: pd.Series,
    threshold_m: float,
    required_hours: float,
) -> dict:
    """Full accessibility summary for one working limit and job length.

    Returns:
        dict with the accessible fraction, window counts and durations, the
        expected wait, and the implied annual standby hours.
    """
    windows = find_windows(hs, threshold_m, required_hours)
    record_hours = _record_hours(hs)
    years = record_hours / HOURS_PER_YEAR if record_hours else float("nan")

    return {
        "threshold_m": threshold_m,
        "required_hours": required_hours,
        "accessible_fraction": accessible_fraction(hs, threshold_m),
        "n_windows": len(windows),
        "windows_per_year": len(windows) / years if years else float("nan"),
        "mean_window_hours": (
            float(windows["duration_hours"].mean()) if not windows.empty else 0.0
        ),
        "longest_window_hours": (
            float(windows["duration_hours"].max()) if not windows.empty else 0.0
        ),
        "expected_wait_hours": expected_waiting_hours(hs, threshold_m, required_hours),
        "record_years": years,
    }


def _record_hours(hs: pd.Series) -> float:
    if len(hs) < 2:
        return 0.0
    return (hs.index[-1] - hs.index[0]).total_seconds() / 3600.0


def seasonal_accessibility(
    hs: pd.Series,
    threshold_m: float,
    required_hours: float,
) -> pd.DataFrame:
    """Accessibility by season.

    Worth separating because the annual mean hides the operationally decisive
    fact: on this coast the energetic season and the inaccessible season are the
    same months. A failure in December is a different proposition from one in
    July, and a maintenance plan that assumes the annual average will be wrong
    exactly when it matters.
    """
    seasons = {
        "winter (DJF)": [12, 1, 2],
        "spring (MAM)": [3, 4, 5],
        "summer (JJA)": [6, 7, 8],
        "autumn (SON)": [9, 10, 11],
    }

    rows = []
    for label, months in seasons.items():
        subset = hs[hs.index.month.isin(months)]
        if subset.empty:
            continue
        stats = window_statistics(subset, threshold_m, required_hours)
        stats["season"] = label
        rows.append(stats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("season")
