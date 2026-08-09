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
    # copy=True is required, not defensive: under pandas 3's copy-on-write,
    # `.to_numpy()` hands back a read-only view and the in-place `&=` below
    # raises "output array is read-only". Local runs on pandas 2.3 did not,
    # which is why this reached CI.
    workable = (hs <= threshold_m).to_numpy(copy=True)
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


def _epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Index as int64 **nanoseconds** since the epoch, timezone or not.

    ``asi8`` returns the raw int64 in the index's *own* resolution, which is
    not always nanoseconds. pandas 3 parses text dates to microsecond
    resolution while ``pd.date_range`` produces nanoseconds, so the real NDBC
    records arrive as ``datetime64[us, UTC]`` and every synthetic fixture
    built with ``date_range`` arrives as ``datetime64[ns, UTC]``.

    Reading ``asi8`` without normalising made every duration here 1000x too
    small on the real data and exactly right on every test. The maximum
    waiting time across a nine-year record came out at 7.3 hours.
    """
    return pd.DatetimeIndex(index).as_unit("ns").asi8


#: Longest interruption that may be bridged when measuring a waiting time.
#:
#: Bridge nothing and every momentary dropout censors the wait, discarding the
#: long waits that the mean is mostly made of. Bridge everything and a
#: year-long outage is priced as a year of bad weather, which is the bug this
#: machinery exists to fix. NDBC coverage on these buoys runs 77-87%.
#:
#: A day is the compromise: long enough to absorb a routine dropout, short
#: enough that bridging adds at most 24 h of uncertainty to a wait measured in
#: hundreds. On synthetic records whose window set is provably unchanged by the
#: dropouts, 24 h, 72 h and no censoring at all agree to the hour, while only
#: the last mis-prices a year-long outage - so the value sits on a plateau
#: rather than a knife edge.
#:
#: An earlier version of this comment justified the tolerance with figures
#: taken from the real records - a 4-7 h mean run of observation, 0.5%
#: downtime. Those came from the 1000x units error in _epoch_ns and were
#: measuring nothing. They are removed rather than restated: the argument
#: above rests only on the synthetic sensitivity, which is unaffected.
DEFAULT_MAX_GAP_HOURS = 24.0


def _bridged_observation(hs: pd.Series, max_gap_hours: float) -> np.ndarray:
    """Observation mask with short interruptions filled in.

    An interruption is bridged when it is flanked by observation on both sides
    and spans no more than ``max_gap_hours``. A leading or trailing run of
    missing data is never bridged - there is nothing on one side to bridge to.
    """
    observed = hs.notna().to_numpy()
    n = len(observed)
    if n == 0 or observed.all():
        return observed

    times = _epoch_ns(hs.index)
    missing = ~observed

    edges = np.diff(missing.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1  # exclusive
    if missing[0]:
        starts = np.concatenate([[0], starts])
    if missing[-1]:
        ends = np.concatenate([ends, [n]])

    flanked = (starts > 0) & (ends < n)
    span_hours = (
        times[np.minimum(ends, n - 1)] - times[np.maximum(starts - 1, 0)]
    ) / 3.6e12

    bridged = observed.copy()
    for start, end in zip(starts[flanked & (span_hours <= max_gap_hours)],
                          ends[flanked & (span_hours <= max_gap_hours)]):
        bridged[start:end] = True
    return bridged


def _observed_segments(
    hs: pd.Series,
    step_hours: float,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
) -> np.ndarray:
    """For each timestep, the end time of its run of continuous observation.

    A run breaks on a sustained interruption - a gap in the index *or* a block
    of missing values longer than ``max_gap_hours``. Both mean the same thing:
    the record stopped telling us what the sea was doing for long enough that a
    wait cannot be timed across it.

    The missing-value half matters more than the index half.
    :func:`~src.data.ndbc.load_station` resamples onto a regular hourly grid,
    so an outage appears as a block of NaN rows and the index stays perfectly
    continuous. Splitting on index gaps alone would find nothing to split.

    Returns:
        Array of segment end times as epoch nanoseconds, ``-1`` outside any
        run. Integers rather than datetimes because the index may be
        timezone-aware, in which case ``to_numpy()`` hands back an object array
        of Timestamps that will not compare against ``datetime64``.
    """
    index = hs.index
    n = len(index)
    if n == 0:
        return np.array([], dtype="int64")

    times = _epoch_ns(index)
    observed = _bridged_observation(hs, max_gap_hours)

    continues = np.ones(n, dtype=bool)
    if n > 1:
        gaps = np.diff(times) / 3.6e12  # nanoseconds to hours
        continues[1:] = gaps <= max(step_hours * 1.5, max_gap_hours)

    previously_observed = np.concatenate([[False], observed[:-1]])
    begins = observed & (~previously_observed | ~continues)

    segment_of = np.cumsum(begins) - 1
    ends = np.full(n, -1, dtype="int64")
    if not begins.any():
        return ends

    positions = np.flatnonzero(observed)
    # Later writes win, so each segment id keeps its final position.
    last_position = np.zeros(int(segment_of[positions].max()) + 1, dtype=int)
    last_position[segment_of[positions]] = positions
    ends[positions] = times[last_position[segment_of[positions]]]
    return ends


def _waiting_hours_array(
    hs: pd.Series,
    threshold_m: float,
    required_hours: float,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
) -> tuple:
    """Per-timestep waiting time, NaN where the wait cannot be observed.

    Returns:
        ``(waits, windows)``. ``waits`` is NaN at any timestep whose next
        window lies beyond the end of its contiguous stretch of record - the
        wait is right-censored there and cannot be measured.
    """
    windows = find_windows(hs, threshold_m, required_hours)
    index = hs.index
    waits = np.full(len(index), np.nan)
    if windows.empty or len(index) == 0:
        return waits, windows

    starts = _epoch_ns(pd.DatetimeIndex(windows["start"]))
    times = _epoch_ns(index)

    # Waits are measured only inside a run of continuous observation.
    #
    # Walking the whole record instead counts an outage as waiting: buoy 46087
    # is missing all of 2021, so every timestep late in 2020 "waited" into
    # 2022. The site reported a 1,900 h mean wait while being the most
    # accessible of the three, needing the shortest window, and offering the
    # most windows per year. Those three facts contradicting the fourth is what
    # exposed it.
    segment_ends = _observed_segments(hs, _median_step_hours(index), max_gap_hours)

    positions = np.searchsorted(starts, times, side="left")
    found = positions < len(starts)
    candidate = starts[np.minimum(positions, len(starts) - 1)]
    # A window beyond the end of this run tells us nothing: observation stops
    # before we learn how long the wait actually was.
    measurable = found & (segment_ends >= 0) & (candidate <= segment_ends)

    waits[measurable] = (candidate[measurable] - times[measurable]) / 3.6e12

    # Timesteps already inside a window wait zero.
    for _, window in windows.iterrows():
        inside = np.asarray((index >= window["start"]) & (index <= window["end"]))
        waits[inside] = 0.0

    return waits, windows


def expected_waiting_hours(
    hs: pd.Series,
    threshold_m: float,
    required_hours: float,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
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

    Waits that run past the end of a contiguous stretch of record are censored
    and excluded rather than measured across the gap. That biases the mean
    slightly *low*, because the excluded waits are the long ones - see
    :func:`window_statistics`, which reports the censored fraction so the size
    of the bias is visible. The alternative biases it catastrophically high.

    Returns:
        Mean waiting hours, or NaN if no adequate window exists in the record.
    """
    waits, windows = _waiting_hours_array(
        hs, threshold_m, required_hours, max_gap_hours
    )
    if windows.empty or np.all(np.isnan(waits)):
        return float("nan")
    return float(np.nanmean(waits))


def window_statistics(
    hs: pd.Series,
    threshold_m: float,
    required_hours: float,
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS,
) -> dict:
    """Full accessibility summary for one working limit and job length.

    Returns:
        dict with the accessible fraction, window counts and durations, the
        expected wait, and the implied annual standby hours.
    """
    waits, windows = _waiting_hours_array(
        hs, threshold_m, required_hours, max_gap_hours
    )
    record_hours = _record_hours(hs)
    years = record_hours / HOURS_PER_YEAR if record_hours else float("nan")
    # Censoring is measured among timesteps that sit inside a run of
    # observation. Counting the ones outside would conflate "the sea was too
    # rough for a long time" with "the sensor was offline for a long time",
    # which is the confusion this whole change is about.
    inside_run = (
        _observed_segments(hs, _median_step_hours(hs.index), max_gap_hours) >= 0
    )
    censored = (
        float(np.mean(np.isnan(waits[inside_run])))
        if inside_run.any()
        else float("nan")
    )

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
        "expected_wait_hours": (
            float(np.nanmean(waits))
            if len(waits) and not np.all(np.isnan(waits))
            else float("nan")
        ),
        # Share of timesteps whose wait ran past the end of their contiguous
        # stretch of record. High values mean the mean wait is optimistic.
        "censored_fraction": censored,
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

        # expected_wait_hours is dropped, not reported.
        #
        # It used to be dropped because it was wrong: subsetting by month
        # concatenates the same season across years, so the wait ran from
        # February into the following December and counted the intervening
        # nine months as waiting - 5,307 hours for a 90-day winter. That flaw
        # is now fixed generally, since waits are measured only inside a run
        # of continuous observation and a season boundary breaks the run.
        #
        # It stays dropped for a second reason the fix makes visible rather
        # than removes: within a 90-day block almost every wait runs past the
        # end of the block and is censored. On a winter subset the censored
        # fraction reaches 1.00 and the mean is undefined. The quantity is not
        # estimable from a season at a time, whatever the arithmetic.
        #
        # accessible_fraction and the window counts are unaffected: those are
        # per-timestep and per-run quantities, and find_windows already breaks
        # a run wherever the index skips.
        stats.pop("expected_wait_hours", None)
        stats["season"] = label
        rows.append(stats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("season")
