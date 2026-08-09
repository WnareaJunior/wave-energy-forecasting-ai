"""Extract GEFSv12 wave reforecast time series at NDBC buoy positions.

This is the *forecast* side of the postprocessing pairing. The buoy record from
:mod:`src.data.ndbc` is the truth side; a model learns the difference.

Which files
-----------
The archive publishes three output types per cycle. This module reads only the
point-output table::

    GEFSv12/reforecast/{year}/{date}/station/gefs.wave.{date}.{member}.tab.nc

At 8.7 MB it is 200x smaller than the gridded GRIB2 field (~1.75 GB) and 60x
smaller than the spectral file (~516 MB), and it already holds time series at
658 buoy positions - so there is nothing to interpolate.

The leakage rule
----------------
``time`` in these files is the **valid time**: the time the forecast is about.
Lead time is not stored. It has to be derived::

    lead_hours = valid_time - cycle_init        (cycle_init = date at 03Z)

Everything downstream depends on that subtraction pointing the right way. A
forecast may only be used to predict a valid time *at or after* its own
initialisation, and :func:`extract_cycle` asserts it.

Volume
------
One cycle per day. Restricting to the control member, the 2015-2019 overlap
with the NDBC records is ~1,800 files, about 16 GB of transfer. Files are
deleted immediately after their handful of rows are extracted, so peak disk
stays at a few tens of megabytes regardless of how many cycles are processed.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    NDBC_STATIONS,
    NOAA_CYCLE_HOUR,
    NOAA_SOURCE_BUCKET,
    NOAA_STATION_TAB_TEMPLATE,
)

logger = logging.getLogger(__name__)

#: Reject a nearest-position match further than this from the buoy. At 0.25
#: degrees grid spacing anything beyond half a degree is a different piece of
#: ocean, and silently pairing it with the buoy would poison the dataset.
MAX_MATCH_DEGREES = 0.5

#: Variables worth carrying forward. `hs` is the forecast target; the rest are
#: candidate predictors for the postprocessing model.
FORECAST_VARS = ("hs", "tr", "fp", "lm", "th1p")


def make_client(max_pool_connections: int = 32):
    """Anonymous S3 client. The reforecast bucket is public.

    The connection pool is sized above the download concurrency; botocore's
    default of 10 produced hundreds of "Connection pool is full, discarding
    connection" warnings on a 609-cycle run. Harmless, but it means connections
    are being torn down and re-established for no reason.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED, max_pool_connections=max_pool_connections
        ),
    )


def cycle_init(date: str) -> pd.Timestamp:
    """Initialisation time of a cycle, from its yyyymmdd date.

    The reforecast runs one cycle per day at 03Z.
    """
    return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{NOAA_CYCLE_HOUR:02d}", tz="UTC")


def cycle_key(date: str, member: str = "c00") -> str:
    return NOAA_STATION_TAB_TEMPLATE.format(year=date[:4], date=date, member=member)


def decode_station_names(ds) -> list[str]:
    """Turn the (station, string40) character array into stripped strings."""
    raw = ds["station_name"].values
    names = []
    for row in raw:
        if isinstance(row, bytes):
            names.append(row.decode("utf-8", errors="replace").strip())
        elif isinstance(row, str):
            names.append(row.strip())
        else:
            joined = b"".join(
                c if isinstance(c, bytes) else str(c).encode() for c in row
            )
            names.append(joined.decode("utf-8", errors="replace").strip())
    return names


def resolve_station_indices(ds, station_ids) -> dict[str, int]:
    """Map NDBC station ids to indices in the archive's station dimension.

    Matches on ``station_name`` first, since an id match is exact. Falls back to
    the nearest archived position, and raises if the nearest is further than
    :data:`MAX_MATCH_DEGREES` - a silent bad match would pair a buoy with a
    different piece of ocean and quietly invalidate everything downstream.

    Args:
        ds: An open ``tab.nc`` dataset.
        station_ids: NDBC ids, e.g. ``("46041", "46087")``.

    Returns:
        Mapping of station id to station index.

    Raises:
        KeyError: if a station can be matched neither by name nor by position.
    """
    names = decode_station_names(ds)
    # Positions are data variables with a time dimension, not coordinates.
    latitudes = np.asarray(ds["latitude"].isel(time=0).values, dtype=float)
    longitudes = np.asarray(ds["longitude"].isel(time=0).values, dtype=float)

    resolved: dict[str, int] = {}

    for station_id in station_ids:
        matches = [i for i, name in enumerate(names) if station_id in name]
        if matches:
            index = matches[0]
            logger.info(
                "Station %s matched by name at index %d (%r)",
                station_id, index, names[index],
            )
            resolved[station_id] = index
            continue

        info = NDBC_STATIONS.get(station_id)
        if info is None:
            raise KeyError(
                f"Station {station_id} not found by name and has no known "
                "position to match against. Add it to NDBC_STATIONS."
            )

        lon = info.longitude
        if np.nanmin(longitudes) >= 0:
            lon = lon % 360

        distance = np.hypot(latitudes - info.latitude, longitudes - lon)
        index = int(np.nanargmin(distance))

        if distance[index] > MAX_MATCH_DEGREES:
            raise KeyError(
                f"Station {station_id} not found by name, and the nearest "
                f"archived position ({latitudes[index]:.3f} N, "
                f"{longitudes[index]:.3f} E) is {distance[index]:.3f} degrees "
                f"away - beyond the {MAX_MATCH_DEGREES} degree limit. Refusing "
                "to pair a buoy with a different piece of ocean."
            )

        logger.warning(
            "Station %s not found by name; using nearest position at index %d "
            "(%r), %.3f degrees away",
            station_id, index, names[index], distance[index],
        )
        resolved[station_id] = index

    return resolved


def extract_cycle(ds, station_indices: dict[str, int], init: pd.Timestamp) -> pd.DataFrame:
    """Pull the forecast series for the given stations out of one cycle.

    Args:
        ds: Open ``tab.nc`` dataset for a single cycle and member.
        station_indices: Mapping from :func:`resolve_station_indices`.
        init: Cycle initialisation time.

    Returns:
        Long DataFrame with columns station, init_time, valid_time, lead_hours
        and one column per forecast variable.

    Raises:
        ValueError: if any valid time precedes initialisation, which would mean
            the lead-time derivation is inverted.
    """
    valid_times = pd.to_datetime(ds["time"].values, utc=True)
    lead_hours = (valid_times - init).total_seconds() / 3600.0

    if (lead_hours < 0).any():
        raise ValueError(
            f"{int((lead_hours < 0).sum())} valid times precede cycle init {init}. "
            "Lead time is derived as valid_time - init; a negative lead means "
            "that subtraction is inverted and the dataset would leak the future."
        )

    frames = []
    for station_id, index in station_indices.items():
        data = {
            "station": station_id,
            "init_time": init,
            "valid_time": valid_times,
            "lead_hours": lead_hours.round().astype(int),
        }
        for var in FORECAST_VARS:
            if var in ds.data_vars:
                data[var] = np.asarray(ds[var].isel(station=index).values, dtype=float)
        frames.append(pd.DataFrame(data))

    return pd.concat(frames, ignore_index=True)


def download_cycle(
    s3,
    date: str,
    member: str = "c00",
    tmp_dir: Path | str = "/tmp/gefs",
) -> Path | None:
    """Download one cycle file. Safe to call from several threads.

    Only the network call happens here. Reading the NetCDF does not, and that
    separation is deliberate - see :func:`build_forecast_dataset`.

    Returns:
        Local path, or None if the cycle does not exist. Not every calendar day
        has one.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local = tmp_dir / f"{date}.{member}.tab.nc"

    try:
        s3.download_file(NOAA_SOURCE_BUCKET, cycle_key(date, member), str(local))
    except Exception as e:
        logger.debug("No cycle for %s (%s): %s", date, member, type(e).__name__)
        if local.exists():
            os.remove(local)
        return None
    return local


def read_cycle(
    path: Path,
    date: str,
    station_indices_cache: dict,
    station_ids,
    delete: bool = True,
) -> pd.DataFrame | None:
    """Extract the stations from a downloaded cycle, then delete the file.

    Must be called from a single thread. Deleting immediately is what keeps
    peak disk flat: the pairing needs a few hundred rows out of each 8.7 MB
    file, and there are hundreds of files.
    """
    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            # Station indices are stable across cycles, so resolve once.
            if "indices" not in station_indices_cache:
                station_indices_cache["indices"] = resolve_station_indices(
                    ds, station_ids
                )
            return extract_cycle(ds, station_indices_cache["indices"], cycle_init(date))
    except Exception as e:
        logger.warning("Failed to read %s: %s: %s", path.name, type(e).__name__, e)
        return None
    finally:
        if delete and path.exists():
            os.remove(path)


def build_forecast_dataset(
    dates,
    station_ids,
    member: str = "c00",
    workers: int = 8,
    tmp_dir: Path | str = "/tmp/gefs",
    progress_every: int = 50,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Extract many cycles into one long-format table.

    Downloads run in parallel; **reads do not**. netCDF4 is backed by HDF5,
    which is normally built without thread safety, so opening these files from
    several threads at once segfaults the interpreter outright - no Python
    traceback, just exit 139. An earlier version did exactly that and died five
    seconds into a 609-cycle run.

    Downloading is the slow part at 8.7 MB a file, so parallelising it and
    reading serially keeps nearly all of the speed. Files are fetched a batch at
    a time and consumed before the next batch starts, which also bounds peak
    disk at ``batch_size`` files.

    Args:
        dates: yyyymmdd strings.
        station_ids: NDBC ids to extract.
        member: Ensemble member. ``c00`` is the control run.
        workers: Concurrent downloads.
        tmp_dir: Scratch directory. Files are removed as they are consumed.
        progress_every: Log a progress line every N cycles.
        batch_size: Cycles downloaded before reading. Peak disk is roughly
            ``batch_size * 8.7 MB``.

    Returns:
        Long DataFrame across all cycles.

    Raises:
        RuntimeError: if no cycle could be read at all.
    """
    s3 = make_client()
    dates = list(dates)
    cache: dict = {}
    frames = []
    processed = 0

    for start in range(0, len(dates), batch_size):
        batch = dates[start : start + batch_size]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(lambda d: download_cycle(s3, d, member, tmp_dir), batch))

        # Serial reads: one HDF5 file open at a time, in this thread only.
        for date, path in zip(batch, paths, strict=True):
            if path is None:
                continue
            result = read_cycle(path, date, cache, station_ids)
            if result is not None:
                frames.append(result)

        processed += len(batch)
        if processed % progress_every < batch_size:
            logger.info(
                "%d/%d cycles processed, %d readable", processed, len(dates), len(frames)
            )

    if not frames:
        raise RuntimeError(
            f"No cycles could be read out of {len(dates)} dates. Check the date "
            f"range, the member ({member}), and that the bucket is reachable."
        )

    logger.info("Read %d of %d cycles", len(frames), len(dates))
    return pd.concat(frames, ignore_index=True)


def daily_dates(start: str, end: str, stride: int = 1) -> list[str]:
    """yyyymmdd strings between two ISO dates.

    Args:
        start: First date, ISO format.
        end: Last date, inclusive.
        stride: Take every Nth day. Subsampling trades statistical power for
            transfer volume; at stride 1 the 2015-2019 window is ~1,800 cycles.
    """
    index = pd.date_range(start, end, freq=f"{stride}D")
    return [d.strftime("%Y%m%d") for d in index]


def align_forecast_to_issue_time(
    at_valid: pd.Series,
    horizon: int,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Move a forecast series from valid time onto issue time.

    A forecast for valid time ``t`` at lead ``h`` was issued at ``t - h``, so
    that is where it belongs in a feature frame: at issue time ``t - h`` the
    model may see the forecast for ``t + 0`` through ``t``, and nothing later.

    This is the single leakage-sensitive operation in the postprocessing
    pipeline. If the shift went the other way the feature at issue time would
    carry a forecast made *after* it, and every metric downstream would look
    excellent and mean nothing. It lives in one tested function for that reason.

    Args:
        at_valid: Forecast values indexed by valid time.
        horizon: Lead time in hours.
        index: Target index (issue times) to align onto.

    Returns:
        Series on ``index``, NaN where no forecast is available.
    """
    if at_valid.empty:
        return pd.Series(np.nan, index=index, name="nwp")

    at_issue = at_valid.copy()
    at_issue.index = at_issue.index - pd.Timedelta(hours=horizon)
    at_issue = at_issue[~at_issue.index.duplicated(keep="first")]
    return at_issue.reindex(index)


def to_wide_csv(
    long_df: pd.DataFrame,
    station: str,
    horizons,
    variable: str = "hs",
) -> pd.DataFrame:
    """Reshape into the CSV contract the pilot's Track B expects.

    That contract is a ``time`` column of valid times plus one ``h{N}`` column
    per lead time, documented in ``docs/pilot_experiment.md``.

    Note the sample count this implies. Cycles are daily, so for a fixed lead
    there is exactly one forecast per day: the 2015-2019 window yields ~1,800
    rows per horizon, not one per hour.

    Args:
        long_df: Output of :func:`build_forecast_dataset`.
        station: Which station to extract.
        horizons: Lead times in hours.
        variable: Forecast variable to pivot.

    Returns:
        DataFrame indexed by valid time with one column per horizon.
    """
    subset = long_df[
        (long_df["station"] == station) & (long_df["lead_hours"].isin(list(horizons)))
    ]
    if subset.empty:
        return pd.DataFrame()

    wide = subset.pivot_table(
        index="valid_time", columns="lead_hours", values=variable, aggfunc="mean"
    )
    wide.columns = [f"h{int(c)}" for c in wide.columns]
    wide.index.name = "time"
    return wide.sort_index()
