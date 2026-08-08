"""Fetch and parse NDBC standard meteorological buoy records.

NDBC publishes one whitespace-delimited text file per station per year at

    https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz

These are real moored-buoy observations — the closest thing to ground truth
available for this project, and the reference a physics forecast gets scored
against. A full station-year is a few hundred kilobytes.

File format (2007 onwards)
--------------------------
Two comment lines, then fixed columns::

    #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES ...
    #yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa ...
    2020 01 01 00 00  120  5.0  6.0  2.50  9.09  6.50 230 1021.0 ...

Files before 2007 lack the minutes column and older ones use a two-digit year;
:func:`parse_stdmet` handles the missing-minutes case but this project uses
2015 onwards, where the modern layout applies throughout.

Wave columns of interest
------------------------
==========  =====================================================  =====
Column      Meaning                                                Units
==========  =====================================================  =====
``WVHT``    Significant wave height (Hm0) - the forecast target    m
``DPD``     Dominant (peak) wave period, i.e. Tp                   s
``APD``     Average wave period, close to the zero-crossing Tm02   s
``MWD``     Mean wave direction waves are coming from              degT
``WSPD``    Wind speed - drives wave generation                    m/s
``WDIR``    Wind direction                                         degT
==========  =====================================================  =====

Missing values are encoded as all-nines sentinels whose width varies by column
(99.00 for WVHT, 999 for MWD, 9999.0 for PRES). Left unhandled these become
enormous outliers that quietly wreck a model, so they are converted to NaN.
"""

from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

NDBC_HISTORICAL_URL = (
    "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"
)

#: Full modern standard-meteorological column set, in file order. Used only as a
#: fallback: the column names are read from the file's own header, because the
#: set has changed over the years (PTDY was added around 2007, the minutes
#: column before that) and any hardcoded list silently mis-assigns every column
#: after the first one that moved.
STDMET_COLUMNS = [
    "YY", "MM", "DD", "hh", "mm",
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "PTDY", "TIDE",
]

#: Historical column names mapped onto their modern equivalents.
COLUMN_ALIASES = {
    "WD": "WDIR",
    "BAR": "PRES",
    "BARO": "PRES",
    "#YY": "YY",
    "YYYY": "YY",
}

TIME_COLUMNS = ("YY", "MM", "DD", "hh", "mm")

#: Per-column missing-data sentinel. NDBC pads with nines to the column width,
#: so the sentinel differs per column and a single global value will not do.
MISSING_SENTINELS = {
    "WDIR": 999,
    "WSPD": 99.0,
    "GST": 99.0,
    "WVHT": 99.00,
    "DPD": 99.00,
    "APD": 99.00,
    "MWD": 999,
    "PRES": 9999.0,
    "ATMP": 999.0,
    "WTMP": 999.0,
    "DEWP": 999.0,
    "VIS": 99.0,
    "TIDE": 99.00,
}

#: Physically plausible ranges. Values outside these are sensor faults that the
#: sentinel replacement does not catch (a stuck sensor, say).
VALID_RANGES = {
    "WVHT": (0.0, 25.0),
    "DPD": (0.5, 30.0),
    "APD": (0.5, 30.0),
    "MWD": (0.0, 360.0),
    "WDIR": (0.0, 360.0),
    "WSPD": (0.0, 60.0),
    "GST": (0.0, 90.0),
}

WAVE_COLUMNS = ["WVHT", "DPD", "APD", "MWD"]
WIND_COLUMNS = ["WSPD", "WDIR", "GST"]


def _resolve_columns(header_line: str | None, n_fields: int) -> list[str]:
    """Column names for a data row, preferring the file's own header.

    NDBC's column set has changed over the decades. Reading the header rather
    than assuming a layout means a file with an unexpected column still parses
    correctly instead of assigning wind speed to the wave-height column.
    """
    if header_line:
        names = [COLUMN_ALIASES.get(c, c) for c in header_line.lstrip("#").split()]
        if len(names) == n_fields:
            return names

    # No usable header: fall back on field count.
    fallback = list(STDMET_COLUMNS)
    if n_fields < len(fallback):
        # Pre-2007 files omit the minutes column, and older ones omit PTDY.
        for optional in ("PTDY", "mm"):
            if len(fallback) > n_fields and optional in fallback:
                fallback.remove(optional)
    return fallback[:n_fields]


def parse_stdmet(text: str) -> pd.DataFrame:
    """Parse the contents of one NDBC standard-meteorological file.

    Args:
        text: Decompressed file contents.

    Returns:
        DataFrame indexed by UTC timestamp, with sentinels converted to NaN and
        out-of-range values dropped. Empty DataFrame if the file has no rows.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    comment_lines = [ln for ln in lines if ln.startswith("#")]
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    if not data_lines:
        return pd.DataFrame()

    n_fields = len(data_lines[0].split())
    columns = _resolve_columns(comment_lines[0] if comment_lines else None, n_fields)

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        names=columns,
        header=None,
    )

    if "mm" not in df.columns:
        df["mm"] = 0

    # Two-digit years appear in the oldest files.
    df["YY"] = df["YY"].astype(int)
    df.loc[df["YY"] < 100, "YY"] += 1900

    df["time"] = pd.to_datetime(
        dict(
            year=df["YY"], month=df["MM"], day=df["DD"],
            hour=df["hh"], minute=df["mm"],
        ),
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df.drop(columns=[c for c in TIME_COLUMNS if c in df.columns])

    for column, sentinel in MISSING_SENTINELS.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df.loc[df[column] >= sentinel, column] = pd.NA

    for column, (low, high) in VALID_RANGES.items():
        if column in df.columns:
            outside = (df[column] < low) | (df[column] > high)
            df.loc[outside, column] = pd.NA

    return df.astype("float64")


def _cache_path(cache_dir: Path, station: str, year: int) -> Path:
    return Path(cache_dir) / f"{station}h{year}.txt.gz"


def fetch_station_year(
    station: str,
    year: int,
    cache_dir: Path | str = "data/raw/ndbc",
    timeout: int = 60,
) -> pd.DataFrame:
    """Download (or read from cache) one station-year of buoy observations.

    The raw ``.txt.gz`` is cached on disk, so re-running an experiment costs no
    network traffic.

    Args:
        station: NDBC station id, e.g. ``"46041"``.
        year: Calendar year.
        cache_dir: Where to store downloaded files.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed observations, or an empty DataFrame if that station-year does not
        exist (common for a buoy's first or last year, and for adrift periods).
    """
    import requests  # imported lazily so parsing works without network deps

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, station, year)

    if not path.exists():
        url = NDBC_HISTORICAL_URL.format(station=station, year=year)
        logger.info("Downloading %s", url)
        response = requests.get(url, timeout=timeout)
        if response.status_code == 404:
            logger.warning("No data for station %s in %s", station, year)
            return pd.DataFrame()
        response.raise_for_status()
        path.write_bytes(response.content)

    with gzip.open(path, "rt", errors="replace") as f:
        return parse_stdmet(f.read())


def load_station(
    station: str,
    years: list[int],
    cache_dir: Path | str = "data/raw/ndbc",
    resample: str | None = "1h",
) -> pd.DataFrame:
    """Load and concatenate several years for one station.

    Args:
        station: NDBC station id.
        years: Years to load. Missing years are skipped with a warning.
        cache_dir: Download cache location.
        resample: Pandas frequency to resample onto, or None to leave the raw
            sampling. Recent buoys report every 10 minutes while older records
            are hourly; resampling to ``"1h"`` puts the whole record on one
            regular grid, which the lag features require.

    Returns:
        DataFrame indexed by UTC timestamp. A regular index means gaps appear as
        NaN rows rather than as silently-skipped time — which is what keeps a
        "lag 3 hours" feature from actually meaning "lag 3 observations".
    """
    frames = [
        df
        for year in years
        if not (df := fetch_station_year(station, year, cache_dir)).empty
    ]
    if not frames:
        raise ValueError(f"No data found for station {station} in years {years}")

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    if resample:
        combined = combined.resample(resample).mean()

    combined.attrs["station"] = station
    return combined


def coverage_report(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Per-column data availability.

    Worth looking at before training: NDBC wave sensors fail for weeks at a
    time, and a station-year that is 40% missing will not support a 72-hour
    forecast evaluation.

    Args:
        df: Observations from :func:`load_station`.
        columns: Columns to report on. Defaults to wave and wind columns.

    Returns:
        DataFrame with counts and percentage present per column.
    """
    columns = columns or [c for c in WAVE_COLUMNS + WIND_COLUMNS if c in df.columns]
    return pd.DataFrame(
        {
            "present": [int(df[c].notna().sum()) for c in columns],
            "missing": [int(df[c].isna().sum()) for c in columns],
            "pct_present": [round(100 * df[c].notna().mean(), 2) for c in columns],
        },
        index=columns,
    )
