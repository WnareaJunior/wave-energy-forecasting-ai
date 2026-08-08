"""Check whether the GEFS point output covers our Washington-coast buoys.

The reforecast's ``*.tab.nc`` files hold time series at 658 "positions of wave
buoys". Whether those positions are *our* buoys, and how they are labelled,
decides how the pairing works:

  * if ``station_name`` carries NDBC ids, forecasts join to observations on the
    id and the pairing is exact
  * if it carries arbitrary labels, we fall back to nearest-position matching
    and must check the nearest point is actually close

This script answers that, and confirms the time axis semantics, from one 8.7 MB
file. It walks nothing - the key layout is already known and recorded in
src.config.

Usage::

    python scripts/probe_gefs_stations.py --date 20160104
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import NDBC_STATIONS, NOAA_SOURCE_BUCKET, PILOT_STATIONS  # noqa: E402


def make_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def find_tab_file(s3, date: str, member: str = "c00") -> str | None:
    """Locate a point-output table for one cycle date.

    Cycles do not exist for every calendar day - the 2000 archive starts on the
    8th of January - so this lists the day's prefix rather than assuming a key.
    """
    prefix = f"GEFSv12/reforecast/{date[:4]}/{date}/station/"
    response = s3.list_objects_v2(Bucket=NOAA_SOURCE_BUCKET, Prefix=prefix)
    contents = response.get("Contents", [])
    if not contents:
        print(f"No objects under {prefix}")
        return None

    tabs = [o["Key"] for o in contents if o["Key"].endswith(f".{member}.tab.nc")]
    if not tabs:
        tabs = [o["Key"] for o in contents if o["Key"].endswith(".tab.nc")]
    return tabs[0] if tabs else None


def decode_station_names(ds) -> list[str]:
    """Turn the (station, string40) character array into a list of strings."""
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default="20160104",
        help="Cycle date, yyyymmdd. Default sits inside the 2015-2019 window "
        "where the reforecast overlaps the NDBC records.",
    )
    parser.add_argument("--member", default="c00")
    args = parser.parse_args(argv)

    import xarray as xr

    s3 = make_client()

    key = find_tab_file(s3, args.date, args.member)
    if not key:
        print(f"FAILED: no point-output table found for {args.date}")
        return 1

    head = s3.head_object(Bucket=NOAA_SOURCE_BUCKET, Key=key)
    print(f"Key:  {key}")
    print(f"Size: {head['ContentLength'] / 1e6:.1f} MB")

    local = Path("/tmp") / Path(key).name
    s3.download_file(NOAA_SOURCE_BUCKET, key, str(local))

    try:
        ds = xr.open_dataset(local)

        print("\n" + "=" * 78)
        print("TIME AXIS")
        print("=" * 78)
        times = ds["time"].values
        print(f"n={len(times)}  first={times[0]}  last={times[-1]}")
        spacing = np.diff(times[:5]).astype("timedelta64[m]")
        print(f"spacing of first steps: {spacing}")
        span_hours = (times[-1] - times[0]) / np.timedelta64(1, "h")
        print(f"span: {span_hours:.0f} h = {span_hours / 24:.1f} days")
        print(
            "time is the VALID time, so lead hours = time - cycle init "
            f"(cycle {args.date} at 03Z)"
        )

        print("\n" + "=" * 78)
        print("STATIONS")
        print("=" * 78)
        names = decode_station_names(ds)
        print(f"{len(names)} stations")
        print(f"first 12 names: {names[:12]}")

        # Positions are data variables with a time dimension, not coordinates,
        # so take the first timestep.
        lats = ds["latitude"].isel(time=0).values
        lons = ds["longitude"].isel(time=0).values

        print("\n" + "=" * 78)
        print("MATCHING OUR PILOT BUOYS")
        print("=" * 78)

        for station_id in PILOT_STATIONS:
            info = NDBC_STATIONS[station_id]
            matches = [i for i, n in enumerate(names) if station_id in n]

            if matches:
                i = matches[0]
                print(
                    f"\n{station_id} ({info.name}): FOUND by name at index {i} "
                    f"as {names[i]!r}"
                )
                print(f"  archive position: {lats[i]:.3f} N, {lons[i]:.3f} E")
                print(f"  NDBC position:    {info.latitude:.3f} N, {info.longitude:.3f} E")
            else:
                # Fall back to nearest position, and report how far off it is.
                lon = info.longitude % 360 if np.nanmin(lons) >= 0 else info.longitude
                distance = np.hypot(lats - info.latitude, lons - lon)
                i = int(np.nanargmin(distance))
                print(f"\n{station_id} ({info.name}): NOT found by name")
                print(f"  nearest station: index {i}, name {names[i]!r}")
                print(f"  at {lats[i]:.3f} N, {lons[i]:.3f} E, {distance[i]:.3f} deg away")

            series = ds["hs"].isel(station=i).values
            finite = series[np.isfinite(series)]
            if len(finite):
                print(
                    f"  hs: n={len(finite)}/{len(series)} finite, "
                    f"first={finite[:5].round(2)}, "
                    f"min={finite.min():.2f}, max={finite.max():.2f} m"
                )
            else:
                print("  hs: all NaN at this station")

        print("\n" + "=" * 78)
        print("VARIABLES")
        print("=" * 78)
        for var in ds.data_vars:
            attrs = ds[var].attrs
            print(
                f"  {var}: {attrs.get('long_name', '?')} "
                f"[{attrs.get('units', '?')}] dims={ds[var].dims}"
            )

        return 0
    finally:
        if local.exists():
            os.remove(local)


if __name__ == "__main__":
    raise SystemExit(main())
