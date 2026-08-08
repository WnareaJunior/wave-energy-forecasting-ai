"""Discover the layout of the NOAA GEFSv12 wave reforecast bucket.

Run this once before writing any extraction code. The archive's key structure is
not documented in a form worth trusting, and the difference between guessing it
and knowing it is the difference between a fetcher that works and one that
silently pulls the wrong field.

The bucket is public, so no credentials are needed - requests are unsigned.

What it reports:
  1. the prefix tree, a few levels deep
  2. sample object keys with sizes
  3. the contents of one GRIB2 file: variables, dimensions, coordinates, and
     crucially how initialisation time, lead time and ensemble member are
     encoded, since those determine how a forecast is aligned to a valid time
  4. the value at the nearest grid point to NDBC 46041, as a sanity check that
     the field is wave height in metres and not something else

Usage::

    python scripts/explore_gefs.py                 # structure only
    python scripts/explore_gefs.py --open-sample   # also download and open a file
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKET = "noaa-nws-gefswaves-reforecast-pds"

#: Do not download anything larger than this when sampling a file.
MAX_SAMPLE_BYTES = 400 * 1024 * 1024


def make_client():
    """Anonymous S3 client. The bucket is public; signing would fail."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_level(s3, prefix: str, max_items: int = 25):
    """One level of the key hierarchy: (sub-prefixes, [(key, size), ...])."""
    paginator = s3.get_paginator("list_objects_v2")
    prefixes: list[str] = []
    keys: list[tuple[str, int]] = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        prefixes.extend(p["Prefix"] for p in page.get("CommonPrefixes", []))
        keys.extend((o["Key"], o["Size"]) for o in page.get("Contents", []))
        if len(prefixes) >= max_items and len(keys) >= max_items:
            break

    return prefixes[:max_items], keys[:max_items]


def walk(s3, prefix: str = "", depth: int = 0, max_depth: int = 4) -> str | None:
    """Print the prefix tree and return the first data file key found."""
    indent = "  " * depth
    prefixes, keys = list_level(s3, prefix)

    for key, size in keys[:5]:
        print(f"{indent}[file] {key}  ({size / 1e6:.1f} MB)")

    first_file = None
    if keys:
        data_files = [
            (k, s) for k, s in keys if k.endswith((".grib2", ".grb2", ".nc", ".zarr"))
        ]
        if data_files:
            first_file = data_files[0][0]

    if depth >= max_depth:
        if prefixes:
            print(f"{indent}... {len(prefixes)} more prefixes, not descending further")
        return first_file

    for sub in prefixes[:6]:
        print(f"{indent}{sub}")
        found = walk(s3, sub, depth + 1, max_depth)
        first_file = first_file or found
        # One complete path down the tree is enough to learn the layout.
        if found and depth >= 1:
            break

    if len(prefixes) > 6:
        print(f"{indent}... and {len(prefixes) - 6} more at this level")

    return first_file


def describe_grib(path: Path) -> None:
    """Open a GRIB2 file and print everything needed to write the extractor."""
    import xarray as xr

    print("\n" + "=" * 78)
    print(f"CONTENTS OF {path.name}")
    print("=" * 78)

    datasets = []
    try:
        datasets = [xr.open_dataset(path, engine="cfgrib")]
    except Exception as e:
        # GRIB2 files often hold messages on several level types, which cfgrib
        # cannot merge into one dataset. open_datasets splits them instead.
        print(f"Single-dataset open failed ({type(e).__name__}: {e})")
        print("Retrying with cfgrib.open_datasets()\n")
        try:
            import cfgrib

            datasets = cfgrib.open_datasets(str(path))
        except Exception as e2:
            print(f"open_datasets also failed: {type(e2).__name__}: {e2}")
            return

    print(f"{len(datasets)} dataset(s) in this file\n")

    for i, ds in enumerate(datasets):
        print(f"--- dataset {i} ---")
        print(f"dims:   {dict(ds.sizes)}")
        print(f"coords: {list(ds.coords)}")
        print(f"vars:   {list(ds.data_vars)}")

        for name in ("time", "step", "valid_time", "number", "latitude", "longitude"):
            if name in ds.coords:
                values = ds[name].values
                flat = values.reshape(-1) if values.ndim else [values]
                head = flat[:4]
                print(f"  {name}: shape={values.shape} first={head}")
                if name in ("latitude", "longitude") and len(flat) > 1:
                    print(f"    range {flat.min()} .. {flat.max()}, n={len(flat)}")

        for var in ds.data_vars:
            attrs = ds[var].attrs
            print(
                f"  var {var!r}: {attrs.get('long_name', '?')} "
                f"[{attrs.get('units', '?')}] dims={ds[var].dims}"
            )
        print()

    _sample_at_buoy(datasets)


def _sample_at_buoy(datasets) -> None:
    """Pull the value nearest NDBC 46041 to confirm units and magnitude."""
    from src.config import NDBC_STATIONS

    station = NDBC_STATIONS["46041"]
    print("=" * 78)
    print(f"SAMPLE AT NDBC 46041 ({station.latitude} N, {station.longitude} E)")
    print("=" * 78)

    for i, ds in enumerate(datasets):
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            continue

        # Archives commonly store longitude as 0-360; the station is -124.7.
        lon = station.longitude
        if float(ds.longitude.min()) >= 0:
            lon = lon % 360

        try:
            point = ds.sel(latitude=station.latitude, longitude=lon, method="nearest")
        except Exception as e:
            print(f"dataset {i}: selection failed: {e}")
            continue

        print(
            f"dataset {i}: nearest grid point "
            f"{float(point.latitude):.3f} N, {float(point.longitude):.3f} E"
        )
        for var in point.data_vars:
            values = point[var].values.reshape(-1)
            finite = values[~(values != values)]  # drop NaN
            if len(finite):
                print(
                    f"  {var}: first={finite[:6]} "
                    f"min={finite.min():.3f} max={finite.max():.3f}"
                )
            else:
                print(f"  {var}: all NaN at this point (likely a land mask)")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="Prefix to start walking from")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--open-sample",
        action="store_true",
        help="Download one data file and describe its contents",
    )
    args = parser.parse_args(argv)

    s3 = make_client()

    print("=" * 78)
    print(f"BUCKET s3://{BUCKET}")
    print("=" * 78)

    try:
        first_file = walk(s3, args.prefix, max_depth=args.max_depth)
    except Exception as e:
        print(f"Listing failed: {type(e).__name__}: {e}")
        return 1

    if not first_file:
        print("\nNo data file found while walking. Widen --max-depth or set --prefix.")
        return 1

    print(f"\nFirst data file found: {first_file}")

    if not args.open_sample:
        print("\nRe-run with --open-sample to download and inspect it.")
        return 0

    head = s3.head_object(Bucket=BUCKET, Key=first_file)
    size = head["ContentLength"]
    print(f"Size: {size / 1e6:.1f} MB")
    if size > MAX_SAMPLE_BYTES:
        print(f"Larger than the {MAX_SAMPLE_BYTES / 1e6:.0f} MB sample cap, skipping.")
        return 0

    local = Path("/tmp") / Path(first_file).name
    print(f"Downloading to {local}")
    s3.download_file(BUCKET, first_file, str(local))

    try:
        describe_grib(local)
    finally:
        if local.exists():
            os.remove(local)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
