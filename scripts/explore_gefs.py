"""Discover the layout of the NOAA GEFSv12 wave reforecast bucket.

Run this once before writing any extraction code. The archive's key structure is
not documented in a form worth trusting, and the difference between guessing it
and knowing it is the difference between a fetcher that works and one that
silently pulls the wrong field.

The bucket is public, so no credentials are needed - requests are unsigned.

What it reports:
  1. the text of the archive's own description PDF, which sits at the bucket
     root and is the authoritative account of what is stored and how
  2. the prefix tree, skipping the static grid-info branch
  3. the smallest GRIB2 forecast file's contents: variables, dimensions and
     coordinates, and crucially how initialisation time, lead time and ensemble
     member are encoded, since those determine how a forecast is aligned to a
     valid time
  4. the value at the nearest grid point to NDBC 46041, as a sanity check that
     the field is wave height in metres and not something else

Exits non-zero if it fails to describe a forecast file. A discovery run that
learns nothing must not report success - the first version of this script
swallowed the failure, returned 0, and produced a green check next to an empty
result.

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

#: The archive's own documentation, at the bucket root.
DESCRIPTION_KEY = "Description_of_reforecast_data.pdf"

#: Gridded forecast fields. Every daily cycle file bundles the whole 16-day,
#: 3-hourly global run, so these are ~1.75 GB each - far too large to sample,
#: and far more than this project needs.
GRIB_EXTENSIONS = (".grib2", ".grb2", ".grb")

#: Point output. Per the archive's own documentation these are time series of
#: significant wave height, period and direction at 658 buoy positions - which
#: is exactly what a postprocessing dataset needs, already extracted at the
#: points of interest. Preferring these over the gridded fields avoids
#: downloading and interpolating gigabytes to recover a handful of numbers.
POINT_EXTENSIONS = (".nc",)

#: Sub-prefixes to try first at each level: point output before gridded.
PREFERRED_PREFIX_PARTS = ("station", "point")

#: Static ancillary data - bathymetry, coastline distance, shapefiles. Large,
#: and nothing to do with forecasts. Descending into it wastes the walk.
SKIP_PREFIX_PARTS = ("gridinfo", "shapefile")

#: Do not download anything larger than this when sampling a file.
MAX_SAMPLE_BYTES = 200 * 1024 * 1024


def make_client():
    """Anonymous S3 client. The bucket is public; signing would fail."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def should_skip(prefix: str) -> bool:
    lowered = prefix.lower()
    return any(part in lowered for part in SKIP_PREFIX_PARTS)


def list_level(s3, prefix: str, max_keys: int = 12):
    """One level of the key hierarchy: (sub-prefixes, [(key, size), ...])."""
    paginator = s3.get_paginator("list_objects_v2")
    prefixes: list[str] = []
    keys: list[tuple[str, int]] = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        prefixes.extend(p["Prefix"] for p in page.get("CommonPrefixes", []))
        keys.extend((o["Key"], o["Size"]) for o in page.get("Contents", []))
        if len(keys) >= max_keys:
            break

    return prefixes, keys[:max_keys]


def _sort_key(prefix: str) -> tuple:
    """Order sub-prefixes so point output is visited before gridded fields."""
    lowered = prefix.lower()
    preferred = any(part in lowered for part in PREFERRED_PREFIX_PARTS)
    return (0 if preferred else 1, prefix)


def walk(s3, prefix: str = "", depth: int = 0, max_depth: int = 6, found=None) -> list:
    """Print the prefix tree and collect every data file seen.

    Returns a list of (key, size, kind) where kind is "point" or "gridded".

    Two things the earlier versions got wrong. Finding a file used to stop the
    walk, so the first branch visited short-circuited discovery - that is how
    the grid-info branch hid the reforecast data. And the walk descended into
    ``gridded/`` first and stopped, never seeing the ``station/`` sibling that
    holds exactly the data this project wants. It now visits every child at the
    level where the two appear, point output first.
    """
    found = [] if found is None else found
    indent = "  " * depth
    prefixes, keys = list_level(s3, prefix)

    for key, size in keys[:8]:
        if key.endswith(GRIB_EXTENSIONS):
            kind, marker = "gridded", " <-- GRIB (gridded field)"
        elif key.endswith(POINT_EXTENSIONS):
            kind, marker = "point", " <-- NetCDF (point output)"
        else:
            kind, marker = None, ""
        print(f"{indent}[file] {key}  ({size / 1e6:.1f} MB){marker}")
        if kind:
            found.append((key, size, kind))
    if len(keys) > 8:
        print(f"{indent}... and more files at this level")

    if depth >= max_depth:
        if prefixes:
            print(f"{indent}... {len(prefixes)} prefixes below, not descending")
        return found

    # Print every sibling so the naming convention is visible.
    for sub in prefixes[:10]:
        print(f"{indent}{sub}")
    if len(prefixes) > 10:
        print(f"{indent}... and {len(prefixes) - 10} more prefixes at this level")

    descendable = sorted(
        (p for p in prefixes if not should_skip(p)), key=_sort_key
    )
    # Visit two branches per level rather than stopping at the first. Listing
    # is cheap; missing a whole output type is not.
    for sub in descendable[:2]:
        print(f"{indent}--> descending into {sub}")
        walk(s3, sub, depth + 1, max_depth, found)

    skipped = [p for p in prefixes if should_skip(p)]
    if skipped:
        print(f"{indent}(skipping static ancillary data: {', '.join(skipped)})")

    return found


def print_description(s3) -> None:
    """Download and print the archive's own description PDF.

    This is the authoritative account of what the archive contains, and reading
    it costs 0.2 MB.
    """
    print("\n" + "=" * 78)
    print(f"ARCHIVE DOCUMENTATION: {DESCRIPTION_KEY}")
    print("=" * 78)

    local = Path("/tmp") / DESCRIPTION_KEY
    try:
        s3.download_file(BUCKET, DESCRIPTION_KEY, str(local))
    except Exception as e:
        print(f"Could not download: {type(e).__name__}: {e}")
        return

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(local))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                print(f"\n--- page {i + 1} ---")
                print(text)
    except Exception as e:
        print(f"Could not extract text: {type(e).__name__}: {e}")
    finally:
        if local.exists():
            os.remove(local)


def describe_file(path: Path) -> bool:
    """Open a data file and print everything needed to write the extractor.

    Returns True if at least one dataset was opened.
    """
    import xarray as xr

    print("\n" + "=" * 78)
    print(f"CONTENTS OF {path.name}")
    print("=" * 78)

    # Dispatch by extension. The first version always used cfgrib, so a NetCDF
    # sample could not possibly open.
    engine = "cfgrib" if path.suffix in GRIB_EXTENSIONS else None

    datasets = []
    try:
        datasets = [xr.open_dataset(path, engine=engine)]
    except Exception as e:
        print(f"Single-dataset open failed ({type(e).__name__}: {e})")
        if engine == "cfgrib":
            # GRIB2 files often hold messages on several level types, which
            # cfgrib cannot merge into one dataset.
            print("Retrying with cfgrib.open_datasets()\n")
            try:
                import cfgrib

                datasets = cfgrib.open_datasets(str(path))
            except Exception as e2:
                print(f"open_datasets also failed: {type(e2).__name__}: {e2}")
                return False
        else:
            return False

    if not datasets:
        print("No datasets found in file")
        return False

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
                print(f"  {name}: shape={values.shape} first={flat[:4]}")
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
    return True


def _sample_at_buoy(datasets) -> None:
    """Pull the value nearest NDBC 46041 to confirm units and magnitude."""
    from src.config import NDBC_STATIONS

    station = NDBC_STATIONS["46041"]
    print("=" * 78)
    print(f"SAMPLE AT NDBC 46041 ({station.latitude} N, {station.longitude} E)")
    print("=" * 78)

    for i, ds in enumerate(datasets):
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            # Say why rather than skipping in silence. In the point-output
            # files latitude and longitude are data variables with a time
            # dimension, not coordinates, so this branch is taken and an
            # earlier version printed the section header and nothing else.
            available = list(ds.coords)
            print(
                f"dataset {i}: no latitude/longitude coordinates "
                f"(coords are {available}). "
                "If this is a station file, positions are data variables and "
                "stations are selected by name - use "
                "scripts/probe_gefs_stations.py instead."
            )
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
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument(
        "--open-sample",
        action="store_true",
        help="Download the smallest GRIB2 file found and describe its contents",
    )
    parser.add_argument(
        "--skip-docs", action="store_true", help="Skip the description PDF"
    )
    args = parser.parse_args(argv)

    s3 = make_client()

    if not args.skip_docs:
        print_description(s3)

    print("\n" + "=" * 78)
    print(f"BUCKET s3://{BUCKET}")
    print("=" * 78)

    try:
        candidates = walk(s3, args.prefix, max_depth=args.max_depth)
    except Exception as e:
        print(f"Listing failed: {type(e).__name__}: {e}")
        return 1

    if not candidates:
        print(
            "\nFAILED: no data file found while walking. "
            "Increase --max-depth, or set --prefix to a known data path."
        )
        return 1

    point = [c for c in candidates if c[2] == "point"]
    gridded = [c for c in candidates if c[2] == "gridded"]
    print(f"\nFound {len(point)} point-output and {len(gridded)} gridded file(s).")

    # Point output first, then smallest. Gridded cycle files bundle the entire
    # 16-day global run and run to ~1.75 GB, so they are never a viable sample
    # and are not what this project needs anyway.
    usable = point or gridded
    usable.sort(key=lambda item: item[1])
    key, size, kind = usable[0]
    print(f"Sampling the smallest {kind} file:")
    print(f"  {key}  ({size / 1e6:.1f} MB)")

    if not args.open_sample:
        print("\nRe-run with --open-sample to download and inspect it.")
        return 0

    if size > MAX_SAMPLE_BYTES:
        print(
            f"\nFAILED: larger than the {MAX_SAMPLE_BYTES / 1e6:.0f} MB sample cap. "
            "Gridded cycle files bundle the whole 16-day global run; use the "
            "station/ point output instead, or a byte-range read via the .idx "
            "sidecar."
        )
        return 1

    local = Path("/tmp") / Path(key).name
    print(f"Downloading to {local}")
    s3.download_file(BUCKET, key, str(local))

    try:
        described = describe_file(local)
    finally:
        if local.exists():
            os.remove(local)

    if not described:
        print("\nFAILED: could not open the sample file.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
