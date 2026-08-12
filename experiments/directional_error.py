"""Why does postprocessing work at Neah Bay and nowhere else?

Track B found GEFS postprocessing gives +14.5% at NDBC 46087 and nothing at
the two open-coast buoys. The explanation offered was geometric: WAVEWATCH III
runs on a 0.25 degree grid, about 28 km, and the entrance to the Strait of Juan
de Fuca is narrower than that, so the model cannot represent the sheltering
that governs the real sea state. What a model gets wrong *systematically* is
what a postprocessor can learn.

That was a story attached to a single number. This script tests it.

The prediction
--------------
Unresolved geometry leaves a **directional signature**. If the grid cell cannot
see the strait, forecast error should depend on where the waves come from:
large when the true state is governed by a coastline the model has smoothed
away, small when it is governed by open ocean the model resolves perfectly
well. If instead 46087 is simply a calmer and easier site, error should be the
same size from every direction, just smaller overall.

So the discriminating quantity is not the size of the error but its
**organisation by direction** - reported here as the spread of bias across
compass sectors, divided by pooled RMSE so that a calm site and a rough one can
be compared without the comparison collapsing into "Neah Bay is sheltered".

What would refute it
--------------------
If 46087's bias spread ratio is comparable to the open-coast buoys, the
geometric story is wrong and the +14.5% needs another explanation. That is a
real possible outcome and the reason this is worth running: a confirmation is
worth much less than the chance of a refutation.

Caveats carried into the reading
--------------------------------
* Observed wave direction (MWD) stratifies the error, not the model's own
  direction, which is subject to the same unresolved geometry.
* 46087 sits inside the strait mouth, so its directional distribution is
  narrow by construction. Sectors below the sample floor are dropped, and a
  site surviving in fewer sectors is itself worth noting rather than hiding.
* This measures *association*, not mechanism. A directional signature is
  consistent with unresolved geometry; it does not prove it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import NDBC_STATIONS, NOAA_LAST_YEAR  # noqa: E402
from src.data.gefs import build_forecast_dataset, daily_dates  # noqa: E402
from src.data.ndbc import load_station  # noqa: E402
from src.eval.directional import (  # noqa: E402
    directional_structure,
    error_by_sector,
)
from src.eval.metrics import bias, rmse  # noqa: E402

logger = logging.getLogger("directional")


def paired_frame(station: str, args) -> pd.DataFrame:
    """Observations joined to the raw GEFS forecast at one lead time."""
    start_year, end_year = (int(y) for y in args.years.split("-"))
    end_year = min(end_year, NOAA_LAST_YEAR)

    observations = load_station(
        station, list(range(start_year, end_year + 1)), cache_dir=args.cache_dir
    )
    dates = daily_dates(f"{start_year}-01-01", f"{end_year}-12-31", args.stride)
    forecasts = build_forecast_dataset(
        dates, [station], member=args.member, workers=args.workers
    )

    at_lead = forecasts[forecasts["lead_hours"] == args.horizon]
    at_lead = at_lead.set_index("valid_time")
    if at_lead.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "obs": observations["WVHT"],
            "mwd": observations["MWD"],
            "fcst": at_lead["hs"],
        }
    ).dropna()


def report_station(station: str, frame: pd.DataFrame, horizon: int) -> dict:
    info = NDBC_STATIONS[station]
    print("\n" + "=" * 78)
    print(f"{station}  {info.name}   (+{horizon} h)")
    print("=" * 78)

    if frame.empty:
        print("  no paired samples")
        return {"station": station}

    print(
        f"  overall: n={len(frame)}  RMSE={rmse(frame['obs'], frame['fcst']):.3f} m  "
        f"bias={bias(frame['obs'], frame['fcst']):+.3f} m"
    )

    by_sector = error_by_sector(frame["obs"], frame["fcst"], frame["mwd"])
    if by_sector.empty:
        print("  no sector had enough samples")
        return {"station": station}

    print("\n  error by direction the waves came from:")
    print(by_sector.round(3).to_string())

    structure = directional_structure(by_sector)
    print(
        f"\n  sectors={structure['n_sectors']}  "
        f"bias spread={structure['bias_spread_m']:.3f} m  "
        f"spread/RMSE={structure['bias_spread_ratio']:.3f}  "
        f"worst={structure['worst_sector']}  best={structure['best_sector']}"
    )
    return {"station": station, "name": info.name, "n": len(frame), **structure}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stations", default="46041,46087,46029")
    parser.add_argument("--years", default="2015-2019")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--member", default="c00")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/raw/ndbc")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    summaries = []
    for station in stations:
        try:
            summaries.append(
                report_station(station, paired_frame(station, args), args.horizon)
            )
        except Exception:
            # One station failing must not lose the others' results, but it
            # must also not be mistaken for a station with no signal.
            logger.exception("station %s failed", station)
            summaries.append({"station": station, "failed": True})

    print("\n" + "=" * 78)
    print("COMPARISON: is the error organised by direction?")
    print("=" * 78)
    table = pd.DataFrame(summaries).set_index("station")
    print(table.round(3).to_string())

    usable = table[table.get("bias_spread_ratio").notna()] if "bias_spread_ratio" in table else table.iloc[0:0]
    if len(usable) < 2:
        print("\nToo few stations produced a ratio to compare. No conclusion.")
        return 1

    ranked = usable["bias_spread_ratio"].sort_values(ascending=False)
    print(f"\nMost directionally structured error: {ranked.index[0]}")
    print(
        "The geometric explanation predicts 46087 at the top. It is not at the "
        "top here."
        if ranked.index[0] != "46087"
        else "The geometric explanation predicts 46087 at the top, and it is."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
