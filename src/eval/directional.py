"""Is a forecast model's error structured by wave direction?

Track B found that postprocessing GEFS helps at NDBC 46087 (Neah Bay, +14.5%
at +12 h) and does not help at the two open-coast buoys. The explanation
offered was geometric: GEFS runs WAVEWATCH III on a 0.25 degree grid, roughly
28 km, and the entrance to the Strait of Juan de Fuca is narrower than that.
A model that cannot see the coastline cannot get the sheltering right, and
what it gets wrong systematically, a postprocessor can learn.

That story is plausible and, so far, untested. It is also the difference
between a curiosity at one buoy and a rule for where postprocessing pays, so
it is worth testing rather than repeating.

The test rests on one idea: **unresolved geometry leaves a directional
signature.** If the grid cell cannot see the strait, the forecast error should
depend on where the waves are coming from - large when the true sea state is
governed by a coastline the model has smoothed away, small when it is governed
by open ocean the model resolves fine. If instead 46087 is simply a calmer,
easier site, the error should be roughly the same size whatever the direction.

So the discriminating quantity is not the size of the error but its
*organisation by direction*, which is what this module measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import bias, rmse

#: Compass sectors, 45 degrees each, centred on the named direction.
SECTOR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SECTOR_WIDTH_DEG = 360.0 / len(SECTOR_NAMES)

#: Minimum paired samples before a sector's statistics are reported. Sectors
#: with fewer are dropped rather than shown noisy: a 40% bias swing computed
#: from nine observations would look like the very signal being hunted.
MIN_SECTOR_SAMPLES = 200


def direction_sector(degrees: pd.Series) -> pd.Series:
    """Bin compass bearings into eight named sectors.

    North straddles 360/0, so bearings are rotated by half a sector before
    binning. Getting this wrong splits the northerly sector in two and hides
    exactly the kind of structure being looked for.
    """
    valid = degrees.where((degrees >= 0) & (degrees <= 360))
    shifted = (valid + SECTOR_WIDTH_DEG / 2.0) % 360.0
    index = (shifted // SECTOR_WIDTH_DEG).astype("Float64").astype("Int64")
    return pd.Series(
        pd.Categorical.from_codes(
            index.fillna(-1).astype(int), categories=list(SECTOR_NAMES)
        ),
        index=degrees.index,
        name="sector",
    )


def error_by_sector(
    observed: pd.Series,
    forecast: pd.Series,
    direction: pd.Series,
    min_samples: int = MIN_SECTOR_SAMPLES,
) -> pd.DataFrame:
    """Forecast error broken down by the direction the waves came from.

    Args:
        observed: Truth series.
        forecast: Model series on the same index.
        direction: Bearing in degrees on the same index. The *observed*
            direction is the right one to use - the model's own direction is
            subject to the same unresolved geometry and would smear the
            stratification it is meant to expose.
        min_samples: Sectors with fewer paired samples are dropped.

    Returns:
        One row per sector with n, RMSE and bias, indexed by sector name.
    """
    frame = pd.DataFrame(
        {
            "obs": observed,
            "fcst": forecast,
            "sector": direction_sector(direction),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["n", "rmse", "bias"])

    rows = []
    for sector, group in frame.groupby("sector", observed=True):
        if len(group) < min_samples:
            continue
        rows.append(
            {
                "sector": str(sector),
                "n": len(group),
                "rmse": rmse(group["obs"], group["fcst"]),
                "bias": bias(group["obs"], group["fcst"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["n", "rmse", "bias"])
    return pd.DataFrame(rows).set_index("sector")


def directional_structure(by_sector: pd.DataFrame) -> dict:
    """How much of the error is organised by direction rather than uniform.

    Two summaries, because they answer different objections:

    * ``bias_spread_m`` - the range of bias across sectors. A model that is
      uniformly too low has a large mean bias and a small spread; a model that
      is too low from one direction and too high from another has a small mean
      bias and a large spread. Only the second is evidence of unresolved
      geometry, and only the second is something a postprocessor can exploit
      using a direction feature.

    * ``bias_spread_ratio`` - that spread divided by the overall RMSE. This is
      the comparable number across sites. A raw spread is larger wherever the
      seas are larger, so comparing 46087 against an open-coast buoy on raw
      spread alone would mostly measure that 46087 is calmer, which is the
      confound the whole exercise is trying to avoid.

    Returns NaN for both when fewer than two sectors survived the sample-count
    filter, since a spread over one sector is not a spread.
    """
    if len(by_sector) < 2:
        return {
            "n_sectors": len(by_sector),
            "bias_spread_m": float("nan"),
            "bias_spread_ratio": float("nan"),
            "worst_sector": None,
            "best_sector": None,
        }

    spread = float(by_sector["bias"].max() - by_sector["bias"].min())
    weights = by_sector["n"]
    pooled_rmse = float(
        np.sqrt((by_sector["rmse"] ** 2 * weights).sum() / weights.sum())
    )
    return {
        "n_sectors": len(by_sector),
        "bias_spread_m": spread,
        "bias_spread_ratio": spread / pooled_rmse if pooled_rmse else float("nan"),
        "worst_sector": str(by_sector["rmse"].idxmax()),
        "best_sector": str(by_sector["rmse"].idxmin()),
    }
