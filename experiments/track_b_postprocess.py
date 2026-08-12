"""Track B: can a statistical model improve on the physics forecast?

The buoy pilot answered "is there skill in the buoy's own history" - yes, about
20% over the best baseline, peaking at +24 h and exhausted by +72 h. This asks
the harder and more useful question: given WAVEWATCH III's forecast for a
Washington-coast buoy, can a model learn its site-specific errors and beat it?

That is where ML reliably earns its place in operational forecasting. Not by
replacing the physics, but by correcting it.

The comparison
--------------
Skill is scored against **the raw GEFS forecast**, not persistence. Two
deliberately awkward baselines sit alongside it:

  ``raw_nwp``       the physics forecast, untouched - the reference
  ``nwp_debiased``  the forecast minus its mean training error

The second matters. Much of what a gradient-boosted postprocessor achieves is
often just the removal of a constant offset, and without this baseline a
complex model takes the credit for it.

Data
----
Truth is the NDBC record. The forecast is the GEFSv12 reforecast point output.
Their overlap is 2015-2019, and cycles are daily, so at a fixed lead time there
is roughly one sample per day - about 1,800 per horizon at stride 1.

Usage::

    python experiments/track_b_postprocess.py --stations 46041 --stride 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import NDBC_STATIONS, NOAA_LAST_YEAR  # noqa: E402
from src.data.gefs import (  # noqa: E402
    aggregate_ensemble,
    align_forecast_to_issue_time,
    build_ensemble_dataset,
    build_forecast_dataset,
    daily_dates,
)
from src.data.ndbc import load_station  # noqa: E402
from src.eval.backtest import (  # noqa: E402
    aggregate_folds,
    format_mean_std,
    run_rolling_backtest,
)
from src.eval.metrics import bias, rmse  # noqa: E402
from src.features.build import build_feature_frame  # noqa: E402
from src.models.baselines import (  # noqa: E402
    BiasCorrectedNWPForecaster,
    ClimatologyForecaster,
    PersistenceForecaster,
    RawNWPForecaster,
)
from src.models.linear import RidgeForecaster  # noqa: E402

logger = logging.getLogger("track_b")

DEFAULT_HORIZONS = (6, 12, 24, 48, 72)

#: Both must be present for a row to be usable: the buoy value at issue time,
#: so persistence can forecast, and the NWP value, so the reference can. Without
#: the second, rows with no forecast survive and raw_nwp is scored on a
#: different sample than the models it is compared against.
REQUIRED_COLUMNS = ["WVHT", "nwp_wvht"]

#: A deliberately small feature set.
#:
#: The first run handed the postprocessors the full 113-column buoy feature
#: frame against ~300 training rows per fold, and both ridge and LightGBM came
#: out *worse* than the uncorrected forecast at every horizon. That is what
#: overfitting looks like, and it says nothing about whether postprocessing
#: works.
#:
#: Cycles are daily, so samples are scarce by construction: one per cycle per
#: lead. The feature set has to be sized for hundreds of rows, not tens of
#: thousands. These are the quantities with a defensible reason to carry
#: information about the forecast's error - the forecast itself and its sea
#: state, the observed state at issue time, and the season.
POSTPROC_FEATURES = [
    "nwp_wvht",   # the forecast being corrected
    "nwp_spread", # ensemble disagreement: the forecast's own uncertainty
    "nwp_tr",     # its mean period: swell and wind sea fail differently
    "nwp_fp",     # its peak frequency
    "WVHT",       # observed height at issue time
    "APD",        # observed period at issue time
    "WVHT_lag6",  # recent trend: is the sea building or dying
    "WVHT_lag24",
    "WVHT_mean24",
    "doy_sin",    # seasonal bias structure
    "doy_cos",
]

#: LightGBM defaults assume far more data than a daily-cycle archive provides.
#: Shallow trees, a high minimum leaf size and heavy regularisation, or it
#: memorises individual storms out of a few hundred rows.
SMALL_DATA_LGBM = {
    "num_leaves": 7,
    "min_data_in_leaf": 30,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "lambda_l2": 5.0,
}


def build_factories(seed: int = 0) -> dict:
    factories = {
        "persistence": PersistenceForecaster,
        "climatology": ClimatologyForecaster,
        "raw_nwp": RawNWPForecaster,
        "nwp_debiased": BiasCorrectedNWPForecaster,
        "ridge_postproc": lambda: RidgeForecaster(alpha=10.0),
    }
    try:
        from src.models.trees import LightGBMForecaster

        factories["lgbm_postproc"] = lambda: LightGBMForecaster(
            params=SMALL_DATA_LGBM, num_boost_round=300, seed=seed
        )
    except ImportError:
        logger.warning("lightgbm not installed - skipping that rung")
    return factories


def report_raw_forecast_error(observations, forecasts, station, horizons) -> None:
    """Characterise the physics model's error before trying to correct it.

    If the raw forecast has little systematic bias there is less for a
    postprocessor to remove, and that is worth knowing before reading any
    skill number.
    """
    print("\n" + "=" * 78)
    print(f"RAW GEFS ERROR AT {station}, BEFORE ANY CORRECTION")
    print("=" * 78)
    print(f"{'lead':>6}  {'n':>6}  {'RMSE':>7}  {'bias':>7}")

    truth = observations["WVHT"]
    for horizon in horizons:
        forecast = forecasts[forecasts["lead_hours"] == horizon].set_index("valid_time")
        if forecast.empty:
            continue
        column = "hs_mean" if "hs_mean" in forecast.columns else "hs"
        joined = pd.DataFrame({"nwp": forecast[column]}).join(
            truth.rename("obs"), how="inner"
        ).dropna()
        if joined.empty:
            continue
        print(
            f"{horizon:>5}h  {len(joined):>6}  "
            f"{rmse(joined['obs'], joined['nwp']):>7.3f}  "
            f"{bias(joined['obs'], joined['nwp']):>+7.3f}"
        )


def run_station(station: str, args) -> pd.DataFrame:
    """Build the paired dataset for one station and run the comparison."""
    info = NDBC_STATIONS[station]
    logger.info("Station %s (%s)", station, info.name)

    start_year, end_year = (int(y) for y in args.years.split("-"))
    if end_year > NOAA_LAST_YEAR:
        logger.warning(
            "Reforecast ends in %d; clipping the requested window", NOAA_LAST_YEAR
        )
        end_year = NOAA_LAST_YEAR

    logger.info("Loading NDBC observations %d-%d", start_year, end_year)
    observations = load_station(
        station, list(range(start_year, end_year + 1)), cache_dir=args.cache_dir
    )

    dates = daily_dates(f"{start_year}-01-01", f"{end_year}-12-31", args.stride)
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    logger.info(
        "Extracting %d GEFS cycles x %d member(s) (%s), stride %d",
        len(dates), len(members), ", ".join(members), args.stride,
    )
    if len(members) == 1:
        forecasts = build_forecast_dataset(
            dates, [station], member=members[0], workers=args.workers
        )
        forecasts["hs_mean"] = forecasts["hs"]
        forecasts["hs_std"] = 0.0
    else:
        raw = build_ensemble_dataset(
            dates, [station], members=members, workers=args.workers
        )
        # Keep the control's period fields; only height is ensembled here.
        control = raw[raw["member"] == members[0]].drop(columns=["member"])
        aggregated = aggregate_ensemble(raw, "hs")
        forecasts = control.merge(
            aggregated, on=["station", "valid_time", "lead_hours"], how="left"
        )
        logger.info(
            "Ensembled %d members; median members per point: %.0f",
            len(members), forecasts["n_members"].median(),
        )
    if forecasts.empty:
        raise RuntimeError(f"No GEFS cycles could be read for {station}")

    horizons = [int(h) for h in args.horizons.split(",")]
    report_raw_forecast_error(observations, forecasts, station, horizons)

    features = build_feature_frame(observations)
    target = observations["WVHT"]

    records = []
    for horizon in horizons:
        at_lead = forecasts[forecasts["lead_hours"] == horizon].set_index("valid_time")

        # Carry the forecast's own sea-state description, not just its height:
        # a 4 m swell and a 4 m wind sea are different forecasts and fail
        # differently.
        horizon_features = features.copy()
        for source, name in (
            ("hs_mean", "nwp_wvht"),
            ("hs_std", "nwp_spread"),
            ("tr", "nwp_tr"),
            ("fp", "nwp_fp"),
        ):
            if source not in at_lead.columns:
                continue
            horizon_features[name] = align_forecast_to_issue_time(
                at_lead[source].sort_index(), horizon, features.index
            )

        available = [c for c in POSTPROC_FEATURES if c in horizon_features.columns]
        missing = [c for c in POSTPROC_FEATURES if c not in horizon_features.columns]
        if missing:
            logger.warning("Missing postprocessing features: %s", missing)
        horizon_features = horizon_features[available]

        usable = horizon_features["nwp_wvht"].notna().sum()
        logger.info(
            "Horizon %sh: %d rows carry a forecast, %d features",
            horizon, usable, len(available),
        )
        if usable < args.min_samples:
            logger.warning(
                "Horizon %sh: only %d paired rows, below --min-samples %d, skipping",
                horizon, usable, args.min_samples,
            )
            continue

        fold_records = run_rolling_backtest(
            horizon_features,
            target,
            [horizon],
            build_factories,
            n_splits=args.n_splits,
            test_size_days=args.test_size_days,
            reference_factory=RawNWPForecaster,
            required_columns=REQUIRED_COLUMNS,
        )
        records.append(fold_records)

    if not records:
        raise RuntimeError(f"No horizon produced results for {station}")

    combined = pd.concat(records, ignore_index=True)
    combined["station"] = station
    return combined


def report(station: str, records: pd.DataFrame) -> None:
    info = NDBC_STATIONS[station]
    print("\n" + "=" * 78)
    print(f"TRACK B: NDBC {station} ({info.name})   [vs raw GEFS, mean ± std]")
    print("=" * 78)

    print("\nRMSE by horizon (metres, lower is better)")
    print(format_mean_std(records, "rmse").to_string())

    print("\nSkill vs RAW GEFS (fraction of RMSE removed)")
    print(format_mean_std(records, "skill_vs_reference").to_string())

    print("\nStorm RMSE, top 10% of observed sea states (metres)")
    print(format_mean_std(records, "rmse_p90").to_string())

    summary = aggregate_folds(records, "rmse")["mean"].unstack("model")
    if "raw_nwp" not in summary.columns:
        return

    # Report the fold spread alongside the gain. Several of these means are
    # smaller than their own standard deviation across folds, which makes them
    # indistinguishable from zero - and a bare percentage hides that.
    spread = aggregate_folds(records, "skill_vs_reference")

    print("\nDid postprocessing beat the physics model?")
    for horizon in summary.index:
        raw = summary.loc[horizon, "raw_nwp"]
        candidates = [c for c in summary.columns if c != "raw_nwp"]
        best = summary.loc[horizon, candidates].idxmin()
        value = summary.loc[horizon, best]
        gain = 1 - value / raw

        if gain <= 0:
            print(f"  +{horizon:>3}h  nothing beat raw GEFS ({raw:.3f})")
            continue

        try:
            std = spread.loc[(horizon, best), "std"]
        except KeyError:
            std = float("nan")

        if pd.notna(std) and std > 0 and abs(gain) < 2 * std:
            verdict = f"   <- within fold spread (+/-{std:.1%}), not separable from zero"
        elif gain < 0.02:
            verdict = "   <- under 2%, not worth the machinery"
        else:
            verdict = ""
        print(
            f"  +{horizon:>3}h  {best:<16} {value:.3f}  vs  raw_nwp {raw:.3f}   "
            f"{gain:+.1%}{verdict}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stations", default="46041")
    parser.add_argument(
        "--years",
        default="2015-2019",
        help="Overlap of the NDBC records and the reforecast, which ends 2019",
    )
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every Nth cycle. Each cycle is an 8.7 MB download; stride 1 "
        "over 2015-2019 is ~1,800 files and ~16 GB.",
    )
    parser.add_argument(
        "--members",
        default="c00,p01,p02,p03,p04",
        help="Ensemble members. Each is a separate download, so this multiplies "
        "transfer volume. A single member skips ensembling entirely.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size-days", type=int, default=240)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Skip a horizon with fewer paired rows than this",
    )
    parser.add_argument("--cache-dir", default="data/raw/ndbc")
    parser.add_argument("--output-dir", default="results/track_b")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    collected = []

    for station in stations:
        try:
            records = run_station(station, args)
        except Exception as e:
            logger.error("Station %s failed: %s: %s", station, type(e).__name__, e)
            continue
        report(station, records)
        collected.append(records)

    if not collected:
        logger.error("No station produced results")
        return 1

    all_records = pd.concat(collected, ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"track_b_{'-'.join(stations)}.csv"
    all_records.to_csv(path, index=False)
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
