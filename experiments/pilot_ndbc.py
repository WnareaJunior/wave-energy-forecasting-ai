"""Pilot experiment: how much wave-height forecast skill is there at a buoy?

Runs the full ladder - mean, persistence, seasonal-naive, climatology, ridge,
LightGBM - across lead times from 1 to 72 hours at a Washington-coast NDBC
buoy, and reports skill relative to persistence.

The question it answers is deliberately narrow: **at what lead time does a
statistical model start to beat "the sea will stay as it is", and by how much?**
Everything heavier - the Transformer, the PINN, the gridded pipeline - should be
justified against this table, not against a strawman.

Usage
-----
Real buoy data (needs network access to ndbc.noaa.gov)::

    python experiments/pilot_ndbc.py --station 46041 --years 2015-2023

Synthetic data, to verify the pipeline runs (no network)::

    python experiments/pilot_ndbc.py --synthetic

Synthetic numbers are NOT results. The generator is easier to forecast than the
real ocean; the run only proves the plumbing works.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import NDBC_STATIONS  # noqa: E402
from src.data.splits import temporal_split  # noqa: E402
from src.eval.backtest import run_backtest, skill_summary  # noqa: E402
from src.features.build import build_feature_frame  # noqa: E402
from src.models.baselines import (  # noqa: E402
    BiasCorrectedNWPForecaster,
    ClimatologyForecaster,
    MeanForecaster,
    PersistenceForecaster,
    RawNWPForecaster,
    SeasonalNaiveForecaster,
)
from src.models.linear import RidgeForecaster  # noqa: E402

logger = logging.getLogger("pilot")

DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48, 72)


def build_model_factories(seed: int = 0, with_nwp: bool = False) -> dict:
    """Model ladder for the pilot, cheapest first.

    LightGBM is included only if installed, so the pilot runs with the base
    requirements and gains a rung when the ML extras are present.

    Args:
        seed: Random seed for stochastic models.
        with_nwp: Add the physics-forecast baselines. Requires an ``nwp_wvht``
            column in the feature frame.
    """
    factories = {
        "mean": MeanForecaster,
        "persistence": PersistenceForecaster,
        "seasonal_naive": SeasonalNaiveForecaster,
        "climatology": ClimatologyForecaster,
        "ridge": lambda: RidgeForecaster(alpha=1.0),
    }

    if with_nwp:
        factories["raw_nwp"] = RawNWPForecaster
        factories["nwp_debiased"] = BiasCorrectedNWPForecaster

    try:
        from src.models.trees import LightGBMForecaster

        factories["lightgbm"] = lambda: LightGBMForecaster(seed=seed)
    except ImportError:
        logger.warning(
            "lightgbm not installed - skipping the gradient-boosting rung. "
            "Install with: pip install -r requirements-ml.txt"
        )

    return factories


def load_observations(args) -> pd.DataFrame:
    """Load the buoy record, real or synthetic."""
    if args.synthetic:
        from src.data.synthetic import generate_buoy_record

        logger.info("Generating synthetic record (NOT real data)")
        return generate_buoy_record(
            start=f"{args.years[0]}-01-01", end=f"{args.years[-1] + 1}-01-01", seed=args.seed
        )

    from src.data.ndbc import coverage_report, load_station

    station = NDBC_STATIONS.get(args.station)
    logger.info(
        "Loading NDBC %s (%s), years %s-%s",
        args.station,
        station.name if station else "unknown station",
        args.years[0],
        args.years[-1],
    )
    df = load_station(args.station, list(args.years), cache_dir=args.cache_dir)
    logger.info("Coverage:\n%s", coverage_report(df).to_string())
    return df


def load_nwp(args, observations, horizons) -> pd.DataFrame | None:
    """Load physics-model forecasts to postprocess, if any were requested.

    Returns:
        DataFrame indexed by valid time with one column per horizon (``h1``,
        ``h3``, ...), or None when running buoy-only.
    """
    if args.nwp_csv is None:
        return None

    if args.nwp_csv == "synthetic":
        if not args.synthetic:
            raise SystemExit("--nwp-csv synthetic requires --synthetic")
        from src.data.synthetic import generate_nwp_forecast

        logger.info("Generating synthetic NWP forecasts (NOT real data)")
        return pd.DataFrame(
            {
                f"h{h}": generate_nwp_forecast(observations, h, seed=args.seed)
                for h in horizons
            }
        )

    logger.info("Loading NWP forecasts from %s", args.nwp_csv)
    nwp = pd.read_csv(args.nwp_csv, parse_dates=["time"]).set_index("time")
    if nwp.index.tz is None:
        nwp.index = nwp.index.tz_localize("UTC")

    missing = [f"h{h}" for h in horizons if f"h{h}" not in nwp.columns]
    if missing:
        raise SystemExit(f"{args.nwp_csv} is missing columns: {missing}")
    return nwp


def run_backtest_with_nwp(
    features, target, nwp, horizons, factories, split, gap_hours
) -> pd.DataFrame:
    """Backtest where each horizon gets its own NWP forecast column.

    The physics forecast for a +24 h lead is a different number from the one for
    +72 h, so the feature has to be swapped in per horizon rather than joined
    once. Skill is scored against the raw NWP.
    """
    from src.eval.backtest import run_horizon
    from src.eval.metrics import results_table

    records = []
    for horizon in horizons:
        column = f"h{horizon}"
        if column not in nwp.columns:
            logger.warning("No NWP column %s, skipping horizon %sh", column, horizon)
            continue

        # The NWP forecast is valid at target time, so shift it back onto issue
        # time: at issue time t the model knows the forecast for t + horizon.
        forecast_at_issue = nwp[column].shift(-horizon)
        horizon_features = features.assign(nwp_wvht=forecast_at_issue)

        records.extend(
            run_horizon(
                horizon_features,
                target,
                horizon,
                factories,
                split,
                gap_hours,
                reference_factory=RawNWPForecaster,
            )
        )
    return results_table(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station", default="46041", help="NDBC station id")
    parser.add_argument("--years", default="2015-2023", help="Inclusive year range, e.g. 2015-2023")
    parser.add_argument("--target", default="WVHT", help="Column to forecast")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated lead times in hours",
    )
    parser.add_argument("--train-end", default=None, help="Last training timestamp")
    parser.add_argument("--validation-end", default=None, help="Last validation timestamp")
    parser.add_argument("--gap-hours", type=int, default=72, help="Buffer between splits")
    parser.add_argument("--cache-dir", default="data/raw/ndbc")
    parser.add_argument("--output-dir", default="results/pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no network)")
    parser.add_argument(
        "--nwp-csv",
        default=None,
        help=(
            "CSV of physics-model forecasts to postprocess. Needs a 'time' column "
            "(the valid time, UTC) plus one column per horizon named 'h1', 'h3', ... "
            "With --synthetic, pass 'synthetic' to generate one."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    start_year, end_year = (int(y) for y in args.years.split("-"))
    args.years = range(start_year, end_year + 1)
    horizons = [int(h) for h in args.horizons.split(",")]

    observations = load_observations(args)
    if args.target not in observations.columns:
        parser.error(f"Target {args.target!r} not in data. Have: {list(observations.columns)}")

    logger.info("Building features")
    features = build_feature_frame(observations)
    target = observations[args.target]

    # Default to the last full year as test and the one before it as validation.
    train_end = args.train_end or f"{end_year - 2}-12-31"
    validation_end = args.validation_end or f"{end_year - 1}-12-31"
    split = temporal_split(features.index, train_end, validation_end, args.gap_hours)
    logger.info("Split:\n%s", split.summary().to_string())

    if len(split.test) == 0:
        parser.error(
            f"Empty test set. Data ends {features.index.max()}, but the split puts "
            f"test after {validation_end}. Pass --train-end/--validation-end."
        )

    nwp = load_nwp(args, observations, horizons)
    factories = build_model_factories(args.seed, with_nwp=nwp is not None)
    logger.info("Models: %s", ", ".join(factories))

    if nwp is None:
        results = run_backtest(
            features, target, horizons, factories, split, args.gap_hours
        )
    else:
        # Skill is measured against the physics forecast, not persistence: the
        # claim under test is "we improve on the NWP", and persistence would
        # make it look far easier than it is.
        results = run_backtest_with_nwp(
            features, target, nwp, horizons, factories, split, args.gap_hours
        )

    if results.empty:
        logger.error("No results produced")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "synthetic" if args.synthetic else args.station
    results.to_csv(output_dir / f"pilot_{tag}_results.csv")

    print("\n" + "=" * 78)
    if args.synthetic:
        print("SYNTHETIC DATA - these numbers are not results, only a pipeline check")
    else:
        station = NDBC_STATIONS.get(args.station)
        print(f"NDBC {args.station} ({station.name if station else 'unknown'})")
    print("=" * 78)

    reference_name = "raw NWP" if nwp is not None else "persistence"

    print("\nRMSE by horizon (metres, lower is better)")
    print(skill_summary(results, "rmse").round(3).to_string())

    print(f"\nSkill vs {reference_name} (fraction of RMSE removed, higher is better)")
    print(skill_summary(results, "skill_vs_reference").round(3).to_string())

    print("\nStorm RMSE, top 10% of observed sea states (metres)")
    print(skill_summary(results, "rmse_p90").round(3).to_string())

    print("\nBias (metres, positive = forecast runs high)")
    print(skill_summary(results, "bias").round(3).to_string())

    # The reference itself always scores exactly zero, so drop it before
    # picking a winner.
    reference_label = "raw_nwp" if nwp is not None else "persistence"
    best = (
        results["skill_vs_reference"]
        .drop(reference_label, level="model", errors="ignore")
        .groupby("horizon")
        .idxmax()
    )
    print(f"\nBest model per horizon (vs {reference_name})")
    for horizon, key in best.dropna().items():
        skill = results.loc[key, "skill_vs_reference"]
        verdict = "" if skill > 0 else "   <- no model beat the reference"
        print(f"  +{horizon:>3}h  {key[1]:<16} {skill:+.1%}{verdict}")

    print(f"\nWritten to {output_dir / f'pilot_{tag}_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
