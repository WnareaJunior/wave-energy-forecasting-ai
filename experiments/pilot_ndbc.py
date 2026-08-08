"""Pilot experiment: how much wave-height forecast skill is there at a buoy?

Runs the ladder - mean, persistence, seasonal-naive, climatology, ridge,
LightGBM - across lead times from 1 to 72 hours at Washington-coast NDBC buoys,
and reports skill relative to persistence.

The question it answers is deliberately narrow: **at what lead time does a
statistical model start to beat "the sea will stay as it is", and by how much?**
Everything heavier - the Transformer, the PINN, the gridded pipeline - should be
justified against this table, not against a strawman.

Read the climatology row, not just the persistence row. Climatology ignores
current conditions entirely, so the lead time where a model stops beating it is
the lead time where the buoy's own history has stopped carrying information.

Usage
-----
Rolling-origin evaluation across three stations (the default, and the one to
trust - it reports an error bar)::

    python experiments/pilot_ndbc.py --stations 46041,46087,46029

Single fixed split, faster, no error bar::

    python experiments/pilot_ndbc.py --station 46041 --no-rolling

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
from src.eval.backtest import (  # noqa: E402
    aggregate_folds,
    format_mean_std,
    run_backtest,
    run_rolling_backtest,
    skill_summary,
)
from src.features.build import build_feature_frame  # noqa: E402
from src.models.baselines import (  # noqa: E402
    ClimatologyForecaster,
    MeanForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from src.models.linear import RidgeForecaster  # noqa: E402

logger = logging.getLogger("pilot")

DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48, 72)


def build_model_factories(seed: int = 0) -> dict:
    """Model ladder for the pilot, cheapest first.

    LightGBM is included only if installed, so the pilot runs with the base
    requirements and gains a rung when the ML extras are present.
    """
    factories = {
        "mean": MeanForecaster,
        "persistence": PersistenceForecaster,
        "seasonal_naive": SeasonalNaiveForecaster,
        "climatology": ClimatologyForecaster,
        "ridge": lambda: RidgeForecaster(alpha=1.0),
    }

    try:
        from src.models.trees import LightGBMForecaster

        factories["lightgbm"] = lambda: LightGBMForecaster(seed=seed)
    except ImportError:
        logger.warning(
            "lightgbm not installed - skipping the gradient-boosting rung. "
            "Install with: pip install -r requirements-ml.txt"
        )

    return factories


def load_observations(station: str, years, args) -> pd.DataFrame:
    """Load one station's record, real or synthetic."""
    if args.synthetic:
        from src.data.synthetic import generate_buoy_record

        logger.info("Generating synthetic record (NOT real data)")
        return generate_buoy_record(
            start=f"{years[0]}-01-01", end=f"{years[-1] + 1}-01-01", seed=args.seed
        )

    from src.data.ndbc import coverage_report, load_station

    info = NDBC_STATIONS.get(station)
    logger.info(
        "Loading NDBC %s (%s), years %s-%s",
        station,
        info.name if info else "unknown station",
        years[0],
        years[-1],
    )
    df = load_station(station, list(years), cache_dir=args.cache_dir)
    logger.info("Coverage:\n%s", coverage_report(df).to_string())
    return df


def run_station(station: str, years, horizons, args) -> pd.DataFrame:
    """Run the full ladder for one station. Returns long-format records."""
    observations = load_observations(station, years, args)
    if args.target not in observations.columns:
        raise SystemExit(
            f"Target {args.target!r} not in data. Have: {list(observations.columns)}"
        )

    logger.info("Building features")
    features = build_feature_frame(observations)
    target = observations[args.target]

    if args.rolling:
        records = run_rolling_backtest(
            features,
            target,
            horizons,
            build_model_factories,
            n_splits=args.n_splits,
            test_size_days=args.test_size_days,
            gap_hours=args.gap_hours,
        )
    else:
        end_year = years[-1]
        train_end = args.train_end or f"{end_year - 2}-12-31"
        validation_end = args.validation_end or f"{end_year - 1}-12-31"
        split = temporal_split(features.index, train_end, validation_end, args.gap_hours)
        logger.info("Split:\n%s", split.summary().to_string())

        if len(split.test) == 0:
            raise SystemExit(
                f"Empty test set. Data ends {features.index.max()}, but the split "
                f"puts test after {validation_end}."
            )
        table = run_backtest(
            features, target, horizons,
            build_model_factories(args.seed), split, args.gap_hours,
        )
        records = table.reset_index() if not table.empty else pd.DataFrame()

    if not records.empty:
        records["station"] = station
    return records


def report_station(station: str, records: pd.DataFrame, rolling: bool) -> None:
    """Print the result tables for one station."""
    info = NDBC_STATIONS.get(station)
    label = f"NDBC {station} ({info.name if info else 'unknown'})"

    print("\n" + "=" * 78)
    print(label + ("   [rolling-origin, mean ± std across folds]" if rolling else ""))
    print("=" * 78)

    if rolling:
        print("\nRMSE by horizon (metres, lower is better)")
        print(format_mean_std(records, "rmse").to_string())
        print("\nSkill vs persistence (fraction of RMSE removed, higher is better)")
        print(format_mean_std(records, "skill_vs_reference").to_string())
        print("\nStorm RMSE, top 10% of observed sea states (metres)")
        print(format_mean_std(records, "rmse_p90").to_string())
        summary = aggregate_folds(records, "rmse")["mean"].unstack("model")
    else:
        indexed = records.set_index(["horizon", "model"])
        print("\nRMSE by horizon (metres, lower is better)")
        print(skill_summary(indexed, "rmse").round(3).to_string())
        print("\nSkill vs persistence (fraction of RMSE removed, higher is better)")
        print(skill_summary(indexed, "skill_vs_reference").round(3).to_string())
        print("\nStorm RMSE, top 10% of observed sea states (metres)")
        print(skill_summary(indexed, "rmse_p90").round(3).to_string())
        summary = skill_summary(indexed, "rmse")

    # Compare against the best baseline, not just persistence. At long leads
    # climatology becomes competitive, and a model that only beats persistence
    # there has not learned anything a seasonal average does not already know.
    baselines = [c for c in ("persistence", "climatology", "seasonal_naive", "mean")
                 if c in summary.columns]
    candidates = [c for c in summary.columns if c not in baselines]
    if not candidates:
        return

    print("\nBest model vs best baseline, per horizon")
    for horizon in summary.index:
        best_baseline = summary.loc[horizon, baselines].idxmin()
        baseline_rmse = summary.loc[horizon, best_baseline]
        best_model = summary.loc[horizon, candidates].idxmin()
        model_rmse = summary.loc[horizon, best_model]
        gain = 1 - model_rmse / baseline_rmse
        verdict = "" if gain > 0.02 else "   <- no real gain over the baseline"
        print(
            f"  +{horizon:>3}h  {best_model:<10} {model_rmse:.3f}  vs  "
            f"{best_baseline:<14} {baseline_rmse:.3f}   {gain:+.1%}{verdict}"
        )


def report_cross_station(all_records: pd.DataFrame, rolling: bool) -> None:
    """Compare stations side by side, to see whether skill replicates."""
    stations = all_records["station"].unique()
    if len(stations) < 2:
        return

    print("\n" + "=" * 78)
    print("CROSS-STATION: ridge skill vs persistence")
    print("=" * 78)
    print("Skill that does not replicate across stations is not a property of the coast.\n")

    ridge = all_records[all_records["model"] == "ridge"]
    if ridge.empty:
        return
    table = ridge.groupby(["horizon", "station"])["skill_vs_reference"].mean().unstack("station")
    print(table.round(3).to_string())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--station", default=None, help="Single NDBC station id")
    parser.add_argument(
        "--stations",
        default=None,
        help="Comma-separated station ids, e.g. 46041,46087,46029",
    )
    parser.add_argument("--years", default="2015-2023", help="Inclusive year range")
    parser.add_argument("--target", default="WVHT", help="Column to forecast")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated lead times in hours",
    )
    parser.add_argument(
        "--no-rolling",
        dest="rolling",
        action="store_false",
        help="Use a single fixed split instead of rolling-origin folds",
    )
    parser.add_argument("--n-splits", type=int, default=4, help="Rolling-origin folds")
    parser.add_argument(
        "--test-size-days", type=int, default=180, help="Length of each test window"
    )
    parser.add_argument("--train-end", default=None, help="Fixed-split training cutoff")
    parser.add_argument("--validation-end", default=None, help="Fixed-split validation cutoff")
    parser.add_argument("--gap-hours", type=int, default=72, help="Buffer between splits")
    parser.add_argument("--cache-dir", default="data/raw/ndbc")
    parser.add_argument("--output-dir", default="results/pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--log-level", default="INFO")
    parser.set_defaults(rolling=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    if args.stations:
        stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    else:
        stations = [args.station or "46041"]

    start_year, end_year = (int(y) for y in args.years.split("-"))
    years = range(start_year, end_year + 1)
    horizons = [int(h) for h in args.horizons.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    for station in stations:
        try:
            records = run_station(station, years, horizons, args)
        except Exception as e:
            # One bad station must not lose the results for the others.
            logger.error("Station %s failed: %s", station, e)
            continue

        if records.empty:
            logger.error("Station %s produced no results", station)
            continue

        report_station(station, records, args.rolling)
        collected.append(records)

    if not collected:
        logger.error("No results produced for any station")
        return 1

    all_records = pd.concat(collected, ignore_index=True)
    report_cross_station(all_records, args.rolling)

    tag = "synthetic" if args.synthetic else "-".join(stations)
    path = output_dir / f"pilot_{tag}_results.csv"
    all_records.to_csv(path, index=False)
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
