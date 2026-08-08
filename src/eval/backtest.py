"""Run a set of forecasters across horizons and collect results.

This is the loop the whole pilot exists to run: for each lead time, fit every
model on the same training rows, score them on the same test rows, and report
skill relative to persistence.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.splits import (
    Split,
    carve_validation,
    check_no_overlap,
    rolling_origin_splits,
)
from src.eval.metrics import evaluate, results_table
from src.features.build import make_supervised
from src.models.base import Forecaster
from src.models.baselines import PersistenceForecaster

logger = logging.getLogger(__name__)


def _fit(model: Forecaster, X_train, y_train, X_val, y_val) -> Forecaster:
    """Fit, passing a validation set to models that accept one."""
    try:
        return model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    except TypeError:
        return model.fit(X_train, y_train)


def run_horizon(
    features: pd.DataFrame,
    target: pd.Series,
    horizon: int,
    model_factories: dict,
    split: Split,
    gap_hours: int = 72,
    storm_percentile: float = 90,
    reference_factory=None,
) -> list[dict]:
    """Fit and score every model at a single lead time.

    Args:
        features: Causal feature frame indexed by issue time.
        target: Observed series to forecast.
        horizon: Lead time in hours.
        model_factories: Mapping of label to zero-argument callable returning a
            fresh :class:`~src.models.base.Forecaster`. Factories rather than
            instances so each horizon gets an untrained model.
        split: Train/validation/test partition.
        gap_hours: Passed to the leakage check.
        storm_percentile: Percentile defining storm conditions.
        reference_factory: Callable returning the forecaster that skill is
            measured against. Defaults to persistence. For a postprocessing
            experiment pass ``RawNWPForecaster`` instead - beating the physics
            model is the claim that matters there, and persistence would be
            flattering.

    Returns:
        One metric dict per model.
    """
    X, y = make_supervised(features, target, horizon)

    train_index = X.index.intersection(split.train)
    val_index = X.index.intersection(split.validation)
    test_index = X.index.intersection(split.test)

    # Log the sample counts every time. An empty training set is otherwise
    # indistinguishable from a model failure, and the counts are the first thing
    # worth checking when results look wrong.
    logger.info(
        "Horizon %sh: %d usable rows -> train %d, val %d, test %d",
        horizon, len(X), len(train_index), len(val_index), len(test_index),
    )

    if len(train_index) == 0 or len(test_index) == 0:
        logger.warning(
            "Horizon %sh: empty train or test set, skipping. "
            "%d rows survived alignment out of %d feature rows - if that ratio "
            "is near zero, check how many features are required to be non-NaN.",
            horizon, len(X), len(features),
        )
        return []

    check_no_overlap(train_index, test_index, gap_hours)

    X_train, y_train = X.loc[train_index], y.loc[train_index]
    X_val, y_val = X.loc[val_index], y.loc[val_index]
    X_test, y_test = X.loc[test_index], y.loc[test_index]

    # The reference is always computed, regardless of what the caller asked for
    # in model_factories, so no result table can be published without one.
    if reference_factory is None:
        reference = PersistenceForecaster(variable=str(target.name))
    else:
        reference = reference_factory()
    reference.fit(X_train, y_train)
    y_reference = reference.predict(X_test)

    records = []
    for label, factory in model_factories.items():
        model = factory()
        try:
            _fit(model, X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_test)
        except Exception as e:  # a broken model should not kill the sweep
            logger.error("Horizon %sh: model %s failed: %s", horizon, label, e)
            continue

        metrics = evaluate(y_test, y_pred, y_reference, storm_percentile)
        records.append({"model": label, "horizon": horizon, **metrics})

    return records


def run_backtest(
    features: pd.DataFrame,
    target: pd.Series,
    horizons,
    model_factories: dict,
    split: Split,
    gap_hours: int = 72,
    storm_percentile: float = 90,
    reference_factory=None,
) -> pd.DataFrame:
    """Run :func:`run_horizon` across all lead times.

    Returns:
        Results table indexed by (horizon, model).
    """
    records = []
    for horizon in horizons:
        logger.info("Horizon %sh", horizon)
        records.extend(
            run_horizon(
                features, target, horizon, model_factories, split,
                gap_hours, storm_percentile, reference_factory,
            )
        )
    return results_table(records)


def run_rolling_backtest(
    features: pd.DataFrame,
    target: pd.Series,
    horizons,
    factory_builder,
    n_splits: int = 4,
    test_size_days: int = 90,
    gap_hours: int = 72,
    val_fraction: float = 0.15,
    storm_percentile: float = 90,
    reference_factory=None,
) -> pd.DataFrame:
    """Score every model across several consecutive test windows.

    A single train/validate/test split gives one number whose value depends on
    which months happened to land in the test window. With a few years of data
    that is a real risk, and it offers no way to tell a genuine difference
    between two models from noise. Rolling-origin evaluation scores each model
    on ``n_splits`` successive periods, so the spread across folds becomes the
    error bar.

    Each fold's validation block is carved from the tail of its own training
    block rather than being a fixed calendar year, which is what broke the
    first real run: 46041's 2022 record is almost entirely missing, so
    LightGBM early-stopped against 1,716 rows.

    Args:
        features: Causal feature frame.
        target: Series to forecast.
        horizons: Lead times in hours.
        factory_builder: Callable taking a seed and returning a dict of model
            factories. Called once per fold with the fold index as the seed, so
            stochastic models vary between folds and the fold spread reflects
            seed variance as well as period variance.
        n_splits: Number of folds.
        test_size_days: Length of each test window.
        gap_hours: Buffer between blocks.
        val_fraction: Share of each training block held out for early stopping.
        storm_percentile: Percentile defining storm conditions.
        reference_factory: Forecaster that skill is measured against.

    Returns:
        Long-format DataFrame, one row per (fold, horizon, model).
    """
    records = []

    for fold, (train_index, test_index) in enumerate(
        rolling_origin_splits(features.index, n_splits, test_size_days, gap_hours)
    ):
        inner_train, validation = carve_validation(train_index, val_fraction, gap_hours)
        split = Split(train=inner_train, validation=validation, test=test_index)

        logger.info(
            "Fold %d: train %s -> %s, test %s -> %s",
            fold,
            inner_train.min().date() if len(inner_train) else None,
            inner_train.max().date() if len(inner_train) else None,
            test_index.min().date() if len(test_index) else None,
            test_index.max().date() if len(test_index) else None,
        )

        factories = factory_builder(fold)
        for horizon in horizons:
            fold_records = run_horizon(
                features, target, horizon, factories, split,
                gap_hours, storm_percentile, reference_factory,
            )
            for record in fold_records:
                record["fold"] = fold
            records.extend(fold_records)

    return pd.DataFrame(records)


def aggregate_folds(records: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Mean and standard deviation of one metric across folds.

    The standard deviation is the point of running folds at all: a difference
    between two models that is smaller than their fold-to-fold spread is not a
    difference worth acting on.

    Returns:
        DataFrame indexed by (horizon, model) with ``mean``, ``std``, and
        ``n_folds`` columns.
    """
    if records.empty:
        return records

    grouped = records.groupby(["horizon", "model"])[metric]
    return pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(),
            "n_folds": grouped.count(),
        }
    )


def format_mean_std(records: pd.DataFrame, metric: str = "rmse", decimals: int = 3):
    """Horizon-by-model table of "mean ± std" strings, for printing."""
    if records.empty:
        return records
    stats = aggregate_folds(records, metric)
    formatted = stats.apply(
        lambda row: f"{row['mean']:.{decimals}f} ± {row['std']:.{decimals}f}"
        if pd.notna(row["std"])
        else f"{row['mean']:.{decimals}f}",
        axis=1,
    )
    return formatted.unstack("model")


def skill_summary(results: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Pivot single-split results into a horizon-by-model matrix for one metric."""
    if results.empty:
        return results
    return results[metric].unstack("model")
