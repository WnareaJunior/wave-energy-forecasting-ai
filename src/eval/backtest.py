"""Run a set of forecasters across horizons and collect results.

This is the loop the whole pilot exists to run: for each lead time, fit every
model on the same training rows, score them on the same test rows, and report
skill relative to persistence.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.splits import Split, check_no_overlap
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

    if len(train_index) == 0 or len(test_index) == 0:
        logger.warning("Horizon %sh: empty train or test set, skipping", horizon)
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


def skill_summary(results: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Pivot results into a horizon-by-model matrix for one metric."""
    if results.empty:
        return results
    return results[metric].unstack("model")
