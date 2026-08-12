"""Forecast verification metrics.

Two conventions here that matter more than the formulas:

* **Everything is reported per horizon.** A single aggregate RMSE across lead
  times is dominated by the easy short-range cases and can hide a model that is
  useless at 48 hours.
* **Skill is relative.** Absolute RMSE in metres means little on its own. Skill
  score against a named reference - persistence, or the raw physics forecast -
  is what tells you whether a model earned its complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    """Root mean squared error. Penalises large misses, so storm errors dominate."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error. Less sensitive to outliers than RMSE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.nanmean(np.abs(y_true - y_pred)))


def bias(y_true, y_pred) -> float:
    """Mean error. Positive means the forecast runs high.

    A systematic bias is the easiest thing to correct and the most common thing
    a physics model gets wrong at a specific site - which is the entire premise
    of the postprocessing approach.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.nanmean(y_pred - y_true))


def skill_score(y_true, y_pred, y_reference) -> float:
    """Fractional RMSE improvement over a reference forecast.

    ``1 - RMSE_model / RMSE_reference``: 0 means no better than the reference,
    0.1 means 10% lower error, negative means worse.

    Returns NaN when the reference is perfect, which happens only on degenerate
    data.
    """
    reference_rmse = rmse(y_true, y_reference)
    if reference_rmse == 0:
        return float("nan")
    return float(1 - rmse(y_true, y_pred) / reference_rmse)


def correlation(y_true, y_pred) -> float:
    """Pearson correlation. High correlation with large bias is a calibration
    problem, not a signal problem - worth distinguishing.

    Returns NaN when either series is constant. Correlation is genuinely
    undefined there, and the constant case is not exotic: the mean forecaster
    predicts one value by construction, so letting numpy divide by a zero
    standard deviation would emit a RuntimeWarning on every single fold.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return float("nan")
    if np.std(y_true[mask]) == 0 or np.std(y_pred[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])


def storm_rmse(y_true, y_pred, percentile: float = 90) -> float:
    """RMSE restricted to the highest observed sea states.

    For a wave energy converter this is the number that matters. Most of the
    annual energy arrives in a handful of big winter events, and survival
    loading is set by the extremes. A model can look excellent overall by
    nailing calm summer conditions and still be useless for the decisions the
    forecast supports.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float("nan")
    threshold = np.percentile(y_true[mask], percentile)
    storm = mask & (y_true >= threshold)
    if storm.sum() == 0:
        return float("nan")
    return rmse(y_true[storm], y_pred[storm])


def evaluate(
    y_true,
    y_pred,
    y_reference=None,
    storm_percentile: float = 90,
) -> dict[str, float]:
    """All point-forecast metrics for one model at one horizon.

    Args:
        y_true: Observations.
        y_pred: Model forecasts.
        y_reference: Reference forecast for the skill score, usually
            persistence. Omit to skip the skill column.
        storm_percentile: Percentile defining "storm" conditions.

    Returns:
        dict of metric name to value.
    """
    # n counts pairs where both the observation and the forecast are present.
    # Counting only valid observations would hide a model that silently failed
    # to predict on part of the test set, making its RMSE look comparable to a
    # model that predicted everywhere.
    valid = ~(
        np.isnan(np.asarray(y_true, float)) | np.isnan(np.asarray(y_pred, float))
    )

    results = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
        f"rmse_p{int(storm_percentile)}": storm_rmse(y_true, y_pred, storm_percentile),
        "n": int(valid.sum()),
    }
    if y_reference is not None:
        results["skill_vs_reference"] = skill_score(y_true, y_pred, y_reference)
    return results


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Pinball (quantile) loss - the proper scoring rule for one quantile."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    error = y_true - y_pred
    return float(np.nanmean(np.maximum(quantile * error, (quantile - 1) * error)))


def coverage(y_true, lower, upper) -> float:
    """Fraction of observations inside the predicted interval.

    A well-calibrated 80% interval contains 80% of observations. Much lower and
    the model is overconfident, which for a siting or dispatch decision is worse
    than being imprecise.
    """
    y_true = np.asarray(y_true, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float("nan")
    inside = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
    return float(np.mean(inside))


def results_table(records: list[dict]) -> pd.DataFrame:
    """Assemble per-model, per-horizon metric dicts into a sorted table.

    Args:
        records: dicts each containing at least ``model``, ``horizon``, and the
            metric keys from :func:`evaluate`.

    Returns:
        DataFrame indexed by (horizon, model).
    """
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.set_index(["horizon", "model"]).sort_index()
