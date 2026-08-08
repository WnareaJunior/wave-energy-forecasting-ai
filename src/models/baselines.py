"""Baseline forecasters.

These exist to make "our model works" a falsifiable claim. Persistence in
particular is far stronger at short lead times than people expect - at +1 hour
it is nearly unbeatable, and a model that only reports aggregate RMSE across all
horizons can hide behind that.

The convention throughout: a feature column named ``WVHT_lag0`` or ``WVHT``
holds the observation at issue time. Persistence returns it unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster


def _current_value_column(X: pd.DataFrame, variable: str) -> str:
    """Find the column holding `variable` at issue time."""
    for candidate in (variable, f"{variable}_lag0"):
        if candidate in X.columns:
            return candidate
    # Fall back to the shortest available lag.
    lag_columns = [c for c in X.columns if c.startswith(f"{variable}_lag")]
    if not lag_columns:
        raise KeyError(
            f"No column for {variable} at issue time. Looked for {variable!r}, "
            f"{variable}_lag0, and {variable}_lag*."
        )
    return min(lag_columns, key=lambda c: int(c.rsplit("lag", 1)[1]))


class PersistenceForecaster(Forecaster):
    """Tomorrow looks like today: forecast the last observed value.

    The reference every other model is scored against. Costs nothing, has no
    parameters, and beats a badly-built ML model at short lead times.
    """

    name = "persistence"
    requires_fit = False

    def __init__(self, variable: str = "WVHT"):
        self.variable = variable
        self._column: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PersistenceForecaster:
        self._column = _current_value_column(X, self.variable)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        column = self._column or _current_value_column(X, self.variable)
        return X[column].to_numpy()


class ClimatologyForecaster(Forecaster):
    """Predict the historical average for this time of year.

    Ignores current conditions entirely, so it is hopeless at short lead times
    and surprisingly competitive at long ones - which is exactly the point.
    Where climatology overtakes persistence marks the lead time past which
    current conditions stop carrying information, and that crossover is the
    interesting region for a real model.

    The day-of-year mean is smoothed with a circular rolling window so that a
    single stormy week in the training years does not create a spike.
    """

    name = "climatology"

    def __init__(self, smooth_days: int = 15):
        self.smooth_days = smooth_days
        self._by_doy: pd.Series | None = None
        self._global_mean: float = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ClimatologyForecaster:
        doy = pd.Index(y.index).dayofyear
        by_doy = y.groupby(doy).mean().reindex(range(1, 367))

        # Wrap the series so the smoother sees December next to January.
        padded = pd.concat([by_doy, by_doy, by_doy])
        smoothed = padded.rolling(self.smooth_days, center=True, min_periods=1).mean()
        self._by_doy = smoothed.iloc[len(by_doy) : 2 * len(by_doy)]
        self._by_doy.index = by_doy.index

        self._global_mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._by_doy is None:
            raise RuntimeError("ClimatologyForecaster must be fitted before predicting")
        doy = pd.Index(X.index).dayofyear
        return self._by_doy.reindex(doy).fillna(self._global_mean).to_numpy()


class SeasonalNaiveForecaster(Forecaster):
    """Predict the value from one seasonal period ago (default 24 hours).

    Captures the diurnal cycle. Weak for waves - unlike temperature, wave height
    has little daily rhythm - but cheap to include and it makes the absence of a
    diurnal signal visible in the results table rather than assumed.

    Where the seasonal lag is missing - buoy records are gapped, so a value
    exactly 24 hours back often is - it falls back to the issue-time value.
    Returning NaN instead would quietly shrink this model's sample and make its
    RMSE incomparable with models that predicted everywhere.
    """

    name = "seasonal_naive"
    requires_fit = False

    def __init__(self, variable: str = "WVHT", period_hours: int = 24):
        self.variable = variable
        self.period_hours = period_hours

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaiveForecaster:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        column = f"{self.variable}_lag{self.period_hours}"
        if column not in X.columns:
            return X[_current_value_column(X, self.variable)].to_numpy()

        seasonal = X[column]
        fallback = X[_current_value_column(X, self.variable)]
        return seasonal.fillna(fallback).to_numpy()


class RawNWPForecaster(Forecaster):
    """Pass through a numerical weather prediction forecast unchanged.

    This is the *real* traditional alternative. WAVEWATCH III and its ensemble
    cousins encode decades of wave physics; a statistical model that cannot beat
    the raw forecast has not earned its place in the pipeline.

    Reporting skill against this rather than against persistence is what makes
    the postprocessing claim falsifiable. It is also a much harder bar.

    Args:
        column: Feature column holding the NWP forecast valid at the target
            time. It must be a genuine forecast issued at or before issue time -
            using a reanalysis here would be leakage dressed up as a baseline.
    """

    name = "raw_nwp"
    requires_fit = False

    def __init__(self, column: str = "nwp_wvht"):
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RawNWPForecaster:
        if self.column not in X.columns:
            raise KeyError(
                f"NWP column {self.column!r} not in features. Available: "
                f"{[c for c in X.columns if 'nwp' in c.lower()]}"
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy()


class BiasCorrectedNWPForecaster(Forecaster):
    """NWP forecast minus its mean training error.

    The simplest postprocessing there is, and a deliberately awkward baseline:
    much of what a gradient-boosted postprocessor achieves is often just this.
    Including it stops a complex model from taking credit for removing a
    constant offset.
    """

    name = "nwp_debiased"

    def __init__(self, column: str = "nwp_wvht"):
        self.column = column
        self._bias: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BiasCorrectedNWPForecaster:
        if self.column not in X.columns:
            raise KeyError(f"NWP column {self.column!r} not in features")
        self._bias = float(np.nanmean(X[self.column].to_numpy() - y.to_numpy()))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy() - self._bias


class MeanForecaster(Forecaster):
    """Predict the training mean. The floor - any model below this is broken."""

    name = "mean"

    def __init__(self):
        self._mean: float = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MeanForecaster:
        self._mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)
