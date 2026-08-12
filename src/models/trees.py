"""Gradient-boosted tree forecasters.

LightGBM is the bar to beat. On tabular features with a few thousand to a few
hundred thousand rows it usually matches or beats a neural network at a fraction
of the cost, and it handles NaN natively - which matters here, because buoy
sensors fail and the alternative is throwing away the surrounding rows.

LightGBM is an optional dependency (``requirements-ml.txt``). Importing this
module without it works; instantiating the forecaster raises with an
installation hint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised by the import-guard test
    lgb = None
    HAS_LIGHTGBM = False


#: Deliberately conservative. The pilot has O(10k) training rows per horizon,
#: so a deep forest would memorise individual storms.
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


class LightGBMForecaster(Forecaster):
    """Gradient-boosted trees, one model per horizon.

    Args:
        params: LightGBM parameters, merged over :data:`DEFAULT_PARAMS`.
        num_boost_round: Maximum boosting iterations.
        early_stopping_rounds: Stop when validation RMSE has not improved for
            this many rounds. Requires a validation set passed to :meth:`fit`.
        seed: Random seed. Bagging and feature sampling are stochastic, so
            results vary run to run without it.
    """

    name = "lightgbm"

    def __init__(
        self,
        params: dict | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int | None = 50,
        seed: int = 0,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is not installed. Install the ML extras with:\n"
                "    pip install -r requirements-ml.txt"
            )
        self.params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self._booster = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LightGBMForecaster:
        train_set = lgb.Dataset(X, label=y)
        callbacks = []
        valid_sets = []

        if X_val is not None and y_val is not None and len(X_val):
            valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=train_set))
            if self.early_stopping_rounds:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                )

        self._booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets or None,
            callbacks=callbacks or None,
        )
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("LightGBMForecaster must be fitted before predicting")
        return self._booster.predict(X)

    @property
    def feature_importance(self) -> pd.Series:
        """Gain-based importance, largest first."""
        gains = self._booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.feature_names_).sort_values(ascending=False)


class LightGBMQuantileForecaster(Forecaster):
    """One LightGBM model per quantile, for probabilistic forecasts.

    Trains an independent model per quantile with the pinball loss. Independent
    fits can cross (the 10th percentile predicting above the 90th) in sparse
    regions; :meth:`predict_quantiles` sorts each row to enforce monotonicity,
    which is a presentation fix, not a modelling one.
    """

    name = "lightgbm_quantile"

    def __init__(
        self,
        quantiles=(0.1, 0.5, 0.9),
        params: dict | None = None,
        num_boost_round: int = 500,
        seed: int = 0,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is not installed. Install the ML extras with:\n"
                "    pip install -r requirements-ml.txt"
            )
        self.quantiles = tuple(quantiles)
        self.params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
        self.num_boost_round = num_boost_round
        self._boosters: dict[float, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMQuantileForecaster:
        train_set = lgb.Dataset(X, label=y)
        for q in self.quantiles:
            params = {**self.params, "objective": "quantile", "alpha": q}
            self._boosters[q] = lgb.train(
                params, train_set, num_boost_round=self.num_boost_round
            )
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median forecast, or the quantile closest to it."""
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return self._boosters[median_q].predict(X)

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> pd.DataFrame:
        quantiles = quantiles or self.quantiles
        missing = set(quantiles) - set(self._boosters)
        if missing:
            raise ValueError(f"Model was not trained for quantiles {sorted(missing)}")

        predictions = pd.DataFrame(
            {q: self._boosters[q].predict(X) for q in sorted(quantiles)},
            index=X.index,
        )
        # Enforce non-crossing.
        return pd.DataFrame(
            np.sort(predictions.to_numpy(), axis=1),
            index=predictions.index,
            columns=predictions.columns,
        )
