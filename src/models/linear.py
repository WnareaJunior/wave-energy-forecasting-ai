"""Linear forecasters.

Ridge on lag features is the honest bridge between persistence and a tree
ensemble: it can weight several lags and the seasonal harmonics, but it cannot
represent interactions. If LightGBM does not beat ridge by much, the signal in
this problem is mostly linear and the deep-learning rungs are unlikely to pay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import Forecaster


class RidgeForecaster(Forecaster):
    """Ridge regression on standardised features.

    Imputation and scaling are inside the pipeline, so both are fitted on
    training data only and travel with the model when it is pickled. Fitting
    either on the full dataset before splitting is the most common way test-set
    information leaks into training.

    The imputer is not optional: buoy records are heavily gapped, and
    :func:`~src.features.build.make_supervised` deliberately leaves NaNs in the
    non-required features rather than throwing away most of the dataset.
    Median imputation is used because wave variables are right-skewed, so the
    mean sits above the typical value.
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeForecaster:
        self._pipeline.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(X)

    @property
    def coefficients(self) -> pd.Series:
        """Fitted coefficients, largest absolute value first.

        On standardised features these are directly comparable, so this doubles
        as a feature-importance readout.
        """
        coef = pd.Series(
            self._pipeline.named_steps["ridge"].coef_, index=self.feature_names_
        )
        return coef.reindex(coef.abs().sort_values(ascending=False).index)
