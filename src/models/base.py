"""The contract every forecaster implements.

The point of this file is that swapping LightGBM for a Transformer should be a
one-line change in an experiment script. Everything a model needs arrives as
``(X, y)`` per horizon; nothing a model does touches data loading, splitting,
scaling, or metrics.

A forecaster is fitted per horizon. Direct multi-horizon (a separate model for
each lead time) rather than recursive (feed predictions back in) because
recursive forecasting compounds its own errors, and at 72 hours that compounding
dominates.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Base class for all wave forecasters.

    Subclasses implement :meth:`fit` and :meth:`predict`. Quantile support is
    optional; :meth:`predict_quantiles` raises by default so a model that cannot
    produce a distribution fails loudly rather than silently returning a point
    forecast three times.
    """

    #: Short identifier used in result tables and the registry.
    name: str = "base"

    #: Set to True by models that need no training data (persistence).
    requires_fit: bool = True

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        """Train on features X and targets y for a single horizon."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecasts for each row of X."""

    def predict_quantiles(self, X: pd.DataFrame, quantiles) -> pd.DataFrame:
        """Quantile forecasts, one column per requested quantile."""
        raise NotImplementedError(f"{self.name} does not produce quantile forecasts")

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> Forecaster:
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
