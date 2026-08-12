"""Name-to-class lookup so experiments can be driven from config.

Adding a model to the whole pipeline is one import and one entry here. The
Transformer and PINN rungs plug in at exactly this point.
"""

from __future__ import annotations

from src.models.base import Forecaster
from src.models.baselines import (
    BiasCorrectedNWPForecaster,
    ClimatologyForecaster,
    MeanForecaster,
    PersistenceForecaster,
    RawNWPForecaster,
    SeasonalNaiveForecaster,
)
from src.models.linear import RidgeForecaster

_REGISTRY: dict[str, type[Forecaster]] = {
    "mean": MeanForecaster,
    "persistence": PersistenceForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "climatology": ClimatologyForecaster,
    "ridge": RidgeForecaster,
}

#: Models that need an NWP forecast column present in the features. Kept out of
#: the main registry sweep so contract tests do not try to fit them on
#: buoy-only data.
NWP_MODELS: dict[str, type[Forecaster]] = {
    "raw_nwp": RawNWPForecaster,
    "nwp_debiased": BiasCorrectedNWPForecaster,
}
_REGISTRY.update(NWP_MODELS)

# LightGBM lives in requirements-ml.txt, so register it only if importable.
try:
    from src.models.trees import LightGBMForecaster, LightGBMQuantileForecaster

    _REGISTRY["lightgbm"] = LightGBMForecaster
    _REGISTRY["lightgbm_quantile"] = LightGBMQuantileForecaster
except ImportError:  # pragma: no cover
    pass


def available_models() -> list[str]:
    """Registered model names, sorted."""
    return sorted(_REGISTRY)


def get_model(name: str, **kwargs) -> Forecaster:
    """Instantiate a registered forecaster.

    Args:
        name: Registered model name.
        **kwargs: Passed to the model's constructor.

    Raises:
        KeyError: with the list of available names, which is more useful than a
            bare KeyError when a config file has a typo.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return _REGISTRY[name](**kwargs)


def register_model(name: str, cls: type[Forecaster]) -> None:
    """Add a forecaster to the registry (for models defined outside this package)."""
    _REGISTRY[name] = cls
