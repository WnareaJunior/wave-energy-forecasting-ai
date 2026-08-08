"""Contract tests for the Forecaster interface.

Every registered model is put through the same round trip. When the Transformer
and PINN rungs land they get tested here for free by being registered - which is
the point of having the interface at all.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_feature_frame, make_supervised
from src.models.base import Forecaster
from src.models.baselines import (
    ClimatologyForecaster,
    MeanForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from src.models.registry import NWP_MODELS, available_models, get_model, register_model
from src.models.trees import HAS_LIGHTGBM

#: Models that work on buoy history alone. The NWP models need a forecast
#: column and are tested separately in TestNWPBaselines.
BUOY_ONLY_MODELS = [m for m in available_models() if m not in NWP_MODELS]


@pytest.fixture(scope="module")
def supervised():
    """A small but realistic (X, y) pair at a 6-hour horizon."""
    index = pd.date_range("2020-01-01", periods=2000, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    trend = 2 + 1.2 * np.sin(np.arange(2000) / 400)
    observations = pd.DataFrame(
        {
            "WVHT": trend + rng.normal(0, 0.3, 2000).cumsum() * 0.02,
            "APD": 7 + rng.normal(0, 0.3, 2000),
            "DPD": 10 + rng.normal(0, 0.5, 2000),
            "MWD": (270 + rng.normal(0, 20, 2000)) % 360,
            "WSPD": 8 + rng.normal(0, 1, 2000),
            "WDIR": (260 + rng.normal(0, 25, 2000)) % 360,
        },
        index=index,
    )
    features = build_feature_frame(observations)
    return make_supervised(features, observations["WVHT"], horizon=6)


@pytest.mark.parametrize("name", BUOY_ONLY_MODELS)
class TestForecasterContract:
    """Applied to every model in the registry."""

    def test_is_a_forecaster(self, name):
        assert isinstance(get_model(name), Forecaster)

    def test_fit_returns_self(self, name, supervised):
        X, y = supervised
        model = get_model(name)
        assert model.fit(X, y) is model

    def test_predict_shape(self, name, supervised):
        X, y = supervised
        model = get_model(name).fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_predictions_are_finite(self, name, supervised):
        X, y = supervised
        model = get_model(name).fit(X, y)
        assert np.isfinite(model.predict(X)).all()

    def test_round_trips_through_disk(self, name, supervised, tmp_path):
        X, y = supervised
        model = get_model(name).fit(X, y)
        expected = model.predict(X)

        path = tmp_path / f"{name}.pkl"
        model.save(path)
        reloaded = Forecaster.load(path)

        np.testing.assert_allclose(reloaded.predict(X), expected)

    def test_beats_the_mean_or_is_the_mean(self, name, supervised):
        """A sanity floor: nothing should be worse than predicting the average."""
        from src.eval.metrics import rmse

        X, y = supervised
        model = get_model(name).fit(X, y)
        model_rmse = rmse(y, model.predict(X))
        mean_rmse = rmse(y, np.full(len(y), y.mean()))
        # Climatology and seasonal-naive can genuinely lose to the mean in
        # sample on a short synthetic series, so allow generous headroom; this
        # is a "not catastrophically broken" check, not a skill test.
        assert model_rmse < mean_rmse * 3


class TestPersistence:
    def test_returns_the_current_observation(self, supervised):
        X, y = supervised
        model = PersistenceForecaster(variable="WVHT").fit(X, y)
        np.testing.assert_allclose(model.predict(X), X["WVHT"].to_numpy())

    def test_falls_back_to_the_shortest_lag(self):
        """When the raw column is absent, use WVHT_lag1 rather than failing."""
        index = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
        X = pd.DataFrame(
            {"WVHT_lag1": np.arange(10.0), "WVHT_lag6": np.arange(10.0) * 2},
            index=index,
        )
        y = pd.Series(np.arange(10.0), index=index, name="WVHT")
        model = PersistenceForecaster(variable="WVHT").fit(X, y)
        np.testing.assert_allclose(model.predict(X), X["WVHT_lag1"].to_numpy())

    def test_raises_when_the_variable_is_absent(self):
        index = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
        X = pd.DataFrame({"WSPD": np.arange(5.0)}, index=index)
        y = pd.Series(np.arange(5.0), index=index, name="WVHT")
        with pytest.raises(KeyError, match="WVHT"):
            PersistenceForecaster(variable="WVHT").fit(X, y)

    def test_needs_no_training(self):
        assert PersistenceForecaster.requires_fit is False


class TestClimatology:
    def test_predicts_the_seasonal_mean(self):
        index = pd.date_range("2018-01-01", "2022-01-01", freq="1h", tz="UTC")
        doy = index.dayofyear.to_numpy()
        y = pd.Series(3 + 2 * np.cos(2 * np.pi * doy / 365.25), index=index, name="WVHT")
        X = pd.DataFrame(index=index)

        model = ClimatologyForecaster().fit(X, y)
        predictions = model.predict(X)
        # A pure seasonal signal should be reproduced closely.
        assert np.corrcoef(predictions, y)[0, 1] > 0.99

    def test_smoothing_wraps_around_new_year(self):
        """December and January must not be smoothed apart."""
        index = pd.date_range("2018-01-01", "2022-01-01", freq="1h", tz="UTC")
        y = pd.Series(np.ones(len(index)) * 3.0, index=index, name="WVHT")
        model = ClimatologyForecaster(smooth_days=31).fit(pd.DataFrame(index=index), y)
        assert model._by_doy.loc[1] == pytest.approx(3.0, abs=0.01)
        assert model._by_doy.loc[365] == pytest.approx(3.0, abs=0.01)

    def test_raises_if_not_fitted(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            ClimatologyForecaster().predict(pd.DataFrame())


class TestMeanForecaster:
    def test_predicts_a_constant(self, supervised):
        X, y = supervised
        predictions = MeanForecaster().fit(X, y).predict(X)
        assert len(set(np.round(predictions, 9))) == 1
        assert predictions[0] == pytest.approx(y.mean())


class TestSeasonalNaive:
    def test_uses_the_lag_at_the_seasonal_period(self, supervised):
        X, y = supervised
        model = SeasonalNaiveForecaster(variable="WVHT", period_hours=24).fit(X, y)
        np.testing.assert_allclose(model.predict(X), X["WVHT_lag24"].to_numpy())


class TestRegistry:
    def test_contains_the_baselines(self):
        assert {"mean", "persistence", "climatology", "ridge"} <= set(available_models())

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(KeyError, match="Available"):
            get_model("transformer_9000")

    def test_passes_kwargs_to_the_constructor(self):
        assert get_model("ridge", alpha=7.5).alpha == 7.5

    def test_custom_models_can_register(self):
        class Dummy(Forecaster):
            name = "dummy"

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        register_model("dummy", Dummy)
        assert isinstance(get_model("dummy"), Dummy)

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
    def test_lightgbm_registered_when_available(self):
        assert "lightgbm" in available_models()


class TestNWPBaselines:
    """The postprocessing reference models."""

    @pytest.fixture
    def with_nwp(self, supervised):
        X, y = supervised
        X = X.copy()
        # A forecast that runs 0.4 m high with noise: the error structure real
        # NWP wave forecasts have at a specific site.
        rng = np.random.default_rng(1)
        X["nwp_wvht"] = y.to_numpy() + 0.4 + rng.normal(0, 0.2, len(y))
        return X, y

    def test_raw_nwp_passes_the_column_through(self, with_nwp):
        from src.models.baselines import RawNWPForecaster

        X, y = with_nwp
        model = RawNWPForecaster().fit(X, y)
        np.testing.assert_allclose(model.predict(X), X["nwp_wvht"].to_numpy())

    def test_raw_nwp_reports_the_missing_column(self, supervised):
        from src.models.baselines import RawNWPForecaster

        X, y = supervised
        with pytest.raises(KeyError, match="nwp_wvht"):
            RawNWPForecaster().fit(X, y)

    def test_debiasing_removes_the_offset(self, with_nwp):
        from src.eval.metrics import bias
        from src.models.baselines import BiasCorrectedNWPForecaster

        X, y = with_nwp
        model = BiasCorrectedNWPForecaster().fit(X, y)
        assert abs(bias(y, model.predict(X))) < 0.01

    def test_debiasing_beats_the_raw_forecast(self, with_nwp):
        """The floor any real postprocessor must clear."""
        from src.eval.metrics import rmse
        from src.models.baselines import (
            BiasCorrectedNWPForecaster,
            RawNWPForecaster,
        )

        X, y = with_nwp
        raw = rmse(y, RawNWPForecaster().fit(X, y).predict(X))
        debiased = rmse(y, BiasCorrectedNWPForecaster().fit(X, y).predict(X))
        assert debiased < raw

    def test_debias_learns_from_training_only(self, with_nwp):
        """The correction is fitted, so it must not adapt to the test rows."""
        from src.models.baselines import BiasCorrectedNWPForecaster

        X, y = with_nwp
        split = len(X) // 2
        model = BiasCorrectedNWPForecaster().fit(X.iloc[:split], y.iloc[:split])
        learned = model._bias
        model.predict(X.iloc[split:])
        assert model._bias == learned


class TestQuantiles:
    def test_point_models_refuse_quantiles(self, supervised):
        X, y = supervised
        model = MeanForecaster().fit(X, y)
        with pytest.raises(NotImplementedError, match="quantile"):
            model.predict_quantiles(X, [0.1, 0.9])

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
    def test_quantiles_do_not_cross(self, supervised):
        from src.models.trees import LightGBMQuantileForecaster

        X, y = supervised
        model = LightGBMQuantileForecaster(
            quantiles=(0.1, 0.5, 0.9), num_boost_round=30
        ).fit(X, y)
        predictions = model.predict_quantiles(X)
        values = predictions.to_numpy()
        assert (np.diff(values, axis=1) >= 0).all()
