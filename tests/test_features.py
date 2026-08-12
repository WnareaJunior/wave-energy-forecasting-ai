"""Causality tests for feature construction.

Every feature must be computable at forecast issue time. The tests below are
deliberately blunt: they mutate the future and assert the present does not
change. That catches lookahead bugs that eyeballing the code will not.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    add_calendar_features,
    add_direction_features,
    add_lag_features,
    add_physics_features,
    add_rolling_features,
    build_feature_frame,
    make_supervised,
)


@pytest.fixture
def observations():
    index = pd.date_range("2020-01-01", periods=500, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "WVHT": 2 + np.sin(np.arange(500) / 24) + rng.normal(0, 0.1, 500),
            "APD": 7 + rng.normal(0, 0.3, 500),
            "DPD": 10 + rng.normal(0, 0.5, 500),
            "MWD": (270 + rng.normal(0, 20, 500)) % 360,
            "WSPD": 8 + rng.normal(0, 1, 500),
            "WDIR": (260 + rng.normal(0, 25, 500)) % 360,
        },
        index=index,
    )


class TestCausality:
    def test_lag_features_ignore_the_future(self, observations):
        """Change the last row; earlier lag features must be identical."""
        before = add_lag_features(observations, ["WVHT"], lags=(1, 3, 6))

        mutated = observations.copy()
        mutated.iloc[-1, mutated.columns.get_loc("WVHT")] = 99.0
        after = add_lag_features(mutated, ["WVHT"], lags=(1, 3, 6))

        lag_columns = [c for c in before.columns if "_lag" in c]
        pd.testing.assert_frame_equal(
            before[lag_columns].iloc[:-1], after[lag_columns].iloc[:-1]
        )

    def test_rolling_features_exclude_the_current_timestamp(self, observations):
        """closed='left' means the window stops before t, not at t."""
        rolled = add_rolling_features(observations, ["WVHT"], windows=(6,))

        spike = observations.copy()
        spike.iloc[100, spike.columns.get_loc("WVHT")] = 50.0
        rolled_spike = add_rolling_features(spike, ["WVHT"], windows=(6,))

        # The spike at t=100 must not appear in the window at t=100.
        assert rolled["WVHT_mean6"].iloc[100] == pytest.approx(
            rolled_spike["WVHT_mean6"].iloc[100]
        )
        # It must appear at t=101.
        assert rolled["WVHT_mean6"].iloc[101] != pytest.approx(
            rolled_spike["WVHT_mean6"].iloc[101]
        )

    def test_lag_n_is_exactly_n_hours_back(self, observations):
        lagged = add_lag_features(observations, ["WVHT"], lags=(3,))
        assert lagged["WVHT_lag3"].iloc[10] == pytest.approx(
            observations["WVHT"].iloc[7]
        )

    def test_full_feature_frame_ignores_the_future(self, observations):
        before = build_feature_frame(observations)

        mutated = observations.copy()
        mutated.iloc[-1] = mutated.iloc[-1] * 3
        after = build_feature_frame(mutated)

        pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


class TestDirectionEncoding:
    def test_replaces_degrees_with_sin_cos(self, observations):
        encoded = add_direction_features(observations, columns=("MWD",))
        assert "MWD" not in encoded.columns
        assert {"MWD_sin", "MWD_cos"} <= set(encoded.columns)

    def test_wraparound_is_continuous(self):
        """359 degrees and 1 degree must be close in the encoded space."""
        df = pd.DataFrame(
            {"MWD": [359.0, 1.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="1h", tz="UTC"),
        )
        encoded = add_direction_features(df, columns=("MWD",))
        distance = np.hypot(
            encoded["MWD_sin"].iloc[0] - encoded["MWD_sin"].iloc[1],
            encoded["MWD_cos"].iloc[0] - encoded["MWD_cos"].iloc[1],
        )
        assert distance < 0.05

    def test_unit_circle(self, observations):
        encoded = add_direction_features(observations, columns=("MWD",))
        norm = encoded["MWD_sin"] ** 2 + encoded["MWD_cos"] ** 2
        assert np.allclose(norm, 1.0)


class TestPhysicsFeatures:
    def test_adds_power_flux(self, observations):
        out = add_physics_features(observations)
        assert "power_flux" in out.columns
        assert (out["power_flux"] > 0).all()

    def test_power_flux_matches_the_formula(self, observations):
        from src.processing.wave_power_flux import (
            ENERGY_PERIOD_FACTORS,
            calculate_wave_power_flux,
        )

        out = add_physics_features(observations)
        expected = calculate_wave_power_flux(
            observations["WVHT"].iloc[0],
            observations["APD"].iloc[0] * ENERGY_PERIOD_FACTORS["VTM02"],
        )
        assert out["power_flux"].iloc[0] == pytest.approx(expected)

    def test_survives_missing_columns(self):
        df = pd.DataFrame(
            {"WVHT": [1.0, 2.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="1h", tz="UTC"),
        )
        out = add_physics_features(df)  # no APD or DPD present
        assert "WVHT" in out.columns


class TestCalendarFeatures:
    def test_harmonics_are_bounded(self, observations):
        out = add_calendar_features(observations)
        for column in ("doy_sin", "doy_cos", "hour_sin", "hour_cos"):
            assert out[column].between(-1, 1).all()

    def test_december_is_adjacent_to_january(self):
        index = pd.DatetimeIndex(["2020-12-31T00", "2021-01-01T00"], tz="UTC")
        out = add_calendar_features(pd.DataFrame(index=index))
        distance = np.hypot(
            out["doy_sin"].iloc[0] - out["doy_sin"].iloc[1],
            out["doy_cos"].iloc[0] - out["doy_cos"].iloc[1],
        )
        assert distance < 0.05


class TestSupervisedAlignment:
    def test_target_is_shifted_forward(self, observations):
        features = build_feature_frame(observations)
        X, y = make_supervised(features, observations["WVHT"], horizon=6)
        issue_time = X.index[0]
        assert y.iloc[0] == pytest.approx(
            observations["WVHT"].loc[issue_time + pd.Timedelta(hours=6)]
        )

    def test_target_and_required_features_are_complete(self, observations):
        """Other features may hold NaN by design - see TestRealisticMissingness."""
        features = build_feature_frame(observations)
        X, y = make_supervised(features, observations["WVHT"], horizon=24)
        assert not y.isna().any()
        assert not X["WVHT"].isna().any()

    def test_x_and_y_stay_aligned(self, observations):
        features = build_feature_frame(observations)
        X, y = make_supervised(features, observations["WVHT"], horizon=12)
        assert X.index.equals(y.index)

    def test_longer_horizon_yields_fewer_samples(self, observations):
        features = build_feature_frame(observations)
        _, y_short = make_supervised(features, observations["WVHT"], horizon=1)
        _, y_long = make_supervised(features, observations["WVHT"], horizon=72)
        assert len(y_long) < len(y_short)


class TestRealisticMissingness:
    """Regression tests for the gapped-data bug.

    A real NDBC record is ~13% missing. An earlier make_supervised dropped any
    row with a NaN anywhere in the feature set; with ~77 lag features that
    leaves P = 0.87^77 of rows, about one in 65,000. It produced an empty
    training set on real data while passing on 3%-missing synthetic input.
    """

    @pytest.fixture
    def gappy_observations(self):
        """Hourly record with 13% scattered missing, matching NDBC 46041."""
        index = pd.date_range("2020-01-01", periods=4000, freq="1h", tz="UTC")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "WVHT": 2 + np.sin(np.arange(4000) / 100) + rng.normal(0, 0.2, 4000),
                "APD": 7 + rng.normal(0, 0.3, 4000),
                "DPD": 10 + rng.normal(0, 0.5, 4000),
                "MWD": (270 + rng.normal(0, 20, 4000)) % 360,
                "WSPD": 8 + rng.normal(0, 1, 4000),
                "WDIR": (260 + rng.normal(0, 25, 4000)) % 360,
            },
            index=index,
        )
        for column in df.columns:
            df.loc[rng.random(4000) < 0.134, column] = np.nan
        return df

    def test_most_rows_survive_alignment(self, gappy_observations):
        """The headline check: a gapped record must still yield a usable set."""
        features = build_feature_frame(gappy_observations)
        X, y = make_supervised(features, gappy_observations["WVHT"], horizon=24)
        # Both target and issue-time value must be present: ~0.866^2 = 75%.
        assert len(X) > 0.6 * len(gappy_observations)

    def test_target_is_never_nan(self, gappy_observations):
        features = build_feature_frame(gappy_observations)
        _, y = make_supervised(features, gappy_observations["WVHT"], horizon=24)
        assert not y.isna().any()

    def test_required_column_is_never_nan(self, gappy_observations):
        """Persistence needs the issue-time value, so it must be complete."""
        features = build_feature_frame(gappy_observations)
        X, _ = make_supervised(features, gappy_observations["WVHT"], horizon=24)
        assert not X["WVHT"].isna().any()

    def test_nans_are_retained_elsewhere(self, gappy_observations):
        """Lag features keep their NaNs - models handle or impute them."""
        features = build_feature_frame(gappy_observations)
        X, _ = make_supervised(features, gappy_observations["WVHT"], horizon=24)
        assert X.isna().any().any()

    def test_extra_required_columns_are_honoured(self, gappy_observations):
        features = build_feature_frame(gappy_observations)
        X, _ = make_supervised(
            features,
            gappy_observations["WVHT"],
            horizon=24,
            required_columns=["WVHT", "WSPD"],
        )
        assert not X[["WVHT", "WSPD"]].isna().any().any()

    def test_ridge_trains_on_gapped_features(self, gappy_observations):
        """Ridge cannot accept NaN, so its pipeline must impute internally."""
        from src.models.linear import RidgeForecaster

        features = build_feature_frame(gappy_observations)
        X, y = make_supervised(features, gappy_observations["WVHT"], horizon=24)
        predictions = RidgeForecaster().fit(X, y).predict(X)
        assert np.isfinite(predictions).all()

    def test_persistence_works_on_gapped_features(self, gappy_observations):
        from src.models.baselines import PersistenceForecaster

        features = build_feature_frame(gappy_observations)
        X, y = make_supervised(features, gappy_observations["WVHT"], horizon=24)
        predictions = PersistenceForecaster().fit(X, y).predict(X)
        assert np.isfinite(predictions).all()
