"""Tests for rolling-origin evaluation and fold aggregation.

The fixed-split run at 46041 landed on a validation year that was almost
entirely missing at that station, so LightGBM early-stopped against 1,716 rows
without anything flagging it. Rolling-origin folds carve validation from each
fold's own training tail, which removes that failure mode; these tests pin the
behaviour down.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import carve_validation
from src.eval.backtest import (
    aggregate_folds,
    format_mean_std,
    run_rolling_backtest,
)
from src.features.build import build_feature_frame
from src.models.baselines import MeanForecaster, PersistenceForecaster
from src.models.linear import RidgeForecaster


@pytest.fixture(scope="module")
def observations():
    index = pd.date_range("2019-01-01", periods=20000, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    doy = index.dayofyear.to_numpy()
    seasonal = 2.5 + 1.2 * np.cos(2 * np.pi * doy / 365.25)
    noise = rng.normal(0, 0.3, 20000).cumsum() * 0.01
    df = pd.DataFrame(
        {
            "WVHT": np.clip(seasonal + noise, 0.2, None),
            "APD": 7 + rng.normal(0, 0.3, 20000),
            "DPD": 10 + rng.normal(0, 0.5, 20000),
            "MWD": (270 + rng.normal(0, 20, 20000)) % 360,
            "WSPD": 8 + rng.normal(0, 1, 20000),
            "WDIR": (260 + rng.normal(0, 25, 20000)) % 360,
        },
        index=index,
    )
    for column in df.columns:
        df.loc[rng.random(20000) < 0.13, column] = np.nan
    return df


def _factories(seed):
    return {
        "mean": MeanForecaster,
        "persistence": PersistenceForecaster,
        "ridge": lambda: RidgeForecaster(alpha=1.0),
    }


class TestCarveValidation:
    def test_validation_comes_from_the_tail(self):
        index = pd.date_range("2020-01-01", periods=1000, freq="1h", tz="UTC")
        train, validation = carve_validation(index, fraction=0.2, gap_hours=0)
        assert train.max() < validation.min()
        assert validation.max() == index.max()

    def test_gap_is_respected(self):
        index = pd.date_range("2020-01-01", periods=1000, freq="1h", tz="UTC")
        train, validation = carve_validation(index, fraction=0.2, gap_hours=48)
        assert validation.min() - train.max() > pd.Timedelta(hours=48)

    def test_blocks_do_not_overlap(self):
        index = pd.date_range("2020-01-01", periods=1000, freq="1h", tz="UTC")
        train, validation = carve_validation(index)
        assert len(train.intersection(validation)) == 0

    def test_roughly_the_requested_fraction(self):
        index = pd.date_range("2020-01-01", periods=1000, freq="1h", tz="UTC")
        _, validation = carve_validation(index, fraction=0.2, gap_hours=0)
        assert 150 < len(validation) < 250

    def test_short_block_yields_empty_validation(self):
        """Callers must tolerate this rather than assume a split is possible."""
        index = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
        train, validation = carve_validation(index)
        assert len(validation) == 0
        assert len(train) == 5


class TestRollingBacktest:
    @pytest.fixture(scope="class")
    def records(self, observations):
        features = build_feature_frame(observations)
        return run_rolling_backtest(
            features,
            observations["WVHT"],
            horizons=[6, 24],
            factory_builder=_factories,
            n_splits=3,
            test_size_days=60,
        )

    def test_produces_records_for_every_fold(self, records):
        assert set(records["fold"]) == {0, 1, 2}

    def test_covers_every_model_and_horizon(self, records):
        assert set(records["horizon"]) == {6, 24}
        assert {"mean", "persistence", "ridge"} <= set(records["model"])

    def test_persistence_scores_zero_skill_against_itself(self, records):
        persistence = records[records["model"] == "persistence"]
        assert np.allclose(persistence["skill_vs_reference"], 0.0)

    def test_ridge_beats_persistence_at_long_lead(self, records):
        """A sanity check on the harness, not a claim about the ocean."""
        ridge = records[(records["model"] == "ridge") & (records["horizon"] == 24)]
        assert ridge["skill_vs_reference"].mean() > 0

    def test_every_fold_has_test_samples(self, records):
        assert (records["n"] > 0).all()


class TestAggregation:
    def test_mean_and_std_across_folds(self):
        records = pd.DataFrame(
            {
                "fold": [0, 1, 2, 0, 1, 2],
                "horizon": [6, 6, 6, 24, 24, 24],
                "model": ["ridge"] * 6,
                "rmse": [0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
            }
        )
        stats = aggregate_folds(records, "rmse")
        assert stats.loc[(6, "ridge"), "mean"] == pytest.approx(0.5)
        assert stats.loc[(6, "ridge"), "n_folds"] == 3
        assert stats.loc[(6, "ridge"), "std"] == pytest.approx(0.1)

    def test_std_is_the_error_bar(self):
        """Identical folds mean zero spread - the difference is certain."""
        records = pd.DataFrame(
            {
                "fold": [0, 1],
                "horizon": [6, 6],
                "model": ["ridge", "ridge"],
                "rmse": [0.5, 0.5],
            }
        )
        assert aggregate_folds(records, "rmse").loc[(6, "ridge"), "std"] == pytest.approx(0.0)

    def test_format_produces_mean_plus_minus_std(self):
        records = pd.DataFrame(
            {
                "fold": [0, 1, 2],
                "horizon": [6, 6, 6],
                "model": ["ridge"] * 3,
                "rmse": [0.4, 0.5, 0.6],
            }
        )
        table = format_mean_std(records, "rmse")
        assert table.loc[6, "ridge"] == "0.500 ± 0.100"

    def test_empty_input(self):
        assert aggregate_folds(pd.DataFrame()).empty
        assert format_mean_std(pd.DataFrame()).empty


class TestEmptyColumnDropping:
    def test_all_nan_columns_are_removed(self):
        """NDBC emits VIS and TIDE even at stations with no such sensor."""
        index = pd.date_range("2020-01-01", periods=500, freq="1h", tz="UTC")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "WVHT": 2 + rng.normal(0, 0.2, 500),
                "APD": 7 + rng.normal(0, 0.3, 500),
                "VIS": np.full(500, np.nan),
                "TIDE": np.full(500, np.nan),
            },
            index=index,
        )
        features = build_feature_frame(df)
        assert "VIS" not in features.columns
        assert "TIDE" not in features.columns
        assert "WVHT" in features.columns

    def test_partially_present_columns_are_kept(self):
        index = pd.date_range("2020-01-01", periods=500, freq="1h", tz="UTC")
        rng = np.random.default_rng(0)
        sparse = np.full(500, np.nan)
        sparse[::50] = 1.0
        df = pd.DataFrame(
            {"WVHT": 2 + rng.normal(0, 0.2, 500), "APD": sparse}, index=index
        )
        assert "APD" in build_feature_frame(df).columns
