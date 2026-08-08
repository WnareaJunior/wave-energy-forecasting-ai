"""Verification-metric tests.

Pinned against hand-computable cases, because a subtly wrong skill score is
worse than none: it looks authoritative and points the whole project the wrong
way.
"""

import numpy as np
import pytest

from src.eval.metrics import (
    bias,
    correlation,
    coverage,
    evaluate,
    mae,
    pinball_loss,
    results_table,
    rmse,
    skill_score,
    storm_rmse,
)


class TestPointMetrics:
    def test_perfect_forecast_scores_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0
        assert bias(y, y) == 0.0

    def test_rmse_hand_computed(self):
        # errors 1, 1, 1 -> RMSE 1
        assert rmse([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_rmse_punishes_outliers_more_than_mae(self):
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        y_pred[0] = 10.0
        assert rmse(y_true, y_pred) > mae(y_true, y_pred)

    def test_bias_sign(self):
        """Positive bias means the forecast runs high."""
        assert bias([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)
        assert bias([2.0, 2.0], [1.0, 1.0]) == pytest.approx(-1.0)

    def test_bias_cancels_out(self):
        """Offsetting errors give zero bias but nonzero RMSE - why both exist."""
        y_true, y_pred = [1.0, 3.0], [2.0, 2.0]
        assert bias(y_true, y_pred) == pytest.approx(0.0)
        assert rmse(y_true, y_pred) > 0

    def test_nans_are_ignored(self):
        assert rmse([1.0, np.nan, 3.0], [2.0, 5.0, 4.0]) == pytest.approx(1.0)

    def test_correlation_of_a_perfect_linear_relation(self):
        y = np.arange(10.0)
        assert correlation(y, 2 * y + 5) == pytest.approx(1.0)

    def test_correlation_needs_two_points(self):
        assert np.isnan(correlation([1.0], [1.0]))


class TestSkillScore:
    def test_zero_when_matching_the_reference(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert skill_score(y_true, y_pred, y_pred) == pytest.approx(0.0)

    def test_positive_when_better(self):
        y_true = np.array([1.0, 2.0, 3.0])
        good = np.array([1.1, 2.1, 3.1])
        bad = np.array([2.0, 3.0, 4.0])
        assert skill_score(y_true, good, bad) > 0

    def test_negative_when_worse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        good = np.array([1.1, 2.1, 3.1])
        bad = np.array([2.0, 3.0, 4.0])
        assert skill_score(y_true, bad, good) < 0

    def test_halving_error_gives_half_skill(self):
        y_true = np.zeros(4)
        reference = np.full(4, 1.0)
        model = np.full(4, 0.5)
        assert skill_score(y_true, model, reference) == pytest.approx(0.5)

    def test_perfect_reference_returns_nan(self):
        y = np.array([1.0, 2.0])
        assert np.isnan(skill_score(y, np.array([1.5, 2.5]), y))


class TestStormMetrics:
    def test_restricts_to_the_upper_tail(self):
        """Errors on calm rows are excluded, so only the storm error counts."""
        y_true = np.array([1.0] * 9 + [10.0])
        y_pred = np.array([5.0] * 9 + [10.0])
        assert storm_rmse(y_true, y_pred, percentile=90) == pytest.approx(0.0)

    def test_catches_a_model_that_misses_storms(self):
        y_true = np.array([1.0] * 9 + [10.0])
        y_pred = np.array([1.0] * 9 + [2.0])
        assert storm_rmse(y_true, y_pred, percentile=90) == pytest.approx(8.0)
        # Overall RMSE looks fine; the storm metric is what exposes it.
        assert rmse(y_true, y_pred) < storm_rmse(y_true, y_pred, 90)

    def test_all_nan_returns_nan(self):
        assert np.isnan(storm_rmse([np.nan, np.nan], [1.0, 2.0]))


class TestEvaluate:
    def test_returns_the_expected_keys(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true + 0.1
        results = evaluate(y_true, y_pred)
        assert {"rmse", "mae", "bias", "correlation", "rmse_p90", "n"} <= set(results)

    def test_skill_column_only_with_a_reference(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true + 0.1
        assert "skill_vs_reference" not in evaluate(y_true, y_pred)
        assert "skill_vs_reference" in evaluate(y_true, y_pred, y_true + 0.5)

    def test_n_counts_valid_observations(self):
        results = evaluate([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
        assert results["n"] == 2


class TestProbabilisticMetrics:
    def test_pinball_is_asymmetric(self):
        """A high quantile should be penalised more for under- than over-shooting."""
        under = pinball_loss([10.0], [8.0], quantile=0.9)
        over = pinball_loss([10.0], [12.0], quantile=0.9)
        assert under > over

    def test_pinball_at_the_median_is_half_the_absolute_error(self):
        assert pinball_loss([10.0], [8.0], quantile=0.5) == pytest.approx(1.0)

    def test_coverage_counts_observations_inside_the_interval(self):
        y_true = np.array([1.0, 2.0, 3.0, 10.0])
        lower = np.zeros(4)
        upper = np.full(4, 5.0)
        assert coverage(y_true, lower, upper) == pytest.approx(0.75)

    def test_full_coverage(self):
        y_true = np.array([1.0, 2.0])
        assert coverage(y_true, np.zeros(2), np.full(2, 5.0)) == pytest.approx(1.0)


class TestResultsTable:
    def test_indexes_by_horizon_and_model(self):
        records = [
            {"model": "persistence", "horizon": 1, "rmse": 0.2},
            {"model": "ridge", "horizon": 1, "rmse": 0.19},
            {"model": "ridge", "horizon": 24, "rmse": 0.6},
        ]
        table = results_table(records)
        assert table.index.names == ["horizon", "model"]
        assert table.loc[(1, "ridge"), "rmse"] == pytest.approx(0.19)

    def test_empty_input(self):
        assert results_table([]).empty
