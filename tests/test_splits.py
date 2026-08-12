"""Leakage tests.

If any of these fail, every skill number the project produces is inflated. They
are the highest-value tests in the repository.
"""

import pandas as pd
import pytest

from src.data.splits import (
    check_no_overlap,
    rolling_origin_splits,
    temporal_split,
)


@pytest.fixture
def hourly_index():
    return pd.date_range("2018-01-01", "2024-01-01", freq="1h", tz="UTC")


class TestTemporalSplit:
    def test_blocks_are_chronological(self, hourly_index):
        split = temporal_split(hourly_index, "2021-12-31", "2022-12-31")
        assert split.train.max() < split.validation.min()
        assert split.validation.max() < split.test.min()

    def test_gap_is_respected(self, hourly_index):
        split = temporal_split(hourly_index, "2021-12-31", "2022-12-31", gap_hours=72)
        assert split.validation.min() - split.train.max() > pd.Timedelta(hours=72)
        assert split.test.min() - split.validation.max() > pd.Timedelta(hours=72)

    def test_no_timestamp_in_two_blocks(self, hourly_index):
        split = temporal_split(hourly_index, "2021-12-31", "2022-12-31")
        assert len(split.train.intersection(split.validation)) == 0
        assert len(split.train.intersection(split.test)) == 0
        assert len(split.validation.intersection(split.test)) == 0

    def test_larger_gap_discards_more(self, hourly_index):
        small = temporal_split(hourly_index, "2021-12-31", "2022-12-31", gap_hours=1)
        large = temporal_split(hourly_index, "2021-12-31", "2022-12-31", gap_hours=240)
        assert len(large.validation) < len(small.validation)

    def test_rejects_inverted_cut_dates(self, hourly_index):
        with pytest.raises(ValueError, match="train_end must be before"):
            temporal_split(hourly_index, "2022-12-31", "2021-12-31")

    def test_summary_reports_all_blocks(self, hourly_index):
        summary = temporal_split(hourly_index, "2021-12-31", "2022-12-31").summary()
        assert list(summary.index) == ["train", "validation", "test"]
        assert summary["n"].sum() < len(hourly_index)  # the gaps are discarded


class TestRollingOrigin:
    def test_yields_requested_number_of_folds(self, hourly_index):
        folds = list(rolling_origin_splits(hourly_index, n_splits=4, test_size_days=90))
        assert len(folds) == 4

    def test_test_blocks_move_forward(self, hourly_index):
        folds = list(rolling_origin_splits(hourly_index, n_splits=4, test_size_days=90))
        starts = [test.min() for _, test in folds]
        assert starts == sorted(starts)

    def test_training_never_reaches_into_test(self, hourly_index):
        for train, test in rolling_origin_splits(hourly_index, n_splits=4, test_size_days=90):
            assert train.max() < test.min()

    def test_expanding_window_grows(self, hourly_index):
        sizes = [
            len(train)
            for train, _ in rolling_origin_splits(hourly_index, n_splits=4, expanding=True)
        ]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_empty_index_yields_nothing(self):
        empty = pd.DatetimeIndex([], tz="UTC")
        assert list(rolling_origin_splits(empty)) == []


class TestOverlapCheck:
    def test_passes_on_a_clean_split(self, hourly_index):
        split = temporal_split(hourly_index, "2021-12-31", "2022-12-31", gap_hours=72)
        check_no_overlap(split.train, split.test, gap_hours=72)

    def test_catches_shared_timestamps(self, hourly_index):
        block = hourly_index[:100]
        with pytest.raises(ValueError, match="overlap"):
            check_no_overlap(block, block, gap_hours=0)

    def test_catches_insufficient_gap(self, hourly_index):
        train = hourly_index[:100]
        test = hourly_index[101:200]
        with pytest.raises(ValueError, match="need"):
            check_no_overlap(train, test, gap_hours=72)
