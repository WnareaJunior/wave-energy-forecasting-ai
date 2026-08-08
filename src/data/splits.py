"""Temporal splitting.

The one place leakage can enter the pipeline, so it is isolated here and tested
directly.

Two rules, both easy to get wrong:

1. **Never shuffle.** Random k-fold on a time series puts tomorrow in the
   training set and today in the test set. Wave height is autocorrelated over
   days, so a shuffled split reports skill that does not exist.
2. **Leave a gap.** A sample issued at time t carries features reaching back
   ``context`` hours and a target reaching forward ``horizon`` hours. Without a
   buffer between the training and test blocks, the last training samples and
   the first test samples overlap in the observations they touch.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """One train/validation/test partition of a time index."""

    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "validation", "test"):
            index = getattr(self, name)
            rows.append(
                {
                    "split": name,
                    "n": len(index),
                    "start": index.min() if len(index) else None,
                    "end": index.max() if len(index) else None,
                }
            )
        return pd.DataFrame(rows).set_index("split")


def temporal_split(
    index: pd.DatetimeIndex,
    train_end: str,
    validation_end: str,
    gap_hours: int = 72,
) -> Split:
    """Split chronologically at two cut dates, with a gap between blocks.

    Args:
        index: Time index to split.
        train_end: Last timestamp of the training block (exclusive of the gap).
        validation_end: Last timestamp of the validation block.
        gap_hours: Buffer discarded between blocks. Set it to at least
            ``max(horizon)`` so no test target overlaps a training feature
            window.

    Returns:
        A :class:`Split`. Timestamps inside the gaps belong to no block.
    """
    index = pd.DatetimeIndex(index).sort_values()
    train_end_ts = pd.Timestamp(train_end, tz=index.tz)
    validation_end_ts = pd.Timestamp(validation_end, tz=index.tz)

    if train_end_ts >= validation_end_ts:
        raise ValueError("train_end must be before validation_end")

    gap = pd.Timedelta(hours=gap_hours)

    train = index[index <= train_end_ts]
    validation = index[(index > train_end_ts + gap) & (index <= validation_end_ts)]
    test = index[index > validation_end_ts + gap]

    return Split(train=train, validation=validation, test=test)


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    n_splits: int = 4,
    test_size_days: int = 90,
    gap_hours: int = 72,
    expanding: bool = True,
):
    """Yield successive train/test splits that walk forward through time.

    A single holdout gives one number that depends heavily on which months
    happened to land in the test window - and with only a few years of data,
    that is a real risk. Rolling-origin evaluation scores the model on several
    consecutive periods instead.

    Args:
        index: Time index to split.
        n_splits: Number of folds.
        test_size_days: Length of each test block.
        gap_hours: Buffer between train and test.
        expanding: If True the training window grows each fold (use all history);
            if False it slides at fixed length (test adaptation to recent regime).

    Yields:
        (train_index, test_index) pairs, earliest fold first.
    """
    index = pd.DatetimeIndex(index).sort_values()
    if len(index) == 0:
        return

    test_size = pd.Timedelta(days=test_size_days)
    gap = pd.Timedelta(hours=gap_hours)
    end = index.max()

    boundaries = [end - test_size * (n_splits - i) for i in range(n_splits)]

    for test_start in boundaries:
        test_end = test_start + test_size
        train_end = test_start - gap

        if expanding:
            train = index[index <= train_end]
        else:
            window_start = train_end - (boundaries[0] - index.min())
            train = index[(index > window_start) & (index <= train_end)]

        test = index[(index > test_start) & (index <= test_end)]
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test


def check_no_overlap(train: pd.DatetimeIndex, test: pd.DatetimeIndex, gap_hours: int) -> None:
    """Raise if train and test are not separated by at least ``gap_hours``.

    Call this after any custom split. It is three lines and it catches the class
    of bug that otherwise shows up as an unexplainably good test score.
    """
    if len(train) == 0 or len(test) == 0:
        return
    shared = train.intersection(test)
    if len(shared):
        raise ValueError(f"Train and test overlap at {len(shared)} timestamps")

    separation = test.min() - train.max()
    required = pd.Timedelta(hours=gap_hours)
    if separation < required:
        raise ValueError(
            f"Only {separation} between train end and test start; need {required}"
        )
