"""Parsing tests for NDBC standard-meteorological files.

The sentinel handling is the part worth testing hardest: an unconverted 99.00
in WVHT is a 99-metre wave, and it will not raise an error - it will just
quietly dominate every mean, threshold, and model fit downstream.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.ndbc import (
    MISSING_SENTINELS,
    VALID_RANGES,
    coverage_report,
    parse_stdmet,
)

MODERN_FILE = """\
#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2020 01 01 00 00 220  8.0  9.5  3.20 11.11  7.50 250 1015.2   9.0  10.5   7.0 99.0 99.0 99.00
2020 01 01 01 00 225  9.0 11.0  3.50 12.50  8.10 255 1014.8   9.1  10.5   7.1 99.0 99.0 99.00
2020 01 01 02 00 999 99.0 99.0 99.00 99.00 99.00 999 9999.0 999.0 999.0 999.0 99.0 99.0 99.00
2020 01 01 03 00 230  7.5  8.8  2.90 10.00  7.00 245 1015.9   8.9  10.4   6.9 99.0 99.0 99.00
"""

LEGACY_FILE = """\
#YY MM DD hh WD   WSPD GST  WVHT  DPD   APD  MWD  BARO   ATMP  WTMP  DEWP  VIS
1998 03 15 12 210  6.0  7.0  2.10  9.00  6.20 240 1018.0   8.0   9.5   6.0 99.0
1998 03 15 13 215  6.5  7.5  2.20  9.50  6.40 245 1017.8   8.1   9.5   6.1 99.0
"""


class TestParsing:
    def test_parses_modern_format(self):
        df = parse_stdmet(MODERN_FILE)
        assert len(df) == 4
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

    def test_reads_wave_height(self):
        df = parse_stdmet(MODERN_FILE)
        assert df["WVHT"].iloc[0] == pytest.approx(3.20)
        assert df["DPD"].iloc[0] == pytest.approx(11.11)
        assert df["APD"].iloc[0] == pytest.approx(7.50)

    def test_index_is_sorted(self):
        df = parse_stdmet(MODERN_FILE)
        assert df.index.is_monotonic_increasing

    def test_empty_input(self):
        assert parse_stdmet("").empty
        assert parse_stdmet("#header only\n#units\n").empty

    def test_handles_legacy_format_without_minutes(self):
        """Pre-2007 files omit the minutes column; the row count must survive."""
        df = parse_stdmet(LEGACY_FILE)
        assert len(df) == 2
        assert df.index[0].year == 1998


class TestSentinels:
    def test_wave_height_sentinel_becomes_nan(self):
        """99.00 in WVHT is 'missing', not a 99-metre wave."""
        df = parse_stdmet(MODERN_FILE)
        assert np.isnan(df["WVHT"].iloc[2])

    def test_all_sentinels_on_the_missing_row(self):
        df = parse_stdmet(MODERN_FILE)
        row = df.iloc[2]
        for column in ("WVHT", "DPD", "APD", "MWD", "WSPD", "WDIR", "PRES"):
            assert np.isnan(row[column]), f"{column} sentinel not converted"

    def test_valid_rows_are_untouched(self):
        df = parse_stdmet(MODERN_FILE)
        assert df["WVHT"].notna().sum() == 3

    def test_no_impossible_wave_heights_survive(self):
        """The whole point: nothing above the physical range gets through."""
        df = parse_stdmet(MODERN_FILE)
        assert df["WVHT"].max() < VALID_RANGES["WVHT"][1]

    def test_every_sentinel_has_a_column(self):
        from src.data.ndbc import STDMET_COLUMNS

        assert set(MISSING_SENTINELS).issubset(set(STDMET_COLUMNS))


class TestRangeFiltering:
    def test_out_of_range_period_dropped(self):
        """A 45-second period is not physical; it is a sensor fault."""
        text = MODERN_FILE.replace("11.11", "45.00")
        df = parse_stdmet(text)
        assert np.isnan(df["DPD"].iloc[0])

    def test_negative_wave_height_dropped(self):
        text = MODERN_FILE.replace(" 3.20", "-1.00")
        df = parse_stdmet(text)
        assert np.isnan(df["WVHT"].iloc[0])


class TestCoverageReport:
    def test_counts_present_and_missing(self):
        df = parse_stdmet(MODERN_FILE)
        report = coverage_report(df, ["WVHT"])
        assert report.loc["WVHT", "present"] == 3
        assert report.loc["WVHT", "missing"] == 1
        assert report.loc["WVHT", "pct_present"] == pytest.approx(75.0)
