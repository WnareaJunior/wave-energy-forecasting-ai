"""Tests for GEFS reforecast extraction.

The lead-time derivation is the part that matters most. ``time`` in these files
is the valid time and lead is not stored, so it must be computed as
``valid_time - cycle_init``. Inverting that would build a dataset in which the
"forecast" already knows the answer, and nothing downstream would complain -
the metrics would just look wonderful.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.gefs import (
    FORECAST_VARS,
    MAX_MATCH_DEGREES,
    align_forecast_to_issue_time,
    cycle_init,
    cycle_key,
    daily_dates,
    decode_station_names,
    extract_cycle,
    resolve_station_indices,
    to_wide_csv,
)

#: Positions matching the real archive, confirmed against the bucket.
ARCHIVE_POSITIONS = {
    "46029": (46.143, -124.485),
    "46041": (47.353, -124.742),
    "46087": (48.493, -124.726),
}


def make_tab_dataset(
    station_names=("46029", "46041", "46087", "51001"),
    init="2016-01-04T03:00",
    n_times=48,
):
    """Build a dataset shaped like a real gefs.wave.*.tab.nc file.

    Mirrors the real schema: station_name as a (station, string40) character
    array, latitude and longitude as data variables with a time dimension
    rather than coordinates, and time holding absolute valid times.
    """
    times = pd.date_range(init, periods=n_times, freq="1h", tz=None)
    n_stations = len(station_names)

    names = np.array(
        [[c.encode() for c in name.ljust(40)] for name in station_names],
        dtype="S1",
    )

    latitudes = np.array(
        [ARCHIVE_POSITIONS.get(n, (24.5, -90.0))[0] for n in station_names]
    )
    longitudes = np.array(
        [ARCHIVE_POSITIONS.get(n, (24.5, -90.0))[1] for n in station_names]
    )

    rng = np.random.default_rng(0)
    shape = (n_times, n_stations)

    return xr.Dataset(
        {
            "station_name": (("station", "string40"), names),
            "latitude": (("time", "station"), np.tile(latitudes, (n_times, 1))),
            "longitude": (("time", "station"), np.tile(longitudes, (n_times, 1))),
            "hs": (("time", "station"), 2.5 + rng.normal(0, 0.3, shape)),
            "tr": (("time", "station"), 8.0 + rng.normal(0, 0.5, shape)),
            "fp": (("time", "station"), 0.1 + rng.normal(0, 0.01, shape)),
            "lm": (("time", "station"), 120 + rng.normal(0, 10, shape)),
            "th1p": (("time", "station"), 270 + rng.normal(0, 15, shape)),
        },
        coords={"time": times},
    )


class TestCycleTiming:
    def test_init_is_at_03z(self):
        assert cycle_init("20160104") == pd.Timestamp("2016-01-04T03:00", tz="UTC")

    def test_init_is_utc(self):
        assert str(cycle_init("20160104").tz) == "UTC"

    def test_key_template(self):
        key = cycle_key("20160104", "c00")
        assert key == (
            "GEFSv12/reforecast/2016/20160104/station/gefs.wave.20160104.c00.tab.nc"
        )

    def test_key_uses_the_member(self):
        assert ".p02.tab.nc" in cycle_key("20160104", "p02")


class TestStationNames:
    def test_decodes_character_array(self):
        ds = make_tab_dataset()
        assert decode_station_names(ds)[:3] == ["46029", "46041", "46087"]

    def test_strips_padding(self):
        """Names are padded to 40 characters in the file."""
        assert all(name == name.strip() for name in decode_station_names(make_tab_dataset()))


class TestStationResolution:
    def test_matches_by_name(self):
        ds = make_tab_dataset()
        assert resolve_station_indices(ds, ["46041"]) == {"46041": 1}

    def test_resolves_several(self):
        ds = make_tab_dataset()
        resolved = resolve_station_indices(ds, ["46029", "46041", "46087"])
        assert resolved == {"46029": 0, "46041": 1, "46087": 2}

    def test_falls_back_to_nearest_position(self):
        """An unnamed station still resolves if a close position exists."""
        ds = make_tab_dataset(station_names=("nearby", "51001"))
        # Put "nearby" essentially on top of 46041.
        ds["latitude"][:] = np.tile([47.35, 24.5], (len(ds.time), 1))
        ds["longitude"][:] = np.tile([-124.74, -90.0], (len(ds.time), 1))
        assert resolve_station_indices(ds, ["46041"]) == {"46041": 0}

    def test_rejects_a_distant_match(self):
        """Refuse to pair a buoy with a different piece of ocean."""
        ds = make_tab_dataset(station_names=("51001", "51002"))
        with pytest.raises(KeyError, match="degrees away"):
            resolve_station_indices(ds, ["46041"])

    def test_unknown_station_without_position_raises(self):
        ds = make_tab_dataset()
        with pytest.raises(KeyError, match="no known"):
            resolve_station_indices(ds, ["99999"])

    def test_match_limit_is_tight_enough_to_matter(self):
        """Half a degree is ~55 km; anything looser is a different sea state."""
        assert MAX_MATCH_DEGREES <= 0.5


class TestExtractCycle:
    def test_lead_starts_at_zero(self):
        ds = make_tab_dataset()
        out = extract_cycle(ds, {"46041": 1}, cycle_init("20160104"))
        assert out["lead_hours"].min() == 0

    def test_lead_increments_hourly(self):
        """Point output is hourly even though the gridded fields are 3-hourly."""
        ds = make_tab_dataset(n_times=10)
        out = extract_cycle(ds, {"46041": 1}, cycle_init("20160104"))
        assert list(out["lead_hours"]) == list(range(10))

    def test_lead_is_valid_minus_init(self):
        ds = make_tab_dataset()
        init = cycle_init("20160104")
        out = extract_cycle(ds, {"46041": 1}, init)
        row = out.iloc[5]
        assert row["valid_time"] == init + pd.Timedelta(hours=int(row["lead_hours"]))

    def test_negative_lead_is_rejected(self):
        """The guard against an inverted lead derivation."""
        ds = make_tab_dataset()
        later_init = cycle_init("20160105")  # after the file's first valid time
        with pytest.raises(ValueError, match="precede cycle init"):
            extract_cycle(ds, {"46041": 1}, later_init)

    def test_carries_the_forecast_variables(self):
        ds = make_tab_dataset()
        out = extract_cycle(ds, {"46041": 1}, cycle_init("20160104"))
        for var in FORECAST_VARS:
            assert var in out.columns

    def test_handles_several_stations(self):
        ds = make_tab_dataset()
        out = extract_cycle(
            ds, {"46029": 0, "46041": 1, "46087": 2}, cycle_init("20160104")
        )
        assert set(out["station"]) == {"46029", "46041", "46087"}
        assert len(out) == 3 * len(ds.time)

    def test_pulls_the_right_station_column(self):
        ds = make_tab_dataset()
        out = extract_cycle(ds, {"46087": 2}, cycle_init("20160104"))
        np.testing.assert_allclose(
            out["hs"].to_numpy(), ds["hs"].isel(station=2).values
        )

    def test_valid_times_are_utc(self):
        ds = make_tab_dataset()
        out = extract_cycle(ds, {"46041": 1}, cycle_init("20160104"))
        assert str(out["valid_time"].dt.tz) == "UTC"


class TestDateRange:
    def test_daily(self):
        dates = daily_dates("2016-01-01", "2016-01-05")
        assert dates == ["20160101", "20160102", "20160103", "20160104", "20160105"]

    def test_stride_subsamples(self):
        assert daily_dates("2016-01-01", "2016-01-10", stride=3) == [
            "20160101", "20160104", "20160107", "20160110"
        ]

    def test_five_year_window_size(self):
        """The 2015-2019 overlap with the NDBC records."""
        assert len(daily_dates("2015-01-01", "2019-12-31")) == 1826


class TestForecastAlignment:
    """The one operation that can leak the future into the postprocessing set.

    A forecast for valid time t at lead h was issued at t - h. If the shift ran
    the other way, the feature at issue time would carry a forecast made after
    it, and every metric downstream would look excellent and mean nothing.
    """

    @pytest.fixture
    def index(self):
        return pd.date_range("2016-01-01", periods=200, freq="1h", tz="UTC")

    def test_forecast_lands_horizon_hours_earlier(self, index):
        valid_time = pd.Timestamp("2016-01-03T00:00", tz="UTC")
        at_valid = pd.Series([3.7], index=pd.DatetimeIndex([valid_time]))

        aligned = align_forecast_to_issue_time(at_valid, 24, index)

        issue_time = valid_time - pd.Timedelta(hours=24)
        assert aligned.loc[issue_time] == pytest.approx(3.7)

    def test_nothing_lands_at_the_valid_time_itself(self, index):
        """The giveaway of an inverted shift."""
        valid_time = pd.Timestamp("2016-01-03T00:00", tz="UTC")
        at_valid = pd.Series([3.7], index=pd.DatetimeIndex([valid_time]))
        aligned = align_forecast_to_issue_time(at_valid, 24, index)
        assert pd.isna(aligned.loc[valid_time])

    def test_never_shifts_forward(self, index):
        """Every placed value must sit strictly before the time it describes."""
        at_valid = pd.Series(
            np.arange(10.0),
            index=pd.date_range("2016-01-02", periods=10, freq="1h", tz="UTC"),
        )
        for horizon in (1, 6, 24, 72):
            aligned = align_forecast_to_issue_time(at_valid, horizon, index)
            placed = aligned.dropna()
            for issue_time, value in placed.items():
                original = at_valid[at_valid == value].index[0]
                assert issue_time < original
                assert original - issue_time == pd.Timedelta(hours=horizon)

    def test_longer_horizon_moves_further_back(self, index):
        at_valid = pd.Series(
            [2.0], index=pd.DatetimeIndex([pd.Timestamp("2016-01-05", tz="UTC")])
        )
        short = align_forecast_to_issue_time(at_valid, 6, index).dropna()
        long = align_forecast_to_issue_time(at_valid, 48, index).dropna()
        assert long.index[0] < short.index[0]

    def test_values_are_unchanged(self, index):
        at_valid = pd.Series(
            [1.5, 2.5, 3.5],
            index=pd.date_range("2016-01-04", periods=3, freq="1h", tz="UTC"),
        )
        aligned = align_forecast_to_issue_time(at_valid, 12, index)
        np.testing.assert_allclose(sorted(aligned.dropna()), [1.5, 2.5, 3.5])

    def test_output_is_on_the_requested_index(self, index):
        at_valid = pd.Series(
            [2.0], index=pd.DatetimeIndex([pd.Timestamp("2016-01-05", tz="UTC")])
        )
        assert align_forecast_to_issue_time(at_valid, 24, index).index.equals(index)

    def test_empty_input_gives_all_nan(self, index):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        aligned = align_forecast_to_issue_time(empty, 24, index)
        assert aligned.isna().all()
        assert aligned.index.equals(index)

    def test_forecasts_outside_the_index_are_dropped(self, index):
        far_future = pd.Series(
            [9.0], index=pd.DatetimeIndex([pd.Timestamp("2030-01-01", tz="UTC")])
        )
        assert align_forecast_to_issue_time(far_future, 24, index).isna().all()


class TestWideCsv:
    @pytest.fixture
    def long_df(self):
        ds = make_tab_dataset(n_times=80)
        frames = [
            extract_cycle(ds, {"46041": 1}, cycle_init(date))
            for date in ("20160104",)
        ]
        return pd.concat(frames, ignore_index=True)

    def test_columns_are_named_by_horizon(self, long_df):
        wide = to_wide_csv(long_df, "46041", [1, 24, 48])
        assert list(wide.columns) == ["h1", "h24", "h48"]

    def test_index_is_valid_time(self, long_df):
        wide = to_wide_csv(long_df, "46041", [24])
        assert wide.index.name == "time"

    def test_values_match_the_long_form(self, long_df):
        wide = to_wide_csv(long_df, "46041", [24])
        expected = long_df[long_df["lead_hours"] == 24].iloc[0]
        assert wide.loc[expected["valid_time"], "h24"] == pytest.approx(expected["hs"])

    def test_unknown_station_yields_empty(self, long_df):
        assert to_wide_csv(long_df, "99999", [24]).empty

    def test_index_is_sorted(self, long_df):
        assert to_wide_csv(long_df, "46041", [1, 24]).index.is_monotonic_increasing
