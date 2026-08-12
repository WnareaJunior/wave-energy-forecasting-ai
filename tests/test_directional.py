"""Tests for directional error stratification.

The claim being supported is that GEFS error at 46087 is *organised by wave
direction* while error at the open-coast buoys is not, because the 0.25 degree
grid cannot resolve the Strait of Juan de Fuca entrance. These tests check the
measurement can tell those two worlds apart on data where the answer is known,
so that a null result on the real record means something.
"""

import numpy as np
import pandas as pd
import pytest

from src.eval.directional import (
    MIN_SECTOR_SAMPLES,
    SECTOR_NAMES,
    direction_sector,
    directional_structure,
    error_by_sector,
)


class TestDirectionSector:
    def test_cardinal_bearings_land_in_their_own_sector(self):
        bearings = pd.Series([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        assert list(direction_sector(bearings)) == list(SECTOR_NAMES)

    def test_north_does_not_split_across_the_wrap(self):
        """359 and 1 degrees are both north, and must bin together.

        Binning without the half-sector rotation puts them in different
        sectors, which manufactures a directional difference where there is
        none - the exact artefact this analysis would otherwise report as a
        finding.
        """
        assert direction_sector(pd.Series([359.0]))[0] == "N"
        assert direction_sector(pd.Series([1.0]))[0] == "N"
        assert direction_sector(pd.Series([22.0]))[0] == "N"
        assert direction_sector(pd.Series([23.0]))[0] == "NE"

    def test_invalid_bearings_become_missing(self):
        sectors = direction_sector(pd.Series([np.nan, -5.0, 400.0, 90.0]))
        assert sectors.isna().sum() == 3
        assert sectors.iloc[3] == "E"


def _paired(n=4000, bias_by_sector=None, noise=0.3, seed=0):
    """Observations and a forecast with a controllable per-sector bias."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2016-01-01", periods=n, freq="1h", tz="UTC")
    direction = pd.Series(rng.uniform(0, 360, n), index=index)
    observed = pd.Series(2.0 + rng.gamma(2.0, 0.3, n), index=index)

    sectors = direction_sector(direction)
    offsets = pd.Series(0.0, index=index)
    for name, value in (bias_by_sector or {}).items():
        offsets[sectors == name] = value

    forecast = observed + offsets + rng.normal(0, noise, n)
    return observed, forecast, direction


class TestErrorBySector:
    def test_uniform_error_shows_little_directional_structure(self):
        """The null case: a model wrong the same way everywhere."""
        observed, forecast, direction = _paired(
            bias_by_sector={name: 0.4 for name in SECTOR_NAMES}
        )
        structure = directional_structure(
            error_by_sector(observed, forecast, direction)
        )
        assert structure["n_sectors"] == len(SECTOR_NAMES)
        assert abs(structure["bias_spread_m"]) < 0.1

    def test_direction_dependent_error_is_detected(self):
        """The signal case: sheltering missed from one quarter only."""
        observed, forecast, direction = _paired(
            bias_by_sector={"W": 0.9, "NW": 0.7}
        )
        by_sector = error_by_sector(observed, forecast, direction)
        structure = directional_structure(by_sector)

        assert structure["bias_spread_m"] > 0.5
        assert structure["worst_sector"] in {"W", "NW"}

    def test_uniform_and_structured_are_separated_by_the_ratio(self):
        """The comparison the whole analysis rests on.

        A uniformly biased model and a directionally biased one can have the
        same mean bias. Only the spread tells them apart, and only the ratio
        makes the spread comparable between a rough site and a calm one.
        """
        uniform = directional_structure(
            error_by_sector(*_paired(bias_by_sector={n: 0.5 for n in SECTOR_NAMES}))
        )
        structured = directional_structure(
            error_by_sector(*_paired(bias_by_sector={"W": 1.0, "NW": 0.8, "SW": 0.4}))
        )
        assert structured["bias_spread_ratio"] > 3 * uniform["bias_spread_ratio"]

    def test_calm_and_rough_sites_compare_fairly_on_the_ratio(self):
        """Guards the confound: 46087 is calmer, not just better resolved.

        The same *relative* directional structure at two different sea-state
        scales must give the same ratio, or the analysis would simply be
        rediscovering that Neah Bay is sheltered.
        """
        rng = np.random.default_rng(5)
        n = 6000
        index = pd.date_range("2016-01-01", periods=n, freq="1h", tz="UTC")
        direction = pd.Series(rng.uniform(0, 360, n), index=index)
        sectors = direction_sector(direction)

        ratios = []
        for scale in (1.0, 3.0):
            observed = pd.Series(scale * (2.0 + rng.gamma(2.0, 0.3, n)), index=index)
            offsets = pd.Series(0.0, index=index)
            offsets[sectors == "W"] = 0.6 * scale
            forecast = observed + offsets + rng.normal(0, 0.3 * scale, n)
            ratios.append(
                directional_structure(
                    error_by_sector(observed, forecast, direction)
                )["bias_spread_ratio"]
            )

        assert ratios[0] == pytest.approx(ratios[1], rel=0.25)

    def test_thin_sectors_are_dropped_not_reported_noisily(self):
        observed, forecast, direction = _paired(n=1000)
        # Force one sector to be nearly empty.
        direction.iloc[:] = 90.0
        direction.iloc[:MIN_SECTOR_SAMPLES // 4] = 270.0
        by_sector = error_by_sector(observed, forecast, direction)
        assert "W" not in by_sector.index
        assert "E" in by_sector.index

    def test_structure_is_undefined_with_one_surviving_sector(self):
        observed, forecast, direction = _paired(n=1000)
        direction.iloc[:] = 90.0
        structure = directional_structure(
            error_by_sector(observed, forecast, direction)
        )
        assert structure["n_sectors"] == 1
        assert np.isnan(structure["bias_spread_m"])

    def test_empty_input_does_not_raise(self):
        empty = pd.Series(dtype=float)
        assert error_by_sector(empty, empty, empty).empty
