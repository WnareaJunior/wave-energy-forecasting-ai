"""Checks on the shared configuration and region definition.

The repository previously carried three different, non-overlapping study regions
hardcoded across the notebooks, the docs, and the two downloaders. These tests
exist to keep that from happening again.
"""

import dataclasses

import pytest

from src.config import (
    COPERNICUS_RESOLUTION_DEG,
    PACIFIC_NORTHWEST,
    STUDY_REGION,
    Region,
)


class TestRegionValidation:
    def test_rejects_inverted_longitudes(self):
        with pytest.raises(ValueError, match="lon_min must be"):
            Region("bad", lon_min=-124.0, lon_max=-130.0, lat_min=46.0, lat_max=50.5)

    def test_rejects_inverted_latitudes(self):
        with pytest.raises(ValueError, match="lat_min must be"):
            Region("bad", lon_min=-130.0, lon_max=-124.0, lat_min=50.5, lat_max=46.0)

    def test_rejects_out_of_range_longitude(self):
        with pytest.raises(ValueError, match="longitudes must be"):
            Region("bad", lon_min=-200.0, lon_max=-124.0, lat_min=46.0, lat_max=50.5)

    def test_rejects_out_of_range_latitude(self):
        with pytest.raises(ValueError, match="latitudes must be"):
            Region("bad", lon_min=-130.0, lon_max=-124.0, lat_min=46.0, lat_max=95.0)


class TestStudyRegion:
    def test_study_region_is_pacific_northwest(self):
        assert STUDY_REGION == PACIFIC_NORTHWEST

    def test_bounds_match_the_published_analysis(self):
        """These bounds appear in the README, the notebooks, and every figure."""
        assert (STUDY_REGION.lon_min, STUDY_REGION.lon_max) == (-130.0, -124.0)
        assert (STUDY_REGION.lat_min, STUDY_REGION.lat_max) == (46.0, 50.5)

    def test_longitudes_are_west(self):
        """Guards against the +124 to +144 E (Western Pacific) regression."""
        assert STUDY_REGION.lon_max < 0

    def test_grid_shape_at_copernicus_resolution(self):
        """0.2 degree grid over the PNW box: 31 x 23 = 713 cells."""
        n_lon, n_lat = STUDY_REGION.grid_shape(COPERNICUS_RESOLUTION_DEG)
        assert (n_lon, n_lat) == (31, 23)
        assert n_lon * n_lat == 713

    def test_region_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            STUDY_REGION.lat_min = 0.0
