"""Tests for fips.problems.flux.transport.stilt.builder."""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from fips.problems.flux.transport.stilt.builder import (
    JacobianBuilder,
    build_jacobian_row_from_coords,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flux_times() -> pd.IntervalIndex:
    return pd.interval_range(
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-02"),
        freq="1h",
    )


def _fake_footprint(
    location_id: str = "site_A",
    time: str = "2023-01-01 12:00",
    agg_value: float = 1.0,
):
    """Mock Footprint whose aggregate() returns a one-cell DataFrame."""
    fp = MagicMock()
    fp.receptor.location_id = location_id
    fp.receptor.time = pd.Timestamp(time)

    flux_times = _flux_times()
    agg_df = pd.DataFrame(
        [[agg_value]],
        index=pd.MultiIndex.from_tuples([(-111.85, 40.77)], names=["lon", "lat"]),
        columns=pd.DatetimeIndex(["2023-01-01"], name="time"),
    )
    fp.aggregate.return_value = agg_df
    return fp


def _model(*footprints):
    """Mock Model whose footprints[name].load() returns the given footprints."""
    model = MagicMock()
    model.footprints.__getitem__.return_value.load.return_value = list(footprints)
    return model


# ---------------------------------------------------------------------------
# build_jacobian_row_from_coords
# ---------------------------------------------------------------------------

def test_row_returns_dict_with_correct_obs_index():
    fp = _fake_footprint()
    result = build_jacobian_row_from_coords(
        fp=fp,
        coords={"DEFAULT": [(-111.85, 40.77)]},
        location_dim="obs_location",
        time_dim="obs_time",
        flux_times=_flux_times(),
    )
    assert result is not None
    assert "DEFAULT" in result
    df = result["DEFAULT"]
    assert df.index.names == ["obs_location", "obs_time"]
    assert df.index[0] == ("site_A", pd.Timestamp("2023-01-01 12:00"))


def test_row_returns_none_when_no_overlap():
    fp = _fake_footprint(agg_value=0.0)
    result = build_jacobian_row_from_coords(
        fp=fp,
        coords={"DEFAULT": [(-111.85, 40.77)]},
        location_dim="obs_location",
        time_dim="obs_time",
        flux_times=_flux_times(),
    )
    assert result is None


def test_row_multi_coord_set():
    fp = _fake_footprint()
    result = build_jacobian_row_from_coords(
        fp=fp,
        coords={"A": [(-111.85, 40.77)], "B": [(-111.85, 40.77)]},
        location_dim="obs_location",
        time_dim="obs_time",
        flux_times=_flux_times(),
    )
    assert result is not None
    assert set(result.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# JacobianBuilder
# ---------------------------------------------------------------------------

def test_builder_init():
    model = _model()
    builder = JacobianBuilder(model)
    assert builder.model is model
    assert builder.location_dim == "obs_location"
    assert builder.time_dim == "obs_time"


def test_build_from_coords_calls_get_footprints():
    fp = _fake_footprint()
    model = _model(fp)
    flux_times = _flux_times()

    builder = JacobianBuilder(model)
    builder.build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=flux_times,
        footprint="slv",
    )

    model.footprints.__getitem__.assert_called_once_with("slv")
    model.footprints["slv"].load.assert_called_once_with(
        mets=None,
        time_range=(flux_times[0].left, flux_times[-1].right),
        location_ids=None,
    )


def test_build_from_coords_passes_filters():
    fp = _fake_footprint()
    model = _model(fp)
    flux_times = _flux_times()
    tr = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-31"))

    builder = JacobianBuilder(model)
    builder.build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=flux_times,
        footprint="slv",
        mets="hrrr",
        time_range=tr,
        location_ids={"site_A"},
    )

    model.footprints.__getitem__.assert_called_once_with("slv")
    model.footprints["slv"].load.assert_called_once_with(
        mets="hrrr",
        time_range=tr,
        location_ids={"site_A"},
    )


def test_subset_hours_filters_footprints():
    fp_noon = _fake_footprint(location_id="A", time="2023-01-01 12:00")
    fp_midnight = _fake_footprint(location_id="B", time="2023-01-01 00:00")
    model = _model(fp_noon, fp_midnight)

    builder = JacobianBuilder(model)
    result = builder.build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=_flux_times(),
        footprint="slv",
        subset_hours=12,
    )
    # Only fp_noon (hour=12) should produce a row
    assert result is not None
    df = result.data
    assert "A" in df.index.get_level_values("obs_location")
    assert "B" not in df.index.get_level_values("obs_location")


def test_raises_when_no_footprints_after_filter():
    model = _model()  # no footprints at all
    builder = JacobianBuilder(model)

    with pytest.raises(ValueError, match="No footprints found"):
        builder.build_from_coords(
            coords=[(-111.85, 40.77)],
            flux_times=_flux_times(),
            footprint="slv",
        )


def test_raises_when_no_rows_produced():
    fp = _fake_footprint(agg_value=0.0)  # aggregate returns all zeros → no overlap
    model = _model(fp)
    builder = JacobianBuilder(model)

    with pytest.raises(ValueError, match="No Jacobian rows"):
        builder.build_from_coords(
            coords=[(-111.85, 40.77)],
            flux_times=_flux_times(),
            footprint="slv",
        )


def test_location_mapper_applied():
    fp = _fake_footprint(location_id="202301011200_-111.85_40.77_5")
    model = _model(fp)
    mapper = {"202301011200_-111.85_40.77_5": "wbb"}

    builder = JacobianBuilder(model)
    result = builder.build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=_flux_times(),
        footprint="slv",
        location_mapper=mapper,
    )
    assert "wbb" in result.data.index.get_level_values("obs_location")
