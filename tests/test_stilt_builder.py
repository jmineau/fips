"""Tests for fips.problems.flux.transport.stilt.builder."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import xarray as xr

from fips.matrix import MatrixBlock
from fips.problems.flux.transport.stilt.builder import (
    JacobianBuilder,
    _build_jacobian_row,
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

    agg_df = pd.DataFrame(
        [[agg_value]],
        index=pd.MultiIndex.from_tuples([(-111.85, 40.77)], names=["lon", "lat"]),
        columns=pd.DatetimeIndex(["2023-01-01"], name="time"),
    )
    fp.aggregate.return_value = agg_df
    return fp


def _fake_path(sim_id: str = "hrrr_202301011200_-111.85_40.77_5") -> Path:
    """Return a footprint path whose parent dir encodes the sim_id (for hour filters)."""
    return Path(f"/fake/{sim_id}/{sim_id}_slv_foot.nc")


def _model(*paths: Path):
    """Mock Model whose footprints[name].paths() returns the given paths."""
    model = MagicMock()
    model.footprints.__getitem__.return_value.paths.return_value = list(paths)
    return model


@pytest.fixture
def load_footprints(monkeypatch):
    """Patch Footprint.from_netcdf to map each path to a prepared fake footprint."""

    def _install(mapping: dict[Path, object]):
        keyed = {str(p): fp for p, fp in mapping.items()}

        def fake_from_netcdf(path, *args, **kwargs):
            return keyed[str(path)]

        monkeypatch.setattr(
            "fips.problems.flux.transport.stilt.builder.Footprint.from_netcdf",
            staticmethod(fake_from_netcdf),
        )

    return _install


# ---------------------------------------------------------------------------
# _build_jacobian_row
# ---------------------------------------------------------------------------


def test_row_returns_dict_with_correct_obs_index():
    """A row is keyed by target name and indexed by (location, time)."""
    fp = _fake_footprint()
    result = _build_jacobian_row(
        fp,
        {"DEFAULT": [(-111.85, 40.77)]},
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
    """A footprint with an all-zero aggregate contributes no row."""
    fp = _fake_footprint(agg_value=0.0)
    result = _build_jacobian_row(
        fp,
        {"DEFAULT": [(-111.85, 40.77)]},
        location_dim="obs_location",
        time_dim="obs_time",
        flux_times=_flux_times(),
    )
    assert result is None


def test_row_multi_target_set():
    """Passing multiple named targets yields one row entry per target."""
    fp = _fake_footprint()
    result = _build_jacobian_row(
        fp,
        {"A": [(-111.85, 40.77)], "B": [(-111.85, 40.77)]},
        location_dim="obs_location",
        time_dim="obs_time",
        flux_times=_flux_times(),
    )
    assert result is not None
    assert set(result.keys()) == {"A", "B"}


def test_row_accepts_xarray_grid_target():
    """A target may be an xarray grid; it is forwarded straight to aggregate()."""
    fp = _fake_footprint()
    grid = xr.Dataset(coords={"lon": [-111.85], "lat": [40.77]})
    result = _build_jacobian_row(
        fp,
        {"DEFAULT": grid},
        location_dim="obs_location",
        time_dim="obs_time",
        flux_times=_flux_times(),
    )
    assert result is not None
    fp.aggregate.assert_called_once()
    # the grid object is passed through unchanged as the aggregate target
    assert fp.aggregate.call_args.args[0] is grid


# ---------------------------------------------------------------------------
# JacobianBuilder
# ---------------------------------------------------------------------------


def test_builder_init():
    """The builder stores the model and default dimension names."""
    model = _model()
    builder = JacobianBuilder(model)
    assert builder.model is model
    assert builder.location_dim == "obs_location"
    assert builder.time_dim == "obs_time"


def test_build_from_coords_queries_paths(load_footprints):
    """build_from_coords queries footprint paths over the flux window."""
    p = _fake_path()
    model = _model(p)
    load_footprints({p: _fake_footprint()})
    flux_times = _flux_times()

    builder = JacobianBuilder(model)
    builder.build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=flux_times,
        footprint="slv",
    )

    model.footprints.__getitem__.assert_any_call("slv")
    model.footprints["slv"].paths.assert_called_once_with(
        mets=None,
        time_range=(flux_times[0].left, flux_times[-1].right),
        location_ids=None,
    )


def test_build_from_coords_passes_filters(load_footprints):
    """met/time_range/location filters are forwarded to paths()."""
    p = _fake_path()
    model = _model(p)
    load_footprints({p: _fake_footprint()})
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

    model.footprints["slv"].paths.assert_called_once_with(
        mets="hrrr",
        time_range=tr,
        location_ids={"site_A"},
    )


def test_build_from_grid_accepts_xarray(load_footprints):
    """build_from_grid accepts an xarray grid target end-to-end."""
    p = _fake_path()
    model = _model(p)
    load_footprints({p: _fake_footprint()})
    grid = xr.Dataset(coords={"lon": [-111.85], "lat": [40.77]})

    builder = JacobianBuilder(model)
    result = builder.build_from_grid(grid, _flux_times(), "slv")

    assert result is not None
    model.footprints["slv"].paths.assert_called_once()


def test_subset_hours_filters_footprints(load_footprints):
    """subset_hours drops paths whose receptor hour is excluded."""
    p_noon = _fake_path("hrrr_202301011200_-111.85_40.77_5")
    p_midnight = _fake_path("hrrr_202301010000_-111.85_40.77_5")
    model = _model(p_noon, p_midnight)
    load_footprints(
        {
            p_noon: _fake_footprint(location_id="A", time="2023-01-01 12:00"),
            p_midnight: _fake_footprint(location_id="B", time="2023-01-01 00:00"),
        }
    )

    builder = JacobianBuilder(model)
    result = builder.build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=_flux_times(),
        footprint="slv",
        subset_hours=12,
    )
    # Only the noon footprint (hour=12) should produce a row
    assert isinstance(result, MatrixBlock)  # list coords -> single block
    df = result.data
    assert "A" in df.index.get_level_values("obs_location")
    assert "B" not in df.index.get_level_values("obs_location")


def test_raises_when_no_footprints_after_filter():
    """An empty path set raises a clear 'No footprints found' error."""
    model = _model()  # paths() returns nothing
    builder = JacobianBuilder(model)

    with pytest.raises(ValueError, match="No footprints found"):
        builder.build_from_coords(
            coords=[(-111.85, 40.77)],
            flux_times=_flux_times(),
            footprint="slv",
        )


def test_raises_when_no_rows_produced(load_footprints):
    """All-zero aggregates across paths raise 'No Jacobian rows'."""
    p = _fake_path()
    model = _model(p)
    load_footprints({p: _fake_footprint(agg_value=0.0)})  # all-zero → no overlap
    builder = JacobianBuilder(model)

    with pytest.raises(ValueError, match="No Jacobian rows"):
        builder.build_from_coords(
            coords=[(-111.85, 40.77)],
            flux_times=_flux_times(),
            footprint="slv",
        )


def test_location_mapper_applied(load_footprints):
    """location_mapper renames location ids in the assembled index."""
    p = _fake_path()
    model = _model(p)
    load_footprints({p: _fake_footprint(location_id="202301011200_-111.85_40.77_5")})
    mapper = {"202301011200_-111.85_40.77_5": "wbb"}

    builder = JacobianBuilder(model)
    result = builder.build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=_flux_times(),
        footprint="slv",
        location_mapper=mapper,
    )
    assert isinstance(result, MatrixBlock)  # list coords -> single block
    assert "wbb" in result.data.index.get_level_values("obs_location")
