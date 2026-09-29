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


def _receptor(time: str = "2023-01-01 12:00", longitude: float = -111.85):
    from stilt import PointReceptor

    return PointReceptor(time=time, longitude=longitude, latitude=40.77, altitude=5.0)


def _project(tmp_path, *receptors, variants=None):
    """Build a real PYSTILT model over *receptors*; outputs exist only once stubbed."""
    from stilt import Model

    settings = {
        "mets": {
            "hrrr": {
                "directory": str(tmp_path / "met"),
                "file_format": "%Y%m%d_%H",
                "file_tres": "6h",
            }
        },
        "grid": {
            "xmin": -112,
            "xmax": -111,
            "ymin": 40,
            "ymax": 41,
            "xres": 0.1,
            "yres": 0.1,
        },
    }
    if variants is not None:
        settings["variants"] = variants
    return Model(project=tmp_path / "project", receptors=list(receptors), **settings)


def _stub(model, receptor, variant: str = "hrrr", output: str = "footprint") -> Path:
    """Write a placeholder output file where PYSTILT expects it and return its path."""
    sim = model.simulations[receptor.id, variant]
    path = sim.footprint_path if output == "footprint" else sim.trajectories_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    return path


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
    model = MagicMock()
    builder = JacobianBuilder(model)
    assert builder.model is model
    assert builder.location_dim == "obs_location"
    assert builder.time_dim == "obs_time"


def test_rows_come_from_the_variant_within_the_flux_window(tmp_path, load_footprints):
    """Only the named variant's footprints inside the flux window make rows."""
    inside, outside = _receptor(), _receptor(time="2023-02-01 12:00")
    model = _project(
        tmp_path, inside, outside, variants={"hrrr": {}, "zi08": {"ziscale": 0.8}}
    )
    load_footprints(
        {
            _stub(model, inside): _fake_footprint(location_id="hrrr-inside"),
            _stub(model, outside): _fake_footprint(location_id="hrrr-outside"),
            _stub(model, inside, "zi08"): _fake_footprint(location_id="zi08-inside"),
        }
    )

    H = JacobianBuilder(model).build_from_coords(
        coords=[(-111.85, 40.77)], flux_times=_flux_times(), variant="hrrr"
    )

    assert isinstance(H, MatrixBlock)
    assert list(H.data.index.get_level_values("obs_location")) == ["hrrr-inside"]


def test_time_range_and_location_filters(tmp_path, load_footprints):
    """time_range and location_ids narrow the simulations used."""
    a = _receptor(time="2023-01-01 06:00", longitude=-111.85)
    b = _receptor(time="2023-01-01 07:00", longitude=-111.80)
    c = _receptor(time="2023-01-01 20:00", longitude=-111.85)
    model = _project(tmp_path, a, b, c)
    load_footprints(
        {
            _stub(model, r): _fake_footprint(location_id=n, time=str(r.time))
            for n, r in zip("abc", (a, b, c), strict=True)
        }
    )

    H = JacobianBuilder(model).build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=_flux_times(),
        variant="hrrr",
        time_range=(pd.Timestamp("2023-01-01 00:00"), pd.Timestamp("2023-01-01 12:00")),
        location_ids={a.location_id},
    )

    assert isinstance(H, MatrixBlock)
    assert list(H.data.index.get_level_values("obs_location")) == ["a"]


def test_build_from_grid_accepts_xarray(tmp_path, load_footprints):
    """build_from_grid accepts an xarray grid target end-to-end."""
    r = _receptor()
    model = _project(tmp_path, r)
    load_footprints({_stub(model, r): _fake_footprint()})
    grid = xr.Dataset(coords={"lon": [-111.85], "lat": [40.77]})

    result = JacobianBuilder(model).build_from_grid(grid, _flux_times(), "hrrr")

    assert isinstance(result, MatrixBlock)


def test_subset_hours_filters_by_receptor_hour(tmp_path, load_footprints):
    """subset_hours keeps only receptors at those UTC hours."""
    noon, midnight = (
        _receptor(time="2023-01-01 12:00"),
        _receptor(time="2023-01-01 00:00"),
    )
    model = _project(tmp_path, noon, midnight)
    load_footprints(
        {
            _stub(model, noon): _fake_footprint(
                location_id="A", time="2023-01-01 12:00"
            ),
            _stub(model, midnight): _fake_footprint(
                location_id="B", time="2023-01-01 00:00"
            ),
        }
    )

    result = JacobianBuilder(model).build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=_flux_times(),
        variant="hrrr",
        subset_hours=12,
    )

    assert isinstance(result, MatrixBlock)  # list coords -> single block
    assert list(result.data.index.get_level_values("obs_location")) == ["A"]


def test_raises_when_no_footprints_after_filter(tmp_path):
    """No stored footprint for the selection raises a clear error."""
    model = _project(tmp_path, _receptor())  # nothing has run

    with pytest.raises(ValueError, match="No footprints found for variant 'hrrr'"):
        JacobianBuilder(model).build_from_coords(
            coords=[(-111.85, 40.77)], flux_times=_flux_times(), variant="hrrr"
        )


def test_unknown_variant_raises(tmp_path):
    """A variant the project does not define is an error, not an empty Jacobian."""
    model = _project(tmp_path, _receptor())

    with pytest.raises(KeyError, match="hrr"):
        JacobianBuilder(model).build_from_coords(
            coords=[(-111.85, 40.77)], flux_times=_flux_times(), variant="hrr"
        )


def test_raises_when_no_rows_produced(tmp_path, load_footprints):
    """All-zero aggregates across footprints raise 'No Jacobian rows'."""
    r = _receptor()
    model = _project(tmp_path, r)
    load_footprints({_stub(model, r): _fake_footprint(agg_value=0.0)})

    with pytest.raises(ValueError, match="No Jacobian rows"):
        JacobianBuilder(model).build_from_coords(
            coords=[(-111.85, 40.77)], flux_times=_flux_times(), variant="hrrr"
        )


def test_location_mapper_applied(tmp_path, load_footprints):
    """location_mapper renames location ids in the assembled index."""
    r = _receptor()
    model = _project(tmp_path, r)
    load_footprints({_stub(model, r): _fake_footprint(location_id=str(r.location_id))})

    result = JacobianBuilder(model).build_from_coords(
        coords=[(-111.85, 40.77)],
        flux_times=_flux_times(),
        variant="hrrr",
        location_mapper={str(r.location_id): "wbb"},
    )

    assert isinstance(result, MatrixBlock)  # list coords -> single block
    assert list(result.data.index.get_level_values("obs_location")) == ["wbb"]


# ---------------------------------------------------------------------------
# build_from_target: spatial targets and regeneration
# ---------------------------------------------------------------------------


def _fake_points_footprint(location_id="site_A", time="2023-01-01 12:00"):
    """Mock Footprint whose aggregate() returns an id-indexed DataFrame."""
    fp = MagicMock()
    fp.receptor.location_id = location_id
    fp.receptor.time = pd.Timestamp(time)
    fp.aggregate.return_value = pd.DataFrame(
        [[1.0], [2.0]],
        index=pd.Index(["landfill", "wwtp"], name="cell"),
        columns=pd.DatetimeIndex(["2023-01-01"], name="time"),
    )
    return fp


def test_build_from_target_mesh_columns_are_cell_time(tmp_path, load_footprints):
    """A labelled Mesh target yields a (cell, time) Jacobian column index."""
    from stilt import Mesh

    r = _receptor()
    model = _project(tmp_path, r)
    fp = _fake_points_footprint()
    load_footprints({_stub(model, r): fp})
    builder = JacobianBuilder(model)
    target = Mesh.from_windows(
        [(-111.97, 40.515), (-112.015, 40.779)], 0.01, ids=["landfill", "wwtp"]
    )

    H = builder.build_from_target(target, _flux_times(), "hrrr")

    assert isinstance(H, MatrixBlock)
    assert fp.aggregate.call_args.args[0] is target
    cols = H.data.columns
    assert cols.names == ["cell", "time"]
    assert list(cols.get_level_values("cell")) == ["landfill", "wwtp"]


def test_build_from_grid_is_alias_of_build_from_target(tmp_path, load_footprints):
    """build_from_grid forwards to build_from_target unchanged."""
    r = _receptor()
    model = _project(tmp_path, r)
    load_footprints({_stub(model, r): _fake_footprint()})
    builder = JacobianBuilder(model)
    grid = xr.Dataset(coords={"lon": [-111.85], "lat": [40.77]})
    a = builder.build_from_grid(grid, _flux_times(), "hrrr")
    b = builder.build_from_target(grid, _flux_times(), "hrrr")
    assert isinstance(a, MatrixBlock) and isinstance(b, MatrixBlock)
    pd.testing.assert_frame_equal(a.data, b.data)


def test_build_from_target_regenerates_from_trajectories(tmp_path, monkeypatch):
    """A FootprintConfig regenerates from the variant's trajectories, not its footprints."""
    from stilt import FootprintConfig

    r = _receptor()
    model = _project(tmp_path, r)
    traj_path = _stub(model, r, output="trajectory")  # no stored footprint at all

    config = FootprintConfig.model_validate(
        {
            "grid": {
                "xmin": -112.0,
                "xmax": -111.0,
                "ymin": 40.0,
                "ymax": 41.0,
                "xres": 0.1,
                "yres": 0.1,
            }
        }
    )
    fp = _fake_footprint()
    traj = MagicMock()
    traj.footprint.return_value = fp

    def fake_from_parquet(path, *args, **kwargs):
        assert str(path) == str(traj_path)
        return traj

    monkeypatch.setattr(
        "fips.problems.flux.transport.stilt.builder.Trajectories.from_parquet",
        staticmethod(fake_from_parquet),
    )

    builder = JacobianBuilder(model)
    H = builder.build_from_target(
        [(-111.85, 40.77)], _flux_times(), "hrrr", footprint=config
    )

    traj.footprint.assert_called_once_with(config)
    assert isinstance(H, MatrixBlock)
