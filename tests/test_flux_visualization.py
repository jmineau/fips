"""Test suite for fips.problems.flux.visualization module."""

import numpy as np
import pandas as pd
import pytest

from fips import Covariance, Operator
from fips.problems.flux.problem import FluxProblem
from fips.vector import Vector

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    """Close every figure after a test, and make ``plt.show`` a no-op."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


@pytest.fixture
def solved_problem():
    """Solved flux problem: 2 times x 2 lat x 2 lon of flux, two sites of daily data."""
    rng = np.random.default_rng(0)
    flux_idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2023-01", periods=2, freq="MS"),
            [40.5, 40.9],
            [-112.0, -111.6],
        ],
        names=["time", "lat", "lon"],
    )
    days = pd.date_range("2023-01-01", periods=40, freq="D")
    # SITE2 has a 10-day gap in its data, which the concentration plot shades.
    obs_idx = pd.MultiIndex.from_tuples(
        [
            (site, day)
            for site in ["SITE1", "SITE2"]
            for i, day in enumerate(days)
            if not (site == "SITE2" and 15 <= i < 25)
        ],
        names=["obs_location", "obs_time"],
    )
    n_flux, n_obs = len(flux_idx), len(obs_idx)

    problem = FluxProblem(
        obs=Vector.from_series(
            pd.Series(
                rng.uniform(400, 410, n_obs), index=obs_idx, name="concentration"
            ),
            block="concentration",
        ),
        prior=Vector.from_series(
            pd.Series(rng.uniform(1, 2, n_flux), index=flux_idx, name="flux"),
            block="flux",
        ),
        forward_operator=Operator.from_frame(
            pd.DataFrame(
                rng.uniform(0, 1, (n_obs, n_flux)), index=obs_idx, columns=flux_idx
            ),
            row_block="concentration",
            col_block="flux",
        ),
        prior_error=Covariance.from_frame(
            pd.DataFrame(np.eye(n_flux) * 0.5, index=flux_idx, columns=flux_idx),
            row_block="flux",
            col_block="flux",
        ),
        modeldata_mismatch=Covariance.from_frame(
            pd.DataFrame(np.eye(n_obs) * 0.1, index=obs_idx, columns=obs_idx),
            row_block="concentration",
            col_block="concentration",
        ),
        constant=Vector.from_series(
            pd.Series(np.full(n_obs, 395.0), index=obs_idx, name="concentration"),
            block="concentration",
        ),
    )
    return problem.solve()


class TestFluxes:
    """Tests for FluxPlotter.fluxes."""

    @pytest.mark.parametrize(
        ("time", "has_truth", "titles"),
        [
            ("mean", False, ["Prior", "Posterior", "Residual"]),
            ("mean", True, ["Truth", "Prior", "Posterior", "Residual"]),
            (1, False, ["Prior", "Posterior", "Residual"]),
            ("std", False, ["Prior", "Posterior"]),
            ("std", True, ["Truth", "Prior", "Posterior"]),
        ],
        ids=["mean", "mean-truth", "int", "std", "std-truth"],
    )
    def test_panels(self, solved_problem, time, has_truth, titles):
        """The truth panel comes first and the residual panel last; ``std`` has none."""
        truth = solved_problem.prior_fluxes * 1.1 if has_truth else None

        fig, axes = solved_problem.plot.fluxes(time=time, truth=truth)

        assert [ax.get_title() for ax in axes] == titles
        # The panels, plus one colorbar axes for the prior and one for the residual.
        assert len(fig.axes) == len(titles) + (1 if time == "std" else 2)

    def test_sites(self, solved_problem):
        """Site markers are drawn on every panel."""
        _, axes = solved_problem.plot.fluxes(sites={"A": (40.7, -111.8)})

        assert all(len(ax.collections) >= 2 for ax in axes)

    def test_sites_must_be_a_dict(self, solved_problem):
        """Anything but a dict of sites is refused."""
        with pytest.raises(NotImplementedError, match="dict"):
            solved_problem.plot.fluxes(sites=True)

    def test_tiler(self, solved_problem):
        """With a tiler, every panel is a map with the tiles added."""
        pytest.importorskip("cartopy")
        from cartopy.io.img_tiles import GoogleTiles

        _, axes = solved_problem.plot.fluxes(tiler=GoogleTiles(), tiler_zoom=5)

        assert [type(ax).__name__ for ax in axes] == ["GeoAxes"] * 3
        assert [len(ax.img_factories) for ax in axes] == [1] * 3

    def test_tiler_without_cartopy(self, solved_problem, monkeypatch):
        """A missing cartopy raises its install hint before a figure is made."""
        import fips.problems.flux.visualization as viz

        def missing():
            raise ImportError("cartopy is required for map plotting")

        monkeypatch.setattr(viz, "_require_cartopy", missing)

        class Tiler:
            crs = None

        with pytest.raises(ImportError, match="cartopy"):
            solved_problem.plot.fluxes(tiler=Tiler())
        assert plt.get_fignums() == []


class TestConcentrations:
    """Tests for FluxPlotter.concentrations."""

    def test_defaults(self, solved_problem):
        """
        The default windows raise no warning.

        pandas 3 deprecates the lowercase alias, ``"30d"``, and the suite turns
        warnings into errors.
        """
        axes = solved_problem.plot.concentrations()

        assert axes.shape == (2,)
        # SITE2's 11-day gap is longer than the default 7-day threshold
        assert [len(ax.patches) for ax in axes] == [0, 1]

    def test_legend_and_gap(self, solved_problem):
        """Each site gets a figure; a site with a data gap adds a "no data" entry."""
        axes = solved_problem.plot.concentrations(
            rolling_window="30D", gap_threshold="5D"
        )

        assert axes.shape == (2,)
        legends = [[t.get_text() for t in ax.get_legend().get_texts()] for ax in axes]
        assert legends[0] == ["observed", "posterior", "prior"]
        assert legends[1] == ["observed", "posterior", "prior", "no data"]
        # the shaded gap is one patch
        assert [len(ax.patches) for ax in axes] == [0, 1]

    def test_location(self, solved_problem):
        """Only the requested site is plotted."""
        axes = solved_problem.plot.concentrations(
            location="SITE1", rolling_window="30D", gap_threshold="7D", show_raw=False
        )

        assert axes.shape == (1,)
        assert axes[0].get_title() == "Concentrations at site: SITE1"
        assert len(axes[0].lines) == 3

    def test_location_type(self, solved_problem):
        """A location that is not a string or a list is refused."""
        with pytest.raises(ValueError, match="location must be"):
            solved_problem.plot.concentrations(location=3)
