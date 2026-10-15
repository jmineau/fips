"""
Atmospheric flux inversion framework.

Provides classes and utilities for estimating surface fluxes (emissions/uptake)
from observed atmospheric concentrations using STILT transport models.

Flux inversion is a specific application of the inverse problem framework where:
- Prior: initial estimate of surface fluxes (from inventories or models)
- Posterior: updated flux estimate after assimilating observations
- Forward Operator (Jacobian): maps fluxes to concentrations via STILT transport
- Observations: measured concentration values at receptor locations/times

The Forward Operator is built from STILT footprints which quantify the sensitivity
of each observation to fluxes at different locations and times.
"""

import logging
from typing import Self

import pandas as pd

from fips.operator import Operator
from fips.posterior import Posterior
from fips.problem import Problem
from fips.problems.flux.visualization import FluxPlotter
from fips.solve import solve
from fips.vector import Vector

logger = logging.getLogger(__name__)

# TODO:
# - Support multiple flux sources (farfield/bio/etc)
# - Enable regridding


def _block_series(vector: Vector, block: str) -> pd.Series:
    """Return one block of `vector` as a Series on the block's own labels."""
    return vector[block].to_series().droplevel("block")


class FluxProblem(Problem):
    """
    Atmospheric flux inversion problem.

    A :class:`~fips.Problem` whose state has a ``flux`` block and whose
    observations have a ``concentration`` block, with accessors for those
    blocks. Supports multi-block state composition (e.g., fluxes + bias
    corrections). :meth:`solve` keeps the posterior as :attr:`result`, which
    the posterior accessors read.
    """

    result: Posterior | None = None
    """The posterior from the last :meth:`solve`, or None."""

    def solve(self, method: str = "analytic", **options) -> Self:
        """
        Solve the flux inversion with :func:`fips.solve` and keep the posterior.

        Parameters
        ----------
        method : str, default 'analytic'
            The method :func:`fips.solve` uses.
        **options
            The method's options (``gamma``).

        Returns
        -------
        Self
            The FluxProblem, its posterior kept as :attr:`result`.
        """
        self.result = solve(self, method, **options)
        return self

    def _posterior(self) -> Posterior:
        """Return :attr:`result`, or raise if the problem has not been solved."""
        if self.result is None:
            raise RuntimeError("Problem has not been solved. Call .solve() first.")
        return self.result

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"FluxProblem("
            f"n_obs={len(self.obs)}, n_flux={len(self.prior_fluxes)}, "
            f"n_state={len(self.prior)}, solved={self.result is not None})"
        )

    def __getstate__(self) -> dict:
        """Return the inputs and the posterior."""
        return {**super().__getstate__(), "result": self.result}

    def __setstate__(self, state: dict) -> None:
        """Restore from :meth:`__getstate__`."""
        state = dict(state)
        self.result = state.pop("result", None)
        super().__setstate__(state)

    @property
    def concentrations(self) -> pd.Series:
        """Observed concentrations."""
        return _block_series(self.obs, "concentration")

    @property
    def enhancement(self) -> pd.Series:
        """The background-subtracted observation vector."""
        if self.background is None:
            return self.concentrations
        return self.concentrations - self.background

    @property
    def prior_fluxes(self) -> pd.Series:
        """Prior flux inventory."""
        return _block_series(self.prior, "flux")

    @property
    def jacobian(self) -> Operator:
        """Forward operator (Jacobian) mapping state to observations."""
        return self.forward["concentration", "flux"]

    @property
    def prior_flux_error(self) -> Operator:
        """Inventory error covariance matrix."""
        return self.prior_cov["flux", "flux"]

    @property
    def concentration_error(self) -> Operator:
        """Model-data mismatch covariance matrix for observations."""
        return self.obs_cov["concentration", "concentration"]

    @property
    def background(self) -> pd.Series | None:
        """Background concentration."""
        if self.offset is None:
            return None
        return _block_series(self.offset, "concentration")

    @property
    def posterior_fluxes(self) -> pd.Series:
        """Posterior flux estimates after inversion."""
        return _block_series(self._posterior().estimate, "flux")

    @property
    def posterior_flux_error(self) -> Operator:
        """Posterior flux error covariance matrix after inversion."""
        return self._posterior().cov()["flux", "flux"]

    @property
    def prior_concentrations(self) -> pd.Series:
        """Modelled concentrations from prior fluxes."""
        return _block_series(self._posterior().prior_modeled_obs, "concentration")

    @property
    def prior_concentration_error(self) -> Operator:
        """Covariance of the prior modelled concentrations from the prior flux error."""
        return self._posterior().prior_obs_cov["concentration", "concentration"]

    @property
    def posterior_concentrations(self) -> pd.Series:
        """Modelled concentrations from posterior fluxes."""
        return _block_series(self._posterior().modeled_obs, "concentration")

    @property
    def plot(self) -> FluxPlotter:
        """FluxPlotter: Plotting interface for flux inversion results."""
        return FluxPlotter(self)
