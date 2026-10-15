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

import pandas as pd

from fips.covariance import Covariance
from fips.operator import Operator
from fips.problem import Problem
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
    corrections). Solve it with :func:`fips.solve`; the posterior's blocks
    are read the same way (``post.estimate["flux"]``), and
    :class:`~fips.problems.flux.FluxPlotter` plots a posterior.
    """

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"FluxProblem("
            f"n_obs={len(self.obs)}, n_flux={len(self.prior_fluxes)}, "
            f"n_state={len(self.prior)})"
        )

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
    def prior_concentrations(self) -> pd.Series:
        """Modelled concentrations from the prior (:meth:`~fips.Problem.predict` of it)."""
        return _block_series(self.predict(self.prior), "concentration")

    @property
    def prior_concentration_error(self) -> Covariance:
        """The prior error mapped to concentrations: the concentration block of ``H S_0 H^T``."""
        H = self.forward.data
        HS_0H = Covariance((H @ self.prior_cov.data) @ H._transpose(), self.obs.space)
        block = HS_0H["concentration", "concentration"]
        return Covariance(block.data, block.rows)
