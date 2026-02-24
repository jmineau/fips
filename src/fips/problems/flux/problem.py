"""Atmospheric flux inversion framework.

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

from fips.estimators import Estimator
from fips.problem import InverseProblem
from fips.problems.flux.visualization import FluxPlotter

logger = logging.getLogger(__name__)

# TODO:
# - Support multiple flux sources (farfield/bio/etc)
# - Enable regridding
# - Build STILT Jacobian from geometries or nested grid


class FluxProblem(InverseProblem):
    """Atmospheric flux inversion problem.

    Subclass of InverseProblem specialized for estimating spatial and temporal surface fluxes
    from observed atmospheric concentrations using a forward transport model.
    Supports multi-block state composition (e.g., fluxes + bias corrections).
    """

    def solve(self, estimator: str | type[Estimator] = "bayesian", **kwargs):
        return super().solve(estimator=estimator, **kwargs)

    @property
    def concentrations(self) -> pd.Series:
        """Observed concentrations."""
        return self.obs["concentration"]

    @property
    def enhancement(self) -> pd.Series:
        """The background-subtracted observation vector."""
        if self.background is None:
            return self.concentrations
        return self.concentrations - self.background

    @property
    def prior_fluxes(self) -> pd.Series:
        """Prior flux inventory."""
        return self.prior["flux"]

    @property
    def jacobian(self) -> pd.DataFrame:
        """Forward operator (Jacobian) mapping state to observations."""
        return self.forward_operator["concentration", "flux"]

    @property
    def prior_flux_error(self) -> pd.DataFrame:
        """Inventory error covariance matrix."""
        return self.prior_error["flux", "flux"]

    @property
    def concentration_error(self) -> pd.DataFrame:
        """Model-data mismatch covariance matrix for observations."""
        return self.modeldata_mismatch["concentration", "concentration"]

    @property
    def background(self) -> pd.Series | None:
        """Background concentration."""
        return None if self.constant is None else self.constant["concentration"]

    @property
    def posterior_fluxes(self) -> pd.Series:
        """Posterior flux estimates after inversion."""
        return self.posterior["flux"]

    @property
    def posterior_flux_error(self) -> pd.DataFrame:
        """Posterior flux error covariance matrix after inversion."""
        return self.posterior_error["flux", "flux"]

    @property
    def prior_concentrations(self) -> pd.Series:
        """Modelled concentrations from prior fluxes."""
        return self.prior_obs["concentration"]

    @property
    def posterior_concentrations(self) -> pd.Series:
        """Modelled concentrations from posterior fluxes."""
        return self.posterior_obs["concentration"]

    @property
    def plot(self) -> FluxPlotter:
        """FluxPlotter: Plotting interface for flux inversion results."""
        return FluxPlotter(self)
