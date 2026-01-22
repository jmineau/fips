"""
Flux inversion module.

This module provides classes and utilities for performing atmospheric flux inversion,
a technique used to estimate surface fluxes (such as greenhouse gas emissions or uptake)
from observed atmospheric concentrations. Flux inversion is a specific application of
the general inverse problem framework, where the goal is to infer unknown fluxes
(posterior) given observed concentrations, a prior flux inventory, and a model
of atmospheric transport.

Key terminology differences from the base inverse problem:
- "Prior" refers to the initial estimate of surface fluxes (e.g., from inventories or models).
- "Posterior" refers to the updated estimate of fluxes after assimilating observations.
- "Jacobian" (or "Forward Operator") maps fluxes to concentrations via atmospheric transport.
- "Concentrations" are the observed values at receptor locations/times.
- "Background" is the baseline (constant) concentration not attributed to local fluxes.

The Jacobian matrix is constructed using atmospheric transport models, currently STILT
(Stochastic Time-Inverted Lagrangian Transport), which simulates the influence of surface
fluxes on observed concentrations by generating footprints for each observation.
These footprints quantify the sensitivity of each observation to fluxes at different
locations and times, forming the basis of the Jacobian.
This module supports building the Jacobian from STILT simulations, specifying time bins,
grid resolutions, and parallel computation. It also provides the FluxInversion class,
which extends the base InverseProblem to handle flux-specific terminology and plotting
interfaces for visualizing results.
"""

import pandas as pd

from fips.estimators import Estimator
from fips.matrices import CovarianceMatrix
from fips.mixins import AttributeMapperMixin
from fips.operators import ForwardOperator as Jacobian
from fips.problem import InverseProblem
from fips.problems.flux.plotter import Plotter
from fips.utils import filter_intervals

# TODO:
# eventually want to support multiple flux source (farfield/bio/etc)
# enable regridding
# build stilt jacobian from geometries or nested grid
# ability to extend state elements


class FluxInversion(AttributeMapperMixin, InverseProblem):
    """
    FluxInversion: Atmospheric Flux Inversion Problem.

    Subclass of InverseProblem for estimating spatial and temporal surface fluxes
    (e.g., greenhouse gas emissions or uptake) from observed atmospheric concentrations.
    Combines observations, prior flux estimates, and a forward model (Jacobian)
    within a statistical estimation framework.

    Attributes
    ----------
    concentrations : pd.Series
        Observed concentrations used in the inversion.
    inventory : pd.Series
        Prior flux inventory.
    jacobian : pd.DataFrame
        Forward operator mapping fluxes to concentrations.
    prior_error : CovarianceMatrix
        Covariance matrix representing uncertainty in the prior flux inventory.
    modeldata_mismatch : CovarianceMatrix
        Covariance matrix representing uncertainty in observed concentrations and model-data mismatch.
    background : pd.Series, float, or None
        Background concentration.
    estimator : type[Estimator] or str, optional
        Estimation method or class to use for the inversion (e.g., 'bayesian'). Default is 'bayesian'.
    posterior_fluxes : pd.Series
        Estimated fluxes after inversion (posterior).
    posterior_concentrations : pd.Series
        Modelled concentrations using posterior fluxes.
    prior_concentrations : pd.Series
        Modelled concentrations using prior fluxes.
    plot : Plotter
        Diagnostic and plotting interface.
    """

    _attribute_map = {
        "concentrations": "obs",
        "inventory": "prior",
        "jacobian": "forward_operator",
        "background": "constant",
        "posterior_fluxes": "posterior",
        "posterior_concentrations": "posterior_obs",
        "prior_concentrations": "prior_obs",
    }
    _read_only_attributes = {
        "posterior_fluxes",
        "posterior_concentrations",
        "prior_concentrations",
    }

    # Type annotations for attributes
    obs_index: pd.MultiIndex
    state_index: pd.MultiIndex
    concentrations: pd.Series
    inventory: pd.Series
    jacobian: Jacobian | pd.DataFrame
    prior_error: CovarianceMatrix
    modeldata_mismatch: CovarianceMatrix
    background: pd.Series
    posterior_fluxes: pd.Series
    posterior_concentrations: pd.Series
    prior_concentrations: pd.Series

    def __init__(
        self,
        concentrations: pd.Series,
        inventory: pd.Series,
        jacobian: Jacobian | pd.DataFrame,
        prior_error: CovarianceMatrix,
        modeldata_mismatch: CovarianceMatrix,
        background: pd.Series | float | None = None,
        state_index: pd.Index | None = None,
        estimator: type[Estimator] | str = "bayesian",
        freq: str | None = "infer",
        min_obs_per_interval: int = 1,
        min_sims_per_interval: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize a flux inversion problem.

        Parameters
        ----------
        concentrations : pd.Series
            Observed concentrations with a multi-index of (obs_location, obs_time).
        inventory : pd.Series
            Prior flux inventory with a multi-index of (time, lat, lon).
        jacobian : Jacobian | pd.DataFrame
            Jacobian matrix mapping fluxes to concentrations.
        prior_error : CovarianceMatrix
            Prior error covariance matrix.
        modeldata_mismatch : CovarianceMatrix
            Model-data mismatch covariance matrix.
        background : pd.Series | float | None, optional
            Background concentration to add to modelled concentrations, by default None.
        estimator : type[Estimator] | str, optional
            Estimator class or name to use for the inversion, by default 'bayesian'.
        kwargs : dict, optional
            Additional keyword arguments to pass to the InverseProblem constructor.
        """
        # Drop time steps with insufficient observations/simulations
        # TODO i think this should be moved after the super init
        # but then we might lose our initial indices and cannot infer time bins
        if min_obs_per_interval > 1 or min_sims_per_interval > 1:
            # Determine state time bins
            state_time_index = state_index or inventory.index
            times = state_time_index.get_level_values("time").unique().sort_values()
            if not isinstance(times, pd.IntervalIndex):  # Assume regular intervals
                if freq == "infer":
                    freq = pd.infer_freq(times)
                    if freq is None:
                        raise ValueError(
                            "Could not infer frequency from inventory time index. Please specify freq."
                        )
                offset = pd.tseries.frequencies.to_offset(freq)
                t0 = times.min()
                tf = times.max() + offset
                bins = pd.interval_range(start=t0, end=tf, freq=freq)
            else:  # Already interval index
                # If the supplied index is already an IntervalIndex, we dont need to infer freq
                bins = times  # The supplied intervals can be regular or irregular

            if min_obs_per_interval > 1:
                concentrations = filter_intervals(
                    concentrations,
                    intervals=bins,
                    threshold=min_obs_per_interval,
                    level="obs_time",
                )
            if min_sims_per_interval > 1:
                if isinstance(jacobian, Jacobian):
                    jacobian = jacobian.data
                jacobian = filter_intervals(
                    jacobian,
                    intervals=bins,
                    threshold=min_sims_per_interval,
                    level="obs_time",
                )

        # Initialize inverse problem, aligning indices
        super().__init__(
            estimator=estimator,
            obs=concentrations,
            prior=inventory,
            forward_operator=jacobian,
            prior_error=prior_error,
            modeldata_mismatch=modeldata_mismatch,
            constant=background,
            state_index=state_index,
            **kwargs,
        )

        # Build plotting interface
        self.plot = Plotter(self)
