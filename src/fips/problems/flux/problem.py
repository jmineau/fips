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
- "Forward Operator" maps fluxes to concentrations via atmospheric transport.
- "Observations" are the observed concentration values at receptor locations/times.
- "Constant" is the baseline (background) concentration not attributed to local fluxes.

The Forward Operator matrix is constructed using atmospheric transport models, currently STILT
(Stochastic Time-Inverted Lagrangian Transport), which simulates the influence of surface
fluxes on observed concentrations by generating footprints for each observation.
These footprints quantify the sensitivity of each observation to fluxes at different
locations and times, forming the basis of the Forward Operator.
This module supports building the Forward Operator from STILT simulations, specifying time bins,
grid resolutions, and parallel computation. It also provides the FluxInversion class,
which extends the base InverseProblem to handle flux-specific terminology and plotting
interfaces for visualizing results.
"""

import pandas as pd

from fips.estimators import Estimator
from fips.matrices import ForwardOperator as Jacobian
from fips.problem import InverseProblem
from fips.problems.flux.plotting import FluxPlotter
from fips.utils import filter_intervals
from fips.vectors import Block, Vector

# TODO:
# - Support multiple flux sources (farfield/bio/etc)
# - Enable regridding
# - Build STILT Jacobian from geometries or nested grid


class FluxInversion(InverseProblem):
    """
    FluxInversion: Atmospheric Flux Inversion Problem.

    Subclass of InverseProblem for estimating spatial and temporal surface fluxes
    (e.g., greenhouse gas emissions or uptake) from observed atmospheric concentrations.
    Combines observations, prior flux estimates, and a forward model (Forward Operator)
    within a statistical estimation framework.

    Supports multi-block state composition (e.g., fluxes + bias corrections).

    Attributes
    ----------
    prior : Vector
        Prior state vector (fluxes, bias, etc. as Blocks).
    obs : Vector
        Observation vector (concentrations, etc. as Blocks).
    forward_operator : ForwardOperator
        Forward operator mapping state to observations via atmospheric transport.
    prior_error : CovarianceMatrix
        Covariance matrix for prior state uncertainty.
    modeldata_mismatch : CovarianceMatrix
        Covariance matrix for observation error.
    constant : pd.Series or float or None
        Background/constant concentration to add to modelled observations.
    plot : Plotter
        Plotting interface for results.
    """

    def __init__(
        self,
        concentrations: pd.Series | Vector,
        inventory: pd.Series | Block | Vector,
        forward_operator: pd.DataFrame | Jacobian,
        prior_error,
        modeldata_mismatch,
        background: pd.Series | float | None = None,
        bias: pd.Series | Block | None = None,
        freq: str | None = "infer",
        min_obs_per_interval: int = 1,
        min_sims_per_interval: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize a flux inversion problem.

        Parameters
        ----------
        concentrations : pd.Series or Vector
            Observed concentrations. If pd.Series, automatically converted to Vector.
        inventory : pd.Series, Block, or Vector
            Prior flux inventory. If pd.Series, converted to Block; if Block, converted to Vector.
            If already a Vector, used as-is.
        forward_operator : pd.DataFrame or ForwardOperator
            Forward operator (Jacobian) mapping state to observations.
        prior_error : CovarianceMatrix
            Prior error covariance matrix for state.
        modeldata_mismatch : CovarianceMatrix
            Model-data mismatch covariance matrix for observations.
        background : pd.Series, float, or None, optional
            Background concentration to add to modelled observations, by default None.
        bias : pd.Series, Block, or None, optional
            Optional bias correction block to add to state, by default None.
        freq : str, optional
            Frequency for time binning (e.g., 'D', 'H'), by default 'infer'.
        min_obs_per_interval : int, optional
            Minimum observations per time interval to retain, by default 1.
        min_sims_per_interval : int, optional
            Minimum forward model evaluations per time interval, by default 1.
        **kwargs
            Additional keyword arguments passed to InverseProblem.

        Raises
        ------
        ValueError
            If time frequency cannot be inferred and is not specified.
        """
        # Handle filtering by time intervals (existing logic)
        if min_obs_per_interval > 1 or min_sims_per_interval > 1:
            # Determine state time bins
            if isinstance(inventory, Vector):
                state_data = list(inventory.blocks.values())[0].data
            elif isinstance(inventory, Block):
                state_data = inventory.data
            else:
                state_data = inventory

            state_time_index = state_data.index
            times = state_time_index.get_level_values("time").unique().sort_values()

            if not isinstance(times, pd.IntervalIndex):  # Assume regular intervals
                if freq == "infer":
                    freq = pd.infer_freq(times)
                    if freq is None:
                        raise ValueError(
                            "Could not infer frequency from inventory time index. "
                            "Please specify freq."
                        )
                offset = pd.tseries.frequencies.to_offset(freq)
                t0 = times.min()
                tf = times.max() + offset
                bins = pd.interval_range(start=t0, end=tf, freq=freq)
            else:  # Already interval index
                bins = times

            # Filter concentrations by interval
            if min_obs_per_interval > 1:
                if isinstance(concentrations, Vector):
                    conc_data = list(concentrations.blocks.values())[0].data
                else:
                    conc_data = concentrations

                filtered_conc = filter_intervals(
                    conc_data,
                    intervals=bins,
                    threshold=min_obs_per_interval,
                    level="obs_time",
                )
                if isinstance(concentrations, Vector):
                    # Reconstruct Vector with filtered data
                    block = Block(
                        data=filtered_conc, name=list(concentrations.blocks.keys())[0]
                    )
                    concentrations = Vector(name="obs", blocks=[block])
                else:
                    concentrations = filtered_conc

            # Filter Jacobian by interval
            if min_sims_per_interval > 1:
                if isinstance(forward_operator, Jacobian):
                    jac_data = forward_operator.data
                else:
                    jac_data = forward_operator

                filtered_jac = filter_intervals(
                    jac_data,
                    intervals=bins,
                    threshold=min_sims_per_interval,
                    level="obs_time",
                )
                forward_operator = filtered_jac

        # Convert concentrations to Vector if needed
        if isinstance(concentrations, pd.Series):
            conc_block = Block(data=concentrations, name="concentration")
            concentrations_vector = Vector(name="obs", blocks=[conc_block])
        else:
            concentrations_vector = concentrations

        # Convert inventory and bias to multi-block state Vector
        if isinstance(inventory, pd.Series):
            inventory_block = Block(data=inventory, name="flux")
        elif isinstance(inventory, Block):
            inventory_block = inventory
        elif isinstance(inventory, Vector):
            # Extract blocks from existing Vector
            state_blocks = list(inventory.blocks.values())
        else:
            raise TypeError("inventory must be pd.Series, Block, or Vector")

        # If inventory is Series/Block, create list of blocks
        if not isinstance(inventory, Vector):
            state_blocks = [inventory_block]

        # Add bias block if provided
        if bias is not None:
            if isinstance(bias, pd.Series):
                bias_block = Block(data=bias, name="bias")
            elif isinstance(bias, Block):
                bias_block = bias
            else:
                raise TypeError("bias must be pd.Series or Block")
            state_blocks.append(bias_block)

        # Create state Vector
        inventory_vector = Vector(name="prior", blocks=state_blocks)

        # Call parent with Vector objects
        super().__init__(
            obs=concentrations_vector,
            prior=inventory_vector,
            forward_operator=forward_operator,
            prior_error=prior_error,
            modeldata_mismatch=modeldata_mismatch,
            constant=background,
            **kwargs,
        )

        # Build plotting interface
        self.plot = FluxPlotter(self)

    def solve(self, estimator: str | type[Estimator] = "bayesian", **kwargs):
        return super().solve(estimator=estimator, **kwargs)
