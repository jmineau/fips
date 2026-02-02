"""Atmospheric flux inversion framework.

Provides classes and utilities for estimating surface fluxes (emissions/uptake)
from observed atmospheric concentrations using STILT transport models.

Flux inversion is a specific application of the inverse problem framework where:
- Prior: initial estimate of surface fluxes (from inventories or models)
- Posterior: updated flux estimate after assimilating observations
- Forward Operator: maps fluxes to concentrations via STILT transport
- Observations: measured concentration values at receptor locations/times

The Forward Operator is built from STILT footprints which quantify the sensitivity
of each observation to fluxes at different locations and times.
"""

import pandas as pd

from fips.converters import to_frame, to_series
from fips.covariance import CovarianceMatrix
from fips.estimators import Estimator
from fips.indices import (
    ensure_block,
    ensure_block_axis,
    outer_align_levels,
    select_intervals_with_min_obs,
)
from fips.operators import ForwardOperator
from fips.problem import InverseProblem
from fips.problems.flux.visualization import FluxPlotter
from fips.structures import Block, Vector

# TODO:
# - Support multiple flux sources (farfield/bio/etc)
# - Enable regridding
# - Build STILT Jacobian from geometries or nested grid


class FluxInversion(InverseProblem):
    """Atmospheric flux inversion problem.

    Subclass of InverseProblem specialized for estimating spatial and temporal surface fluxes
    from observed atmospheric concentrations using a forward transport model.
    Supports multi-block state composition (e.g., fluxes + bias corrections).

    Attributes
    ----------
    prior : Vector
        Prior state vector (fluxes, bias, etc.).
    obs : Vector
        Observation vector (concentrations, etc.).
    forward_operator : ForwardOperator
        Sensitivity matrix mapping state to observations.
    prior_error : CovarianceMatrix
        Prior state uncertainty covariance.
    modeldata_mismatch : CovarianceMatrix
        Observation error covariance.
    constant : pd.Series, float, or None
        Background concentration.
    plot : FluxPlotter
        Plotting interface for results.
    """

    def __init__(
        self,
        concentrations: pd.Series | Block | Vector,
        inventory: pd.Series | Block | Vector,
        jacobian: pd.DataFrame | ForwardOperator,
        inventory_error: pd.DataFrame | CovarianceMatrix,
        modeldata_mismatch: pd.DataFrame | CovarianceMatrix,
        background: pd.Series | Block | Vector | float | None = None,
        bias: pd.Series | Vector | None = None,  # this would be the prior
        bias_error: pd.DataFrame
        | CovarianceMatrix
        | None = None,  # then we would just add an index for each index in the block
        bias_jacobian: float | pd.DataFrame | ForwardOperator | None = 1.0,
        freq: str | None = "infer",
        min_obs_per_interval: int = 1,
        min_sims_per_interval: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize a flux inversion problem.

        Parameters
        ----------
        concentrations : pd.Series, Block, or Vector
            Observed concentrations. If pd.Series, automatically converted to Vector.
        inventory : pd.Series, Block, or Vector
            Prior flux inventory. If pd.Series, converted to Block; if Block, converted to Vector.
            If already a Vector, used as-is.
        jacobian : pd.DataFrame or ForwardOperator
            Jacobian (forward operator) mapping state to observations.
        inventory_error : CovarianceMatrix
            Inventory error covariance matrix for state.
        modeldata_mismatch : CovarianceMatrix
            Model-data mismatch covariance matrix for observations.
        background : pd.Series, float, or None, optional
            Background concentration to add to modelled observations, by default None.
        bias : pd.Series, Block, or Vector, or None, optional
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

        # ----- NORMALIZE INPUTS -----

        inventory = to_series(inventory)
        concentrations = to_series(concentrations)
        jacobian = to_frame(jacobian)
        inventory_error = to_frame(inventory_error)
        modeldata_mismatch = to_frame(modeldata_mismatch)
        background = to_series(background) if background is not None else None

        # ----- FILTER INTERVALS -----

        # Handle filtering by time intervals
        if min_obs_per_interval > 1 or min_sims_per_interval > 1:
            # Determine flux times
            times = inventory.index.get_level_values("time").unique().sort_values()

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
                concentrations = select_intervals_with_min_obs(
                    concentrations,
                    intervals=bins,
                    threshold=min_obs_per_interval,
                    level="obs_time",
                )

            # Filter Jacobian by interval
            if min_sims_per_interval > 1:
                jacobian = select_intervals_with_min_obs(
                    jacobian,
                    intervals=bins,
                    threshold=min_sims_per_interval,
                    level="obs_time",
                )

        # ----- PREPARE BLOCKS -----

        state_blocks = []

        # Prepare inventory block
        inventory = Block(name="flux", data=inventory)
        state_blocks.append(inventory)

        # Prepare concentration block
        concentrations = Block(name="concentration", data=concentrations)

        # Normalize block names for alignment
        if isinstance(background, pd.Series):
            background = Block(name="concentration", data=background)

        jacobian = ensure_block_axis(jacobian, "index", "concentration")
        jacobian = ensure_block_axis(jacobian, "columns", "flux")
        modeldata_mismatch = ensure_block(modeldata_mismatch, "concentration")
        inventory_error = ensure_block(inventory_error, "flux")

        # Add bias block if provided
        if bias is not None:
            # Prepare bias state block
            if isinstance(bias, pd.Series):
                bias_block = Block(data=bias, name="bias")
            elif isinstance(bias, Block):
                bias_block = bias
            else:
                raise TypeError("bias must be pd.Series or Block")

            state_blocks.append(bias_block)

            # Expand prior_error to include bias block
            if bias_error is None:
                raise ValueError("bias_error must be provided if bias block is used")
            elif isinstance(bias_error, CovarianceMatrix):
                bias_error = bias_error.data

            bias_error = ensure_block(bias_error, "bias")

            inventory_error, bias_error = outer_align_levels(
                [inventory_error, bias_error], axis="both"
            )
            prior_error = pd.concat([inventory_error, bias_error], axis=0).fillna(0.0)

            # Expand jacobian to include bias block
            if bias_jacobian is None:
                raise ValueError("bias_jacobian must be provided if bias block is used")
            elif isinstance(bias_jacobian, ForwardOperator):
                bias_jacobian = bias_jacobian.data
            elif isinstance(bias_jacobian, (float, int)):
                bias_jacobian = pd.DataFrame(
                    bias_jacobian,
                    index=jacobian.index,
                    columns=pd.MultiIndex.from_product(
                        [["bias"], bias_block.data.index],
                        names=["block", "state_index"],
                    ),
                )

            bias_jacobian = ensure_block_axis(bias_jacobian, "columns", "bias")
            bias_jacobian = ensure_block_axis(bias_jacobian, "index", "concentration")
            print(f"{bias_jacobian.iloc[-5:, -5:] = }")
            print(f"{jacobian.iloc[-5:, -5:] = }")

            jacobian, bias_jacobian = outer_align_levels(
                [jacobian, bias_jacobian], axis=1
            )
            jacobian = pd.concat([jacobian, bias_jacobian], axis=1).fillna(0.0)
            print(f"merged {jacobian.iloc[-5:, -5:] = }")
        else:
            prior_error = inventory_error

        # ----- Assemble Blocks -----

        # Create state Vector
        prior = Vector(name="prior", blocks=state_blocks)

        # ----- INITIALIZE INVERSE PROBLEM -----

        super().__init__(
            obs=concentrations,
            prior=prior,
            forward_operator=jacobian,
            prior_error=prior_error,
            modeldata_mismatch=modeldata_mismatch,
            constant=background,
            **kwargs,
        )

    def solve(self, estimator: str | type[Estimator] = "bayesian", **kwargs):
        return super().solve(estimator=estimator, **kwargs)

    @property
    def plot(self) -> FluxPlotter:
        """FluxPlotter: Plotting interface for flux inversion results."""
        return FluxPlotter(self)
