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
from fips.filters import select_intervals_with_min_obs
from fips.matrix import MatrixBlock, MatrixBlockLike
from fips.problem import InverseProblem
from fips.problems.flux.visualization import FluxPlotter
from fips.vector import Block, BlockLike

logger = logging.getLogger(__name__)

# TODO:
# - Support multiple flux sources (farfield/bio/etc)
# - Enable regridding
# - Build STILT Jacobian from geometries or nested grid


class FluxInversion(InverseProblem):
    """Atmospheric flux inversion problem.

    Subclass of InverseProblem specialized for estimating spatial and temporal surface fluxes
    from observed atmospheric concentrations using a forward transport model.
    Supports multi-block state composition (e.g., fluxes + bias corrections).
    """

    def __init__(
        self,
        concentrations: BlockLike,
        inventory: BlockLike,
        jacobian: MatrixBlockLike,
        inventory_error: MatrixBlockLike,
        modeldata_mismatch: MatrixBlockLike,
        background: "BlockLike | float | None" = None,
        bias: "BlockLike | None" = None,  # this would be the prior
        bias_error: "MatrixBlockLike | None" = None,  # then we would just add an index for each index in the block
        bias_jacobian: "MatrixBlockLike | float | None" = 1.0,
        freq: str | None = "infer",
        min_obs_per_interval: int = 1,
        min_sims_per_interval: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize a flux inversion problem.

        Parameters
        ----------
        concentrations : BlockLike
            Observed concentrations.
        inventory : BlockLike
            Prior flux inventory.
        jacobian : MatrixBlockLike
            Jacobian (forward operator) mapping state to observations.
        inventory_error : MatrixBlockLike
            Inventory error covariance matrix for state.
        modeldata_mismatch : MatrixBlockLike
            Model-data mismatch covariance matrix for observations.
        background : BlockLike, float, or None, optional
            Background concentration to add to modelled observations, by default None.
        bias : BlockLike, float, or None, optional
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

        # ----- Normalize Inputs -----
        obs_blk = Block(concentrations, name="concentration")
        flux_blk = Block(inventory, name="flux")
        jac_blk = MatrixBlock(jacobian, "concentration", "flux")
        inv_err_blk = MatrixBlock(inventory_error, "flux", "flux")
        mdm_blk = MatrixBlock(modeldata_mismatch, "concentration", "concentration")
        bg_blk = (
            Block(background, name="background") if background is not None else None
        )

        # ----- FIlter Intervals -----

        # Handle filtering by time intervals
        if min_obs_per_interval > 1 or min_sims_per_interval > 1:
            # Get inputs as series
            inventory = flux_blk.to_series()
            concentrations = obs_blk.to_series()

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

            obs_blk = Block(concentrations, name="concentration")
            flux_blk = Block(inventory, name="flux")

        # ----- Aggregate Blocks -----

        state_blocks = [flux_blk]
        fo_blks = [jac_blk]
        S_0_blks = [inv_err_blk]
        S_z_blks = [mdm_blk]

        # Add bias block if provided
        if bias is not None:
            bias_blk = Block(bias, name="bias")
            state_blocks.append(bias_blk)

            if any(x is None for x in [bias_error, bias_jacobian]):
                raise ValueError(
                    "bias error and bias jacobian must be provided if bias block is used"
                )

            if isinstance(bias_jacobian, (float, int)):
                # Create a simple Jacobian block that maps bias to concentrations with the specified scalar value
                bias_jac_blk = MatrixBlock(
                    bias_jacobian,
                    "concentration",
                    "bias",
                    name="bias_jacobian",
                    index=jac_blk.index,
                    columns=bias_blk.index,
                )
            else:
                bias_jac_blk = MatrixBlock(bias_jacobian, "concentration", "bias")
            fo_blks.append(bias_jac_blk)

            S_0_blks.append(MatrixBlock(bias_error, "bias", "bias"))

        # ----- Initialize Inverse Problem -----

        super().__init__(
            obs=obs_blk,
            prior=state_blocks,
            forward_operator=fo_blks,
            prior_error=S_0_blks,
            modeldata_mismatch=S_z_blks,
            constant=bg_blk,
            **kwargs,
        )

    def solve(self, estimator: str | type[Estimator] = "bayesian", **kwargs):
        return super().solve(estimator=estimator, **kwargs)

    @property
    def concentrations(self) -> pd.Series:
        """Observed concentrations."""
        return self.obs["concentration"]

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
        return None if self.constant is None else self.constant["background"]

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
