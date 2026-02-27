"""
Pipeline for atmospheric flux inversion.

This module provides the `FluxInversionPipeline` class, which extends the
base `InversionPipeline` to handle the specific requirements of flux
inversion problems, such as loading STILT footprints and building Jacobians.
"""

from abc import ABC

import numpy as np
import pandas as pd

from fips.filters import select_intervals_with_min_obs
from fips.pipeline import InversionPipeline
from fips.problems.flux.problem import FluxProblem
from fips.vector import Vector


class FluxInversionPipeline(InversionPipeline[FluxProblem], ABC):
    """
    Abstract pipeline for atmospheric flux inversions.
    """

    problem: FluxProblem

    def __init__(self, config):
        super().__init__(
            config=config,
            problem=FluxProblem,
            estimator="bayesian",
        )

    def filter_state_space(self, obs: Vector, prior: Vector) -> tuple[Vector, Vector]:
        """
        Filter state space by removing intervals with insufficient observations.

        Parameters
        ----------
        obs : Vector
            Observation vector.
        prior : Vector
            Prior state vector.

        Returns
        -------
        tuple[Vector, Vector]
            Filtered observation and prior vectors.
        """
        try:
            min_obs_per_interval = self.config.min_obs_per_interval
        except AttributeError:
            min_obs_per_interval = 1
        try:
            min_sims_per_interval = self.config.min_sims_per_interval
        except AttributeError:
            min_sims_per_interval = 1
        try:
            freq = self.config.flux_freq
        except AttributeError:
            freq = "infer"

        # Handle filtering by time intervals
        if min_obs_per_interval > 1 or min_sims_per_interval > 1:
            # Get inputs as series
            obs_series = obs.to_series()

            # Determine flux times
            times = prior.index.get_level_values("time").unique().sort_values()

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
                obs_series = select_intervals_with_min_obs(
                    obs_series,
                    intervals=bins,
                    threshold=min_obs_per_interval,
                    level="obs_time",
                )

            obs = Vector(obs_series, name=obs.name)

        return obs, prior

    def run(self, **kwargs) -> FluxProblem:
        """Run the flux inversion pipeline.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the inverse problem initialization.

        Returns
        -------
        FluxProblem
            The solved flux inversion problem.
        """
        inversion = super().run(**kwargs)

        # Print summary report
        self.summarize()

        return inversion

    def summarize(self) -> None:
        """Print a comprehensive statistical summary of the inversion results."""
        problem = self.problem

        # --- Extract Data ---
        obs = problem.concentrations
        enhancement = problem.enhancement

        prior_flux = problem.prior_fluxes
        post_flux = problem.posterior_fluxes

        prior_obs = problem.prior_concentrations
        post_obs = problem.posterior_concentrations

        est = problem.estimator

        # --- Calculate Atmospheric Metrics ---
        # Mean Enhancements
        mean_enhancement = np.mean(enhancement)

        # Root Mean Square Error (RMSE)
        rmse_prior = np.sqrt(((prior_obs - obs) ** 2).mean())
        rmse_post = np.sqrt(((post_obs - obs) ** 2).mean())

        # Mean Bias (Model - Obs)
        mb_prior = np.mean(prior_obs - obs)
        mb_post = np.mean(post_obs - obs)

        # R-squared (Coefficient of Determination)
        ss_tot = ((obs - obs.mean()) ** 2).sum()
        r2_prior = 1 - (((obs - prior_obs) ** 2).sum() / ss_tot)
        r2_post = 1 - (((obs - post_obs) ** 2).sum() / ss_tot)

        # --- Calculate Flux & Bayesian Metrics ---
        mean_prior_flux = prior_flux.mean()
        mean_post_flux = post_flux.mean()

        # Bayesian Diagnostics
        n = len(prior_flux)  # Size of state vector
        m = len(obs)  # Size of observation vector
        dofs = est.DOFS  # Degrees of Freedom for Signal
        chi2 = est.reduced_chi2  # Reduced Chi-Square
        R2 = est.R2  # Bayesian R-squared
        RMSE = est.RMSE  # Bayesian RMSE
        uncertainty_reduction = est.uncertainty_reduction * 100  # Convert to percentage

        # --- Print Report ---
        print("==================================================")
        print("          FLUX INVERSION SUMMARY REPORT           ")
        print("==================================================")
        print(f"State Vector Size (n):       {n}")
        print(f"Observation Vector Size (m): {m}")
        print(
            f"Degrees of Freedom (DOFS):   {dofs:.2f} ({(dofs / n) * 100:.1f}% of state)"
        )
        print(f"Reduced Chi-Square:          {chi2:.3f}")
        print(f"Bayesian R^2:                {R2:.3f}")
        print(f"Bayesian RMSE:               {RMSE:.3f}")
        print(f"Uncertainty Reduction:       {uncertainty_reduction:.1f}%")
        print("--------------------------------------------------")
        print("FLUX ESTIMATES:")
        print(f"  Mean Prior Flux:     {mean_prior_flux:.2f}")
        print(f"  Mean Posterior Flux: {mean_post_flux:.2f}")
        print("--------------------------------------------------")
        print("ATMOSPHERIC CONCENTRATIONS (Model vs Obs):")
        print(f"  Mean Obs Enhancement: {mean_enhancement:.2f}")
        print(f"  RMSE (Prior):         {rmse_prior:.2f}")
        print(f"  RMSE (Posterior):     {rmse_post:.2f}")
        print(f"  Mean Bias (Prior):    {mb_prior:.2f}")
        print(f"  Mean Bias (Posterior):{mb_post:.2f}")
        print(f"  R^2 (Prior):          {r2_prior:.3f}")
        print(f"  R^2 (Posterior):      {r2_post:.3f}")
        print("==================================================")
