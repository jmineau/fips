"""
Pipeline for atmospheric flux inversion.

This module provides the `FluxInversionPipeline` class, which extends the
base `InversionPipeline` to handle the specific requirements of flux
inversion problems, such as a forward operator from transport footprints.
"""

import logging
from abc import ABC

import numpy as np
import pandas as pd

from fips.filters import select_intervals_with_min_obs
from fips.pipeline import InversionPipeline
from fips.posterior import Posterior
from fips.problems.flux.problem import FluxProblem, _block_series
from fips.vector import Vector

logger = logging.getLogger(__name__)


class FluxInversionPipeline(InversionPipeline[FluxProblem], ABC):
    """
    Abstract pipeline for atmospheric flux inversions.
    """

    problem: FluxProblem

    def __init__(self, config):
        super().__init__(config=config, problem=FluxProblem)

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

        Raises
        ------
        TypeError
            If the frequency of the flux times has to be inferred (``flux_freq``
            is not set in the config) and the ``time`` level of ``prior`` is not
            datetimes.
        ValueError
            If the frequency cannot be inferred from the flux times.
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
                    if not isinstance(times, pd.DatetimeIndex):
                        raise TypeError(
                            "Cannot infer the frequency of the inventory time index: "
                            f"the 'time' level has dtype {times.dtype}, not datetime. "
                            "Convert it to datetimes or specify freq."
                        )
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

            obs = Vector.from_series(obs_series)

        return obs, prior

    def run(self, options: dict | None = None, **kwargs) -> Posterior:
        """
        Run the flux inversion pipeline.

        Parameters
        ----------
        options : dict, optional
            The method's options, passed to :func:`fips.solve` (``gamma``).
        **kwargs
            Additional keyword arguments passed to the problem's constructor.

        Returns
        -------
        Posterior
            The posterior, kept as :attr:`posterior`; the problem is kept as
            :attr:`problem`.
        """
        posterior = super().run(options=options, **kwargs)

        # Print summary report
        self.summarize()

        return posterior

    def summarize(self) -> None:
        """
        Print a comprehensive statistical summary of the inversion results.

        It reads :attr:`problem` and :attr:`posterior`, which :meth:`run` sets.

        The report ends with the trend of the mean posterior flux when the flux
        has more than two timesteps. The trend is a rate per year, so it is left
        out when the ``time`` level of the flux is not datetimes; the rest of the
        report is the same, and the skip is logged at ``INFO`` level.
        """
        from scipy import stats

        problem = self.problem
        post = self.posterior

        # --- Extract Data ---
        obs = problem.concentrations
        enhancement = problem.enhancement

        prior_flux = problem.prior_fluxes
        post_flux = _block_series(post.estimate, "flux")

        prior_obs = problem.prior_concentrations
        post_obs = _block_series(post.modeled_obs, "concentration")

        # --- Calculate Atmospheric Metrics ---
        mean_enhancement = np.mean(enhancement)

        rmse_prior = np.sqrt(((prior_obs - obs) ** 2).mean())
        rmse_post = np.sqrt(((post_obs - obs) ** 2).mean())

        mb_prior = np.mean(prior_obs - obs)
        mb_post = np.mean(post_obs - obs)

        ss_tot = ((obs - obs.mean()) ** 2).sum()
        r2_prior = 1 - (((obs - prior_obs) ** 2).sum() / ss_tot)
        r2_post = 1 - (((obs - post_obs) ** 2).sum() / ss_tot)

        # --- Calculate Flux & Bayesian Metrics ---
        mean_prior_flux = prior_flux.mean()
        mean_post_flux = post_flux.mean()

        n = len(prior_flux)
        m = len(obs)
        dofs = post.dofs
        chi2 = post.reduced_chi2
        R2 = post.r2
        RMSE = post.rmse
        uncertainty_reduction = post.uncertainty_reduction * 100

        # --- Negative cells ---
        n_negative = int((post_flux < 0).sum())
        pct_negative = n_negative / n * 100

        # --- Per-timestep means ---
        time_level = "time"
        prior_by_time = prior_flux.groupby(level=time_level).mean()
        post_by_time = post_flux.groupby(level=time_level).mean()
        change_pct = ((post_by_time / prior_by_time) - 1) * 100

        # --- Trend ---
        trend_str = None
        if len(post_by_time) > 2:
            t = post_by_time.index
            if isinstance(t, pd.DatetimeIndex):
                t_years = t.year + (t.day_of_year - 1) / 365.25
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    t_years, post_by_time.values
                )
                trend_str = (
                    f"  Slope:   {slope:+.4f} /yr\n"
                    f"  R^2:     {r_value**2:.3f}\n"
                    f"  p-value: {p_value:.3g}"
                )
            else:
                logger.info(
                    "Skipping the flux trend: the 'time' level of the flux has "
                    f"dtype {t.dtype}, not datetime, and the trend is a rate per year."
                )

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
        print("FLUX ESTIMATES (per grid cell):")
        print(f"  Mean Prior Flux:     {mean_prior_flux:.4f}")
        print(f"  Mean Posterior Flux: {mean_post_flux:.4f}")
        print(f"  Negative Cells:     {n_negative} / {n} ({pct_negative:.1f}%)")
        print("--------------------------------------------------")
        print("ATMOSPHERIC CONCENTRATIONS (Model vs Obs):")
        print(f"  Mean Obs Enhancement: {mean_enhancement:.4f}")
        print(f"  RMSE (Prior):         {rmse_prior:.4f}")
        print(f"  RMSE (Posterior):     {rmse_post:.4f}")
        print(f"  Mean Bias (Prior):    {mb_prior:.4f}")
        print(f"  Mean Bias (Posterior):{mb_post:.4f}")
        print(f"  R^2 (Prior):          {r2_prior:.3f}")
        print(f"  R^2 (Posterior):      {r2_post:.3f}")
        print("--------------------------------------------------")
        print("MEAN FLUX BY TIMESTEP:")
        timestep_df = pd.DataFrame(
            {
                "prior": prior_by_time,
                "posterior": post_by_time,
                "change_%": change_pct,
            }
        )
        print(timestep_df.to_string(float_format="%.4f"))
        if trend_str is not None:
            print("--------------------------------------------------")
            print("POSTERIOR FLUX TREND:")
            print(trend_str)
        print("==================================================")
