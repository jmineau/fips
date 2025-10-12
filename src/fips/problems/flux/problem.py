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


import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd

from fips import estimators  # register estimators
from fips.core import (
    SymmetricMatrix,
    Estimator,
    ForwardOperator as Jacobian,
    InverseProblem
)

# TODO:
    # eventually want to support multiple flux source (farfield/bio/etc)
    # enable regridding
    # build stilt jacobian from geometries or nested grid
    # ability to extend state elements


class FluxInversion(InverseProblem):
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
    plot : _Plotter
        Diagnostic and plotting interface.
    """

    def __init__(self,
                 concentrations: pd.Series,
                 inventory: pd.Series,
                 jacobian: Jacobian | pd.DataFrame,
                 prior_error: SymmetricMatrix,
                 modeldata_mismatch: SymmetricMatrix,
                 background: pd.Series | float | None = None,
                 estimator: type[Estimator] | str = 'bayesian',
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

        super().__init__(
            estimator=estimator,
            obs=concentrations,
            prior=inventory,
            forward_operator=jacobian,
            prior_error=prior_error,
            modeldata_mismatch=modeldata_mismatch,
            constant=background,
            **kwargs,
        )

        # Build plotting interface
        self.plot = _Plotter(self)

    @property
    def concentrations(self) -> pd.Series:
        return self.obs

    @property
    def inventory(self) -> pd.Series:
        return self.prior

    @property
    def jacobian(self) -> pd.DataFrame:
        return self.forward_operator

    @property
    def background(self) -> pd.Series | float | None:
        return self.constant

    @property
    def posterior_fluxes(self) -> pd.Series:
        return self.posterior

    @property
    def posterior_concentrations(self) -> pd.Series:
        return self.posterior_obs

    @property
    def prior_concentrations(self) -> pd.Series:
        return self.prior_obs


class _Plotter:
    """ Plotting interface for FluxInversion results."""

    def __init__(self, inversion: 'FluxInversion'):
        self.inversion = inversion

    def fluxes(self, time='mean', truth=None, **kwargs):
        """
        Plot prior & Posterior fluxes.

        Parameters
        ----------
        time : 'mean' | 'std' | int | pd.Timestamp, optional
            Time to plot. Can be 'mean' or 'std' to plot the mean or standard deviation
            over time, an integer to plot a specific time index, or a pd.Timestamp to plot a specific time.
            By default 'mean'.
        tiler : cartopy.io.img_tiles.GoogleTiles | None, optional
            Tiler to use for background map, by default None.
            If provided, the tiler will be used to add a background map to the plots.
        truth : pd.Series | None, optional
            Truth fluxes to plot for comparison, by default None.
            Residual will be calculated as posterior - truth if provided,
            otherwise as posterior - prior.
        **kwargs : dict
            Additional keyword arguments to pass to xarray plotting functions.

        Returns
        -------
        fig, axes : matplotlib.figure.Figure, np.ndarray
            Figure and axes objects.
        """
        # Get xarray representations of fluxes
        prior = self.inversion.xr.prior
        posterior = self.inversion.xr.posterior_fluxes

        # Filter/aggregate by time
        if time == 'mean':
            prior = prior.mean(dim='time')
            posterior = posterior.mean(dim='time')
        elif time == 'std':
            prior = prior.std(dim='time')
            posterior = posterior.std(dim='time')
        elif isinstance(time, int):
            prior = prior.isel(time=time)
            posterior = posterior.isel(time=time)
        else:
            prior = prior.sel(time=time)
            posterior = posterior.sel(time=time)

        # Get tiler and projection from kwargs
        tiler = kwargs.pop('tiler', None)
        subplot_kw = kwargs.pop('subplot_kw', {})
        if tiler is not None:
            subplot_kw['projection'] = tiler.crs

        ncols = 3
        if time == 'std':
            ncols -= 1
        if truth is not None:
            ncols += 1
            if isinstance(truth, pd.Series):
                truth = truth.to_xarray()
            if time == 'mean':
                truth = truth.mean(dim='time')
            elif time == 'std':
                truth = truth.std(dim='time')
            elif isinstance(time, int):
                truth = truth.isel(time=time)
            else:
                truth = truth.sel(time=time)

        # Create figure and axes
        fig, axes = plt.subplots(ncols=ncols, sharey=True,
                                 subplot_kw=subplot_kw)

        if truth is None:
            ax_prior = axes[0]
            ax_post = axes[1]
        else:
            ax_truth = axes[0]
            ax_prior = axes[1]
            ax_post = axes[2]
        if time != 'std':
            ax_res = axes[-1]

        # Add background tiles
        if tiler is not None:
            tiler_zoom = kwargs.pop('tiler_zoom', 10)
            extent = [posterior.lon.min(), posterior.lon.max(),
                    posterior.lat.min(), posterior.lat.max()]
            for ax in axes:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_image(tiler, tiler_zoom)
            if 'lat' in posterior.dims:
                kwargs['transform'] = ccrs.PlateCarree()
            else:
                # TODO handle projected data (i could use my crs class)
                raise ValueError('Cannot determine coordinate reference system for plotting.')

        # Colorbar and plot options
        vmin = min(prior.min(), posterior.min())
        vmax = max(prior.max(), posterior.max())
        if truth is not None:
            vmin = min(vmin, truth.min())
            vmax = max(vmax, truth.max())
        alpha = kwargs.pop('alpha', 0.55)
        cmap = kwargs.pop('cmap', 'RdBu_r' if vmin < 0 else 'Reds')
        if vmin < 0:
            center = 0
            vmin = None  # cant set both vmin/vmax and center
        else:
            center = None

        # Set colorbar axis below both plots
        fig.subplots_adjust(bottom=0.15)
        ax1 = axes[0]
        cbar_ax1_width = ax_post.get_position().x1 - ax1.get_position().x0
        cbar_ax1 = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0 - 0.1, cbar_ax1_width, 0.05])
        
        if truth is not None:
            truth.plot(ax=ax_truth, x='lon', y='lat', vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha,
                       add_colorbar=False, center=center, **kwargs)
            ax_truth.set(title='Truth',
                         xlabel=None,
                         ylabel=None)

        prior.plot(ax=ax_prior, x='lon', y='lat', vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha,
                       cbar_ax=cbar_ax1, cbar_kwargs={'orientation': 'horizontal', 'label': 'Flux'},
                       center=center, **kwargs)
        posterior.plot(ax=ax_post, x='lon', y='lat', vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha,
                       add_colorbar=False, center=center, **kwargs)

        # Add title and time text
        fig.suptitle("Flux Maps", fontsize=16, y=ax_prior.get_position().y1 + 0.13)
        fig.text(0.5, ax_prior.get_position().y1 + 0.05, f"time = {time}", ha='center', va='bottom', fontsize=10)

        # Set labels for each subplot
        ax_prior.set(title='Prior',
                     xlabel=None,
                     ylabel=None)
        ax_post.set(title='Posterior',
                    xlabel=None,
                    ylabel=None)

        # Plot residual
        if time != 'std':
            if truth is not None:
                base = truth
                label = 'Posterior - Truth'
            else:
                base = prior
                label = 'Posterior - Prior'
            residual_cmap = kwargs.pop('residual_cmap', 'PiYG')
            cbar_ax2 = fig.add_axes([ax_res.get_position().x0, ax_res.get_position().y0 - 0.1, ax_res.get_position().width, 0.05])
            (posterior - base).plot(ax=ax_res, x='lon', y='lat', cmap=residual_cmap, alpha=alpha, center=0,
                                         cbar_ax=cbar_ax2, cbar_kwargs={'orientation': 'horizontal',
                                                                        'label': label},
                                         **kwargs)

            ax_res.set(title='Residual',
                       xlabel=None,
                       ylabel=None)

        return fig, axes

    def concentrations(self, location=None, **kwargs):
        """
        Plot observed, prior, & posterior concentrations.
        
        Parameters
        ----------
        location : str | list[str] | None, optional
            Observation location(s) to plot. If None, plots all locations.
            By default None.
        **kwargs : dict
            Additional keyword arguments to pass to pandas plotting functions.

        Returns
        -------
        axes : list[matplotlib.axes.Axes]
            List of axes objects.
        """
        obs = self.inversion.concentrations
        posterior = self.inversion.posterior_concentrations
        prior = self.inversion.prior_concentrations

        data = pd.concat([obs, posterior, prior], axis=1)

        if location is None:
            locations = data.index.get_level_values('obs_location').unique()
        elif isinstance(location, str):
            locations = [location]
        elif isinstance(location, list):
            locations = location
        else:
            raise ValueError('location must be None, a string, or a list of strings')

        axes = []
        for location in locations:
            df = data.loc[location]
            df.columns.name = None

            fig, ax = plt.subplots()

            df.plot(ax=ax, style='.', alpha=0.6, color=['black', 'red', 'blue'], markeredgecolor='None', legend=False)
            df.rolling(window=max(1, int(len(df)/10)), center=True).mean().plot(ax=ax, linewidth=2,
                                                                                  color=['black', 'red', 'blue'],
                                                                                  label=['Observed', 'Posterior', 'Prior'],)
            ax.set(title=f'Concentrations at {location}', ylabel='Concentration', xlabel='Time')
            fig.autofmt_xdate()
            axes.append(ax)
            plt.show()
        return axes