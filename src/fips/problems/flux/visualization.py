from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fips.problems.flux.problem import FluxInversion


class FluxPlotter:
    """Plotting interface for FluxInversion results."""

    def __init__(self, inversion: "FluxInversion"):
        self.inversion = inversion

    def fluxes(
        self,
        time="mean",
        truth=None,
        x_dim="lon",
        y_dim="lat",
        time_dim="time",
        sites=False,
        sites_kwargs=None,
        **kwargs,
    ):
        """
        Plot prior & Posterior fluxes.

        Parameters
        ----------
        time : 'mean' | 'std' | int | pd.Timestamp, optional
            Time to plot. Can be 'mean' or 'std' to plot the mean or standard deviation
            over time, an integer to plot a specific time index, or a pd.Timestamp to plot a specific time.
            By default 'mean'.
        truth : pd.Series | None, optional
            Truth fluxes to plot for comparison, by default None.
            Residual will be calculated as posterior - truth if provided,
            otherwise as posterior - prior.
        x_dim : str, optional
            Name of the x-coordinate dimension (e.g., 'lon'), by default 'lon'.
        y_dim : str, optional
            Name of the y-coordinate dimension (e.g., 'lat'), by default 'lat'.
        time_dim : str, optional
            Name of the time dimension, by default 'time'.
        sites : bool | dict | None, optional
            Whether to overlay observation site locations on the plots.
            Can be:
            - False or None: no sites plotted (default)
            - True: plot sites (requires sites to be stored in inversion object)
            - dict: dictionary mapping site/location IDs to (latitude, longitude) tuples.
              Example: {'site_1': (40.5, -111.5), 'site_2': (41.2, -112.0)}
        sites_kwargs : dict | None, optional
            Additional keyword arguments for site marker plotting (e.g., 'marker', 'color', 's').
            By default None (uses default marker style).
        tiler : cartopy.io.img_tiles.GoogleTiles | None, optional
            Tiler to use for background map, by default None.
            If provided, the tiler will be used to add a background map to the plots.
        **kwargs : dict
            Additional keyword arguments to pass to xarray plotting functions.

        Returns
        -------
        fig, axes : matplotlib.figure.Figure, np.ndarray
            Figure and axes objects.
        """
        # Get xarray representations of fluxes
        prior = self.inversion.xr.prior["flux"]
        posterior = self.inversion.xr.posterior["flux"]

        # Filter/aggregate by time
        if time == "mean":
            prior = prior.mean(dim=time_dim)
            posterior = posterior.mean(dim=time_dim)
        elif time == "std":
            prior = prior.std(dim=time_dim)
            posterior = posterior.std(dim=time_dim)
        elif isinstance(time, int):
            prior = prior.isel({time_dim: time})
            posterior = posterior.isel({time_dim: time})
        else:
            prior = prior.sel({time_dim: time})
            posterior = posterior.sel({time_dim: time})

        # Get tiler and projection from kwargs
        tiler = kwargs.pop("tiler", None)
        subplot_kw = kwargs.pop("subplot_kw", {})
        if tiler is not None:
            subplot_kw["projection"] = tiler.crs

        ncols = 3
        if time == "std":
            ncols -= 1
        if truth is not None:
            ncols += 1
            if isinstance(truth, pd.Series):
                truth = truth.to_xarray()
            if time == "mean":
                truth = truth.mean(dim=time_dim)
            elif time == "std":
                truth = truth.std(dim=time_dim)
            elif isinstance(time, int):
                truth = truth.isel({time_dim: time})
            else:
                truth = truth.sel({time_dim: time})

        # Create figure and axes
        fig, axes = plt.subplots(ncols=ncols, sharey=True, subplot_kw=subplot_kw)

        if truth is None:
            ax_prior = axes[0]
            ax_post = axes[1]
        else:
            ax_truth = axes[0]
            ax_prior = axes[1]
            ax_post = axes[2]
        if time != "std":
            ax_res = axes[-1]

        # Add background tiles
        if tiler is not None:
            tiler_zoom = kwargs.pop("tiler_zoom", 10)
            extent = [
                posterior.lon.min(),
                posterior.lon.max(),
                posterior.lat.min(),
                posterior.lat.max(),
            ]
            for ax in axes:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_image(tiler, tiler_zoom)
            if "lat" in posterior.dims:
                kwargs["transform"] = ccrs.PlateCarree()
            else:
                # TODO handle projected data (i could use my crs class)
                raise ValueError(
                    "Cannot determine coordinate reference system for plotting."
                )

        # Colorbar and plot options
        vmin = min(prior.min(), posterior.min())
        vmax = max(prior.max(), posterior.max())
        if truth is not None:
            vmin = min(vmin, truth.min())
            vmax = max(vmax, truth.max())
        alpha = kwargs.pop("alpha", 0.55)
        cmap = kwargs.pop("cmap", "RdBu_r" if vmin < 0 else "Reds")
        if vmin < 0:
            center = 0
            vmin = None  # cant set both vmin/vmax and center
        else:
            center = None

        # Set colorbar axis below both plots
        fig.subplots_adjust(bottom=0.15)
        ax1 = axes[0]
        cbar_ax1_width = ax_post.get_position().x1 - ax1.get_position().x0
        cbar_ax1 = fig.add_axes(
            [ax1.get_position().x0, ax1.get_position().y0 - 0.1, cbar_ax1_width, 0.05]
        )

        if truth is not None:
            truth.plot(
                ax=ax_truth,
                x=x_dim,
                y=y_dim,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                alpha=alpha,
                add_colorbar=False,
                center=center,
                **kwargs,
            )
            ax_truth.set(title="Truth", xlabel=None, ylabel=None)

        prior.plot(
            ax=ax_prior,
            x=x_dim,
            y=y_dim,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            alpha=alpha,
            cbar_ax=cbar_ax1,
            cbar_kwargs={"orientation": "horizontal", "label": "Flux"},
            center=center,
            **kwargs,
        )
        posterior.plot(
            ax=ax_post,
            x=x_dim,
            y=y_dim,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            alpha=alpha,
            add_colorbar=False,
            center=center,
            **kwargs,
        )

        # Add title and time text
        fig.suptitle("Flux Maps", fontsize=16, y=ax_prior.get_position().y1 + 0.13)
        fig.text(
            0.5,
            ax_prior.get_position().y1 + 0.05,
            f"time = {time}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        # Set labels for each subplot
        ax_prior.set(title="Prior", xlabel=None, ylabel=None)
        ax_post.set(title="Posterior", xlabel=None, ylabel=None)

        # Plot residual
        if time != "std":
            if truth is not None:
                base = truth
                label = "Posterior - Truth"
            else:
                base = prior
                label = "Posterior - Prior"
            residual_cmap = kwargs.pop("residual_cmap", "PiYG")
            cbar_ax2 = fig.add_axes(
                [
                    ax_res.get_position().x0,
                    ax_res.get_position().y0 - 0.1,
                    ax_res.get_position().width,
                    0.05,
                ]
            )
            (posterior - base).plot(
                ax=ax_res,
                x=x_dim,
                y=y_dim,
                cmap=residual_cmap,
                alpha=alpha,
                center=0,
                cbar_ax=cbar_ax2,
                cbar_kwargs={"orientation": "horizontal", "label": label},
                **kwargs,
            )

            ax_res.set(title="Residual", xlabel=None, ylabel=None)

        # Plot observation site markers
        if sites:
            if isinstance(sites, dict):
                # Extract coordinates from dict map
                lats = []
                lons = []
                for site_id, (lat, lon) in sites.items():
                    lats.append(lat)
                    lons.append(lon)

                if lons and lats:
                    # Set default site marker kwargs
                    site_plot_kwargs = {
                        "marker": "o",
                        "color": "blue",
                        "s": 80,
                        "alpha": 0.7,
                    }
                    if sites_kwargs:
                        site_plot_kwargs.update(sites_kwargs)

                    # Plot sites on all axes
                    for ax in axes:
                        ax.scatter(
                            lons,
                            lats,
                            **site_plot_kwargs,
                            zorder=5,
                            edgecolors="black",
                            linewidth=1,
                        )
            else:
                raise NotImplementedError(
                    "Sites must be passed as a dict[site_id, (lat, lon)] at present"
                )

        return fig, axes

    def concentrations(self, location=None, location_dim="obs_location", **kwargs):
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
        obs = self.inversion.obs["concentration"].rename("observed")
        posterior = self.inversion.posterior_obs["concentration"].rename("posterior")
        prior = self.inversion.prior_obs["concentration"].rename("prior")

        data = pd.concat([obs, posterior, prior], axis=1)

        if location is None:
            locations = data.index.get_level_values(location_dim).unique()
        elif isinstance(location, str):
            locations = [location]
        elif isinstance(location, list):
            locations = location
        else:
            raise ValueError("location must be None, a string, or a list of strings")

        axes = []
        for location in locations:
            df = data.loc[location]
            df.columns.name = None

            fig, ax = plt.subplots()

            df.plot(
                ax=ax,
                style=".",
                alpha=0.6,
                color=["black", "red", "blue"],
                markeredgecolor="None",
                legend=False,
            )
            df.rolling(window=max(1, int(len(df) / 10)), center=True).mean().plot(
                ax=ax,
                linewidth=2,
                color=["black", "red", "blue"],
                label=["Observed", "Posterior", "Prior"],
            )
            ax.set(
                title=f"Concentrations at site: {location}",
                ylabel="Concentration",
                xlabel="Time",
            )
            fig.autofmt_xdate()
            axes.append(ax)
            plt.show()
        return np.array(axes)
