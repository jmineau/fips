"""Plotting and visualization for flux inversion results."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fips.problems.flux.problem import FluxProblem


def _require_cartopy():
    try:
        import cartopy.crs as ccrs  # type: ignore

        return ccrs
    except ImportError as exc:
        raise ImportError(
            "cartopy is required for map plotting; install with `pip install fips[flux]`."
        ) from exc


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting; install with `pip install fips[flux]`."
        ) from exc


class FluxPlotter:
    """
    Plotting interface for FluxInversion results.

    Provides methods for visualizing prior/posterior fluxes and concentration timeseries.

    Parameters
    ----------
    inversion : FluxInversion
        The inverse problem to visualize.
    """

    inversion: "FluxProblem"
    """The inverse problem to visualize."""

    def __init__(self, inversion: "FluxProblem"):
        self.inversion = inversion

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FluxPlotter(inversion={repr(self.inversion)})"

    def fluxes(
        self,
        time="mean",
        truth=None,
        x_dim="lon",
        y_dim="lat",
        time_dim="time",
        sites: bool | dict = False,
        sites_kwargs=None,
        **kwargs,
    ):
        """
        Plot prior and posterior flux maps.

        Parameters
        ----------
        time : str, int, or pd.Timestamp, default 'mean'
            Time to plot: 'mean', 'std', time index, or timestamp.
        truth : pd.Series, optional
            Truth fluxes for comparison.
        x_dim : str, default 'lon'
            Name of the x-coordinate dimension.
        y_dim : str, default 'lat'
            Name of the y-coordinate dimension.
        time_dim : str, default 'time'
            Name of the time dimension.
        sites : bool or dict, optional
            Site locations to overlay: dict mapping site IDs to (lat, lon).
        sites_kwargs : dict, optional
            Additional plotting kwargs for site markers.
        **kwargs
            Additional arguments passed to xarray plotting.

        Returns
        -------
        fig, axes : matplotlib Figure and Axes
            The created figure and axes.
        """
        plt = _require_matplotlib()

        # Get xarray representations of fluxes
        prior = self.inversion.prior_fluxes.to_xarray()
        posterior = self.inversion.posterior_fluxes.to_xarray()

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
            ccrs = _require_cartopy()
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

        ax_truth = None
        ax_res = None
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
            # Expand extent by half a cell so edge cells aren't clipped
            dx = (
                float(posterior.lon.diff("lon").median())
                if len(posterior.lon) > 1
                else 0
            )
            dy = (
                float(posterior.lat.diff("lat").median())
                if len(posterior.lat) > 1
                else 0
            )
            extent = [
                float(posterior.lon.min()) - dx / 2,
                float(posterior.lon.max()) + dx / 2,
                float(posterior.lat.min()) - dy / 2,
                float(posterior.lat.max()) + dy / 2,
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

            # Align base to posterior to handle floating point coordinate mismatches
            base = base.reindex_like(posterior, method="nearest")

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
                for _site_id, (lat, lon) in sites.items():
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

    def concentrations(
        self,
        location=None,
        location_dim="obs_location",
        rolling_window="30d",
        gap_threshold="7d",
        show_raw=True,
        **kwargs,
    ):
        """
        Plot observed, prior, and posterior concentrations.

        Parameters
        ----------
        location : str, list of str, optional
            Observation location(s) to plot. If None, plots all locations.
        location_dim : str, default 'obs_location'
            Name of the location dimension in the data.
        rolling_window : str or int, default '30d'
            Rolling window for smoothed lines. Passed to ``DataFrame.rolling``.
        gap_threshold : str, default '7d'
            Minimum gap duration to highlight as missing data.
        show_raw : bool, default True
            Whether to show raw data points behind the smoothed lines.
        **kwargs
            Additional arguments passed to pandas plotting.

        Returns
        -------
        axes : np.ndarray
            Array of axes objects, one per location.
        """
        plt = _require_matplotlib()

        obs = self.inversion.concentrations.rename("observed")
        prior = self.inversion.prior_concentrations.rename("prior")
        posterior = self.inversion.posterior_concentrations.rename("posterior")

        data = pd.concat([obs, posterior, prior], axis=1)

        if location is None:
            locations = data.index.get_level_values(location_dim).unique()
        elif isinstance(location, str):
            locations = [location]
        elif isinstance(location, list):
            locations = location
        else:
            raise ValueError("location must be None, a string, or a list of strings")

        colors = ["black", "red", "blue"]
        gap_td = pd.Timedelta(gap_threshold)

        axes = []
        for location in locations:
            df = data.loc[location]
            df.columns.name = None

            fig, ax = plt.subplots()

            if show_raw:
                df.plot(
                    ax=ax,
                    style=".",
                    alpha=0.3,
                    markersize=4,
                    color=colors,
                    markeredgecolor="None",
                    legend=False,
                )

            # Detect gaps in observed data
            times = df.index.sort_values()
            diffs = times.to_series().diff()
            gap_starts = times[diffs > gap_td]

            # Compute smoothed lines, inserting NaNs at gaps to break lines
            smoothed = df.rolling(
                window=rolling_window, center=True, min_periods=1
            ).mean()
            for gap_start in gap_starts:
                gap_end = gap_start
                gap_begin = times[times < gap_start][-1]
                mid = gap_begin + (gap_end - gap_begin) / 2
                smoothed.loc[mid:gap_start] = np.nan

            smoothed.plot(
                ax=ax,
                linewidth=1.5,
                color=colors,
                legend=False,
            )

            # Shade data gaps
            for gap_start in gap_starts:
                gap_begin = times[times < gap_start][-1]
                ax.axvspan(gap_begin, gap_start, color="grey", alpha=0.15)

            # Build legend from smoothed lines only
            handles = [
                plt.Line2D([], [], color=c, linewidth=1.5, label=label)
                for c, label in zip(colors, df.columns, strict=False)
            ]
            if len(gap_starts) > 0:
                from matplotlib.patches import Patch

                handles.append(Patch(facecolor="grey", alpha=0.15, label="no data"))
            ax.legend(handles=handles)

            # Set y-limits based on percentiles to avoid outlier stretching
            all_vals = df.values.ravel()
            all_vals = all_vals[~np.isnan(all_vals)]
            if len(all_vals) > 0:
                p01, p99 = np.percentile(all_vals, [1, 99])
                margin = (p99 - p01) * 0.1
                ax.set_ylim(p01 - margin, p99 + margin)

            ax.set(
                title=f"Concentrations at site: {location}",
                ylabel="Concentration",
                xlabel="Time",
            )
            fig.autofmt_xdate()
            axes.append(ax)
            plt.show()
        return np.array(axes)
