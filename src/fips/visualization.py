"""
Visualization and plotting utilities for inverse problem results.

This module provides functions for plotting error norms, comparing multiple series,
mapping a block of a posterior's state (:func:`plot_map`), plotting the
observations against what the prior and the posterior model (:func:`plot_obs`),
and computing credible intervals for uncertainty visualization. They need
matplotlib, and maps on background tiles need cartopy: both come with the
``viz`` extra (``pip install "fips[viz]"``).
"""

from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fips.covariance import Covariance
from fips.operator import Operator
from fips.posterior import Posterior
from fips.vector import Vector

if TYPE_CHECKING:
    from matplotlib.artist import Artist

ArrayLike = Sequence | np.ndarray


def _require_matplotlib():
    """Import and return ``matplotlib.pyplot``, with an install hint when missing."""
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ImportError as exc:
        raise ImportError(
            'matplotlib is required for plotting; install with `pip install "fips[viz]"`.'
        ) from exc


def _require_cartopy():
    """Import and return ``cartopy.crs``, with an install hint when missing."""
    try:
        import cartopy.crs as ccrs  # type: ignore

        return ccrs
    except ImportError as exc:
        raise ImportError(
            'cartopy is required for map tiles; install with `pip install "fips[viz]"`.'
        ) from exc


def _block_name(vector: Vector, block: str | None, which: str) -> str:
    """
    Return `block`, or the one block of `vector` when `block` is None.

    Raises
    ------
    ValueError
        If `block` is None and `vector` has several blocks.
    KeyError
        If `vector` has no block named `block`.
    """
    names = vector.space.names
    if block is None:
        if len(names) != 1:
            raise ValueError(
                f"The {which} has the blocks {list(names)}: name one with block=."
            )
        return names[0]
    if block not in vector.space:
        raise KeyError(f"The {which} has no block {block!r}; it has {list(names)}.")
    return block


def _unwrap(x: pd.Series | pd.DataFrame | Vector) -> pd.Series | pd.DataFrame:
    """
    Return a ``Vector`` as a Series; anything else as is.

    A vector of one block is the Series on the block's own labels, without
    the ``block`` level; one of several blocks keeps it.
    """
    if not isinstance(x, Vector):
        return x
    series = x.to_series()
    return series.droplevel("block") if len(x.space.names) == 1 else series


def plot_error_norm(
    prior: ArrayLike | Vector,
    posterior: ArrayLike | Vector,
    truth: ArrayLike | Vector,
    t: Iterable[float] | None = None,
    norm: str = "l2",
    figsize: tuple[float, float] | None = None,
):
    """
    Plot normed errors of prior and posterior against truth.

    Parameters
    ----------
    prior : ArrayLike or Vector
        Prior estimates.
    posterior : ArrayLike or Vector
        Posterior estimates.
    truth : ArrayLike or Vector
        True values for comparison.
    t : Iterable[float], optional
        Time or x-axis values. If None, uses integer indices.
    norm : {'l2', 'l1', 'linf'}, default 'l2'
        Norm type to use for error calculation.
    figsize : tuple[float, float], optional
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    plt = _require_matplotlib()

    # Safely extract raw values and force to 2D for consistent norm calculation
    def _format(x):
        """Return ``x`` as a 2-D numpy array."""
        arr = np.asarray(getattr(x, "values", x))
        return arr[:, None] if arr.ndim == 1 else arr

    p_arr, post_arr, t_arr = _format(prior), _format(posterior), _format(truth)

    if norm == "l2":
        prior_err = np.linalg.norm(p_arr - t_arr, axis=1)
        post_err = np.linalg.norm(post_arr - t_arr, axis=1)
    elif norm == "l1":
        prior_err = np.sum(np.abs(p_arr - t_arr), axis=1)
        post_err = np.sum(np.abs(post_arr - t_arr), axis=1)
    elif norm == "linf":
        prior_err = np.max(np.abs(p_arr - t_arr), axis=1)
        post_err = np.max(np.abs(post_arr - t_arr), axis=1)
    else:
        raise ValueError("norm must be one of {'l2', 'l1', 'linf'}")

    x = np.arange(len(prior_err)) if t is None else np.asarray(list(t))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, prior_err, label=f"prior |.|{norm}", linestyle="--", color="tab:blue")
    ax.plot(
        x, post_err, label=f"posterior |.|{norm}", linestyle="-", color="tab:orange"
    )
    ax.set(xlabel="step", ylabel=f"{norm} error")
    ax.legend(loc="best")
    fig.tight_layout()

    return fig, ax


def plot_comparison(
    *series: pd.Series | Vector,
    x: str | int | None = None,
    truth: pd.Series | Vector | None = None,
    errors: Sequence[pd.DataFrame | Operator | pd.Series | None] | None = None,
    kind: str | None = None,
):
    """
    Compare multiple aligned series with optional errors and truth values.

    Plots multiple series (e.g., prior, posterior, observations) on the same axes
    with optional error bars/bands and truth values for comparison.

    Parameters
    ----------
    *series : pd.Series or Vector
        Variable number of series to compare.
    x : str or int, optional
        Name or level of the index to use for x-axis. Required if series have MultiIndex.
    truth : pd.Series or Vector, optional
        True values to plot as reference.
    errors : Sequence of pd.DataFrame or Covariance or Operator or pd.Series or None, optional
        Error estimates for each series. Must match length of series.
        If a Covariance or an Operator, its diagonal's square roots (the
        standard deviations) are used.
        Use None for a series with no error: it is plotted without a band or
        error bars.
    kind : {'line', 'bar'}, optional
        Plot type. If None, auto-detects based on x-axis data type.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    plt = _require_matplotlib()

    # Unpack data
    df = pd.concat([_unwrap(s) for s in series], axis=1)

    # Extract standard deviations for error bounds: one column per series that has
    # one, labeled like the series' column in ``df`` (a series without an error is
    # left out, and so has no column)
    err_list = []
    if errors:
        for i, err in enumerate(errors[: len(series)]):
            if err is None:
                continue

            name = df.columns[i]

            if isinstance(err, Covariance):
                variances = err.variances
                e = pd.Series(
                    np.sqrt(variances.values), index=variances.index, name=name
                )
            elif isinstance(err, Operator):
                diagonal = np.sqrt(err.data.diagonal())
                e = pd.Series(diagonal, index=err.rows.index, name=name)
            else:
                e = _unwrap(err)
                if isinstance(e, pd.Series):
                    e = e.rename(name)  # a copy: the caller's Series keeps its name

            err_list.append(e)

    err_df = pd.concat(err_list, axis=1) if err_list else None

    # Determine X-axis values
    if isinstance(df.index, pd.MultiIndex):
        if x is None:
            raise ValueError("x must be specified when series have a MultiIndex")
        x_vals = df.index.get_level_values(x)
    else:
        x_vals = df.index
        x = x or str(df.index.name or "index")

    # Auto-detect plot kind
    if kind is None:
        if pd.api.types.is_numeric_dtype(
            x_vals
        ) or pd.api.types.is_datetime64_any_dtype(x_vals):
            kind = "line"
        else:
            kind = "bar"

    # Plot
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))
    x_pos = np.arange(len(x_vals))  # Used for bar chart offsets

    for i, col in enumerate(df.columns):
        y = df[col]
        e = err_df[col] if err_df is not None and col in err_df else None

        if kind == "line":
            ax.plot(x_vals, y, label=str(col), color=colors[i])
            if e is not None:
                ax.fill_between(x_vals, y - e, y + e, color=colors[i], alpha=0.3)
        else:
            ax.bar(
                x_pos + i * 0.2,
                y,
                width=0.2,
                label=str(col),
                color=colors[i],
                yerr=e,
                capsize=5,
            )

    if truth is not None:
        t_data = _unwrap(truth)
        if kind == "line":
            ax.plot(x_vals, t_data, label="truth", color="black", linestyle="--")
        else:
            ax.scatter(
                x_pos + (len(series) / 2 - 0.5) * 0.2,
                t_data,
                label="truth",
                color="black",
                marker="x",
                s=100,
                zorder=5,
            )

    if kind == "bar":
        ax.set_xticks(x_pos + (len(series) / 2 - 0.5) * 0.2)
        ax.set_xticklabels(x_vals)

    ax.legend()
    ax.set(xlabel=str(x), ylabel="value")

    return fig, ax


def _at_time(data, time: Any, time_level: str):
    """Reduce or select `data` (xarray) along `time_level`, as :func:`plot_map` takes `time`."""
    if time == "mean":
        return data.mean(dim=time_level)
    if time == "std":
        return data.std(dim=time_level)
    if isinstance(time, int):
        return data.isel({time_level: time})
    return data.sel({time_level: time})


def plot_map(
    post: Posterior,
    block: str | None = None,
    time: Any = "mean",
    truth: pd.Series | Vector | None = None,
    x: str = "lon",
    y: str = "lat",
    time_level: str = "time",
    sites: dict[Hashable, tuple[float, float]] | None = None,
    sites_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    """
    Map a block of the state: the prior, the posterior estimate and their difference.

    The block's labels must have a level for each axis of the map (`x` and
    `y`, longitude and latitude by default) and a `time_level`, which `time`
    reduces or selects. One panel each for the prior and the estimate, on one
    color scale, then the estimate minus the prior; with `truth`, a truth
    panel comes first and the last panel is the estimate minus the truth.

    Parameters
    ----------
    post : Posterior
        The posterior :func:`fips.solve` returned.
    block : str, optional
        The block of the state to map. By default, the state's only block;
        a state of several blocks needs it.
    time : {'mean', 'std'}, int or label, default 'mean'
        ``'mean'`` or ``'std'`` over `time_level`, the position of one time
        (an int), or one time's label (a timestamp). With ``'std'`` there is
        no difference panel.
    truth : pandas.Series or Vector, optional
        The true values, for a synthetic problem: a Series on the block's
        labels, or a Vector with the block.
    x, y : str, default 'lon', 'lat'
        The levels of the block's labels on the horizontal and vertical axes.
    time_level : str, default 'time'
        The level of the block's labels that `time` reduces or selects.
    sites : dict, optional
        Points to mark on every panel, as ``{name: (lat, lon)}``: the
        observation sites, for instance.
    sites_kwargs : dict, optional
        Keyword arguments for the markers, passed to ``Axes.scatter``.
    **kwargs
        Passed to xarray's plotting (``cmap``, ``alpha``, ...), except
        ``tiler`` (a cartopy tile source drawn under each panel; `x` and `y`
        are then longitudes and latitudes in degrees, and cartopy is needed),
        ``tiler_zoom`` (its zoom level, 10 by default), ``subplot_kw`` (passed
        to ``plt.subplots``) and ``residual_cmap`` (the difference panel's
        colormap, ``'PiYG'`` by default).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : numpy.ndarray
        Its panels, in order.

    Raises
    ------
    TypeError
        If `sites` is not a dict.
    ImportError
        If `tiler` is given and cartopy is not installed.

    See Also
    --------
    plot_obs : The observations against what the prior and the posterior model.
    """
    plt = _require_matplotlib()
    if sites is not None and not isinstance(sites, dict):
        raise TypeError(
            f"sites is a dict of {{name: (lat, lon)}}, not {type(sites).__name__}."
        )

    block = _block_name(post.estimate, block, "state")
    prior = _at_time(post.problem.prior[block].to_xarray(), time, time_level)
    posterior = _at_time(post.estimate[block].to_xarray(), time, time_level)
    if truth is not None:
        if isinstance(truth, Vector):
            truth = truth[block].to_series().droplevel("block")
        truth = _at_time(truth.to_xarray(), time, time_level)

    tiler = kwargs.pop("tiler", None)
    tiler_zoom = kwargs.pop("tiler_zoom", 10)
    residual_cmap = kwargs.pop("residual_cmap", "PiYG")
    subplot_kw = kwargs.pop("subplot_kw", {})
    if tiler is not None:
        _require_cartopy()  # fail with the install hint before the figure exists
        subplot_kw["projection"] = tiler.crs

    ncols = 3 if time != "std" else 2
    if truth is not None:
        ncols += 1

    fig, axes = plt.subplots(ncols=ncols, sharey=True, subplot_kw=subplot_kw)

    # The truth panel, when there is one, comes first; the difference panel,
    # when there is one, comes last.
    ax_prior, ax_post = (axes[0], axes[1]) if truth is None else (axes[1], axes[2])

    if tiler is not None:
        ccrs = _require_cartopy()
        # Expand the extent by half a cell so the edge cells are not clipped
        xs, ys = posterior[x], posterior[y]
        dx = float(xs.diff(x).median()) if len(xs) > 1 else 0
        dy = float(ys.diff(y).median()) if len(ys) > 1 else 0
        extent = [
            float(xs.min()) - dx / 2,
            float(xs.max()) + dx / 2,
            float(ys.min()) - dy / 2,
            float(ys.max()) + dy / 2,
        ]
        for ax in axes:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_image(tiler, tiler_zoom)
        kwargs["transform"] = ccrs.PlateCarree()

    # One color scale for the truth, the prior and the estimate
    vmin = min(prior.min(), posterior.min())
    vmax = max(prior.max(), posterior.max())
    if truth is not None:
        vmin = min(vmin, truth.min())
        vmax = max(vmax, truth.max())
    alpha = kwargs.pop("alpha", 0.55)
    cmap = kwargs.pop("cmap", "RdBu_r" if vmin < 0 else "Reds")
    if vmin < 0:
        center = 0
        vmin = None  # xarray cannot take both vmin/vmax and center
    else:
        center = None
    shared = dict(x=x, y=y, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, center=center)

    # The colorbar goes under the panels it covers
    fig.subplots_adjust(bottom=0.15)
    ax1 = axes[0]
    cbar_ax1_width = ax_post.get_position().x1 - ax1.get_position().x0
    cbar_ax1 = fig.add_axes(
        (ax1.get_position().x0, ax1.get_position().y0 - 0.1, cbar_ax1_width, 0.05)
    )

    if truth is not None:
        ax_truth = axes[0]
        truth.plot(ax=ax_truth, add_colorbar=False, **shared, **kwargs)
        ax_truth.set(title="Truth", xlabel=None, ylabel=None)

    prior.plot(
        ax=ax_prior,
        cbar_ax=cbar_ax1,
        cbar_kwargs={"orientation": "horizontal", "label": block},
        **shared,
        **kwargs,
    )
    posterior.plot(ax=ax_post, add_colorbar=False, **shared, **kwargs)

    fig.suptitle(block, fontsize=16, y=ax_prior.get_position().y1 + 0.13)
    fig.text(
        0.5,
        ax_prior.get_position().y1 + 0.05,
        f"{time_level} = {time}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    ax_prior.set(title="Prior", xlabel=None, ylabel=None)
    ax_post.set(title="Posterior", xlabel=None, ylabel=None)

    if time != "std":
        ax_res = axes[-1]
        if truth is not None:
            base, label = truth, "Posterior - Truth"
        else:
            base, label = prior, "Posterior - Prior"
        # Put the base on the estimate's coordinates, which may differ in the
        # last digits
        base = base.reindex_like(posterior, method="nearest")
        cbar_ax2 = fig.add_axes(
            (
                ax_res.get_position().x0,
                ax_res.get_position().y0 - 0.1,
                ax_res.get_position().width,
                0.05,
            )
        )
        (posterior - base).plot(
            ax=ax_res,
            x=x,
            y=y,
            cmap=residual_cmap,
            alpha=alpha,
            center=0,
            cbar_ax=cbar_ax2,
            cbar_kwargs={"orientation": "horizontal", "label": label},
            **kwargs,
        )
        ax_res.set(title="Residual", xlabel=None, ylabel=None)

    if sites:
        lats = [lat for lat, _ in sites.values()]
        lons = [lon for _, lon in sites.values()]
        marker = {"marker": "o", "color": "blue", "s": 80, "alpha": 0.7}
        marker.update(sites_kwargs or {})
        for ax in axes:
            ax.scatter(lons, lats, **marker, zorder=5, edgecolors="black", linewidth=1)

    return fig, axes


def plot_obs(
    post: Posterior,
    block: str | None = None,
    location: Hashable | list | None = None,
    location_level: str = "obs_location",
    time_level: str | None = None,
    rolling_window: str | int = "30D",
    gap_threshold: str = "7D",
    show_raw: bool = True,
) -> np.ndarray:
    """
    Plot the observations against what the prior and the posterior model, one figure per location.

    Three series over time: the observations, the posterior's
    :attr:`~fips.Posterior.modeled_obs` and the prior's
    (``problem.predict(problem.prior)``), each as points and as a rolling
    mean. A gap in a location's times longer than `gap_threshold` is shaded,
    and the rolling means break there.

    Parameters
    ----------
    post : Posterior
        The posterior :func:`fips.solve` returned.
    block : str, optional
        The block of the observations to plot. By default, the only block;
        observations of several blocks need it.
    location : label or list of labels, optional
        The location (or locations) to plot, values of `location_level`. By
        default, every location, in order of appearance.
    location_level : str, default 'obs_location'
        The level of the block's labels that holds the locations.
    time_level : str, optional
        The level that holds the times (datetimes). By default, the one level
        left besides `location_level`.
    rolling_window : str or int, default '30D'
        The window of the rolling means, as ``DataFrame.rolling`` takes it.
    gap_threshold : str, default '7D'
        The shortest gap between two times that is shaded as missing data.
    show_raw : bool, default True
        Whether to draw the points behind the rolling means.

    Returns
    -------
    numpy.ndarray
        The axes, one per location.

    See Also
    --------
    plot_map : Maps of a block of the state.
    plot_comparison : Several series on one axes, with error bands.
    """
    plt = _require_matplotlib()
    from matplotlib.patches import Patch

    problem = post.problem
    block = _block_name(problem.obs, block, "observations")

    def series(vector: Vector, name: str) -> pd.Series:
        """Return the block of `vector` on its own labels, named `name`."""
        return vector[block].to_series().droplevel("block").rename(name)

    data = pd.concat(
        [
            series(problem.obs, "observed"),
            series(post.modeled_obs, "posterior"),
            series(problem.predict(problem.prior), "prior"),
        ],
        axis=1,
    )

    if location is None:
        locations = list(data.index.get_level_values(location_level).unique())
    elif isinstance(location, list):
        locations = location
    else:
        locations = [location]

    colors = ["black", "red", "blue"]
    gap = pd.Timedelta(gap_threshold)

    axes = []
    for loc in locations:
        df = data.xs(loc, level=location_level)
        if time_level is not None:
            df.index = df.index.get_level_values(time_level)
        df = df.sort_index()
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

        # The gaps in the times
        times = df.index
        diffs = times.to_series().diff()
        gap_ends = times[diffs > gap]

        # The rolling means, with NaNs from the middle of each gap to its end
        # to break the lines
        smoothed = df.rolling(window=rolling_window, center=True, min_periods=1).mean()
        for gap_end in gap_ends:
            gap_start = times[times < gap_end][-1]
            smoothed.loc[gap_start + (gap_end - gap_start) / 2 : gap_end] = np.nan
        smoothed.plot(ax=ax, linewidth=1.5, color=colors, legend=False)

        for gap_end in gap_ends:
            gap_start = times[times < gap_end][-1]
            ax.axvspan(gap_start, gap_end, color="grey", alpha=0.15)

        # The legend shows the rolling means only
        handles: list[Artist] = [
            plt.Line2D([], [], color=c, linewidth=1.5, label=label)
            for c, label in zip(colors, df.columns, strict=False)
        ]
        if len(gap_ends) > 0:
            handles.append(Patch(facecolor="grey", alpha=0.15, label="no data"))
        ax.legend(handles=handles)

        # y-limits from the 1st and 99th percentiles, so outliers do not
        # flatten the rest
        values = df.values.ravel()
        values = values[~np.isnan(values)]
        if len(values) > 0:
            p01, p99 = np.percentile(values, [1, 99])
            margin = (p99 - p01) * 0.1
            ax.set_ylim(p01 - margin, p99 + margin)

        ax.set(title=f"{block} at {loc}", ylabel=block, xlabel="time")
        fig.autofmt_xdate()
        axes.append(ax)
    return np.array(axes)


def compute_credible_interval(
    samples: ArrayLike, q: tuple[float, float] = (0.05, 0.95)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute lower and upper quantiles along the first axis of samples.

    Parameters
    ----------
    samples : ArrayLike
        Array of samples with shape (n_samples, ...). Quantiles computed along axis 0.
    q : tuple[float, float], default (0.05, 0.95)
        Lower and upper quantile values in [0, 1].

    Returns
    -------
    lower : np.ndarray
        Lower quantile values.
    upper : np.ndarray
        Upper quantile values.
    """
    samples = np.asarray(samples)
    return np.quantile(samples, q[0], axis=0), np.quantile(samples, q[1], axis=0)
