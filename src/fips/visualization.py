"""
Plotting of inverse problem results.

:func:`plot_comparison` draws several series (a prior, a posterior, the
observations) on one axes, with error bands or bars. It needs matplotlib,
which the ``viz`` extra installs (``pip install "fips[viz]"``). For anything
else, a :class:`~fips.Vector` gives pandas and xarray objects to plot
(``post.estimate["flux"].to_series()``, ``.to_xarray()``).
"""

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from fips.covariance import Covariance
from fips.operator import Operator
from fips.space import _listed
from fips.vector import Vector


def _require_matplotlib():
    """Import and return ``matplotlib.pyplot``, with an install hint when missing."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            'matplotlib is required for plotting; install with `pip install "fips[viz]"`.'
        ) from exc


def _as_series(v: Vector) -> pd.Series:
    """
    Return a ``Vector`` as a Series.

    A vector of one block is the Series on the block's own labels, without
    the ``block`` level; one of several blocks keeps it.
    """
    series = v.to_series()
    return series.droplevel("block") if len(v.space.names) == 1 else series


def _unwrap(x: pd.Series | pd.DataFrame | Vector) -> pd.Series | pd.DataFrame:
    """Return a ``Vector`` as a Series; anything else as is."""
    return _as_series(x) if isinstance(x, Vector) else x


def _errors_on(
    err: pd.DataFrame | Operator | pd.Series | Vector | np.ndarray | list,
    name: Hashable,
    index: pd.Index,
    position: int,
) -> pd.Series | None:
    """
    Return the standard deviations ``err`` gives for the series ``name``, on ``index``.

    A Covariance or an Operator gives the square roots of its diagonal, on the
    labels of its rows. Every error with labels is matched to ``index`` by
    label, in any order, as the truth is; a plain array or list has none, so it
    is read by position. A DataFrame gives its column called ``name``, or None
    if it has none.

    Raises
    ------
    ValueError
        If the error has no value for some label of ``index``, or a plain
        array or list is not as long as ``index``.
    """
    if isinstance(err, (np.ndarray, list, tuple)):
        values = np.asarray(err, dtype=float)
        if values.shape != (len(index),):
            raise ValueError(
                f"errors[{position}] has {values.size} values for {len(index)} "
                "points; a plain array or list is read by position."
            )
        return pd.Series(values, index=index)
    if isinstance(err, Covariance):
        variances = err.variances
        e = _as_series(Vector(np.sqrt(variances.values), variances.space))
    elif isinstance(err, Operator):
        e = _as_series(Vector(np.sqrt(err.data.diagonal()), err.rows))
    else:
        given = _unwrap(err)
        if isinstance(given, pd.DataFrame):
            if name not in given:
                return None
            e = given[name]
        else:
            e = given

    e = e.reindex(index)
    if e.isna().any():
        raise ValueError(
            f"errors[{position}] has no value for some labels of the series: "
            f"{_listed(index[e.isna().to_numpy()])}."
        )
    return e


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
        True values to plot as reference, matched to the series by label (in
        any order), as the errors are.
    errors : Sequence of pd.DataFrame or Covariance or Operator or pd.Series or Vector or np.ndarray or list or None, optional
        Error estimates for each series, in the series' order. A Series, a
        Vector, a DataFrame column or an array is the standard deviations
        themselves, not the variances: the band is ``y - e`` to ``y + e``.
        If a Covariance or an Operator, its diagonal's square roots (the
        standard deviations) are used. An error with labels is matched to the
        series by label (in any order), as `truth` is, so a Series sorted
        differently from the series, or a Covariance on a reordered space,
        still gives each point its own error. A plain array or list has no
        labels: it is read by position, in the series' order. A DataFrame
        gives its column named like the series.
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

    Raises
    ------
    ValueError
        If `x` is not given for series with a MultiIndex, `truth` or an error
        has no value for a label of the series, or a plain array or list of
        errors is not as long as the series.
    """
    plt = _require_matplotlib()

    # Unpack data
    df = pd.concat([_unwrap(s) for s in series], axis=1)

    # Standard deviations for the error bounds, each on the series' labels (as the
    # truth is): ``errors_on[i]`` is None for a series without an error
    errors_on: list[pd.Series | None] = [None] * len(series)
    for i, err in enumerate(errors[: len(series)] if errors else []):
        if err is not None:
            errors_on[i] = _errors_on(err, df.columns[i], df.index, i)

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
        e = errors_on[i]

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
        t_data = _unwrap(truth).reindex(df.index)
        if t_data.isna().any():
            raise ValueError(
                "truth has no value for some labels of the series: "
                f"{_listed(df.index[t_data.isna().to_numpy()])}."
            )
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
