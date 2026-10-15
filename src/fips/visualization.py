"""
Plotting of inverse problem results.

:func:`plot_comparison` draws several series (a prior, a posterior, the
observations) on one axes, with error bands or bars. It needs matplotlib,
which the ``viz`` extra installs (``pip install "fips[viz]"``). For anything
else, a :class:`~fips.Vector` gives pandas and xarray objects to plot
(``post.estimate["flux"].to_series()``, ``.to_xarray()``).
"""

from collections.abc import Sequence

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

    Raises
    ------
    ValueError
        If `x` is not given for series with a MultiIndex, or `truth` has no
        value for a label of the series.
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
