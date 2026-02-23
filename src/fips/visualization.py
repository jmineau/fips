"""Visualization and plotting utilities for inverse problem results.

This module provides functions for plotting error norms, comparing multiple series,
and computing credible intervals for uncertainty visualization.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from fips.covariance import CovarianceMatrix
from fips.matrix import Matrix
from fips.vector import Block, Vector

ArrayLike = Sequence | np.ndarray


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting; install with `pip install matplotlib`."
        ) from exc


def plot_error_norm(
    prior: ArrayLike | Vector | Block,
    posterior: ArrayLike | Vector | Block,
    truth: ArrayLike | Vector | Block,
    t: Iterable[float] | None = None,
    norm: str = "l2",
    figsize: tuple[float, float] | None = None,
):
    """Plot normed errors of prior and posterior against truth."""
    plt = _require_matplotlib()

    # Safely extract raw values and force to 2D for consistent norm calculation
    def _format(x):
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
    *series: pd.Series | Vector | Block,
    x: str | int | None = None,
    truth: pd.Series | Vector | Block | None = None,
    errors: Sequence[pd.DataFrame | CovarianceMatrix | Matrix | pd.Series | None]
    | None = None,
    kind: str | None = None,
):
    """Compare multiple aligned series (e.g., prior, posterior, obs) with optional errors."""
    plt = _require_matplotlib()

    # Unpack data
    df = pd.concat([getattr(s, "data", s) for s in series], axis=1)

    # Extract standard deviations for error bounds
    err_list = []
    if errors:
        for i, (s, err) in enumerate(zip(series, errors, strict=False)):
            if err is None:
                err_list.append(None)
                continue

            name = getattr(s, "name", df.columns[i])

            if isinstance(err, CovarianceMatrix):
                e = np.sqrt(err.get_variances())
            elif isinstance(err, Matrix):
                e = pd.Series(np.sqrt(np.diag(err.values)), index=err.index, name=name)
            else:
                e = getattr(err, "data", err)

            e.name = name
            err_list.append(e)

    err_df = pd.concat(err_list, axis=1) if err_list else None

    # Determine X-axis values
    if isinstance(df.index, pd.MultiIndex):
        if x is None:
            raise ValueError("x must be specified when series have a MultiIndex")
        x_vals = df.index.get_level_values(x)
    else:
        x_vals = df.index
        x = x or df.index.name or "index"

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
        e = err_df[col] if err_df is not None else None

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
        t_data = getattr(truth, "data", truth)
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


def compute_credible_interval(
    samples: ArrayLike, q: tuple[float, float] = (0.05, 0.95)
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower/upper quantiles along the first axis of samples."""
    samples = np.asarray(samples)
    return np.quantile(samples, q[0], axis=0), np.quantile(samples, q[1], axis=0)
