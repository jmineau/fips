"""Visualization and plotting utilities for inverse problem results.

This module provides functions for plotting error norms, comparing multiple series,
and computing credible intervals for uncertainty visualization.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from fips.covariance import CovarianceMatrix
from fips.structures import Vector

ArrayLike = Sequence[Sequence[float]] | np.ndarray
CI = tuple[ArrayLike, ArrayLike]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "matplotlib is required for plotting; install with `pip install matplotlib` or `pip install fips[flux]`."
        ) from exc
    return plt


def _to_2d(data: ArrayLike | None, name: str) -> np.ndarray | None:
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D (time, state_dim)")
    return arr


def _check_shapes(*arrays: np.ndarray | None) -> tuple[int, int]:
    ref_shape = None
    for arr in arrays:
        if arr is None:
            continue
        if ref_shape is None:
            ref_shape = arr.shape
        elif ref_shape != arr.shape:
            raise ValueError(
                f"All arrays must share shape (n_points, state_dim); got {ref_shape} and {arr.shape}"
            )
    if ref_shape is None:
        raise ValueError("At least one series is required")
    n_points, state_dim = ref_shape
    return n_points, state_dim


def plot_error_norm(
    prior: ArrayLike,
    posterior: ArrayLike,
    truth: ArrayLike,
    t: Iterable[float] | None = None,
    norm: str = "l2",
    figsize: tuple[float, float] | None = None,
):
    """Plot normed errors of prior and posterior against truth.

    Parameters
    ----------
    prior, posterior, truth : array-like
        Arrays shaped (n_points, state_dim) or (n_points,).
    t : iterable, optional
        Index values for x-axis; defaults to ``range(n_points)``.
    norm : str, default 'l2'
        Norm to use: 'l2', 'l1', or 'linf'.
    figsize : tuple, optional
        Figure size passed to matplotlib.
    """

    plt = _require_matplotlib()

    prior_arr = _to_2d(prior, "prior")
    posterior_arr = _to_2d(posterior, "posterior")
    truth_arr = _to_2d(truth, "truth")
    n_points, _ = _check_shapes(prior_arr, posterior_arr, truth_arr)

    x = np.arange(n_points) if t is None else np.asarray(list(t))
    if x.shape[0] != n_points:
        raise ValueError(f"t must have length {n_points}; got {x.shape[0]}")

    def _norm(arr: np.ndarray) -> np.ndarray:
        if norm == "l2":
            return np.linalg.norm(arr, axis=1)
        if norm == "l1":
            return np.sum(np.abs(arr), axis=1)
        if norm == "linf":
            return np.max(np.abs(arr), axis=1)
        raise ValueError("norm must be one of {'l2', 'l1', 'linf'}")

    prior_err = _norm(prior_arr - truth_arr)
    posterior_err = _norm(posterior_arr - truth_arr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, prior_err, label=f"prior | |.|_{norm}", linestyle="--", color="tab:blue")
    ax.plot(
        x,
        posterior_err,
        label=f"posterior | |.|_{norm}",
        linestyle="-",
        color="tab:orange",
    )
    ax.set_xlabel("step")
    ax.set_ylabel(f"{norm} error")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_comparison(
    *series: pd.Series | Vector,
    x: str | int | None = None,
    truth: pd.Series | Vector | None = None,
    errors: Sequence[pd.DataFrame | CovarianceMatrix | None] | None = None,
    kind: str | None = None,
):
    """Compare multiple aligned series (e.g., prior, posterior, obs) with optional errors.

    Works with numpy arrays, pandas Series/DataFrames, or fips Vectors.

    Parameters
    ----------
    *series : array-like, pd.Series, Vector
        Series to compare. If pd.Series, uses index as x-axis labels.
        If Vector, uses assembled data.
    truth : array-like, optional
        Reference series with same length.
    errors : sequence of array-like, optional
        Error bounds for each series. 1D arrays (std) or 2D covariance matrices.
        If 2D, assumes covariance for a single dimension (extracts diagonal).
    kind : str, optional
        Plot style: 'line' or 'bar'. If None, auto-detects based on x_labels dtype
        (numeric/date → line, string/other → bar).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    plt = _require_matplotlib()

    series = tuple(s.data if isinstance(s, Vector) else s for s in series)
    df = pd.concat(series, axis=1)

    # Handle truth
    truth = truth.data if isinstance(truth, Vector) else truth

    # Handle errors
    error_vals = []
    if errors is not None:
        for i, err in enumerate(errors):
            if err is None:
                error_vals.append(None)
            else:
                name = series[i].name

                if isinstance(err, CovarianceMatrix):
                    err = err.data
                if isinstance(err, pd.DataFrame):
                    index = err.index
                    err = err.loc[index, index].values
                    err = pd.Series(err, index=index, name=name)
                error_vals.append(err)
    err_df = pd.concat(error_vals, axis=1) if error_vals else None

    # Get x index
    if isinstance(df.index, pd.MultiIndex):
        if x is None:
            raise ValueError("x must be specified when series have MultiIndex")
        x_vals = df.index.get_level_values(x)
    else:
        x_vals = df.index
        if x is None:
            x = df.index.name if df.index.name is not None else "index"

    # Auto-detect plot kind
    if kind is None:
        try:
            if np.issubdtype(x_vals.dtype, np.number) or np.issubdtype(
                x_vals.dtype, np.datetime64
            ):
                kind = "line"
            else:
                kind = "bar"
        except (TypeError, np.core._exceptions._UFuncInputCastingError):
            kind = "bar"

    if kind not in ("line", "bar"):
        raise ValueError("kind must be 'line' or 'bar'")

    # Plot
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))

    for i, col in enumerate(df.columns):
        y = df[col]
        err = err_df[col] if err_df is not None else None

        if kind == "line":
            ax.plot(x_vals, y, label=str(col), color=colors[i])
            if err is not None:
                ax.fill_between(
                    x_vals,
                    y - err,
                    y + err,
                    color=colors[i],
                    alpha=0.3,
                )

        else:  # bar
            ax.bar(
                x_vals + i * 0.2,
                y,
                width=0.2,
                label=str(col),
                color=colors[i],
                yerr=err.values if err is not None else None,
                capsize=5,
            )

    if truth is not None:
        if kind == "line":
            ax.plot(x_vals, truth, label="truth", color="black", linestyle="--")
        else:
            ax.scatter(
                x_vals + (len(df.columns) / 2 - 0.5) * 0.2,
                truth,
                label="truth",
                color="black",
                marker="x",
                s=100,
                zorder=5,
            )

    ax.legend()
    ax.set(xlabel=str(x), ylabel="value")

    return fig, ax


def compute_credible_interval(
    samples: ArrayLike, q: tuple[float, float] = (0.05, 0.95)
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower/upper quantiles along the first axis of samples.

    Parameters
    ----------
    samples : array-like
        Sample stack shaped (n_samples, time, state_dim) or (n_samples, state_dim) or (n_samples,).
    q : tuple, default (0.05, 0.95)
        Quantiles to compute.
    """

    arr = np.asarray(samples)
    if arr.ndim not in (2, 3):
        raise ValueError(
            "samples must be 2D or 3D: (n_samples, time, state_dim) or (n_samples, state_dim)"
        )
    lower = np.quantile(arr, q[0], axis=0)
    upper = np.quantile(arr, q[1], axis=0)
    if lower.shape != upper.shape:
        raise ValueError("Quantile outputs should match shape")
    return np.asarray(lower), np.asarray(upper)
