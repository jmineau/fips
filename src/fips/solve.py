"""
Solving an inverse problem: :func:`solve` and the methods it can use.

Each method is registered under a name with the dataclass of its options,
declared once: :func:`solve` checks the options a caller passes against it
before anything is computed, and an unknown method or option raises with the
names it knows. The one method is ``"analytic"``, the closed-form Gaussian
posterior, computed from the whitened factorization of :mod:`fips._whitened`.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from fips._whitened import Whitened
from fips.posterior import Posterior
from fips.problem import Problem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticOptions:
    r"""
    The options of the ``"analytic"`` method.

    Parameters
    ----------
    gamma : float, default 1.0
        The regularization factor :math:`\gamma`, which multiplies the
        observation term of the cost:
        :math:`J(x) = (x - x_0)^T S_0^{-1} (x - x_0) + \gamma (z - H x - c)^T S_z^{-1} (z - H x - c)`.
        It is the same as dividing :math:`S_z` by :math:`\gamma`: above 1 the
        estimate fits the observations more closely, below 1 it stays closer
        to the prior. It must be positive and finite.

    Raises
    ------
    ValueError
        If `gamma` is not a positive, finite number.
    """

    gamma: float = 1.0
    """The regularization factor: the model-data mismatch is divided by it."""

    def __post_init__(self):
        try:
            gamma = float(self.gamma)
        except (TypeError, ValueError):
            raise ValueError(f"gamma must be a number, got {self.gamma!r}.") from None
        if not (math.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be positive and finite, got {gamma}.")
        object.__setattr__(self, "gamma", gamma)


@dataclass(frozen=True)
class _Method:
    """A registered method: the dataclass of its options and the function that solves."""

    options: type
    solve: Callable[[Problem, Any], Posterior]


_METHODS: dict[str, _Method] = {}


def _register(name: str, options: type) -> Callable:
    """Register the decorated function as the method `name`, taking `options`."""

    def decorator(function: Callable[[Problem, Any], Posterior]) -> Callable:
        """File `function` under `name` and return it unchanged."""
        _METHODS[name] = _Method(options, function)
        return function

    return decorator


def available_methods() -> list[str]:
    """
    Return the names of the methods :func:`solve` can use.

    Returns
    -------
    list of str
        The registered names, such as ``"analytic"``.
    """
    return list(_METHODS)


@_register("analytic", AnalyticOptions)
def _analytic(problem: Problem, options: AnalyticOptions) -> Posterior:
    r"""
    Factor the problem's whitened system once; the posterior derives everything from it.

    :math:`H S_0` and :math:`H S_0 H^T` are formed in the cheapest form of
    their factors (through a Kronecker :math:`S_0`'s factors), then dense.
    """
    H = problem.forward.data
    S_0 = problem.prior_cov.data
    HS_0 = H @ S_0
    HS_0H = HS_0 @ H._transpose()
    innovation = problem.obs.values - (
        H @ problem.prior.values + problem._offset_values()
    )
    factors = Whitened.factor(
        HS_0.dense(),
        HS_0H.dense(),
        problem.obs_cov.data,
        innovation,
        S_0.diagonal(),
    )
    return Posterior(problem, factors, options)


def solve(problem: Problem, method: str = "analytic", **options: Any) -> Posterior:
    """
    Solve an inverse problem and return its posterior.

    The problem is not changed.

    Parameters
    ----------
    problem : Problem
        The problem to solve.
    method : str, default "analytic"
        The name of a method (see :func:`available_methods`). ``"analytic"``
        is the closed-form Gaussian posterior.
    **options
        The method's options, by name: for ``"analytic"``, ``gamma`` (see
        :class:`~fips.solve.AnalyticOptions`).

    Returns
    -------
    Posterior
        The labeled solution; its estimate is computed before it returns.

    Raises
    ------
    ValueError
        If `method` is not a registered name, or an option is unknown to the
        method or has a value it refuses; the message lists what it knows.
    TypeError
        If `problem` is not a :class:`~fips.Problem`.

    See Also
    --------
    fips.Posterior : What it returns.
    available_methods : The names `method` can be.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> cells = pd.Index(["a", "b"], name="cell")
    >>> obs_id = pd.Index(["o1", "o2", "o3"], name="obs_id")
    >>> prior = fips.Vector.from_series(pd.Series(0.0, index=cells), block="flux")
    >>> obs = fips.Vector.from_series(
    ...     pd.Series([1.0, 2.0, 3.0], index=obs_id), block="co2"
    ... )
    >>> problem = fips.Problem(
    ...     prior=prior,
    ...     prior_cov=fips.Independent().with_std(1.0),
    ...     obs=obs,
    ...     obs_cov=fips.Independent().with_std(np.sqrt(0.1)),
    ...     forward=fips.Operator(
    ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), obs.space, prior.space
    ...     ),
    ... )
    >>> post = fips.solve(problem, gamma=0.5)
    >>> post.options
    AnalyticOptions(gamma=0.5)
    >>> fips.solve(problem, gama=0.5)
    Traceback (most recent call last):
    ...
    ValueError: Method 'analytic' has no option 'gama'; its options are: gamma.
    """
    if not isinstance(problem, Problem):
        raise TypeError(f"solve takes a Problem, got {type(problem).__name__}.")
    if method not in _METHODS:
        raise ValueError(
            f"Unknown method {method!r}; the methods are: "
            f"{', '.join(available_methods())}."
        )
    registered = _METHODS[method]
    known = [f.name for f in fields(registered.options)]
    unknown = [name for name in options if name not in known]
    if unknown:
        raise ValueError(
            f"Method {method!r} has no option {', '.join(map(repr, unknown))}; "
            f"its options are: {', '.join(known)}."
        )
    logger.info("Solving with the %s method...", method)
    posterior = registered.solve(problem, registered.options(**options))
    _ = posterior.estimate  # solve now
    return posterior
