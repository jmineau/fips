"""
Solving an inverse problem: :func:`solve` and the methods it can use.

Each method is registered under a name with the dataclass of its options,
declared once: :func:`solve` checks the options a caller passes against it
before anything is computed, and an unknown method or option raises with the
names it knows. The one method is ``"analytic"`` (:mod:`fips.methods.analytic`),
the closed-form Gaussian posterior.

A method is a module here with three things, and a line in ``_METHODS``
below:

- its options, a frozen dataclass with a ``gamma`` field (the regularization
  factor; :class:`~fips.posterior.Options`) and any others it needs;
- its solution, the object it keeps from a solve, which answers the questions
  of :class:`~fips.posterior.Solution` with unlabeled arrays in the order of
  the problem's spaces: ``estimate()``, ``variances()``, ``cov(W)``,
  ``gain(W)``, ``dofs()`` and ``chi2()``. It may also have
  ``with_gamma(gamma)`` and ``with_data(problem)``, which give another
  solution without solving again, and ``estimate_is_mean = True`` when its
  estimate is the posterior mean; a posterior whose solution lacks one of
  them raises when that feature is asked for;
- a function from a :class:`~fips.Problem` and the options to
  ``Posterior(problem, solution, options)``.

The :class:`~fips.Posterior` puts the labels on, keeps each output, and
writes everything else (the averaging kernel, the innovation covariance,
Desroziers, leverage, the scalars, the summary) from those answers and the
problem, so a new method gets them all. The analytic method's solution is
``Whitened``, the factorization of its whitened system. A lognormal method
would keep its estimate (the mode) and the Gaussian of the log state around
it, and answer from those carried back to the state (the gain of the problem
linearized at the mode; no ``estimate_is_mean``); an ensemble method would keep its posterior ensemble and answer from the
sample mean and covariance (``estimate_is_mean = True``). Neither can change
gamma without solving again, so neither has ``with_gamma``. A method imports
:mod:`fips.posterior`; :mod:`fips.posterior` never imports a method.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from fips.methods.analytic import AnalyticOptions, _analytic
from fips.posterior import Posterior
from fips.problem import Problem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Method:
    """A registered method: the dataclass of its options and the function that solves."""

    options: type
    solve: Callable[[Problem, Any], Posterior]


_METHODS: dict[str, _Method] = {"analytic": _Method(AnalyticOptions, _analytic)}
"""The methods by name."""


def available_methods() -> list[str]:
    """
    Return the names of the methods :func:`solve` can use.

    Returns
    -------
    list of str
        The registered names, such as ``"analytic"``.
    """
    return list(_METHODS)


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
        :class:`~fips.methods.analytic.AnalyticOptions`).

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
