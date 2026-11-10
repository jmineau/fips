"""
Solving an inverse problem: :func:`solve` and the methods it can use.

Each method is registered under a name with the dataclass of its options,
declared once: :func:`solve` checks the options a caller passes against it
before anything is computed, and an unknown method or option raises with the
names it knows. The methods are ``"analytic"`` (:mod:`fips.methods.analytic`),
the closed-form Gaussian posterior; ``"bounded"``
(:mod:`fips.methods.bounded`), that posterior with bounds on the state: its
mode, and draws of the Gaussian truncated to the bounds (or rectified
draws, the bounded problem solved again for draws of the prior and the
data); and
``"gauss_newton"`` (:mod:`fips.methods.gauss_newton`), the mode of the
posterior of a problem with a forward model, and draws of that posterior; and
``"ensemble"`` (:mod:`fips.methods.ensemble`), a square-root ensemble Kalman
update of members drawn from the prior, for a model that can only be run;
and ``"variational"`` (:mod:`fips.methods.variational`), the estimate by
conjugate gradients and a low-rank posterior by Lanczos, from products
alone, for a problem too large to factor.

A method is a module here with three things, and a line in ``_METHODS``
below:

- its options, a frozen dataclass with a ``gamma`` field (the regularization
  factor; :class:`~fips.posterior.Options`) and any others it needs;
- its solution, the object it keeps from a solve, which answers the questions
  of :class:`~fips.posterior.Solution` with unlabeled arrays in the order of
  the problem's spaces: ``estimate()``, ``variances()``, ``cov(W)``,
  ``gain(W)``, ``dofs()``, ``dofs_per_element()``, ``information()``,
  ``chi2()``, ``misfits()`` and ``chi2_tests()``. It may also have
  ``with_gamma(gamma)`` and ``with_data(problem)``, which give another
  solution without solving again, ``sample(N, rng)``, which draws from the
  posterior (``Posterior.sample``), ``at_bound()``, which says which state
  elements sit on a bound (``Posterior.at_bound``), ``iterations()`` and
  ``linearized()`` for an iterated method (``Posterior.iterations``, and the
  linear problem its gain is of), ``spectrum()`` for a low-rank posterior
  (``Posterior.eigenvalues``), ``estimate_is_mean = True`` when its
  estimate is the posterior mean, and ``estimate_is_linear = True`` when its
  estimate is :math:`x_0 + K (z - H x_0 - c)` with the gain of ``gain()``
  (what ``Posterior.recovery`` needs), and ``evidence_is_laplace = True``
  when its posterior is the Gaussian at its estimate, exactly or
  approximately, so that ``chi2()`` and ``information()`` give the evidence
  (``Posterior.log_evidence``); a posterior whose solution lacks one
  of them raises when that feature is asked for;
- a function from a :class:`~fips.Problem` and the options to
  ``Posterior(problem, solution, options)``.

A method's default options on a linear problem with no bounds give the
analytic posterior: ``tests/test_oracle.py`` runs every registered method,
with its default options, against the closed form. A method registered with
``exact=False`` (the ensemble, which approaches it only as its members grow)
is left out of the oracle and has its own convergence test. One that is
exact only to the tolerance of its iterative solves (the variational method)
is registered with the options the oracle solves it with (``oracle``) and
the tolerance the oracle holds it to (``oracle_rtol``).

The :class:`~fips.Posterior` puts the labels on, keeps each output, and
writes everything else (the averaging kernel, the innovation covariance,
Desroziers, leverage, the scalars, the summary) from those answers and the
problem, so a new method gets them all. The analytic method's solution is
``Whitened`` or ``Information``, the factorization of its whitened
(``m x m``) or information (``n x n``) system, whichever its size calls for.
The bounded method's solution (``fips.methods.bounded.Bounded``) holds the
analytic one and the mode, and answers the variances from its draws (of its
truncated posterior, or rectified) and the gain from the Gaussian at the mode; it has no
``estimate_is_mean`` and no ``estimate_is_linear`` (the mode is not linear in
the data), and its ``kept`` draws give the posterior's mean and intervals, ``evidence_is_laplace`` only with no finite bound, and its ``with_gamma`` and ``with_data`` find the mode again on the
analytic factors. The Gauss-Newton method's solution
(``fips.methods.gauss_newton.GaussNewton``) keeps its mode and the analytic
solution of the problem linearized at its last iterate (a lognormal flux is
such a forward model), answers the gain from it and the variances, the mean
and the intervals from its ``kept`` draws of the posterior, which it
preconditions, and its evidence is Laplace's from
the Gaussian at the mode; it has no ``with_gamma``, since its
linearization depends on gamma. The ensemble method's solution
(``fips.methods.ensemble.Ensemble``) keeps the prior members, their modeled
observations and the factors of the update, and answers from the updated
members (``estimate_is_mean = True``); its ``with_gamma`` and ``with_data``
reuse the modeled members. The variational method's solution
(``fips.methods.variational.Variational``) keeps the whitened estimate and
the leading eigenpairs of the prior-preconditioned Hessian, and answers from
the low-rank posterior they make (``estimate_is_mean`` for a linear
problem); its ``with_gamma`` and ``with_data`` take one CG solve each, the
eigenpairs kept, for a linear problem. A method imports
:mod:`fips.posterior`; :mod:`fips.posterior` never imports a method.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

from fips.methods.analytic import AnalyticOptions, _analytic
from fips.methods.bounded import BoundedOptions, _bounded
from fips.methods.ensemble import EnsembleOptions, _ensemble
from fips.methods.gauss_newton import GaussNewtonOptions, _gauss_newton
from fips.methods.variational import VariationalOptions, _variational
from fips.posterior import Posterior
from fips.problem import Problem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Method:
    """A registered method: the dataclass of its options, the function that solves, and whether it is exact."""

    options: type
    solve: Callable[[Problem, Any], Posterior]
    exact: bool = True
    """False for a method whose posterior approaches the analytic one only as a number of members grows (``tests/test_oracle.py`` holds the others to it)."""
    oracle: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    """The options ``tests/test_oracle.py`` solves with, for a method exact only to the tolerance of its iterative solves (none for a direct one)."""
    oracle_rtol: float = 1e-10
    """The relative tolerance ``tests/test_oracle.py`` holds the method to."""


_METHODS: dict[str, _Method] = {
    "analytic": _Method(AnalyticOptions, _analytic),
    "bounded": _Method(BoundedOptions, _bounded),
    "gauss_newton": _Method(GaussNewtonOptions, _gauss_newton),
    "ensemble": _Method(EnsembleOptions, _ensemble, exact=False),
    "variational": _Method(
        VariationalOptions,
        _variational,
        oracle=MappingProxyType({"cg_tol": 1e-13}),
        oracle_rtol=1e-8,
    ),
}
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
        is the closed-form Gaussian posterior; ``"bounded"`` the posterior
        with bounds on the state; ``"gauss_newton"`` the mode of the
        posterior of a problem with a forward model, and draws of it;
        ``"ensemble"`` the update of an ensemble of members, for a model that
        can only be run; ``"variational"`` the estimate by conjugate gradients
        and a low-rank posterior, from products alone, for a problem too
        large to factor.
    **options
        The method's options, by name: for ``"analytic"``, ``gamma`` (see
        :class:`~fips.methods.analytic.AnalyticOptions`); for ``"bounded"``,
        ``gamma``, ``lower``, ``upper``, ``draws``, ``burn_in``, ``seed`` and
        ``distribution`` (see :class:`~fips.methods.bounded.BoundedOptions`); for
        ``"gauss_newton"``, ``gamma``, ``max_iterations``, ``tol``, ``draws``,
        ``burn_in`` and ``seed`` (see
        :class:`~fips.methods.gauss_newton.GaussNewtonOptions`); for
        ``"ensemble"``, ``gamma``, ``members``, ``predicted``,
        ``localization``, ``obs_localization`` and ``seed`` (see
        :class:`~fips.methods.ensemble.EnsembleOptions`); for
        ``"variational"``, ``gamma``, ``rank``, ``cg_tol``,
        ``max_cg_iterations``, ``max_iterations`` and ``tol`` (see
        :class:`~fips.methods.variational.VariationalOptions`).

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
