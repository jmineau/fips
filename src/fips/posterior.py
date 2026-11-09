r"""
The posterior of a solved inverse problem.

:func:`fips.solve` returns a :class:`Posterior`: the problem, the options of
the method, and what the method kept from its solve (a :class:`Solution`).
The posterior labels the answer; the method keeps its own machinery. It asks
the solution for unlabeled arrays (the estimate, the variances, the
covariance of aggregates, the gain, DOFS and its share per state element, the
information content, chi-squared and its two misfits), puts the problem's labels on them and keeps
each one once computed. Everything else it gives is
written from those and the problem, so it holds for every method. The
analytic method's solution is the factorization of its whitened system, or of
its information system when that is smaller (:mod:`fips.methods.analytic`).

The :class:`~fips.Problem` is the question and the posterior the answer: the
posterior holds what depends on the solve or on the ``m x m`` matrix it
factors; what needs neither (:meth:`Problem.predict <fips.Problem.predict>`)
is on the problem.
"""

import dataclasses
import logging
import numbers
from collections.abc import Callable
from functools import cached_property
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.linalg import inv, solve
from scipy.special import chdtrc, ndtr, ndtri  # chi-squared sf, normal CDF, inverse

from fips.covariance import Covariance, CovarianceModel
from fips.linalg import LinearOperator
from fips.linalg.forms import _adopt, _solve_psd
from fips.operator import Operator
from fips.problem import Problem, _covariance
from fips.vector import Vector

logger = logging.getLogger(__name__)


class Solution(Protocol):
    r"""
    What a :class:`Posterior` asks of the object a method keeps from its solve.

    A method solves a :class:`~fips.Problem` and keeps whatever it needs to
    answer these questions (the analytic method keeps a factorization of
    its system; a sampling method would keep its samples). Each answer is an unlabeled NumPy array (or a float), in the order of the
    problem's spaces: ``n`` the state elements (``problem.prior``), ``m`` the
    observations (``problem.obs``). The posterior labels them, and computes
    each one once.

    Nine members are optional, and a posterior whose solution lacks one says
    so when it is asked for the feature (or, for ``linearized``, uses its own
    problem):

    - ``with_gamma(gamma) -> Solution``: the solution at another
      regularization factor, without solving again (:meth:`Posterior.with_gamma`).
    - ``with_data(problem) -> Solution``: the solution of `problem`, the same
      problem with other values of the observations, without solving again
      (:meth:`Posterior.with_data`).
    - ``innovation_precision(groups) -> list[np.ndarray]``: for each array of
      observation positions :math:`g`, the block :math:`P_{gg}` of
      :math:`P = (H S_0 H^T + S_z/\gamma)^{-1}` (``m_g x m_g``), and
      ``self_sensitivity() -> np.ndarray``: :math:`\operatorname{diag}(H K)`
      (``m``), whose sum is DOFS (:meth:`Posterior.data_worth`).
    - ``sample(N, rng) -> np.ndarray``: ``N`` draws from the posterior, one per
      column (``n x N``), made with the :class:`numpy.random.Generator`
      `rng` (:meth:`Posterior.sample`).
    - ``at_bound() -> np.ndarray``: which state elements sit on a bound at the
      estimate, booleans (``n``; :attr:`Posterior.at_bound`).
    - ``iterations() -> tuple[np.ndarray, ...]``: for each iteration of
      an iterated method, the size of its step and the cost after it, and
      for a method that solves by conjugate gradients their iterations
      (:attr:`Posterior.iterations`).
    - ``spectrum() -> np.ndarray``: for a low-rank posterior, the eigenvalues
      of the prior-preconditioned Hessian it keeps, from the largest
      (:attr:`Posterior.eigenvalues`).
    - ``linearized() -> Problem``: for a method that solves a problem with a
      forward model, the linear problem its gain is of (the problem
      linearized where its last solve was made). The averaging kernel and
      the innovation covariance read their forward operator from it; without
      it, from the posterior's problem.

    And a solution whose :meth:`estimate` is the posterior mean says so with a
    class attribute ``estimate_is_mean = True``; without it, a posterior has
    no :attr:`~Posterior.mean`. One whose estimate is linear in the
    observations, :math:`\hat{x} = x_0 + K (z - H x_0 - c)` with the
    :math:`K` of :meth:`gain` whatever the values of :math:`z`, says so with
    ``estimate_is_linear = True``; without it, a posterior has no
    :meth:`~Posterior.recovery`. One whose posterior is the Gaussian at its
    estimate, exactly or as Laplace's approximation, so that its
    :meth:`chi2` and :meth:`information` give the evidence, says so with
    ``evidence_is_laplace = True``; without it, a posterior has no
    :attr:`~Posterior.log_evidence`.

    A method whose posterior is not Gaussian in closed form answers each
    question as its docstring says. The bounded method's estimate is the mode;
    its variances and covariances are those of draws of its posterior (a
    Gaussian truncated to the bounds); its gain, DOFS and information content
    are those of the Gaussian at the mode, with the elements on a bound held
    there; and its chi-squared terms are those of the mode. The
    ``"gauss_newton"`` method's estimate is the mode; its variances and
    covariances are those of draws of its posterior; its gain, DOFS and
    information content those of the Gaussian of the problem linearized where
    its last step was made; and its chi-squared terms those of the model at
    its estimate. The ``"ensemble"`` method's estimate is the mean of its
    updated members; its variances, covariances and gain are those of the
    members, and its DOFS, information content and chi-squared the analytic
    formulas with the members' covariance of the modeled observations. The
    ``"variational"`` method's estimate is the solution of conjugate
    gradients (the mode, with a forward model); its variances, covariances,
    gain, DOFS and information content are those of its low-rank posterior,
    exact when its rank reaches the problem's.
    Such answers are estimates or approximations, and
    :meth:`Posterior.trials` measures how good they are.

    See Also
    --------
    Posterior : The labeled answer built on it.
    fips.methods : How a method is added.
    """

    def estimate(self) -> np.ndarray:
        r"""Return the point estimate of the state, :math:`\hat{x}` (``n``)."""
        ...

    def variances(self) -> np.ndarray:
        r"""Return the posterior error variances, the diagonal of :math:`\hat{S}` (``n``)."""
        ...

    def cov(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""
        Return the posterior error covariance of the state, or of aggregates of it.

        Parameters
        ----------
        W : LinearOperator, optional
            An aggregation, ``k x n``, its columns in the state's order.

        Returns
        -------
        np.ndarray
            Without `W`, :math:`\hat{S}` (``n x n``); with it,
            :math:`W \hat{S} W^T` (``k x k``). The posterior writes into
            neither, and keeps the first.
        """
        ...

    def gain(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""
        Return the gain :math:`K`, or :math:`W K` for an aggregation.

        Parameters
        ----------
        W : LinearOperator, optional
            An aggregation, ``k x n``, its columns in the state's order.

        Returns
        -------
        np.ndarray
            Without `W`, :math:`K` (``n x m``), with
            :math:`\hat{x} = x_0 + K (z - H x_0 - c)` for a linear method;
            with it, :math:`W K` (``k x m``). The averaging kernel is
            written from it: :math:`A = K H`.
        """
        ...

    def dofs(self) -> float:
        r"""Return the degrees of freedom for signal, :math:`\operatorname{tr}(K H)`."""
        ...

    def dofs_per_element(self) -> np.ndarray:
        r"""Return the diagonal of the averaging kernel, :math:`\operatorname{diag}(K H)` (``n``), whose sum is :meth:`dofs`."""
        ...

    def information(self) -> float:
        r"""Return the Shannon information content, :math:`-\tfrac{1}{2} \ln \lvert I - K H \rvert`, in nats."""
        ...

    def chi2(self) -> float:
        r"""Return the chi-squared of the solution, :math:`d^T (H S_0 H^T + S_z/\gamma)^{-1} d`: gamma times the data misfit plus the state misfit."""
        ...

    def misfits(self) -> tuple[float, float]:
        r"""Return the data misfit :math:`r^T S_z^{-1} r` and the state misfit :math:`(\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0)`, with :math:`r = z - H \hat{x} - c`."""
        ...

    def chi2_tests(self) -> dict[str, tuple[float, int]]:
        r"""
        Return Rodgers' four chi-squared tests, each its chi-squared and degrees of freedom.

        Keyed ``innovation`` (:math:`z - H x_0 - c`), ``residual``
        (:math:`z - H \hat{x} - c`), ``modeled_increment`` (:math:`H (\hat{x}
        - x_0)`) and ``increment`` (:math:`\hat{x} - x_0`): for each,
        :math:`v^T C^{+} v` and the rank of :math:`C`, its covariance
        (:meth:`Posterior.chi2_tests`, which adds the p-values). A method
        whose estimate is linear in the observations gives two pairs of the
        same numbers.
        """
        ...


class Options(Protocol):
    """
    What a :class:`Posterior` reads from the options of a method.

    A method's options are a frozen dataclass of its own
    (:class:`~fips.methods.analytic.AnalyticOptions`); the posterior keeps them
    as :attr:`Posterior.options`, reads only the regularization factor, and
    makes another gamma's with :func:`dataclasses.replace`.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

    @property
    def gamma(self) -> float:
        r"""The regularization factor :math:`\gamma`: the solve divided the model-data mismatch by it."""
        ...


def _obs_groups(index: pd.Index, by: Any) -> tuple[pd.Index, list[np.ndarray]]:
    """
    Return the groups `by` makes of the labels `index`, and the positions of each group's labels.

    The groups are those of :meth:`pandas.DataFrame.groupby` on the labels'
    levels, in its order; a label whose block lacks a level is in no group of
    it, and an empty group (a time bin with no label) is left out.

    Raises
    ------
    ValueError
        If no label is in a group.
    """
    grouped = pd.DataFrame({"position": np.arange(len(index))}, index=index).groupby(
        by
    )["position"]
    names = list(grouped.size().index.names)
    keys, positions = [], []
    for key, part in grouped:
        if len(part):
            if len(names) == 1 and isinstance(key, tuple) and len(key) == 1:
                key = key[0]  # a list of one key gives one-element tuples
            keys.append(key)
            positions.append(part.to_numpy())
    if not keys:
        raise ValueError("`by` puts no observation in a group.")
    if len(names) == 1:
        return pd.Index(keys, name=names[0]), positions
    return pd.MultiIndex.from_tuples(keys, names=names), positions


def _worth_table(
    groups: pd.Index,
    positions: list[np.ndarray],
    dofs: np.ndarray,
    aggregates: pd.Index,
    var_without: np.ndarray,
    var_with: np.ndarray,
    scale: np.ndarray,
    prior: np.ndarray | None = None,
    var_alone: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Return the table of what each group of observations does to each aggregate.

    One row per group and aggregate (the groups' levels, then the aggregates'):
    ``n_obs`` and ``dofs`` of the group, and the std of the aggregate without
    and with the group (`var_without` and `var_with` are ``groups x
    aggregates``; `scale` is the largest prior variance each aggregate could
    have, :func:`_largest_variance`, which a variance below zero is compared
    to: :func:`_std`). With `var_alone` (``groups x aggregates``: the variance
    with only the group's observations, on the prior) and `prior` (the prior
    variance of each aggregate), also ``std_alone`` and ``reduction_alone``. A
    group level named like a level of the aggregates (``block``) is called
    ``obs_<name>``: an index cannot have two levels of one name.
    """
    names = [
        f"obs_{name}" if name is not None and name in aggregates.names else name
        for name in groups.names
    ]
    n_groups, k = var_without.shape
    g, a = np.repeat(np.arange(n_groups), k), np.tile(np.arange(k), n_groups)
    index = pd.MultiIndex.from_arrays(
        [groups.get_level_values(i).take(g) for i in range(groups.nlevels)]
        + [aggregates.get_level_values(j).take(a) for j in range(aggregates.nlevels)],
        names=names + list(aggregates.names),
    )
    std_without = _std(var_without, scale, "variances without the group").ravel()
    std_with = _std(var_with, scale, "variances with the group").ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction = np.where(std_without > 0, 1.0 - std_with / std_without, np.nan)
    table = {
        "n_obs": np.repeat([len(rows) for rows in positions], k),
        "dofs": np.repeat(dofs, k),
        "std_without": std_without,
        "std_with": std_with,
        "reduction": reduction,
    }
    if var_alone is not None and prior is not None:
        std_alone = _std(var_alone, scale, "variances with the group alone").ravel()
        prior_std = np.tile(_std(prior, scale, "prior variances"), n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            table["std_alone"] = std_alone
            table["reduction_alone"] = np.where(
                prior_std > 0, 1.0 - std_alone / prior_std, np.nan
            )
    return pd.DataFrame(table, index=index)


def _stacked(top: LinearOperator, bottom: LinearOperator) -> LinearOperator:
    """Return the rows of `top` above those of `bottom`, sparse if both are."""
    a, b = top._csr(), bottom._csr()
    if a is not None and b is not None:
        return _adopt(sp.vstack([a, b], format="csr"))
    return _adopt(np.vstack([top.dense(), bottom.dense()]))


def _times_pseudo_inverse(XWt: np.ndarray, W: LinearOperator) -> np.ndarray:
    """
    Return ``X W*`` given ``X W^T``, where ``W* = W^T (W W^T)^-1`` is the pseudo-inverse of `W`.

    Raises
    ------
    ValueError
        If the rows of `W` are linearly dependent, so ``W W^T`` is singular.
    """
    G = (W @ W.T).dense()
    try:
        # X W^T (W W^T)^-1 is the transpose of (W W^T)^-1 (X W^T)^T
        return solve(G, XWt.T, assume_a="pos").T
    except LinAlgError as err:
        raise ValueError(
            "The rows of the aggregation are linearly dependent, so it has no "
            "pseudo-inverse."
        ) from err


def _largest_variance(Xt: np.ndarray | None, cov: LinearOperator) -> np.ndarray:
    r"""
    Return the largest variance of each column of `Xt` under any covariance with `cov`'s variances.

    That is :math:`(\sum_i |x_i| \sigma_i)^2`, the variance of :math:`x^T e`
    were the errors :math:`e` perfectly correlated. A positive semidefinite
    :math:`S` has :math:`|S_{ij}| \le \sigma_i \sigma_j`, so this bounds
    :math:`|x|^T |S| |x|`, the size of the terms :math:`x^T S x` is summed
    from: the scale its roundoff is relative to. `Xt` None stands for the
    identity (each element's own variance).
    """
    sigma = np.sqrt(np.maximum(cov.diagonal(), 0.0))
    if Xt is None:
        return sigma**2
    return (np.abs(Xt).T @ sigma) ** 2


_VARIANCE_RTOL = float(np.sqrt(np.finfo(float).eps))
"""How far below zero a variance may be and be roundoff: 1.5e-8 of its scale (:func:`_std`)."""


def _std(variance: np.ndarray, scale: np.ndarray, what: str) -> np.ndarray:
    r"""
    Return the square root of `variance`: 0 where it is below zero by roundoff, NaN (logged) where it is below that.

    A variance :math:`x^T S x` of a positive semidefinite :math:`S` is not
    negative, but computed it can come out below zero: by a few ``eps`` of
    :math:`|x|^T |S| |x|` (which `scale` bounds, :func:`_largest_variance`)
    when it is summed, and by more when it comes from a solve, which loses
    digits with the condition number. Below zero by at most
    ``sqrt(eps) * scale`` (1.5e-8 of it: what a double-precision solve keeps
    up to a condition number of 1e8) is roundoff, and its square root is 0.
    One further below zero is not: a covariance is not positive
    semidefinite, and its square root is NaN, with one warning naming `what`.
    """
    roundoff = _VARIANCE_RTOL * scale
    negative = variance < -roundoff
    if negative.any():
        lowest = np.unravel_index(
            np.argmax(np.where(negative, -variance, -np.inf)), variance.shape
        )
        logger.warning(
            "%d of the %d %s are below zero by more than roundoff (the lowest is "
            "%.3g, where roundoff is %.3g); their standard deviation is NaN. A "
            "variance is negative only when a covariance is not positive "
            "semidefinite: check the prior error and model-data mismatch "
            "covariances.",
            int(negative.sum()),
            negative.size,
            what,
            variance[lowest],
            np.broadcast_to(roundoff, variance.shape)[lowest],
        )
    return np.sqrt(np.where(negative, np.nan, np.maximum(variance, 0.0)))


class Posterior:
    r"""
    The solution of an inverse problem, labeled.

    :func:`fips.solve` builds it; there is no need to build one by hand. It
    keeps what the method kept from the solve (its :class:`Solution`), and
    each output is computed from it when first asked for, labeled with the
    problem's spaces, then kept. For the ``"analytic"`` method the posterior
    is Gaussian: :attr:`estimate` is its mean (so :attr:`mean` is the same
    object), and the posterior covariance is
    :math:`\hat{S} = (H^T S_z^{-1} H + S_0^{-1})^{-1}`, with :math:`S_z`
    divided by the regularization factor :math:`\gamma`. It holds what depends
    on the solve or on the ``m x m`` innovation covariance
    (:attr:`innovation_cov`); what needs neither is on the
    :class:`~fips.Problem` (:meth:`~fips.Problem.predict`).

    Every member works for every method, except those that need more of the
    method than a :class:`Solution` gives: :attr:`mean` (where the estimate is
    the mean), :meth:`with_gamma` and :meth:`with_data` (where the method can
    answer without solving again), :meth:`recovery` and
    :meth:`candidate_worth` (where the estimate is linear in the
    observations), :meth:`data_worth`, :meth:`sample` (where the method can
    draw from its posterior), :attr:`at_bound` (where it has bounds),
    :attr:`iterations` (where it iterates) and :attr:`log_evidence` (where
    the posterior is the Gaussian at the estimate). The analytic method has
    all but :attr:`at_bound` and :attr:`iterations`. The ``"bounded"`` method
    (:class:`~fips.methods.bounded.BoundedOptions`) has :meth:`with_gamma`,
    :meth:`with_data`, :meth:`sample` and :attr:`at_bound`, and
    :attr:`log_evidence` only with no finite bound; its estimate is
    the mode of the posterior with bounds on the state, and its variances are
    those of draws of that posterior. The ``"gauss_newton"`` method
    (:class:`~fips.methods.gauss_newton.GaussNewtonOptions`), for a problem
    with a forward model, has :meth:`with_data`, :meth:`sample`,
    :attr:`iterations` and :attr:`log_evidence` (Laplace's approximation);
    its estimate is the mode, its variances are those of
    draws of its posterior, and its gain and averaging kernel are those of
    the Gaussian at the mode. Its
    :attr:`problem` is the problem with the model, as solved; what needs a
    linear forward operator (the averaging kernel, :attr:`innovation_cov`)
    reads the Jacobian of the last step.
    The ``"ensemble"`` method
    (:class:`~fips.methods.ensemble.EnsembleOptions`) has :attr:`mean`,
    :meth:`with_gamma`, :meth:`with_data` and :meth:`sample` (its updated
    members, at most as many as it has); its estimate, variances, gain and
    diagnostics are those of its members, which approach the analytic
    posterior as they grow, and what needs :math:`H` (the averaging kernel,
    :attr:`dofs_per_element`) raises for a model.

    Parameters
    ----------
    problem : Problem
        The problem that was solved.
    solution : Solution
        What the method kept from the solve, which gives the unlabeled
        arrays (for the analytic method, a factorization:
        ``fips.methods.analytic.Whitened`` or ``Information``).
    options : Options
        The options of the method, such as
        :class:`fips.methods.analytic.AnalyticOptions`.

    See Also
    --------
    fips.solve : Solve a problem and return its posterior.

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
    >>> post = fips.solve(problem)
    >>> post.estimate.values.round(3)
    array([0.997, 1.906])
    >>> round(post.dofs, 3)
    1.877
    """

    def __init__(self, problem: Problem, solution: Solution, options: Options):
        self._problem = problem
        self._solution = solution
        self._options = options

    @property
    def problem(self) -> Problem:
        """The problem that was solved."""
        return self._problem

    @property
    def options(self) -> Options:
        """The options the method solved with (``gamma``)."""
        return self._options

    @property
    def _gamma(self) -> float:
        """The regularization factor."""
        return self._options.gamma

    def __repr__(self) -> str:
        """Return the options and the sizes."""
        return (
            f"Posterior({self._options!r}, n_state={len(self._problem.prior)}, "
            f"n_obs={len(self._problem.obs)})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return the problem, the options and the solution; the outputs are rebuilt."""
        return {
            "problem": self._problem,
            "options": self._options,
            "solution": self._solution,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`."""
        self._problem = state["problem"]
        self._options = state["options"]
        self._solution = state["solution"]

    @cached_property
    def _linear(self) -> Problem:
        """
        The linear problem the gain, the averaging kernel and the innovation covariance are of.

        The posterior's own problem, or, for a method that solves a problem
        with a forward model, the problem linearized where its last solve was
        made (the solution's ``linearized``).
        """
        linearized = getattr(self._solution, "linearized", None)
        return self._problem if linearized is None else linearized()

    def _optional(self, name: str, feature: str) -> Any:
        """
        Return the solution's optional member `name`.

        Raises
        ------
        TypeError
            If the solution does not have it: the method cannot give `feature`.
        """
        member = getattr(self._solution, name, None)
        if member is None:
            raise TypeError(
                f"The method that solved this problem cannot give {feature} "
                f"({type(self._solution).__name__} has no {name}); solve the "
                "problem again with fips.solve."
            )
        return member

    # -- the state -----------------------------------------------------------

    @cached_property
    def estimate(self) -> Vector:
        r"""
        The posterior state estimate, :math:`\hat{x}`, on the prior's space.

        The point estimate of the method; for the analytic method the
        posterior mean, :math:`\hat{x} = x_0 + K (z - H x_0 - c)`.

        See Also
        --------
        mean : The posterior mean.
        variances : The posterior error variances.
        """
        values = self._solution.estimate()
        return Vector(values, self._problem.prior.space, name="estimate")

    @property
    def mean(self) -> Vector:
        """
        The posterior mean, :attr:`estimate` itself.

        The analytic method's posterior is Gaussian, so its estimate is its
        mean. A posterior whose method gives another point estimate (a mode)
        has no mean.

        Raises
        ------
        AttributeError
            If the method's estimate is not the posterior mean.
        """
        if not getattr(self._solution, "estimate_is_mean", False):
            raise AttributeError(
                "The method that solved this problem does not give the posterior "
                "mean; its point estimate is `estimate`."
            )
        return self.estimate

    @cached_property
    def variances(self) -> Vector:
        r"""
        The posterior error variances, the diagonal of :math:`\hat{S}`.

        The analytic method computes them from its factors in
        :math:`O(m n)`, without the :math:`n \times n` covariance. The
        ``"bounded"`` method gives the variances of its kept draws of the
        truncated posterior (``draws`` of them: a Monte Carlo estimate, a
        standard deviation good to about 2% at the default 2000), about
        their mean, not about the mode; the ``"gauss_newton"`` method those
        of its kept draws of the posterior of its forward model, the same
        way; the ``"ensemble"`` method those of its updated members.

        See Also
        --------
        cov : The posterior error covariance, of the state or of aggregates.
        """
        values = self._solution.variances()
        return Vector(values, self._problem.prior.space, name="variance")

    @cached_property
    def at_bound(self) -> pd.Series:
        r"""
        Which state elements sit on a bound at the estimate: True where one does.

        A method with bounds on the state (``"bounded"``) moves an element that
        the analytic estimate puts outside its bounds onto the bound, and may
        hold others there that the move would push out. These are the elements
        the bounds decide; the observations decide the others. Its estimate is
        the mode, so an element on a bound still has posterior spread
        (:attr:`variances`, from draws of the truncated posterior), on one side
        of the bound.

        Returns
        -------
        pandas.Series
            Booleans on the state's labels, named ``at_bound``; its sum is the
            number of elements on a bound.

        Raises
        ------
        TypeError
            If the method has no bounds (its solution has no ``at_bound``),
            as for the analytic method, and for ``"bounded"`` when no finite
            bound is given (its posterior is then the analytic one).

        See Also
        --------
        estimate : The estimate, the mode for a bounded method.
        fips.methods.bounded.BoundedOptions : The bounds.

        Examples
        --------
        Three cells seen through two sums, :math:`a + b = -0.4` and :math:`b
        + c = 1.2`. Without a bound, ``a`` goes below zero. Held at or above
        zero, ``a`` goes to its bound and ``b``, which the first sum ties to
        it, is held there too; ``c`` moves up by what the second sum says
        ``b`` gave:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> cells = pd.Index(["a", "b", "c"], name="cell")
        >>> prior = fips.Vector.from_series(pd.Series(0.5, index=cells), block="flux")
        >>> obs = fips.Vector(
        ...     np.array([-0.4, 1.2]),
        ...     fips.Space({"co2": pd.Index(["o1", "o2"], name="obs_id")}),
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(0.5),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(0.1),
        ...     forward=fips.Operator(
        ...         np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), obs.space, prior.space
        ...     ),
        ... )
        >>> fips.solve(problem).estimate.values.round(3)
        array([-0.467,  0.105,  1.072])
        >>> post = fips.solve(problem, "bounded", lower=0.0)
        >>> post.estimate.values.round(3)
        array([0.   , 0.   , 1.173])
        >>> post.at_bound.tolist()
        [True, True, False]
        """
        at = self._optional("at_bound", "which state elements sit on a bound")
        return pd.Series(
            np.asarray(at(), dtype=bool),
            index=self._problem.prior.index,
            name="at_bound",
        )

    @cached_property
    def iterations(self) -> pd.DataFrame:
        r"""
        The steps of an iterated method: how far each moved the estimate, and the cost after it.

        ``"gauss_newton"`` solves the problem linearized at its last estimate,
        from the prior, until a step is small (Rodgers' rule): ``d2`` is the
        step :math:`\delta = x_{i+1} - x_i` measured in the posterior
        covariance of the solve that made it, :math:`\delta^T \hat{S}_i^{-1}
        \delta`, and the iterations stop when it is below ``tol`` times the
        number of state elements. ``cost`` is the cost at the step's estimate,
        :math:`\gamma` times the data misfit plus the state misfit; the last is
        :attr:`chi2`. Near the mode, Gauss-Newton's ``d2`` falls by orders of
        magnitude a step, and the cost settles. ``"variational"`` takes the
        same steps for a forward model (one for a linear problem), each solved
        by conjugate gradients, and adds their number, ``cg_iterations`` (each
        a product with :math:`H` and one with :math:`H^T`).

        Returns
        -------
        pandas.DataFrame
            One row per step, indexed by ``iteration`` (from 1), with the
            columns ``d2`` and ``cost`` (and ``cg_iterations`` for a method
            that solves by conjugate gradients); its length is the number of
            steps.

        Raises
        ------
        TypeError
            If the method does not iterate (its solution has no
            ``iterations``), as for the analytic method, and for
            ``"gauss_newton"`` on a problem whose forward is an operator (its
            posterior is the analytic one).

        See Also
        --------
        fips.methods.gauss_newton.GaussNewtonOptions : ``max_iterations`` and ``tol``.

        Examples
        --------
        Two observations of the squares of two state elements:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> from fips.linalg import DiagonalOperator
        >>> class Square(fips.ForwardModel):
        ...     def __call__(self, x):
        ...         return fips.Vector(x.values**2, self.rows)
        ...
        ...     def jacobian(self, x):
        ...         form = DiagonalOperator(2 * x.values)
        ...         return fips.Operator(form, self.rows, x.space)
        >>> state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
        >>> observed = fips.Space({"co2": pd.Index(["o1", "o2"], name="obs_id")})
        >>> problem = fips.Problem(
        ...     prior=fips.Vector([1.0, 2.0], state),
        ...     prior_cov=fips.Independent().with_std(0.5),
        ...     obs=fips.Vector([1.5, 3.0], observed),
        ...     obs_cov=fips.Independent().with_std(0.1),
        ...     forward=Square(observed, state),
        ... )
        >>> post = fips.solve(problem, "gauss_newton")
        >>> post.iterations.round(4).reset_index()
           iteration        d2    cost
        0          1  124.5031  1.2294
        1          2    0.7422  0.4870
        2          3    0.0000  0.4869
        3          4    0.0000  0.4869
        >>> post.estimate.values.round(3)
        array([1.223, 1.733])
        >>> post.modeled_obs.values.round(3)  # F(x) at the mode, near the data
        array([1.496, 3.003])
        """
        history = self._optional("iterations", "the steps of an iterated solve")
        d2, cost, *cg = history()
        columns = {
            "d2": np.asarray(d2, dtype=float),
            "cost": np.asarray(cost, dtype=float),
        }
        if cg:
            columns["cg_iterations"] = np.asarray(cg[0], dtype=int)
        return pd.DataFrame(
            columns, index=pd.RangeIndex(1, len(d2) + 1, name="iteration")
        )

    @cached_property
    def eigenvalues(self) -> pd.DataFrame:
        r"""
        The eigenvalues a low-rank posterior keeps, and the DOFS each adds: whether its rank was enough.

        ``"variational"`` keeps the ``rank`` leading eigenpairs of the
        prior-preconditioned Hessian, :math:`\tilde{H} = \gamma R^T H^T
        S_z^{-1} H R` (:math:`S_0 = R R^T`). Along the eigenvector of
        :math:`\tilde\lambda_j` the data remove the share
        :math:`\tilde\lambda_j / (1 + \tilde\lambda_j)` of the prior's
        variance, which is that pair's DOFS; the directions left out keep
        the prior's variance. When the last pairs add little to DOFS (their
        eigenvalues well below 1), ``rank`` held what the data inform; when
        they add much, a larger ``rank`` lowers the variances further.

        Returns
        -------
        pandas.DataFrame
            One row per pair, indexed by ``pair`` (from 1, the largest
            eigenvalue first), with the columns ``eigenvalue``
            (:math:`\tilde\lambda_j`), ``dofs`` (its share) and
            ``cumulative_dofs``, whose last value is :attr:`dofs`.

        Raises
        ------
        TypeError
            If the method keeps no low-rank posterior (its solution has no
            ``spectrum``), as for the analytic method.

        See Also
        --------
        fips.methods.variational.VariationalOptions : ``rank``.
        """
        spectrum = self._optional("spectrum", "the eigenvalues of a low-rank posterior")
        eigenvalues = np.asarray(spectrum(), dtype=float)
        dofs = eigenvalues / (1.0 + eigenvalues)
        return pd.DataFrame(
            {
                "eigenvalue": eigenvalues,
                "dofs": dofs,
                "cumulative_dofs": np.cumsum(dofs),
            },
            index=pd.RangeIndex(1, len(eigenvalues) + 1, name="pair"),
        )

    def _aggregation(self, W: Operator) -> LinearOperator:
        """
        Return an aggregation's matrix with its columns in the state's order.

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If its columns do not have exactly the state's labels.
        """
        if not isinstance(W, Operator):
            raise TypeError(
                f"An aggregation is an Operator, not {type(W).__name__}: build one "
                "with Operator.from_frame or Operator.from_blocks."
            )
        return W._on(
            W.rows,
            self._problem.prior.space,
            "the aggregation",
            ("its rows", "the state"),
        )

    @cached_property
    def _cov(self) -> Covariance:
        """The full posterior covariance, formed once."""
        S_hat = self._solution.cov()
        return Covariance._computed(
            _adopt(S_hat), self._problem.prior.space, name="posterior_cov"
        )

    def cov(self, W: Operator | None = None) -> Covariance:
        r"""
        Return the posterior error covariance, of the state or of aggregates of it.

        Parameters
        ----------
        W : Operator, optional
            An aggregation, ``k x n``: one row per aggregate (a region, a
            month, a total), one column per state element, with the state's
            labels as its columns (in any order). Row ``i`` holds the weight
            of each state element in aggregate ``i``: ones for a sum,
            ``1/count`` for a mean. It may be sparse.

        Returns
        -------
        Covariance
            Without `W`, :math:`\hat{S}`, dense, ``n x n`` (formed once, the
            same object each time). With `W`, :math:`W \hat{S} W^T` on the
            aggregation's rows, ``k x k``, computed without the ``n x n``
            covariance. For the ``"bounded"`` and ``"gauss_newton"``
            methods, the covariance of their kept draws of the posterior (or
            of the draws aggregated, :math:`W x`): a Monte Carlo estimate.

        Raises
        ------
        ValueError
            If the columns of `W` do not have exactly the state's labels.

        See Also
        --------
        variances : The diagonal of the full covariance.
        averaging_kernel : The averaging kernel, of the state or of aggregates.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> cells = pd.Index(["a", "b", "c"], name="cell")
        >>> prior = fips.Vector.from_series(pd.Series(0.0, index=cells), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0], index=pd.Index(["o1", "o2"], name="obs_id")),
        ...     block="co2",
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(np.sqrt(0.1)),
        ...     forward=fips.Operator(
        ...         np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), obs.space, prior.space
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> total = fips.Operator(
        ...     np.ones((1, 3)),
        ...     fips.Space({"total": pd.Index(["all"], name="region")}),
        ...     prior.space,
        ... )
        >>> post.cov(total).dense().round(3)
        array([[0.419]])
        """
        if W is None:
            return self._cov
        values = self._solution.cov(self._aggregation(W))
        return Covariance._computed(_adopt(values), W.rows, name="posterior_cov")

    def _prior_scale(self, W_form: LinearOperator | None) -> np.ndarray:
        """
        Return the largest prior variance each aggregate could have, which its roundoff is relative to.

        :func:`_largest_variance` of the prior error covariance for the rows of
        `W_form` (the state's elements for None).
        """
        Wt = None if W_form is None else W_form._transpose().dense()  # n x k
        return _largest_variance(Wt, self._problem.prior_cov.data)

    def uncertainty(self, W: Operator | None = None) -> pd.DataFrame:
        r"""
        Return the prior and posterior uncertainty of each aggregate, and how much the observations reduce it.

        The standard deviation of each row of :math:`W x` before the
        observations, :math:`\sqrt{\operatorname{diag} W S_0 W^T}`, and after,
        :math:`\sqrt{\operatorname{diag} W \hat{S} W^T}`, from
        :meth:`Covariance.aggregate <fips.Covariance.aggregate>` and
        :meth:`cov`, neither forming an ``n x n`` matrix. A row of `W` can be
        any linear combination of the state: a total, a mean, or a combination
        of totals such as a trend or the difference of two seasons (``C @ W``,
        with ``C`` the weights of the rows of ``W``), whose uncertainty
        includes the covariance between the totals.

        Parameters
        ----------
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it: one row per aggregate. By
            default, one row per state element.

        Returns
        -------
        pandas.DataFrame
            One row per aggregate, on the aggregation's rows (or the state's
            labels): ``prior_std``, ``posterior_std`` and ``reduction``,
            :math:`1 - \sigma_{posterior} / \sigma_{prior}` (a fraction; NaN
            where the prior standard deviation is 0). A variance below zero by
            more than roundoff, which only a prior error or model-data
            mismatch covariance that is not positive semidefinite can give,
            makes its standard deviation and the reduction NaN, with a
            logged warning (as in :meth:`recovery`).

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If the columns of `W` do not have exactly the state's labels.

        See Also
        --------
        cov : The posterior covariance, of the state or of aggregates.
        fips.Covariance.aggregate : The prior covariance of aggregates, ``problem.prior_cov.aggregate(W)``.
        uncertainty_reduction : One number for the whole state.

        Examples
        --------
        Two months of two cells, observed six times; the total of each month,
        and the change from the first month to the second:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = pd.MultiIndex.from_product(
        ...     [pd.Index([1, 2], name="month"), pd.Index(["a", "b"], name="cell")]
        ... )
        >>> prior = fips.Vector.from_series(pd.Series(1.0, index=state), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series(0.0, index=pd.RangeIndex(6, name="obs_id")), block="ch4"
        ... )
        >>> rng = np.random.default_rng(0)
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(0.5),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(1.0),
        ...     forward=fips.Operator(
        ...         rng.uniform(0.0, 1.0, (6, 4)), obs.space, prior.space
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> monthly = fips.aggregate.sum(prior.space, by="month")
        >>> post.uncertainty(monthly).droplevel("block").round(3).reset_index()
           month  prior_std  posterior_std  reduction
        0      1      0.707          0.528      0.253
        1      2      0.707          0.586      0.171
        >>> change = fips.Operator(
        ...     np.array([[-1.0, 1.0]]),
        ...     fips.Space({"change": pd.Index(["2 - 1"], name="months")}),
        ...     monthly.rows,
        ... )
        >>> post.uncertainty(change @ monthly).round(3).reset_index()
            block months  prior_std  posterior_std  reduction
        0  change  2 - 1        1.0          0.921      0.079

        The change is more uncertain than either month's total: the two
        totals' posterior errors are negatively correlated (the observations
        see their sum better than how it splits), and the covariance between
        the rows enters the change's variance.
        """
        if W is None:
            prior = self._problem.prior_cov.variances.values
            posterior = self.variances.values
            scale = self._prior_scale(None)
            index = self._problem.prior.index
        else:
            posterior = self.cov(W).variances.values  # checks W against the state
            prior = self._problem.prior_cov.aggregate(W).variances.values
            scale = self._prior_scale(self._aggregation(W))
            index = W.rows.index
        prior_std = _std(prior, scale, "prior variances")
        posterior_std = _std(posterior, scale, "posterior variances")
        with np.errstate(divide="ignore", invalid="ignore"):
            reduction = np.where(prior_std > 0, 1.0 - posterior_std / prior_std, np.nan)
        return pd.DataFrame(
            {
                "prior_std": prior_std,
                "posterior_std": posterior_std,
                "reduction": reduction,
            },
            index=index,
        )

    def sample(
        self,
        n: int,
        rng: "int | np.random.Generator | None" = None,
        *,
        W: Operator | None = None,
    ) -> pd.DataFrame:
        r"""
        Return `n` draws from the posterior, one column each.

        The distribution of any function of the state, which :meth:`cov` and
        :meth:`uncertainty` give only for a linear one and only as a spread:
        the probability that a total is above a value or below zero, a
        quantile of a ratio of two totals, the share of cells below zero. Take
        the function of each draw, and read the answer off the draws.

        The analytic method draws exactly, by perturbing the prior and the
        data and solving. With :math:`x_p \sim N(x_0, S_0)` and :math:`e \sim
        N(0, S_z/\gamma)`, the solution of the perturbed problem is

        .. math::

            x_s = x_p + K (z - c - e - H x_p) = \hat{x} + (x_p - x_0) - K \big(H (x_p - x_0) + e\big),

        and :math:`x_s \sim N(\hat{x}, \hat{S})` exactly: the posterior, for
        any :math:`\gamma`. The cost is one product with each covariance's
        root and one with the factors for all `n` draws at once (the gain
        :math:`K` is never formed, and nothing ``n x n``), and the draws are
        held as a ``state x n`` array: for many draws of a large state, ask
        for them in blocks (each call with its own generator, or the same one
        passed along).

        The ``"bounded"`` method draws its posterior, the Gaussian truncated
        to the bounds, by exact Hamiltonian Monte Carlo (Pakman and Paninski,
        2014): each call runs a chain from the mode, leaves out its first
        ``burn_in`` draws, and returns the next `n`. Each draw starts from the
        last, so the draws are correlated (in the tests, about one effective
        draw per draw for most elements, one per four or five for an element
        whose analytic estimate is far outside its bound). A call costs a
        reflection off a bound, an ``O(n)`` step, for each bound a path
        reaches: about as many per draw as there are elements near a bound.

        The ``"gauss_newton"`` method draws the posterior of its forward
        model, :math:`p(x \mid z) \propto e^{-\chi^2(x)/2}`, by Hamiltonian
        Monte Carlo with the Gaussian at the mode as its mass and the exactly
        solved part of its paths, and a Metropolis test: each call continues
        the chain from where the solve's burn-in ended, at its tuned step
        (``GaussNewtonOptions``), and returns the next `n` draws. A draw of
        the state (log scale factors), carried through the model or its
        transform, is a draw of the result: a total of fluxes, its interval.
        The draws are correlated (an autocorrelation time of 1 to 3 draws for
        most elements in a Salt Lake Valley-shaped test; hundreds along a
        curve the data pin, which the totals they constrain do not share). A
        draw costs a few evaluations of the model and of the product of its
        Jacobian's transpose (one for a linear model), each with a product of
        the Gaussian at the mode's covariance.

        The ``"ensemble"`` method's draws are its updated members: `n` of
        them, chosen without replacement with `rng`, in the members' order.
        It has no more, and raises for `n` above their number.

        Parameters
        ----------
        n : int
            The number of draws, a positive integer.
        rng : int, numpy.random.Generator or None, default None
            A seed, or the generator to draw from, as :meth:`trials` takes it
            (used as given, so one generator passed to every call gives
            independent draws; None draws from fresh entropy). The prior's
            standard normals are drawn first, as a ``columns x n`` array, then
            the model-data mismatch's, so a seed gives the same draws for the
            same problem and gamma (to roundoff, whichever form the method factored in).
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it: one row per total, one
            column per state element. The draws are then those of
            :math:`W x`: the same state draws, aggregated, so ``sample(n,
            rng=0, W=W)`` is ``W @ sample(n, rng=0)``.

        Returns
        -------
        pandas.DataFrame
            One row per state element, labeled as the :attr:`estimate` is (or
            one per row of `W`), and one column per draw, a ``RangeIndex``
            named ``draw`` from 0 to ``n - 1``. Its ``groupby`` over the rows
            totals the draws with pandas
            (``draws.groupby(level="time").sum()``).

        Raises
        ------
        TypeError
            If `W` is not an Operator; or the method cannot draw from its
            posterior (its solution has no ``sample``).
        ValueError
            If `n` is not a positive integer (or, for the ensemble method, is
            above its number of members), or the columns of `W` do not have
            exactly the state's labels.

        See Also
        --------
        cov : The posterior covariance, of the state or of aggregates.
        uncertainty : The standard deviation of aggregates, before and after.
        trials : The estimate's error over many sets of data.
        fips.Covariance.sample : One draw from a covariance, as the prior and the mismatch are drawn here.

        Examples
        --------
        The total of three cells, and how likely it is to be below 3 (its
        estimate is 2.35, its posterior standard deviation 0.65):

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> cells = pd.Index(["a", "b", "c"], name="cell")
        >>> prior = fips.Vector.from_series(pd.Series(1.0, index=cells), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0], index=pd.Index(["o1", "o2"], name="obs_id")),
        ...     block="co2",
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(np.sqrt(0.1)),
        ...     forward=fips.Operator(
        ...         np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), obs.space, prior.space
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> total = fips.Operator(
        ...     np.ones((1, 3)),
        ...     fips.Space({"total": pd.Index(["all"], name="region")}),
        ...     prior.space,
        ... )
        >>> draws = post.sample(5000, rng=0, W=total)
        >>> draws.shape
        (1, 5000)
        >>> draws.mean(axis=1).round().tolist()  # near the estimate's total, 2.35
        [2.0]
        >>> round(float((draws.iloc[0] < 3.0).mean()), 1)  # exactly 0.84 for a Gaussian
        0.8
        """
        sample = self._optional("sample", "draws from the posterior")
        if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
            raise ValueError(
                f"n is the number of draws, a positive integer, got {n!r}."
            )
        form = None if W is None else self._aggregation(W)
        draws = sample(int(n), np.random.default_rng(rng))
        if form is None:
            values, index = draws, self._problem.prior.index
        else:
            assert W is not None  # form is None exactly when W is
            values, index = np.asarray(form @ draws), W.rows.index
        return pd.DataFrame(
            values, index=index, columns=pd.RangeIndex(int(n), name="draw")
        )

    def data_worth(self, by: Any, W: Operator) -> pd.DataFrame:
        r"""
        Return what each group of observations adds to the certainty of each aggregate, and its share of DOFS.

        For each group (a site, a platform, a month), the posterior standard
        deviation of each aggregate without the group's observations and with
        them (this posterior), without solving again. Removing the
        observations :math:`g` changes the posterior covariance by a rank
        :math:`m_g` update,

        .. math::

            \hat{S}_{-g} = \hat{S} + K_g P_{gg}^{-1} K_g^T, \qquad
            P = (H S_0 H^T + S_z/\gamma)^{-1},

        with :math:`K_g` the gain's columns for the group and :math:`P_{gg}`
        its block of :math:`P` (the block inverse of the innovation
        covariance). It is exact whatever the model-data mismatch, also when
        it correlates the errors of one group with another's: it is the
        posterior of the problem without the group's rows and columns. For
        :math:`W`, :math:`W K_g` is a part of :math:`W K` (``k x m``), so
        nothing ``n x n`` is formed.

        A group's share of DOFS is the sum of :math:`(H K)_{ii}` over its
        observations (the observation self-sensitivity, Cardinali et al.,
        2004); the shares of groups that hold every observation once sum to
        :attr:`dofs`.

        What a group is worth by itself is the other reading: the aggregate's
        standard deviation with only the group's observations, on the prior,
        from a rank :math:`m_g` update of :math:`S_0`,

        .. math::

            S_{g} = S_0 - S_0 H_g^T F_g^{-1} H_g S_0, \qquad
            F_g = H_g S_0 H_g^T + R_{gg} / \gamma,

        with :math:`H_g` the group's rows of :math:`H` and :math:`R_{gg}` its own
        block of the model-data mismatch. It is not the difference of the other
        columns: two groups that see the same part of the state each take much
        off the prior alone and little on top of one another, and the two
        readings agree only for a single group. Because it uses :math:`R_{gg}`
        only, it ignores any correlation between the group's errors and the
        other groups', which ``std_without`` and ``std_with`` keep: a problem
        whose mismatch correlates sites gives each site, alone, the benefit of
        independent errors from the rest.

        Parameters
        ----------
        by : str, list or pandas.Grouper
            The levels of the observations' labels to group by, as
            :meth:`desroziers` and :meth:`pandas.DataFrame.groupby` take them:
            ``"site"``, ``"block"`` (a platform per block), ``["site",
            pd.Grouper(level="obs_time", freq="MS")]``. Observations whose
            block lacks a level are in no group of it (they are never
            removed).
        W : Operator
            An aggregation, as :meth:`cov` takes it: one row per aggregate (a
            regional annual total, a trend as ``C @ W``). For every state
            element, the identity on the state's space.

        Returns
        -------
        pandas.DataFrame
            One row per group and aggregate, indexed by the group's levels
            and then the aggregation's rows (a group level with the name of
            one of those, ``block``, is called ``obs_block``): ``n_obs`` (the
            group's observations), ``dofs`` (its share of DOFS), ``std_without``
            (the aggregate's posterior standard deviation without the group),
            ``std_with`` (with it: :meth:`uncertainty`'s ``posterior_std``) and
            ``reduction``, :math:`1 - \sigma_{with} / \sigma_{without}`, the
            fraction the group takes off on top of the other observations;
            ``std_alone`` (the standard deviation with only the group's
            observations, on the prior) and ``reduction_alone``, :math:`1 -
            \sigma_{alone} / \sigma_{prior}`, the fraction the group takes off
            the prior by itself (``prior_std`` in :meth:`uncertainty`).
            A variance below zero by more than roundoff, which only a
            covariance that is not positive semidefinite can give, makes its
            standard deviation and the reduction NaN, with a logged warning
            (as in :meth:`uncertainty`), and so does one in the alone column or
            in the prior's.

        Raises
        ------
        TypeError
            If `W` is not an Operator, or the method cannot give the blocks of
            :math:`P` and the self-sensitivity (its solution lacks
            ``innovation_precision`` or ``self_sensitivity``).
        ValueError
            If the columns of `W` do not have exactly the state's labels, or
            `by` puts no observation in a group.

        See Also
        --------
        uncertainty : The prior and posterior uncertainty of each aggregate.
        candidate_worth : What observations not in the problem would add.
        dofs : The degrees of freedom for signal, the sum of the shares.

        References
        ----------
        Cardinali, C., Pezzulli, S., & Andersson, E. (2004). Influence-matrix
        diagnostic of a data assimilation system. Q.J.R. Meteorol. Soc.,
        130, 2767-2786.

        Examples
        --------
        One flux observed at two sites, three times at A and once at B; the
        flux's own row (``W`` the identity):

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = fips.Space({"flux": pd.Index(["a"], name="cell")})
        >>> labels = pd.MultiIndex.from_tuples(
        ...     [("A", 0), ("A", 1), ("A", 2), ("B", 0)], names=["site", "hour"]
        ... )
        >>> observed = fips.Space({"co2": labels})
        >>> problem = fips.Problem(
        ...     prior=fips.Vector(np.zeros(1), state),
        ...     prior_cov=fips.Covariance.diagonal(1.0, state),
        ...     obs=fips.Vector(np.ones(4), observed),
        ...     obs_cov=fips.Covariance.diagonal(1.0, observed),
        ...     forward=fips.Operator(np.ones((4, 1)), observed, state),
        ... )
        >>> post = fips.solve(problem)
        >>> W = fips.Operator(np.eye(1), state, state)
        >>> table = post.data_worth(by="site", W=W).droplevel("block").reset_index()
        >>> table.iloc[:, :7].round(3)
          site cell  n_obs  dofs  std_without  std_with  reduction
        0    A    a      3   0.6        0.707     0.447      0.368
        1    B    a      1   0.2        0.500     0.447      0.106

        Alone on the prior, site A (three observations) leaves the flux at a
        standard deviation of 0.5 and site B (one) at 0.707, taking 0.5 and
        0.293 off the prior's 1; on top of the other site, B adds only 0.106 and
        A 0.368:

        >>> table[["site", "std_alone", "reduction_alone"]].round(3)
          site  std_alone  reduction_alone
        0    A      0.500            0.500
        1    B      0.707            0.293
        """
        precision = self._optional(
            "innovation_precision", "the data worth of groups of observations"
        )
        sensitivity = self._optional(
            "self_sensitivity", "the data worth of groups of observations"
        )
        groups, positions = _obs_groups(self._problem.obs.index, by)
        W_form = self._aggregation(W)
        WK = self._solution.gain(W_form)  # k x m
        var_with = self.cov(W).variances.values
        shares = sensitivity()
        # the prior side of the alone column: W S_0 and the group's rows of H
        prior_cov, H = self._problem.prior_cov.data, self._linear._operator.data
        R = self._problem.obs_cov.data
        prior = self._problem.prior_cov.aggregate(W).variances.values
        WS_0 = W_form @ prior_cov
        columns = np.arange(len(self._problem.prior))
        var_without, var_alone, dofs = [], [], []
        for P_gg, rows in zip(precision(positions), positions, strict=True):
            WK_g = WK[:, rows]  # k x m_g
            P_gg = 0.5 * (P_gg + P_gg.T)
            added = np.einsum("ki,ik->k", WK_g, _solve_psd(P_gg, WK_g.T))
            var_without.append(var_with + added)
            dofs.append(float(shares[rows].sum()))
            # the group alone on the prior: W S_0 H_g^T and H_g S_0 H_g^T + R_gg/gamma
            H_g = H._select(rows, columns)
            S_wg = (WS_0 @ H_g._transpose()).dense()  # k x m_g
            F = ((H_g @ prior_cov) @ H_g._transpose()).dense()
            F = F + R._select(rows, rows).dense() / self._gamma
            Y = _solve_psd(0.5 * (F + F.T), S_wg.T)
            var_alone.append(prior - np.einsum("ki,ik->k", S_wg, Y))
        return _worth_table(
            groups,
            positions,
            np.array(dofs),
            W.rows.index,
            np.array(var_without),
            np.tile(var_with, (len(positions), 1)),
            self._prior_scale(W_form),
            prior,
            np.array(var_alone),
        )

    def candidate_worth(
        self,
        forward: Operator,
        obs_cov: "Covariance | CovarianceModel",
        by: Any,
        W: Operator,
    ) -> pd.DataFrame:
        r"""
        Return what each group of candidate observations, not in the problem, would add to the certainty of each aggregate.

        Candidate observations are rows of a forward operator that the
        problem does not have (a new site, another platform, a season not yet
        measured) and the covariance of their errors; their values are not
        needed. For each group of candidates, the posterior standard deviation
        of each aggregate without the group (this posterior) and with it, from
        a rank :math:`m_c` update of :math:`\hat{S}`, without solving again:

        .. math::

            \hat{S}_{+c} = \hat{S} - \hat{S} H_c^T F^{-1} H_c \hat{S}, \qquad
            F = H_c \hat{S} H_c^T + R_c / \gamma,

        with :math:`H_c` the group's rows of `forward` and :math:`R_c` its
        block of `obs_cov`. The errors of the candidates are taken to be
        independent of the problem's observations, and are divided by the
        regularization factor :math:`\gamma` as the problem's are, so the
        result is the posterior of the problem solved again with the group's
        observations added. For :math:`W`, the three blocks
        :math:`W \hat{S} W^T`, :math:`W \hat{S} H_c^T` and
        :math:`H_c \hat{S} H_c^T` are one :meth:`cov` of :math:`W` and
        :math:`H_c` stacked: nothing ``n x n`` is formed. Each group is added
        on its own, to the problem as it is.

        Parameters
        ----------
        forward : Operator
            The candidates' forward operator, from the state (its columns, the
            state's labels in any order) to the candidate observations (its
            rows, labeled as observations are, on the levels `by` groups).
        obs_cov : Covariance or CovarianceModel
            The covariance of the candidates' errors, on the rows of
            `forward`; a model (``fips.Independent().with_std(5.0)``) is built
            there. It must be positive-definite.
        by : str, list or pandas.Grouper
            The levels of the candidates' labels to group by, as
            :meth:`data_worth` takes them (``"site"``).
        W : Operator
            An aggregation, as :meth:`cov` takes it: one row per aggregate.

        Returns
        -------
        pandas.DataFrame
            As :meth:`data_worth`: one row per group and aggregate, with
            ``n_obs``, ``dofs`` (the group's share of DOFS in the problem with
            it added, :math:`m_c - \operatorname{tr}(R_c F^{-1}) / \gamma`),
            ``std_without`` (this posterior's), ``std_with`` (with the group
            added) and ``reduction``, :math:`1 - \sigma_{with} /
            \sigma_{without}`. A variance below zero by more than roundoff,
            which only a covariance that is not positive semidefinite can
            give, makes its standard deviation and the reduction NaN, with a
            logged warning (as in :meth:`uncertainty`).

        Raises
        ------
        TypeError
            If the method's estimate is not linear in the observations (its
            solution does not set ``estimate_is_linear``), `forward` or `W`
            is not an Operator, or `obs_cov` is not a covariance.
        ValueError
            If the columns of `forward` or `W` do not have exactly the state's
            labels, `obs_cov` is not on the rows of `forward`, or `by` puts
            no candidate in a group.

        See Also
        --------
        data_worth : What the groups of observations in the problem add.
        uncertainty : The prior and posterior uncertainty of each aggregate.

        Notes
        -----
        A candidate group is always added on top of the problem's observations,
        so there are no ``std_alone`` and ``reduction_alone`` columns, which
        :meth:`data_worth` has for the groups in the problem.

        Examples
        --------
        One flux observed four times at site A; would a candidate site B,
        three observations with errors of the same size, help more than one
        more observation at A?

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = fips.Space({"flux": pd.Index(["a"], name="cell")})
        >>> observed = fips.Space(
        ...     {
        ...         "co2": pd.MultiIndex.from_product(
        ...             [["A"], range(4)], names=["site", "hour"]
        ...         )
        ...     }
        ... )
        >>> problem = fips.Problem(
        ...     prior=fips.Vector(np.zeros(1), state),
        ...     prior_cov=fips.Covariance.diagonal(1.0, state),
        ...     obs=fips.Vector(np.ones(4), observed),
        ...     obs_cov=fips.Covariance.diagonal(1.0, observed),
        ...     forward=fips.Operator(np.ones((4, 1)), observed, state),
        ... )
        >>> post = fips.solve(problem)
        >>> candidates = fips.Space(
        ...     {
        ...         "co2": pd.MultiIndex.from_tuples(
        ...             [("A", 4), ("B", 0), ("B", 1), ("B", 2)], names=["site", "hour"]
        ...         )
        ...     }
        ... )
        >>> H_new = fips.Operator(np.ones((4, 1)), candidates, state)
        >>> W = fips.Operator(np.eye(1), state, state)
        >>> table = post.candidate_worth(
        ...     H_new, fips.Independent().with_std(1.0), by="site", W=W
        ... )
        >>> table.droplevel("block").round(3).reset_index()
          site cell  n_obs   dofs  std_without  std_with  reduction
        0    A    a      1  0.167        0.447     0.408      0.087
        1    B    a      3  0.375        0.447     0.354      0.209
        """
        if not getattr(self._solution, "estimate_is_linear", False):
            raise TypeError(
                "The method that solved this problem does not give an estimate "
                f"linear in the observations ({type(self._solution).__name__} does "
                "not set estimate_is_linear), so adding observations is not a "
                "rank update of its covariance: solve the problem with them."
            )
        if not isinstance(forward, Operator):
            raise TypeError(
                f"forward must be an Operator, got {type(forward).__name__}."
            )
        state = self._problem.prior.space
        candidates = forward.rows
        H_c = forward._on(candidates, state, "forward", ("the candidates", "the state"))
        R_c = _covariance(obs_cov, candidates, "obs_cov", "the candidates").data
        groups, positions = _obs_groups(candidates.index, by)
        W_form = self._aggregation(W)
        k = W_form.shape[0]
        columns = np.arange(len(state))
        var_without, var_with, dofs = [], [], []
        for rows in positions:
            # W S^ W^T, W S^ H_c^T and H_c S^ H_c^T, from one aggregated covariance
            S = self._solution.cov(_stacked(W_form, H_c._select(rows, columns)))
            S_ww, S_wc, S_cc = S[:k, :k], S[:k, k:], S[k:, k:]
            R_g = R_c._select(rows, rows).dense() / self._gamma
            F = S_cc + R_g
            Y = _solve_psd(0.5 * (F + F.T), np.hstack([S_wc.T, R_g]))
            var_without.append(np.diagonal(S_ww))
            var_with.append(np.diagonal(S_ww) - np.einsum("ki,ik->k", S_wc, Y[:, :k]))
            dofs.append(len(rows) - float(np.trace(Y[:, k:])))
        return _worth_table(
            groups,
            positions,
            np.array(dofs),
            W.rows.index,
            np.array(var_without),
            np.array(var_with),
            self._prior_scale(W_form),
        )

    @cached_property
    def kalman_gain(self) -> Operator:
        r"""
        The gain :math:`K`, state by observations, dense.

        :math:`\hat{x} = x_0 + K (z - H x_0 - c)`.
        """
        return Operator(
            _adopt(self._solution.gain()),
            self._problem.prior.space,
            self._problem.obs.space,
            name="kalman_gain",
        )

    @cached_property
    def _averaging_kernel(self) -> Operator:
        """The full averaging kernel, formed once."""
        A = self.kalman_gain.dense() @ self._linear._operator.data
        space = self._problem.prior.space
        return Operator(_adopt(A), space, space, name="averaging_kernel")

    def averaging_kernel(self, W: Operator | None = None) -> Operator:
        r"""
        Return the averaging kernel, of the state or of aggregates of it.

        Without `W`, :math:`A = K H`, state by state (dense, ``n x n``; formed
        once, the same object each time). With an aggregation :math:`W`,
        :math:`W A W^*`, where :math:`W^* = W^T (W W^T)^{-1}` is the
        pseudo-inverse of :math:`W`: it spreads a change of the aggregates back
        over the state elements (for a sum, evenly over the elements summed),
        so :math:`W A W^*` is how the aggregated estimate responds to the
        aggregated truth. This is the form IMI uses; :math:`W A W^T` is not an
        averaging kernel. It is computed without the ``n x n`` kernel.

        Parameters
        ----------
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it. Its rows must be linearly
            independent.

        Returns
        -------
        Operator
            ``n x n`` on the state's space, or ``k x k`` on the aggregation's
            rows.

        Raises
        ------
        ValueError
            If the columns of `W` do not have exactly the state's labels, or
            its rows are linearly dependent.

        See Also
        --------
        dofs : The trace of the full averaging kernel.
        cov : The posterior covariance, of the state or of aggregates.
        """
        if W is None:
            return self._averaging_kernel
        W_form = self._aggregation(W)
        WK = self._solution.gain(W_form)  # k x m
        HWt = np.asarray(self._linear._operator.data @ W_form.T)  # m x k
        values = _times_pseudo_inverse(WK @ HWt, W_form)
        return Operator(_adopt(values), W.rows, W.rows, name="averaging_kernel")

    # -- the observations ----------------------------------------------------

    @cached_property
    def modeled_obs(self) -> Vector:
        r"""
        The observations the estimate models, :math:`H \hat{x} + c`.

        ``problem.predict(post.estimate)``, kept. The observations another
        state models are :meth:`Problem.predict <fips.Problem.predict>` of it
        (the prior's: ``problem.predict(problem.prior)``).

        See Also
        --------
        fips.Problem.predict : The observations any state models.
        """
        values = self._problem.predict(self.estimate).values
        return Vector(values, self._problem.obs.space, name="modeled_obs")

    @cached_property
    def _HS_0H(self) -> LinearOperator:
        r""":math:`H S_0 H^T`, in the cheapest form of its factors."""
        H = self._linear._operator.data
        return (H @ self._problem.prior_cov.data) @ H._transpose()

    @cached_property
    def innovation_cov(self) -> Covariance:
        r"""
        The covariance of the innovation, :math:`H S_0 H^T + S_z / \gamma`, dense.

        The innovation is :math:`z - H x_0 - c`, the observations minus what
        the prior models; this is its covariance when the prior error and the
        model-data mismatch are right, and the ``m x m`` matrix the solve
        factors. Kalman filtering calls it the innovation covariance; Rodgers
        (2000) writes it :math:`S_y`. It includes the regularization factor:
        the mismatch is divided by :math:`\gamma`. Its diagonal, square-rooted,
        is the spread expected for each observation's innovation, of which
        :math:`H S_0 H^T` is the prior's share.

        See Also
        --------
        desroziers : The observation error variances the residuals diagnose.
        """
        G = self._HS_0H + self._problem.obs_cov.data / self._gamma
        return Covariance._computed(
            _adopt(G.dense()), self._problem.obs.space, name="innovation_cov"
        )

    def desroziers(self, by: Any = None) -> pd.DataFrame:
        r"""
        Compare the observation error variances Desroziers et al. (2005) diagnose with those the solve used.

        With :math:`d^b = z - H x_0 - c` (the innovation) and
        :math:`d^a = z - H \hat{x} - c` (the residual), the expected value of
        :math:`d^a_i d^b_i` is the variance of the error of observation
        :math:`i` when the covariances of the problem are right. One
        observation's product is noisy; its mean over a group of observations
        (a site, a month) is the diagnostic.

        Parameters
        ----------
        by : str, list or pandas.Grouper, optional
            The levels of the observations' labels to group by, as
            :meth:`pandas.DataFrame.groupby` takes them: ``"site"``,
            ``["site", "block"]``, or ``pd.Grouper(level="obs_time",
            freq="MS")``. Observations whose block lacks a level are left out
            of a grouping by it. By default, one row per observation.

        Returns
        -------
        pandas.DataFrame
            ``diagnosed`` (:math:`d^a_i d^b_i`, or its mean over a group),
            ``specified`` (the diagonal of :math:`S_z / \gamma`, the
            variances the solve used, or their mean), and ``ratio``
            (diagnosed over specified; near 1 when they agree). Indexed by
            the observations' labels, or by the groups.

        References
        ----------
        Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). Diagnosis
        of observation, background and analysis-error statistics in
        observation space. Q.J.R. Meteorol. Soc., 131, 3385-3396.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> sites = pd.MultiIndex.from_product(
        ...     [["A", "B"], [0, 1, 2]], names=["site", "hour"]
        ... )
        >>> cell = pd.Index(["c"], name="cell")
        >>> prior = fips.Vector.from_series(pd.Series(0.0, index=cell), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0, 0.5, 3.0, 2.5, 1.0], index=sites), block="co2"
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(1.0),
        ...     forward=fips.Operator(np.ones((6, 1)), obs.space, prior.space),
        ... )
        >>> fips.solve(problem).desroziers(by="site").round(3).reset_index()
          site  diagnosed  specified  ratio
        0    A      0.083        1.0  0.083
        1    B      2.321        1.0  2.321
        """
        problem = self._problem
        before = problem.obs.values - problem.predict(problem.prior).values
        after = problem.obs.values - self.modeled_obs.values
        frame = pd.DataFrame(
            {
                "diagnosed": after * before,
                "specified": problem.obs_cov.data.diagonal() / self._gamma,
            },
            index=problem.obs.index,
        )
        if by is not None:
            frame = frame.groupby(by).mean()
        frame["ratio"] = frame["diagnosed"] / frame["specified"]
        return frame

    def leverage(self, x: Vector) -> Operator:
        r"""
        Return the leverage matrix of the state `x`: which observations weigh most on it.

        .. math::
            L = h \left(h^T G^{-1} h\right)^{-1} h^T G^{-1},
            \qquad h = H x + c, \quad G = H S_0 H^T + S_z / \gamma

        This is the generalized-least-squares hat matrix of the one regressor
        :math:`h`, the observations that `x` models (:meth:`~fips.Problem.predict`,
        which includes the offset :math:`c`): the projection onto :math:`h` in
        the metric :math:`G^{-1}`. It is idempotent and its
        trace is 1, so its diagonal shares one unit of leverage among the
        observations; with correlated errors an entry can be negative.

        Parameters
        ----------
        x : Vector
            A state, on the prior's labels (in any order), such as
            :attr:`estimate` or the problem's prior.

        Returns
        -------
        Operator
            ``m x m``, dense, on the observations' space.

        Raises
        ------
        ValueError
            If `x` does not have exactly the state's labels.
        """
        problem = self._problem
        Hx = problem.predict(x).values[:, None]
        G = self.innovation_cov.dense()
        term = _solve_psd(G, Hx)
        L = Hx @ inv(Hx.T @ term) @ term.T
        space = problem.obs.space
        return Operator(_adopt(L), space, space, name="leverage")

    # -- scalars -------------------------------------------------------------

    @property
    def dofs(self) -> float:
        r"""
        Degrees of freedom for signal, the trace of :meth:`averaging_kernel`.

        The analytic method reads it from its factors without the kernel:
        :math:`\sum_k \gamma\lambda_k / (\gamma\lambda_k + 1)` over the
        eigenvalues :math:`\lambda_k` of its factorization (the same in
        both of its forms), or, at the gamma of :func:`fips.solve`, from the
        Cholesky factor it solved with.
        """
        return self._solution.dofs()

    @cached_property
    def dofs_per_element(self) -> Vector:
        r"""
        Each state element's share of :attr:`dofs`: the diagonal of :meth:`averaging_kernel`, on the prior's space.

        :math:`A_{jj}`, how much the estimate of element :math:`j` responds to
        its own true value: near 1 for an element the observations constrain,
        near 0 for one left at its prior. The values sum to :attr:`dofs`, so
        a group's share (a region, a block, a month) is a sum over its
        labels. The analytic method computes the diagonal from its factors,
        without forming the gain or the ``n x n`` kernel.

        A diagonal entry is not the information on that element alone: the
        rest of its row of :math:`A` says how much of the estimate comes
        from the true values of the other elements.

        See Also
        --------
        dofs : Their sum.
        averaging_kernel : The full kernel :math:`A = K H`.
        information_content : The information the observations give, in nats.

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
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> post.dofs_per_element.to_series().round(3)
        block  cell
        flux   a       0.938
               b       0.938
        Name: dofs_per_element, dtype: float64
        >>> round(post.dofs, 3)
        1.877
        """
        values = self._solution.dofs_per_element()
        return Vector(values, self._problem.prior.space, name="dofs_per_element")

    @property
    def information_content(self) -> float:
        r"""
        The Shannon information content of the observations about the state, in nats.

        .. math::
            H = -\tfrac{1}{2} \ln \lvert I - A \rvert
              = \tfrac{1}{2} \ln \frac{\lvert S_0 \rvert}{\lvert \hat{S} \rvert}
              = \tfrac{1}{2} \sum_k \ln(1 + s_k^2)

        with :math:`A` the averaging kernel and :math:`s_k^2` the eigenvalues
        of :math:`\gamma R_0^T H^T S_z^{-1} H R_0` (:math:`S_0 = R_0 R_0^T`),
        the observations' information on the whitened state;
        :attr:`dofs` is :math:`\sum_k s_k^2 / (1 + s_k^2)`. It is how much
        the observations reduce the entropy of the state, from the prior to
        the posterior: the log of how many times smaller the volume of
        uncertainty is. Divide by :math:`\ln 2` for bits. A direction the
        observations constrain counts at most 1 in :attr:`dofs`, but its
        information keeps growing with the precision of the observations.

        The analytic method reads the log-determinant from the factor it
        solved with (the Cholesky factor at the gamma of :func:`fips.solve`,
        or the eigenvalues after :meth:`with_gamma`), so it costs nothing
        more than the solve.

        References
        ----------
        Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding:
        Theory and Practice*. World Scientific.

        See Also
        --------
        dofs : The degrees of freedom for signal.
        dofs_per_element : DOFS, element by element.

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
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> round(post.information_content, 3)
        2.916
        >>> round(float(post.information_content / np.log(2)), 3)  # in bits
        4.207
        """
        return self._solution.information()

    @property
    def chi2(self) -> float:
        r"""
        The chi-squared of the solution: the minimum of the cost, the data misfit times gamma plus the state misfit.

        .. math::
            \chi^2 = (z - H\hat{x} - c)^T (S_z/\gamma)^{-1} (z - H\hat{x} - c)
                + (\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0)
            = \gamma \, \mathrm{data\_misfit} + \mathrm{state\_misfit}

        which equals :math:`d^T (H S_0 H^T + S_z/\gamma)^{-1} d` with
        :math:`d = z - H x_0 - c`; the analytic method reads it from its
        factors.

        See Also
        --------
        reduced_chi2 : It, divided by the number of observations.
        chi2_tests : Its degrees of freedom and p-value, and Rodgers' other three tests.
        data_misfit : Its first term, without gamma.
        state_misfit : Its second term.
        """
        return self._solution.chi2()

    @cached_property
    def _misfits(self) -> tuple[float, float]:
        """The data and the state misfits, computed together once."""
        return self._solution.misfits()

    @property
    def data_misfit(self) -> float:
        r"""
        How far the modeled observations are from the observations, weighed by the model-data mismatch: :math:`r^T S_z^{-1} r`.

        :math:`r = z - H \hat{x} - c` is the residual and :math:`S_z` the
        model-data mismatch as the problem gives it, not divided by
        :math:`\gamma`, so that it can be compared across a sweep over
        gamma: with :attr:`state_misfit` it is the L-curve (Hansen, 1992), a
        point per gamma, whose corner is where fitting the observations more
        closely starts to cost much more departure from the prior.
        :attr:`chi2` is :math:`\gamma` times this plus :attr:`state_misfit`.

        References
        ----------
        Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means
        of the L-curve. *SIAM Review*, 34(4), 561-580.
        https://doi.org/10.1137/1034115

        See Also
        --------
        state_misfit : The other term, the departure from the prior.
        chi2 : The two together, at the solve's gamma.
        with_gamma : The posterior at another gamma, for a sweep.

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
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> round(post.data_misfit, 3)
        0.182
        >>> sweep = [post.with_gamma(g) for g in (0.1, 1.0, 10.0)]
        >>> [
        ...     round(p.data_misfit, 3) for p in sweep
        ... ]  # closer to the data as gamma grows
        [9.688, 0.182, 0.002]
        """
        return self._misfits[0]

    @property
    def state_misfit(self) -> float:
        r"""
        How far the estimate is from the prior, weighed by the prior error covariance: :math:`(\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0)`.

        The analytic method computes it without inverting :math:`S_0`, so it
        holds for a semi-definite prior error covariance too (the increment
        lies in its range). With :attr:`data_misfit` it is the L-curve: plot
        one against the other over a sweep of gamma. :attr:`chi2` is
        :math:`\gamma` times :attr:`data_misfit` plus this.

        See Also
        --------
        data_misfit : The other term, the misfit to the observations.
        chi2 : The two together, at the solve's gamma.
        with_gamma : The posterior at another gamma, for a sweep.

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
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> round(post.state_misfit, 3)
        4.628
        >>> sweep = [post.with_gamma(g) for g in (0.1, 1.0, 10.0)]
        >>> [
        ...     round(p.state_misfit, 3) for p in sweep
        ... ]  # farther from the prior as gamma grows
        [2.656, 4.628, 4.96]
        """
        return self._misfits[1]

    @property
    def reduced_chi2(self) -> float:
        r"""
        :attr:`chi2` divided by the number of observations; near 1 when the covariances are right.

        :attr:`chi2` is :math:`d^T G^{-1} d`, with :math:`d = z - H x_0 - c`
        the innovation and :math:`G = H S_0 H^T + S_z / \gamma` its
        covariance (:attr:`innovation_cov`). When the covariances are right,
        :math:`d \sim N(0, G)`, so :attr:`chi2` follows a chi-squared
        distribution with :math:`m` degrees of freedom: its mean is :math:`m`
        and its standard deviation :math:`\sqrt{2m}`, so this is 1 within
        about :math:`\sqrt{2/m}`. It is the first test of
        :meth:`chi2_tests`, which gives its p-value (Rodgers, 2000, chapter
        12). Used, for example, by Kunik et al. (2019),
        https://doi.org/10.1525/elementa.375.

        References
        ----------
        Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding:
        Theory and Practice*. World Scientific.

        See Also
        --------
        chi2 : The chi-squared itself.
        chi2_tests : The test, with its p-value, and Rodgers' other three tests.
        """
        return self.chi2 / len(self._problem.obs)

    @property
    def log_evidence(self) -> float:
        r"""
        The log of the evidence: the probability density of the observations given the problem's settings, with the state integrated out.

        With the state integrated out, the observations of a linear problem
        are Gaussian, :math:`z \sim N(H x_0 + c, G)` with :math:`G = H S_0 H^T
        + S_z/\gamma` (:attr:`innovation_cov`), so

        .. math::
            \ln p(z) = -\tfrac{1}{2} \left( d^T G^{-1} d + \ln \lvert G \rvert
                + m \ln 2\pi \right)
            = -\tfrac{1}{2} \left( \chi^2 + 2 \mathcal{H} + \ln \lvert S_z \rvert
                - m \ln \gamma + m \ln 2\pi \right)

        with :math:`d = z - H x_0 - c`, :math:`\chi^2` the :attr:`chi2`,
        :math:`\mathcal{H}` the :attr:`information_content` and :math:`m` the number of
        observations: :math:`\lvert G \rvert = \lvert S_z/\gamma \rvert \,
        \lvert I + \gamma S_z^{-1} H S_0 H^T \rvert`, and the log of the
        second factor is twice the information content. Every term but
        :math:`\ln \lvert S_z \rvert` is one the solve computed, and that one
        comes from the form's own factors (a Cholesky factorization of a dense
        :math:`S_z`, the log of a diagonal one's entries), once per problem:
        the posteriors of a sweep over gamma share it, and :meth:`with_gamma`
        and :meth:`with_data` give the evidence without a new solve.
        It holds for a singular :math:`S_0`; :math:`S_z` must be
        positive-definite.

        It is the likelihood of the settings the problem was built with, the
        prior error covariance, the model-data mismatch and gamma: the
        settings that maximize it are their maximum-likelihood estimates
        (Michalak et al., 2005). Write a function that builds the problem from
        the settings and maximize this over them, with
        :func:`scipy.optimize.minimize` on its negative
        (:doc:`/user_guide/settings`). The difference between two settings'
        values is the log of their Bayes factor. It is a density of the
        observations, so it depends on their units: compare only problems of
        the same observations.

        For the ``"gauss_newton"`` method, it is Laplace's approximation: the
        same formula with the chi-squared of the model at the mode
        (:attr:`chi2`) and the information content of the Gaussian there (the
        Gauss-Newton curvature of the cost, at the last linearization, one
        step from the mode). It is exact for a linear forward model; for a
        model that bends, it is the evidence of the Gaussian at the mode, and
        leaves out how the posterior departs from it (the skew its draws show,
        :meth:`sample`): for a lognormal flux in a few cells with a log-std of
        0.8, it was within about 0.1 of the evidence by importance sampling. The
        ``"bounded"`` method gives it only with no finite bound, when its
        posterior is the analytic one.

        Raises
        ------
        TypeError
            If the method's posterior is not the Gaussian at its estimate (a
            bounded posterior with a finite bound).
        numpy.linalg.LinAlgError
            If :math:`S_z` is not positive-definite.

        References
        ----------
        Michalak, A. M., Hirsch, A., Bruhwiler, L., Gurney, K. R., Peters, W.,
        and Tans, P. P. (2005). Maximum likelihood estimation of covariance
        parameters for Bayesian atmospheric trace gas surface flux inversions.
        *Journal of Geophysical Research*, 110, D24107.
        https://doi.org/10.1029/2005JD005970

        Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding:
        Theory and Practice*. World Scientific.

        See Also
        --------
        chi2 : The chi-squared of the innovation, its first term.
        information_content : Its second term, halved.
        with_gamma : The posterior at another gamma, whose evidence costs a few products.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from scipy.stats import multivariate_normal
        >>> import fips
        >>> cells = pd.Index(["a", "b"], name="cell")
        >>> obs_id = pd.Index(["o1", "o2", "o3"], name="obs_id")
        >>> prior = fips.Vector.from_series(pd.Series(0.0, index=cells), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0, 3.0], index=obs_id), block="co2"
        ... )
        >>> H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(np.sqrt(0.1)),
        ...     forward=fips.Operator(H, obs.space, prior.space),
        ... )
        >>> post = fips.solve(problem)
        >>> round(post.log_evidence, 4)
        -4.6236
        >>> G = H @ H.T + 0.1 * np.eye(3)  # the innovation covariance
        >>> round(float(multivariate_normal(np.zeros(3), G).logpdf(obs.values)), 4)
        -4.6236

        A sweep over gamma reads it from one eigendecomposition:

        >>> [round(post.with_gamma(g).log_evidence, 3) for g in (0.1, 1.0, 10.0)]
        [-5.609, -4.624, -3.5]
        """
        if not getattr(self._solution, "evidence_is_laplace", False):
            raise TypeError(
                "The method that solved this problem gives no log evidence "
                f"({type(self._solution).__name__}'s evidence_is_laplace is "
                "false or unset): its posterior is not the Gaussian at its "
                "estimate, so the chi-squared and the information content do not "
                "give it. With bounds on the state, the exact evidence is the "
                "Gaussian one times the probability of the bounds under the "
                "posterior over their probability under the prior, both "
                "orthant probabilities of a correlated Gaussian, which fips does "
                "not compute. The evidence of the problem without bounds is "
                "fips.solve(problem).log_evidence."
            )
        m, gamma = len(self._problem.obs), self._gamma
        return -0.5 * float(
            self.chi2
            + 2.0 * self.information_content
            + self._problem._obs_logdet
            - m * np.log(gamma)
            + m * np.log(2.0 * np.pi)
        )

    def chi2_tests(self) -> pd.DataFrame:
        r"""
        Test whether the innovation, the residual and the increment are as large as the covariances say: Rodgers' four chi-squared tests.

        Each test takes a vector that is :math:`N(0, C)` when the prior error
        and model-data mismatch covariances are right, and its chi-squared
        :math:`v^T C^{+} v` (:math:`^{+}` the pseudo-inverse), which then
        follows a chi-squared distribution whose degrees of freedom are the
        rank of :math:`C`: it averages them. With :math:`G = H S_0 H^T +
        S_z / \gamma` (:attr:`innovation_cov`) and :math:`r` the rank of
        :math:`H S_0`, the number of directions of the state the
        observations constrain at all:

        - ``innovation``: the observations against what the prior models,
          :math:`d = z - H x_0 - c`, with :math:`C = G`: :math:`m` degrees
          of freedom. Its chi-squared is :attr:`chi2`: this row is
          :attr:`chi2` and :attr:`reduced_chi2` with their p-value.
        - ``residual``: the observations against what the estimate models,
          :math:`z - H \hat{x} - c`, with :math:`C = (S_z/\gamma) G^{-1}
          (S_z/\gamma)`: :math:`m` degrees of freedom.
        - ``modeled_increment``: how far the estimate moves the modeled
          observations from the prior's, :math:`H (\hat{x} - x_0)`, with
          :math:`C = H S_0 H^T G^{-1} H S_0 H^T`: :math:`r` degrees of freedom.
        - ``increment``: the estimate against the prior,
          :math:`\hat{x} - x_0`, with :math:`C = S_0 - \hat{S} = S_0 H^T
          G^{-1} H S_0`: :math:`r` degrees of freedom.

        Rodgers (2000, chapter 12) names them the measurement against the
        prior, the retrieval against the measurement, and the retrieval
        against the prior in the measurement and in the state space, as
        pyOptimalEstimation computes them. The first two test both
        covariances at once; the last two whether the observations move the
        estimate as far from the prior as the prior error covariance allows.

        **In a linear problem the four are two pairs of the same numbers**,
        so a table with duplicate rows is expected: the residual is
        :math:`(S_z/\gamma) G^{-1} d`, so its chi-squared is :math:`d^T G^{-1}
        d`, the innovation's; and the two increments are the same
        components of the innovation, scaled, so their chi-squared is one
        sum (below). When the observations constrain as many directions as
        there are observations (:math:`r = m`, usual when :math:`n \ge m`),
        all four are the same number. They differ when the estimate is not
        linear in the observations (a nonlinear method, whose :math:`H` is
        the forward model linearized at the estimate): the table has four
        rows so that every method gives it the same shape.

        The degrees of freedom are the ranks of the four :math:`C`. :math:`G`
        and :math:`S_z` are positive definite, so the first two have rank
        :math:`m`. For a positive-definite :math:`G^{-1}`, :math:`B^T G^{-1}
        B` has the rank of :math:`B`: :math:`B = H S_0 H^T` for the third
        and :math:`B = H S_0` for the fourth, both of rank :math:`r` (with a
        positive-definite :math:`S_0`, :math:`r` is the rank of :math:`H`).
        So :math:`r \le \min(m, n)`: with fewer state elements than
        observations the increments have at most :math:`n`, and observations
        that see nothing (:math:`H S_0 = 0`) give them none.

        The analytic method computes the increments from the
        eigendecomposition of its factors: with :math:`s_k^2` its
        eigenvalues times :math:`\gamma` and :math:`e'_k` the innovation
        whitened by :math:`S_z / \gamma` and rotated onto its eigenvectors
        (variance :math:`1 + s_k^2`), the increment has the components
        :math:`s_k e'_k / (1 + s_k^2)` in the state whitened by :math:`S_0`
        and the modeled increment :math:`s_k^2 e'_k / (1 + s_k^2)` in the
        observations whitened by :math:`S_z / \gamma`; both chi-squared are
        :math:`\sum_k e_k'^2 / (1 + s_k^2)` over the :math:`r` eigenvalues
        above zero. After :func:`fips.solve` this eigendecomposes once, as
        the first :meth:`with_gamma` does, and keeps it; a Monte Carlo over
        :meth:`with_data` should start from ``post.with_gamma(gamma)``, whose
        eigendecomposition every trial shares. An eigenvalue counts when it
        is above roundoff relative to the largest, so the units of the
        problem do not change the degrees of freedom.

        Returns
        -------
        pandas.DataFrame
            One row per test (``innovation``, ``residual``,
            ``modeled_increment``, ``increment``), indexed by ``test``:
            ``chi2``, ``dof`` (the degrees of freedom), ``reduced_chi2``
            (``chi2 / dof``, near 1 when the covariances are right) and
            ``p_value``, the probability of a chi-squared at least this large
            when they are right. A small p-value says the vector is larger
            than the covariances allow (they are too small); one near 1 that it
            is smaller (they are too large). With no degrees of freedom (the
            observations constrain nothing) the last two are NaN.

        References
        ----------
        Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding:
        Theory and Practice*, chapter 12. World Scientific.

        See Also
        --------
        chi2 : The chi-squared of the innovation.
        desroziers : The observation error variances the residuals diagnose.

        Examples
        --------
        Three observations of two state elements: the innovation and the
        residual have three degrees of freedom, the two increments two. The
        problem is linear, so the rows come in pairs of the same numbers.

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> cells = pd.Index(["a", "b"], name="cell")
        >>> obs_id = pd.Index(["o1", "o2", "o3"], name="obs_id")
        >>> prior = fips.Vector.from_series(pd.Series(0.0, index=cells), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0, 4.0], index=obs_id), block="co2"
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(np.sqrt(0.1)),
        ...     forward=fips.Operator(
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> fips.solve(problem).chi2_tests().round(3).reset_index()
                        test    chi2  dof  reduced_chi2  p_value
        0         innovation  10.293    3         3.431    0.016
        1           residual  10.293    3         3.431    0.016
        2  modeled_increment   6.960    2         3.480    0.031
        3          increment   6.960    2         3.480    0.031
        """
        names = ["innovation", "residual", "modeled_increment", "increment"]
        tests = self._solution.chi2_tests()
        chi2 = np.array([tests[name][0] for name in names], dtype=float)
        dof = np.array([tests[name][1] for name in names], dtype=int)
        has_dof = dof > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            reduced = np.where(has_dof, chi2 / dof, np.nan)
        p_value = np.where(has_dof, chdtrc(np.maximum(dof, 1), chi2), np.nan)
        return pd.DataFrame(
            {"chi2": chi2, "dof": dof, "reduced_chi2": reduced, "p_value": p_value},
            index=pd.Index(names, name="test"),
        )

    @property
    def r2(self) -> float:
        r"""
        The R² of the enhancement, :math:`\operatorname{corr}(z - c, H\hat{x})^2`.

        How well the estimate fits the part of the observations the state
        explains: the observations :math:`z` less the offset :math:`c` (in a
        flux inversion, the enhancement over the background), against
        :math:`H\hat{x}`, which is :attr:`modeled_obs` less the offset.

        The offset is taken out because it is given, not fitted. One that
        varies from observation to observation, as a background does from
        day to day, would otherwise raise the correlation by how well it
        follows the observations, whatever the fit of the state. A constant
        offset, or none, changes no correlation: then this is the squared
        correlation of the observations and :attr:`modeled_obs`.

        It is a squared correlation, not :math:`1 - SS_{res} / SS_{tot}`: a
        fit that is off by a constant or a factor scores as well as an exact
        one. :attr:`rmse` measures the misfit itself.

        See Also
        --------
        modeled_obs : :math:`H\hat{x} + c`.
        rmse : The root mean square of the residuals.

        Examples
        --------
        Four days of one flux over a background that rises from day to day,
        given as the offset:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = fips.Space({"flux": pd.Index(["a"], name="cell")})
        >>> days = fips.Space({"co2": pd.RangeIndex(4, name="day")})
        >>> background = np.array([400.0, 402.0, 404.0, 406.0])
        >>> problem = fips.Problem(
        ...     prior=fips.Vector(np.zeros(1), state),
        ...     prior_cov=fips.Covariance.diagonal(1.0, state),
        ...     obs=fips.Vector(background + [1.0, 0.0, 2.0, 1.0], days),
        ...     obs_cov=fips.Covariance.diagonal(0.5, days),
        ...     forward=fips.Operator(
        ...         np.array([[1.0], [0.5], [1.5], [0.0]]), days, state
        ...     ),
        ...     offset=fips.Vector(background, days),
        ... )
        >>> post = fips.solve(problem)
        >>> round(post.r2, 3)
        0.4

        With the background left in, its rise alone would make the fit look
        nearly perfect:

        >>> z, modeled = problem.obs.values, post.modeled_obs.values
        >>> round(float(np.corrcoef(z, modeled)[0, 1] ** 2), 3)
        0.983
        """
        z = self._problem.obs.values
        offset = self._problem.offset
        if offset is not None:
            z = z - offset.values
        fit = self._problem._apply(self.estimate.values)
        return float(np.corrcoef(z, fit)[0, 1] ** 2)

    @property
    def rmse(self) -> float:
        """The root mean square of the observations minus :attr:`modeled_obs`."""
        r = self._problem.obs.values - self.modeled_obs.values
        return float(np.sqrt((r**2).sum() / len(r)))

    @property
    def uncertainty_reduction(self) -> float:
        r"""
        One minus the ratio of the posterior to the prior total error.

        :math:`1 - \sqrt{\operatorname{tr}(\hat{S})} / \sqrt{\operatorname{tr}(S_0)}`.
        """
        prior = self._problem.prior_cov.data.trace()
        return float(1 - np.sqrt(self.variances.values.sum()) / np.sqrt(prior))

    def summary(self) -> pd.DataFrame:
        """
        Return the diagnostics of the solve as a table of one row.

        Nothing is printed: the numbers are returned, and the row of one
        posterior stacks with those of others (``pd.concat``), as in a sweep
        over the regularization factor.

        Returns
        -------
        pandas.DataFrame
            One row: ``gamma`` (the regularization factor), ``n_state`` and
            ``n_obs`` (the sizes of the state and of the observations), then
            :attr:`dofs`, :attr:`information_content`, :attr:`chi2`,
            :attr:`reduced_chi2`, :attr:`r2`, :attr:`rmse` and
            :attr:`uncertainty_reduction`.

        See Also
        --------
        chi2_tests : The chi-squared tests, with their p-values.
        desroziers : The observation error variances the residuals diagnose.
        with_gamma : The posterior with another regularization factor.

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
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> table = pd.concat(
        ...     [post.with_gamma(g).summary() for g in (0.1, 1.0, 10.0)],
        ...     ignore_index=True,
        ... )
        >>> table[["gamma", "dofs", "reduced_chi2"]].round(3)
           gamma   dofs  reduced_chi2
        0    0.1  1.250         1.208
        1    1.0  1.877         1.603
        2   10.0  1.987         1.660
        """
        problem = self._problem
        row = {
            "gamma": self._gamma,
            "n_state": len(problem.prior),
            "n_obs": len(problem.obs),
            "dofs": self.dofs,
            "information_content": self.information_content,
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "r2": self.r2,
            "rmse": self.rmse,
            "uncertainty_reduction": self.uncertainty_reduction,
        }
        return pd.DataFrame([row])

    # -- reusing the factorization --------------------------------------------

    def with_gamma(self, gamma: float) -> "Posterior":
        r"""
        Return the posterior of the same problem with another regularization factor.

        :func:`fips.solve` factors the analytic method's matrix by Cholesky
        at its own gamma. The first call eigendecomposes it (about the cost
        of a solve) and keeps that, since the eigendecomposition reads the
        factors at any gamma: every later call on this posterior, or on one
        it returned, costs :math:`O(m n)` at most, so a sweep over gamma
        eigendecomposes once.

        Parameters
        ----------
        gamma : float
            The regularization factor, as ``fips.solve(problem, gamma=...)``
            takes it.

        Returns
        -------
        Posterior
            The posterior ``fips.solve(problem, gamma=gamma)`` would give, its
            estimate computed.

        Raises
        ------
        TypeError
            If the method cannot change gamma without solving again (its
            solution has no ``with_gamma``).

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
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> [round(post.with_gamma(g).dofs, 3) for g in (0.1, 1.0, 10.0)]
        [1.25, 1.877, 1.987]
        """
        sweep = self._optional("with_gamma", "another gamma's posterior")
        options = dataclasses.replace(self._options, gamma=gamma)
        post = type(self)(self._problem, sweep(options.gamma), options)
        _ = post.estimate
        return post

    def with_data(self, obs: Vector) -> "Posterior":
        r"""
        Return the posterior of the same problem with other observations.

        The analytic method's factorization depends on where the
        observations are and how uncertain they are, not on their values, so
        this costs a product with the factors instead of a new solve (a
        bootstrap, or synthetic observations of a known truth): one
        :math:`m \times m` product in the whitened form, a solve with the
        model-data mismatch and :math:`O(n^2)` in the information form.

        Parameters
        ----------
        obs : Vector
            New observations, with exactly the labels of the problem's (in
            any order).

        Returns
        -------
        Posterior
            The posterior of the problem with `obs`, its estimate computed.

        Raises
        ------
        ValueError
            If `obs` does not have exactly the observations' labels.
        TypeError
            If the method cannot take new observations without solving again
            (its solution has no ``with_data``).
        """
        resolve = self._optional("with_data", "the posterior of other observations")
        problem = self._problem
        space = problem.obs.space
        positions = space._align(
            obs.space, "the new observations", "The new observations and the problem's"
        )
        new = type(problem)(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=Vector(obs.values[positions], space, name=obs.name),
            obs_cov=problem.obs_cov,
            forward=problem.forward,
            offset=problem.offset,
        )
        post = type(self)(new, resolve(new), self._options)
        _ = post.estimate
        return post

    # -- the error of the estimate in a world ----------------------------------

    def _on_labels(self, world: Problem, own_state: bool = False) -> Problem:
        """
        Return `world` with its inputs in the order of this posterior's labels.

        With `own_state`, the world keeps its state (on any labels) and only
        its observations are put in the posterior's order.

        Raises
        ------
        TypeError
            If `world` is not a Problem.
        ValueError
            If its observations, or its state unless `own_state`, do not have
            exactly the labels of this posterior's; the message names those
            on one side only.
        """
        if not isinstance(world, Problem):
            raise TypeError(f"world must be a Problem, got {type(world).__name__}.")
        state, observed = self._problem.prior.space, self._problem.obs.space
        if world.obs.space == observed and (own_state or world.prior.space == state):
            return world
        if own_state:
            prior = world.prior
        else:
            try:
                s = state._align(
                    world.prior.space,
                    "the world's state",
                    "The world's state and the posterior's",
                )
            except ValueError as error:
                raise ValueError(
                    f"{error} A world on other state labels is compared on totals: "
                    "give W on the posterior's state and W_world on the world's."
                ) from None
            prior = Vector(world.prior.values[s], state)
        o = observed._align(
            world.obs.space,
            "the world's observations",
            "The world's observations and the posterior's",
        )
        # Problem puts the operators, covariances and offset on these spaces
        return Problem(
            prior=prior,
            prior_cov=world.prior_cov,
            obs=Vector(world.obs.values[o], observed),
            obs_cov=world.obs_cov,
            forward=world.forward,
            offset=world.offset,
        )

    def recovery(
        self,
        world: Problem | None = None,
        *,
        truth: Vector | None = None,
        W: Operator | None = None,
        W_world: Operator | None = None,
        level: float = 0.95,
    ) -> pd.DataFrame:
        r"""
        Return how well the estimate recovers the truth when the data come from `world`: exactly, without sampling.

        The exact answer of the Monte Carlo loop over ``world.simulate(rng)``
        and :meth:`with_data`. The estimate is linear in the observations,
        :math:`\hat{x} = x_0 + K (z - H x_0 - c)`, with the gain :math:`K` of
        this posterior (its problem's :math:`H`, :math:`c`, :math:`S_0` and
        :math:`S_z / \gamma`). When the world makes the data, with a truth
        :math:`x \sim N(\mu, P)` (its prior, or a given `truth` with
        :math:`P = 0`) and observations :math:`z = H_w x + c_w + e`,
        :math:`e \sim N(0, R_w)` (its forward operator, offset and model-data
        mismatch, not divided by :math:`\gamma`), the error
        :math:`\hat{x} - x` is Gaussian, with

        .. math::

            b = (x_0 - \mu) + K (H_w \mu + c_w - H x_0 - c), \qquad
            V = (K H_w - I) P (K H_w - I)^T + K R_w K^T.

        For an aggregation :math:`W`, :math:`W b` and :math:`W V W^T`, from
        :math:`W K` (``k x m``) and :math:`W K H_w - W` (``k x n``), with
        :math:`P` and :math:`R_w` applied to ``k`` columns in their own forms
        (through the factors of a Kronecker covariance): nothing
        :math:`n \times n` or :math:`m \times m` beyond what the posterior
        holds. With the world the problem itself and :math:`\gamma = 1`, the
        bias is zero and :math:`V` is :math:`\hat{S}` (in Joseph form), so
        ``std`` equals ``posterior_std``.

        The world's state may have other labels than the posterior's (a
        monthly truth recovered by an annual inversion, the whole state where
        the solve was restricted, a gridded truth and a basin inversion) when
        both are added up to the same totals: `W` on the posterior's state and
        `W_world` on the world's, with the same rows. The error of a total is
        then :math:`W \hat{x} - W_w x`, with mean :math:`W x_0 + W K (H_w \mu
        + c_w - H x_0 - c) - W_w \mu` and covariance :math:`G P G^T + W K R_w
        K^T W^T`, :math:`G = W K H_w - W_w` (``k x n_w``): the same formula,
        with :math:`W` on each side its own.

        Parameters
        ----------
        world : Problem, optional
            The problem that makes the data, with exactly this posterior's
            observation labels (in any order) and, unless `W_world` is given,
            its state labels; its own observations are not used. By default
            the posterior's problem.
        truth : Vector, optional
            A fixed truth, on the world's state labels in any order, instead
            of the world's prior (another inventory, 1.5 times the prior).
            Only the observation error is then random.
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it: one row per total. By
            default, one row per state element. Required with `W_world`.
        W_world : Operator, optional
            The aggregation of the world's state to the same totals, for a
            world whose state has other labels than the posterior's: its
            columns the world's state labels, its rows exactly the rows of
            `W` (in any order). By default `W`, so the world must then have
            the posterior's state labels.
        level : float, default 0.95
            The probability of the posterior's interval that ``coverage``
            is computed for, :math:`\hat{x} \pm z_{level} \hat{\sigma}`.

        Returns
        -------
        pandas.DataFrame
            One row per aggregate (or state element), on the aggregation's
            rows (or the state's labels): ``prior`` (:math:`W x_0`),
            ``truth`` (:math:`W \mu`), ``estimate`` (the expected
            :math:`W \hat{x}`), ``bias`` (``estimate - truth``), ``std`` (the
            standard deviation of the error, :math:`\sqrt{\operatorname{diag}
            W V W^T}`), ``posterior_std`` (what the posterior claims,
            :math:`\sqrt{\operatorname{diag} W \hat{S} W^T}`) and ``coverage``
            (the probability that the `level` interval of the posterior holds
            the truth, for an error :math:`N(\text{bias}, \text{std}^2)`).
            A variance below zero by roundoff gives a standard deviation of 0;
            one further below zero, which only a covariance that is not
            positive semidefinite gives, gives NaN (and a NaN ``coverage``),
            with a warning logged. Roundoff is ``sqrt(eps)`` (1.5e-8) times
            the largest variance the row could have under covariances with
            the same variances, :math:`(\sum_i |x_i| \sigma_i)^2`.

        Raises
        ------
        TypeError
            If the method's estimate is not linear in the observations (its
            solution does not set ``estimate_is_linear``), or `world` is not
            a Problem, or `truth` not a Vector.
        ValueError
            If the world's observations or, without `W_world`, its state, the
            truth or the columns of `W` do not have exactly this posterior's
            labels (the message names those on one side only); if `W_world`
            is given without `W`, its columns are not the world's state or
            its rows are not `W`'s; or `level` is not between 0 and 1.

        See Also
        --------
        trials : The same question answered by sampling, for any world.
        fips.Problem.simulate : One draw of a truth and its observations.
        with_data : The posterior of one draw's observations.
        cov : The posterior covariance, of the state or of aggregates.

        Examples
        --------
        Paper 1's experiment: a truth of 1.5 times the prior, observed without
        noise, and the fraction of the true change of each monthly total that
        the inversion recovers. The expected estimate is the one from
        noise-free observations, since the noise has mean zero; ``std`` is the
        spread the noise adds.

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = pd.MultiIndex.from_product(
        ...     [pd.Index([1, 2], name="month"), pd.Index(["a", "b"], name="cell")]
        ... )
        >>> prior = fips.Vector.from_series(pd.Series(1.0, index=state), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series(0.0, index=pd.RangeIndex(6, name="obs_id")), block="ch4"
        ... )
        >>> rng = np.random.default_rng(0)
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(0.5),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(1.0),
        ...     forward=fips.Operator(
        ...         rng.uniform(0.0, 1.0, (6, 4)), obs.space, prior.space
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> monthly = fips.aggregate.sum(prior.space, by="month")
        >>> truth = fips.Vector(prior.values * 1.5, prior.space)
        >>> table = post.recovery(truth=truth, W=monthly)
        >>> table = table.droplevel("block")
        >>> table.round(3).reset_index()
           month  prior  truth  estimate   bias    std  posterior_std  coverage
        0      1    2.0    3.0     2.667 -0.333  0.309          0.528     0.988
        1      2    2.0    3.0     2.538 -0.462  0.285          0.586     0.992
        >>> moved = table["estimate"] - table["prior"]
        >>> needed = table["truth"] - table["prior"]
        >>> (moved / needed).round(3).tolist()  # the recovered fraction
        [0.667, 0.538]

        In its own world (the default), an inversion at :math:`\gamma = 1` has
        no bias, and its intervals hold the truth as often as they claim:

        >>> honest = post.recovery(W=monthly).droplevel("block")
        >>> honest[["bias", "std", "posterior_std", "coverage"]].round(3).reset_index()
           month  bias    std  posterior_std  coverage
        0      1   0.0  0.528          0.528      0.95
        1      2   0.0  0.586          0.586      0.95
        """
        if not getattr(self._solution, "estimate_is_linear", False):
            raise TypeError(
                "The method that solved this problem does not give an estimate "
                f"linear in the observations ({type(self._solution).__name__} does "
                "not set estimate_is_linear), so its error has no closed form: "
                "measure it by sampling, with Posterior.trials."
            )
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be between 0 and 1, got {level}.")
        problem = self._problem
        if W_world is not None and W is None:
            raise ValueError(
                "W_world needs W: a world on other state labels is compared on "
                "totals, W on the posterior's state and W_world on the world's."
            )
        world = (
            problem
            if world is None
            else self._on_labels(world, own_state=W_world is not None)
        )
        if not isinstance(world.forward, Operator):
            raise TypeError(
                "recovery's exact answer needs a world whose forward is a linear "
                f"operator, and this world's is a model, {type(world.forward).__name__}: "
                "a model's data are not Gaussian. Measure the error by sampling, "
                "with Posterior.trials(world)."
            )
        state, world_state = problem.prior.space, world.prior.space
        if truth is None:
            mu, P = world.prior.values, world.prior_cov.data
        else:
            if not isinstance(truth, Vector):
                raise TypeError(f"truth must be a Vector, got {type(truth).__name__}.")
            mu = truth.values
            if truth.space != world_state:
                mu = mu[
                    world_state._align(truth.space, "truth", "The truth and the prior")
                ]
            P = None  # a fixed truth has no spread
        # The world's mean data minus what the problem's prior predicts
        d = (
            world.predict(Vector(mu, world_state)).values
            - problem.predict(problem.prior).values
        )

        if W is None:
            WK = self._solution.gain()  # n x m
            Wx0, Wmu = problem.prior.values, mu
            posterior_var = self.variances.values
            Wt_post = Wt = np.eye(len(state))  # n x n: the per-element case
            index = state.index
        else:
            W_form = self._aggregation(W)
            WK = self._solution.gain(W_form)  # k x m
            Wx0 = np.asarray(W_form @ problem.prior.values)
            posterior_var = self.cov(W).variances.values
            Wt_post = W_form._transpose().dense()  # n x k
            index = W.rows.index
            if W_world is None:
                Ww_form = W_form
            else:
                if not isinstance(W_world, Operator):
                    raise TypeError(
                        f"W_world is an Operator, not {type(W_world).__name__}."
                    )
                Ww_form = W_world._on(
                    W.rows,
                    world_state,
                    "W_world",
                    ("the rows of W", "the world's state"),
                )
            Wmu = np.asarray(Ww_form @ mu)
            Wt = Ww_form._transpose().dense()  # n_w x k

        estimate = Wx0 + WK @ d
        bias = estimate - Wmu
        # diag(W K R_w K^T W^T), with R_w applied to k columns in its form
        KT = WK.T  # m x k
        variance = np.einsum("ik,ik->k", KT, np.asarray(world.obs_cov.data @ KT))
        scale = _largest_variance(KT, world.obs_cov.data)
        if P is not None:
            # G^T = (W K H_w - W)^T = H_w^T (W K)^T - W^T, n x k
            Gt = np.asarray(world._operator.data._transpose() @ KT) - Wt
            variance += np.einsum("ik,ik->k", Gt, np.asarray(P @ Gt))
            scale += _largest_variance(Gt, P)
        std = _std(variance, scale, "error variances")
        prior_scale = _largest_variance(
            None if W is None else Wt_post, problem.prior_cov.data
        )
        posterior_std = _std(posterior_var, prior_scale, "posterior variances")

        half = ndtri(0.5 + level / 2) * posterior_std  # half the interval
        with np.errstate(divide="ignore", invalid="ignore"):
            random = ndtr((half - bias) / std) - ndtr((-half - bias) / std)
        # An error with no spread is inside the interval or not
        coverage = np.where(std > 0, random, (np.abs(bias) <= half).astype(float))
        coverage[np.isnan(std) | np.isnan(posterior_std)] = np.nan
        return pd.DataFrame(
            {
                "prior": Wx0,
                "truth": Wmu,
                "estimate": estimate,
                "bias": bias,
                "std": std,
                "posterior_std": posterior_std,
                "coverage": coverage,
            },
            index=index,
        )

    def trials(
        self,
        world: "Problem | Callable[[np.random.Generator], tuple[Vector, Vector]] | None" = None,
        n: int = 400,
        rng: "int | np.random.Generator | None" = None,
        *,
        W: Operator | None = None,
        W_world: Operator | None = None,
    ) -> pd.DataFrame:
        r"""
        Return the truth and the estimate of each total in `n` trials of data made by `world`: the sampled :meth:`recovery`.

        Each trial, the world makes a truth and its observations, this
        posterior solves the observations with its own assumptions
        (:meth:`with_data`, one product), and the totals of the two go into
        one row per total. The table is the loop :meth:`recovery` answers
        exactly, kept trial by trial, for what the exact answer cannot take: a
        world whose truth is not Gaussian or whose error is not additive, a
        statistic of each trial (its reduced chi-squared), or a method whose
        estimate is not linear in the observations (the spread it claims is
        then each trial's own). Where both apply, the
        table's bias, spread and coverage agree with ``recovery`` within their
        Monte Carlo standard errors.

        Parameters
        ----------
        world : Problem or callable, optional
            What makes the data. A :class:`~fips.Problem` makes them with
            :meth:`~fips.Problem.simulate` (a truth drawn from its prior, the
            observations with its error); its observations must have this
            posterior's labels, in any order, and its state this posterior's
            unless `W_world` is given. Or a function of a
            :class:`numpy.random.Generator` that returns ``(truth, obs)``, two
            :class:`~fips.Vector`: a lognormal truth, an error the inversion
            cannot see. By default the posterior's own problem.
        n : int, default 400
            The number of trials. With 400, a bias is known to a twentieth of
            the spread and a coverage near 0.95 to about 0.01.
        rng : int, numpy.random.Generator or None, default None
            A seed, or the generator to draw from, as
            :meth:`~fips.Problem.simulate` takes it. The same seed gives two
            posteriors the same data, trial for trial, so their tables pair.
        W : Operator, optional
            An aggregation, as :meth:`recovery` takes it: one row per total.
            By default, one row per state element.
        W_world : Operator, optional
            The aggregation of the world's state to the same totals, as
            :meth:`recovery` takes it, for a world whose state has other
            labels than the posterior's.

        Returns
        -------
        pandas.DataFrame
            One row per trial and total, indexed by ``trial`` (0 to ``n - 1``)
            and then the rows of `W` (or the state's labels): ``truth``
            (:math:`W_w x`), ``estimate`` (:math:`W \hat{x}`),
            ``posterior_std`` (the spread the posterior claims,
            :math:`\sqrt{\operatorname{diag} W \hat{S} W^T}`: the same in
            every trial of a method whose estimate is linear in the
            observations, and for any other that of the trial's own
            posterior) and ``reduced_chi2`` (the trial's
            :attr:`reduced_chi2`, the same on every row of a trial). Its
            ``groupby`` over the totals (with ``dropna=False``: a state of
            several blocks is NaN on the levels a block lacks) gives the
            performance measures: the mean of ``estimate - truth`` is the
            bias, its standard deviation the spread, and the fraction of
            trials with ``|estimate - truth| <= z * posterior_std`` the
            coverage.

        Raises
        ------
        TypeError
            If `world` is neither a Problem nor callable, or returns something
            other than two Vectors; or the method cannot take new
            observations (its solution has no ``with_data``).
        ValueError
            If `n` is not a positive integer; if the world's observations do
            not have exactly this posterior's labels, or its truth not the
            state's (without `W_world`); if `W_world` is given without `W`,
            or its columns are not the truth's labels or its rows not `W`'s.

        See Also
        --------
        recovery : The exact bias, spread and coverage, without sampling.
        fips.Problem.simulate : One draw of a truth and its observations.
        with_data : The posterior of one draw's observations.

        Examples
        --------
        The honest world of a small problem, sampled, beside the exact answer:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> cells = pd.Index(["a", "b"], name="cell")
        >>> prior = fips.Vector.from_series(pd.Series(1.0, index=cells), block="flux")
        >>> obs = fips.Vector(
        ...     np.zeros(3), fips.Space({"co2": pd.RangeIndex(3, name="obs_id")})
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(0.5),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(0.2),
        ...     forward=fips.Operator(
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ... )
        >>> post = fips.solve(problem)
        >>> table = post.trials(n=2000, rng=0)
        >>> table.index.names
        FrozenList(['trial', 'block', 'cell'])
        >>> error = table["estimate"] - table["truth"]
        >>> error.groupby(level="cell").std().round(
        ...     2
        ... ).tolist()  # the spread over trials
        [0.16, 0.15]
        >>> post.recovery()["std"].round(2).tolist()  # the exact spread
        [0.15, 0.15]
        """
        resolve = self._optional("with_data", "the posterior of other observations")
        if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
            raise ValueError(
                f"n is the number of trials, a positive integer, got {n!r}."
            )
        if W_world is not None and W is None:
            raise ValueError(
                "W_world needs W: a world on other state labels is compared on "
                "totals, W on the posterior's state and W_world on the world's."
            )
        problem = self._problem
        if world is None:
            make = problem.simulate
        elif isinstance(world, Problem):
            make = world.simulate
        elif callable(world):
            make = world
        else:
            raise TypeError(
                "world is a Problem (its simulate makes the data) or a function of "
                f"a numpy Generator returning (truth, obs), not {type(world).__name__}."
            )
        if W_world is not None and not isinstance(W_world, Operator):
            raise TypeError(f"W_world is an Operator, not {type(W_world).__name__}.")
        generator = np.random.default_rng(rng)
        state = problem.prior.space
        if W is None:
            W_form, rows_of_W, index = None, state, state.index
        else:
            W_form, rows_of_W, index = self._aggregation(W), W.rows, W.rows.index

        def claims(post: "Posterior") -> np.ndarray:
            """Return the spread `post` claims for each row: the root of its variances."""
            variances = post.variances if W is None else post.cov(W).variances
            return np.sqrt(np.maximum(variances.values, 0.0))

        # a linear method claims the same in every trial (its posterior does not
        # depend on the data); any other is asked of each trial's own posterior
        linear = getattr(self._solution, "estimate_is_linear", False)
        claimed = claims(self) if linear else None
        del resolve  # with_data checks the observations' labels each trial

        truths, estimates, chi2, stds = [], [], [], []
        for _ in range(n):
            drawn = make(generator)
            if (
                not isinstance(drawn, tuple)
                or len(drawn) != 2
                or not all(isinstance(v, Vector) for v in drawn)
            ):
                raise TypeError(
                    "The world must return (truth, obs), two Vectors, as "
                    "Problem.simulate does."
                )
            truth, obs = drawn
            trial = self.with_data(obs)
            estimate = trial.estimate.values
            if W_world is not None:
                Ww_form = W_world._on(
                    rows_of_W,
                    truth.space,
                    "W_world",
                    ("the rows of W", "the world's truth"),
                )
                x = np.asarray(Ww_form @ truth.values)
            else:
                x = truth.values
                if truth.space != state:
                    try:
                        x = x[
                            state._align(
                                truth.space, "the truth", "The truth and the state"
                            )
                        ]
                    except ValueError as error:
                        raise ValueError(
                            f"{error} A world on other state labels is compared on "
                            "totals: give W on the posterior's state and W_world on "
                            "the world's."
                        ) from None
                if W_form is not None:
                    x = np.asarray(W_form @ x)
            truths.append(x)
            estimates.append(
                estimate if W_form is None else np.asarray(W_form @ estimate)
            )
            chi2.append(trial.reduced_chi2)
            if claimed is None:
                stds.append(claims(trial))
        k = len(index)
        rows = pd.MultiIndex.from_arrays(
            [np.repeat(np.arange(n), k)]
            + [
                index.get_level_values(i).take(np.tile(np.arange(k), n))
                for i in range(index.nlevels)
            ],
            names=["trial", *index.names],
        )
        return pd.DataFrame(
            {
                "truth": np.concatenate(truths),
                "estimate": np.concatenate(estimates),
                "posterior_std": (
                    np.tile(claimed, n) if claimed is not None else np.concatenate(stds)
                ),
                "reduced_chi2": np.repeat(np.asarray(chi2, dtype=float), k),
            },
            index=rows,
        )
