r"""
The posterior of a solved inverse problem.

:func:`fips.solve` returns a :class:`Posterior`: the problem, the options of
the method, and what the method kept from its solve (a :class:`Solution`).
The posterior labels the answer; the method keeps its own machinery. It asks
the solution for unlabeled arrays (the estimate, the variances, the
covariance of aggregates, the gain, DOFS, chi-squared), puts the problem's
labels on them and keeps each one once computed. Everything else it gives is
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
from functools import cached_property
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.linalg import inv, solve
from scipy.special import ndtr, ndtri  # the normal CDF and its inverse

from fips.covariance import Covariance
from fips.linalg import LinearOperator
from fips.linalg.forms import _adopt, _solve_psd
from fips.operator import Operator
from fips.problem import Problem
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

    Two members are optional, and a posterior whose solution lacks one says
    so when it is asked for the feature:

    - ``with_gamma(gamma) -> Solution``: the solution at another
      regularization factor, without solving again (:meth:`Posterior.with_gamma`).
    - ``with_data(problem) -> Solution``: the solution of `problem`, the same
      problem with other values of the observations, without solving again
      (:meth:`Posterior.with_data`).

    And a solution whose :meth:`estimate` is the posterior mean says so with a
    class attribute ``estimate_is_mean = True``; without it, a posterior has
    no :attr:`~Posterior.mean`. One whose estimate is linear in the
    observations, :math:`\hat{x} = x_0 + K (z - H x_0 - c)` with the
    :math:`K` of :meth:`gain` whatever the values of :math:`z`, says so with
    ``estimate_is_linear = True``; without it, a posterior has no
    :meth:`~Posterior.recovery`.

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

    def chi2(self) -> float:
        r"""Return the chi-squared of the solution: the sum of the data and the state misfits."""
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
    on the solve or on the ``m x m`` matrix the solve factors
    (:attr:`innovation_cov`); what needs neither is on the
    :class:`~fips.Problem` (:meth:`~fips.Problem.predict`).

    Every member works for every method, except four that need more of the
    method than a :class:`Solution` gives: :attr:`mean` (where the estimate is
    the mean), :meth:`with_gamma` and :meth:`with_data` (where the method can
    answer without solving again), and :meth:`recovery` (where the estimate is
    linear in the observations). The analytic method has all four.

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
        :math:`O(m n)`, without the :math:`n \times n` covariance.

        See Also
        --------
        cov : The posterior error covariance, of the state or of aggregates.
        """
        values = self._solution.variances()
        return Vector(values, self._problem.prior.space, name="variance")

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
            covariance.

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
        A = self.kalman_gain.dense() @ self._problem.forward.data
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
        HWt = np.asarray(self._problem.forward.data @ W_form.T)  # m x k
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
        H = self._problem.forward.data
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
            L = Hx \left((Hx)^T G^{-1} Hx\right)^{-1} (Hx)^T G^{-1},
            \qquad G = H S_0 H^T + S_z / \gamma

        This is the generalized-least-squares hat matrix of the one regressor
        :math:`Hx` (which includes the offset :math:`c`): the projection onto
        :math:`Hx` in the metric :math:`G^{-1}`. It is idempotent and its
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

    @property
    def chi2(self) -> float:
        r"""
        The chi-squared of the solution: the sum of the data and the state misfits.

        .. math::
            \chi^2 = (z - H\hat{x} - c)^T (S_z/\gamma)^{-1} (z - H\hat{x} - c)
                + (\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0)

        which equals :math:`d^T (H S_0 H^T + S_z/\gamma)^{-1} d` with
        :math:`d = z - H x_0 - c`; the analytic method reads it from its
        factors.

        See Also
        --------
        reduced_chi2 : It, divided by the number of observations.
        """
        return self._solution.chi2()

    @property
    def reduced_chi2(self) -> float:
        """
        :attr:`chi2` divided by the number of observations; near 1 when the covariances are right.

        Tarantola (1987), as in Kunik et al. (2019),
        https://doi.org/10.1525/elementa.375.
        """
        return self.chi2 / len(self._problem.obs)

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
        fit = self._problem.forward.data @ self.estimate.values
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
            :attr:`dofs`, :attr:`chi2`, :attr:`reduced_chi2`, :attr:`r2`,
            :attr:`rmse` and :attr:`uncertainty_reduction`.

        See Also
        --------
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

    def _on_labels(self, world: Problem) -> Problem:
        """
        Return `world` with its inputs in the order of this posterior's labels.

        Raises
        ------
        TypeError
            If `world` is not a Problem.
        ValueError
            If its state or observations do not have exactly the labels of
            this posterior's; the message names those on one side only.
        """
        if not isinstance(world, Problem):
            raise TypeError(f"world must be a Problem, got {type(world).__name__}.")
        state, observed = self._problem.prior.space, self._problem.obs.space
        if world.prior.space == state and world.obs.space == observed:
            return world
        s = state._align(
            world.prior.space,
            "the world's state",
            "The world's state and the posterior's",
        )
        o = observed._align(
            world.obs.space,
            "the world's observations",
            "The world's observations and the posterior's",
        )
        # Problem puts the operators, covariances and offset on these spaces
        return Problem(
            prior=Vector(world.prior.values[s], state),
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

        Parameters
        ----------
        world : Problem, optional
            The problem that makes the data, with exactly this posterior's
            state and observation labels (in any order); its own observations
            are not used. By default the posterior's problem. A world on
            other state labels (the whole state where the solve was
            restricted) is not taken: use ``world.simulate`` and
            :meth:`with_data` in a loop.
        truth : Vector, optional
            A fixed truth, on the state's labels in any order, instead of the
            world's prior (another inventory, 1.5 times the prior). Only the
            observation error is then random.
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it: one row per total. By
            default, one row per state element.
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
            If the world's state or observations, the truth or the columns of
            `W` do not have exactly this posterior's labels (the message names
            those on one side only), or `level` is not between 0 and 1.

        See Also
        --------
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
                "use Problem.simulate and with_data in a loop."
            )
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be between 0 and 1, got {level}.")
        problem = self._problem
        world = problem if world is None else self._on_labels(world)
        state = problem.prior.space
        if truth is None:
            mu, P = world.prior.values, world.prior_cov.data
        else:
            if not isinstance(truth, Vector):
                raise TypeError(f"truth must be a Vector, got {type(truth).__name__}.")
            mu = truth.values
            if truth.space != state:
                mu = mu[state._align(truth.space, "truth", "The truth and the prior")]
            P = None  # a fixed truth has no spread
        # The world's mean data minus what the problem's prior predicts
        d = (
            world.predict(Vector(mu, state)).values
            - problem.predict(problem.prior).values
        )

        if W is None:
            WK = self._solution.gain()  # n x m
            Wx0, Wmu = problem.prior.values, mu
            posterior_var = self.variances.values
            Wt = np.eye(len(state))  # n x n: the per-element case
            index = state.index
        else:
            W_form = self._aggregation(W)
            WK = self._solution.gain(W_form)  # k x m
            Wx0 = np.asarray(W_form @ problem.prior.values)
            Wmu = np.asarray(W_form @ mu)
            posterior_var = self.cov(W).variances.values
            Wt = W_form._transpose().dense()  # n x k
            index = W.rows.index

        estimate = Wx0 + WK @ d
        bias = estimate - Wmu
        # diag(W K R_w K^T W^T), with R_w applied to k columns in its form
        KT = WK.T  # m x k
        variance = np.einsum("ik,ik->k", KT, np.asarray(world.obs_cov.data @ KT))
        scale = _largest_variance(KT, world.obs_cov.data)
        if P is not None:
            # G^T = (W K H_w - W)^T = H_w^T (W K)^T - W^T, n x k
            Gt = np.asarray(world.forward.data._transpose() @ KT) - Wt
            variance += np.einsum("ik,ik->k", Gt, np.asarray(P @ Gt))
            scale += _largest_variance(Gt, P)
        std = _std(variance, scale, "error variances")
        prior_scale = _largest_variance(
            None if W is None else Wt, problem.prior_cov.data
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
