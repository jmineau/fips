r"""
The posterior of a solved inverse problem.

:func:`fips.solve` returns a :class:`Posterior`: the problem, the options of
the method, and what the method kept from its solve (a :class:`Solution`).
The posterior labels the answer; the method keeps its own machinery. It asks
the solution for unlabeled arrays (the estimate, the variances, the
covariance of aggregates, the gain, DOFS, chi-squared), puts the problem's
labels on them and keeps each one once computed. Everything else it gives is
written from those and the problem, so it holds for every method. The
analytic method's solution is the factorization of its whitened system
(:mod:`fips.methods.analytic`).

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
    answer these questions (the analytic method keeps the factorization of
    its whitened system; a sampling method would keep its samples). Each
    answer is an unlabeled NumPy array (or a float), in the order of the
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

    A third is optional because the posterior can do without it: ``HS_0H``,
    the attribute holding :math:`H S_0 H^T` (``m x m``, a ``LinearOperator``)
    for a method that formed it. :attr:`Posterior.innovation_cov` and
    :meth:`Posterior.leverage` use it instead of forming it again, and form it
    themselves when the solution has none.

    And a solution whose :meth:`estimate` is the posterior mean says so with a
    class attribute ``estimate_is_mean = True``; without it, a posterior has
    no :attr:`~Posterior.mean`.

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

    Every member works for every method, except three that need more of the
    method than a :class:`Solution` gives: :attr:`mean` (where the estimate is
    the mean), :meth:`with_gamma` and :meth:`with_data` (where the method can
    answer without solving again). The analytic method has all three.

    Parameters
    ----------
    problem : Problem
        The problem that was solved.
    solution : Solution
        What the method kept from the solve, which gives the unlabeled
        arrays (for the analytic method, the factorization of its whitened
        system, ``fips.methods.analytic.Whitened``).
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
        r""":math:`H S_0 H^T`, in the cheapest form of its factors: the solution's, or formed here."""
        kept = getattr(self._solution, "HS_0H", None)
        if kept is not None:
            return kept
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

        The analytic method reads it from its factors,
        :math:`\sum_k \gamma\lambda_k / (\gamma\lambda_k + 1)` over the
        eigenvalues :math:`\lambda_k` of the whitened system, without the
        kernel.
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
        """The squared correlation of the observations and :attr:`modeled_obs`."""
        z = self._problem.obs.values
        return float(np.corrcoef(z, self.modeled_obs.values)[0, 1] ** 2)

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

        The analytic method's factorization does not depend on gamma, so
        this costs :math:`O(m n)` instead of a new solve: a sweep over gamma
        factors once.

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
        this costs one :math:`m \times m` product instead of a new solve (a
        bootstrap, or synthetic observations of a known truth).

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
        state = problem.__getstate__()
        state["obs"] = Vector(obs.values[positions], space, name=obs.name)
        new = type(problem).__new__(type(problem))
        new.__setstate__(state)
        post = type(self)(new, resolve(new), self._options)
        _ = post.estimate
        return post
