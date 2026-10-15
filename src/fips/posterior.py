"""
The posterior of a solved inverse problem.

:func:`fips.solve` fits an estimator and returns a :class:`~fips.Posterior`, which
puts the problem's labels on what the estimator computed: the state index on
the state axes, the observation index on the observation axes.
"""

import logging
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from fips.base import Pickleable
from fips.covariance import Covariance
from fips.estimators import Estimator
from fips.linalg import LinearOperator
from fips.operator import Operator
from fips.space import Space
from fips.vector import Vector

logger = logging.getLogger(__name__)


class Posterior(Pickleable):
    """
    The solution of an inverse problem, labeled.

    :func:`fips.solve` builds it; there is no need to build one by hand. Each
    output is computed when first asked for, then kept, so asking again
    returns the same object. The outputs are shared: change one in place and
    the next access shows the change.

    Parameters
    ----------
    estimator : Estimator
        The fitted estimator.
    state_index : pd.Index
        The problem's state index, which labels every state axis.
    obs_index : pd.Index
        The problem's observation index, which labels every observation axis.

    See Also
    --------
    fips.solve : Solve a problem and return its posterior.
    fips.InverseProblem.solve : Solve a problem in place.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> state = pd.Index(["a", "b"], name="cell")
    >>> obs = pd.Index(["o1", "o2", "o3"], name="obs_id")
    >>> problem = fips.InverseProblem(
    ...     obs=pd.Series([1.0, 2.0, 3.0], index=obs, name="obs"),
    ...     prior=pd.Series([0.0, 0.0], index=state, name="prior"),
    ...     forward_operator=pd.DataFrame(
    ...         [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=obs, columns=state
    ...     ),
    ...     prior_error=1.0,
    ...     modeldata_mismatch=0.1,
    ... )
    >>> post = fips.solve(problem)
    >>> post.estimate.values.round(3)
    array([0.997, 1.906])
    >>> round(post.dofs, 3)
    1.877
    """

    estimator: Estimator
    """The fitted estimator, which holds the unlabeled arrays."""
    state_index: pd.Index
    """The state index, which labels every state axis."""
    obs_index: pd.Index
    """The observation index, which labels every observation axis."""

    def __init__(
        self, estimator: Estimator, state_index: pd.Index, obs_index: pd.Index
    ):
        self.estimator = estimator
        self.state_index = state_index
        self.obs_index = obs_index

    @cached_property
    def state_space(self) -> Space:
        """
        The state's labels as a :class:`~fips.Space`, which labels every state axis of a matrix.

        Raises
        ------
        ValueError
            If the state index has a block's rows apart (the estimator's
            arrays would not be in the space's order).
        """
        return _space_of(self.state_index, "state")

    @cached_property
    def obs_space(self) -> Space:
        """The observations' labels as a :class:`~fips.Space`, which labels every observation axis of a matrix."""
        return _space_of(self.obs_index, "observation")

    def __getstate__(self) -> dict[str, Any]:
        """Return the estimator and the two indexes; the labeled outputs are rebuilt."""
        return {
            "estimator": self.estimator,
            "state_index": self.state_index,
            "obs_index": self.obs_index,
        }

    def __setstate__(self, state: dict[str, Any]):
        """Restore the estimator and the two indexes."""
        self.estimator = state["estimator"]
        self.state_index = state["state_index"]
        self.obs_index = state["obs_index"]

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"estimator={type(self.estimator).__name__}, "
            f"n_state={len(self.state_index)}, n_obs={len(self.obs_index)})"
        )

    # The names given to the outputs below are the names they had as
    # properties of InverseProblem, so ``problem.posterior`` and
    # ``post.estimate`` are one object with one name.

    def _state_vector(self, values: np.ndarray, name: str) -> Vector:
        """Label ``values`` with the state space."""
        return Vector(values, self.state_space, name=name)

    def _obs_vector(self, values: np.ndarray, name: str) -> Vector:
        """Label ``values`` with the observation space."""
        return Vector(values, self.obs_space, name=name)

    @cached_property
    def estimate(self) -> Vector:
        r"""
        The posterior state estimate, :math:`\hat{x}`, on the state index.

        The point estimate of the method. For the analytic Bayesian solve it
        is the posterior mean, so it is also :attr:`mean`.

        See Also
        --------
        mean : The posterior mean, where the method gives it.
        covariance : The posterior error covariance.
        """
        return self._state_vector(self.estimator.x_hat, "posterior")

    @cached_property
    def mean(self) -> Vector:
        r"""
        The posterior mean, on the state index.

        Every estimator in fips gives the posterior mean as its estimate
        (:attr:`~fips.Estimator.x_hat`), so this is the same object as
        :attr:`estimate`. A method whose estimate is not the mean, such as a
        bounded one that gives the mode, will not have it.

        See Also
        --------
        estimate : The point estimate.
        """
        return self.estimate

    @cached_property
    def covariance(self) -> Covariance:
        r"""
        The posterior error covariance, :math:`\hat{S}`, on the state space.

        A dense n x n matrix.

        See Also
        --------
        variances : Its diagonal.
        """
        return Covariance(
            np.asarray(self.estimator.S_hat), self.state_space, name="posterior_error"
        )

    @cached_property
    def variances(self) -> Vector:
        r"""
        The posterior error variances, the diagonal of :math:`\hat{S}`.

        The estimator's :attr:`~fips.Estimator.S_hat_diagonal`: for the
        Bayesian solve, computed from its factorization without forming the
        ``n x n`` :attr:`covariance`.

        See Also
        --------
        covariance : The full posterior error covariance.
        cov : The covariance of aggregates of the state.
        """
        return self._state_vector(self.estimator.S_hat_diagonal, "posterior_variance")

    def _aggregation(self, W: Operator) -> LinearOperator:
        """
        Return an aggregation's matrix with its columns in state order.

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If the columns are not exactly the state's labels (a label
            missing or extra).
        """
        if not isinstance(W, Operator):
            raise TypeError(
                f"An aggregation is an Operator, not {type(W).__name__}: build one "
                "with Operator.from_frame or Operator.from_blocks."
            )
        try:
            return W._on(W.rows, self.state_space)
        except ValueError as err:
            raise ValueError(
                "The aggregation's columns must have exactly the state's labels "
                f"(the other space below is the aggregation's columns). {err}"
            ) from None

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
            Without `W`, :attr:`covariance` (dense, ``n x n``). With `W`,
            :math:`W \hat{S} W^T` on the aggregation's rows, ``k x k``,
            computed without the ``n x n`` covariance for the Bayesian solve.

        Raises
        ------
        ValueError
            If the columns of `W` are not exactly the state's labels: a state
            label missing, or a label that is not in the state.

        See Also
        --------
        variances : The diagonal of the full covariance.
        averaging_kernel : The averaging kernel, of the state or of aggregates.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = pd.Index(["a", "b", "c"], name="cell")
        >>> obs = pd.Index(["o1", "o2"], name="obs_id")
        >>> problem = fips.InverseProblem(
        ...     obs=pd.Series([1.0, 2.0], index=obs, name="obs"),
        ...     prior=pd.Series(0.0, index=state, name="prior"),
        ...     forward_operator=pd.DataFrame(
        ...         [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], index=obs, columns=state
        ...     ),
        ...     prior_error=1.0,
        ...     modeldata_mismatch=0.1,
        ... )
        >>> post = fips.solve(problem)
        >>> total = fips.Operator.from_frame(
        ...     pd.DataFrame(
        ...         1.0, index=pd.Index(["total"], name="region"), columns=state
        ...     ),
        ...     row_block="region",
        ...     col_block="prior",
        ... )
        >>> post.cov(total).dense().round(3)
        array([[0.722]])
        """
        if W is None:
            return self.covariance
        return Covariance(
            self.estimator.aggregate_S_hat(self._aggregation(W)),
            W.rows,
            name="posterior_error",
        )

    @cached_property
    def modeled_obs(self) -> Vector:
        r"""
        The observations the estimate models, :math:`H \hat{x} + c`.

        See Also
        --------
        prior_modeled_obs : The observations the prior models.
        """
        return self._obs_vector(self.estimator.y_hat, "modeled_obs")

    @cached_property
    def prior_modeled_obs(self) -> Vector:
        r"""
        The observations the prior models, :math:`H x_0 + c`.

        See Also
        --------
        modeled_obs : The observations the estimate models.
        prior_obs_covariance : Their covariance from the prior error.
        """
        return self._obs_vector(self.estimator.y_0, "prior_obs")

    @cached_property
    def prior_obs_covariance(self) -> Covariance:
        r"""
        The prior error in observation space, :math:`H S_0 H^T`.

        The uncertainty of :attr:`prior_modeled_obs` that the prior error
        implies, on the observation index. Its diagonal, square-rooted, is the
        prior's share of each observation's error budget; add the model-data
        mismatch in quadrature for the total the solve expects.
        """
        return Covariance(
            np.asarray(self.estimator.S_y0), self.obs_space, name="prior_obs_error"
        )

    @cached_property
    def kalman_gain(self) -> Operator:
        r"""
        The gain :math:`K`, state by observations.

        :math:`\hat{x} = x_0 + K (z - H x_0 - c)`.
        """
        return Operator(
            np.asarray(self.estimator.K),
            self.state_space,
            self.obs_space,
            name="kalman_gain",
        )

    @cached_property
    def _averaging_kernel(self) -> Operator:
        """The full averaging kernel, built once."""
        return Operator(
            np.asarray(self.estimator.A),
            self.state_space,
            self.state_space,
            name="averaging_kernel",
        )

    def averaging_kernel(self, W: Operator | None = None) -> Operator:
        r"""
        Return the averaging kernel, of the state or of aggregates of it.

        Without `W`, :math:`A = K H`, state by state (dense, ``n x n``; the same
        object each time). With an aggregation :math:`W`, :math:`W A W^*`,
        where :math:`W^* = W^T (W W^T)^{-1}` is the pseudo-inverse of
        :math:`W`: it spreads a change of the aggregates back over the state
        elements (for a sum, evenly over the elements summed), so :math:`W A
        W^*` is how the aggregated estimate responds to the aggregated truth.
        This is the form IMI uses; :math:`W A W^T` is not an averaging kernel.
        For the Bayesian solve it is computed without the ``n x n`` kernel.

        Parameters
        ----------
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it. Its rows must be linearly
            independent.

        Returns
        -------
        Operator
            ``n x n`` on the state space, or ``k x k`` on the aggregation's
            rows.

        Raises
        ------
        ValueError
            If the columns of `W` are not exactly the state's labels, or its
            rows are linearly dependent.

        See Also
        --------
        dofs : The trace of the full averaging kernel.
        cov : The posterior covariance, of the state or of aggregates.
        """
        if W is None:
            return self._averaging_kernel
        return Operator(
            self.estimator.aggregate_A(self._aggregation(W)),
            W.rows,
            W.rows,
            name="averaging_kernel",
        )

    def with_gamma(self, gamma: float) -> "Posterior":
        r"""
        Return the posterior of the same problem with another regularization factor.

        The Bayesian solve keeps its factorization, which does not depend on
        gamma, so the new posterior costs :math:`O(m n)` instead of a new
        solve: a sweep over gamma factors once. The problem is unchanged.

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
            If the estimator has no regularization factor to change.

        Examples
        --------
        >>> import pandas as pd
        >>> import fips
        >>> state = pd.Index(["a", "b"], name="cell")
        >>> obs = pd.Index(["o1", "o2", "o3"], name="obs_id")
        >>> problem = fips.InverseProblem(
        ...     obs=pd.Series([1.0, 2.0, 3.0], index=obs, name="obs"),
        ...     prior=pd.Series([0.0, 0.0], index=state, name="prior"),
        ...     forward_operator=pd.DataFrame(
        ...         [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=obs, columns=state
        ...     ),
        ...     prior_error=1.0,
        ...     modeldata_mismatch=0.1,
        ... )
        >>> post = fips.solve(problem)
        >>> [round(post.with_gamma(g).dofs, 3) for g in (0.1, 1.0, 10.0)]
        [0.886, 1.877, 1.986]
        """
        with_gamma = getattr(self.estimator, "with_gamma", None)
        if with_gamma is None:
            raise TypeError(
                f"{type(self.estimator).__name__} has no regularization factor to change."
            )
        post = type(self)(with_gamma(gamma), self.state_index, self.obs_index)
        _ = post.estimate
        return post

    @cached_property
    def desroziers(self) -> Covariance:
        """
        The observation error covariance diagnosed by Desroziers et al. (2005).

        On the observation index. See :attr:`~fips.Estimator.desroziers` for the
        formula.
        """
        return Covariance(self.estimator.desroziers, self.obs_space, name="desroziers")

    @property
    def dofs(self) -> float:
        """
        Degrees of freedom for signal, the trace of :meth:`averaging_kernel`.

        The estimator's :attr:`~fips.Estimator.DOFS`.
        """
        return self.estimator.DOFS

    @property
    def reduced_chi2(self) -> float:
        """
        The reduced chi-squared of the solution.

        The estimator's :attr:`~fips.Estimator.reduced_chi2`.
        """
        return self.estimator.reduced_chi2

    @property
    def r2(self) -> float:
        """
        The squared correlation of the observations and :attr:`modeled_obs`.

        The estimator's :attr:`~fips.Estimator.R2`.
        """
        return self.estimator.R2

    @property
    def rmse(self) -> float:
        """
        The root mean square of the observations minus :attr:`modeled_obs`.

        The estimator's :attr:`~fips.Estimator.RMSE`.
        """
        return self.estimator.RMSE

    @property
    def uncertainty_reduction(self) -> float:
        r"""
        One minus the ratio of the posterior to the prior total error.

        The estimator's :attr:`~fips.Estimator.uncertainty_reduction`: the square
        roots of the traces of :math:`\hat{S}` and :math:`S_0`.
        """
        return self.estimator.uncertainty_reduction


def _space_of(index: pd.Index, what: str) -> Space:
    """
    Return the space of `index`, checking that it keeps the index's order.

    :meth:`Space.from_index` puts each block's rows together, so an index
    whose blocks are interleaved would label the estimator's arrays wrongly.
    """
    space = Space.from_index(index)
    blocks = index.get_level_values("block").astype(str)
    rank = pd.Index(space.names).get_indexer(blocks)
    if (np.diff(rank) < 0).any():
        raise ValueError(
            f"The {what} index has the rows of a block apart; put each block's "
            "rows together (InverseProblem does)."
        )
    return space
