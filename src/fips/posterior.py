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
from fips.covariance import CovarianceMatrix
from fips.estimators import Estimator
from fips.matrix import Matrix
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
        """Label ``values`` with the state index."""
        return Vector(values, index=self.state_index, name=name)

    def _obs_vector(self, values: np.ndarray, name: str) -> Vector:
        """Label ``values`` with the observation index."""
        return Vector(values, index=self.obs_index, name=name)

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
    def covariance(self) -> CovarianceMatrix:
        r"""
        The posterior error covariance, :math:`\hat{S}`, on the state index.

        A dense n x n matrix.

        See Also
        --------
        variances : Its diagonal.
        """
        return CovarianceMatrix(
            np.asarray(self.estimator.S_hat),
            index=self.state_index,
            columns=self.state_index,
            name="posterior_error",
        )

    @cached_property
    def variances(self) -> Vector:
        r"""
        The posterior error variances, the diagonal of :math:`\hat{S}`.

        Taken from the dense :attr:`covariance`.

        See Also
        --------
        covariance : The full posterior error covariance.
        """
        return self._state_vector(
            np.diag(np.asarray(self.estimator.S_hat)).copy(), "posterior_variance"
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
    def prior_obs_covariance(self) -> CovarianceMatrix:
        r"""
        The prior error in observation space, :math:`H S_0 H^T`.

        The uncertainty of :attr:`prior_modeled_obs` that the prior error
        implies, on the observation index. Its diagonal, square-rooted, is the
        prior's share of each observation's error budget; add the model-data
        mismatch in quadrature for the total the solve expects.
        """
        return CovarianceMatrix(
            np.asarray(self.estimator.S_y0),
            index=self.obs_index,
            columns=self.obs_index,
            name="prior_obs_error",
        )

    @cached_property
    def kalman_gain(self) -> Matrix:
        r"""
        The gain :math:`K`, state by observations.

        :math:`\hat{x} = x_0 + K (z - H x_0 - c)`.
        """
        return Matrix(
            np.asarray(self.estimator.K),
            index=self.state_index,
            columns=self.obs_index,
            name="kalman_gain",
        )

    @cached_property
    def averaging_kernel(self) -> Matrix:
        r"""
        The averaging kernel :math:`A = K H`, state by state.

        See Also
        --------
        dofs : Its trace.
        """
        return Matrix(
            np.asarray(self.estimator.A),
            index=self.state_index,
            columns=self.state_index,
            name="averaging_kernel",
        )

    @cached_property
    def desroziers(self) -> CovarianceMatrix:
        """
        The observation error covariance diagnosed by Desroziers et al. (2005).

        On the observation index. See :attr:`~fips.Estimator.desroziers` for the
        formula.
        """
        return CovarianceMatrix(
            self.estimator.desroziers,
            index=self.obs_index,
            columns=self.obs_index,
            name="desroziers",
        )

    @property
    def dofs(self) -> float:
        """
        Degrees of freedom for signal, the trace of :attr:`averaging_kernel`.

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
