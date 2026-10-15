r"""
The analytic method: the closed-form Gaussian posterior of a linear problem.

``_analytic`` factors the problem's whitened system once, in a ``Whitened``
(see :mod:`fips.posterior`), and the :class:`~fips.Posterior` it returns
derives everything from the factors, for any regularization factor
:math:`\gamma`. :class:`AnalyticOptions` holds that factor.
"""

import math
from dataclasses import dataclass

from fips.posterior import Posterior, Whitened
from fips.problem import Problem


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
