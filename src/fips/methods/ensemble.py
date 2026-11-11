r"""
The ensemble method: a square-root ensemble Kalman update, for a forward model that can only be run.

A model with no Jacobian (an atmospheric transport model, such as a TM5 or
WRF configuration) cannot be linearized by hand, but it can be run. The ensemble method needs only its
runs: :meth:`Problem.ensemble <fips.Problem.ensemble>` draws ``N`` members
from the prior, the model is run on each (outside fips, or by fips when the
problem's forward is an :class:`~fips.Operator` or a
:class:`~fips.ForwardModel`), and ``fips.solve(problem, "ensemble",
members=..., predicted=...)`` updates them with the observations. With an
explicit :math:`H` it approximates what ``fips.solve(problem)`` gives exactly,
with an error that falls as :math:`1/\sqrt{N}`.

With the members' deviations from their mean, scaled, :math:`A = (X -
\bar{x}) / \sqrt{N - 1}` (``n x N``), the modeled observations' :math:`B =
(Y - \bar{y}) / \sqrt{N - 1}` (``m x N``, :math:`Y_{\cdot j} = F(x_j)`),
the innovation :math:`d = z - c - \bar{y}` and :math:`R = S_z / \gamma`, the
ensemble's prior covariance is :math:`P = A A^T` and the update is one batch
update with every observation (a deterministic square-root filter, EnSRF:
Whitaker and Hamill, 2002):

.. math::

    \bar{x}^a = \bar{x} + K d, \qquad
    K = P_{xy} (P_{yy} + R)^{-1}, \qquad
    A^a = A - \tilde{K} B,

with :math:`P_{xy} = A B^T` and :math:`P_{yy} = B B^T`, each multiplied
element by element by the user's localization weights when they are given
(:math:`\rho_{xy} \circ P_{xy}`, :math:`\rho_{yy} \circ P_{yy}`). The
observations are first whitened by :math:`S_z = L L^T` (its Cholesky factor,
or the root of its diagonal), so that the mismatch is :math:`I / \gamma`;
then with :math:`M = L^{-1} P_{yy} L^{-T} = Q \Lambda Q^T`, the reduced gain
of the batch EnSRF (its symmetric square roots, Andrews 1968) is

.. math::

    \tilde{K} = P_{xy} L^{-T} \, G^{-1/2} (G^{1/2} + \gamma^{-1/2} I)^{-1},
    \qquad G = M + I / \gamma,

so that :math:`(I - \tilde{K} H) P (I - \tilde{K} H)^T = (I - K H) P`: the
deviations' covariance after the update is the Kalman posterior of the
ensemble's own :math:`P`, exactly, for any :math:`P` and any :math:`S_z`.
Both roots are functions of :math:`M`, read from its eigendecomposition at
any :math:`\gamma`. Without localization the same update is the ensemble
transform :math:`A^a = A (I + S^T S)^{-1/2}` with :math:`S = \sqrt{\gamma}
L^{-1} B` (the symmetric root, as in the ETKF), which keeps the members'
mean, and fips computes it from the thin SVD :math:`L^{-1} B = U \Sigma V^T`
(:math:`Q = U`, :math:`\Lambda = \Sigma^2`): :math:`O(m N \min(m, N))`, no
``m x m`` or ``N x N`` factorization. With localization the update needs
:math:`\rho_{yy} \circ P_{yy}`'s eigendecomposition (``m x m``) and
:math:`\rho_{xy} \circ P_{xy}` (``n x m``, dense); the deviations' covariance
is then not the localized Kalman posterior (a rank :math:`N - 1` ensemble
cannot hold it), as in every localized square-root filter.

Neither factorization depends on the values of the observations or on
:math:`\gamma`, so ``with_data`` and ``with_gamma`` reuse the modeled
ensemble: :meth:`Posterior.trials <fips.Posterior.trials>` runs the model
once, not once per trial.

The solution (:class:`Ensemble`) answers from the ensemble: the estimate is
the mean of the updated members (``estimate_is_mean``); the variances, the
covariance of aggregates and the gain are those of the deviations, and the
prior's variances (which the reduction of uncertainty is measured against)
those of the prior members, :math:`A A^T`; DOFS,
:math:`\operatorname{tr}(P_{yy} (P_{yy} + R)^{-1})`, the information content
and chi-squared, :math:`d^T (P_{yy} + R)^{-1} d`, are those of the
ensemble's covariances, which equal :math:`H P H^T` with a linear model;
:meth:`Posterior.sample <fips.Posterior.sample>` returns the updated
members themselves. Its estimate is not claimed to be linear in the
observations (``recovery`` refuses; ``trials`` works), and its log evidence
is not claimed at all: the ensemble's chi-squared and information content are
those of a sample covariance, with an error that does not fall below the
number of observations over :math:`N`.

References
----------
Whitaker, J. S., and Hamill, T. M. (2002). Ensemble data assimilation without
perturbed observations. *Monthly Weather Review*, 130(7), 1913-1924.
https://doi.org/10.1175/1520-0493(2002)130<1913:EDAWPO>2.0.CO;2

Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., and Whitaker,
J. S. (2003). Ensemble square root filters. *Monthly Weather Review*, 131(7),
1485-1490. https://doi.org/10.1175/1520-0493(2003)131<1485:ESRF>2.0.CO;2

Houtekamer, P. L., and Mitchell, H. L. (2001). A sequential ensemble Kalman
filter for atmospheric data assimilation. *Monthly Weather Review*, 129(1),
123-137. https://doi.org/10.1175/1520-0493(2001)129<0123:ASEKFF>2.0.CO;2

Andrews, A. (1968). A square root formulation of the Kalman covariance
equations. *AIAA Journal*, 6(6), 1165-1166. https://doi.org/10.2514/3.4696
"""

import dataclasses
import logging
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, eigh, solve_triangular, svd

from fips.linalg import DiagonalOperator, LinearOperator
from fips.linalg.forms import _read_only
from fips.methods.analytic import AnalyticOptions
from fips.operator import Operator
from fips.posterior import Posterior
from fips.problem import Problem
from fips.space import Space

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class EnsembleOptions:
    r"""
    The options of the ``"ensemble"`` method.

    Parameters
    ----------
    gamma : float, default 1.0
        The regularization factor :math:`\gamma`, as for the analytic method
        (:class:`~fips.methods.analytic.AnalyticOptions`): the model-data
        mismatch is divided by it.
    members : int or pandas.DataFrame
        The prior ensemble. A DataFrame is the members
        :meth:`Problem.ensemble <fips.Problem.ensemble>` returned (or any
        others): one row per state element, labeled as ``problem.prior.index``
        (in any order), one column per member (at least two, their labels
        unique). A number draws that many members with
        ``problem.ensemble(members, rng=seed)``. Without `predicted`, the
        problem's forward (an :class:`~fips.Operator`, or a
        :class:`~fips.ForwardModel`'s ``__call__``; no Jacobian is needed)
        is run on each member.
    predicted : pandas.DataFrame, optional
        The model's output for each member, without the problem's offset (as
        ``ForwardModel.__call__`` returns it, :math:`F(x_j)`): one row per
        observation, labeled as ``problem.obs.index`` (in any order), and the
        columns of `members` (in any order). Given, fips runs no model: this
        is the second stage of a model that runs outside Python. It needs
        `members` as a DataFrame, the members it was modeled from.
    localization : Operator, optional
        Weights :math:`\rho_{xy}` on the ensemble's covariance of the state
        with the modeled observations: rows the state's labels, columns the
        observations' (any order, any form). fips computes no distances:
        the weights are yours (Gaspari and Cohn's function of the distance
        between each state element and each observation, say). Without it, no
        localization. It needs `obs_localization`.
    obs_localization : Operator, optional
        Weights :math:`\rho_{yy}` on the ensemble's covariance of the modeled
        observations with each other: rows and columns the observations'
        labels. It must be symmetric and give a positive semidefinite
        :math:`\rho_{yy} \circ P_{yy}` (Gaspari and Cohn's function does).
        It needs `localization`.
    seed : int, default 0
        The seed of the members drawn when `members` is a number.

    Raises
    ------
    ValueError
        If `gamma` is not a positive, finite number; `members` is missing, a
        number below 2 or a DataFrame with fewer than two columns or with
        repeated column labels; `predicted` is given with a number of
        members; only one of `localization` and `obs_localization` is given;
        or `seed` is not a nonnegative integer.
    TypeError
        If `members` is neither an integer nor a DataFrame, `predicted` is not
        a DataFrame, or a localization is not an Operator.
    """

    gamma: float = 1.0
    """The regularization factor: the model-data mismatch is divided by it."""
    members: "int | pd.DataFrame | None" = None
    """The prior members (a DataFrame), or how many to draw."""
    predicted: "pd.DataFrame | None" = None
    """The model's output for each member, without the offset; None runs the problem's forward."""
    localization: Operator | None = None
    r"""The weights :math:`\rho_{xy}` (state x observations), or None."""
    obs_localization: Operator | None = None
    r"""The weights :math:`\rho_{yy}` (observations x observations), or None."""
    seed: int = 0
    """The seed of the members drawn when `members` is a number."""

    def __post_init__(self):
        object.__setattr__(self, "gamma", AnalyticOptions(self.gamma).gamma)
        members = self.members
        if members is None:
            raise ValueError(
                "The ensemble method needs members=: a number of members to draw "
                "from the prior (the problem's forward is run on each), or the "
                "DataFrame problem.ensemble(N) returned, with predicted= the "
                "model's output for each."
            )
        if isinstance(members, pd.DataFrame):
            if members.shape[1] < 2:
                raise ValueError(
                    f"members needs at least two columns, got {members.shape[1]}."
                )
            if members.columns.has_duplicates:
                raise ValueError(
                    "The columns of members are the members' labels, and repeat: "
                    f"{list(members.columns[members.columns.duplicated()][:5])}."
                )
        elif isinstance(members, numbers.Integral) and not isinstance(members, bool):
            if members < 2:
                raise ValueError(
                    f"members must be at least 2 (or a DataFrame), got {members}."
                )
            object.__setattr__(self, "members", int(members))
        else:
            raise TypeError(
                "members is a number of members or the DataFrame of the members, "
                f"not {type(members).__name__}."
            )
        if self.predicted is not None:
            if not isinstance(self.predicted, pd.DataFrame):
                raise TypeError(
                    "predicted is a DataFrame (observations x members), not "
                    f"{type(self.predicted).__name__}."
                )
            if not isinstance(members, pd.DataFrame):
                raise ValueError(
                    "predicted needs members= as the DataFrame of the members it "
                    "was modeled from (problem.ensemble returns it), not a number."
                )
        for name in ("localization", "obs_localization"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Operator):
                raise TypeError(
                    f"{name} is an Operator of weights, not {type(value).__name__}."
                )
        if (self.localization is None) != (self.obs_localization is None):
            raise ValueError(
                "Localization needs both weights: localization= (state x "
                "observations) and obs_localization= (observations x observations)."
            )
        seed = self.seed
        if not isinstance(seed, numbers.Integral) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"seed must be a nonnegative integer, got {seed!r}.")
        object.__setattr__(self, "seed", int(seed))

    def __repr__(self) -> str:
        """Return the options, with the shapes of the DataFrames and Operators instead of their values."""

        def shape(value: Any) -> str:
            """Return the shape of a DataFrame or an Operator, or the repr of anything else."""
            if isinstance(value, pd.DataFrame | Operator):
                rows, cols = value.shape
                return f"<{rows} x {cols} {type(value).__name__}>"
            return repr(value)

        return (
            f"EnsembleOptions(gamma={self.gamma!r}, members={shape(self.members)}, "
            f"predicted={shape(self.predicted)}, "
            f"localization={shape(self.localization)}, "
            f"obs_localization={shape(self.obs_localization)}, seed={self.seed!r})"
        )


@dataclass(frozen=True, eq=False)
class Ensemble:
    r"""
    The ensemble method's solution: the prior ensemble, its modeled observations, and the factors of the update.

    The :class:`~fips.Posterior` asks it for every array it reports: it is
    the method's :class:`~fips.posterior.Solution`. Every answer is written
    from five arrays and the eigenvalues :math:`\lambda_k` (module docstring,
    with :math:`Q` the eigenvectors of :math:`M` in the whitened
    observations): :math:`F = P_{xy} L^{-T} Q` (``n x k``), :math:`E = Q^T
    L^{-1} B` (``k x N``), the whitened innovation :math:`e = Q^T L^{-1} d`
    (``k``) and the part of :math:`L^{-1} d` outside :math:`Q`. With
    :math:`w_k = \gamma / (1 + \gamma \lambda_k)`:

    .. math::

        \bar{x}^a = \bar{x} + F (w \circ e), \qquad
        K = F \operatorname{diag}(w) Q^T L^{-1}, \qquad
        A^a = A - F \operatorname{diag}(c) E,

    with :math:`c_k = \gamma / \big(\sqrt{1 + \gamma\lambda_k} (\sqrt{1 +
    \gamma\lambda_k} + 1)\big)`, the reduced gain's weights. Only :math:`w`
    and :math:`c` depend on :math:`\gamma`, and only :math:`e` on the
    observations' values. Its answers:

    - the estimate is the mean of the updated members, the posterior mean of
      the ensemble (``estimate_is_mean``); the variances and the covariance
      of aggregates are those of the updated deviations, :math:`A^a A^{aT}`;
      :meth:`sample` returns updated members, at most :math:`N`; the prior's
      variances, which the reduction of uncertainty is measured against, are
      those of the prior members, :math:`A A^T`, the prior the update used;
    - the gain is the ensemble's :math:`K`; DOFS is :math:`\sum_k w_k
      \lambda_k = \operatorname{tr}(P_{yy} (P_{yy} + R)^{-1})`, the
      information content :math:`\tfrac{1}{2} \sum_k \ln(1 + \gamma
      \lambda_k)`, and chi-squared :math:`d^T (P_{yy} + R)^{-1} d`: the
      analytic method's formulas with the ensemble's :math:`P_{yy}` for
      :math:`H S_0 H^T`;
    - the data misfit is that of the updated modeled observations,
      :math:`\bar{y} + P_{yy} (P_{yy} + R)^{-1} d` (the modeled observations
      of the estimate, for a linear model), and the state misfit is the
      increment's in the ensemble's prior covariance, chi-squared less
      gamma times the data misfit; Rodgers' four tests are written the same
      way, so their degrees of freedom count at most :math:`N - 1`
      directions for the increments;
    - the diagonal of the averaging kernel, :math:`\operatorname{diag}(K
      H)`, and the averaging kernel and innovation covariance the posterior
      writes, need the problem's :math:`H` (an :class:`~fips.Operator`), and
      raise for a model.

    :meth:`with_gamma` and :meth:`with_data` reuse the factors. The updated
    deviations at each gamma are computed once and shared by the solutions
    ``with_data`` returns (not pickled).
    """

    estimate_is_mean: ClassVar[bool] = True
    """The estimate is the mean of the updated ensemble."""

    no_evidence: ClassVar[str] = (
        "the ensemble's chi-squared and information content are those of the "
        "sample covariance of its members' modeled observations, with an error "
        "that does not fall below the number of observations over the number "
        "of members, so the evidence from them is not claimed. For a linear "
        "problem, the exact evidence is fips.solve(problem).log_evidence."
    )
    """Why an ensemble posterior gives no log evidence."""

    problem: Problem
    """The problem that was solved."""
    gamma: float
    r"""The regularization factor :math:`\gamma` the factors are read at."""
    prior_mean: np.ndarray
    r"""The mean of the prior members, :math:`\bar{x}` (``n``)."""
    deviations: np.ndarray
    r"""The prior members' deviations, scaled: :math:`A = (X - \bar{x}) / \sqrt{N - 1}` (``n x N``)."""
    predicted_mean: np.ndarray
    r"""The mean of the modeled observations, :math:`\bar{y}` (``m``)."""
    obs_root: np.ndarray
    r"""The root of :math:`S_z`: the square roots of a diagonal one (``m``), or its lower Cholesky factor :math:`L` (``m x m``)."""
    eigenvectors: np.ndarray
    r""":math:`Q`, the eigenvectors of :math:`M` with an eigenvalue kept (``m x k``)."""
    eigenvalues: np.ndarray
    r""":math:`\lambda_k`, the eigenvalues of :math:`M = L^{-1} P_{yy} L^{-T}` (``k``), clipped at zero."""
    F: np.ndarray
    r""":math:`F = P_{xy} L^{-T} Q` (``n x k``), with :math:`P_{xy}` localized when weights were given."""
    E: np.ndarray
    r""":math:`E = Q^T L^{-1} B` (``k x N``)."""
    innovation: np.ndarray
    r"""The whitened innovation on the eigenvectors, :math:`e = Q^T L^{-1} d` (``k``)."""
    outside: float
    r""":math:`\lVert L^{-1} d - Q e \rVert^2`, the whitened innovation outside the eigenvectors (where :math:`\lambda = 0`)."""
    localized: bool
    """Whether the update was localized."""
    shared: dict = field(default_factory=dict, kw_only=True, repr=False)
    """The updated deviations by gamma, computed once and shared with the solutions of other observations (not pickled)."""

    # -- the update at gamma -------------------------------------------------------

    def weights(self) -> np.ndarray:
        r"""Return :math:`w_k = \gamma / (1 + \gamma \lambda_k)`."""
        return self.gamma / (1.0 + self.gamma * self.eigenvalues)

    def _reduced(self) -> np.ndarray:
        r"""Return :math:`c_k = \gamma / (\sqrt{1 + \gamma\lambda_k} (\sqrt{1 + \gamma\lambda_k} + 1))`, the weights of the reduced gain."""
        root = np.sqrt(1.0 + self.gamma * self.eigenvalues)
        return self.gamma / (root * (root + 1.0))

    def _updated(self) -> np.ndarray:
        r"""Return the updated deviations :math:`A^a = A - F \operatorname{diag}(c) E` (``n x N``), computed once per gamma."""
        updated = self.shared.get(self.gamma)
        if updated is None:
            updated = self.deviations - self.F @ (self._reduced()[:, None] * self.E)
            updated = self.shared[self.gamma] = _read_only(updated)
        return updated

    def _whitened_basis(self) -> np.ndarray:
        r"""Return :math:`L^{-T} Q` (``m x k``): :math:`K = F \operatorname{diag}(w) (L^{-T} Q)^T`."""
        return _whiten(self.obs_root, self.eigenvectors, transpose=True)

    # -- the Solution ----------------------------------------------------------------

    def estimate(self) -> np.ndarray:
        r"""Return the mean of the updated members, :math:`\bar{x} + F (w \circ e)`."""
        return self.prior_mean + self.F @ (self.weights() * self.innovation)

    def variances(self) -> np.ndarray:
        """Return the variances of the updated members."""
        updated = self._updated()
        return np.einsum("ij,ij->i", updated, updated)

    def cov(self, W: LinearOperator | None = None) -> np.ndarray:
        """Return the covariance of the updated members, ``n x n``, or of their aggregates ``W x``, ``k x k``."""
        updated = self._updated()
        Y = updated if W is None else np.asarray(W @ updated, dtype=float)
        return Y @ Y.T

    def used_prior_variances(self, W: LinearOperator | None = None) -> np.ndarray:
        """Return the variances of the prior members (``n``), or of their aggregates ``W x`` (``k``): the prior the update used, which the reduction of uncertainty is measured against."""
        Y = (
            self.deviations
            if W is None
            else np.asarray(W @ self.deviations, dtype=float)
        )
        return np.einsum("ij,ij->i", Y, Y)

    def gain(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""Return the ensemble's gain, :math:`K = P_{xy} (P_{yy} + R)^{-1}`, ``n x m``, or :math:`W K`."""
        F = self.F if W is None else np.asarray(W @ self.F, dtype=float)
        return (F * self.weights()) @ self._whitened_basis().T

    def dofs(self) -> float:
        r"""Return :math:`\operatorname{tr}(P_{yy} (P_{yy} + R)^{-1}) = \sum_k \gamma\lambda_k / (1 + \gamma\lambda_k)`."""
        return float(np.sum(self.weights() * self.eigenvalues))

    def dofs_per_element(self) -> np.ndarray:
        r"""
        Return :math:`\operatorname{diag}(K H)`, with the problem's :math:`H`.

        :math:`K H = F \operatorname{diag}(w) (H^T L^{-T} Q)^T`: one product
        of :math:`H^T` with ``m x k``. With a linear model and no
        localization its sum is :meth:`dofs`, since :math:`H P_{xy} =
        P_{yy}`.

        Raises
        ------
        TypeError
            If the problem's forward is a model: the averaging kernel needs
            :math:`H`.
        """
        H = self.problem._operator.data
        HtZ = np.asarray(H._transpose() @ self._whitened_basis(), dtype=float)
        return np.einsum("jk,k,jk->j", self.F, self.weights(), HtZ)

    def information(self) -> float:
        r"""Return :math:`\tfrac{1}{2} \ln \lvert I + \gamma M \rvert = \tfrac{1}{2} \sum_k \ln(1 + \gamma\lambda_k)`."""
        return 0.5 * float(np.sum(np.log1p(self.gamma * self.eigenvalues)))

    def chi2(self) -> float:
        r"""Return :math:`d^T (P_{yy} + R)^{-1} d = \gamma \lVert L^{-1} d - Q e \rVert^2 + \sum_k w_k e_k^2`."""
        return self.gamma * self.outside + float(
            np.sum(self.weights() * self.innovation**2)
        )

    def misfits(self) -> tuple[float, float]:
        r"""
        Return the data misfit of the updated modeled observations and the state misfit of the increment, in the ensemble's prior covariance.

        The updated modeled observations are :math:`\bar{y} + P_{yy} (P_{yy}
        + R)^{-1} d`, so the whitened residual is :math:`e_k / (1 +
        \gamma\lambda_k)` on the eigenvectors and the innovation outside
        them: the data misfit is :math:`\lVert L^{-1} d - Q e \rVert^2 +
        \sum_k e_k^2 / (1 + \gamma\lambda_k)^2`. The state misfit is
        :math:`\sum_k \gamma^2 \lambda_k e_k^2 / (1 + \gamma\lambda_k)^2`, so
        that :meth:`chi2` is gamma times the first plus the second (without
        localization, the increment :math:`A u`'s :math:`\lVert u \rVert^2`
        for the least :math:`u`).
        """
        scale = 1.0 + self.gamma * self.eigenvalues
        e2 = self.innovation**2
        data = self.outside + float(np.sum(e2 / scale**2))
        state = float(np.sum(self.gamma**2 * self.eigenvalues * e2 / scale**2))
        return data, state

    def chi2_tests(self) -> dict[str, tuple[float, int]]:
        r"""
        Return Rodgers' four chi-squared tests, with the ensemble's covariances.

        As the analytic method writes them, with the ensemble's
        :math:`P_{yy}` for :math:`H S_0 H^T`: ``innovation`` and
        ``residual`` are :meth:`chi2` with :math:`m` degrees of freedom; the
        increments are :math:`\sum_k w_k e_k^2` over the eigenvalues above
        roundoff, with their number (at most :math:`N - 1`) as the degrees of
        freedom.
        """
        chi2, m = self.chi2(), len(self.problem.obs)
        eigenvalues = self.eigenvalues
        roundoff = len(eigenvalues) * np.finfo(float).eps * eigenvalues.max(initial=0.0)
        kept = eigenvalues > roundoff
        increment = float(np.sum((self.weights() * self.innovation**2)[kept]))
        rank = int(kept.sum())
        return {
            "innovation": (chi2, m),
            "residual": (chi2, m),
            "modeled_increment": (increment, rank),
            "increment": (increment, rank),
        }

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """
        Return `N` of the updated members (``n x N``), chosen without replacement by `rng`, in the members' order.

        Raises
        ------
        ValueError
            If `N` is more than the members.
        """
        members = self.deviations.shape[1]
        if members < N:
            raise ValueError(
                f"The ensemble has {members} members, and its draws are its "
                f"members: ask for at most {members}, not {N}."
            )
        chosen = np.sort(rng.choice(members, size=N, replace=False))
        scale = math.sqrt(members - 1)
        return self.estimate()[:, None] + scale * self._updated()[:, chosen]

    def with_gamma(self, gamma: float) -> "Ensemble":
        """Return the same ensemble's update at `gamma`: the factors do not depend on it."""
        return dataclasses.replace(self, gamma=gamma)

    def with_data(self, problem: Problem) -> "Ensemble":
        """Return the update of `problem`, this one's with other observation values: the same members and modeled observations, the innovation alone computed again."""
        innovation, outside = _innovation(
            problem, self.predicted_mean, self.obs_root, self.eigenvectors
        )
        return dataclasses.replace(
            self, problem=problem, innovation=innovation, outside=outside
        )

    def __getstate__(self) -> dict:
        """Return the fields for a pickle or a copy, without the shared updated deviations."""
        state = dict(self.__dict__)
        state["shared"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle or a copy, the arrays read-only again."""
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                _read_only(value)


def _whiten(root: np.ndarray, X: np.ndarray, transpose: bool = False) -> np.ndarray:
    r"""Return :math:`L^{-1} X`, or :math:`L^{-T} X` with `transpose`, for the root of :math:`S_z` (``Ensemble.obs_root``)."""
    if root.ndim == 1:
        return X / (root[:, None] if X.ndim == 2 else root)
    trans = "T" if transpose else "N"
    # np.asarray: SciPy is not annotated, and gives no array type
    return np.asarray(solve_triangular(root, X, lower=True, trans=trans))


def _obs_root(S_z: LinearOperator) -> np.ndarray:
    """
    Return the root of the model-data mismatch: the square roots of a diagonal one, or the Cholesky factor of any other, made dense.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the mismatch is not positive-definite.
    """
    if isinstance(S_z, DiagonalOperator):
        if not np.all(S_z.values > 0):
            raise LinAlgError(
                "The model-data mismatch has a variance that is not positive, so "
                "it cannot whiten the observations."
            )
        return np.sqrt(np.asarray(S_z.values, dtype=float))
    try:
        # np.asarray: SciPy is not annotated, and gives no array type
        return np.asarray(cholesky(S_z.dense(), lower=True))
    except LinAlgError as err:
        raise LinAlgError(
            "The model-data mismatch is not positive-definite, so it cannot "
            "whiten the observations."
        ) from err


def _innovation(
    problem: Problem, predicted_mean: np.ndarray, root: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, float]:
    r"""Return :math:`e = Q^T L^{-1} d` and :math:`\lVert L^{-1} d - Q e \rVert^2`, for :math:`d = z - c - \bar{y}`."""
    d = problem.obs.values - problem._offset_values() - predicted_mean
    whitened = _whiten(root, d)
    e = Q.T @ whitened
    outside = float(np.sum((whitened - Q @ e) ** 2))
    return _read_only(e), outside


def _frame_values(
    frame: pd.DataFrame, space: Space, columns: pd.Index | None, what: str
) -> tuple[np.ndarray, pd.Index]:
    """
    Return the values of `frame` with its rows in the order of `space` (and its columns in that of `columns`), and its columns.

    Raises
    ------
    ValueError
        If the rows do not have exactly the space's labels, the columns not
        exactly `columns`, or a value is not finite.
    """
    try:
        given = Space.from_index(frame.index)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"The rows of {what} are not labels fips can read ({error}); label "
            "them as the problem's are (problem.prior.index, problem.obs.index)."
        ) from None
    rows = space._align(
        given, f"the rows of {what}", f"The rows of {what} and the problem's"
    )
    values = frame.to_numpy(dtype=float)
    if columns is not None:
        if frame.columns.has_duplicates or not frame.columns.sort_values().equals(
            columns.sort_values()
        ):
            only_here = frame.columns.difference(columns)
            only_there = columns.difference(frame.columns)
            raise ValueError(
                f"The columns of {what} must be the members' labels, each once: "
                f"{list(only_here[:5])} only in {what}, {list(only_there[:5])} "
                "only in the members."
            )
        values = values[:, frame.columns.get_indexer(columns)]
    else:
        columns = frame.columns
    values = values[rows]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{what} has values that are not finite.")
    return values, columns


def _members(
    problem: Problem, options: EnsembleOptions
) -> tuple[np.ndarray, np.ndarray]:
    """Return the prior members (``n x N``) and their modeled observations (``m x N``), in the problem's orders."""
    members = options.members
    if isinstance(members, pd.DataFrame):
        X, columns = _frame_values(members, problem.prior.space, None, "members")
    else:
        assert isinstance(members, int)  # EnsembleOptions checked it
        frame = problem.ensemble(members, rng=options.seed)
        X, columns = frame.to_numpy(), frame.columns
    if options.predicted is not None:
        Y, _ = _frame_values(options.predicted, problem.obs.space, columns, "predicted")
        return X, Y
    forward = problem.forward
    if isinstance(forward, Operator):
        return X, np.asarray(forward.data @ X, dtype=float)
    logger.info(
        "Running the forward model %s on %d members...",
        type(forward).__name__,
        X.shape[1],
    )
    Y = np.column_stack([problem._apply(X[:, j]) for j in range(X.shape[1])])
    if not np.all(np.isfinite(Y)):
        raise ValueError("The forward model gave values that are not finite.")
    return X, Y


def _weights(
    problem: Problem, options: EnsembleOptions
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the localization weights :math:`\rho_{xy}` (``n x m``) and :math:`\rho_{yy}` (``m x m``), dense, in the problem's orders."""
    state, observed = problem.prior.space, problem.obs.space
    assert options.localization is not None and options.obs_localization is not None
    rho_xy = options.localization._on(
        state, observed, "localization", ("the prior", "the observations")
    ).dense()
    rho_yy = options.obs_localization._on(
        observed,
        observed,
        "obs_localization",
        ("the observations", "the observations"),
    ).dense()
    if not np.allclose(rho_yy, rho_yy.T, rtol=0.0, atol=1e-12 * np.abs(rho_yy).max()):
        raise ValueError("obs_localization must be symmetric.")
    return np.asarray(rho_xy, dtype=float), np.asarray(rho_yy, dtype=float)


def _factor(problem: Problem, options: EnsembleOptions) -> Ensemble:
    """Draw or read the members, run or read their modeled observations, and factor the update."""
    X, Y = _members(problem, options)
    N = X.shape[1]
    scale = math.sqrt(N - 1)
    x_bar, y_bar = X.mean(axis=1), Y.mean(axis=1)
    A = (X - x_bar[:, None]) / scale
    B = (Y - y_bar[:, None]) / scale
    root = _obs_root(problem.obs_cov.data)
    B_white = _whiten(root, B)  # L^{-1} B, m x N
    if options.localization is None:
        # The thin SVD of L^{-1} B: M = U S^2 U^T, P_xy L^{-T} U = A V S
        U, s, Vt = svd(B_white, full_matrices=False)
        Q, s, Vt = np.asarray(U), np.asarray(s, dtype=float), np.asarray(Vt)
        eigenvalues = s**2
        F = A @ (Vt.T * s)
        E = s[:, None] * Vt
    else:
        rho_xy, rho_yy = _weights(problem, options)
        P_yy = _whiten(root, rho_yy * (B @ B.T))  # L^{-1} P_yy
        M = _whiten(root, P_yy.T).T  # L^{-1} P_yy L^{-T}
        M = 0.5 * (M + M.T)
        eigenvalues, Q = eigh(M)
        eigenvalues, Q = np.asarray(eigenvalues), np.asarray(Q)
        largest = np.abs(eigenvalues).max(initial=0.0)
        if eigenvalues.min(initial=0.0) < -math.sqrt(np.finfo(float).eps) * largest:
            raise ValueError(
                "The localized covariance of the modeled observations has a "
                f"negative eigenvalue ({eigenvalues.min():.3g}, the largest "
                f"{largest:.3g}): obs_localization must give a positive "
                "semidefinite weighting (Gaspari and Cohn's function of the "
                "distance does)."
            )
        eigenvalues = np.maximum(eigenvalues, 0.0)
        P_xy = rho_xy * (A @ B.T)  # n x m
        F = _whiten(root, P_xy.T).T @ Q  # P_xy L^{-T} Q
        E = Q.T @ B_white
    innovation, outside = _innovation(problem, y_bar, root, Q)
    return Ensemble(
        problem=problem,
        gamma=options.gamma,
        prior_mean=_read_only(x_bar),
        deviations=_read_only(A),
        predicted_mean=_read_only(y_bar),
        obs_root=_read_only(root),
        eigenvectors=_read_only(Q),
        eigenvalues=_read_only(eigenvalues),
        F=_read_only(np.ascontiguousarray(F)),
        E=_read_only(np.ascontiguousarray(E)),
        innovation=innovation,
        outside=outside,
        localized=options.localization is not None,
    )


def _ensemble(problem: Problem, options: EnsembleOptions) -> Posterior:
    """Update the prior ensemble with the observations."""
    return Posterior(problem, _factor(problem, options), options)
