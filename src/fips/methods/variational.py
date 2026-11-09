r"""
The variational method: the posterior of a problem too large to factor, from products alone.

``fips.solve`` factors an ``m x m`` or an ``n x n`` matrix and forms
:math:`H S_0` (``m x n``). With an hourly state on a fine grid both ``n``
and ``m`` can pass :math:`10^5`, and then none of them fits. This method
needs only products: :math:`H v` and :math:`H^T w` (a sparse :math:`H`, or a
:class:`~fips.linalg.MatrixFreeOperator`, or a forward model's
:meth:`~fips.ForwardModel.jvp` and :meth:`~fips.ForwardModel.vjp`), solves
with :math:`S_z`, and a root :math:`R` of the prior error covariance,
:math:`S_0 = R R^T` (``n x r``, the root a draw is made with, applied
through a Kronecker or block-diagonal form's factors).

**The estimate.** In the state whitened by the prior, :math:`x = x_0 + R
\xi`, the cost is :math:`\xi^T \xi + \gamma (d - H R \xi)^T S_z^{-1} (d - H R
\xi)`, with the innovation :math:`d = z - H x_0 - c`, and its minimum solves
the whitened normal equations

.. math::

    (I + \tilde{H})\, \xi = \gamma R^T H^T S_z^{-1} d, \qquad
    \tilde{H} = \gamma R^T H^T S_z^{-1} H R,

:math:`\tilde{H}` the prior-preconditioned Hessian of the data misfit. Their
matrix is the identity plus a positive semidefinite one, so its eigenvalues
are at least 1, and conjugate gradients (``_cg``) solve them in a number of
products set by how many eigenvalues of :math:`\tilde{H}` stand above 1,
not by ``n``. CG stops when the residual is below ``tol`` times the right
side. With a forward model the same solve is the inner step of Gauss-Newton
(incremental 4D-Var): at each iterate :math:`x_i` the problem linearized
there (:meth:`Problem.linearized <fips.Problem.linearized>`) is solved by
CG, started from :math:`\xi_i`, and the iterations stop by Rodgers' rule, as
``"gauss_newton"``'s do: the step measured in the posterior covariance of
that solve, :math:`d^2 = \Delta\xi^T (I + \tilde{H}_i) \Delta\xi`, below
``outer_tol`` times ``n``. A step that raises the cost is halved until it
does not.

**The uncertainty.** The posterior covariance is :math:`\hat{S} = R (I +
\tilde{H})^{-1} R^T`. Its ``k`` leading eigenpairs :math:`\tilde{H} v_j =
\tilde\lambda_j v_j` (``rank``; by Lanczos, :func:`scipy.sparse.linalg.eigsh`
on the products, or from the dense :math:`\tilde{H}`, made of ``r``
products, when ``k`` is at least half of ``r``) give the low-rank posterior
of Bui-Thanh et al. (2013) and Spantini et al. (2015), as hippylib makes it:

.. math::

    \hat{S}_k = R \big(I - V \operatorname{diag}(\tilde\lambda_j / (1 +
        \tilde\lambda_j))\, V^T\big) R^T = S_0 - G D G^T, \qquad G = R V
        \ (n \times k).

It is the posterior of the ``k`` directions the data inform most, with the
prior left in the others: so each variance is the prior's less the rank-``k``
term, an upper bound of the exact one that falls with each pair added, and
equals it when ``k`` reaches the rank of :math:`\tilde{H}` (at most
``min(m, r)``). The gain, the averaging kernel and DOFS
(:math:`\sum_j \tilde\lambda_j / (1 + \tilde\lambda_j)`, a lower bound), the
information content (:math:`\tfrac{1}{2} \sum_j \ln(1 + \tilde\lambda_j)`)
and draws are those of the same posterior, and none of them forms anything
``n x n``: the covariance of aggregates is :math:`W S_0 W^T - (W G) D (W
G)^T`, a draw :math:`\hat{x} + R (w - V \operatorname{diag}(1 - (1 +
\tilde\lambda_j)^{-1/2}) V^T w)` with :math:`w` standard normal.
:attr:`Posterior.eigenvalues <fips.Posterior.eigenvalues>` lists the pairs and
the DOFS each adds, so a user can see whether ``rank`` was enough; a solve
warns when the last pair still holds more information than the prior
(:math:`\tilde\lambda_k > 1`).

The solution (:class:`Variational`) has ``estimate_is_mean`` for a linear
problem (the estimate is the posterior mean, to the CG tolerance) and not for
a forward model (it is the mode), no ``estimate_is_linear`` (CG's estimate is
linear in the data only to its tolerance, and the gain of ``k`` pairs is not
its gain unless ``k`` is the rank), and ``evidence_is_laplace``: the log
evidence from the cost at the estimate and the information content of the
``k`` pairs (exact at full rank for a linear problem), unless ``rank`` left
out pairs that still hold information (the last kept has an eigenvalue above
1): the pairs left out would add to the information content, so the evidence
is biased high, and :attr:`Posterior.log_evidence <fips.Posterior.log_evidence>`
refuses it. For a linear problem
``with_gamma`` and ``with_data`` cost one CG solve each: the eigenvectors do
not depend on the data, and with gamma only the eigenvalues scale. With a
forward model ``with_data`` solves again from the last linearization, and
there is no ``with_gamma`` (the linearization depends on gamma).

References
----------
Bui-Thanh, T., Ghattas, O., Martin, J., and Stadler, G. (2013). A
computational framework for infinite-dimensional Bayesian inverse problems.
Part I. *SIAM Journal on Scientific Computing*, 35(6), A2494-A2523.
https://doi.org/10.1137/12089586X

Spantini, A., Solonen, A., Cui, T., Martin, J., Tenorio, L., and Marzouk, Y.
(2015). Optimal low-rank approximations of Bayesian linear inverse problems.
*SIAM Journal on Scientific Computing*, 37(6), A2451-A2487.
https://doi.org/10.1137/140977308

Courtier, P., Thépaut, J.-N., and Hollingsworth, A. (1994). A strategy for
operational implementation of 4D-Var, using an incremental approach.
*Quarterly Journal of the Royal Meteorological Society*, 120(519),
1367-1387. https://doi.org/10.1002/qj.49712051912
"""

import dataclasses
import logging
import math
import numbers
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from fips.linalg import LinearOperator
from fips.linalg.forms import _read_only
from fips.methods.analytic import AnalyticOptions
from fips.operator import Operator
from fips.posterior import Posterior, UnavailableError
from fips.problem import Problem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariationalOptions:
    r"""
    The options of the ``"variational"`` method.

    Parameters
    ----------
    gamma : float, default 1.0
        The regularization factor :math:`\gamma`, as for the analytic method
        (:class:`~fips.methods.analytic.AnalyticOptions`): the model-data
        mismatch is divided by it.
    rank : int, default 100
        The number of leading eigenpairs of the prior-preconditioned Hessian
        kept for the low-rank posterior (at most ``min(m, r)``, the most it
        has). The variances, the covariance of aggregates, DOFS and the
        draws are those of the ``rank`` directions the data inform most, with
        the prior in the others: exact when ``rank`` reaches the problem's
        rank, conservative (variances too large, DOFS too small) below it.
        Each pair costs about two products with :math:`H` and :math:`H^T`.
    tol : float, default 1e-8
        The tolerance of the iterative solves: conjugate gradients stop when
        the residual of the whitened normal equations is below ``tol`` times
        their right side, and Lanczos when each eigenvalue is good to ``tol``
        relative (``eigsh``'s ``tol``). The error of the estimate, measured in
        the posterior covariance, is at most ``tol`` times the length of the
        right side. Below 1: the prior, where CG starts, is at 1.
    max_cg_iterations : int, default 1000
        The most conjugate-gradient iterations of one solve; a solve that has
        not reached ``tol`` by then raises.
    max_iterations : int, default 50
        For a forward model, the most Gauss-Newton steps (each a CG solve of
        the problem linearized at the last estimate), as for
        ``"gauss_newton"``; a solve that has not stopped by then raises.
    outer_tol : float, default 1e-6
        For a forward model, the step at which the Gauss-Newton steps stop,
        :math:`d^2 <` ``outer_tol`` :math:`\cdot\, n`, Rodgers' rule as
        ``"gauss_newton"``'s ``tol`` (its default too).

    Raises
    ------
    ValueError
        If `gamma` is not a positive, finite number, `rank`,
        `max_cg_iterations` or `max_iterations` is not a positive integer,
        `tol` or `outer_tol` is not a positive, finite number, or `tol` is
        not below 1.
    """

    gamma: float = 1.0
    """The regularization factor: the model-data mismatch is divided by it."""
    rank: int = 100
    """The number of eigenpairs of the low-rank posterior."""
    tol: float = 1e-8
    """The relative tolerance of the conjugate gradients and of Lanczos."""
    max_cg_iterations: int = 1000
    """The most conjugate-gradient iterations of one solve."""
    max_iterations: int = 50
    """For a forward model, the most Gauss-Newton steps."""
    outer_tol: float = 1e-6
    r"""For a forward model, the steps stop when :math:`d^2 <` ``outer_tol`` :math:`\cdot\, n`."""

    def __post_init__(self):
        object.__setattr__(self, "gamma", AnalyticOptions(self.gamma).gamma)
        for name in ("rank", "max_cg_iterations", "max_iterations"):
            value = getattr(self, name)
            if (
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
            object.__setattr__(self, name, int(value))
        for name in ("tol", "outer_tol"):
            value = getattr(self, name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a number, got {value!r}.") from None
            if not (math.isfinite(number) and number > 0):
                raise ValueError(f"{name} must be positive and finite, got {number}.")
            object.__setattr__(self, name, number)
        if self.tol >= 1.0:
            raise ValueError(
                f"tol must be below 1, got {self.tol}: it is the residual of the "
                "conjugate gradients relative to the right side, and the start, "
                "the prior, has a relative residual of 1 (tol=1 returns the prior)."
            )


class _System:
    r"""
    The whitened normal equations of a linear problem, by products: :math:`(I + \tilde{H}) \xi = b`, :math:`\tilde{H} = \gamma R^T H^T S_z^{-1} H R`.

    It holds the problem's forms (``R`` the prior's root, ``n x r``) and the
    solve with :math:`S_z`, factored once, and counts the products with
    :math:`\tilde{H}` it makes.
    """

    def __init__(self, linear: Problem, gamma: float):
        self.problem = linear
        self.gamma = gamma
        R = linear.prior_cov.data._root
        self.R, self.R_T = R, R._transpose()
        H = linear._operator.data
        self.H, self.H_T = H, H._transpose()
        self.solve_S_z = linear.obs_cov.data._solver()
        self.products = 0

    @property
    def r(self) -> int:
        """The length of the whitened state, the columns of the prior's root."""
        return self.R.shape[1]

    def whiten(self, D: np.ndarray) -> np.ndarray:
        r"""Return :math:`\gamma R^T H^T S_z^{-1} D` (``r x q``) for observation-space columns `D` (``m x q``)."""
        Y = np.asarray(self.solve_S_z(D), dtype=float)
        return self.gamma * np.asarray(self.R_T @ (self.H_T @ Y), dtype=float)

    def modeled(self, V: np.ndarray) -> np.ndarray:
        """Return :math:`H R V` (``m x q``) for whitened columns `V` (``r x q``)."""
        return np.asarray(self.H @ np.asarray(self.R @ V, dtype=float), dtype=float)

    def hessian(self, V: np.ndarray) -> np.ndarray:
        r"""Return :math:`\tilde{H} V` (``r x q``) for whitened columns `V`."""
        self.products += V.shape[1]
        return self.whiten(self.modeled(V))

    def weighted_norm(self, Y: np.ndarray) -> float:
        r"""Return :math:`y^T S_z^{-1} y` for an observation-space vector `y`."""
        return float(Y @ np.asarray(self.solve_S_z(Y), dtype=float))

    def innovation(self) -> np.ndarray:
        """Return the innovation :math:`d = z - H x_0 - c` of the problem."""
        problem = self.problem
        return problem.obs.values - problem.predict(problem.prior).values


class _Hessian(LinearOperator):
    r"""The prior-preconditioned Hessian :math:`\tilde{H}` of a system as a (symmetric) form, for Lanczos."""

    def __init__(self, system: _System):
        self._system = system
        super().__init__((system.r, system.r))

    def _matmat(self, X):
        r"""Return :math:`\tilde{H} X`."""
        return self._system.hessian(np.asarray(X, dtype=float))

    def _transpose(self):
        r"""Return itself: :math:`\tilde{H}` is symmetric."""
        return self


def _cg(
    system: _System, b: np.ndarray, start: np.ndarray, tol: float, max_iterations: int
) -> tuple[np.ndarray, int]:
    r"""
    Return the solution of :math:`(I + \tilde{H}) \xi = b` by conjugate gradients from `start`, and the iterations taken.

    It stops when the residual's length is at most `tol` times `b`'s; a
    start that already is (a Gauss-Newton iterate at the mode) takes none.
    The matrix is the identity plus a positive semidefinite one, so every
    search direction has positive curvature.

    Raises
    ------
    RuntimeError
        If the residual is above the tolerance after `max_iterations`.
    """
    size = float(np.linalg.norm(b))
    if size == 0.0:
        return np.zeros_like(b), 0
    target = tol * size
    xi = np.array(start, dtype=float)
    r = b - xi - system.hessian(xi[:, None])[:, 0] if xi.any() else np.array(b)
    rr = float(r @ r)
    if math.sqrt(rr) <= target:
        return xi, 0
    p = r.copy()
    for k in range(1, max_iterations + 1):
        Ap = p + system.hessian(p[:, None])[:, 0]
        alpha = rr / float(p @ Ap)
        xi += alpha * p
        r -= alpha * Ap
        rr_new = float(r @ r)
        if math.sqrt(rr_new) <= target:
            return xi, k
        p = r + (rr_new / rr) * p
        rr = rr_new
    raise RuntimeError(
        f"The conjugate gradients did not converge in {max_iterations} iterations: "
        f"the residual is {math.sqrt(rr) / size:.3g} of the right side, above "
        f"tol = {tol:.3g}. Raise max_cg_iterations, or tol; a prior-preconditioned "
        "Hessian with very large eigenvalues (observations far more precise than "
        "the prior) converges slowly."
    )


def _shrinkage(eigenvalues: np.ndarray) -> np.ndarray:
    r"""Return :math:`\tilde\lambda / (1 + \tilde\lambda)`, how much of the prior's variance along each eigenvector the data remove (each pair's DOFS)."""
    return eigenvalues / (1.0 + eigenvalues)


def _eigenpairs(
    system: _System, rank: int, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Return the leading eigenvalues of :math:`\tilde{H} / \gamma` (the gamma-free matrix :math:`N`, from the largest) and their eigenvectors (``r x k``), with ``k`` at most `rank`.

    ``k`` is at most ``min(m, r)``, the most nonzero eigenvalues
    :math:`\tilde{H}` has. When it is at least half of ``r``, :math:`\tilde{H}`
    is formed from ``r`` products and eigendecomposed (``eigh``); otherwise
    Lanczos finds the pairs from products (``eigsh``, ``which="LA"``, a
    seeded start so that a solve is reproducible). Eigenvalues at roundoff of
    the largest (below ``r * eps`` of it, or for Lanczos ``tol`` of it) are
    dropped: they are directions the data do not see.

    Raises
    ------
    RuntimeError
        If Lanczos does not converge.
    """
    r, m = system.r, system.H.shape[0]
    k = min(rank, r, m)
    if k == 0:
        return np.zeros(0), np.zeros((r, 0))
    if 2 * k >= r:
        Ht = system.hessian(np.eye(r))
        values, vectors = (np.asarray(a) for a in eigh(0.5 * (Ht + Ht.T)))
        eigenvalues, vectors = values[::-1][:k], vectors[:, ::-1][:, :k]
        floor = r * np.finfo(float).eps
    else:
        start = np.random.default_rng(0).standard_normal(r)
        try:
            values, vectors = (
                np.asarray(a)
                for a in eigsh(_Hessian(system), k=k, which="LA", tol=tol, v0=start)
            )
        except ArpackNoConvergence as error:
            raise RuntimeError(
                f"Lanczos did not find the {k} leading eigenpairs of the "
                "prior-preconditioned Hessian; lower rank, or raise tol."
            ) from error
        order = np.argsort(values)[::-1]
        eigenvalues, vectors = values[order], vectors[:, order]
        floor = max(r * np.finfo(float).eps, tol)
    kept = eigenvalues > floor * eigenvalues.max(initial=0.0)
    return (
        np.ascontiguousarray(eigenvalues[kept]) / system.gamma,
        np.ascontiguousarray(vectors[:, kept]),
    )


_UNRESOLVED = 1.0
r"""The eigenvalue :math:`\tilde\lambda` of the last pair kept above which a solve warns that ``rank`` was too small: that direction still holds more information from the data than from the prior."""


def _unresolved(eigenvalues: np.ndarray, limited: bool) -> bool:
    r"""
    Return whether ``rank`` left out pairs that still hold information: the pairs were `limited` by ``rank`` (below ``min(m, r)``, the most the Hessian has), and the last kept has :math:`\tilde\lambda >` :data:`_UNRESOLVED`.

    When every pair was computed (``rank`` at least ``min(m, r)``), those
    dropped are at roundoff and the low-rank posterior is the exact one,
    however few pairs are kept.
    """
    return limited and len(eigenvalues) > 0 and bool(eigenvalues[-1] > _UNRESOLVED)


def _report(eigenvalues: np.ndarray, limited: bool) -> None:
    """Log the pairs kept and the DOFS the last tenth of them added, and warn when ``rank`` left out pairs that still hold information (`limited`: ``rank`` was below the most pairs the Hessian has)."""
    k = len(eigenvalues)
    if not k:
        logger.info(
            "The low-rank posterior has no pairs: the data inform no direction."
        )
        return
    dofs = _shrinkage(eigenvalues)
    last = max(1, k // 10)
    logger.info(
        "The low-rank posterior has %d pairs, DOFS at least %.4g; the last %d added "
        "%.3g of it.",
        k,
        float(dofs.sum()),
        last,
        float(dofs[-last:].sum()),
    )
    if _unresolved(eigenvalues, limited):
        logger.warning(
            "The last of the %d pairs of the low-rank posterior still has an "
            "eigenvalue of %.3g (it adds %.2f to DOFS): the data inform more "
            "directions than rank keeps, so the variances are too large, DOFS "
            "too small and the log evidence biased high (it is refused). Raise "
            "rank (see post.eigenvalues).",
            k,
            float(eigenvalues[-1]),
            float(dofs[-1]),
        )


@dataclass(frozen=True, eq=False)
class Variational:
    r"""
    The variational method's solution: the estimate by conjugate gradients and the low-rank posterior.

    It keeps the problem (``problem``), the linear problem the posterior is
    of (``linear``: the problem itself, or the problem linearized at the last
    Gauss-Newton iterate, one step from the mode), the whitened estimate
    :math:`\xi` (:math:`\hat{x} = x_0 + R \xi`), the ``k`` leading
    eigenvalues of :math:`N = \tilde{H} / \gamma` and their eigenvectors
    :math:`V` (``r x k``), :math:`G = R V` (``n x k``), and each step's
    :math:`d^2`, cost and CG iterations. The :class:`~fips.Posterior` asks it
    for every array it reports (the module's docstring says how each is
    made); nothing it holds or computes is ``n x n`` unless asked for
    (``cov()`` without an aggregation, ``gain()`` is ``n x m``).
    """

    problem: Problem
    """The problem solved."""
    linear: Problem
    """The linear problem the low-rank posterior is of: the problem, or its linearization at :attr:`point`."""
    gamma: float
    r"""The regularization factor :math:`\gamma`."""
    xi: np.ndarray
    r"""The whitened estimate, :math:`\xi` with :math:`\hat{x} = x_0 + R \xi` (``r``)."""
    mode: np.ndarray
    r"""The estimate :math:`\hat{x}`: the posterior mean of a linear problem, the mode with a forward model."""
    point_xi: np.ndarray
    r"""The whitened state of the last linearization (:math:`\xi` of :attr:`mode` for a linear problem)."""
    eigenvalues: np.ndarray
    r"""The ``k`` leading eigenvalues of :math:`N = R^T H^T S_z^{-1} H R`, from the largest; :math:`\tilde\lambda = \gamma \lambda`."""
    V: np.ndarray
    """Their eigenvectors, orthonormal (``r x k``)."""
    G: np.ndarray
    """:math:`R V` (``n x k``)."""
    d2: np.ndarray
    r"""Each step's :math:`d^2`, the step in the metric of the posterior covariance of its solve."""
    cost: np.ndarray
    """The cost after each step."""
    cg_iterations: np.ndarray
    """The conjugate-gradient iterations of each step's solve."""
    data_misfit: float
    r"""The data misfit at the estimate, :math:`r^T S_z^{-1} r`."""
    state_misfit: float
    r"""The state misfit at the estimate, :math:`\xi^T \xi`."""
    options: VariationalOptions
    """The options it solved with."""

    def __getstate__(self) -> dict:
        """Return the fields for a pickle or a copy, without the cached system of products."""
        return {k: v for k, v in self.__dict__.items() if k != "_system_"}

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle or a copy, the arrays read-only again."""
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                _read_only(value)

    @property
    def estimate_is_mean(self) -> bool:
        """True for a linear problem, whose posterior mean the estimate is (to the CG tolerance); False with a forward model (the mode)."""
        return isinstance(self.problem.forward, Operator)

    def _unresolved(self) -> bool:
        """Whether ``rank`` left out pairs that still hold information (``_unresolved``), at this solution's gamma."""
        limited = self.options.rank < min(self.V.shape[0], len(self.linear.obs))
        return _unresolved(self._tilde(), limited)

    @property
    def evidence_is_laplace(self) -> bool:
        r"""
        Whether the log evidence is Laplace's, the cost at the estimate and the information content of the low-rank posterior: unless ``rank`` left out pairs that still hold information.

        The information content of ``k`` pairs leaves out :math:`\tfrac{1}{2}
        \sum_{j > k} \ln(1 + \tilde\lambda_j)`, so below the problem's rank
        the evidence is biased high by that much. When the last pair kept has
        :math:`\tilde\lambda_k` above 1, the solve warns and the evidence is
        refused. It is exact for a linear problem at full rank.
        """
        return not self._unresolved()

    @property
    def no_evidence(self) -> str:
        """Why a solution whose rank left out informative pairs gives no log evidence."""
        tilde = self._tilde()
        return (
            f"the low-rank posterior keeps {len(tilde)} pairs (rank="
            f"{self.options.rank}), and the last still has an eigenvalue of "
            f"{tilde[-1]:.3g}, above 1: the information content of the pairs "
            "left out is missing, so the evidence at this rank is biased high, "
            "by half the sum of log(1 + eigenvalue) over them. Raise rank until "
            "the last pairs' eigenvalues are well below 1 (post.eigenvalues)."
        )

    def _system(self) -> _System:
        """Return the products of :attr:`linear`, made once (not pickled)."""
        system = self.__dict__.get("_system_")
        if system is None:
            system = self.__dict__["_system_"] = _System(self.linear, self.gamma)
        return system

    def _tilde(self) -> np.ndarray:
        r"""Return :math:`\tilde\lambda = \gamma \lambda`."""
        return self.gamma * self.eigenvalues

    # -- the Solution ------------------------------------------------------------

    def estimate(self) -> np.ndarray:
        """Return the estimate."""
        return self.mode

    def variances(self) -> np.ndarray:
        r"""Return the diagonal of :math:`\hat{S}_k`: the prior's variances less :math:`\sum_j G_{ij}^2 \tilde\lambda_j / (1 + \tilde\lambda_j)`."""
        prior = self.linear.prior_cov.data.diagonal()
        return prior - (self.G**2) @ _shrinkage(self._tilde())

    def cov(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""Return :math:`\hat{S}_k = S_0 - G D G^T` (``n x n``), or :math:`W S_0 W^T - (W G) D (W G)^T` (``k x k``) without it."""
        S_0 = self.linear.prior_cov.data
        D = _shrinkage(self._tilde())
        if W is None:
            prior, WG = S_0.dense(), self.G
        else:
            prior = np.asarray((W @ S_0 @ W._transpose()).dense(), dtype=float)
            WG = np.asarray(W @ self.G, dtype=float)
        S = prior - (WG * D) @ WG.T
        return 0.5 * (S + S.T)

    def _weighted_modeled(self) -> np.ndarray:
        r"""Return :math:`\gamma S_z^{-1} H G` (``m x k``)."""
        system = self._system()
        HG = system.modeled(self.V)
        return self.gamma * np.asarray(system.solve_S_z(HG), dtype=float)

    def gain(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""Return the gain of the low-rank posterior, :math:`K = G \operatorname{diag}(1/(1 + \tilde\lambda)) (\gamma S_z^{-1} H G)^T` (``n x m``), or :math:`W K`."""
        Z = self._weighted_modeled()
        WG = self.G if W is None else np.asarray(W @ self.G, dtype=float)
        return (WG / (1.0 + self._tilde())) @ Z.T

    def dofs(self) -> float:
        r"""Return :math:`\sum_j \tilde\lambda_j / (1 + \tilde\lambda_j)` over the ``k`` pairs."""
        return float(_shrinkage(self._tilde()).sum())

    def dofs_per_element(self) -> np.ndarray:
        r"""Return :math:`\operatorname{diag}(K H) = \sum_j G_{ij} (H^T Z)_{ij} / (1 + \tilde\lambda_j)`, :math:`Z = \gamma S_z^{-1} H G`: ``k`` products with :math:`H^T`."""
        HtZ = np.asarray(self._system().H_T @ self._weighted_modeled(), dtype=float)
        return np.einsum("ij,j,ij->i", self.G, 1.0 / (1.0 + self._tilde()), HtZ)

    def information(self) -> float:
        r"""Return :math:`\tfrac{1}{2} \sum_j \ln(1 + \tilde\lambda_j)` over the ``k`` pairs."""
        return 0.5 * float(np.log1p(self._tilde()).sum())

    def misfits(self) -> tuple[float, float]:
        """Return the data and the state misfits at the estimate."""
        return self.data_misfit, self.state_misfit

    def chi2(self) -> float:
        r"""Return the cost at the estimate, :math:`\gamma` times the data misfit plus the state misfit (for a linear problem, :math:`d^T (H S_0 H^T + S_z/\gamma)^{-1} d` to second order in the CG error)."""
        return self.gamma * self.data_misfit + self.state_misfit

    def _increment_chi2(self) -> tuple[float, int]:
        r"""Return :math:`\sum_j (v_j^T \xi)^2 (1 + \tilde\lambda_j) / \tilde\lambda_j` over the ``k`` pairs, and ``k``: the increment's chi-squared in the directions the observations constrain."""
        tilde = self._tilde()
        components = self.V.T @ self.xi
        return float(np.sum(components**2 * (1.0 + tilde) / tilde)), len(tilde)

    def chi2_tests(self) -> dict[str, tuple[float, int]]:
        r"""
        Return Rodgers' four chi-squared tests: the innovation, the residual, and the increment in the directions of the ``k`` pairs (``k`` degrees of freedom).

        For a linear problem the innovation and the residual are both
        :meth:`chi2` (as for the analytic method). With a forward model the
        innovation is :math:`d^T (J S_0 J^T + S_z/\gamma)^{-1} d` with the
        model's own :math:`d = z - F(x_0) - c` and the Jacobian of the last
        linearization, :math:`\gamma d^T S_z^{-1} d - b^T (I +
        \tilde{H})^{-1} b` by Woodbury (one more CG solve), and the residual
        :math:`u^T (J S_0 J^T + S_z/\gamma) u` with :math:`u = \gamma
        S_z^{-1} r`, as ``"gauss_newton"``'s.
        """
        m = len(self.problem.obs)
        increment, k = self._increment_chi2()
        if self.estimate_is_mean:
            chi2 = self.chi2()
            innovation = residual = chi2
        else:
            system, gamma = self._system(), self.gamma
            problem = self.problem
            z = problem.obs.values
            d = z - problem.predict(problem.prior).values
            b = system.whiten(d[:, None])[:, 0]
            options = self.options
            solved, _ = _cg(
                system, b, np.zeros_like(b), options.tol, options.max_cg_iterations
            )
            innovation = gamma * system.weighted_norm(d) - float(b @ solved)
            r = z - problem._apply(self.mode) - problem._offset_values()
            u = gamma * np.asarray(system.solve_S_z(r), dtype=float)
            RJu = np.asarray(system.R_T @ (system.H_T @ u), dtype=float)
            S_z = problem.obs_cov.data
            residual = float(RJu @ RJu) + float(u @ np.asarray(S_z @ u)) / gamma
        return {
            "innovation": (innovation, m),
            "residual": (residual, m),
            "modeled_increment": (increment, k),
            "increment": (increment, k),
        }

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        r"""Return `N` exact draws of the low-rank posterior (``n x N``): :math:`\hat{x} + R (w - V (c \circ V^T w))`, :math:`c = 1 - (1 + \tilde\lambda)^{-1/2}`, :math:`w` standard normal (``r x N``)."""
        system = self._system()
        w = rng.standard_normal((system.r, N))
        c = 1.0 - 1.0 / np.sqrt(1.0 + self._tilde())
        whitened = w - self.V @ (c[:, None] * (self.V.T @ w))
        return self.mode[:, None] + np.asarray(system.R @ whitened, dtype=float)

    def iterations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return each step's :math:`d^2`, the cost after it, and its CG iterations."""
        return self.d2, self.cost, self.cg_iterations

    def linearized(self) -> Problem:
        """Return the linear problem the posterior is of."""
        return self.linear

    def spectrum(self) -> np.ndarray:
        r"""Return the eigenvalues :math:`\tilde\lambda_j` of the prior-preconditioned Hessian, from the largest."""
        return self._tilde()

    def with_gamma(self, gamma: float) -> "Variational":
        r"""
        Return the solution at another gamma: one CG solve, the eigenvectors kept (:math:`\tilde{H}` is :math:`\gamma N`, so only :math:`\tilde\lambda` scales).

        Raises
        ------
        UnavailableError
            With a forward model, whose linearization depends on gamma: solve
            again.
        """
        if not self.estimate_is_mean:
            raise UnavailableError(
                "The variational solution of a problem with a forward model has no "
                "with_gamma: its linearization depends on gamma. Solve again with "
                'fips.solve(problem, "variational", gamma=...).'
            )
        options = dataclasses.replace(self.options, gamma=gamma)
        return _linear(self.problem, options, keep=self)

    def with_data(self, problem: Problem) -> "Variational":
        """Return the solution of `problem`, this one's with other observation values: one CG solve with the eigenpairs kept (a linear problem), or Gauss-Newton again from the last linearization."""
        if self.estimate_is_mean:
            return _linear(problem, self.options, keep=self)
        return _iterate(problem, self.options, start=self)


def _linear(
    problem: Problem, options: VariationalOptions, keep: Variational | None = None
) -> Variational:
    """Return the solution of a linear problem: one CG solve from the prior, and the eigenpairs (those of `keep`, a solution of the same problem with other data or gamma, when given)."""
    gamma = options.gamma
    system = _System(problem, gamma)
    d = system.innovation()
    b = system.whiten(d[:, None])[:, 0]
    xi, iterations = _cg(
        system, b, np.zeros_like(b), options.tol, options.max_cg_iterations
    )
    modeled = system.modeled(xi[:, None])[:, 0]  # H R xi
    data = system.weighted_norm(d - modeled)
    state = float(xi @ xi)
    d2 = state + gamma * system.weighted_norm(modeled)
    if keep is None:
        eigenvalues, V = _eigenpairs(system, options.rank, options.tol)
        _report(gamma * eigenvalues, options.rank < min(system.r, len(d)))
        G = np.asarray(system.R @ V, dtype=float)
    else:
        eigenvalues, V, G = keep.eigenvalues, keep.V, keep.G
    logger.info(
        "The conjugate gradients took %d iterations (%d products with the "
        "prior-preconditioned Hessian in all).",
        iterations,
        system.products,
    )
    mode = problem.prior.values + np.asarray(system.R @ xi, dtype=float)
    solution = Variational(
        problem=problem,
        linear=problem,
        gamma=gamma,
        xi=_read_only(xi),
        mode=_read_only(mode),
        point_xi=_read_only(np.zeros_like(xi)),
        eigenvalues=_read_only(np.asarray(eigenvalues)),
        V=_read_only(np.asarray(V)),
        G=_read_only(np.asarray(G)),
        d2=_read_only(np.array([d2])),
        cost=_read_only(np.array([gamma * data + state])),
        cg_iterations=_read_only(np.array([iterations])),
        data_misfit=data,
        state_misfit=state,
        options=options,
    )
    solution.__dict__["_system_"] = system
    return solution


_HALVINGS = 30
"""The most times a step is halved before the cost stops rising along it."""

_RISE = 1e-12
"""How much the cost may rise along a step, relative, and the step still be taken (roundoff near the mode)."""

_FAILED = (ValueError, FloatingPointError, OverflowError)
"""What a model raises at a state it cannot model: a step that reaches one is taken shorter."""


def _iterate(
    problem: Problem, options: VariationalOptions, start: Variational | None = None
) -> Variational:
    r"""
    Return the solution of a problem with a forward model: Gauss-Newton steps, each a CG solve of the problem linearized at the last iterate (from the prior, or from the last linearization of `start`), and the eigenpairs at the last.

    Raises
    ------
    RuntimeError
        If the steps have not stopped after ``max_iterations``, or the cost
        rises along a step however short.
    """
    gamma, n = options.gamma, len(problem.prior)
    x_0, z = problem.prior.values, problem.obs.values
    c = problem._offset_values()
    limit = options.outer_tol * n
    root = problem.prior_cov.data._root
    solve_S_z = problem.obs_cov.data._solver()

    def cost(xi: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray] | None:
        """Return the data and state misfits at ``x_0 + R xi``, the state and the model there; None where the model fails."""
        x = x_0 + np.asarray(root @ xi, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                modeled = problem._apply(x)
            except _FAILED:
                return None
        if not np.isfinite(modeled).all():
            return None
        r = z - modeled - c
        data = float(r @ np.asarray(solve_S_z(r), dtype=float))
        return data, float(xi @ xi), x, modeled

    xi = np.zeros(root.shape[1]) if start is None else np.array(start.point_xi)
    here = cost(xi)
    if here is None:
        raise RuntimeError("The forward model cannot model the starting state.")
    current = gamma * here[0] + here[1]
    d2s: list[float] = []
    costs: list[float] = []
    cgs: list[int] = []
    for i in range(1, options.max_iterations + 1):
        _, _, x, modeled = here
        linear = problem._linearized(x, modeled)
        system = _System(linear, gamma)
        b = system.whiten(system.innovation()[:, None])[:, 0]
        new, iterations = _cg(system, b, xi, options.tol, options.max_cg_iterations)
        step = new - xi
        d2 = float(step @ step) + gamma * system.weighted_norm(
            system.modeled(step[:, None])[:, 0]
        )
        converged = d2 < limit
        fraction, taken = 1.0, None
        for _ in range(_HALVINGS):
            trial = cost(xi + fraction * step)
            if trial is not None and (
                converged
                or gamma * trial[0] + trial[1] <= current + _RISE * abs(current)
            ):
                taken = trial
                break
            fraction /= 2
        if taken is None:
            raise RuntimeError(
                f"The cost rises along Gauss-Newton step {i} however short the step: "
                "the Jacobian is not the derivative of the model (check it with "
                "fips.taylortest(problem.forward, x)), or the model does not give "
                "finite observations near the last estimate."
            )
        d2s.append(d2 * fraction**2)
        cgs.append(iterations)
        current = gamma * taken[0] + taken[1]
        costs.append(current)
        logger.debug(
            "Gauss-Newton step %d: %d CG iterations, d2 = %.3g (it stops below %.3g), "
            "cost %.6g.",
            i,
            iterations,
            d2s[-1],
            limit,
            current,
        )
        if converged:
            point_xi = xi
            xi = xi + fraction * step
            eigenvalues, V = _eigenpairs(system, options.rank, options.tol)
            _report(gamma * eigenvalues, options.rank < min(system.r, len(z)))
            logger.info(
                "Gauss-Newton converged in %d steps, %d CG iterations in all: the "
                "last step's d2 is %.3g, below outer_tol * n = %.3g.",
                i,
                sum(cgs),
                d2,
                limit,
            )
            solution = Variational(
                problem=problem,
                linear=linear,
                gamma=gamma,
                xi=_read_only(xi),
                mode=_read_only(np.array(taken[2])),
                point_xi=_read_only(np.array(point_xi)),
                eigenvalues=_read_only(eigenvalues),
                V=_read_only(V),
                G=_read_only(np.asarray(system.R @ V, dtype=float)),
                d2=_read_only(np.array(d2s)),
                cost=_read_only(np.array(costs)),
                cg_iterations=_read_only(np.array(cgs)),
                data_misfit=taken[0],
                state_misfit=taken[1],
                options=options,
            )
            solution.__dict__["_system_"] = system
            return solution
        xi, here = xi + fraction * step, taken
    raise RuntimeError(
        f"Gauss-Newton did not converge in {options.max_iterations} steps: the last "
        f"step's d2 is {d2s[-1]:.3g}, above outer_tol * n = {limit:.3g}. Raise "
        "max_iterations, or check the Jacobian with fips.taylortest(problem.forward, x)."
    )


def _variational(problem: Problem, options: VariationalOptions) -> Posterior:
    """Solve by conjugate gradients and Lanczos: one CG solve for a linear problem, Gauss-Newton steps with CG inside for a forward model."""
    if isinstance(problem.forward, Operator):
        return Posterior(problem, _linear(problem, options), options)
    return Posterior(problem, _iterate(problem, options), options)
