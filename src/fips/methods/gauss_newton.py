r"""
The Gauss-Newton method: the mode of the posterior of a problem with a forward model.

With a forward model :math:`F` (:class:`~fips.ForwardModel`) the observations
are :math:`z = F(x) + c + e`, and the posterior is not Gaussian. Its mode, the
estimate, is the state where the cost

.. math::

    \chi^2(x) = (x - x_0)^T S_0^{-1} (x - x_0)
        + \gamma\, (z - F(x) - c)^T S_z^{-1} (z - F(x) - c)

is least. ``_gauss_newton`` finds it as Rodgers (2000, chapter 5) does:
from the prior, each step solves the problem linearized at the last estimate
(:meth:`Problem.linearized <fips.Problem.linearized>`, with the Jacobian
:math:`J_i` at :math:`x_i`) by the analytic method,

.. math::

    x_{i+1} = x_0 + K_i \big(z - F(x_i) - c + J_i (x_i - x_0)\big),

and it stops when the step is small in the metric of the posterior
covariance of that solve (Rodgers' rule, :math:`d_i^2 \ll n`):

.. math::

    d_i^2 = (x_{i+1} - x_i)^T \hat{S}_i^{-1} (x_{i+1} - x_i)
        < \mathrm{tol} \cdot n, \qquad
    \hat{S}_i^{-1} = S_0^{-1} + \gamma J_i^T S_z^{-1} J_i.

A step that overshoots, so that the cost at its estimate is higher than at
the last one, is halved until it is not (a line search on the cost; the
mode is the same), and only a whole step below the rule stops the
iterations. The rule measures the step, not the distance to the mode: when
the steps fall by a constant factor near 1 (a model that bends much over the
posterior's spread), the solve logs a warning that the estimate may be
further from the mode (``_warn_if_slow``).

For a transform (:class:`~fips.LogNormalForward`,
:class:`~fips.SoftplusForward`), whose second derivatives are diagonal, the
cost's whole Hessian is :math:`\hat{S}_i^{-1} - D_i`, with :math:`D_i =
\gamma \operatorname{diag}(T''(u) \circ H^T S_z^{-1} r)` on the block
(``ForwardModel._curvature``), and each iteration also takes Newton's step,
solved by conjugate gradients preconditioned by :math:`\hat{S}_i`
(``_newton``): it goes to whichever of the two estimates has the lower cost
(each halved as above), and Gauss-Newton's alone where the Newton matrix is
not positive definite. The iterations stop when both steps are below the
rule, with Gauss-Newton's, so the mode is the same and the Gaussian at the
mode is still the analytic solution of the problem linearized at the last
iterate. Newton converges quadratically where Gauss-Newton crawls: on the
user guide's comparison problem (400 state elements, 120 observations, a log
standard deviation of 1, 200 trials) the lognormal model took a median of
7 steps and at most 14, against Gauss-Newton's 9 and 23; the softplus model 9
and 14, against 18 and 153. A model fips did not write has no curvature, and
its steps are Gauss-Newton's.

:math:`\hat{S}_i^{-1}` is never formed. Every estimate is :math:`x_0 + S_0 y`
with :math:`y = J_i^T u_i` (:math:`u_i` the analytic solve's :math:`(J_i S_0
J_i^T + S_z/\gamma)^{-1}` times its innovation), so the prior term of
:math:`d^2` is :math:`\Delta y^T S_0 \Delta y` (no inverse of :math:`S_0`,
which may be singular) and its data term :math:`\gamma (J_i \delta)^T
S_z^{-1} (J_i \delta)`, a solve with the mismatch factored once.

The solution (:class:`GaussNewton`) keeps the analytic solution of the last
linearized problem, the Gaussian at the mode: the analytic posterior of the
problem linearized at the last iterate, one step of :math:`d^2 <
\mathrm{tol} \cdot n` from the mode. Its estimate is the mode, and its gain,
DOFS and information content are those of the Gaussian at the mode. Its
variances, the covariance of aggregates, the mean, the intervals and its draws
are those of the posterior itself, :math:`p(x \mid z) \propto e^{-\chi^2(x)/2}`, drawn by
Hamiltonian Monte Carlo (``_Sampler``): the Gaussian at the mode is its mass
and its exactly solved part, the rest of the cost its kicks, and a Metropolis
test keeps the chain exact. Where the model bends over the posterior's
spread, the Gaussian at the mode is not the posterior (a total of lognormal
values from its draws runs high and covers the truth too rarely); the draws
are. The misfits and the chi-squared are the cost's terms at the mode, from
the model itself. With a forward operator (a linear problem)
``_gauss_newton`` returns the analytic solution itself.

References
----------
Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and
Practice*, chapter 5. World Scientific. https://doi.org/10.1142/3171

Shahbaba, B., Lan, S., Johnson, W. O., and Neal, R. M. (2014). Split
Hamiltonian Monte Carlo. *Statistics and Computing*, 24(3), 339-349.
https://doi.org/10.1007/s11222-012-9373-1

Hoffman, M. D., and Gelman, A. (2014). The No-U-Turn sampler: adaptively
setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning
Research*, 15, 1593-1623.
"""

import logging
import math
import numbers
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.linalg import LinAlgError

from fips.linalg import LinearOperator
from fips.linalg.forms import _read_only
from fips.methods._draws import _warn_if_imprecise
from fips.methods.analytic import AnalyticOptions, Information, Whitened, _factor
from fips.operator import Operator
from fips.posterior import Posterior
from fips.problem import Problem
from fips.vector import Vector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaussNewtonOptions:
    r"""
    The options of the ``"gauss_newton"`` method.

    Parameters
    ----------
    gamma : float, default 1.0
        The regularization factor :math:`\gamma`, as for the analytic method
        (:class:`~fips.methods.analytic.AnalyticOptions`): the model-data
        mismatch is divided by it.
    max_iterations : int, default 50
        The most steps to take. A solve that has not stopped by then raises.
        A mildly nonlinear problem stops in a few; a strongly nonlinear one
        can take tens, its steps' ``d2`` falling by a constant factor.
    tol : float, default 1e-6
        The step at which the iterations stop: when :math:`d^2`, the step
        measured in the posterior covariance, is below ``tol`` times the
        number of state elements :math:`n`. Rodgers' rule asks for
        :math:`d^2 \ll n`; at the default the last step is a thousandth of a
        posterior standard deviation (root mean square), and the estimate is
        within about that of the mode: far inside its uncertainty, for an
        iteration or two more than a looser rule. Each step costs a factoring
        of the problem linearized at the last estimate.
    draws : int, default 2000
        The number of draws of the posterior kept for its variances and the
        covariance of aggregates (Monte Carlo estimates: the relative error of
        a standard deviation is about :math:`\sqrt{\tau / (2\,
        \text{draws}})`, with :math:`\tau` the draws' autocorrelation time,
        about 1 to 3 for most elements, 2% to 3% at 2000). They are made at
        the first use of the variances or covariances, and each trial of
        :meth:`Posterior.trials <fips.Posterior.trials>` makes its own, so
        fewer make trials quicker. There is no setting without draws: the
        Gaussian at the mode, which is not the posterior where the model
        bends, is ``fips.solve(problem.linearized(post.estimate))``.
    burn_in : int, default 200
        The draws made and left out before those kept, from the mode, while
        the sampler tunes its step to an acceptance rate of 0.9.
    seed : int, default 0
        The seed of the tuning and the kept draws, so that a solve is
        reproducible.

    Raises
    ------
    ValueError
        If `gamma` is not a positive, finite number, `max_iterations` is not a
        positive integer, `tol` is not a positive, finite number, `draws` is
        not an integer of at least 2, or `burn_in` or `seed` is not a
        nonnegative one.
    """

    gamma: float = 1.0
    """The regularization factor: the model-data mismatch is divided by it."""
    max_iterations: int = 50
    """The most steps to take."""
    tol: float = 1e-6
    r"""The iterations stop when :math:`d^2 <` ``tol`` :math:`\cdot\, n`."""
    draws: int = 2000
    """The number of draws of the posterior kept for its variances and covariances."""
    burn_in: int = 200
    """The draws left out before those kept, while the sampler tunes its step."""
    seed: int = 0
    """The seed of the tuning and of the kept draws."""

    def __post_init__(self):
        object.__setattr__(self, "gamma", AnalyticOptions(self.gamma).gamma)
        value = self.max_iterations
        if (
            not isinstance(value, numbers.Integral)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(
                f"max_iterations must be a positive integer, got {value!r}."
            )
        object.__setattr__(self, "max_iterations", int(value))
        try:
            tol = float(self.tol)
        except (TypeError, ValueError):
            raise ValueError(f"tol must be a number, got {self.tol!r}.") from None
        if not (math.isfinite(tol) and tol > 0):
            raise ValueError(f"tol must be positive and finite, got {tol}.")
        object.__setattr__(self, "tol", tol)
        for name, least in (("draws", 2), ("burn_in", 0), ("seed", 0)):
            value = getattr(self, name)
            if (
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value < least
            ):
                kind = {2: "an", 0: "a nonnegative"}[least]
                at_least = (
                    " of at least 2 (a variance needs two draws)" if least else ""
                )
                raise ValueError(
                    f"{name} must be {kind} integer{at_least}, got {value!r}."
                )
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True, eq=False)
class GaussNewton:
    r"""
    The Gauss-Newton method's solution: the mode, the Gaussian of the problem linearized at the last iterate, and draws of the posterior.

    It keeps the problem with the forward model (``problem``), the analytic
    solution of the last linearized problem (``analytic``, whose estimate is
    the mode), the state that problem was linearized at, and each step's
    :math:`d^2` and the cost after it. The :class:`~fips.Posterior` asks it
    for every array it reports: it is the method's
    :class:`~fips.posterior.Solution`. The draws of the posterior are made at
    their first use and kept (not pickled: a posterior read back draws them
    again, the same, from the seed). Its answers, by what each needs:

    - the estimate is the mode, not the mean, and is not linear in the
      observations (no ``estimate_is_mean``, no ``estimate_is_linear``);
    - the variances and the covariance of aggregates are those of the kept
      draws of the posterior (:meth:`kept`: ``draws`` of them, after a
      ``burn_in`` that tunes the sampler, from ``seed``), about their mean,
      and so are the posterior's mean and intervals (the draws' quantiles);
      :meth:`sample` draws more;
    - the gain, DOFS and the information content are those of the Gaussian
      at the mode (``analytic``'s); the posterior reads the averaging kernel
      and the innovation covariance from the linearized problem
      (``linearized``);
    - the misfits and the chi-squared are the terms of the cost at the mode,
      with the residual :math:`z - F(\hat{x}) - c` of the model itself;
    - :meth:`with_data` solves again, starting from the last linearization,
      and its draws are its own; there is no ``with_gamma``: the
      linearization depends on gamma.

    Every estimate is :math:`x_0 + S_0 y`, and the solution keeps the
    :math:`y` of the mode and of the last linearization point, so the state
    misfit, :math:`d^2` and the sampler need no inverse of :math:`S_0`.
    """

    evidence_is_laplace: ClassVar[bool] = True
    """Its log evidence is Laplace's: the cost at the mode and the information content of the Gaussian there (exact for a linear model)."""

    problem: Problem
    """The problem with the forward model."""
    analytic: "Whitened | Information"
    """The analytic solution of the problem linearized at :attr:`point`; its estimate is the mode."""
    mode: np.ndarray
    r"""The mode, :math:`\hat{x}`."""
    mode_y: np.ndarray
    r""":math:`y` with :math:`\hat{x} = x_0 + S_0 y`."""
    point: np.ndarray
    """The state of the last linearization, one step from the mode."""
    point_y: np.ndarray
    r""":math:`y` with :attr:`point` :math:`= x_0 + S_0 y`."""
    d2: np.ndarray
    r"""Each step's :math:`d^2`, in the metric of its solve's posterior covariance (a step taken at half its length has a quarter of its whole one's)."""
    cost: np.ndarray
    r"""The cost at each step's estimate, :math:`\gamma` times the data misfit plus the state misfit."""
    data_misfit: float
    r"""The data misfit at the mode, :math:`r^T S_z^{-1} r` with :math:`r = z - F(\hat{x}) - c`."""
    state_misfit: float
    r"""The state misfit at the mode, :math:`(\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0) = y^T S_0 y`."""
    options: GaussNewtonOptions
    """The options it solved with (and :meth:`with_data` solves with)."""

    # -- the draws, kept outside the frozen fields -------------------------------

    def _sampler(self) -> "_Sampler":
        """Return the sampler of this solution's posterior, made once."""
        sampler = self.__dict__.get("_sampler_")
        if sampler is None:
            sampler = self.__dict__["_sampler_"] = _Sampler(self)
        return sampler

    def _tuning(self) -> "_Tuning":
        """Return the sampler's step, tuned over the burn-in from the mode at the first call (its generator a child of the seed's, apart from the kept draws')."""
        tuning = self.__dict__.get("_tuning_")
        if tuning is None:
            (child,) = np.random.SeedSequence(self.options.seed).spawn(1)
            tuning = self._sampler().tune(
                self.options.burn_in, np.random.default_rng(child)
            )
            self.__dict__["_tuning_"] = tuning
            steps = _steps(tuning.step)
            logger.info(
                "The posterior sampler's step is %.3g, %d steps a draw, after a "
                "burn-in of %d draws accepted at %.2f.",
                tuning.step,
                steps,
                self.options.burn_in,
                tuning.acceptance,
            )
            if steps >= _MANY_STEPS:
                logger.warning(
                    "The posterior sampler takes %d steps a draw: the posterior "
                    "bends very much over its spread, or the Jacobian is not the "
                    "derivative of the model (a wrong one leaves the draws exact "
                    "but makes each cost hundreds of steps; check it with "
                    "fips.taylortest(problem.forward, post.estimate)).",
                    steps,
                )
        return tuning

    def kept(self) -> np.ndarray:
        """Return the kept draws (``n x draws``), which the variances, the covariances, the mean and the intervals are read from: drawn from the seed at the first call, with a warning if they mix too slowly for their variances."""
        kept = self.__dict__.get("_draws")
        if kept is None:
            draws = self.options.draws
            kept = self.sample(draws, np.random.default_rng(self.options.seed))
            kept = self.__dict__["_draws"] = _read_only(kept)
            _warn_if_imprecise(kept, logger, _SLOW_DRAWS)
        return kept

    def __getstate__(self) -> dict:
        """Return the fields for a pickle or a copy, without the caches (the draws are made again, the same, from the seed)."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ("_sampler_", "_tuning_", "_draws")
        }

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle or a copy, the arrays read-only again."""
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                _read_only(value)

    # -- the Solution ------------------------------------------------------------

    def estimate(self) -> np.ndarray:
        """Return the mode."""
        return self.mode

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """Return `N` new draws of the posterior (``n x N``): the chain from where the burn-in ended, at its tuned step."""
        tuning = self._tuning()
        draws, _, acceptance = self._sampler().run(tuning.point, N, rng, tuning.step)
        logger.debug("%d draws of the posterior, accepted at %.2f.", N, acceptance)
        return draws

    def variances(self) -> np.ndarray:
        """Return the variances of the kept draws."""
        kept = self.kept()
        centered = kept - kept.mean(axis=1, keepdims=True)
        return np.einsum("ij,ij->i", centered, centered) / (kept.shape[1] - 1)

    def cov(self, W: LinearOperator | None = None) -> np.ndarray:
        """Return the covariance of the kept draws, ``n x n``, or of their aggregates ``W x``, ``k x k``."""
        kept = self.kept()
        Y = kept if W is None else np.asarray(W @ kept, dtype=float)
        centered = Y - Y.mean(axis=1, keepdims=True)
        S = centered @ centered.T / (Y.shape[1] - 1)
        return 0.5 * (S + S.T)

    def gain(self, W: LinearOperator | None = None) -> np.ndarray:
        """Return the gain of the problem linearized at the last iterate, or :math:`W` times it."""
        return self.analytic.gain(W)

    def dofs(self) -> float:
        """Return DOFS of the Gaussian at the mode."""
        return self.analytic.dofs()

    def dofs_per_element(self) -> np.ndarray:
        """Return the diagonal of the averaging kernel at the mode."""
        return self.analytic.dofs_per_element()

    def information(self) -> float:
        """Return the information content of the Gaussian at the mode."""
        return self.analytic.information()

    def misfits(self) -> tuple[float, float]:
        """Return the data and the state misfits at the mode."""
        return self.data_misfit, self.state_misfit

    def chi2(self) -> float:
        r"""Return the cost at the mode, :math:`\gamma` times the data misfit plus the state misfit."""
        return self.analytic.gamma * self.data_misfit + self.state_misfit

    def chi2_tests(self) -> dict[str, tuple[float, int]]:
        r"""
        Return Rodgers' four chi-squared tests at the mode, with the covariances of the Gaussian at the mode.

        With :math:`G = J S_0 J^T + S_z/\gamma` (:math:`J` the Jacobian of the
        last linearization): the innovation :math:`d = z - F(x_0) - c` of the
        model itself gives :math:`d^T G^{-1} d`; the residual :math:`r = z -
        F(\hat{x}) - c` gives :math:`u^T G u`, with :math:`u = \gamma S_z^{-1}
        r`; the increment :math:`\hat{x} - x_0 = S_0 y` is measured in the
        directions the observations constrain, as the analytic method does
        (``_increment_chi2_at``), and the modeled increment with it. The
        chi-squared values are those of the model at the mode; the degrees of
        freedom are those of the linearized problem, so the p-values are
        approximate.
        """
        problem, analytic = self.problem, self.analytic
        gamma, m = analytic.gamma, len(problem.obs)
        z = problem.obs.values
        d = z - problem.predict(problem.prior).values
        innovation = float(d @ analytic._precision_times(d[:, None])[:, 0])
        r = z - problem.predict(Vector(self.mode, problem.prior.space)).values
        S_z = problem.obs_cov.data
        u = gamma * np.asarray(S_z.solve(r), dtype=float)
        Jtu = np.asarray(analytic.problem._operator.data._transpose() @ u)
        residual = (
            float(Jtu @ np.asarray(problem.prior_cov.data @ Jtu))
            + float(u @ np.asarray(S_z @ u)) / gamma
        )
        increment, rank = analytic._increment_chi2_at(self.mode_y)
        return {
            "innovation": (innovation, m),
            "residual": (residual, m),
            "modeled_increment": (increment, rank),
            "increment": (increment, rank),
        }

    def iterations(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Return each step's :math:`d^2` and the cost after it."""
        return self.d2, self.cost

    def linearized(self) -> Problem:
        """Return the problem linearized at the last iterate, which the gain is of."""
        return self.analytic.problem

    def with_data(self, problem: Problem) -> "GaussNewton":
        """
        Return the solution of `problem`, this one's with other observation values: Gauss-Newton again, from the last linearization.

        The first step reuses the factors of the last linearized problem (they
        do not depend on the observations' values), so it costs a product.
        """
        return _iterate(problem, self.options, start=self)


_QUARTER = 0.5 * math.pi
"""The time each draw travels: a quarter period of the Gaussian at the mode, after which a draw of that Gaussian would be independent of its start."""

_TARGET = 0.9
"""The acceptance rate the step is tuned to during the burn-in."""

_JITTER = 0.2
"""Each draw's step is the tuned one times a uniform number within this fraction of 1, so that no path is periodic."""

_MOST_STEPS = 1000
"""The most steps a draw takes: the step is never tuned below a quarter period over this."""

_MANY_STEPS = 200
"""The steps a draw takes at which a solve warns that the posterior bends very much or the Jacobian is wrong."""

_BLOCK = 512
"""The number of velocities the sampler draws in one block."""

_FAILED = (ValueError, FloatingPointError, OverflowError, LinAlgError)
"""What a model (or a transform) raises at a state it cannot model: a path that reaches one is rejected."""


@dataclass(frozen=True)
class _Point:
    r"""A state of the chain with what the next path needs there: the remainder :math:`R(x)`, its gradient :math:`\rho(x)` and :math:`\hat{S} \rho(x)`."""

    x: np.ndarray
    R: float
    rho: np.ndarray
    S_rho: np.ndarray


@dataclass(frozen=True)
class _Tuning:
    """What the burn-in leaves: the tuned step, the state it ended at, and its acceptance rate."""

    step: float
    point: _Point
    acceptance: float


class _Sampler:
    r"""
    Hamiltonian Monte Carlo for the posterior of a problem with a forward model, split at the Gaussian at the mode.

    The posterior is :math:`p(x \mid z) \propto e^{-U(x)}`, :math:`U` half
    the cost. Write it as the Gaussian at the mode times the rest,
    :math:`U = U_G + R`, with :math:`U_G(x) = \tfrac{1}{2} (x - \hat{x})^T
    \hat{S}^{-1} (x - \hat{x})` (:math:`\hat{S}` the covariance of the
    problem linearized at the last iterate, ``analytic``). The momentum's
    mass is :math:`\hat{S}^{-1}`, so a velocity is a draw of :math:`N(0,
    \hat{S})`: the analytic method's exact draws less their mean
    (``analytic.sample``). The dynamics of :math:`U_G` with that mass are a
    rotation, solved exactly, as for the bounded method's sampler:

    .. math::

        x(t) = \hat{x} + (x - \hat{x}) \cos t + v \sin t, \qquad
        v(t) = v \cos t - (x - \hat{x}) \sin t,

    and :math:`R` acts as kicks, :math:`v \leftarrow v - h \hat{S} \nabla R`,
    half a step before and after each rotation by :math:`h` (Strang
    splitting: Shahbaba et al., 2014). The path is symplectic and reversible,
    so a Metropolis test on the change of :math:`H = U + \tfrac{1}{2} v^T
    \hat{S}^{-1} v` makes the chain exact whatever :math:`\hat{S}` and
    :math:`h` are; they set only how fast it mixes. Where the posterior is
    the Gaussian at the mode (a linear model), :math:`\nabla R = 0`, every
    path is accepted, and one step of a quarter period is an exact,
    independent draw.

    :math:`S_0^{-1}` is never needed. With :math:`\hat{x} = x_0 + S_0
    \hat{y}` (``mode_y``) and every state in :math:`x_0` plus the range of
    :math:`S_0` (the velocities and kicks are), in the state whitened by
    :math:`S_0` the remainder is

    .. math::

        R(x) = \hat{y}^T (x - \hat{x}) + \tfrac{\gamma}{2} \big(r^T S_z^{-1} r
            - d^T S_z^{-1} d\big), \qquad r = z - F(x) - c, \quad
            d = J_* (x - \hat{x}),

    up to a constant, with :math:`J_*` the Jacobian of the last
    linearization; its gradient is :math:`\rho = \hat{y} - \gamma J(x)^T
    S_z^{-1} r - \gamma J_*^T S_z^{-1} d`, so a kick is :math:`\hat{S} \rho`
    (``_cov_times``). The kinetic energy is never formed either: a kick of
    :math:`h` changes it by :math:`-h \rho^T v + \tfrac{1}{2} h^2 \rho^T
    \hat{S} \rho`, and a rotation not at all. A singular :math:`S_0` (elements
    with no prior spread) leaves those elements at the mode.

    Each draw starts from the last with a new velocity and travels a quarter
    period, :data:`_QUARTER`, in steps of the tuned :math:`h` (jittered by
    :data:`_JITTER`). The burn-in, from the mode, tunes :math:`h` to an
    acceptance rate of :data:`_TARGET` by dual averaging (Hoffman and
    Gelman, 2014), starting from one step a draw; its draws are left out.
    A path the model cannot follow (it overflows) is rejected.
    """

    def __init__(self, solution: "GaussNewton"):
        problem, analytic = solution.problem, solution.analytic
        self.problem = problem
        self.analytic = analytic
        self.gamma = analytic.gamma
        self.mode = solution.mode
        self.mode_y = solution.mode_y
        self.mean = analytic.estimate()
        J = analytic.problem._operator.data
        self.J, self.J_T = J, J._transpose()
        self.solve_S_z = problem.obs_cov.data._solver()
        self.z = problem.obs.values
        self.c = problem._offset_values()

    def point(self, x: np.ndarray) -> _Point:
        r"""
        Return the state `x` with :math:`R`, :math:`\rho` and :math:`\hat{S} \rho` there.

        Raises
        ------
        ValueError, FloatingPointError, OverflowError, numpy.linalg.LinAlgError
            If the model cannot model `x`, or the result is not finite.
        """
        problem, gamma = self.problem, self.gamma
        with np.errstate(over="raise", invalid="raise"):
            r = self.z - problem._apply(x) - self.c
            dx = x - self.mode
            d = np.asarray(self.J @ dx, dtype=float)
            weighted = np.asarray(self.solve_S_z(np.column_stack([r, d])))
            R = float(self.mode_y @ dx) + 0.5 * gamma * (
                float(r @ weighted[:, 0]) - float(d @ weighted[:, 1])
            )
            rho = self.mode_y - gamma * (
                problem._gradient(x, weighted[:, 0])
                + np.asarray(self.J_T @ weighted[:, 1], dtype=float)
            )
            S_rho = self.analytic._cov_times(rho[:, None])[:, 0]
        if not (math.isfinite(R) and np.isfinite(S_rho).all()):
            raise FloatingPointError("The remainder is not finite.")
        return _Point(x, R, rho, S_rho)

    def travel(
        self, start: _Point, v: np.ndarray, h: float, steps: int
    ) -> tuple[_Point, float]:
        """Return where the path from `start` with the velocity `v` is after `steps` steps of `h`, and the change of the energy along it."""
        mode = self.mode
        cos, sin = math.cos(h), math.sin(h)
        here = start
        change = -0.5 * h * float(here.rho @ v) + 0.125 * h * h * float(
            here.rho @ here.S_rho
        )
        v = v - 0.5 * h * here.S_rho
        for step in range(steps):
            a = here.x - mode
            x = mode + a * cos + v * sin
            v = v * cos - a * sin
            here = self.point(x)
            kick = h if step < steps - 1 else 0.5 * h
            change += -kick * float(here.rho @ v) + 0.5 * kick * kick * float(
                here.rho @ here.S_rho
            )
            v = v - kick * here.S_rho
        return here, change + (here.R - start.R)

    def move(
        self, here: _Point, v: np.ndarray, test: float, h: float, steps: int
    ) -> tuple[_Point, float]:
        """Return the chain's next state from `here` (the path's end if the Metropolis test `test` passes, else `here`) and the path's acceptance probability."""
        try:
            with np.errstate(over="raise", invalid="raise"):
                end, change = self.travel(here, v, h, steps)
        except _FAILED:
            return here, 0.0
        if change <= 0.0:
            alpha = 1.0
        else:  # a NaN change is rejected
            alpha = math.exp(-change) if change < math.inf else 0.0
        return (end if test < alpha else here), alpha

    def moves(
        self, count: int, rng: np.random.Generator
    ) -> Iterator[tuple[np.ndarray, float, float]]:
        """Yield `count` velocities, each with the uniform number of its test and its jitter, drawn from `rng` in blocks of :data:`_BLOCK`."""
        done = 0
        while done < count:
            block = min(_BLOCK, count - done)
            V = self.analytic.sample(block, rng) - self.mean[:, None]
            tests = rng.random(block)
            jitters = rng.uniform(1.0 - _JITTER, 1.0 + _JITTER, block)
            for j in range(block):
                yield V[:, j], float(tests[j]), float(jitters[j])
            done += block

    def tune(self, burn_in: int, rng: np.random.Generator) -> _Tuning:
        """
        Return the step tuned over `burn_in` draws from the mode, by dual averaging (Hoffman and Gelman, 2014, algorithm 5) of its logarithm, and where they ended.

        It starts from one step a draw, and the step stays between
        :data:`_QUARTER` over :data:`_MOST_STEPS` and :data:`_QUARTER`. With
        no burn-in, the step is a whole quarter period.
        """
        here = self.point(np.array(self.mode))
        longest, shortest = math.log(_QUARTER), math.log(_QUARTER / _MOST_STEPS)
        target = math.log(10.0 * _QUARTER)
        log_h, log_mean, error, accepted = longest, longest, 0.0, 0.0
        for m, (v, test, jitter) in enumerate(self.moves(burn_in, rng), start=1):
            steps = _steps(math.exp(log_h))
            here, alpha = self.move(here, v, test, _QUARTER / steps * jitter, steps)
            accepted += alpha
            error += ((_TARGET - alpha) - error) / (m + 10)
            log_h = min(max(target - math.sqrt(m) / 0.05 * error, shortest), longest)
            weight = m**-0.75
            log_mean = weight * log_h + (1.0 - weight) * log_mean
        acceptance = accepted / burn_in if burn_in else math.nan
        return _Tuning(math.exp(log_mean), here, acceptance)

    def run(
        self, start: _Point, N: int, rng: np.random.Generator, step: float
    ) -> tuple[np.ndarray, _Point, float]:
        """Return `N` draws (``n x N``) of the chain from `start` at the step `step`, with velocities and tests drawn from `rng`, where it ended, and its acceptance rate."""
        steps = _steps(step)
        draws = np.empty((len(start.x), N))
        here, accepted = start, 0.0
        for j, (v, test, jitter) in enumerate(self.moves(N, rng)):
            here, alpha = self.move(here, v, test, _QUARTER / steps * jitter, steps)
            accepted += alpha
            draws[:, j] = here.x
        return draws, here, accepted / N


_SLOW_DRAWS = (
    "The posterior bends much over its spread, along a direction the Gaussian "
    "at the mode does not follow; raise draws= for those elements' spread, or "
    "read the totals the data constrain, which mix faster."
)
"""What the slow-mixing warning of the kept draws says of the cause and the remedy."""


def _steps(step: float) -> int:
    """Return the number of steps of about `step` that make a quarter period, at most :data:`_MOST_STEPS`."""
    return max(1, min(_MOST_STEPS, math.ceil(_QUARTER / step - 1e-9)))


def _with_obs(linear: Problem, obs: Vector) -> Problem:
    """Return the linear problem `linear` with the observations `obs`, on its labels."""
    return type(linear)(
        prior=linear.prior,
        prior_cov=linear.prior_cov,
        obs=obs,
        obs_cov=linear.obs_cov,
        forward=linear.forward,
        offset=linear.offset,
    )


def _d2(
    dy: np.ndarray,
    J_step: np.ndarray,
    S_0: LinearOperator,
    solve_S_z: Callable[[np.ndarray], np.ndarray],
    gamma: float,
) -> float:
    r"""
    Return a step's :math:`d^2 = \delta^T (S_0^{-1} + \gamma J^T S_z^{-1} J) \delta`, with neither inverse of :math:`S_0` nor :math:`\hat{S}^{-1}`.

    The step is :math:`\delta = S_0 \Delta y` (both estimates are :math:`x_0 +
    S_0 y`), so its prior term is :math:`\Delta y^T S_0 \Delta y`, and its
    data term :math:`\gamma (J \delta)^T S_z^{-1} (J \delta)`, with `J_step`
    :math:`= J \delta` and `solve_S_z` the solve with :math:`S_z`.
    """
    prior = float(dy @ np.asarray(S_0 @ dy))
    return prior + gamma * float(J_step @ np.asarray(solve_S_z(J_step)))


_HALVINGS = 30
"""The most times a step is halved before the cost stops rising along it."""

_SLOW = 10.0
"""How many times ``tol * n`` the distance left to the mode (in d2), read from the last two steps, may be before a solve warns that it converged slowly."""

_RISE = 1e-12
"""How much the cost may rise along a step, relative, and the step still be taken (roundoff near the mode)."""


def _warn_if_slow(d2s: list[float], limit: float) -> None:
    r"""
    Log a warning when the steps fell so slowly that the estimate may be further from the mode than the stopping rule meant.

    Rodgers' rule measures the step, not the distance to the mode. When each
    step is a factor :math:`q` of the last (Gauss-Newton converges so, by a
    constant factor, when the model bends much over the posterior's spread),
    the steps still to come add up to :math:`q / (1 - q)` of the last one, so
    the distance left is about :math:`(q / (1 - q))^2 d_k^2` in :math:`d^2`.
    :math:`q` is read from the last two steps, :math:`q^2 = d_k^2 /
    d_{k-1}^2`. It warns when that distance is above ``_SLOW`` times
    ``tol * n`` (`limit`): for a last step near the limit, :math:`q` above
    about 3/4.
    """
    if len(d2s) < 2 or not d2s[-2] > 0:
        return
    q = math.sqrt(d2s[-1] / d2s[-2])
    left = (q / (1 - q)) ** 2 * d2s[-1] if q < 1 else math.inf
    if left <= _SLOW * limit:
        return
    logger.warning(
        "Gauss-Newton converged slowly: each step was about %.2f of the last, so "
        "the estimate may still be about %.3g from the mode (as d2), above tol * "
        "n = %.3g. Lower tol to get closer. A model that bends this much over the "
        "posterior's spread is also where the Gaussian at the mode is least right "
        "(check it with post.trials).",
        q,
        left,
        limit,
    )


_NEWTON_ITERATIONS = 50
"""The most conjugate-gradient iterations a Newton step's solve takes; one that has not converged by then gives the Gauss-Newton step."""

_SOLVED = 1e-12
r"""The Newton step is solved when its residual, measured as a :math:`d^2`, is below this fraction of the Gauss-Newton step's :math:`d^2` (a relative error of :math:`10^{-6}`)."""


@dataclass(frozen=True)
class _Taken:
    """A step taken: the fraction of its length, the state it leads to (``x_0 + S_0 y``), the model there, and the cost's terms there (with the weighted residual :math:`S_z^{-1} r`)."""

    fraction: float
    x: np.ndarray
    y: np.ndarray
    modeled: np.ndarray
    data: float
    state: float
    weighted: np.ndarray

    def cost(self, gamma: float) -> float:
        """Return the cost, gamma times the data misfit plus the state misfit."""
        return gamma * self.data + self.state


def _newton(
    analytic: "Whitened | Information",
    curvature: np.ndarray,
    gradient: np.ndarray,
    step: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""
    Return the Newton step :math:`\delta` and :math:`\sigma` with :math:`\delta = \hat{S} \sigma`, or None where the Newton matrix is not positive definite.

    The Newton matrix is the Gauss-Newton one less the model's curvature,
    :math:`A = \hat{S}^{-1} - D` (`curvature` is :math:`D`'s diagonal,
    :math:`\gamma` included), and the step solves :math:`A \delta = b`, with
    `gradient` :math:`b = -g` (minus half the cost's gradient) and `step`
    :math:`\hat{S} b`, the Gauss-Newton step. It is solved by conjugate
    gradients preconditioned by :math:`\hat{S}`, the analytic solve's own
    covariance (``_cov_times``, one product an iteration; in exact arithmetic
    at most one iteration more than the elements :math:`D` is not zero on, and a
    few near the mode, where :math:`D` is small beside :math:`\hat{S}^{-1}`).
    Each search direction :math:`p` is :math:`\hat{S} s` with :math:`s`
    carried beside it, so :math:`A p = s - D p` needs no inverse of
    :math:`\hat{S}` or :math:`S_0`, and :math:`\delta = \hat{S} \sigma` with
    :math:`\sigma` the same sum of the :math:`s`. It stops when the residual
    :math:`\rho` has :math:`\rho^T \hat{S} \rho` (the :math:`d^2` of the
    correction still to make) below :data:`_SOLVED` times the Gauss-Newton
    step's.

    A direction along which :math:`p^T A p \le 0` proves :math:`A` is not
    positive definite, the conjugate gradients' own test at no extra cost
    (a Cholesky of :math:`A` would need it formed, ``n x n``, which the
    whitened form never is): it returns None, and the step is Gauss-Newton's.
    So does a solve that has not converged in :data:`_NEWTON_ITERATIONS`.
    A step it returns is a descent direction of the cost (each :math:`p` it
    took had positive curvature).
    """
    delta, sigma = np.zeros_like(step), np.zeros_like(step)
    rho, zeta = gradient, step
    p, s = zeta, rho
    rz = rz_0 = float(rho @ zeta)
    for k in range(1, _NEWTON_ITERATIONS + 1):
        Dp = curvature * p
        metric = float(p @ s)  # p^T S_hat^-1 p
        bend = metric - float(p @ Dp)
        if not bend > 0.0:
            logger.debug(
                "The Newton matrix is not positive definite along a direction "
                "(its curvature %.3g against Gauss-Newton's %.3g): a Gauss-Newton step.",
                bend,
                metric,
            )
            return None
        alpha = rz / bend
        delta = delta + alpha * p
        sigma = sigma + alpha * s
        rho = rho - alpha * (s - Dp)
        zeta = analytic._cov_times(rho[:, None])[:, 0]
        rz_new = float(rho @ zeta)
        if rz_new <= _SOLVED * rz_0:
            logger.debug("The Newton step solved in %d iterations.", k)
            return delta, sigma
        p = zeta + (rz_new / rz) * p
        s = rho + (rz_new / rz) * s
        rz = rz_new
    logger.debug(
        "The Newton step's solve did not converge in %d iterations: a Gauss-Newton step.",
        _NEWTON_ITERATIONS,
    )
    return None


def _newton_step(
    problem: Problem,
    analytic: "Whitened | Information",
    here: _Taken,
    step: np.ndarray,
    gamma: float,
    solve_S_z: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    r"""
    Return Newton's step from `here`, with its :math:`\Delta y` and :math:`d^2`, or None for a model with no curvature or where the Newton matrix is not positive definite (``_newton``).

    `step` is the Gauss-Newton step, :math:`\hat{S} b` with :math:`b = \gamma
    J^T S_z^{-1} r - y` (minus half the cost's gradient; :math:`y` stands for
    :math:`S_0^{-1} (x - x_0)`). With :math:`\delta = \hat{S} \sigma`,
    :math:`\Delta y = \sigma - \gamma J^T S_z^{-1} J \delta`, since
    :math:`S_0 (\sigma - \gamma J^T S_z^{-1} J \hat{S} \sigma) = \hat{S}
    \sigma` (from :math:`\hat{S}^{-1} = S_0^{-1} + \gamma J^T S_z^{-1} J`;
    it holds for a singular :math:`S_0` too), so the next state is still
    :math:`x_0 + S_0 y`.
    """
    curvature = problem._curvature(here.x, here.weighted)
    if curvature is None:
        return None
    J = analytic.problem._operator.data
    J_T = J._transpose()
    gradient = gamma * np.asarray(J_T @ here.weighted, dtype=float) - here.y
    solved = _newton(analytic, gamma * curvature, gradient, step)
    if solved is None:
        return None
    delta, sigma = solved
    J_delta = np.asarray(J @ delta, dtype=float)
    dy = sigma - gamma * np.asarray(J_T @ np.asarray(solve_S_z(J_delta)), dtype=float)
    return (
        delta,
        dy,
        _d2(dy, J_delta, analytic.problem.prior_cov.data, solve_S_z, gamma),
    )


def _along(
    problem: Problem,
    cost: Callable[[np.ndarray, np.ndarray], tuple[float, float, np.ndarray]],
    x: np.ndarray,
    y: np.ndarray,
    step: np.ndarray,
    dy: np.ndarray,
    limit: float,
    gamma: float,
) -> tuple[_Taken | None, Exception | None]:
    """
    Return the step from `x` taken as far along as the cost does not rise above `limit` (any length if `limit` is infinite), halved until it does not, or None if it rises however short; with the model's last failure.

    A step too far can overflow the model (inf, or the NaN of 0 * inf, which a
    Vector refuses): it is taken shorter.
    """
    fraction, failure = 1.0, None
    for _ in range(_HALVINGS):
        x_new, y_new = x + fraction * step, y + fraction * dy
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                modeled = problem._apply(x_new)
            except (ValueError, FloatingPointError, OverflowError) as error:
                failure, modeled = error, None
        if modeled is not None and np.isfinite(modeled).all():
            data, state, weighted = cost(modeled, y_new)
            if gamma * data + state <= limit:
                return _Taken(
                    fraction, x_new, y_new, modeled, data, state, weighted
                ), None
        fraction /= 2
    return None, failure


def _iterate(
    problem: Problem, options: GaussNewtonOptions, start: GaussNewton | None = None
) -> GaussNewton:
    """
    Return the Gauss-Newton solution of `problem`, from the prior or from the last linearization of `start` (a solution of the same problem with other observations).

    Each step goes to the analytic estimate of the problem linearized at the
    last estimate; for a model with a curvature (a transform), to Newton's
    estimate instead (``_newton_step``) where its matrix is positive
    definite and its cost is the lower. A step whose cost is higher than the
    last estimate's (it overshot) is halved until it is not. Only a whole
    Gauss-Newton step small enough stops the iterations (so a halved step,
    or a Newton one, never does), and with a curvature only when Newton's
    step is small enough too: where the Newton matrix is nearly singular,
    the Gauss-Newton step is much shorter than the distance to the mode.

    Raises
    ------
    RuntimeError
        If the iterations have not stopped after ``max_iterations``, or the
        cost rises along a step however short.
    """
    gamma = options.gamma
    S_0 = problem.prior_cov.data
    solve_S_z = problem.obs_cov.data._solver()  # factored once for every step
    n = len(problem.prior)
    z = problem.obs.values
    c = problem._offset_values()
    limit = options.tol * n

    def cost(modeled: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Return the data and the state misfits of the state ``x_0 + S_0 y`` that models `modeled`, and the weighted residual ``S_z^-1 r``."""
        r = z - modeled - c
        weighted = np.asarray(solve_S_z(r))
        return float(r @ weighted), float(y @ np.asarray(S_0 @ y)), weighted

    analytic: Whitened | Information | None
    if start is None:
        x, y = problem.prior.values, np.zeros(n)
        analytic = None
    else:
        x, y = start.point, start.point_y
        linear = _with_obs(start.analytic.problem, problem.obs)
        analytic = start.analytic.with_data(linear)
    modeled = problem._apply(x)
    data, state, weighted = cost(modeled, y)
    here = _Taken(1.0, x, y, modeled, data, state, weighted)
    current = here.cost(gamma)

    d2s: list[float] = []
    whole: list[float] = []  # the d2 the stopping rule compares, of each whole step
    costs: list[float] = []
    newton_steps = 0
    for i in range(1, options.max_iterations + 1):
        if analytic is None:
            analytic = _factor(problem._linearized(here.x, here.modeled), gamma)
        J = analytic.problem._operator.data
        x_full = analytic.estimate()
        y_full = np.asarray(J._transpose() @ analytic._innovation_precision_times())
        step, dy = x_full - here.x, y_full - here.y
        d2 = _d2(dy, np.asarray(J @ step, dtype=float), S_0, solve_S_z, gamma)
        # None for a model without a curvature: Gauss-Newton's step alone
        newton_step = _newton_step(problem, analytic, here, step, gamma, solve_S_z)
        converged = d2 < limit and (newton_step is None or newton_step[2] < limit)
        whole.append(d2 if newton_step is None else max(d2, newton_step[2]))
        rise = math.inf if converged else current + _RISE * abs(current)
        # Newton's step first, so that it is kept on a tie
        candidates = [(step, dy, d2, False)]
        if newton_step is not None and not converged:
            candidates.insert(0, (*newton_step, True))
        taken, failure, newton = None, None, False
        for step_i, dy_i, d2_i, newton_i in candidates:
            along, failure = _along(
                problem, cost, here.x, here.y, step_i, dy_i, rise, gamma
            )
            if along is None:
                logger.debug(
                    "The cost rises along %s step %d however short.",
                    "Newton's" if newton_i else "Gauss-Newton's",
                    i,
                )
            elif taken is None or along.cost(gamma) < taken.cost(gamma):
                taken, d2, newton = along, d2_i, newton_i
        if taken is None:
            raise RuntimeError(
                f"The cost rises along Gauss-Newton's step {i} however short the "
                "step: the Jacobian is not the derivative of the model at the last "
                "estimate (check it with fips.taylortest(problem.forward, x)), or "
                "the model does not give finite observations near it."
            ) from failure
        kind = "Newton" if newton else "Gauss-Newton"
        newton_steps += newton
        if taken.fraction < 1.0:
            logger.debug(
                "%s step %d overshot: taken at %g of its length.",
                kind,
                i,
                taken.fraction,
            )
        trial = taken.cost(gamma)
        d2s.append(d2 * taken.fraction**2)
        costs.append(trial)
        logger.debug(
            "%s step %d: d2 = %.3g (it stops below %.3g), cost %.6g.",
            kind,
            i,
            d2s[-1],
            limit,
            trial,
        )
        if converged:
            logger.info(
                "Gauss-Newton converged in %d steps%s: the last step's d2 is %.3g, "
                "below tol * n = %.3g.",
                i,
                f" ({newton_steps} of them Newton's)" if newton_steps else "",
                d2,
                limit,
            )
            _warn_if_slow(d2s, limit)
            return GaussNewton(
                problem=problem,
                analytic=analytic,
                mode=_read_only(taken.x),
                mode_y=_read_only(taken.y),
                point=_read_only(np.array(here.x)),
                point_y=_read_only(np.array(here.y)),
                d2=_read_only(np.array(d2s)),
                cost=_read_only(np.array(costs)),
                data_misfit=taken.data,
                state_misfit=taken.state,
                options=options,
            )
        here, current, analytic = taken, trial, None
    raise RuntimeError(_unconverged(options.max_iterations, whole, d2s, limit))


def _shown(values: list[float]) -> str:
    """Return the values as text, the first three and the last four when there are more than eight."""
    shown = [f"{d:.3g}" for d in values]
    if len(shown) > 8:
        shown = [*shown[:3], "...", *shown[-4:]]
    return ", ".join(shown)


def _unconverged(
    steps: int, whole: list[float], taken: list[float], limit: float
) -> str:
    """Return the message of iterations that did not stop: the d2 the rule compared (each whole step's, before its line search), and the d2 of the steps as taken."""
    return (
        f"Gauss-Newton did not converge in {steps} steps: the last whole step's d2 "
        f"is {whole[-1]:.3g}, above tol * n = {limit:.3g} (the rule compares each "
        f"step's d2 before it is shortened: {_shown(whole)}; as taken: "
        f"{_shown(taken)}). If they are still falling, raise max_iterations. A "
        "whole step that does not shrink while the steps taken do (each shortened "
        "to keep the cost falling) points to the Jacobian: check it with "
        "fips.taylortest(problem.forward, x). Or the problem may be too nonlinear "
        "for Gauss-Newton from the prior."
    )


def _gauss_newton(problem: Problem, options: GaussNewtonOptions) -> Posterior:
    """Find the mode by Gauss-Newton; with a forward operator, the analytic posterior itself."""
    if isinstance(problem.forward, Operator):
        return Posterior(problem, _factor(problem, options.gamma), options)
    return Posterior(problem, _iterate(problem, options), options)
