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
direction is still Gauss-Newton's, so the mode is the same), and only a whole
step below the rule stops the iterations. The rule measures the step, not the
distance to the mode: when the steps fall by a constant factor near 1 (a model
that bends much over the posterior's spread), the solve logs a warning that
the estimate may be further from the mode (``_warn_if_slow``).

:math:`\hat{S}_i^{-1}` is never formed. Every estimate is :math:`x_0 + S_0 y`
with :math:`y = J_i^T u_i` (:math:`u_i` the analytic solve's :math:`(J_i S_0
J_i^T + S_z/\gamma)^{-1}` times its innovation), so the prior term of
:math:`d^2` is :math:`\Delta y^T S_0 \Delta y` (no inverse of :math:`S_0`,
which may be singular) and its data term :math:`\gamma (J_i \delta)^T
S_z^{-1} (J_i \delta)`, a solve with the mismatch factored once.

The solution (:class:`GaussNewton`) keeps the analytic solution of the last
linearized problem. Its estimate is the mode, and its covariance, gain, DOFS
and draws are those of the Gaussian at the mode: the analytic posterior of
the problem linearized at the last iterate, one step of :math:`d^2 <
\mathrm{tol} \cdot n` from the mode. The misfits and the chi-squared are the
cost's terms at the mode, from the model itself. With a forward operator
(a linear problem) ``_gauss_newton`` returns the analytic solution itself.

References
----------
Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and
Practice*, chapter 5. World Scientific. https://doi.org/10.1142/3171
"""

import logging
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fips.linalg import LinearOperator
from fips.linalg.forms import _read_only
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

    Raises
    ------
    ValueError
        If `gamma` is not a positive, finite number, `max_iterations` is not a
        positive integer, or `tol` is not a positive, finite number.
    """

    gamma: float = 1.0
    """The regularization factor: the model-data mismatch is divided by it."""
    max_iterations: int = 50
    """The most steps to take."""
    tol: float = 1e-6
    r"""The iterations stop when :math:`d^2 <` ``tol`` :math:`\cdot\, n`."""

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


@dataclass(frozen=True, eq=False)
class GaussNewton:
    r"""
    The Gauss-Newton method's solution: the mode, and the Gaussian of the problem linearized at the last iterate.

    It keeps the problem with the forward model (``problem``), the analytic
    solution of the last linearized problem (``analytic``, whose estimate is
    the mode), the state that problem was linearized at, and each step's
    :math:`d^2` and the cost after it. The :class:`~fips.Posterior` asks it
    for every array it reports: it is the method's
    :class:`~fips.posterior.Solution`. Its answers, by what each needs:

    - the estimate is the mode, not the mean, and is not linear in the
      observations (no ``estimate_is_mean``, no ``estimate_is_linear``);
    - the variances, the covariance of aggregates, the gain, DOFS, the
      information content and the draws are those of the Gaussian at the
      mode (``analytic``'s); the posterior reads the averaging kernel and the
      innovation covariance from the linearized problem (``linearized``);
    - the misfits and the chi-squared are the terms of the cost at the mode,
      with the residual :math:`z - F(\hat{x}) - c` of the model itself;
    - :meth:`with_data` solves again, starting from the last linearization;
      there is no ``with_gamma``: the linearization depends on gamma.

    Every estimate is :math:`x_0 + S_0 y`, and the solution keeps the
    :math:`y` of the mode and of the last linearization point, so the state
    misfit and :math:`d^2` need no inverse of :math:`S_0`.
    """

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

    def __getstate__(self) -> dict:
        """Return the fields for a pickle or a copy."""
        return dict(self.__dict__)

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle or a copy, the arrays read-only again."""
        self.__dict__.update(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                _read_only(value)

    def estimate(self) -> np.ndarray:
        """Return the mode."""
        return self.mode

    def variances(self) -> np.ndarray:
        """Return the variances of the Gaussian at the mode."""
        return self.analytic.variances()

    def cov(self, W: LinearOperator | None = None) -> np.ndarray:
        """Return the covariance of the Gaussian at the mode, or of aggregates of it."""
        return self.analytic.cov(W)

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

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """Return `N` draws of the Gaussian at the mode (``n x N``)."""
        return self.analytic.sample(N, rng)

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


def _iterate(
    problem: Problem, options: GaussNewtonOptions, start: GaussNewton | None = None
) -> GaussNewton:
    """
    Return the Gauss-Newton solution of `problem`, from the prior or from the last linearization of `start` (a solution of the same problem with other observations).

    Each step goes to the analytic estimate of the problem linearized at the
    last estimate. A step whose cost is higher than the last estimate's (the
    linearization overshot) is halved until it is not; only a whole step
    small enough stops the iterations, so a halved step never does.

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

    def cost(modeled: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return the data and the state misfits of the state ``x_0 + S_0 y`` that models `modeled`."""
        r = z - modeled - c
        return float(r @ np.asarray(solve_S_z(r))), float(y @ np.asarray(S_0 @ y))

    analytic: Whitened | Information | None
    if start is None:
        x, y = problem.prior.values, np.zeros(n)
        analytic = None
    else:
        x, y = start.point, start.point_y
        linear = _with_obs(start.analytic.problem, problem.obs)
        analytic = start.analytic.with_data(linear)
    modeled = problem._apply(x)
    data, state = cost(modeled, y)
    current = gamma * data + state

    d2s: list[float] = []
    costs: list[float] = []
    for i in range(1, options.max_iterations + 1):
        if analytic is None:
            analytic = _factor(problem._linearized(x, modeled), gamma)
        J = analytic.problem._operator.data
        x_full = analytic.estimate()
        y_full = np.asarray(J._transpose() @ analytic._innovation_precision_times())
        step, dy = x_full - x, y_full - y
        d2 = _d2(dy, np.asarray(J @ step, dtype=float), S_0, solve_S_z, gamma)
        converged = d2 < limit
        fraction, failure, modeled_new = 1.0, None, modeled
        for _ in range(_HALVINGS):
            x_new, y_new = x + fraction * step, y + fraction * dy
            # a step too far can overflow the model (inf, or the NaN of 0 * inf,
            # which a Vector refuses): it is taken shorter
            with np.errstate(over="ignore", invalid="ignore"):
                try:
                    modeled_new = problem._apply(x_new)
                except (ValueError, FloatingPointError, OverflowError) as error:
                    failure, finite = error, False
                else:
                    finite = bool(np.isfinite(modeled_new).all())
                    if finite:
                        data, state = cost(modeled_new, y_new)
            trial = gamma * data + state if finite else math.inf
            if finite and (converged or trial <= current + _RISE * abs(current)):
                break
            fraction /= 2
        else:
            raise RuntimeError(
                f"The cost rises along Gauss-Newton's step {i} however short the "
                "step: the Jacobian is not the derivative of the model at the last "
                "estimate (check it with fips.taylortest(problem.forward, x)), or "
                "the model does not give finite observations near it."
            ) from failure
        if fraction < 1.0:
            logger.debug(
                "Gauss-Newton step %d overshot: taken at %g of its length.",
                i,
                fraction,
            )
        d2s.append(d2 * fraction**2)
        costs.append(trial)
        logger.debug(
            "Gauss-Newton step %d: d2 = %.3g (it stops below %.3g), cost %.6g.",
            i,
            d2s[-1],
            limit,
            trial,
        )
        if converged:
            logger.info(
                "Gauss-Newton converged in %d steps: the last step's d2 is %.3g, "
                "below tol * n = %.3g.",
                i,
                d2,
                limit,
            )
            _warn_if_slow(d2s, limit)
            return GaussNewton(
                problem=problem,
                analytic=analytic,
                mode=_read_only(x_new),
                mode_y=_read_only(y_new),
                point=_read_only(np.array(x)),
                point_y=_read_only(np.array(y)),
                d2=_read_only(np.array(d2s)),
                cost=_read_only(np.array(costs)),
                data_misfit=data,
                state_misfit=state,
                options=options,
            )
        x, y, modeled, current, analytic = x_new, y_new, modeled_new, trial, None
    shown = [f"{d:.3g}" for d in d2s]
    if len(shown) > 8:
        shown = [*shown[:3], "...", *shown[-4:]]
    raise RuntimeError(
        f"Gauss-Newton did not converge in {options.max_iterations} steps: the last "
        f"step's d2 is {d2s[-1]:.3g}, above tol * n = {limit:.3g} (the steps' d2: "
        f"{', '.join(shown)}). If they are still falling, raise max_iterations; if not, check "
        "the Jacobian with fips.taylortest(problem.forward, x), or the problem may "
        "be too nonlinear for Gauss-Newton from the prior."
    )


def _gauss_newton(problem: Problem, options: GaussNewtonOptions) -> Posterior:
    """Find the mode by Gauss-Newton; with a forward operator, the analytic posterior itself."""
    if isinstance(problem.forward, Operator):
        return Posterior(problem, _factor(problem, options.gamma), options)
    return Posterior(problem, _iterate(problem, options), options)
