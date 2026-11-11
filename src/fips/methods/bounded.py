r"""
The bounded method: the posterior with bounds on the state.

The analytic posterior :math:`N(\hat{x}, \hat{S})` with the state held to a box,
:math:`l \le x \le u` (a flux that cannot be negative, a fraction between 0 and
1), is that Gaussian truncated to the box: the prior times the likelihood, with
the prior set to zero outside the bounds. ``_bounded`` factors the problem as
the analytic method does (``_factor``) and keeps two things from the factors:

- **the mode**, the estimate (:meth:`Bounded.estimate`): the point of the box
  where the analytic cost is least. The cost is quadratic, so that is the point
  of the box nearest :math:`\hat{x}` in the metric of the posterior,
  :math:`(x - \hat{x})^T \hat{S}^{-1} (x - \hat{x})`: the analytic estimate
  moved onto the bounds along the posterior's correlations. An element that
  was outside goes to its bound, and the others move by what the posterior
  says they would if that element were known to be at its bound. It is found
  by an active set (``_mode``) that needs :math:`\hat{S}` only at the columns
  of the elements on a bound, from the factors (``_cov_times``);
- **draws of the truncated posterior** (:meth:`Bounded.sample`), by exact
  Hamiltonian Monte Carlo (Pakman and Paninski, 2014: ``_Sampler``), for its
  uncertainty: the variances and the covariance of aggregates are those of the
  draws, and so are the posterior mean and the intervals (their quantiles).

The gain, DOFS and the information content are those of the analytic posterior
with the elements on a bound held there (the Gaussian at the mode); the
chi-squared and the misfits are the analytic cost's terms at the mode. With no
finite bound, the posterior is the analytic one, and ``_bounded`` returns the
analytic solution itself.

References
----------
Pakman, A., and Paninski, L. (2014). Exact Hamiltonian Monte Carlo for truncated
multivariate Gaussians. *Journal of Computational and Graphical Statistics*,
23(2), 518-542. https://doi.org/10.1080/10618600.2013.788448

Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, second
edition, chapter 16. Springer.

Lawson, C. L., and Hanson, R. J. (1974). *Solving Least Squares Problems*.
Prentice-Hall.
"""

import logging
import math
import numbers
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve

from fips.linalg import DenseOperator, LinearOperator
from fips.linalg.forms import _read_only
from fips.methods._draws import _warn_if_imprecise
from fips.methods.analytic import AnalyticOptions, Information, Whitened, _factor
from fips.operator import Operator
from fips.posterior import Posterior
from fips.problem import Problem
from fips.space import Space, _listed
from fips.vector import Vector

logger = logging.getLogger(__name__)

_KKT_RTOL = float(np.sqrt(np.finfo(float).eps))
"""How far from the optimality conditions the mode may be, relative to the scale of the state (1.5e-8)."""

_DECISION_RTOL = 1e-10
"""How wrong the sign of a multiplier must be before the active set releases its element, relative to the scale of the state."""

_BLOCK = 512
"""The number of velocities the sampler draws in one block."""

_COLUMNS = 64
"""The number of neighboring columns of the posterior covariance computed in one product."""

_NEAR = 4.0
"""The sampler fetches, before it starts, the columns of the elements whose analytic estimate is within this many posterior standard deviations inside a bound, or outside it."""


@dataclass(frozen=True)
class _ByBlock(Mapping):
    """
    Bounds given by block, as the options keep them: a read-only mapping of each block's name to its bound.

    A dict is not hashable, and the options are; this is, through its pairs
    (a Vector hashes as itself).
    """

    parts: tuple[tuple[str, "float | Vector"], ...]
    """Each block's name and its bound, a number or a Vector on its labels."""

    def __getitem__(self, block: str) -> "float | Vector":
        """Return the bound of `block`."""
        return dict(self.parts)[block]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the block names."""
        return (block for block, _ in self.parts)

    def __len__(self) -> int:
        """Return the number of blocks."""
        return len(self.parts)

    def __repr__(self) -> str:
        """Return the bounds as the dict they were given as."""
        return repr(dict(self.parts))


def _checked_value(value: Any, side: str, where: str) -> "float | Vector":
    """
    Return one bound, a number or a Vector, checked.

    Raises
    ------
    TypeError
        If it is neither.
    ValueError
        If it is NaN, or a bound no value can satisfy (``+inf`` below,
        ``-inf`` above).
    """
    impossible = math.inf if side == "lower" else -math.inf
    if isinstance(value, Vector):
        if np.any(value.values == impossible):
            raise ValueError(
                f"{side}{where} is {impossible} somewhere, a bound no value can "
                f"satisfy; {-impossible} means no bound."
            )
        return value
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number):
            raise ValueError(
                f"{side}{where} is NaN; give a number, or None for no bound."
            )
        if number == impossible:
            raise ValueError(
                f"{side}{where} is {impossible}, a bound no value can satisfy; "
                f"{-impossible} (or None) means no bound."
            )
        return number
    raise TypeError(
        f"{side}{where} is a number or a Vector, not {type(value).__name__} "
        "(a Series becomes a Vector with Vector.from_series)."
    )


def _checked_bound(bound: Any, side: str) -> "float | Vector | _ByBlock | None":
    """
    Return a bound as the options keep it: None, a number, a Vector, or the bounds of each block.

    A dict by block is kept as a ``_ByBlock`` (a dict is not hashable, and the
    options are); copying the options checks it again.
    """
    if bound is None:
        return None
    if isinstance(bound, Mapping):
        parts = []
        for block, value in bound.items():
            if not isinstance(block, str):
                raise TypeError(
                    f"The keys of {side} are block names, strings; got {block!r}."
                )
            parts.append((block, _checked_value(value, side, f" of block {block!r}")))
        return _ByBlock(tuple(parts))
    return _checked_value(bound, side, "")


@dataclass(frozen=True)
class BoundedOptions:
    r"""
    The options of the ``"bounded"`` method.

    Parameters
    ----------
    gamma : float, default 1.0
        The regularization factor :math:`\gamma`, as for the analytic method
        (:class:`~fips.methods.analytic.AnalyticOptions`): the model-data
        mismatch is divided by it.
    lower, upper : float, Vector, dict or None, default None
        The bounds on the state: ``lower <= x <= upper``. A number bounds
        every state element; a :class:`~fips.Vector` gives each element its
        own bound, with exactly the state's labels (in any order), ``-inf``
        (or ``inf`` for `upper`) where an element has none; a dict maps block
        names to a number or a Vector on that block's labels, and leaves the
        blocks it does not name unbounded (``lower={"flux": 0.0}`` keeps a
        bias block free). A finite number on a state of several blocks bounds
        them all, and the solve logs a warning naming them. None is no bound.
        With no finite bound, the posterior is the analytic one.
    draws : int, default 2000
        The number of draws of the truncated posterior kept for its variances
        and the covariance of aggregates. They are Monte Carlo estimates from a
        chain: each draw is exact, but starts from the last. At 2000, a
        standard deviation is good to about 1.5% for an element away from its
        bounds (:math:`1 / \sqrt{2\, \text{draws}}`, as for independent
        draws), and to about 12% (root mean square over 24 solves; 25% at
        worst) for an element whose analytic estimate is one to four standard
        deviations outside its bound: its draws pile against the bound and mix
        slowly (an autocorrelation time of 13 to 17 draws). One far outside
        mixes faster. The error falls as :math:`1/\sqrt{\text{draws}}`; the
        solve warns when a standard deviation is uncertain by more than 10%,
        and says how many draws would do.
    burn_in : int, default 20
        The draws made and left out before those kept, from the mode. Each
        draw of the sampler travels a quarter period, after which a Gaussian
        without bounds would be independent of its start, so few are needed.
    seed : int, default 0
        The seed of the kept draws, so that a solve is reproducible.

    Raises
    ------
    ValueError
        If `gamma` is not a positive, finite number; a bound is NaN, or a
        bound no value can satisfy (``lower`` of ``inf``); both bounds are
        numbers and `lower` is not below `upper`; or `draws` is not an
        integer of at least 2, `burn_in` not a nonnegative one, or `seed` not
        a nonnegative integer. A bound's labels are checked against the state by
        :func:`fips.solve`.
    TypeError
        If a bound is not a number, a Vector, a dict or None, or a dict has a
        key that is not a string.
    """

    gamma: float = 1.0
    """The regularization factor: the model-data mismatch is divided by it."""
    lower: "float | Vector | Mapping[str, float | Vector] | None" = None
    """The lower bounds on the state, as given (a dict by block is kept as a mapping of its blocks)."""
    upper: "float | Vector | Mapping[str, float | Vector] | None" = None
    """The upper bounds on the state, as given (a dict by block is kept as a mapping of its blocks)."""
    draws: int = 2000
    """The number of draws of the truncated posterior kept for its variances and covariances."""
    burn_in: int = 20
    """The draws left out before those kept."""
    seed: int = 0
    """The seed of the kept draws."""

    def __post_init__(self):
        object.__setattr__(self, "gamma", AnalyticOptions(self.gamma).gamma)
        lower = _checked_bound(self.lower, "lower")
        upper = _checked_bound(self.upper, "upper")
        if isinstance(lower, float) and isinstance(upper, float) and not lower < upper:
            raise ValueError(
                f"lower ({lower}) must be below upper ({upper}): an element held "
                "at one value is not bounded but known."
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
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


def _resolve(
    bound: "float | Vector | Mapping[str, float | Vector] | None",
    space: Space,
    side: str,
) -> np.ndarray:
    """
    Return the bound of each state element, in the order of `space`: ``-inf`` (lower) or ``inf`` (upper) where there is none.

    Raises
    ------
    ValueError
        If a Vector does not have exactly the labels it bounds (the state's,
        or its block's), or a dict names a block the state does not have.
    """
    none = -math.inf if side == "lower" else math.inf
    values = np.full(len(space), none)
    if bound is None:
        return values
    if isinstance(bound, float):
        values[:] = bound
        return values
    if isinstance(bound, Vector):
        positions = space._align(
            bound.space, f"the {side} bounds", f"The {side} bounds and the state"
        )
        return bound.values[positions].copy()
    for block, value in bound.items():
        if block not in space:
            raise ValueError(
                f"{side} names a block the state does not have, {block!r}; the "
                f"state's blocks are {_listed(space.names)}."
            )
        where = space.slice(block)
        if isinstance(value, Vector):
            own = Space({block: space.select(block)})
            positions = own._align(
                value.space,
                f"the {side} bounds of block {block!r}",
                f"The {side} bounds of block {block!r} and the block",
            )
            values[where] = value.values[positions]
        else:
            values[where] = value
    return values


_SLOW_DRAWS = (
    "The draws are exact but each starts from the last, and an element whose "
    "analytic estimate is one to a few posterior standard deviations outside "
    "its bound mixes slowest; raise draws= for those elements' spread."
)
"""What the slow-mixing warning of the kept draws says of the cause and the remedy."""


class _Columns:
    r"""
    Columns of the analytic posterior covariance :math:`\hat{S}`, computed from the factors in blocks and kept.

    The active set and the sampler need :math:`\hat{S}` only at the columns of
    the elements on (or near) a bound. They are computed as
    ``analytic._cov_times`` of columns of the identity, in fixed blocks of
    :data:`_COLUMNS` neighbors (``0:64``, ``64:128``, ...), whichever column of
    a block is asked for first: a product's last bits depend on its shape, so
    a column is the same whatever was fetched before it, and a chain of draws
    is the same after a pickle. They are kept in one Fortran-ordered array, so
    a column is a contiguous view. One cache serves every posterior of other
    observations (``with_data``): :math:`\hat{S}` does not depend on their
    values.
    """

    def __init__(self, analytic: "Whitened | Information"):
        self._analytic = analytic
        n = len(analytic.problem.prior)
        self._slot = np.full(n, -1, dtype=np.intp)
        self._store = np.empty((n, 0), order="F")
        self._count = 0

    def fetch(self, indices: np.ndarray) -> None:
        """Compute the blocks of the columns `indices` not yet kept."""
        n = len(self._slot)
        indices = np.asarray(indices, dtype=np.intp)
        blocks = np.unique(indices[self._slot[indices] < 0] // _COLUMNS)
        if not len(blocks):
            return
        need = self._count + _COLUMNS * len(blocks)
        if need > self._store.shape[1]:
            capacity = min(n, max(need, 2 * self._store.shape[1]))
            store = np.empty((n, capacity), order="F")
            store[:, : self._count] = self._store[:, : self._count]
            self._store = store
        for block in blocks:
            columns = np.arange(block * _COLUMNS, min(n, (block + 1) * _COLUMNS))
            E = np.zeros((n, len(columns)))
            E[columns, np.arange(len(columns))] = 1.0
            slots = np.arange(self._count, self._count + len(columns))
            self._store[:, slots] = self._analytic._cov_times(E)
            self._slot[columns] = slots
            self._count += len(columns)

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        r"""Return :math:`\hat{S}` at the columns `indices` (``n x k``), fetching those not yet kept."""
        self.fetch(indices)
        return self._store[:, self._slot[indices]]

    def column(self, i: int) -> np.ndarray:
        r"""Return column `i` of :math:`\hat{S}`, a view of the kept array."""
        if self._slot[i] < 0:
            self.fetch(np.array([i]))
        return self._store[:, self._slot[i]]


def _scale(
    x_hat: np.ndarray, variances: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Return the scale of the state the mode's tolerances are relative to: its largest value, bound or posterior standard deviation."""
    bounds = np.concatenate([lower, upper])
    finite = bounds[np.isfinite(bounds)]
    return float(
        max(
            np.abs(x_hat).max(initial=0.0),
            np.sqrt(np.maximum(variances, 0.0)).max(initial=0.0),
            np.abs(finite).max(initial=0.0),
            np.finfo(float).tiny,
        )
    )


def _mode(
    x_hat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    columns: _Columns,
    variances: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Return the mode in the box, which side of its bound each element is held at (``1`` lower, ``-1`` upper, ``0`` free), and the multipliers.

    The mode minimizes :math:`\tfrac{1}{2} (x - \hat{x})^T P (x - \hat{x})`,
    :math:`P = \hat{S}^{-1}`, over :math:`l \le x \le u`, by the primal
    active-set method for a convex quadratic program (Nocedal and Wright,
    2006, algorithm 16.3), here with bounds only: the method of which Lawson
    and Hanson's nonnegative least squares, and its bounded variant, are the
    least-squares case, with the step back to the box at the first element
    that would leave it. It keeps a feasible
    point and a working set :math:`a` of elements held at a bound,
    :math:`b_a`. Holding them there, the least cost is the analytic posterior
    conditioned on :math:`x_a = b_a`,

    .. math::

        x(a) = \hat{x} + \hat{S}_{:a} \lambda_a, \qquad
        \hat{S}_{aa} \lambda_a = b_a - \hat{x}_a,

    whose gradient is :math:`P (x(a) - \hat{x}) = \lambda` (zero off
    :math:`a`): :math:`\lambda_a` are the multipliers, positive at a lower
    bound and negative at an upper one when :math:`a` is right. Each step
    moves toward :math:`x(a)` and stops at the first free element that would
    leave the box, which joins :math:`a`; at :math:`x(a)` itself, an element
    whose multiplier has the wrong sign leaves :math:`a` (the most wrong
    first, measured as :math:`\lambda_i \hat{S}_{ii}`, the distance it pulls
    the element), and with none the point is the mode. It starts from
    :math:`\hat{x}` put into the box, with the elements it put on a bound in
    :math:`a`. Each step lowers the cost or changes :math:`a` at the same
    cost, so it ends (a degenerate problem could cycle among sets of the
    same cost: ``10 n + 100`` steps at most, then it raises); only
    :math:`\hat{S}_{:a}` is used (``columns``). The solve then checks the
    optimality conditions of what it returns (``_kkt``).

    Raises
    ------
    numpy.linalg.LinAlgError
        If :math:`\hat{S}_{aa}` is not positive definite (an element on a
        bound whose posterior variance is zero, or two whose errors the
        posterior ties together), or the active set does not end.
    """
    n = len(x_hat)
    side = np.zeros(n, dtype=np.int8)
    side[x_hat < lower] = 1
    side[x_hat > upper] = -1
    x = np.clip(x_hat, lower, upper)
    multipliers = np.zeros(n)
    tolerance = _DECISION_RTOL * scale
    for _ in range(10 * n + 100):
        active = np.flatnonzero(side)
        if not len(active):
            return x_hat.copy(), side, multipliers
        bound = np.where(side[active] > 0, lower[active], upper[active])
        S_a = columns(active)
        lam = _conditioned_solve(S_a[active], bound - x_hat[active])
        target = x_hat + S_a @ lam
        target[active] = bound
        step = target - x
        free = side == 0
        down = free & (step < 0) & np.isfinite(lower)
        up = free & (step > 0) & np.isfinite(upper)
        alpha, blocking, held = 1.0, -1, 0
        if down.any():
            reach = np.maximum(x[down] - lower[down], 0.0) / -step[down]
            k = int(np.argmin(reach))
            if reach[k] < alpha:
                alpha, blocking, held = float(reach[k]), int(np.flatnonzero(down)[k]), 1
        if up.any():
            reach = np.maximum(upper[up] - x[up], 0.0) / step[up]
            k = int(np.argmin(reach))
            if reach[k] < alpha:
                alpha, blocking, held = float(reach[k]), int(np.flatnonzero(up)[k]), -1
        if blocking >= 0:
            x = x + alpha * step
            x[active] = bound
            x[blocking] = lower[blocking] if held > 0 else upper[blocking]
            side[blocking] = held
            continue
        x = target
        pull = side[active] * lam * variances[active]  # below zero: the wrong sign
        worst = int(np.argmin(pull))
        if pull[worst] >= -tolerance:
            multipliers[active] = lam
            return x, side, multipliers
        side[active[worst]] = 0
    raise LinAlgError(
        "The bounded mode was not found: the active set did not end after "
        f"{10 * n + 100} steps."
    )


def _conditioned_solve(S_aa: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    r"""
    Return :math:`\hat{S}_{aa}^{-1}` times `rhs`, by Cholesky.

    Raises
    ------
    numpy.linalg.LinAlgError
        If :math:`\hat{S}_{aa}` is not positive definite.
    """
    try:
        factor = cho_factor(0.5 * (S_aa + S_aa.T), lower=True)
    except LinAlgError as err:
        raise LinAlgError(
            "The posterior covariance of the elements on a bound is not positive "
            "definite: an element whose posterior variance is zero, or elements "
            "whose posterior errors are tied together, cannot be held at their "
            "bounds. Check the bounds of the elements the observations or the "
            "prior fix."
        ) from err
    return np.asarray(cho_solve(factor, rhs))


def _kkt(
    x_hat: np.ndarray,
    mode: np.ndarray,
    side: np.ndarray,
    multipliers: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    columns: _Columns,
    variances: np.ndarray,
) -> dict[str, float]:
    r"""
    Return how far the mode is from each optimality condition, in the units of the state.

    ``feasible``: how far an element is outside its bounds. ``stationary``:
    :math:`|x - \hat{x} - \hat{S}_{:a} \lambda_a|`, which is zero when the
    gradient :math:`P (x - \hat{x})` is :math:`\lambda`, zero at the free
    elements; at the elements on a bound it is how far :math:`\hat{x}_a +
    \hat{S}_{aa} \lambda_a` is from :math:`b_a`, the error of the solve that
    made :math:`\lambda_a`. ``sign``: how far a multiplier is
    of the wrong sign, as :math:`\lambda_i \hat{S}_{ii}`. ``complementary``:
    :math:`|\lambda_i \hat{S}_{ii}|` times how far element :math:`i` is from
    its bound.
    """
    active = np.flatnonzero(side)
    implied = x_hat.copy()
    if len(active):
        implied += columns(active) @ multipliers[active]
    bound = np.where(side > 0, lower, np.where(side < 0, upper, mode))
    pull = side * multipliers * variances
    return {
        "feasible": float(np.maximum(lower - mode, mode - upper).max(initial=0.0)),
        "stationary": float(np.abs(implied - mode).max(initial=0.0)),
        "sign": float(np.maximum(-pull, 0.0).max(initial=0.0)),
        "complementary": float(
            np.abs(multipliers * variances * (mode - bound)).max(initial=0.0)
        ),
    }


def _first_hit(
    alpha: np.ndarray, beta: np.ndarray, gap: np.ndarray, gap2: np.ndarray
) -> tuple[float, int]:
    r"""
    Return when the first of the paths :math:`\alpha_k \cos t + \beta_k \sin t` falls below its `gap`, and which (``inf`` and ``-1`` for none).

    With :math:`\alpha \cos t + \beta \sin t = r \cos(t - \phi)`, a path
    reaches its gap only if :math:`r^2 > \text{gap}^2` (`gap2`), and leaves
    :math:`\{\ge \text{gap}\}` at :math:`t = \phi + \theta` (modulo
    :math:`2\pi`), :math:`\theta = \arccos(\text{gap} / r)`, where it falls
    (at :math:`\phi - \theta` it comes back). By the sum of the two angles,
    :math:`t` is the angle of :math:`(\alpha\, \text{gap} - \beta q,\ \beta\,
    \text{gap} + \alpha q)`, :math:`q = \sqrt{r^2 - \text{gap}^2}`: one
    ``arctan2`` and one square root, for the paths that reach their gap
    only. A path already below its gap and falling leaves now (:math:`t = 0`:
    roundoff put it there); a root within roundoff of 0 is the one a path
    has just come back through, and is skipped.
    """
    reach = np.flatnonzero(alpha * alpha + beta * beta > gap2)
    if not len(reach):
        return math.inf, -1
    a, b, g = alpha[reach], beta[reach], gap[reach]
    leaving = np.flatnonzero((a <= g) & (b < 0.0))
    if len(leaving):
        return 0.0, int(reach[leaving[0]])
    q = np.sqrt(a * a + b * b - g * g)
    t = np.mod(np.arctan2(b * g + a * q, a * g - b * q), 2.0 * np.pi)
    t[t < 1e-12] = math.inf
    k = int(np.argmin(t))
    return float(t[k]), int(reach[k])


def _reflect(velocity: np.ndarray, column: np.ndarray, i: int) -> np.ndarray:
    r"""
    Return the velocity reflected off the bound of element `i`, in the metric of :math:`\hat{S}`.

    In the whitened state :math:`y` (:math:`x = \hat{x} + R y`, :math:`R R^T
    = \hat{S}`), the bound of element :math:`i` is a plane with normal
    :math:`f = R^T e_i`, and the velocity reflects off it:
    :math:`\dot{y}' = \dot{y} - 2 (f^T \dot{y} / f^T f) f`. In the state's
    own coordinates (:math:`\dot{x} = R \dot{y}`) that is
    :math:`\dot{x}' = \dot{x} - 2 (\dot{x}_i / \hat{S}_{ii}) \hat{S}_{:i}`:
    one column of :math:`\hat{S}` (`column`), and no root of it.
    """
    return velocity - (2.0 * velocity[i] / column[i]) * column


class _Sampler:
    r"""
    Exact Hamiltonian Monte Carlo for the Gaussian :math:`N(\hat{x}, \hat{S})` truncated to the box (Pakman and Paninski, 2014).

    In the whitened state the Hamiltonian is :math:`\tfrac{1}{2} y^T y +
    \tfrac{1}{2} s^T s`, and its paths are :math:`y(t) = y_0 \cos t + s \sin
    t`: no step size, no rejection. In the state's own coordinates a path is

    .. math::

        x(t) = \hat{x} + (x_0 - \hat{x}) \cos t + v \sin t, \qquad v \sim N(0, \hat{S}),

    so each element's time to reach its bound has a closed form
    (:func:`_first_hit`), and at the first bound reached the velocity reflects
    (:func:`_reflect`), keeping the energy. Each draw starts from the last
    with a new velocity and travels :math:`\pi / 2` (Pakman and Paninski's
    choice): without bounds :math:`x(\pi/2) = \hat{x} + v`, a draw independent
    of the start. The velocities are the analytic method's exact draws less
    :math:`\hat{x}` (``analytic.sample``), :data:`_BLOCK` at a time; only the
    bounded elements are checked for a hit, and only the columns of the
    elements that hit are used, from ``columns``.
    """

    def __init__(
        self,
        analytic: "Whitened | Information",
        lower: np.ndarray,
        upper: np.ndarray,
        columns: _Columns,
    ):
        self.analytic = analytic
        self.mean = mean = analytic.estimate()
        self.lower, self.upper = lower, upper
        self.columns = columns
        self.std = std = np.sqrt(np.maximum(analytic.variances(), 0.0))
        # Each finite bound is a wall: the element, +1 (lower) or -1 (upper), and
        # the gap, so that sign * (x_i - mean_i) >= gap inside. An element the
        # posterior fixes (no variance) cannot move, and is left out.
        moves = std > 0
        low = np.flatnonzero(np.isfinite(lower) & moves)
        high = np.flatnonzero(np.isfinite(upper) & moves)
        self.cell = np.concatenate([low, high])
        self.sign = np.concatenate([np.ones(len(low)), -np.ones(len(high))])
        self.bound = np.concatenate([lower[low], upper[high]])
        self.gap = self.sign * (self.bound - mean[self.cell])
        self.gap2 = self.gap**2
        # The walls of one block's lower bounds, the usual case, are a slice
        cell = self.cell
        contiguous = len(cell) > 0 and np.array_equal(
            cell, np.arange(cell[0], cell[0] + len(cell))
        )
        self.walls = (
            slice(int(cell[0]), int(cell[0]) + len(cell)) if contiguous else cell
        )
        self.lower_only = bool(np.all(self.sign > 0))
        # The columns of the elements that will hit, fetched in blocks first
        near = self.sign * (mean[self.cell] - self.bound) < _NEAR * std[self.cell]
        columns.fetch(self.cell[near])

    def start(self, mode: np.ndarray) -> np.ndarray:
        """Return the mode moved strictly inside the box, by a thousandth of a standard deviation (at most a quarter of the box)."""
        lower, upper = self.lower, self.upper
        inside = np.minimum(1e-3 * self.std, 0.25 * (upper - lower))
        return np.clip(mode, lower + inside, upper - inside)

    def run(
        self, start: np.ndarray, N: int, burn_in: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return `N` draws (``n x N``) after `burn_in`, from `start`, with velocities drawn from `rng`."""
        n = len(self.mean)
        draws = np.empty((n, N))
        x = start
        total = burn_in + N
        done = 0
        while done < total:
            V = self.analytic.sample(min(_BLOCK, total - done), rng)
            V -= self.mean[:, None]
            for j in range(V.shape[1]):
                x = self._travel(x, V[:, j])
                if done >= burn_in:
                    draws[:, done - burn_in] = x
                done += 1
        return draws

    def _travel(self, x: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        r"""Return where the path from `x` with `velocity` is after :math:`\pi / 2`, reflecting off each bound it reaches."""
        mean, cell, walls, sign = self.mean, self.cell, self.walls, self.sign
        gap, gap2, lower_only = self.gap, self.gap2, self.lower_only
        a, v = x - mean, velocity
        left = 0.5 * np.pi
        most = 1000 + 100 * len(cell)
        for _ in range(most):
            if lower_only:
                t, k = _first_hit(a[walls], v[walls], gap, gap2)
            else:
                t, k = _first_hit(sign * a[walls], sign * v[walls], gap, gap2)
            if not t < left:
                position = mean + a * np.cos(left) + v * np.sin(left)
                return np.clip(position, self.lower, self.upper)
            i = int(cell[k])
            cos, sin = np.cos(t), np.sin(t)
            a, v = a * cos + v * sin, v * cos - a * sin
            a[i] = self.bound[k] - mean[i]
            v = _reflect(v, self.columns.column(i), i)
            left -= t
        raise RuntimeError(
            "A draw of the truncated posterior reflected off the bounds more than "
            f"{most} times."
        )


@dataclass(frozen=True, eq=False)
class Bounded:
    r"""
    The bounded method's solution: the mode of the truncated posterior, and its draws.

    It keeps the analytic factors (``analytic``), the bounds of each state
    element, the mode, which bound each element is held at, and the
    multipliers :math:`\lambda`, with :math:`x = \hat{x} + \hat{S}_{:a}
    \lambda_a` (``_mode``). The :class:`~fips.Posterior` asks it for every
    array it reports: it is the bounded method's
    :class:`~fips.posterior.Solution`. The draws of the truncated posterior
    are made at their first use and kept (not pickled: a posterior read back
    draws them again, the same, from the seed).

    Its answers, by what each needs:

    - the estimate is the mode, not the mean, and is not linear in the
      observations (no ``estimate_is_mean``, no ``estimate_is_linear``);
    - the variances and the covariance of aggregates are those of the kept
      draws (:meth:`kept`: ``draws`` of them, after ``burn_in``, from
      ``seed``), and so are the posterior's mean and intervals (the draws'
      quantiles); :meth:`sample` draws more;
    - the gain, DOFS, DOFS per element and the information content are those
      of the analytic posterior with the elements on a bound held there,
      :math:`x_a = b_a` (the Gaussian at the mode), with :math:`C =
      \hat{S}_{aa}`:

      .. math::

          K_a = K - \hat{S}_{:a} C^{-1} K_{a:}, \qquad
          A_a = K_a H = A - \hat{S}_{:a} C^{-1} A_{a:};

      an element on a bound has a row of zeros in both, and no DOFS;
    - the chi-squared and the misfits are the cost's terms at the mode.
    """

    analytic: "Whitened | Information"
    """The analytic factors of the problem, read at the solve's gamma."""
    lower: np.ndarray
    """The lower bound of each state element (``-inf`` for none)."""
    upper: np.ndarray
    """The upper bound of each state element (``inf`` for none)."""
    mode: np.ndarray
    """The mode in the box."""
    side: np.ndarray
    """``1`` where an element is held at its lower bound, ``-1`` at its upper bound, ``0`` where it is free."""
    multipliers: np.ndarray
    r""":math:`\lambda`, zero at the free elements: :math:`x = \hat{x} + \hat{S} \lambda`."""
    draws: int
    """The number of draws kept."""
    burn_in: int
    """The draws left out before those kept."""
    seed: int
    """The seed of the kept draws."""

    @classmethod
    def solve(
        cls,
        analytic: "Whitened | Information",
        lower: np.ndarray,
        upper: np.ndarray,
        draws: int,
        burn_in: int,
        seed: int,
        columns: "_Columns | None" = None,
    ) -> "Bounded":
        """
        Return the solution of the analytic factors held to the bounds: the mode, by the active set, checked.

        `columns` is a cache of the posterior covariance's columns to share
        (another solution of the same factors with other observations).

        Raises
        ------
        numpy.linalg.LinAlgError
            If the active set fails (``_mode``), or the mode it ends at does not
            meet the optimality conditions to ``sqrt(eps)`` of the scale of the
            state (the largest of its values, bounds and posterior standard
            deviations).
        """
        if columns is None:
            columns = _Columns(analytic)
        x_hat = analytic.estimate()
        variances = analytic.variances()
        scale = _scale(x_hat, variances, lower, upper)
        mode, side, multipliers = _mode(x_hat, lower, upper, columns, variances, scale)
        conditions = _kkt(
            x_hat, mode, side, multipliers, lower, upper, columns, variances
        )
        worst = max(conditions, key=lambda name: conditions[name])
        if conditions[worst] > _KKT_RTOL * scale:
            raise LinAlgError(
                "The bounded mode does not meet the optimality conditions: it is "
                f"off by {conditions[worst]:.3g} in the '{worst}' condition, where "
                f"{_KKT_RTOL * scale:.3g} is allowed. The posterior covariance of "
                "the elements on a bound is too ill-conditioned to hold them there."
            )
        logger.debug(
            "Bounded mode: %d of %d elements on a bound; optimality conditions %s.",
            int(np.count_nonzero(side)),
            len(side),
            conditions,
        )
        solution = cls(
            analytic=analytic,
            lower=_read_only(lower),
            upper=_read_only(upper),
            mode=_read_only(np.clip(mode, lower, upper)),
            side=_read_only(side),
            multipliers=_read_only(multipliers),
            draws=draws,
            burn_in=burn_in,
            seed=seed,
        )
        solution.__dict__["_columns_"] = columns
        return solution

    # -- the caches, outside the frozen fields ----------------------------------

    def _columns(self) -> _Columns:
        """Return the kept columns of the posterior covariance (a new cache after a pickle)."""
        columns = self.__dict__.get("_columns_")
        if columns is None:
            columns = self.__dict__["_columns_"] = _Columns(self.analytic)
        return columns

    def _sampler(self) -> _Sampler:
        """Return the sampler of this solution's truncated posterior, made once."""
        sampler = self.__dict__.get("_sampler_")
        if sampler is None:
            sampler = _Sampler(self.analytic, self.lower, self.upper, self._columns())
            self.__dict__["_sampler_"] = sampler
        return sampler

    def kept(self) -> np.ndarray:
        """Return the kept draws (``n x draws``), which the variances, the covariances, the mean and the intervals are read from: drawn from the seed at the first call, with a warning if they mix too slowly for their variances."""
        kept = self.__dict__.get("_draws")
        if kept is None:
            logger.debug(
                "Drawing %d draws of the truncated posterior (after %d left out).",
                self.draws,
                self.burn_in,
            )
            kept = self.sample(self.draws, np.random.default_rng(self.seed))
            kept = self.__dict__["_draws"] = _read_only(kept)
            _warn_if_imprecise(kept, logger, _SLOW_DRAWS)
        return kept

    def _conditioned(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return the active elements :math:`a`, :math:`\hat{S}_{:a}` and :math:`\hat{S}_{:a} C^{-1}` (``n x |a|``), made once."""
        kept = self.__dict__.get("_conditioned_")
        if kept is None:
            active = np.flatnonzero(self.side)
            S_a = self._columns()(active)
            # S_:a C^-1, the transpose of C^-1 S_a:
            Y = _conditioned_solve(S_a[active], S_a.T).T if len(active) else S_a
            kept = self.__dict__["_conditioned_"] = (active, S_a, Y)
        return kept

    def __getstate__(self) -> dict:
        """Return the fields for a pickle or a copy, without the caches (the draws are made again, the same, from the seed)."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ("_columns_", "_sampler_", "_draws", "_conditioned_")
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

    def at_bound(self) -> np.ndarray:
        """Return which elements are held at a bound at the mode (boolean, ``n``)."""
        return self.side != 0

    @property
    def evidence_is_laplace(self) -> bool:
        """
        Whether the log evidence is the Gaussian one: only with no finite bound, when the posterior is the analytic one.

        With a bound, the exact evidence is the Gaussian one times the ratio
        of the probabilities of the box under the posterior and under the
        prior, orthant probabilities of a correlated Gaussian, and a Gaussian
        at a mode on the bound would not give it.
        """
        return not (np.isfinite(self.lower).any() or np.isfinite(self.upper).any())

    no_evidence: ClassVar[str] = (
        "with bounds on the state, the exact evidence is the Gaussian one "
        "times the probability of the bounds under the posterior over their "
        "probability under the prior, both orthant probabilities of a "
        "correlated Gaussian, which fips does not compute. The evidence of the "
        "problem without bounds is fips.solve(problem).log_evidence."
    )
    """Why a bounded posterior gives no log evidence."""

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """Return `N` new draws of the truncated posterior (``n x N``): a chain from the mode, its first ``burn_in`` left out."""
        sampler = self._sampler()
        return sampler.run(sampler.start(self.mode), N, self.burn_in, rng)

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

    def _gain_rows(self, active: np.ndarray) -> np.ndarray:
        r"""Return the rows of the analytic gain at `active`, :math:`K_{a:}` (``|a| x m``)."""
        n = len(self.mode)
        E = np.zeros((len(active), n))
        E[np.arange(len(active)), active] = 1.0
        return self.analytic.gain(DenseOperator._adopt(E))

    def gain(self, W: LinearOperator | None = None) -> np.ndarray:
        r"""Return the gain at the mode, :math:`K - \hat{S}_{:a} C^{-1} K_{a:}`, or :math:`W` times it."""
        active, S_a, Y = self._conditioned()
        K = self.analytic.gain(W)
        if not len(active):
            return K
        if W is not None:
            return K - np.asarray(W @ Y, dtype=float) @ self._gain_rows(active)
        K = K - Y @ self._gain_rows(active)
        K[active] = 0.0  # exactly: the bounds, not the data, decide these elements
        return K

    def _kernel_rows(self, active: np.ndarray) -> np.ndarray:
        r"""Return the rows of the analytic averaging kernel at `active`, :math:`A_{a:} = K_{a:} H` (``|a| x n``)."""
        H = self.analytic.problem._operator.data
        return np.asarray(H._transpose() @ self._gain_rows(active).T, dtype=float).T

    def dofs(self) -> float:
        r"""Return :math:`\operatorname{tr}(A_a) = \operatorname{tr}(A) - \operatorname{tr}(C^{-1} A_{a:} \hat{S}_{:a})`."""
        active, S_a, Y = self._conditioned()
        dofs = self.analytic.dofs()
        if not len(active):
            return dofs
        return dofs - float(np.einsum("ik,ki->", Y, self._kernel_rows(active)))

    def dofs_per_element(self) -> np.ndarray:
        r"""Return :math:`\operatorname{diag}(A_a) = \operatorname{diag}(A) - \operatorname{diag}(\hat{S}_{:a} C^{-1} A_{a:})`: zero on a bound."""
        active, S_a, Y = self._conditioned()
        diagonal = self.analytic.dofs_per_element()
        if not len(active):
            return diagonal
        diagonal = diagonal - np.einsum("ik,ki->i", Y, self._kernel_rows(active))
        diagonal[active] = 0.0  # exactly, as the gain's rows
        return diagonal

    def information(self) -> float:
        r"""
        Return the information content at the mode, :math:`-\tfrac{1}{2} \ln \lvert I - A_a \rvert`.

        By the matrix determinant lemma, with :math:`(I - A)^{-1} \hat{S} =
        S_0`, :math:`\lvert I - A_a \rvert = \lvert I - A \rvert \,
        \lvert S_{0,aa} \rvert / \lvert C \rvert`: the analytic information
        less what the observations tell of the elements on a bound, which the
        bounds fix. It is the information on the free elements given those.
        """
        active, S_a, Y = self._conditioned()
        information = self.analytic.information()
        if not len(active):
            return information
        prior = self.analytic.problem.prior_cov.data._select(active, active).dense()
        return information - 0.5 * (_logdet(prior) - _logdet(S_a[active]))

    def _residual_weights(self) -> np.ndarray:
        r"""
        Return :math:`u_a = u - (H S_0 H^T + S_z/\gamma)^{-1} H S_0 \lambda`, with :math:`u` the analytic method's.

        The residual at the mode is :math:`(S_z / \gamma) u_a` and its
        increment :math:`x - x_0 = S_0 (H^T u_a + \lambda)`: since :math:`x -
        \hat{x} = \hat{S} \lambda = S_0 (\lambda - H^T P H S_0 \lambda)`, with
        :math:`P` the inverse.
        """
        u = self.analytic._innovation_precision_times()
        problem = self.analytic.problem
        HS_0l = np.asarray(
            problem._operator.data @ (problem.prior_cov.data @ self.multipliers),
            dtype=float,
        )
        return u - self.analytic._precision_times(HS_0l[:, None])[:, 0]

    def misfits(self) -> tuple[float, float]:
        r"""
        Return the data misfit :math:`r^T S_z^{-1} r` and the state misfit :math:`(x - x_0)^T S_0^{-1} (x - x_0)` at the mode.

        With :math:`u_a` (``_residual_weights``), :math:`r = (S_z/\gamma) u_a`
        and :math:`x - x_0 = S_0 y`, :math:`y = H^T u_a + \lambda`: the data
        misfit is :math:`u_a^T S_z u_a / \gamma^2` and the state misfit
        :math:`y^T S_0 y`, with no inverse of either covariance.
        """
        problem = self.analytic.problem
        u = self._residual_weights()
        gamma = self.analytic.gamma
        data = float(u @ np.asarray(problem.obs_cov.data @ u)) / gamma**2
        y = np.asarray(problem._operator.data._transpose() @ u) + self.multipliers
        state = float(y @ np.asarray(problem.prior_cov.data @ y))
        return data, state

    def chi2(self) -> float:
        r"""Return the cost at the mode, :math:`\gamma` times the data misfit plus the state misfit."""
        data, state = self.misfits()
        return self.analytic.gamma * data + state

    def chi2_tests(self) -> dict[str, tuple[float, int]]:
        r"""
        Return Rodgers' four chi-squared tests at the mode.

        The innovation does not depend on the estimate: its test is the
        analytic one. The residual at the mode, :math:`(S_z/\gamma) u_a`,
        against :math:`C = (S_z/\gamma) G^{-1} (S_z/\gamma)` gives
        :math:`u_a^T G u_a`. The increment, :math:`S_0 y`, and the modeled
        increment are measured in the directions the observations constrain
        (``analytic._increment_chi2_at``), where both give the same number;
        the part of the increment the bounds alone move is left out. The
        chi-squared values are those of the mode; their degrees of freedom are
        the analytic ones, and the p-values from them are approximate, since
        the mode is not a linear function of Gaussian data.
        """
        problem = self.analytic.problem
        gamma = self.analytic.gamma
        m = len(problem.obs)
        u = self._residual_weights()
        Htu = np.asarray(problem._operator.data._transpose() @ u)
        residual = (
            float(Htu @ np.asarray(problem.prior_cov.data @ Htu))
            + float(u @ np.asarray(problem.obs_cov.data @ u)) / gamma
        )
        increment, rank = self.analytic._increment_chi2_at(Htu + self.multipliers)
        return {
            "innovation": (self.analytic.chi2(), m),
            "residual": (residual, m),
            "modeled_increment": (increment, rank),
            "increment": (increment, rank),
        }

    def with_gamma(self, gamma: float) -> "Bounded":
        """Return the solution at `gamma`: the active set again, on the analytic factors read at `gamma`."""
        return type(self).solve(
            self.analytic.with_gamma(gamma),
            self.lower,
            self.upper,
            self.draws,
            self.burn_in,
            self.seed,
        )

    def with_data(self, problem: Problem) -> "Bounded":
        """Return the solution of `problem`, this one's with other observation values: the active set again, sharing the kept columns."""
        return type(self).solve(
            self.analytic.with_data(problem),
            self.lower,
            self.upper,
            self.draws,
            self.burn_in,
            self.seed,
            columns=self._columns(),
        )


def _logdet(S: np.ndarray) -> float:
    """Return the log-determinant of a positive-definite matrix, by Cholesky."""
    factor, _ = cho_factor(0.5 * (S + S.T), lower=True)
    return 2.0 * float(np.log(np.diagonal(factor)).sum())


def _warn_if_every_block(options: BoundedOptions, state: Space) -> None:
    """Log a warning when a finite bound given as a number bounds a state of several blocks: every block, a bias block too, which is rarely meant."""
    if len(state.names) < 2:
        return
    for side in ("lower", "upper"):
        bound = getattr(options, side)
        if isinstance(bound, float) and math.isfinite(bound):
            logger.warning(
                "%s=%g bounds every block of the state: %s. A bias or an offset "
                "block is usually free; to bound only some blocks, give a dict, "
                "such as %s={%r: %g}.",
                side,
                bound,
                _listed(state.names),
                side,
                state.names[0],
                bound,
            )


def _bounded(problem: Problem, options: BoundedOptions) -> Posterior:
    """Factor the problem as the analytic method does, then hold the estimate to the bounds; with no finite bound, the analytic posterior."""
    if not isinstance(problem.forward, Operator):
        raise TypeError(
            "The bounded method solves a linear problem, and this problem's "
            f"forward is a model, {type(problem.forward).__name__}: bounds with a "
            'forward model are not available. Solve it with "gauss_newton", '
            "without bounds, or write the model on a state that cannot leave "
            "the bounds (the logarithm of a positive flux)."
        )
    state = problem.prior.space
    _warn_if_every_block(options, state)
    lower = _resolve(options.lower, state, "lower")
    upper = _resolve(options.upper, state, "upper")
    crossed = ~(lower < upper)
    if crossed.any():
        where = "; ".join(
            f"{block!r}: {_listed(state.select(block)[crossed[state.slice(block)]])}"
            for block in state.names
            if crossed[state.slice(block)].any()
        )
        raise ValueError(
            "lower must be below upper at every state element, and is not in "
            f"block {where}: an element held at one value is not bounded but known."
        )
    analytic = _factor(problem, options.gamma)
    if not (np.isfinite(lower).any() or np.isfinite(upper).any()):
        return Posterior(problem, analytic, options)
    solution = Bounded.solve(
        analytic, lower, upper, options.draws, options.burn_in, options.seed
    )
    return Posterior(problem, solution, options)
