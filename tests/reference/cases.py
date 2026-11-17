"""
The reference problems, each written twice: as a fips problem, and as NumPy arrays with its log density.

Every problem here is ``z = H T(u) + c + e``, ``u ~ N(u_0, S_0)``,
``e ~ N(0, S_z)``, with ``T`` element-wise: ``T(u) = r e^u`` (``"exp"``) or
``T(u) = s log(1 + e^{u/s})`` (``"softplus"``). The arrays are what a
reference sampler draws (:mod:`tests.reference.generate`), what a grid
integrates, and where a calibration world's truth and data are drawn from:
none of that runs fips code. The fips problem is what the suite solves.

- :data:`CURVES`: issue #203's two curves of two elements, ``e^a + 2 e^b``
  (a :class:`~fips.ForwardModel` the test writes) and the user guide's two
  cells (:class:`~fips.LogNormalForward`), against their grids.
- :data:`OUTSIDE`: the four ten-element problems of
  ``tests/test_outside_sampler.py``, lognormal and softplus, mild and strong.
- :data:`SWEEP`: problems of 100 to 900 elements, both transforms: an area of
  square cells seen by 120 observations of three receptors, each looking
  upwind in a random direction, around a base setting (400 cells, error 0.3,
  log standard deviation 1, prior correlated over 2 cells), with one setting
  changed at a time.
"""

import dataclasses
import functools
import math

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.optimize import minimize

import fips
from fips import Covariance, Operator, Problem, Space, Vector
from tests.test_gauss_newton import two_cells, two_exp
from tests.test_outside_sampler import REGIMES, Transformed


@dataclasses.dataclass(frozen=True, eq=False)
class Arrays:
    """``z = H T(u) + c + e`` as arrays, in the fips problem's order of the state and the observations."""

    u_0: np.ndarray
    S_0: np.ndarray
    H: np.ndarray
    S_z: np.ndarray
    z: np.ndarray
    c: np.ndarray
    transform: str
    """``"exp"`` (``T(u) = r e^u``) or ``"softplus"`` (``T(u) = s log(1 + e^{u/s})``)."""
    r: np.ndarray
    s: float
    aggregates: dict[str, np.ndarray]
    """Each aggregate's weights on the values ``T(u)``: a sum of some of them."""

    def value(self, u: np.ndarray) -> np.ndarray:
        """``T(u)``, element-wise, for a state (``n``) or draws (``n x k``)."""
        r = self.r if u.ndim == 1 else self.r[:, None]
        if self.transform == "exp":
            return r * np.exp(u)
        return self.s * np.logaddexp(0.0, u / self.s)

    def slope(self, u: np.ndarray) -> np.ndarray:
        """``T'(u)``, element-wise, for a state (``n``)."""
        if self.transform == "exp":
            return self.r * np.exp(u)
        return 0.5 * (1.0 + np.tanh(0.5 * u / self.s))  # the logistic function

    @functools.cached_property
    def precisions(self) -> tuple[np.ndarray, np.ndarray]:
        """The inverses of ``S_0`` and ``S_z``."""
        return np.linalg.inv(self.S_0), np.linalg.inv(self.S_z)

    def log_density(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """The log posterior at `u` (``n``), up to a constant, and its gradient."""
        P_0, P_z = self.precisions
        d = u - self.u_0
        r = self.z - self.H @ self.value(u) - self.c
        P_0d, P_zr = P_0 @ d, P_z @ r
        value = -0.5 * float(d @ P_0d) - 0.5 * float(r @ P_zr)
        return value, -P_0d + self.slope(u) * (self.H.T @ P_zr)

    def laplace(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The mode, by L-BFGS on the log density, and the covariance of the Gaussian there (the prior's precision and the linearized data's).

        Only a reference sampler's coordinates are made from it, so a mode
        found loosely costs it draws, not accuracy.
        """

        def cost(u):
            value, gradient = self.log_density(u)
            return -value, -gradient

        found = minimize(
            cost,
            self.u_0,
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=50_000, maxcor=50, gtol=1e-9, ftol=1e-15),
        )
        P_0, P_z = self.precisions
        J = self.H * self.slope(found.x)
        cov = np.linalg.inv(P_0 + J.T @ P_z @ J)
        return found.x, 0.5 * (cov + cov.T)

    def world(self, seed: int) -> tuple[np.ndarray, "Arrays"]:
        """A truth drawn from the prior and data drawn from it, by NumPy from `seed`: the truth and the arrays with those data."""
        rng = np.random.default_rng([seed, 7919])
        truth = self.u_0 + np.linalg.cholesky(self.S_0) @ rng.standard_normal(
            len(self.u_0)
        )
        noise = np.linalg.cholesky(self.S_z) @ rng.standard_normal(len(self.z))
        z = self.H @ self.value(truth) + self.c + noise
        return truth, dataclasses.replace(self, z=z)


@dataclasses.dataclass(frozen=True, eq=False)
class Case:
    """A reference problem: its fips problem and its arrays, which describe the same posterior."""

    name: str
    problem: Problem
    arrays: Arrays

    def with_data(self, z: np.ndarray) -> Problem:
        """The fips problem with the observations `z`."""
        p = self.problem
        return Problem(
            prior=p.prior,
            prior_cov=p.prior_cov,
            obs=Vector(z, p.obs.space),
            obs_cov=p.obs_cov,
            forward=p.forward,
            offset=p.offset,
        )

    def state_aggregation(self) -> Operator:
        """The aggregates' weights as an aggregation of the state (``u``), which :meth:`fips.Posterior.cov` checks the chains on."""
        names = list(self.arrays.aggregates)
        W = np.array([self.arrays.aggregates[name] for name in names])
        rows = Space({"aggregate": pd.Index(names, name="name")})
        return Operator(W, rows, self.problem.prior.space)


def _pair(problem: Problem, weights: list[float], name: str) -> Case:
    """One of issue #203's curves, ``weights @ exp(u)`` observed once: the fips problem as the tests write it, and its arrays."""
    arrays = Arrays(
        u_0=problem.prior.values.copy(),
        S_0=problem.prior_cov.dense(),
        H=np.array([weights]),
        S_z=problem.obs_cov.dense(),
        z=problem.obs.values.copy(),
        c=np.zeros(1),
        transform="exp",
        r=np.ones(2),
        s=1.0,
        aggregates={"total": np.asarray(weights, dtype=float)},
    )
    return Case(name, problem, arrays)


def _outside(regime: str) -> Case:
    """One of ``tests/test_outside_sampler.py``'s problems, at its seed 7."""
    made = Transformed.drawn(**REGIMES[regime], seed=7)
    n, m = len(made.u_0), len(made.z)
    arrays = Arrays(
        u_0=made.u_0,
        S_0=made.S_0,
        H=made.H,
        S_z=made.S_z,
        z=made.z,
        c=np.zeros(m),
        transform="exp" if made.kind == "lognormal" else "softplus",
        r=np.ones(n),
        s=made.scale,
        aggregates={"total": np.ones(n)},
    )
    return Case(regime, made.problem(), arrays)


BASE = dict(side=20, m=120, receptors=3, noise=0.3, log_std=1.0, length=2.0)
"""The sweep's base setting: 400 cells, 120 observations of three receptors, error 0.3, log standard deviation 1, prior correlated over 2 cells."""

SETTINGS = {
    "400": {},
    "100": dict(side=10),
    "900": dict(side=30),
    "logstd1.5": dict(log_std=1.5),
    "noise0.1": dict(noise=0.1),
}
"""The settings of the sweep the suite holds, each one change from :data:`BASE`; a problem's name is the transform's and the setting's, ``"lognormal-400"``."""


def _sweep(
    name: str, kind: str, side, m, receptors, noise, log_std, length, world=0
) -> Case:
    """
    A problem of the sweep: ``side x side`` cells seen by `m` observations of `receptors` receptors, its truth and data drawn by :meth:`Arrays.world` from `world`.

    Each observation sees the cells upwind of its receptor, in a direction
    drawn for it, falling off across the wind and along it; its row sums to
    5. A lognormal prior is a factor on a reference value of each cell (the
    "inventory", drawn lognormal) whose value has a coefficient of variation
    of ``sqrt(e^{log_std^2} - 1)``; a softplus prior has that mean and
    standard deviation, its scale the inventory's median. The prior is
    correlated as ``exp(-distance / length)``, the observations are
    independent with error `noise`, and their offset is 1900.
    """
    rng = np.random.default_rng(0)
    n = side * side
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    xy = np.c_[gx.ravel(), gy.ravel()].astype(float)
    cells = Space({"value": pd.MultiIndex.from_arrays(xy.T, names=["x", "y"])})
    obs = Space({"obs": pd.RangeIndex(m, name="obs_id")})
    receptor = rng.uniform(0, side - 1, (receptors, 2))
    which = rng.integers(0, receptors, m)
    H = np.zeros((m, n))
    for i in range(m):
        angle = rng.uniform(0, 2 * np.pi)
        d = xy - receptor[which[i]]
        along = d @ np.array([np.cos(angle), np.sin(angle)])
        across = d @ np.array([-np.sin(angle), np.cos(angle)])
        H[i] = np.exp(-np.abs(across) - np.maximum(along, 0) / 3.0) * (along > -1)
    H[H < 0.02] = 0.0
    H = H / H.sum(axis=1, keepdims=True) * 5.0
    inventory = pd.Series(rng.lognormal(0.0, 0.8, n), index=cells.index)
    H_op = Operator(sps.csr_array(H), obs, cells)
    cv = math.sqrt(math.expm1(log_std**2))
    if kind == "lognormal":
        model = fips.LogNormalForward(H_op, inventory)
        mu, sigma = model.prior_for(model.reference, model.reference * cv)
        r, s = inventory.to_numpy(), 1.0
    else:
        scale = float(inventory.median())
        model = fips.SoftplusForward(H_op, scale=scale)
        mu, sigma = model.prior_for(inventory, inventory * cv)
        r, s = np.ones(n), scale
    distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    std = sigma.values
    S_0 = std[:, None] * np.exp(-distance / length) * std[None, :]
    half = side / 2
    west, south = xy[:, 0] < half, xy[:, 1] < half
    aggregates = {
        "total": np.ones(n),
        "seen": (H > 0).any(axis=0).astype(float),
        "sw": (west & south).astype(float),
        "se": (~west & south).astype(float),
        "nw": (west & ~south).astype(float),
        "ne": (~west & ~south).astype(float),
    }
    arrays = Arrays(
        u_0=mu.values.copy(),
        S_0=S_0,
        H=H,
        S_z=noise**2 * np.eye(m),
        z=np.zeros(m),
        c=np.full(m, 1900.0),
        transform="exp" if kind == "lognormal" else "softplus",
        r=r,
        s=s,
        aggregates=aggregates,
    )
    _, arrays = arrays.world(world)
    problem = Problem(
        prior=mu,
        prior_cov=Covariance(S_0, cells),
        obs=Vector(arrays.z, obs),
        obs_cov=fips.Independent().with_std(noise),
        forward=model,
        offset=1900.0,
    )
    return Case(name, problem, arrays)


CURVES = ("two_exp", "two_cells")
"""Issue #203's two curves, held to their grids."""

OUTSIDE = tuple(REGIMES)
"""The ten-element problems of ``tests/test_outside_sampler.py``, held to emcee and to nutpie."""

SWEEP = tuple(
    f"{kind}-{name}" for kind in ("lognormal", "softplus") for name in SETTINGS
)
"""The sweep's problems, held to nutpie."""


@functools.cache
def case(name: str) -> Case:
    """The reference problem `name`, built once a process."""
    if name == "two_exp":
        return _pair(two_exp(), [1.0, 2.0], name)
    if name == "two_cells":
        return _pair(two_cells(), [1.0, 1.0], name)
    if name in REGIMES:
        return _outside(name)
    kind, _, setting = name.partition("-")
    if kind in ("lognormal", "softplus") and setting in SETTINGS:
        return _sweep(name, kind, **{**BASE, **SETTINGS[setting]})
    raise KeyError(f"no reference problem {name!r}")
