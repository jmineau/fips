"""
Tests for ``fips.LognormalForward`` and ``fips.SoftplusForward``: forward models that keep one block of the state positive.

A wrong Jacobian, a wrong prior correspondence or a misread posterior still
gives a labeled answer of the right shape, so each is held to an answer that
does not use the code under test, in layers:

1. **The Taylor test** of each model's Jacobian: dense and sparse ``H``, a
   bias block that passes through, labels in another order, a zero reference.
2. **Hand-written models.** ``LognormalForward`` gives the mode and the
   posterior of the user guide's ``ScaledInventory``; both models' values and
   Jacobians are the formulas written with NumPy.
3. **The mode against an independent optimizer**: SciPy's ``least_squares``
   on the whitened residual of the cost, polished by Newton's method with the
   cost's whole Hessian, with fewer and more state elements than
   observations, a bias block and a Kronecker prior.
4. **The prior correspondence**: ``prior_for`` gives a ``u`` whose flux has
   the mean and standard deviation asked for, against the closed form
   (lognormal), SciPy's adaptive ``quad`` (softplus) and Monte Carlo draws
   (both); the lognormal's flux correlations against their formula.
5. **Its own world**, through each trial's posterior and its draws through
   the transform: the intervals of the cells and of the total cover about as
   often as they claim when the prior is narrow.
6. **Wrong variants** (a Jacobian without the reference, the median
   convention in place of the mean, softplus's Jacobian as ``exp(u)``) fail
   the tests above.
7. **Refusals and pickling.**
8. **Newton under the transform.** The curvature is the derivative of
   ``J^T v`` by differences; the mode is Gauss-Newton's alone (both analytic
   forms, a bias block, a singular prior, labels in another order, an
   aggregation) and SciPy's BFGS minimum; each step is the dense iteration's
   (Newton's with the whole Hessian, or Gauss-Newton's), which a curvature
   with its sign flipped, doubled or halved fails; the draws are
   Gauss-Newton's given the same mode; fewer steps in a Salt Lake
   Valley-shaped world.
"""

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from scipy import integrate
from scipy.linalg import block_diag, cholesky
from scipy.optimize import least_squares, minimize
from scipy.special import ndtri
from scipy.stats import norm

import fips
from fips import Covariance, LognormalForward, Operator, SoftplusForward, Space, Vector
from fips.linalg import (
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    SparseOperator,
)
from fips.transforms import _softplus_moments, _Transformed

# ---------------------------------------------------------------------------
# The user guide's model, at the top level so that it pickles
# ---------------------------------------------------------------------------


class ScaledInventory(fips.ForwardModel):
    """The observations of an inventory scaled by exp(x): F(x) = H (r * exp(x)), as the nonlinear guide writes it."""

    def __init__(self, H, reference):
        super().__init__(H.rows, H.cols)
        self.H = H
        self.reference = reference

    def __call__(self, x):
        return self.H @ (self.reference * np.exp(x))

    def jacobian(self, x):
        flux = (self.reference * np.exp(x)).values
        return self.H @ fips.Operator(DiagonalOperator(flux), x.space, x.space)


# ---------------------------------------------------------------------------
# Problems as arrays, and the references written with them
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Case:
    """
    A problem as arrays: ``z = H [T(u); b] + c + e``, the state ``x = [u; b]`` with ``x ~ N(x_0, S_0)``.

    `n` flux cells and `k` bias elements; ``H`` is ``m x (n + k)``. The
    transform is the lognormal's ``r e^u`` or the softplus's ``s log(1 +
    e^{u/s})``, written here with NumPy.
    """

    kind: str
    n: int
    k: int
    reference: np.ndarray
    scale: float
    x_0: np.ndarray
    S_0: np.ndarray
    H: np.ndarray
    S_z: np.ndarray
    z: np.ndarray
    c: np.ndarray
    flux_index: pd.Index | None = None
    """The flux cells' labels; by default ``c0, c1, ...`` named ``cell``."""

    @classmethod
    def random(
        cls,
        kind: str,
        seed: int,
        n: int,
        m: int,
        k: int = 0,
        std: float = 0.5,
        zeros: int = 0,
        scale: float = 0.7,
        S_0: np.ndarray | None = None,
    ) -> "Case":
        """A correlated prior, a dense footprint, and data from a truth drawn from the prior (`zeros` cells with no reference)."""
        rng = np.random.default_rng(seed)
        i = np.arange(n)
        if S_0 is None:
            C = np.exp(-np.abs(i[:, None] - i[None, :]) / 3.0)
            s = std * rng.uniform(0.7, 1.3, n)
            if kind == "softplus":
                s = s * 2.0
            S_0 = s[:, None] * C * s[None, :]
        if k:
            S_0 = block_diag(S_0, np.eye(k) * 0.3**2)
        reference = rng.uniform(0.5, 2.0, n)
        reference[rng.choice(n, zeros, replace=False)] = 0.0
        if kind == "lognormal":
            u_0 = rng.normal(0.0, 0.3, n)
        else:  # some cells near the bend, some far above it
            u_0 = rng.normal(0.8, 1.0, n)
        x_0 = np.concatenate([u_0, np.zeros(k)])
        H = rng.uniform(0.0, 1.0, (m, n + k))
        H[:, n:] = (rng.integers(0, k, m)[:, None] == np.arange(k)) if k else 0.0
        S_z = np.diag(rng.uniform(0.01, 0.04, m))
        c = rng.normal(2.0, 0.1, m)
        case = cls(kind, n, k, reference, scale, x_0, S_0, H, S_z, np.zeros(m), c)
        truth = x_0 + cholesky(S_0, lower=True) @ rng.standard_normal(n + k)
        case.z = case.F(truth) + c + rng.normal(0.0, np.sqrt(np.diag(S_z)))
        return case

    # the transform, its first and second derivatives
    def T(self, u: np.ndarray) -> np.ndarray:
        """The flux of `u`."""
        if self.kind == "lognormal":
            return self.reference * np.exp(u)
        return self.scale * np.log1p(np.exp(u / self.scale))

    def dT(self, u: np.ndarray) -> np.ndarray:
        """Its slope."""
        if self.kind == "lognormal":
            return self.reference * np.exp(u)
        return 1.0 / (1.0 + np.exp(-u / self.scale))

    def d2T(self, u: np.ndarray) -> np.ndarray:
        """Its curvature."""
        if self.kind == "lognormal":
            return self.reference * np.exp(u)
        g = self.dT(u)
        return g * (1.0 - g) / self.scale

    def F(self, x: np.ndarray) -> np.ndarray:
        """The model."""
        return self.H @ np.concatenate([self.T(x[: self.n]), x[self.n :]])

    def J(self, x: np.ndarray) -> np.ndarray:
        """Its Jacobian."""
        slopes = np.concatenate([self.dT(x[: self.n]), np.ones(self.k)])
        return self.H * slopes[None, :]

    def residual(self, x: np.ndarray) -> np.ndarray:
        """The data's residual."""
        return self.z - self.F(x) - self.c

    def cost(self, x: np.ndarray) -> float:
        """The MAP cost, the chi-squared at `x`."""
        r, dx = self.residual(x), x - self.x_0
        return float(
            r @ np.linalg.solve(self.S_z, r) + dx @ np.linalg.solve(self.S_0, dx)
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Half the gradient of the cost."""
        w = np.linalg.solve(self.S_z, self.residual(x))
        return np.linalg.solve(self.S_0, x - self.x_0) - self.J(x).T @ w

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Half the cost's whole Hessian, the transform's curvature included (which Gauss-Newton leaves out)."""
        w = np.linalg.solve(self.S_z, self.residual(x))
        J = self.J(x)
        curvature = np.concatenate(
            [self.d2T(x[: self.n]) * (self.H[:, : self.n].T @ w), np.zeros(self.k)]
        )
        return (
            np.linalg.inv(self.S_0)
            + J.T @ np.linalg.solve(self.S_z, J)
            - np.diag(curvature)
        )

    def least_squares(self) -> np.ndarray:
        """The minimizer of the cost by SciPy's Levenberg-Marquardt on the whitened, stacked residual."""
        L_z = cholesky(self.S_z, lower=True)
        L_0 = cholesky(self.S_0, lower=True)

        def residual(x):
            return np.concatenate(
                [
                    np.linalg.solve(L_z, self.residual(x)),
                    np.linalg.solve(L_0, x - self.x_0),
                ]
            )

        def jacobian(x):
            return np.vstack([-np.linalg.solve(L_z, self.J(x)), np.linalg.inv(L_0)])

        fit = least_squares(
            residual,
            self.x_0,
            jac=jacobian,
            method="lm",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
        return fit.x

    def mode(self) -> np.ndarray:
        """The minimizer to roundoff: :meth:`least_squares`, then Newton's method with the whole Hessian."""
        x = self.least_squares()
        for _ in range(4):
            x = x - np.linalg.solve(self.hessian(x), self.gradient(x))
        return x

    def gaussian_std(self, x: np.ndarray) -> np.ndarray:
        """The standard deviations of the Gaussian at `x` (Gauss-Newton's)."""
        J = self.J(x)
        P = np.linalg.inv(self.S_0) + J.T @ np.linalg.solve(self.S_z, J)
        return np.sqrt(np.diag(np.linalg.inv(P)))

    # the labeled problem
    def spaces(self) -> tuple[Space, Space]:
        """The state (a ``flux`` block, and a ``bias`` block if `k`) and the observations."""
        flux = self.flux_index
        if flux is None:
            flux = pd.Index([f"c{j}" for j in range(self.n)], name="cell")
        blocks = {"flux": flux}
        if self.k:
            blocks["bias"] = pd.Index([f"s{j}" for j in range(self.k)], name="site")
        m = len(self.z)
        return Space(blocks), Space(
            {"conc": pd.Index([f"o{j}" for j in range(m)], name="obs_id")}
        )

    def operator(self, sparse: bool = False, permuted: bool = False) -> Operator:
        """``H`` labeled; `sparse` holds it as a SciPy matrix, `permuted` reverses its rows and columns."""
        state, obs = self.spaces()
        H = Operator(sps.csr_array(self.H) if sparse else self.H, obs, state)
        if permuted:
            rows = Space({"conc": obs.select("conc")[::-1]})
            cols = Space({b: state.select(b)[::-1] for b in state.names})
            H = H.align(rows, cols)
        return H

    def model(self, H: Operator | None = None) -> "LognormalForward | SoftplusForward":
        """The fips model of this case."""
        H = self.operator() if H is None else H
        if self.kind == "lognormal":
            labels = self.spaces()[0].select("flux")  # matched to H's by label
            return LognormalForward(
                H, pd.Series(self.reference, index=labels), block="flux"
            )
        return SoftplusForward(H, block="flux", scale=self.scale)

    def problem(self, prior_cov=None, **operator) -> fips.Problem:
        """The labeled problem; `operator` as :meth:`operator` takes it."""
        state, obs = self.spaces()
        return fips.Problem(
            prior=Vector(self.x_0, state),
            prior_cov=Covariance(self.S_0, state) if prior_cov is None else prior_cov,
            obs=Vector(self.z, obs),
            obs_cov=Covariance(self.S_z, obs),
            forward=self.model(self.operator(**operator)),
            offset=Vector(self.c, obs),
        )


KINDS = ["lognormal", "softplus"]
SIZES = [(6, 15), (15, 6)]
"""Fewer, then more state elements than observations."""
TIGHT = {"tol": 1e-20, "max_iterations": 200}
"""Options with which the iterations go on close to roundoff, for the modes compared at 1e-9 of a standard deviation (the step counts are compared at ``STEPS``)."""

STEPS = {"tol": 1e-14, "max_iterations": 200}
"""Options for counting the steps to the mode: ``tol * n`` far above where Newton's steps bounce, so the count does not depend on the BLAS (they bounced from 1e-18 to 1e-15 in the whitened form on NumPy 1.24 and SciPy 1.12, at 1e-23 on newer ones)."""


def assert_mode(post: fips.Posterior, case: Case, atol: float = 1e-9) -> None:
    """The estimate is the minimizer of the cost within `atol` posterior standard deviations, against SciPy's least squares and Newton's method, and the cost's gradient is zero there."""
    x = post.estimate.values
    scipy_mode = case.least_squares()
    s = case.gaussian_std(scipy_mode)
    assert np.abs((x - scipy_mode) / s).max() < 1e-6
    reference = case.mode()
    error = np.abs((x - reference) / s).max()
    assert error < atol, error
    assert np.abs(case.gradient(x) * s).max() < atol
    assert post.chi2 == pytest.approx(case.cost(reference), rel=1e-9)


# ---------------------------------------------------------------------------
# 1. The Taylor test
# ---------------------------------------------------------------------------


def check_taylor(case: Case, **operator) -> None:
    """The Taylor test passes at the prior and at a random state, in a random direction and in one that moves the bias block alone."""
    model = case.model(case.operator(**operator))
    state, _ = case.spaces()
    rng = np.random.default_rng(1)
    for x in (case.x_0, case.x_0 + rng.normal(0.0, 0.5, len(case.x_0))):
        assert fips.taylortest(model, Vector(x, state))
        if case.k:
            dx = np.concatenate([np.zeros(case.n), rng.normal(0.0, 1.0, case.k)])
            assert fips.taylortest(model, Vector(x, state), Vector(dx, state))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_the_jacobian_passes_the_taylor_test(kind, sparse, permuted):
    """With a bias block, dense or sparse H, labels in another order, and two cells without a reference."""
    case = Case.random(kind, 0, 8, 12, k=2, zeros=2 if kind == "lognormal" else 0)
    check_taylor(case, sparse=sparse, permuted=permuted)


@pytest.mark.parametrize("kind", KINDS)
def test_the_jacobian_keeps_the_form_of_H(kind):
    """A sparse H gives a sparse Jacobian with H's pattern, a dense one a dense Jacobian."""
    case = Case.random(kind, 1, 6, 9, k=1)
    case.H[case.H < 0.6] = 0.0
    state, _ = case.spaces()
    x = Vector(case.x_0, state)
    sparse = case.model(case.operator(sparse=True)).jacobian(x)
    assert isinstance(sparse.data, SparseOperator)
    assert sparse.data.matrix.nnz == np.count_nonzero(case.H)
    assert isinstance(case.model().jacobian(x).data, DenseOperator)


# ---------------------------------------------------------------------------
# 2. Hand-written models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("permuted", [False, True])
def test_the_values_and_the_jacobian_are_the_formulas(kind, permuted):
    """F(x) and J(x), labeled in the model's order, are the NumPy formulas on the problem's."""
    case = Case.random(kind, 2, 7, 10, k=2, zeros=1 if kind == "lognormal" else 0)
    model = case.model(case.operator(permuted=permuted))
    state, obs = case.spaces()
    x = case.x_0 + np.random.default_rng(3).normal(0.0, 0.6, len(case.x_0))
    modeled = model(Vector(x, state)).align(obs).values
    np.testing.assert_allclose(modeled, case.F(x), rtol=1e-13)
    J = model.jacobian(Vector(x, state)).align(obs, state).dense()
    np.testing.assert_allclose(J, case.J(x), rtol=1e-13, atol=1e-15)
    flux = model.transform(Vector(x, state))
    assert flux.space is state
    np.testing.assert_allclose(flux["flux"].values, case.T(x[: case.n]), rtol=1e-13)
    np.testing.assert_array_equal(flux["bias"].values, x[case.n :])


def test_softplus_is_computed_without_overflow():
    """Far above the scale the flux is u and the slope 1, far below it about s e^{u/s} and 0, with no warning."""
    case = Case.random("softplus", 3, 4, 5, scale=0.5)
    state, _ = case.spaces()
    model = case.model()
    u = np.array([1e4, 40.0, -40.0, -1e4])
    flux = model.transform(Vector(u, state)).values
    np.testing.assert_allclose(flux[:2], u[:2], rtol=1e-15)
    np.testing.assert_allclose(flux[2], 0.5 * np.exp(-80.0), rtol=1e-12)
    assert flux[3] == 0.0
    slopes = (model.jacobian(Vector(u, state)).dense() / case.H)[0]
    assert slopes[0] == 1.0 and slopes[3] == 0.0


def test_lognormal_is_the_guides_scaled_inventory():
    """On the guide's problem, LognormalForward and ScaledInventory give the same model, Jacobian, mode, iterations and posterior."""
    rng = np.random.default_rng(0)
    cells = Space({"flux": pd.RangeIndex(10, name="cell")})
    observed = Space({"ch4": pd.RangeIndex(25, name="obs_id")})
    footprint = rng.uniform(0.0, 1.0, (25, 10)) * (rng.uniform(size=(25, 10)) < 0.4)
    H = Operator(footprint, observed, cells)
    inventory = Vector(rng.uniform(0.5, 2.0, 10), cells)

    def build(model, obs):
        return fips.Problem(
            prior=Vector(np.zeros(10), cells),
            prior_cov=fips.Exponential("cell", 2.0).with_std(0.5),
            obs=obs,
            obs_cov=fips.Independent().with_std(0.2),
            forward=model,
            offset=1900.0,
        )

    guide = ScaledInventory(H, inventory)
    ours = LognormalForward(H, inventory)
    _, obs = build(guide, Vector(np.zeros(25), observed)).simulate(rng=1)
    x = Vector(rng.normal(0.0, 0.5, 10), cells)
    np.testing.assert_allclose(ours(x).values, guide(x).values, rtol=1e-15)
    np.testing.assert_allclose(
        ours.jacobian(x).dense(), guide.jacobian(x).dense(), rtol=1e-15
    )
    a = fips.solve(build(guide, obs), "gauss_newton")
    b = fips.solve(build(ours, obs), "gauss_newton")
    assert len(a.iterations) == len(b.iterations)
    np.testing.assert_allclose(
        b.estimate.values, a.estimate.values, rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        b._solution.analytic.variances(), a._solution.analytic.variances(), rtol=1e-12
    )
    assert b.chi2 == pytest.approx(a.chi2, rel=1e-13)
    draws = b.sample(200, rng=2)
    np.testing.assert_allclose(
        ours.transform(draws).to_numpy(),
        inventory.values[:, None] * np.exp(draws.to_numpy()),
        rtol=1e-15,
    )


# ---------------------------------------------------------------------------
# 3. The mode against an independent optimizer
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n, m", SIZES)
@pytest.mark.parametrize("k", [0, 2])
def test_the_mode_is_the_minimizer_of_the_cost(kind, n, m, k):
    """The estimate minimizes the cost, with and without a bias block, in both analytic forms."""
    case = Case.random(kind, 4, n, m, k=k)
    post = fips.solve(case.problem(), "gauss_newton", **TIGHT)
    assert len(post.iterations) > 3  # nonlinear enough to iterate
    assert_mode(post, case)


@pytest.mark.parametrize("kind", KINDS)
def test_a_sparse_permuted_operator_gives_the_same_mode(kind):
    """H held sparse, its labels in another order, gives the mode of the dense one."""
    case = Case.random(kind, 5, 7, 11, k=2)
    dense = fips.solve(case.problem(), "gauss_newton", **TIGHT)
    other = fips.solve(
        case.problem(sparse=True, permuted=True), "gauss_newton", **TIGHT
    )
    np.testing.assert_allclose(other.estimate.values, dense.estimate.values, atol=1e-12)
    assert_mode(other, case)


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("kind", KINDS)
def test_a_kronecker_prior_gives_the_mode_of_the_dense_prior(kind):
    """A Kronecker prior on the flux block, beside a bias block, through the iterations."""
    rng = np.random.default_rng(7)
    times, cells = 3, 4
    T = np.exp(-np.abs(np.subtract.outer(np.arange(times), np.arange(times))) / 2.0)
    X = np.exp(-np.abs(np.subtract.outer(np.arange(cells), np.arange(cells))) / 1.5)
    std = rng.uniform(0.3, 0.6, times * cells) * (2.0 if kind == "softplus" else 1.0)
    kron = KroneckerOperator([T, X], std=std)
    case = Case.random(kind, 7, times * cells, 9, k=2, S_0=kron.dense())
    case.flux_index = pd.MultiIndex.from_product(
        [range(times), range(cells)], names=["time", "cell"]
    )
    state, _ = case.spaces()
    prior_cov = Covariance.block_diag(
        flux=Covariance(kron, Space({"flux": state.select("flux")})),
        bias=Covariance(np.eye(2) * 0.3**2, Space({"bias": state.select("bias")})),
    )
    post = fips.solve(case.problem(prior_cov=prior_cov), "gauss_newton", **TIGHT)
    assert isinstance(post._linear.prior_cov.data.blocks[0], KroneckerOperator)
    assert_mode(post, case)


def test_a_cell_without_a_reference_changes_nothing_else():
    """Its flux stays zero, and whatever prior its u has, the other cells' mode is the same (the guide's claim)."""
    case = Case.random("lognormal", 8, 8, 12, zeros=2)
    zero = case.reference == 0
    post = fips.solve(case.problem(), "gauss_newton", **TIGHT)
    assert_mode(post, case)
    model = post.problem.forward
    np.testing.assert_array_equal(model.transform(post.estimate).values[zero], 0.0)
    # no prior spread at those cells (as prior_for gives them): the same mode elsewhere
    s = np.sqrt(np.diag(case.S_0)).copy()
    C = case.S_0 / np.outer(s, s)
    s[zero] = 0.0
    other = dataclasses.replace(case, S_0=np.outer(s, s) * C)
    again = fips.solve(other.problem(), "gauss_newton", **TIGHT)
    np.testing.assert_allclose(
        again.estimate.values[~zero], post.estimate.values[~zero], rtol=1e-9, atol=1e-11
    )


# ---------------------------------------------------------------------------
# 4. The prior correspondence
# ---------------------------------------------------------------------------


def one_cell_model(kind: str, n: int, reference=1.0, scale: float = 1.0):
    """A model on `n` cells and one observation of them all."""
    state = Space({"flux": pd.RangeIndex(n, name="cell")})
    H = Operator(
        np.ones((1, n)), Space({"conc": pd.RangeIndex(1, name="obs_id")}), state
    )
    if kind == "lognormal":
        return LognormalForward(H, reference)
    return SoftplusForward(H, scale=scale)


def check_prior_by_draws(
    model, mean: np.ndarray, std: np.ndarray, draws: int = 400_000
) -> None:
    """Draws of u from prior_for, through the transform, have the mean and the standard deviation asked for, within 4.5 Monte Carlo standard errors."""
    mu, sigma = model.prior_for(
        pd.Series(mean, index=model.H.cols.select("flux")),
        pd.Series(std, index=model.H.cols.select("flux")),
    )
    u = mu.values[:, None] + sigma.values[:, None] * np.random.default_rng(
        0
    ).standard_normal((len(mean), draws))
    flux = model.transform(pd.DataFrame(u, index=model.cols.index)).to_numpy()
    se_mean = flux.std(axis=1) / np.sqrt(draws)
    assert np.all(np.abs(flux.mean(axis=1) - mean) < 4.5 * se_mean), (
        flux.mean(axis=1),
        mean,
    )
    # the standard error of a standard deviation, from the fourth moment
    dev = flux - flux.mean(axis=1, keepdims=True)
    m4, var = (dev**4).mean(axis=1), (dev**2).mean(axis=1)
    se_std = np.sqrt((m4 - var**2) / draws) / (2 * np.sqrt(var))
    assert np.all(np.abs(flux.std(axis=1) - std) < 4.5 * se_std), (
        flux.std(axis=1),
        std,
    )


def test_the_lognormal_prior_has_the_flux_moments_exactly():
    """Mu and sigma give the mean and the standard deviation in closed form, and by draws; the reference need not be the mean."""
    reference = np.array([1.0, 2.0, 0.5, 3.0])
    mean = np.array([1.0, 2.0, 1.0, 0.3])
    std = np.array([0.3, 2.0, 0.1, 0.6])
    model = one_cell_model(
        "lognormal", 4, pd.Series(reference, index=pd.RangeIndex(4, name="cell"))
    )
    mu, sigma = model.prior_for(
        pd.Series(mean, index=pd.RangeIndex(4, name="cell")),
        pd.Series(std, index=pd.RangeIndex(4, name="cell")),
    )
    m, v = mu.values, sigma.values**2
    np.testing.assert_allclose(reference * np.exp(m + v / 2), mean, rtol=1e-14)
    np.testing.assert_allclose(
        reference * np.exp(m + v / 2) * np.sqrt(np.expm1(v)), std, rtol=1e-13
    )
    check_prior_by_draws(model, mean, std)


def test_the_lognormal_flux_correlation_is_the_formula():
    """Two cells whose u correlate by rho: their fluxes correlate by (e^{rho s_i s_j} - 1) / sqrt((e^{s_i^2} - 1)(e^{s_j^2} - 1)), as prior_for says."""
    s_i, s_j, rho = 1.0, 0.6, 0.5
    L = cholesky(
        np.array([[s_i**2, rho * s_i * s_j], [rho * s_i * s_j, s_j**2]]), lower=True
    )
    u = L @ np.random.default_rng(1).standard_normal((2, 2_000_000))
    flux = np.exp(u)
    measured = np.corrcoef(flux)[0, 1]
    formula = np.expm1(rho * s_i * s_j) / np.sqrt(np.expm1(s_i**2) * np.expm1(s_j**2))
    assert abs(measured - formula) < 0.01
    assert formula < rho


def softplus_moments_by_quad(a: float, b: float) -> tuple[float, float]:
    """The mean and standard deviation of log(1 + e^t), t ~ N(a, b^2), by SciPy's adaptive quad in the standard normal variable, split at the bend."""
    bend = -a / b
    points = [bend] if -14 < bend < 14 else None

    def f(x):
        return np.logaddexp(0.0, a + b * x)

    mean = integrate.quad(
        lambda x: f(x) * norm.pdf(x),
        -14,
        14,
        points=points,
        limit=500,
        epsabs=0,
        epsrel=1e-12,
    )[0]
    var = integrate.quad(
        lambda x: (f(x) - mean) ** 2 * norm.pdf(x),
        -14,
        14,
        points=points,
        limit=500,
        epsabs=0,
        epsrel=1e-12,
    )[0]
    return mean, np.sqrt(var)


@pytest.mark.parametrize(
    "a, b",
    [
        (-20.0, 0.5),
        (-3.0, 2.0),
        (-1.0, 0.3),
        (0.0, 1.0),
        (1.5, 4.0),
        (5.0, 0.2),
        (30.0, 8.0),
        (0.0, 1e3),
        (-10.0, 30.0),
    ],
)
def test_the_softplus_moments_are_quads(a, b):
    """The quadrature behind prior_for gives SciPy's quad to 1e-11, far below the bend, across it, far above it and far wider than it."""
    mean, std, _ = _softplus_moments(np.array([a]), np.array([b]))
    ref_mean, ref_std = softplus_moments_by_quad(a, b)
    assert mean[0] == pytest.approx(ref_mean, rel=1e-11)
    assert std[0] == pytest.approx(ref_std, rel=1e-11)


def test_the_softplus_moments_derivatives_are_their_differences():
    """The derivatives Newton's method steps with are the moments' central differences."""
    a = np.array([-3.0, 0.0, 2.0, 50.0])
    b = np.array([0.7, 2.0, 5.0, 3.0])
    _, _, J = _softplus_moments(a, b)
    h = 1e-6
    for k, (da, db) in enumerate([(h, 0.0), (0.0, h)]):
        up = _softplus_moments(a + da, b + db)
        down = _softplus_moments(a - da, b - db)
        for moment in (0, 1):
            np.testing.assert_allclose(
                J[:, moment, k],
                (up[moment] - down[moment]) / (2 * h),
                rtol=1e-6,
                atol=1e-9,
            )


@pytest.mark.parametrize("scale", [1.0, 0.05, 30.0])
def test_the_softplus_prior_has_the_flux_moments(scale):
    """By quad to 1e-9 and by draws, from far below the scale to far above it, narrow to wide, at three scales."""
    mean = scale * np.array([0.01, 0.3, 1.0, 1.0, 3.0, 50.0, 0.5])
    std = mean * np.array([0.5, 2.0, 1.0, 0.1, 0.8, 0.2, 10.0])
    model = one_cell_model("softplus", len(mean), scale=scale)
    labels = model.H.cols.select("flux")
    mu, sigma = model.prior_for(
        pd.Series(mean, index=labels), pd.Series(std, index=labels)
    )
    for j in range(len(mean)):
        ref_mean, ref_std = softplus_moments_by_quad(
            mu.values[j] / scale, sigma.values[j] / scale
        )
        assert ref_mean * scale == pytest.approx(mean[j], rel=1e-9)
        assert ref_std * scale == pytest.approx(std[j], rel=1e-9)
    check_prior_by_draws(model, mean, std)


def test_the_softplus_prior_matches_over_many_orders_of_magnitude():
    """Means from 1e-6 to 1e6 scales and standard deviations from 1e-6 to 1e3 times the mean, each matched to 1e-9 by its own quadrature."""
    rng = np.random.default_rng(0)
    mean = 10 ** rng.uniform(-6, 6, 400)
    std = mean * 10 ** rng.uniform(-6, 3, 400)
    model = one_cell_model("softplus", 400)
    labels = model.H.cols.select("flux")
    mu, sigma = model.prior_for(
        pd.Series(mean, index=labels), pd.Series(std, index=labels)
    )
    got_mean, got_std, _ = _softplus_moments(mu.values, sigma.values)
    np.testing.assert_allclose(got_mean, mean, rtol=1e-9)
    np.testing.assert_allclose(got_std, std, rtol=1e-9)


def test_far_above_the_scale_the_softplus_prior_is_the_flux_prior():
    """Where the flux is far above the bend, u's prior is the flux's; with no spread, the u whose flux is the mean."""
    model = one_cell_model("softplus", 3, scale=0.1)
    labels = model.H.cols.select("flux")
    mu, sigma = model.prior_for(
        pd.Series([100.0, 50.0, 0.2], index=labels),
        pd.Series([1.0, 2.0, 0.0], index=labels),
    )
    np.testing.assert_allclose(mu.values[:2], [100.0, 50.0], rtol=1e-12)
    np.testing.assert_allclose(sigma.values, [1.0, 2.0, 0.0], rtol=1e-12)
    assert 0.1 * np.logaddexp(0.0, mu.values[2] / 0.1) == pytest.approx(0.2, rel=1e-14)


def test_prior_for_on_labels_in_another_order_and_with_a_bias_block():
    """Mean and std are matched by label, and the answer is on the block's labels, in the model's order."""
    case = Case.random("lognormal", 9, 5, 7, k=2, zeros=1)
    model = case.model(case.operator(permuted=True))
    labels = model.H.cols.select("flux")
    order = pd.Index([f"c{j}" for j in range(5)]).get_indexer(labels)
    mean = pd.Series(
        case.reference * 1.5, index=pd.Index([f"c{j}" for j in range(5)], name="cell")
    )
    mu, sigma = model.prior_for(mean[::-1], 0.5 * mean)
    assert mu.space.names == ("flux",)
    assert mu.space.select("flux").equals(labels)
    r = case.reference[order]
    expected = np.where(r > 0, np.log(1.5) - np.log1p(0.25) / 2, 0.0)
    np.testing.assert_allclose(mu.values, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(
        sigma.values, np.where(r > 0, np.sqrt(np.log1p(0.25)), 0.0), rtol=1e-14
    )


# ---------------------------------------------------------------------------
# 5. Its own world
# ---------------------------------------------------------------------------


def own_world(
    kind: str, seed: int, n: int = 8, m: int = 20, std: float = 0.2, **options
):
    """A problem whose prior is prior_for's (each cell's flux of mean 1 to 2, standard deviation `std` times it for lognormal), its model and the posterior of one draw's data, solved with `options`."""
    rng = np.random.default_rng(seed)
    state = Space({"flux": pd.Index([f"c{j}" for j in range(n)], name="cell")})
    obs = Space({"conc": pd.RangeIndex(m, name="obs_id")})
    H = Operator(rng.uniform(0.0, 1.0, (m, n)), obs, state)
    mean = pd.Series(rng.uniform(1.0, 2.0, n), index=state.select("flux"))
    if kind == "lognormal":
        model = LognormalForward(H, mean)
        cv = np.sqrt(np.expm1(std**2))
    else:
        model = SoftplusForward(H, scale=0.5)
        cv = std
    mu, sigma = model.prior_for(mean, cv * mean)
    i = np.arange(n)
    C = np.exp(-np.abs(i[:, None] - i[None, :]) / 2.0)
    s = sigma.values

    def problem(z: np.ndarray) -> fips.Problem:
        return fips.Problem(
            prior=mu,
            prior_cov=Covariance(s[:, None] * C * s[None, :], state),
            obs=Vector(z, obs),
            obs_cov=fips.Independent().with_std(0.3),
            forward=model,
        )

    _, z = problem(np.zeros(m)).simulate(rng=rng)
    return model, fips.solve(problem(z.values), "gauss_newton", **options)


def coverage_of_cells_and_total(model, post, trials: int, levels):
    """
    The fraction of trials whose interval at each level holds the truth: each cell's (the posterior standard deviation about the mode, as ``trials`` reports it; the transform is monotone, so an interval in u is the flux's) and the total flux's (the quantiles of the posterior's kept draws, through the transform).

    Returns, for each level, both with their Monte Carlo standard errors.
    """
    rng = np.random.default_rng(10)
    world = post.problem
    index = world.prior.space.index
    z_cells, totals, true_totals = [], [], []
    for _ in range(trials):
        truth, obs = world.simulate(rng=rng)
        trial = post.with_data(obs)
        z_cells.append(
            (trial.estimate.values - truth.values) / np.sqrt(trial.variances.values)
        )
        kept = pd.DataFrame(trial._solution._kept(), index=index)
        totals.append(model.transform(kept).sum().to_numpy())
        true_totals.append(model.transform(truth).values.sum())
    z_cells = np.array(z_cells)
    out = {}
    for level in levels:
        inside = (np.abs(z_cells) <= ndtri(0.5 + level / 2)).mean(axis=1)
        covered = [
            np.quantile(t, 0.5 - level / 2) <= true <= np.quantile(t, 0.5 + level / 2)
            for t, true in zip(totals, true_totals, strict=True)
        ]
        out[level] = (
            (float(inside.mean()), float(inside.std() / np.sqrt(trials))),
            (float(np.mean(covered)), np.sqrt(level * (1 - level) / trials)),
        )
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_in_a_narrow_prior_the_intervals_of_cells_and_totals_cover(kind):
    """
    A truth drawn from prior_for's prior and data through the model: the posterior's draws cover near their level, for the cells and for the total.

    The flux's prior standard deviation is a fifth of its mean (lognormal:
    a log standard deviation of 0.2). Measured (seed 0, 150 trials, 300 kept
    draws each), at 90% and 68%: lognormal cells 0.911 and 0.700 (Monte Carlo
    standard errors 0.008, 0.015), totals 0.900 and 0.680 (0.024, 0.038);
    softplus cells 0.910 and 0.701, totals 0.893 and 0.700. The totals from
    draws of the Gaussian at the mode leaned low here (lognormal 0.865 and
    0.625 at 200 trials); the user guide measures that in wider priors.
    """
    model, post = own_world(kind, 0, draws=300, burn_in=50)
    for level, ((cells, cells_se), (total, total_se)) in coverage_of_cells_and_total(
        model, post, 150, (0.9, 0.68)
    ).items():
        assert abs(cells - level) < 4 * cells_se + 0.01, (level, cells, cells_se)
        assert abs(total - level) < 4 * total_se + 0.01, (level, total, total_se)


# ---------------------------------------------------------------------------
# 6. Wrong variants fail the tests above
# ---------------------------------------------------------------------------


def without_the_reference(monkeypatch):
    """The lognormal's Jacobian scaled by e^u, without the reference."""
    monkeypatch.setattr(LognormalForward, "_slopes", lambda self, u: np.exp(u))


def the_median_convention(monkeypatch):
    """prior_for with the reference as the median: mu = log(m / r), without -sigma^2 / 2."""
    real = LognormalForward.prior_for

    def wrong(self, mean, std):
        mu, sigma = real(self, mean, std)
        return mu + sigma * sigma / 2, sigma

    monkeypatch.setattr(LognormalForward, "prior_for", wrong)


def softplus_slope_as_exp(monkeypatch):
    """Softplus's Jacobian with e^{u/s} in place of the logistic function."""
    monkeypatch.setattr(
        SoftplusForward, "_slopes", lambda self, u: np.exp(u / self.scale)
    )


@pytest.mark.parametrize(
    "mutant, kind, check",
    [
        (without_the_reference, "lognormal", "taylor"),
        (without_the_reference, "lognormal", "mode"),
        (the_median_convention, "lognormal", "prior"),
        (softplus_slope_as_exp, "softplus", "taylor"),
        (softplus_slope_as_exp, "softplus", "mode"),
    ],
)
def test_a_wrong_variant_fails(monkeypatch, mutant, kind, check):
    """Each wrong variant fails a check the right one passes."""
    mutant(monkeypatch)
    case = Case.random(kind, 4, 6, 15, k=2)
    with pytest.raises((AssertionError, RuntimeError)):
        if check == "taylor":
            check_taylor(case)
        elif check == "mode":
            assert_mode(fips.solve(case.problem(), "gauss_newton", **TIGHT), case)
        else:
            model = one_cell_model(
                kind, 3, pd.Series([1.0, 2.0, 0.5], index=pd.RangeIndex(3, name="cell"))
            )
            check_prior_by_draws(
                model, np.array([1.0, 2.0, 0.5]), np.array([0.5, 1.0, 0.25])
            )


# ---------------------------------------------------------------------------
# 7. Refusals and pickling
# ---------------------------------------------------------------------------


def two_block_operator() -> Operator:
    """H on a flux block of three cells and a bias block of one site."""
    state = Space(
        {
            "flux": pd.Index(["a", "b", "c"], name="cell"),
            "bias": pd.Index(["s"], name="site"),
        }
    )
    return Operator(
        np.ones((2, 4)), Space({"conc": pd.RangeIndex(2, name="obs_id")}), state
    )


@pytest.mark.parametrize(
    "reference, error, match",
    [
        (
            pd.Series([1.0, 2.0], index=pd.Index(["a", "b"], name="cell")),
            ValueError,
            r"labels of block 'flux' missing.*'c'",
        ),
        (
            pd.Series(
                [1.0, 2.0, 3.0, 4.0], index=pd.Index(["a", "b", "c", "d"], name="cell")
            ),
            ValueError,
            r"only in reference.*'d'",
        ),
        (
            pd.Series([1.0, -2.0, 3.0], index=pd.Index(["a", "b", "c"], name="cell")),
            ValueError,
            r"negative at 1 labels.*'b'",
        ),
        (
            pd.Series([1.0, np.inf, 3.0], index=pd.Index(["a", "b", "c"], name="cell")),
            ValueError,
            r"not finite.*'b'",
        ),
        (np.ones(3), TypeError, "array has no labels"),
        (
            pd.Series([1.0, 2.0, 3.0], index=pd.Index(["a", "b", "c"], name="site")),
            ValueError,
            "levels",
        ),
    ],
)
def test_a_wrong_reference_is_refused(reference, error, match):
    """A reference on other labels, negative, not finite or without labels raises, naming what is wrong."""
    with pytest.raises(error, match=match):
        LognormalForward(two_block_operator(), reference, block="flux")


def test_the_block_and_the_operator_are_checked():
    """An unknown block, a missing block= with several blocks, and an H that is not an Operator."""
    H = two_block_operator()
    with pytest.raises(KeyError, match=r"No block named 'fluxes'"):
        SoftplusForward(H, block="fluxes")
    with pytest.raises(ValueError, match=r"block="):
        LognormalForward(H, 1.0)
    with pytest.raises(TypeError, match="Operator"):
        SoftplusForward(H.dense(), block="flux")
    model = SoftplusForward(H, block="flux")
    assert model.block == "flux" and model.H is H
    assert "SoftplusForward(block: 'flux'" in repr(model)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, "big"])
def test_a_wrong_scale_is_refused(scale):
    """The scale is a positive, finite number."""
    with pytest.raises(ValueError, match="scale"):
        SoftplusForward(two_block_operator(), block="flux", scale=scale)


def test_an_overflow_is_refused_naming_the_cell():
    """A u whose flux overflows raises a ValueError naming its label (which Gauss-Newton's step halving catches); a cell with no reference never overflows."""
    H = two_block_operator()
    model = LognormalForward(
        H, pd.Series([1.0, 0.0, 1.0], index=H.cols.select("flux")), block="flux"
    )
    x = Vector([1e3, 1e4, 0.0, 0.0], H.cols)
    with pytest.raises(ValueError, match=r"overflows at 1 labels.*'a'"):
        model(x)
    with pytest.raises(ValueError, match="overflows"):
        model.jacobian(x)
    np.testing.assert_array_equal(
        model.transform(Vector([0.0, 1e4, 0.0, 2.0], H.cols)).values,
        [1.0, 0.0, 1.0, 2.0],
    )


def test_transform_takes_states_and_draws_by_label():
    """A state in another order comes back in its order; draws with rows shuffled are transformed row by row; anything else is refused."""
    H = two_block_operator()
    model = LognormalForward(
        H, pd.Series([1.0, 2.0, 3.0], index=H.cols.select("flux")), block="flux"
    )
    x = Vector([0.0, np.log(2.0), 1.0, 5.0], H.cols)
    order = [3, 1, 0, 2]
    shuffled = Vector(x.values[order], Space.from_index(H.cols.index[order]))
    np.testing.assert_allclose(
        model.transform(shuffled).align(H.cols).values, [1.0, 4.0, 3 * np.e, 5.0]
    )
    draws = pd.DataFrame(np.outer(x.values, [1.0, 0.0]), index=H.cols.index).iloc[
        [2, 3, 0, 1]
    ]
    flux = model.transform(draws)
    assert flux.index.equals(draws.index)
    np.testing.assert_allclose(
        flux.loc[H.cols.index].to_numpy()[:, 1], [1.0, 2.0, 3.0, 0.0]
    )
    with pytest.raises(TypeError, match="Vector.*DataFrame"):
        model.transform(x.values)
    with pytest.raises(ValueError, match="draws' rows"):
        model.transform(draws.iloc[1:])


def test_prior_for_refuses_what_it_cannot_make():
    """A lognormal flux is never zero nor below zero, and is zero where the reference is; a softplus flux is above zero."""
    H = two_block_operator()
    labels = H.cols.select("flux")
    lognormal = LognormalForward(
        H, pd.Series([1.0, 0.0, 1.0], index=labels), block="flux"
    )
    with pytest.raises(
        ValueError,
        match=r"mean is not zero at 1 labels where the reference is zero, .*'b'",
    ):
        lognormal.prior_for(1.0, pd.Series([1.0, 0.0, 1.0], index=labels))
    with pytest.raises(
        ValueError, match=r"std is not zero .* reference is zero, .*'b'"
    ):
        lognormal.prior_for(pd.Series([1.0, 0.0, 1.0], index=labels), 1.0)
    with pytest.raises(ValueError, match=r"mean is zero at 1 labels.*'c'"):
        lognormal.prior_for(pd.Series([1.0, 0.0, 0.0], index=labels), 0.0)
    with pytest.raises(ValueError, match=r"std is negative"):
        lognormal.prior_for(
            pd.Series([1.0, 0.0, 1.0], index=labels),
            pd.Series([1.0, 0.0, -1.0], index=labels),
        )
    with pytest.raises(TypeError, match="array"):
        lognormal.prior_for(np.ones(3), 1.0)
    softplus = SoftplusForward(H, block="flux")
    with pytest.raises(ValueError, match=r"not positive at 1 labels.*'a'"):
        softplus.prior_for(pd.Series([0.0, 1.0, 1.0], index=labels), 1.0)


def test_a_singular_softplus_newton_matrix_is_a_runtime_error(monkeypatch):
    """
    Where Newton's 2 x 2 matrix is singular in a cell, its step is skipped and the cell fails to match: a RuntimeError naming the cells, as the docstring says, not numpy's LinAlgError.

    ``prior_for(1e-8, 1e3)`` with a scale of 0.5 (a standard deviation 1e11
    times the mean) met a singular matrix on NumPy 2.5 and not on 1.24, so
    the solve is made singular everywhere here.
    """
    H = two_block_operator()
    softplus = SoftplusForward(H, block="flux")

    def singular(A, b):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "solve", singular)
    with pytest.raises(RuntimeError, match=r"did not match .* in 3 cells.*'a'.*'c'"):
        softplus.prior_for(1.0, 1.0)


@pytest.mark.parametrize("kind", KINDS)
def test_a_problem_and_its_posterior_pickle(kind):
    """The problem with each model and its posterior pickle, and give the same answers."""
    case = Case.random(kind, 16, 5, 8, k=1)
    problem = case.problem(sparse=True)
    post = fips.solve(problem, "gauss_newton")
    again = pickle.loads(pickle.dumps(post))
    np.testing.assert_array_equal(again.estimate.values, post.estimate.values)
    np.testing.assert_array_equal(again.variances.values, post.variances.values)
    model, loaded = post.problem.forward, again.problem.forward
    np.testing.assert_array_equal(
        loaded.transform(again.estimate).values, model.transform(post.estimate).values
    )
    obs = Vector(case.z + 0.1, problem.obs.space)
    np.testing.assert_array_equal(
        again.with_data(obs).estimate.values, post.with_data(obs).estimate.values
    )
    copy = pickle.loads(pickle.dumps(problem))
    np.testing.assert_array_equal(
        copy.predict(copy.prior).values, problem.predict(problem.prior).values
    )


@pytest.mark.parametrize("kind", KINDS)
def test_an_aggregated_problem_keeps_the_model(kind):
    """Problem.aggregate follows the model with the aggregation, and solves."""
    case = Case.random(kind, 17, 5, 8)
    problem = case.problem()
    W = fips.aggregate.sum(
        problem.obs.space,
        by=pd.Series(np.arange(8) // 2, index=problem.obs.space.select("conc")),
    )
    post = fips.solve(problem.aggregate(W), "gauss_newton")
    assert len(post.modeled_obs) == 4


# ---------------------------------------------------------------------------
# 8. Newton under the transform
# ---------------------------------------------------------------------------


def gauss_newton_only(monkeypatch) -> None:
    """The transforms without their curvature: every step Gauss-Newton's, as for a model fips did not write."""
    monkeypatch.setattr(_Transformed, "_curvature", lambda self, x, v: None)


def steps_to(post: fips.Posterior, d2: float = 1e-12) -> int:
    """
    The steps a solve took to bring a step's d2 below `d2`, a tolerance above roundoff.

    ``TIGHT``'s tolerance is at roundoff, where a step's d2 wanders about the
    limit: how many steps a solve spends there differs from one platform's
    BLAS to another's, so the whole count compares nothing.
    """
    below = post.iterations["d2"].to_numpy() < d2
    assert below.any()
    return int(np.argmax(below)) + 1


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("permuted", [False, True])
def test_the_curvature_is_the_derivative_of_the_gradient(kind, permuted):
    """The diagonal the model returns is the Jacobian of ``J(x)^T v`` by central differences, which is diagonal on the block and zero off it; through the problem's order too."""
    case = Case.random(kind, 20, 6, 9, k=2)
    problem = case.problem(permuted=permuted)
    rng = np.random.default_rng(3)
    x = case.x_0 + rng.normal(0.0, 0.5, len(case.x_0))
    v = rng.normal(0.0, 1.0, len(case.z))
    curvature = problem._curvature(x, v)
    h = 1e-6
    columns = []
    for j in range(len(x)):
        e = np.zeros(len(x))
        e[j] = h
        columns.append(
            (problem._gradient(x + e, v) - problem._gradient(x - e, v)) / (2 * h)
        )
    np.testing.assert_allclose(
        np.column_stack(columns), np.diag(curvature), rtol=1e-6, atol=1e-8
    )
    np.testing.assert_array_equal(curvature[case.n :], 0.0)
    expected = case.d2T(x[: case.n]) * (case.H[:, : case.n].T @ v)
    np.testing.assert_allclose(curvature[: case.n], expected, rtol=1e-12)


def test_a_model_fips_did_not_write_has_no_curvature():
    """A hand-written model (the guide's) has none, so its steps are Gauss-Newton's."""
    case = Case.random("lognormal", 21, 4, 6)
    state, obs = case.spaces()
    H = Operator(case.H, obs, state)
    model = ScaledInventory(H, Vector(case.reference, state))
    assert model._curvature(case.x_0, np.ones(6)) is None


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("k", [0, 2])
def test_newton_reaches_the_gauss_newton_mode(monkeypatch, kind, k):
    """With and without a bias block, in both analytic forms: the same mode as Gauss-Newton's alone, to roundoff at a tight tol and well inside the default tol's error, in fewer steps (counted at a tol far above roundoff)."""
    case = Case.random(kind, 22, 9, 14, k=k, std=0.8)
    newton = fips.solve(case.problem(), "gauss_newton", **TIGHT)
    default = fips.solve(case.problem(), "gauss_newton")
    newton_steps = len(fips.solve(case.problem(), "gauss_newton", **STEPS).iterations)
    gauss_newton_only(monkeypatch)
    alone = fips.solve(case.problem(), "gauss_newton", **TIGHT)
    alone_steps = len(fips.solve(case.problem(), "gauss_newton", **STEPS).iterations)
    s = case.gaussian_std(alone.estimate.values)
    assert np.abs((newton.estimate.values - alone.estimate.values) / s).max() < 1e-9
    assert np.abs((default.estimate.values - alone.estimate.values) / s).max() < 1e-3
    assert newton_steps < alone_steps
    assert_mode(newton, case)


@pytest.mark.parametrize("kind", KINDS)
def test_newton_with_a_singular_prior_and_labels_in_another_order(monkeypatch, kind):
    """Cells with no prior spread stay at the prior and the others reach Gauss-Newton's mode, H sparse on reversed labels; no inverse of S_0 is needed."""
    case = Case.random(kind, 23, 8, 12, k=2, std=0.8)
    s = np.sqrt(np.diag(case.S_0)).copy()
    C = case.S_0 / np.outer(s, s)
    s[[1, 4]] = 0.0
    singular = dataclasses.replace(case, S_0=np.outer(s, s) * C)
    newton = fips.solve(
        singular.problem(sparse=True, permuted=True), "gauss_newton", **TIGHT
    )
    np.testing.assert_array_equal(newton.estimate.values[[1, 4]], case.x_0[[1, 4]])
    gauss_newton_only(monkeypatch)
    alone = fips.solve(singular.problem(), "gauss_newton", **TIGHT)
    std = np.sqrt(alone._solution.analytic.variances())
    moved = std > 0
    difference = (newton.estimate.values - alone.estimate.values)[moved] / std[moved]
    assert np.abs(difference).max() < 1e-9
    # by a wide margin here: 9 against 14 (lognormal) and 11 against 71
    assert steps_to(newton) < steps_to(alone)


@pytest.mark.parametrize("kind", KINDS)
def test_an_aggregated_problem_takes_newton_steps_to_the_same_mode(monkeypatch, kind):
    """Problem.aggregate passes the curvature through the aggregation: W^T v weights the model's."""
    case = Case.random(kind, 24, 6, 16, std=0.8)
    problem = case.problem()
    W = fips.aggregate.sum(
        problem.obs.space,
        by=pd.Series(np.arange(16) // 2, index=problem.obs.space.select("conc")),
    )
    aggregated = problem.aggregate(W)
    v = np.random.default_rng(0).normal(size=8)
    np.testing.assert_allclose(
        aggregated._curvature(case.x_0, v),
        problem._curvature(case.x_0, W.dense().T @ v),
        rtol=1e-13,
    )
    newton = fips.solve(aggregated, "gauss_newton", **TIGHT)
    gauss_newton_only(monkeypatch)
    alone = fips.solve(aggregated, "gauss_newton", **TIGHT)
    std = np.sqrt(alone._solution.analytic.variances())
    assert np.abs((newton.estimate.values - alone.estimate.values) / std).max() < 1e-9


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n, m", SIZES)
def test_the_newton_mode_is_bfgs_minimum(kind, n, m):
    """SciPy's BFGS on the cost written densely (no Gauss-Newton, no Newton) finds the same mode, to its own precision."""
    case = Case.random(kind, 25, n, m, k=1, std=0.8)
    post = fips.solve(case.problem(), "gauss_newton", **TIGHT)
    fit = minimize(
        case.cost,
        case.x_0,
        jac=lambda x: 2 * case.gradient(x),
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 5000},
    )
    s = case.gaussian_std(fit.x)
    assert np.abs((post.estimate.values - fit.x) / s).max() < 1e-5
    assert post.chi2 == pytest.approx(fit.fun, rel=1e-10)


def dense_steps(
    case: Case, tol: float, steps: int
) -> tuple[list[float], list[float], list[str]]:
    r"""
    The iterations written densely: the cost after each step, its :math:`d^2`, and which step it was.

    At each state, Gauss-Newton's step :math:`-A^{-1} g` (with :math:`A =
    S_0^{-1} + J^T S_z^{-1} J`) and, where the whole Hessian (``Case.hessian``)
    has a Cholesky factor, Newton's; each halved until the cost does not rise;
    the one of lower cost taken (Newton's on a tie). A whole Gauss-Newton step
    with :math:`d^2 = \delta^T A \delta` below ``tol * n`` is the last; each
    step's :math:`d^2` is in the :math:`A` of the state it starts from.
    """
    x = case.x_0.copy()
    current = case.cost(x)
    costs, d2s, kinds = [], [], []
    for _ in range(steps):
        g = case.gradient(x)
        J = case.J(x)
        A = np.linalg.inv(case.S_0) + J.T @ np.linalg.solve(case.S_z, J)
        gauss_newton = -np.linalg.solve(A, g)
        d2 = gauss_newton @ A @ gauss_newton
        if d2 < tol * len(x):
            costs.append(case.cost(x + gauss_newton))
            d2s.append(d2)
            kinds.append("last")
            break
        candidates = {"gauss_newton": gauss_newton}
        hessian = case.hessian(x)
        try:
            np.linalg.cholesky(hessian)
        except np.linalg.LinAlgError:
            pass
        else:
            candidates["newton"] = -np.linalg.solve(hessian, g)
        taken = {}
        for kind, step in candidates.items():
            fraction = 1.0
            while case.cost(x + fraction * step) > current + 1e-12 * abs(current):
                fraction /= 2
            taken[kind] = x + fraction * step
        kind = min(taken, key=lambda k: (case.cost(taken[k]), k != "newton"))
        d2s.append((taken[kind] - x) @ A @ (taken[kind] - x))
        x = taken[kind]
        current = case.cost(x)
        costs.append(current)
        kinds.append(kind)
    return costs, d2s, kinds


def assert_dense_steps(post: fips.Posterior, case: Case, tol: float) -> None:
    """
    The steps' costs are the dense iterations', and their d2 where a step is long enough for its kind to be clear.

    Near the mode Newton's estimate and Gauss-Newton's have the same cost to
    roundoff, so which of the two a step takes there differs from one
    platform's BLAS to another's, and with it the step's d2 (by a few percent:
    9.03e-10 for 9.35e-10 on one) and whether one more step follows. The costs
    do not. A wrong curvature moves the first Newton step's d2, of 1e-5 to
    1e-4 here, by 1% or more.
    """
    costs, d2s, kinds = dense_steps(case, tol, len(post.iterations))
    assert abs(len(kinds) - len(post.iterations)) <= 1
    k = min(len(kinds), len(post.iterations))
    np.testing.assert_allclose(post.iterations["cost"].values[:k], costs[:k], rtol=1e-9)
    d2s = np.array(d2s[:k])
    clear = d2s > 1e-6
    assert clear.sum() >= 4
    np.testing.assert_allclose(
        post.iterations["d2"].values[:k][clear], d2s[clear], rtol=1e-3
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n, m", SIZES)
def test_each_step_is_the_lower_of_newton_and_gauss_newton(kind, n, m):
    """Each step's cost is that of the same iterations written densely, with the whole Hessian inverted: Newton's where it is positive definite and lower."""
    case = Case.random(kind, 26, n, m, k=1, std=0.8)
    post = fips.solve(case.problem(), "gauss_newton", tol=1e-12)
    assert "newton" in dense_steps(case, 1e-12, len(post.iterations))[2]
    assert_dense_steps(post, case, 1e-12)


def wrong_curvature(factor: float):
    """A mutant whose curvature is the right one times `factor` (-1: the sign of D flipped)."""
    real = _Transformed._curvature

    def wrong(self, x, v):
        return factor * real(self, x, v)

    return wrong


@pytest.mark.parametrize("factor", [-1.0, 2.0, 0.5])
@pytest.mark.parametrize("kind", KINDS)
def test_a_wrong_curvature_fails(monkeypatch, kind, factor):
    """
    The curvature with its sign flipped, doubled or halved gives other steps than the dense iterations.

    The mode is still the mode (the steps are halved until the cost falls,
    and the last is Gauss-Newton's), so a wrong curvature shows only in the
    steps: here, and as more steps in the Salt Lake Valley-shaped world.
    """
    monkeypatch.setattr(_Transformed, "_curvature", wrong_curvature(factor))
    case = Case.random(kind, 26, 6, 15, k=1, std=0.8)
    post = fips.solve(case.problem(), "gauss_newton", tol=1e-12)
    assert_mode(fips.solve(case.problem(), "gauss_newton", **TIGHT), case)
    with pytest.raises(AssertionError):
        assert_dense_steps(post, case, 1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_the_draws_are_gauss_newtons_given_the_same_mode(monkeypatch, kind):
    """The sampler's mass is the Gaussian of the problem linearized at the last iterate: at a tight tol both paths end at the mode, and give the same draws from the same seed."""
    case = Case.random(kind, 27, 6, 10, k=1, std=0.8)
    newton = fips.solve(case.problem(), "gauss_newton", draws=300, burn_in=50, **TIGHT)
    gauss_newton_only(monkeypatch)
    alone = fips.solve(case.problem(), "gauss_newton", draws=300, burn_in=50, **TIGHT)
    std = np.sqrt(alone._solution.analytic.variances())
    np.testing.assert_allclose(
        newton._solution.point, alone._solution.point, rtol=0, atol=1e-9 * std.min()
    )
    np.testing.assert_allclose(newton._solution.analytic.variances(), std**2, rtol=1e-9)
    a, b = newton.sample(200, rng=3).to_numpy(), alone.sample(200, rng=3).to_numpy()
    assert np.abs((a - b) / std[:, None]).max() < 1e-7  # the paths amplify 1e-9
    np.testing.assert_allclose(
        newton.variances.values, alone.variances.values, rtol=1e-8
    )


def slv_shaped(kind: str, side: int = 10, m: int = 40, log_std: float = 1.0):
    """
    A Salt Lake Valley-shaped problem: `side` x `side` cells, `m` observations at three receptors, each seeing the cells upwind in a random direction (a sparse footprint), a lognormal inventory, and prior_for's prior with a flux uncertainty of a log standard deviation `log_std`, correlated over two cells.

    Returns a function of a seed that gives the problem of a truth drawn from
    the prior and its data.
    """
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    xy = np.c_[gx.ravel(), gy.ravel()].astype(float)
    cells = Space({"flux": pd.MultiIndex.from_arrays(xy.T, names=["x", "y"])})
    obs = Space({"ch4": pd.RangeIndex(m, name="obs_id")})
    receptors = rng.uniform(0, side - 1, (3, 2))[rng.integers(0, 3, m)]
    H = np.zeros((m, side * side))
    for i in range(m):
        angle = rng.uniform(0, 2 * np.pi)
        d = xy - receptors[i]
        along = d @ np.array([np.cos(angle), np.sin(angle)])
        across = d @ np.array([-np.sin(angle), np.cos(angle)])
        H[i] = np.exp(-np.abs(across) - np.maximum(along, 0) / 3.0) * (along > -1)
    H[H < 0.02] = 0.0
    H = Operator(sps.csr_array(H / H.sum(axis=1, keepdims=True) * 5.0), obs, cells)
    inventory = pd.Series(rng.lognormal(0.0, 0.8, side * side), index=cells.index)
    cv = np.sqrt(np.expm1(log_std**2))
    if kind == "lognormal":
        model = LognormalForward(H, inventory)
        mu, sigma = model.prior_for(model.reference, model.reference * cv)
    else:
        model = SoftplusForward(H, scale=float(inventory.median()))
        mu, sigma = model.prior_for(inventory, inventory * cv)
    distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    s = sigma.values
    prior_cov = Covariance(s[:, None] * np.exp(-distance / 2.0) * s[None, :], cells)

    def problem(seed: int) -> fips.Problem:
        def with_obs(z):
            return fips.Problem(
                prior=mu,
                prior_cov=prior_cov,
                obs=Vector(z, obs),
                obs_cov=fips.Independent().with_std(0.3),
                forward=model,
                offset=1900.0,
            )

        _, z = with_obs(np.zeros(m)).simulate(rng=seed)
        return with_obs(z.values)

    return problem


def steps_in_the_slv_shaped_world(kind: str, trials: int, **world) -> np.ndarray:
    """The steps each trial's solve takes, at the default options."""
    problem = slv_shaped(kind, **world)
    return np.array(
        [
            len(
                fips.solve(problem(seed), "gauss_newton", max_iterations=500).iterations
            )
            for seed in range(trials)
        ]
    )


@pytest.mark.parametrize("kind", KINDS)
def test_newton_takes_fewer_steps_in_a_slv_shaped_world(monkeypatch, kind):
    """
    At a log standard deviation of 1, on 100 cells and 40 observations, Newton takes fewer steps than Gauss-Newton alone, and never many.

    Measured (20 trials): lognormal, Gauss-Newton a median of 8 steps and at
    most 25, Newton 6 and 7; softplus, Gauss-Newton 13.5 and 40, Newton 9 and
    11. In the user guide's world, 400 cells and 120 observations (200
    trials): lognormal, Gauss-Newton 9 and 23, Newton 7 and 14; softplus,
    Gauss-Newton 18 and 153, Newton 9 and 14. There a wrong curvature takes
    more steps (50 trials, medians: lognormal 10 with its sign flipped and 10
    doubled, against 7; softplus 17, at most 75, and 12, against 9), but half
    the curvature takes about as many (8 and 8): the dense steps above catch
    all three.
    """
    newton = steps_in_the_slv_shaped_world(kind, 20)
    gauss_newton_only(monkeypatch)
    alone = steps_in_the_slv_shaped_world(kind, 20)
    assert np.median(newton) < np.median(alone)
    assert newton.max() < alone.max()
    assert newton.sum() < 0.8 * alone.sum()


@pytest.mark.parametrize("kind, seed", [("lognormal", 1), ("softplus", 1)])
def test_where_newton_is_not_positive_definite_the_step_is_gauss_newtons(
    caplog, kind, seed
):
    """
    A wide prior (a log standard deviation of 1.5): far from the mode the Newton matrix has a direction of negative curvature, that step is Gauss-Newton's, and the solve still ends at the mode.

    Far from the mode: the step after the warning has a d2 above 1, not one
    at roundoff (the softplus case of seed 3 met it only there, at a
    curvature of -5e-27, and not on another BLAS).
    """
    case = Case.random(kind, seed, 8, 12, k=1, std=1.5)
    with caplog.at_level("DEBUG", logger="fips.methods.gauss_newton"):
        post = fips.solve(case.problem(), "gauss_newton", **TIGHT)
    messages = [record.getMessage() for record in caplog.records]
    warned = [
        i
        for i, m in enumerate(messages)
        if "Newton matrix is not positive definite" in m
    ]
    assert warned
    after = next(m for m in messages[warned[0] :] if "d2 = " in m)
    assert after.startswith("Gauss-Newton step")
    assert float(after.split("d2 = ")[1].split(" ")[0]) > 1.0
    assert "Newton's)" in caplog.text  # and Newton steps elsewhere
    assert_mode(post, case)
