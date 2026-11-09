"""
Tests for the ``"gauss_newton"`` method: the mode of the posterior of a problem with a forward model.

A wrong mode still gives a labeled map of the right shape, so the method is
held to answers that do not use its algorithm, in layers that each catch what
the one before misses:

1. **Reduction.** With a forward operator the posterior is the analytic one,
   bit for bit (the oracle in ``test_oracle.py`` runs ``"gauss_newton"`` with
   its default options on every problem); a linear map written as a model
   gives the analytic posterior to roundoff, in both analytic forms and with
   a Kronecker prior.
2. **The mode, against an independent optimizer.** SciPy's
   ``least_squares`` on the whitened, stacked residual of the cost, for
   ``F(x) = H exp(x)``, with fewer and with more state elements than
   observations, another gamma, a Kronecker prior and a model on labels in
   another order.
3. **The Gaussian at the mode** against the dense formulas (its covariance,
   the gain, averaging kernel, innovation covariance, the chi-squared terms),
   each step's ``d2`` against the same steps written densely, and
   ``with_data`` against a fresh solve.
4. **The Taylor test** passes a right Jacobian and fails a wrong one.
5. **Wrong variants** of the method (the Jacobian at the prior, the offset
   without ``-J x``, ``d2`` in the prior covariance) fail the tests above.
6. **Its own world.** A truth drawn from the prior, data through the model:
   with a mild nonlinearity the intervals the draws give cover the truth about
   as often as they claim.
7. **Refusals and pickling.**

The draws of the posterior, which a sampler that has not mixed or draws the
wrong distribution still gives with the right shape, are held to:

8. **Exact answers**: one skewed cell (lognormal and softplus) against the
   posterior on a fine grid (mean, variance, 5% and 95% quantiles); the two
   cells of the guide (the total's mean and standard deviation, and with a
   looser observation the cells' correlation) against a 2-D grid; a linear model
   against the analytic posterior; within Monte Carlo errors from the spread
   of independent chains, which counts their autocorrelation.
9. **Its own world**: the truth's rank among the draws is uniform, for each
   element, the total flux and a difference.
10. **Wrong variants** of the sampler (no Metropolis test; velocities from
    the prior) fail the one-cell check.
11. **Reproducibility**: a seed gives the same draws, after a pickle too, and
    ``with_data`` draws the new data's posterior.
"""

import dataclasses
import math
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cholesky
from scipy.optimize import least_squares
from scipy.special import ndtri
from scipy.stats import chi2 as chi2_distribution

import fips
from fips import Covariance, ForwardModel, Operator, Problem, Space, Vector
from fips.linalg import DiagonalOperator, KroneckerOperator
from fips.methods import gauss_newton
from fips.methods._draws import _autocorrelation_times
from fips.methods.analytic import Information, Whitened
from fips.methods.gauss_newton import GaussNewton, GaussNewtonOptions
from fips.problem import Problem as _Problem

# ---------------------------------------------------------------------------
# Models, at the top level of the module so that they pickle
# ---------------------------------------------------------------------------


class LinearModel(ForwardModel):
    """A linear map written as a model: ``F(x) = H x``, the Jacobian ``H``."""

    def __init__(self, H: Operator):
        super().__init__(H.rows, H.cols)
        self.H = H

    def __call__(self, x):
        return self.H @ x

    def jacobian(self, x):
        return self.H


class ExpModel(ForwardModel):
    """``F(x) = H exp(x)``, the Jacobian ``H diag(exp(x))``; `sign` or `scale` make the Jacobian wrong."""

    def __init__(self, H: Operator, sign: float = 1.0, scale: float = 1.0):
        super().__init__(H.rows, H.cols)
        self.H = H
        self.sign = sign
        self.scale = scale

    def __call__(self, x):
        return self.H @ np.exp(x)

    def jacobian(self, x):
        factor = self.sign * self.scale * np.exp(x.values)
        return self.H @ Operator(DiagonalOperator(factor), x.space, x.space)


class SumModel(ForwardModel):
    """``F(x) = H exp(x) + B x``; with `missing`, a Jacobian without its ``B`` term."""

    def __init__(self, H: Operator, B: Operator, missing: bool = False):
        super().__init__(H.rows, H.cols)
        self.H, self.B, self.missing = H, B, missing

    def __call__(self, x):
        return self.H @ np.exp(x) + self.B @ x

    def jacobian(self, x):
        J = self.H @ Operator(DiagonalOperator(np.exp(x.values)), x.space, x.space)
        return J if self.missing else J + self.B


# ---------------------------------------------------------------------------
# Problems, written as arrays, and the references written with them
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Arrays:
    """A problem as arrays: ``z = H exp(x) + c + e``, ``x ~ N(x_0, S_0)``, ``e ~ N(0, S_z)``."""

    x_0: np.ndarray
    S_0: np.ndarray
    H: np.ndarray
    S_z: np.ndarray
    z: np.ndarray
    c: np.ndarray

    @classmethod
    def random(
        cls, seed: int, n: int, m: int, std: float = 0.5, corr: float = 3.0
    ) -> "Arrays":
        """A correlated prior of standard deviation about `std`, and data from a truth drawn from it."""
        rng = np.random.default_rng(seed)
        i = np.arange(n)
        C = np.exp(-np.abs(i[:, None] - i[None, :]) / corr)
        s = std * rng.uniform(0.7, 1.3, n)
        S_0 = s[:, None] * C * s[None, :]
        x_0 = rng.normal(0.0, 0.3, n)
        H = rng.uniform(0.0, 1.0, (m, n))
        S_z = np.diag(rng.uniform(0.01, 0.04, m))
        c = rng.normal(2.0, 0.1, m)
        truth = x_0 + cholesky(S_0, lower=True) @ rng.standard_normal(n)
        z = H @ np.exp(truth) + c + rng.normal(0.0, np.sqrt(np.diag(S_z)))
        return cls(x_0, S_0, H, S_z, z, c)

    def spaces(self) -> tuple[Space, Space]:
        """The state (one ``flux`` block) and the observations (one ``conc`` block)."""
        n, m = len(self.x_0), len(self.z)
        state = Space({"flux": pd.Index([f"c{k}" for k in range(n)], name="cell")})
        obs = Space({"conc": pd.Index([f"o{k}" for k in range(m)], name="obs_id")})
        return state, obs

    def problem(
        self,
        state: Space | None = None,
        prior_cov=None,
        model: type = ExpModel,
        permuted: bool = False,
    ) -> Problem:
        """The labeled problem with the forward `model` of ``H``; `permuted` puts the model's rows and columns in another order."""
        default_state, obs = self.spaces()
        state = state or default_state
        H = Operator(self.H, obs, state)
        if permuted:
            rows = Space({"conc": obs.select("conc")[::-1]})
            cols = Space({b: state.select(b)[::-1] for b in state.names})
            H = H.align(rows, cols)
        return Problem(
            prior=Vector(self.x_0, state),
            prior_cov=prior_cov
            if prior_cov is not None
            else Covariance(self.S_0, state),
            obs=Vector(self.z, obs),
            obs_cov=Covariance(self.S_z, obs),
            forward=model(H),
            offset=Vector(self.c, obs),
        )

    def F(self, x: np.ndarray) -> np.ndarray:
        """The model."""
        return self.H @ np.exp(x)

    def J(self, x: np.ndarray) -> np.ndarray:
        """Its Jacobian."""
        return self.H * np.exp(x)[None, :]

    def cost(self, x: np.ndarray, gamma: float = 1.0) -> float:
        """The MAP cost, the chi-squared at `x`."""
        r = self.z - self.F(x) - self.c
        dx = x - self.x_0
        return float(
            gamma * r @ np.linalg.solve(self.S_z, r)
            + dx @ np.linalg.solve(self.S_0, dx)
        )

    def least_squares(self, gamma: float = 1.0) -> np.ndarray:
        """
        The minimizer of the cost by SciPy's least squares on the whitened, stacked residual.

        Levenberg-Marquardt stops on the change of the cost, which is
        quadratic near the minimum: it is good to about 1e-7 of a posterior
        standard deviation (:meth:`mode` polishes it).
        """
        L_z = cholesky(self.S_z / gamma, lower=True)
        L_0 = cholesky(self.S_0, lower=True)

        def residual(x):
            return np.concatenate(
                [
                    np.linalg.solve(L_z, self.z - self.F(x) - self.c),
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

    def mode(self, gamma: float = 1.0) -> np.ndarray:
        """
        The minimizer of the cost to roundoff: :meth:`least_squares`, then Newton's method on the cost.

        Newton's steps use the cost's whole Hessian, the second derivatives
        of the model included (``exp`` is its own derivative), which
        Gauss-Newton leaves out: an algorithm Gauss-Newton is not, converging
        quadratically from the optimizer's answer.
        """
        x = self.least_squares(gamma)
        S_z = self.S_z / gamma
        for _ in range(3):
            w = np.linalg.solve(S_z, self.z - self.F(x) - self.c)
            J = self.J(x)
            gradient = np.linalg.solve(self.S_0, x - self.x_0) - J.T @ w
            hessian = (
                np.linalg.inv(self.S_0)
                + J.T @ np.linalg.solve(S_z, J)
                - np.diag(np.exp(x) * (self.H.T @ w))
            )
            x = x - np.linalg.solve(hessian, gradient)
        return x

    def gradient(self, x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Half the gradient of the cost at `x`: zero at the mode."""
        w = np.linalg.solve(self.S_z / gamma, self.z - self.F(x) - self.c)
        return np.linalg.solve(self.S_0, x - self.x_0) - self.J(x).T @ w

    def gaussian(self, x: np.ndarray, gamma: float = 1.0) -> dict[str, np.ndarray]:
        """The Gaussian at `x`, from the dense formulas: precision, covariance, gain, averaging kernel."""
        J = self.J(x)
        S_z = self.S_z / gamma
        P = np.linalg.inv(self.S_0) + J.T @ np.linalg.solve(S_z, J)
        S = np.linalg.inv(P)
        S = 0.5 * (S + S.T)
        K = S @ J.T @ np.linalg.inv(S_z)
        G = J @ self.S_0 @ J.T + S_z
        return {"P": P, "S": S, "K": K, "A": K @ J, "G": G, "J": J}

    def steps(self, k: int, gamma: float = 1.0) -> tuple[list[np.ndarray], list[float]]:
        """The first `k` Gauss-Newton iterates from the prior, and each step's d2 in the posterior precision of its solve, written densely."""
        x, iterates, d2 = self.x_0, [], []
        for _ in range(k):
            ref = self.gaussian(x, gamma)
            J = ref["J"]
            innovation = self.z - self.F(x) - self.c + J @ (x - self.x_0)
            new = self.x_0 + self.S_0 @ J.T @ np.linalg.solve(ref["G"], innovation)
            d2.append(float((new - x) @ ref["P"] @ (new - x)))
            iterates.append(new)
            x = new
        return iterates, d2


TIGHT = {"tol": 1e-22, "max_iterations": 100}
"""Options with which the iterations go on to roundoff, for the comparisons with the dense references."""

SIZES = [(6, 15), (15, 6)]
"""Fewer, then more state elements than observations."""


def sigma(arrays: Arrays, x: np.ndarray) -> np.ndarray:
    """The posterior standard deviations at `x`: the scale differences are measured in."""
    return np.sqrt(np.diag(arrays.gaussian(x)["S"]))


# ---------------------------------------------------------------------------
# 1. Reduction
# ---------------------------------------------------------------------------


def linear_problem(arrays: Arrays, model: bool, prior_cov=None, state=None) -> Problem:
    """The linear problem ``z = H x + c + e`` of `arrays`, with ``H`` as an operator or as a model."""
    default_state, obs = arrays.spaces()
    state = state or default_state
    H = Operator(arrays.H, obs, state)
    return Problem(
        prior=Vector(arrays.x_0, state),
        prior_cov=prior_cov if prior_cov is not None else Covariance(arrays.S_0, state),
        obs=Vector(arrays.z, obs),
        obs_cov=Covariance(arrays.S_z, obs),
        forward=LinearModel(H) if model else H,
        offset=Vector(arrays.c, obs),
    )


@pytest.mark.parametrize("n, m", SIZES)
def test_an_operator_forward_gives_the_analytic_solution_itself(n, m):
    """With a forward operator the method returns the analytic solution, so every number is the analytic one, bit for bit."""
    problem = linear_problem(Arrays.random(0, n, m), model=False)
    post = fips.solve(problem, "gauss_newton", gamma=2.0)
    analytic = fips.solve(problem, gamma=2.0)
    assert isinstance(post._solution, (Whitened, Information))
    assert post.options == GaussNewtonOptions(gamma=2.0)
    np.testing.assert_array_equal(post.estimate.values, analytic.estimate.values)
    np.testing.assert_array_equal(post.variances.values, analytic.variances.values)
    assert post.chi2 == analytic.chi2
    with pytest.raises(TypeError, match="iterations"):
        _ = post.iterations


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", SIZES)
@pytest.mark.parametrize("gamma", [1.0, 3.0])
def test_a_linear_model_gives_the_analytic_posterior(n, m, gamma):
    """
    A linear map written as a model converges in two steps to the analytic posterior, every member to roundoff but those read from draws.

    The variances and covariances are those of draws of the posterior, held
    to the analytic ones within their Monte Carlo errors by
    ``test_a_linear_model_draws_the_analytic_posterior``; the Gaussian at the
    mode they are drawn with is the analytic posterior to roundoff.
    """
    arrays = Arrays.random(1, n, m)
    post = fips.solve(linear_problem(arrays, model=True), "gauss_newton", gamma=gamma)
    ref = fips.solve(linear_problem(arrays, model=False), gamma=gamma)
    assert len(post.iterations) == 2
    assert isinstance(post._solution, GaussNewton)
    close = {"rtol": 1e-9, "atol": 1e-11}
    np.testing.assert_allclose(post.estimate.values, ref.estimate.values, **close)
    gaussian = post._solution.analytic
    np.testing.assert_allclose(gaussian.variances(), ref.variances.values, **close)
    np.testing.assert_allclose(gaussian.cov(), ref.cov().dense(), **close)
    np.testing.assert_allclose(
        post.kalman_gain.dense(), ref.kalman_gain.dense(), **close
    )
    np.testing.assert_allclose(
        post.averaging_kernel().dense(), ref.averaging_kernel().dense(), **close
    )
    np.testing.assert_allclose(
        post.innovation_cov.dense(), ref.innovation_cov.dense(), **close
    )
    np.testing.assert_allclose(post.modeled_obs.values, ref.modeled_obs.values, **close)
    for name in (
        "dofs",
        "information_content",
        "chi2",
        "data_misfit",
        "state_misfit",
        "r2",
    ):
        assert getattr(post, name) == pytest.approx(getattr(ref, name), rel=1e-9), name
    pd.testing.assert_frame_equal(post.chi2_tests(), ref.chi2_tests(), rtol=1e-8)


@pytest.mark.usefixtures("analytic_form")
def test_a_linear_model_with_a_kronecker_prior_gives_the_analytic_posterior():
    """A prior kept as Kronecker factors stays one through the linearization, and gives the analytic posterior."""
    rng = np.random.default_rng(3)
    times, cells = 3, 4
    T = np.exp(-np.abs(np.subtract.outer(np.arange(times), np.arange(times))) / 2.0)
    X = np.exp(-np.abs(np.subtract.outer(np.arange(cells), np.arange(cells))) / 1.5)
    kron = KroneckerOperator([T, X], std=rng.uniform(0.3, 0.6, times * cells))
    arrays = Arrays.random(3, times * cells, 9)
    arrays.S_0 = kron.dense()
    state = Space(
        {
            "flux": pd.MultiIndex.from_product(
                [range(times), range(cells)], names=["time", "cell"]
            )
        }
    )
    prior_cov = Covariance(kron, state)
    post = fips.solve(linear_problem(arrays, True, prior_cov, state), "gauss_newton")
    ref = fips.solve(linear_problem(arrays, False, prior_cov, state))
    assert isinstance(post._linear.prior_cov.data, KroneckerOperator)
    np.testing.assert_allclose(post.estimate.values, ref.estimate.values, rtol=1e-10)
    np.testing.assert_allclose(
        post._solution.analytic.variances(), ref.variances.values, rtol=1e-10
    )
    assert post.dofs == pytest.approx(ref.dofs, rel=1e-10)
    # the draws, through the Kronecker root and products: 2000, nearly independent
    np.testing.assert_allclose(post.variances.values, ref.variances.values, rtol=0.15)


# ---------------------------------------------------------------------------
# 2. The mode, against an independent optimizer
# ---------------------------------------------------------------------------


def assert_mode(
    post: fips.Posterior, arrays: Arrays, gamma: float = 1.0, atol: float = 1e-9
):
    """
    The estimate is the minimizer of the cost, within `atol` posterior standard deviations, and its chi-squared is the cost there.

    Three references, none of them Gauss-Newton: SciPy's least squares, to
    its own precision (1e-6); the same polished by Newton's method, to `atol`;
    and the gradient of the cost at the estimate, zero to `atol` (in the
    units of the state's posterior standard deviations).
    """
    x = post.estimate.values
    scipy_mode = arrays.least_squares(gamma)
    s = sigma(arrays, scipy_mode)
    assert np.abs((x - scipy_mode) / s).max() < 1e-6
    reference = arrays.mode(gamma)
    error = np.abs((x - reference) / s).max()
    assert error < atol, error
    assert np.abs(arrays.gradient(x, gamma) * s).max() < atol
    assert post.chi2 == pytest.approx(arrays.cost(reference, gamma), rel=1e-10)


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", SIZES)
@pytest.mark.parametrize("gamma", [1.0, 2.5])
def test_the_mode_is_the_minimizer_of_the_cost(n, m, gamma):
    """The estimate is the least-squares minimizer of the cost, to roundoff at a tight tol."""
    arrays = Arrays.random(4, n, m)
    post = fips.solve(arrays.problem(), "gauss_newton", gamma=gamma, **TIGHT)
    assert len(post.iterations) > 3  # the problem is nonlinear enough to iterate
    assert_mode(post, arrays, gamma)


@pytest.mark.parametrize("n, m", SIZES)
def test_the_default_tol_finds_the_mode_well_within_its_uncertainty(n, m):
    """At the default tol the mode is off by about a thousandth of a posterior standard deviation, at most."""
    arrays = Arrays.random(5, n, m)
    post = fips.solve(arrays.problem(), "gauss_newton")
    reference = arrays.mode()
    error = (post.estimate.values - reference) / sigma(arrays, reference)
    assert np.abs(error).max() < 2e-3


def test_a_model_on_labels_in_another_order_gives_the_same_mode():
    """A model whose rows and columns are in another order than the observations and the prior is called on its own order."""
    arrays = Arrays.random(6, 7, 11)
    ordered = fips.solve(arrays.problem(), "gauss_newton", **TIGHT)
    permuted = fips.solve(arrays.problem(permuted=True), "gauss_newton", **TIGHT)
    np.testing.assert_allclose(
        permuted.estimate.values, ordered.estimate.values, rtol=0, atol=1e-12
    )
    assert_mode(permuted, arrays)


@pytest.mark.usefixtures("analytic_form")
def test_a_kronecker_prior_gives_the_mode_of_the_dense_prior():
    """A Kronecker prior through the iterations gives the mode of the same prior written dense."""
    rng = np.random.default_rng(7)
    times, cells = 3, 4
    T = np.exp(-np.abs(np.subtract.outer(np.arange(times), np.arange(times))) / 2.0)
    X = np.exp(-np.abs(np.subtract.outer(np.arange(cells), np.arange(cells))) / 1.5)
    kron = KroneckerOperator([T, X], std=rng.uniform(0.3, 0.6, times * cells))
    arrays = Arrays.random(7, times * cells, 9)
    arrays.S_0 = kron.dense()
    state = Space(
        {
            "flux": pd.MultiIndex.from_product(
                [range(times), range(cells)], names=["time", "cell"]
            )
        }
    )
    post = fips.solve(
        arrays.problem(state, Covariance(kron, state)), "gauss_newton", **TIGHT
    )
    assert isinstance(post._linear.prior_cov.data, KroneckerOperator)
    assert_mode(post, arrays)


# ---------------------------------------------------------------------------
# 3. The posterior at the mode
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", SIZES)
def test_the_posterior_is_the_gaussian_at_the_mode(n, m):
    """The Gaussian at the mode (the sampler's mass), the gain, averaging kernel, innovation covariance and the modeled observations are the dense formulas at the mode."""
    gamma = 1.7
    arrays = Arrays.random(8, n, m)
    problem = arrays.problem()
    post = fips.solve(problem, "gauss_newton", gamma=gamma, **TIGHT)
    x_hat = arrays.mode(gamma)
    ref = arrays.gaussian(x_hat, gamma)
    close = {"rtol": 1e-8, "atol": 1e-12}
    assert post.problem is problem
    gaussian = post._solution.analytic
    np.testing.assert_allclose(gaussian.cov(), ref["S"], **close)
    np.testing.assert_allclose(gaussian.variances(), np.diag(ref["S"]), **close)
    np.testing.assert_allclose(post.kalman_gain.dense(), ref["K"], **close)
    np.testing.assert_allclose(post.averaging_kernel().dense(), ref["A"], **close)
    np.testing.assert_allclose(post.dofs_per_element.values, np.diag(ref["A"]), **close)
    assert post.dofs == pytest.approx(np.trace(ref["A"]), rel=1e-8)
    assert post.information_content == pytest.approx(
        -0.5 * np.linalg.slogdet(np.eye(n) - ref["A"])[1], rel=1e-8
    )
    np.testing.assert_allclose(post.innovation_cov.dense(), ref["G"], **close)
    Wd = np.vstack([np.arange(n) % 2 == 0, np.arange(n) % 2 == 1]).astype(float)
    W = Operator(
        Wd,
        Space({"total": pd.Index(["even", "odd"], name="part")}),
        problem.prior.space,
    )
    np.testing.assert_allclose(gaussian.cov(W.data), Wd @ ref["S"] @ Wd.T, **close)
    # the modeled observations, the residuals and the misfits are the model's at the mode
    modeled = arrays.F(x_hat) + arrays.c
    np.testing.assert_allclose(post.modeled_obs.values, modeled, rtol=1e-11)
    r = arrays.z - modeled
    assert post.data_misfit == pytest.approx(
        r @ np.linalg.solve(arrays.S_z, r), rel=1e-9
    )
    dx = x_hat - arrays.x_0
    assert post.state_misfit == pytest.approx(
        dx @ np.linalg.solve(arrays.S_0, dx), rel=1e-9
    )
    assert post.rmse == pytest.approx(np.sqrt(np.mean(r**2)), rel=1e-9)
    assert post.r2 == pytest.approx(
        np.corrcoef(arrays.z - arrays.c, arrays.F(x_hat))[0, 1] ** 2, rel=1e-12
    )


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", SIZES)
def test_the_chi2_tests_are_the_models_at_the_mode(n, m):
    """The innovation and the residual are the model's, against the innovation covariance at the mode; the increment's chi-squared is Rodgers' on the constrained directions."""
    gamma = 1.3
    arrays = Arrays.random(9, n, m)
    post = fips.solve(arrays.problem(), "gauss_newton", gamma=gamma, **TIGHT)
    x_hat = arrays.mode(gamma)
    ref = arrays.gaussian(x_hat, gamma)
    G = ref["G"]
    d = arrays.z - arrays.F(arrays.x_0) - arrays.c
    r = arrays.z - arrays.F(x_hat) - arrays.c
    C = (arrays.S_z / gamma) @ np.linalg.solve(G, arrays.S_z / gamma)
    table = post.chi2_tests()
    assert table.loc["innovation", "chi2"] == pytest.approx(
        d @ np.linalg.solve(G, d), rel=1e-8
    )
    assert table.loc["residual", "chi2"] == pytest.approx(
        r @ np.linalg.solve(C, r), rel=1e-8
    )
    increment = x_hat - arrays.x_0
    C_inc = arrays.S_0 @ ref["J"].T @ np.linalg.solve(G, ref["J"] @ arrays.S_0)
    expected = increment @ np.linalg.pinv(C_inc, rcond=1e-10) @ increment
    assert table.loc["increment", "chi2"] == pytest.approx(expected, rel=1e-6)
    assert table.loc["increment", "dof"] == min(n, m)
    assert (table["dof"].loc[["innovation", "residual"]] == m).all()


@pytest.mark.parametrize("n, m", SIZES)
def test_each_step_is_gauss_newton_and_its_d2_is_in_the_posterior_metric(n, m):
    """The steps' d2 are those of the same iterations written densely, in the posterior precision of the step's own solve."""
    arrays = Arrays.random(10, n, m)
    post = fips.solve(arrays.problem(), "gauss_newton", gamma=1.4, tol=1e-10)
    k = len(post.iterations)
    iterates, d2 = arrays.steps(k, gamma=1.4)
    np.testing.assert_allclose(
        post.estimate.values, iterates[-1], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(post.iterations["d2"].values[:3], d2[:3], rtol=1e-8)
    later = post.iterations["d2"].values[3:]
    np.testing.assert_allclose(later, d2[3:], rtol=1e-4, atol=1e-20)
    costs = [arrays.cost(x, 1.4) for x in iterates]
    np.testing.assert_allclose(post.iterations["cost"].values, costs, rtol=1e-10)
    assert post.iterations["d2"].iloc[-1] < 1e-10 * n <= post.iterations["d2"].iloc[-2]


@pytest.mark.parametrize("n, m", SIZES)
def test_with_data_is_a_fresh_solve(n, m):
    """The posterior of other observations, from the last linearization, is the mode a fresh solve finds."""
    arrays = Arrays.random(11, n, m)
    problem = arrays.problem()
    post = fips.solve(problem, "gauss_newton", **TIGHT)
    other = dataclasses.replace(
        arrays, z=arrays.z + np.random.default_rng(1).normal(0.0, 0.2, m)
    )
    again = post.with_data(Vector(other.z, problem.obs.space))
    fresh = fips.solve(other.problem(), "gauss_newton", **TIGHT)
    assert isinstance(again._solution, GaussNewton)
    assert again.problem.forward is problem.forward
    assert_mode(again, other)
    np.testing.assert_allclose(again.estimate.values, fresh.estimate.values, atol=1e-12)
    np.testing.assert_allclose(
        again._solution.analytic.variances(),
        fresh._solution.analytic.variances(),
        rtol=1e-9,
    )


def strongly_nonlinear(seed: int) -> Arrays:
    """A prior standard deviation of 1.5 in ``x`` and data five times noisier: ``exp`` bends much over the posterior."""
    arrays = Arrays.random(seed, 8, 6, std=1.5)
    return dataclasses.replace(arrays, S_z=arrays.S_z * 25)


def test_a_step_that_overshoots_is_halved_and_the_cost_falls(caplog):
    """A step that would raise the cost is taken shorter: the cost falls at every step, and the end is still the mode."""
    arrays = strongly_nonlinear(0)
    with caplog.at_level("DEBUG", logger="fips.methods.gauss_newton"):
        post = fips.solve(arrays.problem(), "gauss_newton", **TIGHT)
    assert "overshot" in caplog.text
    cost = post.iterations["cost"].values
    assert np.all(np.diff(cost) <= 1e-12 * cost[:-1])
    assert_mode(post, arrays)


def test_a_slow_solve_warns_that_it_stopped_short_of_the_mode(caplog):
    """Steps that fall by a constant factor near 1 leave more to go than the last step: the estimate is further from the mode than the rule meant, and the solve says so."""
    arrays = strongly_nonlinear(4)
    with caplog.at_level("WARNING", logger="fips.methods.gauss_newton"):
        post = fips.solve(arrays.problem(), "gauss_newton")
    assert "converged slowly" in caplog.text and "Lower tol" in caplog.text
    with pytest.raises(AssertionError):
        assert_mode(post, arrays, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. The Taylor test
# ---------------------------------------------------------------------------


def test_the_taylor_test_passes_a_right_jacobian_and_fails_a_wrong_one():
    """A right Jacobian passes; a sign flipped, a term missing or a scale off by 1% fails."""
    arrays = Arrays.random(12, 6, 9)
    state, obs = arrays.spaces()
    H = Operator(arrays.H, obs, state)
    B = Operator(np.random.default_rng(0).normal(size=(9, 6)), obs, state)
    x = Vector(arrays.x_0, state)
    assert fips.taylortest(ExpModel(H), x)
    assert fips.taylortest(SumModel(H, B), x)
    assert fips.taylortest(LinearModel(H), x)  # exact: the remainder is roundoff
    assert not fips.taylortest(ExpModel(H, sign=-1.0), x)
    assert not fips.taylortest(SumModel(H, B, missing=True), x)
    assert not fips.taylortest(ExpModel(H, scale=1.01), x)
    # another direction, and the state on labels in another order
    dx = Vector(np.ones(6), state)
    assert fips.taylortest(ExpModel(H), x, dx=dx)
    assert not fips.taylortest(ExpModel(H, sign=-1.0), x, dx=dx)
    reversed_state = Space({"flux": state.select("flux")[::-1]})
    assert fips.taylortest(ExpModel(H), x.align(reversed_state))


def test_the_taylor_test_refuses_what_is_not_a_model():
    """An operator is checked with dottest; a state on other labels raises."""
    arrays = Arrays.random(12, 3, 4)
    state, obs = arrays.spaces()
    H = Operator(arrays.H, obs, state)
    with pytest.raises(TypeError, match="dottest"):
        fips.taylortest(H, Vector(arrays.x_0, state))
    other = Space({"flux": pd.Index(["x", "y", "z"], name="cell")})
    with pytest.raises(ValueError, match="same labels"):
        fips.taylortest(ExpModel(H), Vector(arrays.x_0, other))


def test_the_taylor_test_refuses_a_step_that_tests_nothing():
    """A ``dx`` of zeros, an infinite one, or one so small that the output does not move above roundoff, has a zero remainder at every step, which a Jacobian wrong by 2 passed: each is refused by name; the default ``dx`` fails it."""
    arrays = Arrays.random(12, 6, 9)
    state, obs = arrays.spaces()
    H = Operator(arrays.H, obs, state)
    x = Vector(arrays.x_0, state)
    wrong = ExpModel(H, scale=2.0)
    assert not fips.taylortest(wrong, x)
    for values in (np.zeros(6), np.r_[np.inf, np.ones(5)]):  # a Vector refuses NaN
        with pytest.raises(ValueError, match="dx must be finite and not zero"):
            fips.taylortest(wrong, x, dx=Vector(values, state))
    with pytest.raises(ValueError, match="barely changes along dx"):
        fips.taylortest(wrong, x, dx=Vector(np.full(6, 1e-16), state))
    # a little above roundoff at the largest step, the remainder is at roundoff
    # whatever the Jacobian: the probe's case, a step of 1e-12 on its model
    for scale in (1.0, 2.0):
        with pytest.raises(ValueError, match="barely changes along dx"):
            fips.taylortest(
                ExpModel(H, scale=scale), x, dx=Vector(np.full(6, 1e-13), state)
            )
    # a direction the observations do not see: nothing to test either
    blind = Operator(np.c_[arrays.H[:, :5], np.zeros(9)], obs, state)
    with pytest.raises(ValueError, match="barely changes along dx"):
        fips.taylortest(ExpModel(blind), x, dx=Vector(np.r_[np.zeros(5), 1.0], state))


# ---------------------------------------------------------------------------
# 5. Wrong variants of the method fail the tests above
# ---------------------------------------------------------------------------


def at_the_prior(monkeypatch):
    """The Jacobian always at the prior, the model at the last iterate (a chord method: its fixed point is not the mode)."""
    real = _Problem._linearized

    def wrong(self, values, modeled=None):
        linear = real(self, self.prior.values)
        J = linear.forward.data
        modeled = self._apply(values) if modeled is None else modeled
        offset = self._offset_values() + modeled - np.asarray(J @ values)
        return Problem(
            self.prior,
            self.prior_cov,
            self.obs,
            self.obs_cov,
            linear.forward,
            Vector(offset, self.obs.space),
        )

    monkeypatch.setattr(_Problem, "_linearized", wrong)


def without_minus_Jx(monkeypatch):
    """The linearized offset ``c + F(x)``, without ``- J x``."""
    real = _Problem._linearized

    def wrong(self, values, modeled=None):
        linear = real(self, values, modeled)
        J = linear.forward.data
        offset = linear.offset.values + np.asarray(J @ values)
        return Problem(
            self.prior,
            self.prior_cov,
            self.obs,
            self.obs_cov,
            linear.forward,
            Vector(offset, self.obs.space),
        )

    monkeypatch.setattr(_Problem, "_linearized", wrong)


def d2_in_the_prior(monkeypatch):
    """The step measured in the prior covariance, without the observations' term."""
    monkeypatch.setattr(
        gauss_newton,
        "_d2",
        lambda dy, J_step, S_0, solve_S_z, gamma: float(dy @ np.asarray(S_0 @ dy)),
    )


@pytest.mark.parametrize(
    "mutant, check",
    [
        (at_the_prior, "mode"),
        (without_minus_Jx, "mode"),
        (without_minus_Jx, "linear"),
        (d2_in_the_prior, "steps"),
    ],
)
def test_a_wrong_variant_fails(monkeypatch, mutant, check):
    """Each wrong variant of the method fails a check of the right one."""
    mutant(monkeypatch)
    arrays = Arrays.random(4, 6, 15)
    with pytest.raises((AssertionError, RuntimeError)):
        if check == "mode":
            assert_mode(fips.solve(arrays.problem(), "gauss_newton", **TIGHT), arrays)
        elif check == "linear":
            post = fips.solve(linear_problem(arrays, model=True), "gauss_newton")
            ref = fips.solve(linear_problem(arrays, model=False))
            np.testing.assert_allclose(
                post.estimate.values, ref.estimate.values, rtol=1e-9
            )
        else:
            post = fips.solve(arrays.problem(), "gauss_newton", tol=1e-10)
            _, d2 = arrays.steps(len(post.iterations))
            np.testing.assert_allclose(
                post.iterations["d2"].values[:3], d2[:3], rtol=1e-8
            )


# ---------------------------------------------------------------------------
# 6. Its own world
# ---------------------------------------------------------------------------


def coverage(
    post: fips.Posterior, trials: int, levels
) -> dict[float, tuple[float, float]]:
    """The fraction of trials whose interval at each level holds the truth, over the elements, and its Monte Carlo standard error."""
    table = post.trials(n=trials, rng=0)
    error = (table["estimate"] - table["truth"]).abs()
    out = {}
    for level in levels:
        inside = error <= ndtri(0.5 + level / 2) * table["posterior_std"]
        by_trial = inside.groupby(level="trial").mean()
        out[level] = (float(by_trial.mean()), float(by_trial.std() / np.sqrt(trials)))
    return out


def test_in_a_mildly_nonlinear_world_the_intervals_cover_as_claimed():
    """
    A truth drawn from the prior and data through the model: the intervals ``trials`` reads (the draws' standard deviation about the mode) cover near their level when the model is nearly linear over the prior's spread.

    With a prior standard deviation of 0.1 in ``x``, the slope of ``exp(x)``
    changes by about 20% over two standard deviations. Measured (seed 0, 150
    trials of 8 elements, 150 kept draws each): coverage 0.954 at 95% and 0.680
    at 68%, with Monte Carlo standard errors of 0.006 and 0.015. Stronger
    worlds are in section 9 (the rank of the truth among the draws) and the
    guide.
    """
    arrays = Arrays.random(13, 8, 20, std=0.1)
    post = fips.solve(arrays.problem(), "gauss_newton", draws=150, burn_in=30)
    for level, (covered, error) in coverage(post, 150, (0.95, 0.68)).items():
        assert abs(covered - level) < 4 * error + 0.01, (level, covered, error)


# ---------------------------------------------------------------------------
# 7. Refusals and pickling
# ---------------------------------------------------------------------------


def test_the_analytic_and_bounded_methods_refuse_a_model():
    """Neither linearizes a model silently: each says which method and which member to use."""
    problem = Arrays.random(14, 4, 6).problem()
    with pytest.raises(TypeError, match=r'"gauss_newton".*problem\.linearized'):
        fips.solve(problem)
    with pytest.raises(TypeError, match=r"bounded method.*gauss_newton"):
        fips.solve(problem, "bounded", lower=0.0)


def test_the_posterior_refuses_what_its_method_cannot_give():
    """No mean, no other gamma without a solve, no closed-form recovery, no data worth."""
    problem = Arrays.random(14, 4, 6).problem()
    post = fips.solve(problem, "gauss_newton")
    with pytest.raises(AttributeError, match="mode|point estimate"):
        _ = post.mean
    with pytest.raises(TypeError, match="with_gamma"):
        post.with_gamma(2.0)
    with pytest.raises(TypeError, match="trials"):
        post.recovery()
    W = fips.aggregate.sum(problem.prior.space, by="cell")
    with pytest.raises(TypeError, match="data worth"):
        post.data_worth("obs_id", W)
    analytic = fips.solve(linear_problem(Arrays.random(14, 4, 6), model=False))
    with pytest.raises(TypeError, match=r"world.*model.*trials"):
        analytic.recovery(problem)


def test_a_solve_that_does_not_converge_raises():
    """Too few steps raise, naming the steps' d2 and what to try."""
    problem = Arrays.random(15, 6, 10).problem()
    with pytest.raises(RuntimeError, match=r"did not converge in 2 steps.*taylortest"):
        fips.solve(problem, "gauss_newton", max_iterations=2)


@pytest.mark.parametrize(
    "options, match",
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": 2.5}, "max_iterations"),
        ({"max_iterations": True}, "max_iterations"),
        ({"tol": 0.0}, "tol"),
        ({"tol": np.inf}, "tol"),
        ({"tol": "a"}, "tol"),
        ({"gamma": -1.0}, "gamma"),
        ({"draws": 0}, "draws"),
        ({"draws": True}, "draws"),
        ({"burn_in": -1}, "burn_in"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_the_options_are_checked(options, match):
    """Each option refuses a value it cannot use."""
    with pytest.raises(ValueError, match=match):
        GaussNewtonOptions(**options)


def test_a_problem_and_its_posterior_pickle():
    """With a model at the top level of a module, the problem and the posterior pickle and give the same answers."""
    arrays = Arrays.random(16, 5, 8)
    problem = arrays.problem()
    post = fips.solve(problem, "gauss_newton")
    again = pickle.loads(pickle.dumps(post))
    np.testing.assert_array_equal(again.estimate.values, post.estimate.values)
    np.testing.assert_array_equal(again.variances.values, post.variances.values)
    pd.testing.assert_frame_equal(again.iterations, post.iterations)
    assert not again._solution.mode.flags.writeable
    obs = Vector(arrays.z + 0.1, problem.obs.space)
    np.testing.assert_array_equal(
        again.with_data(obs).estimate.values, post.with_data(obs).estimate.values
    )
    loaded = pickle.loads(pickle.dumps(problem))
    np.testing.assert_array_equal(
        loaded.predict(loaded.prior).values, problem.predict(problem.prior).values
    )


def test_a_problem_whose_model_does_not_pickle_says_so():
    """A model class defined in a function does not pickle, and pickle names it."""

    class Local(ExpModel):
        pass

    arrays = Arrays.random(16, 3, 4)
    state, obs = arrays.spaces()
    problem = arrays.problem(model=Local)
    with pytest.raises((AttributeError, pickle.PicklingError), match="Local"):
        pickle.dumps(problem)
    assert state == problem.prior.space and obs == problem.obs.space


# ---------------------------------------------------------------------------
# 8. Draws of the posterior, against exact answers
# ---------------------------------------------------------------------------


def chain_estimates(post: fips.Posterior, statistic, chains: int, draws: int):
    """
    The mean over `chains` chains of `statistic` (of each chain's draws, ``n x draws``), and its Monte Carlo standard error.

    The error is the spread of the chains' values over the root of their
    number: it counts each chain's autocorrelation, whatever it is, where an
    estimate from one chain misses the rare long visits to a tail. Every
    chain starts where the burn-in ended, so the first 50 draws of each are
    left out, and the chains are independent.
    """
    values = np.array(
        [
            statistic(post.sample(draws + 50, rng=100 + k).to_numpy()[:, 50:])
            for k in range(chains)
        ]
    )
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(chains)


def assert_within(estimate, error, exact, within: float = 4.5):
    """Assert that each estimate is within `within` of its Monte Carlo errors of the exact value."""
    estimate, error, exact = (
        np.atleast_1d(np.asarray(v, float)) for v in (estimate, error, exact)
    )
    assert np.all(np.abs(estimate - exact) < within * error), (estimate, exact, error)


def one_cell(kind: str) -> tuple[Problem, "callable"]:
    """One cell seen once, near zero and with a wide prior, so that the posterior of ``u`` is skewed: the problem and the transform as a NumPy function."""
    state = Space({"flux": pd.Index(["a"], name="cell")})
    obs = Space({"conc": pd.RangeIndex(1, name="obs_id")})
    H = Operator(np.ones((1, 1)), obs, state)
    if kind == "lognormal":
        model, std, z, noise = fips.LognormalForward(H, 1.0), 1.5, 0.3, 0.1
        transform = np.exp
    else:
        model, std, z, noise = fips.SoftplusForward(H, scale=0.5), 1.0, 0.05, 0.05
        transform = lambda u: 0.5 * np.logaddexp(0.0, u / 0.5)  # noqa: E731
    problem = Problem(
        prior=Vector([0.0], state),
        prior_cov=Covariance(np.array([[std**2]]), state),
        obs=Vector([z], obs),
        obs_cov=Covariance(np.array([[noise**2]]), obs),
        forward=model,
    )
    return problem, transform


def grid_posterior(problem: Problem, transform, lower: float, upper: float):
    """The posterior of one element on a fine grid: its points and normalized weights (the trapezoidal rule on a smooth density is exact to roundoff here)."""
    u = np.linspace(lower, upper, 400_001)
    x_0 = problem.prior.values[0]
    s_0 = np.sqrt(problem.prior_cov.dense()[0, 0])
    z = problem.obs.values[0]
    s_z = np.sqrt(problem.obs_cov.dense()[0, 0])
    log_p = -0.5 * ((u - x_0) / s_0) ** 2 - 0.5 * ((z - transform(u)) / s_z) ** 2
    weights = np.exp(log_p - log_p.max())
    return u, weights / weights.sum()


@pytest.mark.parametrize("kind", ["lognormal", "softplus"])
def test_one_cell_draws_are_the_posterior(kind):
    """
    One skewed cell: the draws' mean, variance and 5% and 95% quantiles are those of the posterior on a fine grid, within their Monte Carlo errors.

    The Gaussian at the mode is off here by more than those errors: its mean
    (the mode) by 0.17 (lognormal) and 0.26 (softplus), and the lognormal's
    5% quantile by 0.4, where the posterior has the prior's long tail.
    """
    post = check_one_cell(kind, chains=16, draws=1000)
    # and the kept draws are the variances' and the covariance's
    kept = post._solution._kept()
    np.testing.assert_allclose(
        post.variances.values, kept.var(axis=1, ddof=1), rtol=1e-12
    )


def check_one_cell(kind: str, chains: int, draws: int) -> fips.Posterior:
    """Assert that the draws of the one-cell posterior have its mean, variance and 5% and 95% quantiles, within their Monte Carlo errors; return the posterior."""
    problem, transform = one_cell(kind)
    post = fips.solve(problem, "gauss_newton")
    u, w = grid_posterior(problem, transform, -18.0, 18.0)
    mean = float(w @ u)
    variance = float(w @ (u - mean) ** 2)
    cdf = np.cumsum(w)
    q05, q95 = np.interp([0.05, 0.95], cdf, u)
    assert abs(post.estimate.values[0] - mean) > 0.15  # the mode is not the mean

    def statistic(draws):
        d = draws[0]
        return [d.mean(), d.var(), *np.quantile(d, [0.05, 0.95])]

    estimate, error = chain_estimates(post, statistic, chains=chains, draws=draws)
    assert_within(estimate, error, [mean, variance, q05, q95])
    return post


def two_cells(noise: float = 0.05) -> Problem:
    """
    The user guide's two cells: their total observed as 2 with error `noise`, each with a lognormal prior of mean 1 and log standard deviation 1.

    The states that give the total lie on a curve, ``exp(u_a) + exp(u_b) =
    2``, and the Gaussian at the mode along its tangent: its draws give a total
    of 2.6 +- 0.9, the posterior 2.00 +- 0.05.
    """
    state = Space({"flux": pd.Index(["a", "b"], name="cell")})
    obs = Space({"conc": pd.RangeIndex(1, name="obs_id")})
    model = fips.LognormalForward(Operator(np.ones((1, 2)), obs, state), 1.0)
    mu, sigma = model.prior_for(1.0, np.sqrt(np.e - 1.0))
    return Problem(
        prior=mu,
        prior_cov=Covariance(np.diag(sigma.values**2), state),
        obs=Vector([2.0], obs),
        obs_cov=Covariance(np.array([[noise**2]]), obs),
        forward=model,
    )


def two_cell_moments(problem: Problem) -> np.ndarray:
    """The posterior's total mean and standard deviation, and the correlation of the two cells' ``u``, on a fine grid (the trapezoidal rule, exact to roundoff on this smooth density)."""
    x_0 = problem.prior.values
    s_0 = np.sqrt(np.diag(problem.prior_cov.dense()))
    s_z = np.sqrt(problem.obs_cov.dense()[0, 0])
    grid = np.linspace(-8.0, 3.0, 1601)
    a, b = np.meshgrid(grid, grid, indexing="ij")
    total = np.exp(a) + np.exp(b)
    log_p = (
        -0.5 * ((a - x_0[0]) / s_0[0]) ** 2
        - 0.5 * ((b - x_0[1]) / s_0[1]) ** 2
        - 0.5 * ((problem.obs.values[0] - total) / s_z) ** 2
    )
    w = np.exp(log_p - log_p.max())
    w /= w.sum()
    mean = float((w * total).sum())
    std = math.sqrt(float((w * (total - mean) ** 2).sum()))
    ma, mb = float((w * a).sum()), float((w * b).sum())
    cov = float((w * (a - ma) * (b - mb)).sum())
    corr = cov / math.sqrt(
        float((w * (a - ma) ** 2).sum()) * float((w * (b - mb) ** 2).sum())
    )
    return np.array([mean, std, corr])


def two_cell_statistic(draws: np.ndarray) -> list[float]:
    """The draws' total mean and standard deviation, and the correlation of the two cells."""
    total = np.exp(draws).sum(axis=0)
    return [total.mean(), total.std(), np.corrcoef(draws)[0, 1]]


def test_two_cells_draws_give_the_observed_total():
    """
    The two cells of the guide: the draws' total has the posterior's mean and standard deviation, within their Monte Carlo errors; with a total observed less closely, the cells have the posterior's correlation too.

    Measured (12 chains of 400 draws): the total 1.998 +- 0.051 against
    1.998 +- 0.050 on the grid, where the Gaussian at the mode's draws give
    2.6 +- 0.9. The total, which the data constrain, mixes in a few draws.
    The cells along the curve mix slowly when it is pinned this closely (an
    autocorrelation time of 50 draws by Sokal's window, hundreds by batch
    means: the Gaussian at the mode is the sampler's mass, narrow across the
    curve where the curve has turned), so chains this short give their
    correlation as -0.92 against -0.79, and it is not tested here. With the
    total's error 0.5 (the cells' autocorrelation time 3) the correlation
    is -0.61 +- 0.01 against -0.60.
    """
    problem = two_cells()
    post = fips.solve(problem, "gauss_newton", burn_in=300)
    gaussian = np.exp(
        post._solution.analytic.sample(4000, np.random.default_rng(0))
    ).sum(axis=0)
    exact = two_cell_moments(problem)
    assert gaussian.mean() > exact[0] + 0.4  # the Gaussian at the mode runs high
    estimate, error = chain_estimates(post, two_cell_statistic, chains=12, draws=400)
    assert_within(estimate[:2], error[:2], exact[:2])
    assert abs(estimate[0] - exact[0]) < 0.01 and abs(estimate[1] - exact[1]) < 0.01
    loose = two_cells(noise=0.5)
    post = fips.solve(loose, "gauss_newton")
    estimate, error = chain_estimates(post, two_cell_statistic, chains=12, draws=400)
    assert_within(estimate, error, two_cell_moments(loose))


@pytest.mark.slow
def test_two_cells_draws_give_the_observed_total_many_draws():
    """The two-cell test with chains of ten times the draws (the closely observed total, and the loosely observed cells' correlation)."""
    problem = two_cells()
    post = fips.solve(problem, "gauss_newton", burn_in=300)
    estimate, error = chain_estimates(post, two_cell_statistic, chains=12, draws=4000)
    assert_within(estimate[:2], error[:2], two_cell_moments(problem)[:2])
    loose = two_cells(noise=0.5)
    post = fips.solve(loose, "gauss_newton")
    estimate, error = chain_estimates(post, two_cell_statistic, chains=12, draws=4000)
    assert_within(estimate, error, two_cell_moments(loose))


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", SIZES)
def test_a_linear_model_draws_the_analytic_posterior(n, m):
    """
    A linear map written as a model: the draws have the analytic mean and covariance, and each is a quarter period of the Gaussian, accepted.

    The posterior is the Gaussian at the mode, so the remainder of the cost
    has no gradient, the burn-in tunes the step to one quarter period, every
    path is accepted, and the draws are exact and independent: the sampler
    costs one gradient a draw. So each mean and covariance is held to the
    error of independent draws.
    """
    arrays = Arrays.random(1, n, m)
    post = fips.solve(linear_problem(arrays, model=True), "gauss_newton")
    ref = fips.solve(linear_problem(arrays, model=False))
    tuning = post._solution._tuning()
    assert tuning.step == pytest.approx(gauss_newton._QUARTER)
    assert tuning.acceptance > 1 - 1e-9
    N = 4000
    draws = post.sample(N, rng=0).to_numpy()
    assert _autocorrelation_times(draws).max() < 1.3
    mean, S = ref.estimate.values, ref.cov().dense()
    variance = np.diag(S)
    assert_within(draws.mean(axis=1), np.sqrt(variance / N), mean)
    centered = draws - mean[:, None]
    i, j = np.triu_indices(n)
    products = (centered[i] * centered[j]).mean(axis=1)
    error = np.sqrt((variance[i] * variance[j] + S[i, j] ** 2) / N)
    assert_within(products, error, S[i, j])


# ---------------------------------------------------------------------------
# 9. Its own world: the truth's rank among the draws
# ---------------------------------------------------------------------------


def _calibration(trials: int, draws: int, seed: int) -> tuple[np.ndarray, float]:
    """
    Return, in the method's own world, the p-value of the uniformity of the truth's rank among the draws, for each quantity, and the coverage of the 90% intervals.

    A truth drawn from the prior of ``u`` (log standard deviation 0.5), data
    through ``F(x) = H exp(x)``. The quantities are each element, the total of
    ``exp(x)`` and the difference of the first two. The rank of the truth
    among a trial's draws (the fraction below it) is uniform over trials when
    the draws are of the posterior (simulation-based calibration, Talts et al.
    2018); its histogram in tenths is tested with a chi-squared test of 9
    degrees of freedom.
    """
    rng = np.random.default_rng(seed)
    n, m = 5, 4
    state = Space({"flux": pd.RangeIndex(n, name="cell")})
    obs = Space({"conc": pd.RangeIndex(m, name="obs_id")})
    x_0 = np.array([0.1, -0.3, 0.2, 0.0, -0.1])
    std = np.array([0.5, 0.4, 0.6, 0.5, 0.45])
    H = np.random.default_rng(99).uniform(0.0, 1.0, (m, n))
    problem = Problem(
        prior=Vector(x_0, state),
        prior_cov=Covariance.diagonal(std**2, state),
        obs=Vector(np.zeros(m), obs),
        obs_cov=Covariance.diagonal(np.full(m, 0.2**2), obs),
        forward=ExpModel(Operator(H, obs, state)),
    )
    post = fips.solve(problem, "gauss_newton", draws=draws, burn_in=50)
    ranks = np.empty((trials, n + 2))
    for k in range(trials):
        truth, z = problem.simulate(rng=rng)
        kept = post.with_data(z)._solution._kept()
        flux, true = np.exp(kept), np.exp(truth.values)
        quantities = np.vstack([kept, flux.sum(axis=0), flux[0] - flux[1]])
        values = np.concatenate([truth.values, [true.sum(), true[0] - true[1]]])
        ranks[k] = (quantities < values[:, None]).mean(axis=1)
    counts = np.stack([np.histogram(r, bins=10, range=(0.0, 1.0))[0] for r in ranks.T])
    expected = trials / 10
    p_values = chi2_distribution.sf(
        ((counts - expected) ** 2 / expected).sum(axis=1), 9
    )
    coverage = float(np.mean((ranks > 0.05) & (ranks < 0.95)))
    return p_values, coverage


def test_draws_are_calibrated_in_their_own_world():
    """
    In the world the posterior assumes, the truth is a draw like the others: its rank among the draws is uniform, for each element, the total flux and a difference.

    Measured (seed 0, 80 trials of 100 kept draws): the smallest p-value of
    the seven 0.12, and the 90% intervals' coverage 0.88.
    """
    p_values, coverage = _calibration(trials=80, draws=100, seed=0)
    assert p_values.min() > 1e-4, p_values
    assert abs(coverage - 0.9) < 0.06, coverage


@pytest.mark.slow
def test_draws_are_calibrated_in_their_own_world_many_trials():
    """The calibration test with six times the trials and five times the draws."""
    p_values, coverage = _calibration(trials=600, draws=1000, seed=1)
    assert p_values.min() > 1e-4, p_values
    assert abs(coverage - 0.9) < 0.02, coverage


# ---------------------------------------------------------------------------
# 10. Wrong variants of the sampler fail the tests above
# ---------------------------------------------------------------------------


def without_the_metropolis_test(self, here, v, test, h, steps):
    """Every path accepted: the splitting's error is kept, and the draws are not of the posterior."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            end, _ = self.travel(here, v, h, steps)
    except gauss_newton._FAILED:
        return here, 0.0
    return end, 1.0


def velocities_from_the_prior(self, count, rng):
    """Velocities drawn from the prior, ``N(0, S_0)``, while the energy counts them as ``N(0, S_hat)``."""
    root = self.problem.prior_cov.data._root
    done = 0
    while done < count:
        block = min(gauss_newton._BLOCK, count - done)
        V = np.asarray(root @ rng.standard_normal((root.shape[1], block)))
        tests = rng.random(block)
        jitters = rng.uniform(1 - gauss_newton._JITTER, 1 + gauss_newton._JITTER, block)
        for j in range(block):
            yield V[:, j], float(tests[j]), float(jitters[j])
        done += block


@pytest.mark.parametrize(
    "name, wrong, chains",
    [
        ("move", without_the_metropolis_test, 16),
        ("moves", velocities_from_the_prior, 4),
    ],
)
def test_a_wrong_sampler_fails(monkeypatch, name, wrong, chains):
    """
    Each wrong variant of the sampler draws another distribution, and the one-cell check against the grid fails (with a quarter of its chains for the second, which fails by far more).

    Measured with 8 chains of 1500 draws, in Monte Carlo
    errors: without the Metropolis test, the lognormal cell's mean off by
    +7.4 and its quantiles by +7.5 and +8.1 (the softplus cell's 95% quantile
    by +20); with velocities from the prior, the mean by -15 and the 5%
    quantile by -26.
    """
    monkeypatch.setattr(gauss_newton._Sampler, name, wrong)
    with pytest.raises(AssertionError):
        check_one_cell("lognormal", chains=chains, draws=1000)


def test_a_wrong_jacobian_leaves_the_draws_exact_but_slow_and_warns(
    monkeypatch, caplog
):
    """
    A Jacobian off by a factor steers the paths wrong, which the Metropolis test corrects: the draws stay exact, and the tuned step falls until a draw takes hundreds of steps, which the solve warns of.

    Measured on the one-cell lognormal problem with the Jacobian's transpose
    1.5 times too large: a step of 0.002 (905 steps a draw, against 2), and
    the one-cell check passes.
    """
    problem, _ = one_cell("lognormal")
    model = problem.forward
    real = type(model)._gradient

    def wrong(self, x, v):
        return 1.5 * real(self, x, v)

    monkeypatch.setattr(type(model), "_gradient", wrong)
    with caplog.at_level("WARNING", logger="fips.methods.gauss_newton"):
        post = fips.solve(problem, "gauss_newton", burn_in=100)
        steps = gauss_newton._steps(post._solution._tuning().step)
    assert steps >= gauss_newton._MANY_STEPS
    assert "steps a draw" in caplog.text and "taylortest" in caplog.text


# ---------------------------------------------------------------------------
# 11. Reproducible draws
# ---------------------------------------------------------------------------


def test_the_draws_are_reproducible_and_the_variances_are_theirs():
    """A seed gives the same draws, in a new solve and after a pickle; another seed others; the variances and covariances are the kept draws', ``sample(draws, rng=seed)``."""
    arrays = Arrays.random(17, 5, 8)
    problem = arrays.problem()
    post = fips.solve(problem, "gauss_newton", draws=300, burn_in=50, seed=7)
    draws = post.sample(300, rng=7).to_numpy()
    np.testing.assert_allclose(
        post.variances.values, draws.var(axis=1, ddof=1), rtol=1e-12
    )
    np.testing.assert_allclose(
        post.cov().dense(), np.cov(draws), rtol=1e-12, atol=1e-15
    )
    W = Operator(
        np.ones((1, 5)),
        Space({"total": pd.Index(["all"], name="part")}),
        problem.prior.space,
    )
    np.testing.assert_allclose(
        post.cov(W).dense(),
        np.atleast_2d(np.var(draws.sum(axis=0), ddof=1)),
        rtol=1e-12,
    )
    again = fips.solve(problem, "gauss_newton", draws=300, burn_in=50, seed=7)
    np.testing.assert_array_equal(again.variances.values, post.variances.values)
    state = post._solution.__getstate__()
    assert not {"_sampler_", "_tuning_", "_draws"} & set(state)
    back = pickle.loads(pickle.dumps(post))
    np.testing.assert_array_equal(back.variances.values, post.variances.values)
    np.testing.assert_array_equal(
        back.sample(50, rng=3).to_numpy(), post.sample(50, rng=3).to_numpy()
    )
    other = fips.solve(problem, "gauss_newton", draws=300, burn_in=50, seed=8)
    assert not np.array_equal(other.variances.values, post.variances.values)


def test_with_data_draws_the_posterior_of_the_new_data():
    """The posterior of other observations draws around its own mode, as a fresh solve of them does, within their Monte Carlo errors."""
    arrays = Arrays.random(18, 5, 8)
    problem = arrays.problem()
    post = fips.solve(problem, "gauss_newton", draws=1000, burn_in=100)
    other = dataclasses.replace(
        arrays, z=arrays.z + np.random.default_rng(2).normal(0.0, 0.2, 8)
    )
    again = post.with_data(Vector(other.z, problem.obs.space))
    fresh = fips.solve(other.problem(), "gauss_newton", draws=1000, burn_in=100)
    a, b = again._solution._kept(), fresh._solution._kept()
    # each chain's errors, with its own autocorrelation time
    tau_a, tau_b = (_autocorrelation_times(d) for d in (a, b))
    se = np.sqrt((a.var(axis=1) * tau_a + b.var(axis=1) * tau_b) / 1000)
    assert np.all(np.abs(a.mean(axis=1) - b.mean(axis=1)) < 4.5 * se)
    se_log = np.sqrt(2 * (tau_a + tau_b) / 1000)
    assert np.all(np.abs(np.log(a.var(axis=1) / b.var(axis=1))) < 4.5 * se_log)
    assert not np.array_equal(a.mean(axis=1), post._solution._kept().mean(axis=1))
