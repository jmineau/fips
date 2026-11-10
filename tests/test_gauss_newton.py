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
3. **The posterior at the mode** against the dense formulas of the Gaussian
   at the mode (covariance, gain, averaging kernel, innovation covariance,
   the chi-squared terms), each step's ``d2`` against the same steps written
   densely, and ``with_data`` against a fresh solve.
4. **The Taylor test** passes a right Jacobian and fails a wrong one.
5. **Wrong variants** of the method (the Jacobian at the prior, the offset
   without ``-J x``, ``d2`` in the prior covariance) fail the tests above.
6. **Its own world.** A truth drawn from the prior, data through the model:
   with a mild nonlinearity the intervals of the Gaussian at the mode cover
   the truth about as often as they claim.
7. **Refusals and pickling.**
"""

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cholesky
from scipy.optimize import least_squares
from scipy.special import ndtri

import fips
from fips import Covariance, ForwardModel, Operator, Problem, Space, Vector
from fips.linalg import DiagonalOperator, KroneckerOperator
from fips.methods import gauss_newton
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
    """A linear map written as a model converges in two steps to the analytic posterior, every member to roundoff."""
    arrays = Arrays.random(1, n, m)
    post = fips.solve(linear_problem(arrays, model=True), "gauss_newton", gamma=gamma)
    ref = fips.solve(linear_problem(arrays, model=False), gamma=gamma)
    assert len(post.iterations) == 2
    assert isinstance(post._solution, GaussNewton)
    close = {"rtol": 1e-9, "atol": 1e-11}
    np.testing.assert_allclose(post.estimate.values, ref.estimate.values, **close)
    np.testing.assert_allclose(post.variances.values, ref.variances.values, **close)
    np.testing.assert_allclose(post.cov().dense(), ref.cov().dense(), **close)
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
    np.testing.assert_allclose(post.variances.values, ref.variances.values, rtol=1e-10)
    assert post.dofs == pytest.approx(ref.dofs, rel=1e-10)


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
    """Covariance, gain, averaging kernel, innovation covariance and the modeled observations are the dense formulas at the mode."""
    gamma = 1.7
    arrays = Arrays.random(8, n, m)
    problem = arrays.problem()
    post = fips.solve(problem, "gauss_newton", gamma=gamma, **TIGHT)
    x_hat = arrays.mode(gamma)
    ref = arrays.gaussian(x_hat, gamma)
    close = {"rtol": 1e-8, "atol": 1e-12}
    assert post.problem is problem
    np.testing.assert_allclose(post.cov().dense(), ref["S"], **close)
    np.testing.assert_allclose(post.variances.values, np.diag(ref["S"]), **close)
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
    np.testing.assert_allclose(post.cov(W).dense(), Wd @ ref["S"] @ Wd.T, **close)
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
        again.variances.values, fresh.variances.values, rtol=1e-9
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


def coverage(post: fips.Posterior, trials: int, level: float) -> tuple[float, float]:
    """The fraction of trials whose interval at `level` holds the truth, over the elements, and its Monte Carlo standard error."""
    table = post.trials(n=trials, rng=0)
    z = ndtri(0.5 + level / 2)
    inside = (table["estimate"] - table["truth"]).abs() <= z * table["posterior_std"]
    by_trial = inside.groupby(level="trial").mean()
    return float(by_trial.mean()), float(by_trial.std() / np.sqrt(trials))


def test_in_a_mildly_nonlinear_world_the_intervals_cover_as_claimed():
    """
    A truth drawn from the prior and data through the model: the Gaussian at the mode covers near its level when the model is nearly linear over the prior's spread.

    With a prior standard deviation of 0.1 in ``x``, the slope of ``exp(x)``
    changes by about 20% over two standard deviations. Measured (seed 0, 300
    trials of 8 elements): coverage 0.946 at 95% and 0.686 at 68%, with Monte
    Carlo standard errors of 0.005 and 0.010. Stronger worlds are in the
    guide, where the Gaussian at the mode under-covers.
    """
    arrays = Arrays.random(13, 8, 20, std=0.1)
    post = fips.solve(arrays.problem(), "gauss_newton")
    for level in (0.95, 0.68):
        covered, error = coverage(post, 300, level)
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
