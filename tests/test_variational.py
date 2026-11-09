"""
The variational method: conjugate gradients and a low-rank posterior, from products alone.

``tests/test_oracle.py`` holds it to the closed form at its registered
``tol`` (``oracle``), with the default ``rank`` above every oracle problem's
rank. Here: the same agreement on larger problems, with Kronecker and
block-diagonal priors used through their factors, and with a matrix-free
forward operator; the low-rank posterior as ``rank`` grows (variances that
fall to the exact ones and DOFS that rise to them, monotonically); a forward
model against ``"gauss_newton"``; nothing ``n x n`` formed; the products of
:class:`fips.ForwardModel` and their adjoint test; and two wrong variants that
the comparisons catch.
"""

import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator as SciPyOperator

import fips
from fips.linalg import (
    BlockDiagonalOperator,
    DiagonalOperator,
    KroneckerOperator,
    LinearOperator,
    MatrixFreeOperator,
    dottest,
)
from fips.methods import variational
from fips.methods.variational import VariationalOptions

TIGHT = 1e-12
"""The ``tol`` the comparisons with the analytic posterior solve at."""

CLOSE = 1e-8
"""The relative tolerance they hold the posterior to, at ``TIGHT``."""


def correlation(rng: np.random.Generator, k: int, length: float) -> np.ndarray:
    """Return an exponential correlation on ``k`` points, a random length scale wide."""
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    return np.exp(-d / (length * rng.uniform(0.5, 1.5)))


def build(
    seed: int = 0,
    times: int = 4,
    cells: int = 6,
    m: int = 30,
    prior: str = "kronecker",
    sparse: bool = True,
    std: float = 1.0,
) -> fips.Problem:
    """A problem with a state of ``times x cells``, a prior of the given form, a sparse (or dense) H, and observations of a drawn truth."""
    rng = np.random.default_rng(seed)
    n = times * cells
    state = fips.Space(
        {
            "flux": pd.MultiIndex.from_product(
                [range(times), range(cells)], names=["time", "cell"]
            )
        }
    )
    observed = fips.Space({"co2": pd.RangeIndex(m, name="obs_id")})
    C_t, C_c = correlation(rng, times, 2.0), correlation(rng, cells, 3.0)
    sd = std * rng.uniform(0.5, 1.5, n)
    if prior == "kronecker":
        form = KroneckerOperator([C_t, C_c], std=sd)
    elif prior == "block":
        # the first times by Kronecker factors, the last one a dense block
        first = n - cells
        dense = sd[first:, None] * correlation(rng, cells, 4.0) * sd[None, first:]
        early = KroneckerOperator([C_t[:-1, :-1], C_c], std=sd[:first])
        form = BlockDiagonalOperator([early, dense])
    elif prior == "dense":
        form = sd[:, None] * np.kron(C_t, C_c) * sd[None, :]
    else:
        raise ValueError(prior)
    if sparse:  # each observation sees about a third of the cells, at most 60
        per_row = min(60, max(1, n // 3))
        rows = np.repeat(np.arange(m), per_row)
        cols = rng.integers(0, n, m * per_row)
        H = sp.csr_array((rng.uniform(0.0, 2.0, m * per_row), (rows, cols)), (m, n))
    else:
        H = rng.standard_normal((m, n))
    forward = fips.Operator(H, observed, state)
    obs_std = rng.uniform(0.3, 0.8, m)
    problem = fips.Problem(
        prior=fips.Vector(rng.normal(size=n), state),
        prior_cov=fips.Covariance(form, state),
        obs=fips.Vector(np.zeros(m), observed),
        obs_cov=fips.Covariance.diagonal(obs_std**2, observed),
        forward=forward,
        offset=0.2,
    )
    _, obs = problem.simulate(rng=rng)
    return fips.Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=obs,
        obs_cov=problem.obs_cov,
        forward=forward,
        offset=0.2,
    )


def regions(problem: fips.Problem) -> fips.Operator:
    """Return the sum of the state over two groups of cells (an aggregation ``W``), at each time."""
    cells = problem.prior.space.select("flux").get_level_values("cell").unique()
    by = pd.Series(
        np.where(np.asarray(cells) < 3, "west", "east"),
        index=pd.Index(cells, name="cell"),
        name="region",
    )
    return fips.aggregate.sum(problem.prior.space, by=by)


def assert_same(got, want, what: str, rtol: float = CLOSE) -> None:
    """Compare at `rtol`, with an absolute floor of `rtol` times the largest of `want`."""
    want = np.asarray(want, dtype=float)
    scale = max(float(np.abs(want).max(initial=0.0)), 1e-300)
    np.testing.assert_allclose(
        np.asarray(got, dtype=float), want, rtol=rtol, atol=rtol * scale, err_msg=what
    )


def assert_matches_analytic(post: fips.Posterior, problem: fips.Problem) -> None:
    """The variational posterior at full rank is the analytic one: estimate, variances, cov(W), gain, DOFS and the scalars."""
    exact = fips.solve(problem, gamma=post.options.gamma)
    assert_same(post.estimate.values, exact.estimate.values, "estimate")
    assert_same(post.variances.values, exact.variances.values, "variances")
    W = regions(problem)
    assert_same(post.cov(W).dense(), exact.cov(W).dense(), "cov(W)")
    assert_same(post.kalman_gain.dense(), exact.kalman_gain.dense(), "gain")
    assert_same(post.dofs_per_element.values, exact.dofs_per_element.values, "A_ii")
    for name in ("dofs", "information_content", "chi2", "log_evidence"):
        assert_same(getattr(post, name), getattr(exact, name), name)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": 2.5},
        {"rank": True},
        {"tol": 0.0},
        {"tol": "small"},
        {"outer_tol": np.inf},
        {"max_cg_iterations": 0},
        {"max_iterations": -1},
        {"gamma": -1.0},
    ],
)
def test_options_refuse_bad_values(kwargs):
    """Every option is checked when the options are made."""
    with pytest.raises(ValueError):
        VariationalOptions(**kwargs)


def test_options_are_registered():
    """``fips.solve`` knows the method and its options."""
    assert "variational" in fips.available_methods()
    with pytest.raises(ValueError, match="rank, tol"):
        fips.solve(build(), "variational", ranks=3)


# ---------------------------------------------------------------------------
# A linear problem: the analytic posterior at full rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prior", ["kronecker", "block", "dense"])
@pytest.mark.parametrize("sparse", [True, False], ids=["sparse H", "dense H"])
@pytest.mark.parametrize("gamma", [1.0, 0.4])
def test_full_rank_is_the_analytic_posterior(prior, sparse, gamma):
    """With rank at the problem's rank, every output equals the analytic posterior's to the CG tolerance."""
    problem = build(seed=3, prior=prior, sparse=sparse)
    post = fips.solve(problem, "variational", gamma=gamma, tol=TIGHT)
    assert len(post.eigenvalues) == min(len(problem.prior), len(problem.obs))
    assert_matches_analytic(post, problem)
    assert post.mean is post.estimate
    assert list(post.iterations.columns) == ["d2", "cost", "cg_iterations"]


def test_lanczos_and_the_dense_eigendecomposition_agree():
    """A rank below half the whitened state goes through Lanczos; the pairs are the dense eigendecomposition's."""
    problem = build(seed=4, times=6, cells=8, m=40)
    by_lanczos = fips.solve(problem, "variational", rank=10, tol=TIGHT)
    by_dense = fips.solve(problem, "variational", rank=40, tol=TIGHT)
    assert_same(
        by_lanczos.eigenvalues["eigenvalue"],
        by_dense.eigenvalues["eigenvalue"].iloc[:10],
        "eigenvalues",
        rtol=1e-9,
    )


def test_default_tol_is_close():
    """At the default tol the estimate is far inside its uncertainty from the analytic one (within 1e-4 of a posterior standard deviation)."""
    problem = build(seed=5, times=6, cells=10, m=50)
    post = fips.solve(problem, "variational")
    exact = fips.solve(problem)
    error = np.abs(post.estimate.values - exact.estimate.values)
    assert (error / np.sqrt(exact.variances.values)).max() < 1e-4


def test_matrix_free_forward_operator():
    """H given only as products (a SciPy LinearOperator) gives the same posterior, and fips.solve can still use it."""
    problem = build(seed=6, sparse=False)
    H = problem.forward.data.dense()
    products = SciPyOperator(
        H.shape, matvec=lambda v: H @ v, rmatvec=lambda w: H.T @ w, dtype=float
    )
    forward = fips.Operator(products, problem.obs.space, problem.prior.space)
    assert isinstance(forward.data, MatrixFreeOperator)
    free = fips.Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=problem.obs,
        obs_cov=problem.obs_cov,
        forward=forward,
        offset=problem.offset,
    )
    post = fips.solve(free, "variational", tol=TIGHT)
    assert_matches_analytic(post, problem)
    assert_same(
        fips.solve(free).estimate.values,
        fips.solve(problem).estimate.values,
        "analytic",
    )


def test_matrix_free_forward_in_another_order():
    """A matrix-free H on labels in another order is reordered by its products, not formed."""
    problem = build(seed=7, sparse=False)
    rng = np.random.default_rng(0)
    rows = rng.permutation(len(problem.obs))
    cols = rng.permutation(len(problem.prior))
    H = problem.forward.data.dense()[np.ix_(rows, cols)]
    obs_space = fips.Space.from_index(problem.obs.index[rows])
    state_space = fips.Space.from_index(problem.prior.index[cols])
    products = SciPyOperator(
        H.shape, matvec=lambda v: H @ v, rmatvec=lambda w: H.T @ w, dtype=float
    )
    free = fips.Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=problem.obs,
        obs_cov=problem.obs_cov,
        forward=fips.Operator(products, obs_space, state_space),
        offset=problem.offset,
    )
    assert isinstance(free.forward.data, MatrixFreeOperator)
    assert_matches_analytic(fips.solve(free, "variational", tol=TIGHT), problem)


# ---------------------------------------------------------------------------
# The low-rank posterior below full rank
# ---------------------------------------------------------------------------


def test_rank_curve_is_monotone():
    """
    As rank grows, the variances fall to the exact ones from above and DOFS rises to it from below, never the other way.

    The low-rank posterior keeps the prior in the directions it leaves out,
    so each variance is an upper bound that each added pair lowers.
    """
    problem = build(seed=8, times=6, cells=10, m=50)
    exact = fips.solve(problem)
    errors, dofs = [], []
    for rank in (1, 2, 5, 10, 20, 35, 50):
        post = fips.solve(problem, "variational", rank=rank, tol=TIGHT)
        excess = post.variances.values - exact.variances.values
        assert excess.min() > -1e-10 * exact.variances.values.max()
        errors.append(float(excess.max()))
        dofs.append(post.dofs)
        assert post.eigenvalues["cumulative_dofs"].iloc[-1] == pytest.approx(post.dofs)
    assert np.all(np.diff(errors) <= 1e-12) and np.all(np.diff(dofs) >= -1e-12)
    assert errors[-1] < 1e-8 * exact.variances.values.max()
    assert dofs[-1] == pytest.approx(exact.dofs, rel=1e-10)
    assert errors[0] > 100 * errors[-1]


def test_rank_too_small_warns(caplog):
    """A solve warns when the last pair it keeps still holds more information than the prior."""
    problem = build(seed=9, times=6, cells=10, m=50, std=5.0)
    with caplog.at_level("WARNING", logger="fips.methods.variational"):
        post = fips.solve(problem, "variational", rank=3)
    assert "Raise rank" in caplog.text
    assert post.eigenvalues["eigenvalue"].iloc[-1] > 1.0
    caplog.clear()
    with caplog.at_level("WARNING", logger="fips.methods.variational"):
        fips.solve(problem, "variational", rank=50)
    assert "Raise rank" not in caplog.text


def test_the_evidence_below_the_problems_rank_is_refused():
    """
    Below the problem's rank the low-rank evidence leaves out half the log of ``1 +`` each eigenvalue not kept, so it is biased high: refused while the last pair kept has an eigenvalue above 1, exact at full rank.

    Six cells seen by nine observations (exact log evidence -10.005): rank 1
    gave -4.60, rank 3 -7.86, rank 5 -9.60, each high by half the log of ``1
    +`` the eigenvalues left out, and a sweep of the prior scale peaked at 2
    or 4 instead of 1.
    """
    rng = np.random.default_rng(9)
    state = fips.Space({"flux": pd.RangeIndex(6, name="cell")})
    observed = fips.Space({"co2": pd.RangeIndex(9, name="obs_id")})
    problem = fips.Problem(
        prior=fips.Vector(np.zeros(6), state),
        prior_cov=fips.Independent().with_std(1.0),
        obs=fips.Vector(rng.normal(size=9), observed),
        obs_cov=fips.Independent().with_std(0.3),
        forward=fips.Operator(rng.normal(size=(9, 6)), observed, state),
    )
    exact = fips.solve(problem).log_evidence
    full = fips.solve(problem, "variational", rank=6, tol=TIGHT)
    assert full.log_evidence == pytest.approx(exact, rel=1e-10)
    for rank in (1, 3, 5):
        low = fips.solve(problem, "variational", rank=rank, tol=TIGHT)
        tilde = full.eigenvalues["eigenvalue"].to_numpy()
        assert low.eigenvalues["eigenvalue"].iloc[-1] > 1.0
        # what it would be: high by the information of the pairs left out
        biased = exact + 0.5 * np.log1p(tilde[rank:]).sum()
        terms = low._solution.information() - full._solution.information()
        assert terms == pytest.approx(-0.5 * np.log1p(tilde[rank:]).sum(), rel=1e-8)
        assert biased > exact + 0.1
        with pytest.raises(TypeError, match="biased high.*Raise rank"):
            _ = low.log_evidence
    # at a gamma where the last pair kept holds little, the evidence is given
    small = fips.solve(problem, "variational", rank=5, tol=TIGHT, gamma=1e-4)
    assert small.eigenvalues["eigenvalue"].iloc[-1] < 1.0
    assert np.isfinite(small.log_evidence)


def test_a_rank_deficient_forward_at_full_rank_neither_warns_nor_refuses(caplog):
    """Every pair computed (``rank`` at least ``min(m, r)``) and those at roundoff dropped: the low-rank posterior is the exact one, so the solve does not warn and the evidence is the analytic one."""
    rng = np.random.default_rng(0)
    state = fips.Space({"flux": pd.RangeIndex(6, name="cell")})
    observed = fips.Space({"co2": pd.RangeIndex(9, name="obs_id")})
    H = 5.0 * rng.normal(size=(9, 2)) @ rng.normal(size=(2, 6))  # rank 2
    problem = fips.Problem(
        prior=fips.Vector(np.zeros(6), state),
        prior_cov=fips.Independent().with_std(1.0),
        obs=fips.Vector(rng.normal(size=9), observed),
        obs_cov=fips.Independent().with_std(0.1),
        forward=fips.Operator(H, observed, state),
    )
    with caplog.at_level("WARNING", logger="fips.methods.variational"):
        post = fips.solve(problem, "variational", rank=100, tol=TIGHT)
    assert len(post.eigenvalues) == 2 and post.eigenvalues["eigenvalue"].min() > 1e3
    assert "Raise rank" not in caplog.text
    assert post.log_evidence == pytest.approx(fips.solve(problem).log_evidence)


def test_draws_are_the_low_rank_posterior():
    """Draws at a rank below the problem's have the low-rank posterior's mean and covariance (Monte Carlo error)."""
    problem = build(seed=10)
    post = fips.solve(problem, "variational", rank=5)
    N = 40_000
    draws = post.sample(N, rng=1).to_numpy()
    S = post.cov().dense()
    sd = np.sqrt(np.diag(S))
    np.testing.assert_array_less(
        np.abs(draws.mean(axis=1) - post.estimate.values), 5 * sd / np.sqrt(N)
    )
    np.testing.assert_allclose(
        np.cov(draws), S, atol=6 * np.outer(sd, sd).max() / np.sqrt(N)
    )


# ---------------------------------------------------------------------------
# Wrong variants the comparisons catch
# ---------------------------------------------------------------------------


def test_lambda_in_place_of_its_shrinkage_fails(monkeypatch):
    r"""The low-rank term with :math:`\lambda` in place of :math:`\lambda / (1 + \lambda)` gives variances the comparison refuses."""
    problem = build(seed=11)
    monkeypatch.setattr(variational, "_shrinkage", lambda eigenvalues: eigenvalues)
    post = fips.solve(problem, "variational", tol=TIGHT)
    with pytest.raises(AssertionError, match="variances"):
        assert_matches_analytic(post, problem)


def test_solve_without_the_prior_preconditioning_fails(monkeypatch):
    """CG on the normal equations without the prior's root (as if S_0 were the identity) gives an estimate the comparison refuses."""
    problem = build(seed=12)
    init = variational._System.__init__

    def unpreconditioned(self, linear, gamma):
        init(self, linear, gamma)
        identity = DiagonalOperator(np.ones(self.R.shape[0]))
        self.R, self.R_T = identity, identity

    monkeypatch.setattr(variational._System, "__init__", unpreconditioned)
    post = fips.solve(problem, "variational", tol=TIGHT)
    with pytest.raises(AssertionError, match="estimate"):
        assert_matches_analytic(post, problem)


# ---------------------------------------------------------------------------
# Nothing n x n
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prior", ["kronecker", "block"])
def test_no_n_by_n_matrix_is_formed(prior, monkeypatch):
    """
    With a Kronecker or block-diagonal prior, the solve and every output but cov() itself stay below n x n.

    The prior's dense matrix is refused outright, and the peak memory traced
    is a small fraction of one ``n x n`` array (2.6 GB at n = 18,000).
    """
    problem = build(seed=13, times=60, cells=300, m=400, prior=prior)
    n = len(problem.prior)

    def refuse(self):
        raise AssertionError(f"{type(self).__name__}.dense() was called")

    for cls in (KroneckerOperator, BlockDiagonalOperator):
        monkeypatch.setattr(cls, "dense", refuse)
    W = regions(problem)
    tracemalloc.start()
    try:
        post = fips.solve(problem, "variational", rank=20)
        # the log evidence's terms: at this rank the evidence itself is refused
        _ = post.variances, post.dofs, post.dofs_per_element, post.information_content
        _ = post.cov(W), post.sample(3, rng=0), post.eigenvalues, post.chi2_tests()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 0.05 * 8 * n * n, f"peak {peak / 1e6:.0f} MB"
    assert post.cov(W).shape == (W.shape[0], W.shape[0])


@pytest.mark.slow
def test_a_problem_whose_covariance_would_not_fit():
    """A state of 400,000 (an n x n array would be 1.3 TB), a Kronecker prior and a sparse H: the solve and its outputs run."""
    problem = build(seed=14, times=400, cells=1000, m=2000, prior="kronecker")
    post = fips.solve(problem, "variational", rank=30)
    assert np.isfinite(post.variances.values).all()
    assert (post.variances.values <= problem.prior_cov.data.diagonal() + 1e-12).all()
    assert post.dofs > 0


# ---------------------------------------------------------------------------
# Twins: with_data, with_gamma, pickle
# ---------------------------------------------------------------------------


def test_with_data_and_with_gamma_are_a_fresh_solve():
    """``with_data`` and ``with_gamma`` give what a fresh solve gives (one CG solve each, the eigenvectors kept)."""
    problem = build(seed=15)
    post = fips.solve(problem, "variational", tol=TIGHT)
    _, obs = problem.simulate(rng=np.random.default_rng(99))
    again = fips.solve(
        fips.Problem(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=obs,
            obs_cov=problem.obs_cov,
            forward=problem.forward,
            offset=problem.offset,
        ),
        "variational",
        tol=TIGHT,
    )
    twin = post.with_data(obs)
    assert_same(twin.estimate.values, again.estimate.values, "with_data estimate")
    assert_same(twin.variances.values, again.variances.values, "with_data variances")
    fresh = fips.solve(problem, "variational", gamma=3.0, tol=TIGHT)
    scaled = post.with_gamma(3.0)
    for name in ("estimate", "variances", "dofs_per_element"):
        assert_same(getattr(scaled, name).values, getattr(fresh, name).values, name)
    assert_same(scaled.chi2, fresh.chi2, "chi2")


def test_pickle_round_trip():
    """A posterior pickles and reads back the same, its arrays read-only."""
    post = fips.solve(build(seed=16), "variational")
    back = pickle.loads(pickle.dumps(post))
    assert_same(back.variances.values, post.variances.values, "variances", rtol=0)
    assert not back._solution.G.flags.writeable


def test_no_recovery_but_trials():
    """The estimate is not claimed linear in the data (recovery refuses); trials runs on with_data."""
    problem = build(seed=17)
    post = fips.solve(problem, "variational")
    with pytest.raises(TypeError, match="estimate_is_linear"):
        post.recovery(problem)
    table = post.trials(problem, 5, rng=0)
    assert len(table) == 5 * len(problem.prior)


def test_cg_that_does_not_converge_raises():
    """A solve whose conjugate gradients have not reached tol raises, saying what to change."""
    with pytest.raises(RuntimeError, match="max_cg_iterations"):
        fips.solve(build(seed=18), "variational", max_cg_iterations=1)


# ---------------------------------------------------------------------------
# A forward model: incremental 4D-Var against gauss_newton
# ---------------------------------------------------------------------------


def lognormal_problem(seed: int = 0, n: int = 24, m: int = 16):
    """A lognormal flux (``fips.LognormalForward``) and the same model written with products only."""
    rng = np.random.default_rng(seed)
    state = fips.Space({"flux": pd.RangeIndex(n, name="cell")})
    observed = fips.Space({"co2": pd.RangeIndex(m, name="obs_id")})
    H = fips.Operator(np.abs(rng.standard_normal((m, n))), observed, state)
    reference = rng.uniform(0.5, 2.0, n)
    model = fips.LognormalForward(
        H, reference=fips.Vector(reference, state), block="flux"
    )
    truth = fips.Vector(rng.normal(0.0, 0.5, n), state)
    obs = fips.Vector(model(truth).values + 0.1 * rng.standard_normal(m), observed)

    def problem(forward):
        return fips.Problem(
            prior=fips.Vector(np.zeros(n), state),
            prior_cov=fips.Exponential("cell", 3.0).with_std(0.5),
            obs=obs,
            obs_cov=fips.Independent().with_std(0.1),
            forward=forward,
        )

    return problem(model), problem(Products(H.data.dense(), reference, observed, state))


class Products(fips.ForwardModel):
    """``H (reference * exp(x))`` with its Jacobian as products only: a tangent linear and an adjoint."""

    def __init__(self, H, reference, rows, cols):
        super().__init__(rows, cols)
        self.H, self.reference = H, reference

    def __call__(self, x):
        return fips.Vector(self.H @ (self.reference * np.exp(x.values)), self.rows)

    def jvp(self, x, v):
        return fips.Vector(
            self.H @ (self.reference * np.exp(x.values) * v.values), self.rows
        )

    def vjp(self, x, w):
        return fips.Vector(
            self.reference * np.exp(x.values) * (self.H.T @ w.values), self.cols
        )


class WrongAdjoint(Products):
    """Its adjoint leaves out the derivative of the exponential."""

    def vjp(self, x, w):
        return fips.Vector(self.reference * (self.H.T @ w.values), self.cols)


def test_forward_model_matches_gauss_newton():
    """
    The mode is gauss_newton's, and the posterior is the Gaussian of the problem linearized one step from it.

    ``"gauss_newton"``'s shipped variances are from draws of the posterior
    itself; the variational ones are the Gaussian at the mode's (Laplace),
    and are compared with the analytic solve of the same linearization (to
    the CG tolerance) and with gauss_newton's Gaussian at its mode (its last
    linearization is another point one step from the mode).
    """
    problem, _ = lognormal_problem(seed=1)
    post = fips.solve(problem, "variational", tol=TIGHT, outer_tol=1e-14)
    newton = fips.solve(problem, "gauss_newton", tol=1e-14, draws=10, burn_in=0)
    sd = np.sqrt(newton._solution.analytic.variances())
    assert np.abs(post.estimate.values - newton.estimate.values).max() < 1e-6 * sd.min()
    linear = post._solution.linear
    gaussian = fips.solve(linear)
    assert_same(post.variances.values, gaussian.variances.values, "variances")
    W = fips.Operator(
        np.ones((1, len(problem.prior))),
        fips.Space({"total": pd.RangeIndex(1, name="t")}),
        problem.prior.space,
    )
    assert_same(post.cov(W).dense(), gaussian.cov(W).dense(), "cov(W)")
    assert_same(
        post.variances.values, newton._solution.analytic.variances(), "GN", rtol=1e-6
    )
    assert not hasattr(post, "mean") or pytest.raises(AttributeError, lambda: post.mean)
    with pytest.raises(AttributeError):
        _ = post.mean
    assert post.iterations["d2"].iloc[-1] < 1e-14 * len(problem.prior)
    assert_same(
        post.chi2_tests()["chi2"], newton.chi2_tests()["chi2"], "chi2 tests", 1e-6
    )


def test_products_only_model_is_the_jacobian_model():
    """A model written with jvp and vjp gives the posterior of the same model written with jacobian, never forming its Jacobian."""
    explicit, products = lognormal_problem(seed=2)
    a = fips.solve(explicit, "variational", tol=TIGHT, outer_tol=1e-14)
    b = fips.solve(products, "variational", tol=TIGHT, outer_tol=1e-14)
    assert isinstance(b._solution.linear.forward.data, MatrixFreeOperator)
    assert_same(b.estimate.values, a.estimate.values, "estimate", rtol=1e-9)
    assert_same(b.variances.values, a.variances.values, "variances", rtol=1e-7)
    back = pickle.loads(pickle.dumps(b))
    assert_same(back.variances.values, b.variances.values, "pickle", rtol=0)


def test_forward_model_with_data_and_no_with_gamma():
    """With a forward model, with_data solves again (the same as a fresh solve) and with_gamma refuses."""
    problem, _ = lognormal_problem(seed=3)
    post = fips.solve(problem, "variational", tol=TIGHT, outer_tol=1e-14)
    with pytest.raises(TypeError, match="with_gamma"):
        post.with_gamma(2.0)
    obs = fips.Vector(problem.obs.values * 1.05, problem.obs.space)
    fresh = fips.solve(
        fips.Problem(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=obs,
            obs_cov=problem.obs_cov,
            forward=problem.forward,
        ),
        "variational",
        tol=TIGHT,
        outer_tol=1e-14,
    )
    assert_same(
        post.with_data(obs).estimate.values, fresh.estimate.values, "estimate", 1e-7
    )


# ---------------------------------------------------------------------------
# The products of a forward model
# ---------------------------------------------------------------------------


def test_transforms_have_products_from_their_jacobian():
    """``jvp`` and ``vjp`` default to the Jacobian, so the transforms (and every model with a jacobian) have them."""
    problem, _ = lognormal_problem(seed=4)
    model = problem.forward
    rng = np.random.default_rng(0)
    x = fips.Vector(rng.normal(size=len(problem.prior)), model.cols)
    v = fips.Vector(rng.normal(size=len(problem.prior)), model.cols)
    w = fips.Vector(rng.normal(size=len(problem.obs)), model.rows)
    J = model.jacobian(x).dense()
    np.testing.assert_allclose(model.jvp(x, v).values, J @ v.values, rtol=1e-13)
    np.testing.assert_allclose(model.vjp(x, w).values, J.T @ w.values, rtol=1e-13)


def test_the_adjoint_test():
    """``taylortest`` checks the transpose against the products: a model with a wrong adjoint fails, a right one passes."""
    explicit, products = lognormal_problem(seed=5)
    model = products.forward
    x = fips.Vector(
        np.random.default_rng(1).normal(size=len(products.prior)), model.cols
    )
    assert fips.taylortest(model, x)
    wrong = WrongAdjoint(model.H, model.reference, model.rows, model.cols)
    assert not fips.taylortest(wrong, x)
    J = wrong.jacobian(x)
    assert isinstance(J.data, MatrixFreeOperator)
    assert not dottest(J.data, rtol=1e-10)
    assert fips.taylortest(explicit.forward, x)


def test_a_model_with_one_product_has_no_jacobian():
    """A model that writes jvp but not vjp (or neither) has no Jacobian, and says what to write."""

    class Half(fips.ForwardModel):
        def __call__(self, x):
            return x

        def jvp(self, x, v):
            return v

    class Neither(fips.ForwardModel):
        def __call__(self, x):
            return x

    space = fips.Space({"s": pd.RangeIndex(3, name="i")})
    x = fips.Vector(np.zeros(3), space)
    for cls in (Half, Neither):
        model = cls(space, space)
        with pytest.raises(NotImplementedError, match="jvp and vjp"):
            model.jacobian(x)
    with pytest.raises(NotImplementedError, match="neither jacobian nor vjp"):
        Half(space, space).vjp(x, x)


# ---------------------------------------------------------------------------
# The matrix-free form
# ---------------------------------------------------------------------------


def test_matrix_free_form_stays_matrix_free():
    """Products, sums, scalings, transposes and selections of a matrix-free form are applied term by term, and equal the dense ones."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 4))
    calls = []

    def matvec(v):
        calls.append(1)
        return A @ v

    form = MatrixFreeOperator(
        SciPyOperator((5, 4), matvec=matvec, rmatvec=lambda w: A.T @ w, dtype=float)
    )
    D = DiagonalOperator(np.arange(1.0, 5.0))
    S = fips.linalg.SparseOperator(sp.csr_array(rng.standard_normal((4, 3))))
    cases = {
        "A D": (form @ D, A @ np.diag(np.arange(1.0, 5.0))),
        "A S": (form @ S, A @ S.dense()),
        "(A S)^T": ((form @ S)._transpose(), (A @ S.dense()).T),
        "A + A": (form + form, 2 * A),
        "-A / 2": (-form / 2, -A / 2),
        "A^T A": (form._transpose() @ form, A.T @ A),
        "selected": (
            form._select(np.array([4, -1, 0]), np.array([1, 1, 3])),
            np.vstack([A[4, [1, 1, 3]], np.zeros(3), A[0, [1, 1, 3]]]),
        ),
    }
    for what, (got, want) in cases.items():
        assert isinstance(got, MatrixFreeOperator), what
        np.testing.assert_allclose(got.dense(), want, rtol=1e-13, err_msg=what)
        assert dottest(got), what
    calls.clear()
    _ = form @ D
    assert not calls  # nothing applied until a product is asked for
    assert isinstance(fips.linalg.as_linear_operator(form._op), MatrixFreeOperator)
    with pytest.raises(TypeError):
        MatrixFreeOperator(D)


def test_kronecker_and_block_roots_have_transposes():
    """The roots draws are made with (Kronecker on part of its grid, block diagonal) have transposes, which the whitening needs."""
    rng = np.random.default_rng(2)
    K = KroneckerOperator(
        [correlation(rng, 3, 2.0), correlation(rng, 4, 2.0)],
        std=rng.uniform(0.5, 1.5, 9),
        positions=np.unravel_index(rng.permutation(12)[:9], (3, 4)),
    )
    B = BlockDiagonalOperator([K, DiagonalOperator([1.0, 4.0])])
    for form in (K, B):
        R = form._root
        assert isinstance(R, LinearOperator)
        np.testing.assert_allclose(R._transpose().dense(), R.dense().T, rtol=1e-13)
        np.testing.assert_allclose(R.dense() @ R.dense().T, form.dense(), atol=1e-12)
        assert dottest(R)
