"""
Tests for the whitened solve: the factorization ``BayesianSolver`` keeps.

The oracle (``test_oracle.py``) holds the estimate, the full posterior
covariance, the gain, the averaging kernel, DOFS and the reduced chi-squared to
the closed form. Here: what the factorization adds (the variances, aggregated
covariances and averaging kernels, ``with_gamma``), the direct solve the
factorization replaced, the inputs it refuses, its pickle, and that nothing
``n x n`` is allocated unless the full covariance or averaging kernel is read.
"""

import logging
import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from numpy.linalg import LinAlgError

import fips
from fips import (
    Covariance,
    Exponential,
    InverseProblem,
    Kronecker,
    Operator,
    Space,
    Vector,
)
from fips.estimators import BayesianSolver, Estimator
from fips.kronecker import KroneckerOperator
from fips.linalg import DiagonalOperator, SparseOperator

RTOL = 1e-10
GAMMAS = [0.1, 0.5, 1.0, 3.0, 20.0]


def _close(actual, expected, what, rtol=RTOL):
    """Compare at `rtol`, with an absolute floor of `rtol` times the largest entry."""
    expected = np.asarray(expected, dtype=float)
    floor = rtol * max(np.abs(expected).max(), 1e-300)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), expected, rtol=rtol, atol=floor, err_msg=what
    )


def _spd(rng, k):
    """A well-conditioned SPD matrix."""
    B = rng.standard_normal((k, k))
    return B @ B.T + k * np.eye(k)


# ---------------------------------------------------------------------------
# Problems, in the forms the solver takes
# ---------------------------------------------------------------------------

FORMS = ["dense", "sparse H, diagonal S_z", "Kronecker S_0", "sparse S_0 and S_z"]


def _arrays(form: str, seed: int, n_time: int = 3, n_cell: int = 4, m: int = 7):
    """Return ``z, x_0, H, S_0, S_z, c`` with ``n = n_time * n_cell`` in one of `FORMS`."""
    rng = np.random.default_rng(seed)
    n = n_time * n_cell
    H = rng.standard_normal((m, n))
    S_0 = _spd(rng, n)
    S_z = _spd(rng, m)
    if form == "sparse H, diagonal S_z":
        H[np.abs(H) < 0.8] = 0.0
        H = sp.csr_matrix(H)
        S_z = sp.diags(rng.uniform(0.5, 2.0, m)).tocsr()
    elif form == "Kronecker S_0":
        S_0 = KroneckerOperator(
            [_spd(rng, n_time) / n_time, _spd(rng, n_cell) / n_cell],
            std=rng.uniform(0.5, 2.0, n),
        )
    elif form == "sparse S_0 and S_z":
        S_0 = sp.csr_matrix(
            np.diag(rng.uniform(1, 3, n))
            + np.diag(np.full(n - 1, 0.3), 1)
            + np.diag(np.full(n - 1, 0.3), -1)
        )
        S_z = sp.csr_matrix(
            np.diag(rng.uniform(1, 3, m))
            + np.diag(np.full(m - 1, 0.2), 1)
            + np.diag(np.full(m - 1, 0.2), -1)
        )
    x_0 = rng.normal(2.0, 1.0, n)
    z = np.asarray(H @ x_0) + rng.normal(0.0, 2.0, m)
    c = rng.normal(0.5, 0.1, m)
    return z, x_0, H, S_0, S_z, c


def _dense(a):
    """A matrix in any form, as a dense array."""
    return a.toarray() if sp.issparse(a) else np.asarray(a, dtype=float)


def _closed_form(z, x_0, H, S_0, S_z, c, gamma):
    """The posterior and its diagnostics, with explicit inverses (as the oracle writes them)."""
    inv = np.linalg.inv
    H, S_0, S_z = _dense(H), _dense(S_0), _dense(S_z) / gamma
    K = S_0 @ H.T @ inv(H @ S_0 @ H.T + S_z)
    x_hat = x_0 + K @ (z - H @ x_0 - c)
    S_hat = inv(H.T @ inv(S_z) @ H + inv(S_0))
    r, dx = z - H @ x_hat - c, x_hat - x_0
    return {
        "x_hat": x_hat,
        "S_hat": S_hat,
        "S_hat_diagonal": np.diag(S_hat),
        "K": K,
        "A": K @ H,
        "DOFS": np.trace(K @ H),
        "reduced_chi2": (r @ inv(S_z) @ r + dx @ inv(S_0) @ dx) / len(z),
        "U_red": (1 - np.sqrt(np.diag(S_hat)) / np.sqrt(np.diag(S_0))) * 100,
        "uncertainty_reduction": 1 - np.sqrt(np.trace(S_hat)) / np.sqrt(np.trace(S_0)),
    }


class _Direct(Estimator):
    """
    The direct solve the factorization replaced.

    The base class's gain, from ``(H S_0 H^T + S_z)`` by Cholesky, and
    ``S_hat = S_0 - K H S_0``.
    """

    def __init__(self, z, x_0, H, S_0, S_z, c=None, gamma=1.0):
        super().__init__(z, x_0, H, S_0, S_z, c)
        self._S_z_op = self._S_z_op / gamma

    def cost(self, x):
        """Not used."""
        raise NotImplementedError

    @property
    def x_hat(self):
        """The estimate from the base class's gain."""
        return self.x_0 + self.K @ self.residual(self.x_0)

    @property
    def S_hat(self):
        """The posterior covariance from the base class's gain."""
        return self._S_0_op.dense() - self.K @ self._HS_0

    @property
    def reduced_chi2(self):  # type: ignore[override]
        """The two misfits, each by a solve with its covariance."""
        r, dx = self.residual(self.x_hat), self.x_hat - self.x_0
        return float(
            (r @ self._S_z_op.solve(r) + dx @ self._S_0_op.solve(dx)) / self.n_z
        )


OUTPUTS = [
    "x_hat", "S_hat", "S_hat_diagonal", "K", "A", "DOFS", "reduced_chi2", "U_red",
    "uncertainty_reduction", "y_hat", "y_0", "S_y0", "R2", "RMSE", "desroziers",
]  # fmt: skip


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("seed", range(5))
def test_outputs_match_the_closed_form(form, gamma, seed):
    """Every output the factors give, at 1e-10 of the closed form."""
    z, x_0, H, S_0, S_z, c = _arrays(form, seed)
    solver = BayesianSolver(z, x_0, H, S_0, S_z, c=c, gamma=gamma)
    for name, expected in _closed_form(z, x_0, H, S_0, S_z, c, gamma).items():
        _close(getattr(solver, name), expected, f"{name}, {form}, gamma {gamma}")


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("gamma", [0.5, 1.0, 4.0])
@pytest.mark.parametrize("shape", [(3, 4, 7), (2, 2, 9), (4, 5, 2)], ids=str)
def test_outputs_match_the_direct_solve(form, gamma, shape):
    """Every output agrees with the direct solve (the code before the factorization) at 1e-10."""
    args = _arrays(form, 3, *shape)
    solver = BayesianSolver(*args[:5], c=args[5], gamma=gamma)
    direct = _Direct(*args[:5], c=args[5], gamma=gamma)
    for name in OUTPUTS:
        _close(getattr(solver, name), getattr(direct, name), f"{name}, {form}")


@pytest.mark.parametrize("form", FORMS)
def test_with_gamma_is_a_fresh_solve(form):
    """``with_gamma`` gives what a new solver with that gamma gives, from the same factors."""
    z, x_0, H, S_0, S_z, c = _arrays(form, 7)
    solver = BayesianSolver(z, x_0, H, S_0, S_z, c=c)
    _ = solver.x_hat
    for gamma in GAMMAS:
        swept = solver.with_gamma(gamma)
        assert swept._whitened is solver._whitened
        assert swept.gamma == gamma
        fresh = BayesianSolver(z, x_0, H, S_0, S_z, c=c, gamma=gamma)
        for name in OUTPUTS:
            _close(getattr(swept, name), getattr(fresh, name), f"{name}, gamma {gamma}")
    assert solver.gamma == 1.0


# ---------------------------------------------------------------------------
# The labeled posterior
# ---------------------------------------------------------------------------


def _labeled(seed: int = 0, n_time: int = 3, n_cell: int = 4, m: int = 9):
    """A labeled problem with a Kronecker prior error and a sparse ``H``."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_time, freq="D")
    cells = [f"c{k}" for k in range(n_cell)]
    state = pd.MultiIndex.from_product(
        [["flux"], times, cells], names=["block", "time", "cell"]
    )
    obs = pd.MultiIndex.from_product([["conc"], range(m)], names=["block", "obs_id"])
    H = rng.uniform(0.0, 1.0, (m, len(state)))
    H[H < 0.5] = 0.0
    S_0 = KroneckerOperator(
        [_spd(rng, n_time) / n_time, _spd(rng, n_cell) / n_cell],
        std=rng.uniform(0.5, 2.0, len(state)),
    )
    x_0 = rng.normal(2.0, 1.0, len(state))
    problem = InverseProblem(
        obs=pd.Series(H @ x_0 + rng.normal(0, 1, m), index=obs),
        prior=pd.Series(x_0, index=state),
        forward_operator=Operator(
            sp.csr_array(H), Space.from_index(obs), Space.from_index(state)
        ),
        prior_error=S_0,
        modeldata_mismatch=rng.uniform(0.2, 0.5),
    )
    return problem, state


def _aggregation(state, sparse=False, shuffle=False):
    """Sums by time and the mean of cell ``c0`` over time, labeled, ``4 x n``."""
    times = state.get_level_values("time")
    rows = [(times == t).astype(float) for t in times.unique()]
    first = (state.get_level_values("cell") == "c0").astype(float)
    rows.append(first / first.sum())
    labels = [str(t.date()) for t in times.unique()] + ["c0 mean"]
    index = pd.MultiIndex.from_arrays(
        [["region"] * len(labels), labels], names=["block", "agg"]
    )
    W = pd.DataFrame(np.array(rows), index=index, columns=state)
    if shuffle:
        W = W.iloc[:, np.random.default_rng(1).permutation(len(state))]
    Wa = W.reindex(columns=state).to_numpy()
    if sparse:
        W = W.astype(pd.SparseDtype(float, 0.0))
    return Operator.from_frame(W), Wa


@pytest.mark.parametrize("sparse", [False, True], ids=["dense W", "sparse W"])
@pytest.mark.parametrize("shuffle", [False, True], ids=["in order", "shuffled"])
def test_cov_and_averaging_kernel_of_an_aggregation(sparse, shuffle):
    """``cov(W)`` is ``W S_hat W^T`` and ``averaging_kernel(W)`` is ``W A W*``, labeled."""
    problem, state = _labeled()
    post = fips.solve(problem, gamma=0.8)
    W, Wa = _aggregation(state, sparse=sparse, shuffle=shuffle)
    assert isinstance(W.data, SparseOperator) == sparse
    S_hat = post.covariance.dense()
    A = post.averaging_kernel().dense()

    cov = post.cov(W)
    assert isinstance(cov, Covariance)
    assert cov.rows.index.equals(W.rows.index) and cov.cols.index.equals(W.rows.index)
    assert cov.name == "posterior_error"
    _close(cov.dense(), Wa @ S_hat @ Wa.T, "cov(W)")

    kernel = post.averaging_kernel(W)
    assert isinstance(kernel, Operator)
    assert kernel.rows.index.equals(W.rows.index)
    assert kernel.cols.index.equals(W.rows.index)
    _close(kernel.dense(), Wa @ A @ np.linalg.pinv(Wa), "averaging_kernel(W)")
    # For sums over disjoint sets, W* = W^T diag(1 / count): an even spread
    sums = post.averaging_kernel(Operator.from_frame(W.to_frame().iloc[:3]))
    counts = Wa[:3].sum(axis=1)
    _close(sums.dense(), Wa[:3] @ A @ (Wa[:3].T / counts), "sums")


def test_full_outputs_without_an_aggregation():
    """Without ``W``, ``cov`` is ``covariance`` and ``averaging_kernel`` is the same object each time."""
    problem, _ = _labeled()
    post = fips.solve(problem)
    assert post.cov() is post.covariance
    assert post.averaging_kernel() is post.averaging_kernel()
    problem.solve("bayesian")
    assert problem.averaging_kernel is problem.result.averaging_kernel()


def test_variances_without_the_covariance():
    """``variances`` is the diagonal of ``covariance``, read without forming it."""
    problem, state = _labeled()
    post = fips.solve(problem)
    variances = post.variances
    assert "S_hat" not in vars(post.estimator)
    assert variances.index.equals(state) and variances.name == "posterior_variance"
    _close(variances.values, np.diag(post.covariance.dense()), "variances", 1e-12)


def test_aggregation_labels_must_be_the_state():
    """An aggregation whose columns are not exactly the state index raises."""
    problem, state = _labeled()
    post = fips.solve(problem)
    W, _ = _aggregation(state)
    frame = W.to_frame()
    with pytest.raises(TypeError, match="An aggregation is an Operator"):
        post.cov(frame)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"must have exactly the state's labels.*block 'flux' missing from the other",
    ):
        post.cov(Operator.from_frame(frame.iloc[:, 1:]))
    extra = pd.DataFrame(
        1.0,
        index=frame.index,
        columns=pd.MultiIndex.from_tuples(
            [("flux", pd.Timestamp("2030-01-01"), "c0")], names=state.names
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"must have exactly the state's labels.*block 'flux' only the other space has",
    ):
        post.averaging_kernel(Operator.from_frame(pd.concat([frame, extra], axis=1)))
    # A repeated state label cannot label an Operator at all
    with pytest.raises(ValueError, match="Block 'flux' has duplicate labels"):
        Operator.from_frame(pd.concat([frame, frame.iloc[:, :1]], axis=1))


def test_dependent_aggregation_rows_raise():
    """An averaging kernel needs the pseudo-inverse, so dependent rows raise."""
    problem, state = _labeled()
    post = fips.solve(problem)
    W, _ = _aggregation(state)
    frame = W.to_frame()
    twice = pd.concat(
        [frame.iloc[:1], frame.iloc[:1].rename(index={"2024-01-01": "again"})]
    )
    with pytest.raises(ValueError, match="linearly dependent"):
        post.averaging_kernel(Operator.from_frame(twice))


def test_aggregation_across_blocks():
    """A multi-block state (NaN-padded levels) aggregates with an exact join."""
    rng = np.random.default_rng(4)
    flux = pd.MultiIndex.from_product(
        [["flux"], ["a", "b", "c"]], names=["block", "cell"]
    )
    bias = pd.MultiIndex.from_product([["bias"], ["s1", "s2"]], names=["block", "site"])
    obs = pd.MultiIndex.from_product([["conc"], range(6)], names=["block", "obs_id"])
    prior = fips.Vector(
        [
            fips.Block(
                pd.Series(rng.normal(size=3), index=flux.droplevel(0), name="flux")
            ),
            fips.Block(pd.Series([0.1, -0.1], index=bias.droplevel(0), name="bias")),
        ]
    )
    state = prior.index
    H = pd.DataFrame(rng.uniform(0, 1, (6, 5)), index=obs, columns=state)
    problem = InverseProblem(
        obs=pd.Series(rng.normal(size=6), index=obs),
        prior=prior,
        forward_operator=H,
        prior_error=1.0,
        modeldata_mismatch=0.5,
    )
    post = fips.solve(problem)
    rows = pd.MultiIndex.from_tuples(
        [("total", "flux"), ("total", "bias")], names=["block", "agg"]
    )
    Wa = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=float)
    order = np.array([4, 2, 0, 3, 1])
    W = Operator.from_frame(
        pd.DataFrame(Wa[:, order], index=rows, columns=state[order])
    )
    _close(
        post.cov(W).dense(),
        Wa @ post.covariance.dense() @ Wa.T,
        "cov across blocks",
    )


@pytest.mark.parametrize("gamma", [0.25, 2.0, 7.5])
def test_posterior_with_gamma(gamma):
    """``Posterior.with_gamma`` is the posterior of a fresh solve with that gamma."""
    problem, state = _labeled(seed=3)
    post = fips.solve(problem)
    swept = post.with_gamma(gamma)
    fresh = fips.solve(problem, gamma=gamma)
    assert "x_hat" in vars(swept.estimator)
    assert swept.estimator._whitened is post.estimator._whitened
    assert problem._estimator is None and post.estimator.gamma == 1.0
    W, _ = _aggregation(state)
    pairs = [
        (swept.estimate, fresh.estimate),
        (swept.variances, fresh.variances),
        (swept.covariance, fresh.covariance),
        (swept.cov(W), fresh.cov(W)),
        (swept.averaging_kernel(W), fresh.averaging_kernel(W)),
        (swept.kalman_gain, fresh.kalman_gain),
        (swept.modeled_obs, fresh.modeled_obs),
    ]
    for got, expected in pairs:
        if isinstance(expected, Vector):
            assert got.index.equals(expected.index)
            _close(got.values, expected.values, got.name)
        else:
            assert got.rows.index.equals(expected.rows.index)
            assert got.cols.index.equals(expected.cols.index)
            _close(got.dense(), expected.dense(), got.name)
    for name in ("dofs", "reduced_chi2", "r2", "rmse", "uncertainty_reduction"):
        _close(getattr(swept, name), getattr(fresh, name), name)


def test_with_gamma_needs_a_gamma():
    """An estimator without a regularization factor has no ``with_gamma``."""

    class NoGamma(Estimator):
        def cost(self, x):
            return 0.0

        @property
        def x_hat(self):
            return self.x_0

        @property
        def S_hat(self):
            return self.S_0

    problem, _ = _labeled()
    post = fips.solve(problem, NoGamma)
    with pytest.raises(TypeError, match="no regularization factor"):
        post.with_gamma(2.0)
    # The generic forms of the new outputs, from S_hat and A
    W, Wa = _aggregation(problem.state_index)
    _close(post.variances.values, np.diag(post.covariance.dense()), "variances")
    _close(post.cov(W).dense(), Wa @ post.covariance.dense() @ Wa.T, "cov(W)")
    A = post.averaging_kernel().dense()
    _close(post.averaging_kernel(W).dense(), Wa @ A @ np.linalg.pinv(Wa), "A(W)")


# ---------------------------------------------------------------------------
# Memory: nothing n x n
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prior", ["Kronecker", "diagonal"])
def test_nothing_n_by_n_is_allocated(prior):
    """
    Solving and reading the variances, DOFS, chi-squared and an aggregated covariance never allocate ``n x n``.

    ``n = 2,400`` and ``m = 30``: one ``n x n`` array is 46 MB, while what the
    solve needs is ``O(m n)`` (0.6 MB for ``W``) and the Kronecker products work
    on ``n x m`` blocks. The traced peak must stay under half of one ``n x n``
    array, so any such array, or a copy of the prior error, fails it. The rest
    is label work, ``O(n)``: the peak was 3 MB on pandas 3 and 14 MB on pandas
    2.0.3.
    """
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=48, freq="D")
    lats, lons = np.arange(10) * 0.1, np.arange(5) * 0.1
    state = pd.MultiIndex.from_product(
        [["flux"], times, lats, lons], names=["block", "time", "lat", "lon"]
    )
    n, m = len(state), 30
    obs = pd.MultiIndex.from_product([["conc"], range(m)], names=["block", "obs_id"])
    H = sp.random(m, n, density=0.05, random_state=1, format="csr")
    if prior == "Kronecker":
        model = Kronecker(
            Exponential("time", "5D"),
            Exponential(["lat", "lon"], 20.0, metric="haversine"),
        )
        S_0 = model.with_std(np.sqrt(0.5)).build(Space.from_index(state)).data
    else:
        S_0 = DiagonalOperator(rng.uniform(0.5, 1.0, n))
    problem = InverseProblem(
        obs=pd.Series(rng.normal(size=m), index=obs),
        prior=pd.Series(rng.normal(size=n), index=state),
        forward_operator=Operator(H, Space.from_index(obs), Space.from_index(state)),
        prior_error=S_0,
        modeldata_mismatch=0.1,
        round_index=None,
    )
    W = Operator(
        sp.csr_array(np.ones((1, n))),
        Space.from_index(
            pd.MultiIndex.from_tuples([("t", "all")]).set_names(["block", "agg"])
        ),
        Space.from_index(state),
    )
    n_by_n = n * n * 8
    tracemalloc.start()
    try:
        post = fips.solve(problem)
        _ = post.variances, post.dofs, post.reduced_chi2, post.uncertainty_reduction
        _ = post.cov(W), post.with_gamma(2.0).variances
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < n_by_n / 2, (
        f"peak {peak / 1e6:.1f} MB; one n x n is {n_by_n / 1e6:.1f} MB"
    )
    big = [k for k, v in vars(post.estimator).items() if getattr(v, "size", 0) >= n * n]
    assert not big, big


# ---------------------------------------------------------------------------
# Pickle, refused inputs, logged clips
# ---------------------------------------------------------------------------


def test_pickle_keeps_the_factors_and_drops_n_by_n():
    """A posterior pickles its factors, not ``S_hat``, ``A`` or ``K``, and sweeps gamma after loading."""
    problem, _ = _labeled()
    post = fips.solve(problem)
    _ = post.covariance, post.averaging_kernel(), post.kalman_gain, post.variances
    state = post.estimator.__getstate__()
    assert not {"S_hat", "A", "K", "_HS_0"} & set(state)
    assert isinstance(state["_whitened"], dict)
    loaded = pickle.loads(pickle.dumps(post))
    assert not {"S_hat", "A", "K"} & set(vars(loaded.estimator))
    np.testing.assert_array_equal(loaded.variances.values, post.variances.values)
    np.testing.assert_array_equal(loaded.covariance.dense(), post.covariance.dense())
    _close(
        loaded.with_gamma(3.0).estimate.values,
        fips.solve(problem, gamma=3.0).estimate.values,
        "with_gamma after a pickle",
    )


def test_singular_mismatch_raises():
    """``S_z`` whitens the observations, so it must be positive-definite."""
    z, x_0, H, S_0, _, c = _arrays("dense", 0)
    m = len(z)
    with pytest.raises(LinAlgError, match="not positive-definite"):
        _ = BayesianSolver(z, x_0, H, S_0, np.ones((m, m)), c=c).x_hat
    zero = sp.diags(np.r_[0.0, np.ones(m - 1)]).tocsr()
    with pytest.raises(LinAlgError, match="not positive"):
        _ = BayesianSolver(z, x_0, H, S_0, zero, c=c).x_hat


def test_negative_eigenvalues_are_clipped_and_logged(caplog):
    """A prior error that is not positive semidefinite gives eigenvalues below zero: clipped, logged."""
    z, x_0, H, _, S_z, c = _arrays("dense", 0, n_time=1, n_cell=3, m=3)
    S_0 = np.diag([1.0, 1.0, -0.5])
    with caplog.at_level(logging.WARNING, logger="fips._whitened"):
        solver = BayesianSolver(z, x_0, np.eye(3), S_0, S_z, c=c)
        _ = solver.x_hat
    assert any("negative eigenvalue" in r.message for r in caplog.records)
    assert (solver._whitened.eigenvalues >= 0).all()


def test_roundoff_is_clipped_quietly(caplog):
    """Eigenvalues negative only by roundoff (a near-singular prior) are clipped without a warning."""
    rng = np.random.default_rng(0)
    t = np.arange(10)
    T = np.exp(-((t[:, None] - t[None, :]) ** 2) / 200.0)  # near rank-deficient
    S_0 = np.kron(T, np.eye(3))
    H = rng.uniform(0, 1, (40, 30))
    with caplog.at_level(logging.WARNING, logger="fips._whitened"):
        solver = BayesianSolver(rng.normal(size=40), np.zeros(30), H, S_0, np.eye(40))
        _ = solver.x_hat
    assert not caplog.records
    assert np.isfinite(solver.S_hat_diagonal).all()


def test_more_observations_than_state_logs(caplog, monkeypatch):
    """A large solve with ``m > n`` says that the state-space form would be cheaper."""
    monkeypatch.setattr("fips._whitened._LARGE_M", 5)
    z, x_0, H, S_0, S_z, c = _arrays("dense", 0, n_time=1, n_cell=3, m=8)
    with caplog.at_level(logging.WARNING, logger="fips._whitened"):
        _ = BayesianSolver(z, x_0, H, S_0, S_z, c=c).x_hat
    assert any("state-space (information) form" in r.message for r in caplog.records)
    caplog.clear()
    z, x_0, H, S_0, S_z, c = _arrays("dense", 0, n_time=2, n_cell=4, m=8)
    with caplog.at_level(logging.WARNING, logger="fips._whitened"):
        _ = BayesianSolver(z, x_0, H, S_0, S_z, c=c).x_hat
    assert not caplog.records
