"""
Tests for the analytic method's factorization, in its two forms: what a ``Posterior`` keeps.

The oracle (``test_oracle.py``) holds the estimate, the full posterior
covariance, the gain, the averaging kernel, DOFS and the reduced chi-squared to
the closed form. Here: what the factorization adds (the variances, aggregated
covariances and averaging kernels, ``with_gamma``, ``with_data``), the direct
solve the factorization replaced, the inputs it refuses, its pickle, and that
nothing ``n x n`` is allocated unless the full covariance or averaging kernel
is read. Most tests run in both forms (the ``analytic_form`` fixture forces
one); the last section tests how the method chooses between the whitened form
(``Whitened``, ``m x m``) and the information form (``Information``,
``n x n``), and what each holds.
"""

import copy
import dataclasses
import logging
import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.linalg import eigh as scipy_eigh

import fips
from fips import (
    Covariance,
    Exponential,
    Kronecker,
    Operator,
    Problem,
    Space,
    Vector,
)
from fips.linalg import (
    BlockDiagonalOperator,
    DiagonalOperator,
    KroneckerOperator,
    SparseOperator,
    as_linear_operator,
)
from fips.methods.analytic import AnalyticOptions, Information, Whitened
from fips.posterior import Posterior

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
    chi2 = r @ inv(S_z) @ r + dx @ inv(S_0) @ dx
    return {
        "estimate": x_hat,
        "cov": S_hat,
        "variances": np.diag(S_hat),
        "kalman_gain": K,
        "averaging_kernel": K @ H,
        "dofs": np.trace(K @ H),
        "chi2": chi2,
        "reduced_chi2": chi2 / len(z),
        "uncertainty_reduction": 1 - np.sqrt(np.trace(S_hat)) / np.sqrt(np.trace(S_0)),
    }


def _direct(z, x_0, H, S_0, S_z, c, gamma):
    """
    The direct solve the factorization replaced.

    The gain from ``(H S_0 H^T + S_z / gamma)`` by its form's solve, ``S_hat =
    S_0 - K H S_0``, and each misfit of the chi-squared by a solve with its
    covariance.
    """
    H, S_0 = as_linear_operator(H), as_linear_operator(S_0)
    S_z = as_linear_operator(S_z) / gamma
    HS_0 = H @ S_0
    HS_0H = HS_0 @ H._transpose()
    K = (HS_0H + S_z).solve(HS_0).T
    x_hat = x_0 + K @ (z - (H @ x_0 + c))
    S_hat = S_0.dense() - K @ HS_0
    y_hat, y_0 = H @ x_hat + c, H @ x_0 + c
    r, dx = z - y_hat, x_hat - x_0
    chi2 = float(r @ S_z.solve(r) + dx @ S_0.solve(dx))
    return {
        "estimate": x_hat,
        "cov": S_hat,
        "variances": np.diag(S_hat),
        "kalman_gain": K,
        "averaging_kernel": K @ H,
        "dofs": np.trace(K @ H),
        "chi2": chi2,
        "reduced_chi2": chi2 / len(z),
        "uncertainty_reduction": 1 - np.sqrt(np.trace(S_hat)) / np.sqrt(S_0.trace()),
        "modeled_obs": y_hat,
        "predict(prior)": y_0,
        "innovation_cov": (HS_0H + S_z).dense(),
        "r2": np.corrcoef(z - c, H @ x_hat)[0, 1] ** 2,
        "rmse": np.sqrt((r**2).sum() / len(z)),
        "desroziers": r * (z - y_0),
    }


def _problem(z, x_0, H, S_0, S_z, c, *, check=True) -> Problem:
    """
    Label the arrays: one block each, numbered; each matrix keeps its form.

    ``check=False`` builds the prior error covariance as fips builds one it
    computes, without the checks of ``Covariance(...)``: a sum or difference
    of covariances can be indefinite, and the method's own guard is tested
    with one.
    """
    state = Space({"flux": pd.Index(range(len(x_0)), name="cell")})
    obs = Space({"conc": pd.Index(range(len(z)), name="obs_id")})
    return Problem(
        prior=Vector(x_0, state),
        prior_cov=(Covariance if check else Covariance._computed)(S_0, state),
        obs=Vector(z, obs),
        obs_cov=Covariance(S_z, obs),
        forward=Operator(H, obs, state),
        offset=Vector(c, obs),
    )


def _outputs(post) -> dict:
    """Every output of a posterior, unlabeled, by the names of `_direct`."""
    return {
        "estimate": post.estimate.values,
        "cov": post.cov().dense(),
        "variances": post.variances.values,
        "kalman_gain": post.kalman_gain.dense(),
        "averaging_kernel": post.averaging_kernel().dense(),
        "dofs": post.dofs,
        "chi2": post.chi2,
        "reduced_chi2": post.reduced_chi2,
        "uncertainty_reduction": post.uncertainty_reduction,
        "modeled_obs": post.modeled_obs.values,
        "predict(prior)": post.problem.predict(post.problem.prior).values,
        "innovation_cov": post.innovation_cov.dense(),
        "r2": post.r2,
        "rmse": post.rmse,
        "desroziers": post.desroziers()["diagnosed"].to_numpy(),
    }


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("seed", range(5))
def test_outputs_match_the_closed_form(form, gamma, seed, analytic_form):
    """Every output the factors give, at 1e-10 of the closed form."""
    z, x_0, H, S_0, S_z, c = _arrays(form, seed)
    outputs = _outputs(fips.solve(_problem(z, x_0, H, S_0, S_z, c), gamma=gamma))
    for name, expected in _closed_form(z, x_0, H, S_0, S_z, c, gamma).items():
        _close(outputs[name], expected, f"{name}, {form}, gamma {gamma}")


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("gamma", [0.5, 1.0, 4.0])
@pytest.mark.parametrize("shape", [(3, 4, 7), (2, 2, 9), (4, 5, 2)], ids=str)
def test_outputs_match_the_direct_solve(form, gamma, shape, analytic_form):
    """Every output agrees with the direct solve (the code before the factorization) at 1e-10."""
    args = _arrays(form, 3, *shape)
    outputs = _outputs(fips.solve(_problem(*args), gamma=gamma))
    for name, expected in _direct(*args, gamma).items():
        _close(outputs[name], expected, f"{name}, {form}")


@pytest.mark.parametrize("form", FORMS)
def test_with_gamma_is_a_fresh_solve(form, analytic_form):
    """``with_gamma`` gives what a new solve with that gamma gives, every sweep from one eigendecomposition."""
    problem = _problem(*_arrays(form, 7))
    post = fips.solve(problem)
    assert post._solution.eigenvalues is None  # a Cholesky at gamma 1
    eigenvalues = post.with_gamma(1.0)._solution.eigenvalues
    assert eigenvalues is not None
    for gamma in GAMMAS:
        swept = post.with_gamma(gamma)
        assert swept._solution.eigenvalues is eigenvalues
        assert swept.with_gamma(2.0)._solution.eigenvalues is eigenvalues
        assert swept.options.gamma == gamma
        assert "estimate" in vars(swept)  # computed, as fips.solve does
        fresh = _outputs(fips.solve(problem, gamma=gamma))
        for name, value in _outputs(swept).items():
            _close(value, fresh[name], f"{name}, gamma {gamma}")
    assert post.options.gamma == 1.0


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("gamma", [0.5, 2.0])
def test_with_data_is_a_fresh_solve(form, gamma, analytic_form):
    """``with_data`` gives what a solve of the problem with those observations gives."""
    z, x_0, H, S_0, S_z, c = _arrays(form, 11)
    post = fips.solve(_problem(z, x_0, H, S_0, S_z, c), gamma=gamma)
    rng = np.random.default_rng(12)
    z2 = z + rng.normal(0.0, 1.0, len(z))
    # Given in another order: the labels decide
    order = rng.permutation(len(z))
    new = Vector(
        z2[order],
        Space({"conc": pd.Index(order, name="obs_id")}),
        name="bootstrap",
    )
    swept = post.with_data(new)
    held = "whitener" if analytic_form is Whitened else "G"
    assert getattr(swept._solution, held) is getattr(post._solution, held)
    assert swept.problem.obs.space == post.problem.obs.space
    np.testing.assert_array_equal(swept.problem.obs.values, z2)
    fresh = _outputs(fips.solve(_problem(z2, x_0, H, S_0, S_z, c), gamma=gamma))
    for name, value in _outputs(swept).items():
        _close(value, fresh[name], f"{name}, with_data")
    np.testing.assert_array_equal(post.problem.obs.values, z)


@pytest.mark.parametrize("form", FORMS)
def test_the_factors_are_read_only(form, analytic_form):
    """No array of the factors can be written to, however the solution was made: a solve, a sweep, new data, a pickle, a copy."""
    post = fips.solve(_problem(*_arrays(form, 6)))
    swept = post.with_gamma(2.0)
    solutions = {
        "factor": post._solution,
        "with_gamma": swept._solution,
        "with_data": post.with_data(post.problem.obs)._solution,
        "with_data after with_gamma": swept.with_data(post.problem.obs)._solution,
        "pickle": pickle.loads(pickle.dumps(post))._solution,
        "pickle after with_gamma": pickle.loads(pickle.dumps(swept))._solution,
        "deepcopy": copy.deepcopy(post._solution),
    }
    for how, solution in solutions.items():
        assert isinstance(solution, analytic_form)
        arrays = [
            f.name
            for f in dataclasses.fields(solution)
            if isinstance(getattr(solution, f.name), np.ndarray)
        ]
        eigendecomposed = solution.eigenvalues is not None
        assert eigendecomposed == ("with_gamma" in how), how
        assert len(arrays) == (4 if analytic_form is Whitened else 2) + eigendecomposed
        for name in arrays:
            array = getattr(solution, name)
            assert not array.flags.writeable, f"{name} after {how}"
            with pytest.raises(ValueError, match="read-only"):
                array.fill(0.0)


class _Counted(Problem):
    """A problem that counts how many times its constructor ran."""

    built = 0

    def __init__(self, *args, **kwargs):
        type(self).built += 1
        super().__init__(*args, **kwargs)


def test_with_data_builds_its_problem_with_the_constructor():
    """The new problem is made by the problem's own class, from the same inputs (their forms are shared)."""
    p = _problem(*_arrays("dense", 4))
    problem = _Counted(
        prior=p.prior,
        prior_cov=p.prior_cov,
        obs=p.obs,
        obs_cov=p.obs_cov,
        forward=p.forward,
        offset=p.offset,
    )
    post = fips.solve(problem)
    before = _Counted.built
    swept = post.with_data(Vector(problem.obs.values + 1.0, problem.obs.space))
    assert _Counted.built == before + 1
    new = swept.problem
    assert type(new) is _Counted and new is not problem
    assert new.prior is problem.prior and new.obs.space is problem.obs.space
    for name in ("prior_cov", "obs_cov", "forward"):
        assert getattr(new, name).data is getattr(problem, name).data, name
    np.testing.assert_array_equal(new.offset.values, problem.offset.values)
    np.testing.assert_array_equal(new.obs.values, problem.obs.values + 1.0)


def test_with_data_needs_the_same_labels():
    """New observations on other labels raise, naming them."""
    z, x_0, H, S_0, S_z, c = _arrays("dense", 0)
    post = fips.solve(_problem(z, x_0, H, S_0, S_z, c))
    other = Vector(z, Space({"conc": pd.Index(range(1, len(z) + 1), name="obs_id")}))
    with pytest.raises(
        ValueError,
        match=r"The new observations and the problem's do not have the same labels:"
        r".*missing from the new observations: 0;.*only in the new observations: 7",
    ):
        post.with_data(other)


def test_the_factors_hold_no_m_by_m_matrix_beyond_the_whitener():
    """H S_0 H^T is formed when ``innovation_cov`` or ``leverage`` asks, not held after a solve (and not pickled)."""
    z, x_0, H, S_0, S_z, c = _arrays("dense", 5)
    post = fips.solve(_problem(z, x_0, H, S_0, S_z, c))
    m = len(z)
    assert "_HS_0H" not in vars(post)
    assert not hasattr(post._solution, "HS_0H")
    assert not [
        k
        for k, v in vars(post._solution).items()
        if getattr(v, "shape", ()) == (m, m) and k != "whitener"
    ]
    assert "_HS_0H" not in vars(pickle.loads(pickle.dumps(post)))
    _ = post.innovation_cov
    assert "_HS_0H" in vars(post)


class _Asked(Problem):
    """A problem that counts how often it is asked for a prediction."""

    asked = 0

    def predict(self, x):
        self.asked += 1
        return super().predict(x)


def test_the_innovation_is_read_through_predict(analytic_form):
    """The innovation each form keeps is built from ``problem.predict(prior)``, with its offset, not around it."""
    z, x_0, H, S_0, S_z, c = _arrays("dense", 2)
    p = _problem(z, x_0, H, S_0, S_z, c)
    problem = _Asked(
        prior=p.prior,
        prior_cov=p.prior_cov,
        obs=p.obs,
        obs_cov=p.obs_cov,
        forward=p.forward,
        offset=p.offset,
    )
    post = fips.solve(problem)
    assert problem.asked == 1
    solution, d = post._solution, z - problem.predict(problem.prior).values
    if analytic_form is Whitened:  # e = B d
        B = solution.whitener
        np.testing.assert_array_equal(solution.innovation, B @ d)
        assert not np.array_equal(solution.innovation, B @ (z - H @ x_0))
    else:  # b = G^T H^T S_z^-1 d
        G = solution.G
        _close(solution.innovation, G.T @ (H.T @ np.linalg.solve(S_z, d)), "b")
        without_c = G.T @ (H.T @ np.linalg.solve(S_z, z - H @ x_0))
        assert not np.allclose(solution.innovation, without_c)


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
    state_space, obs_space = Space.from_index(state), Space.from_index(obs)
    problem = Problem(
        obs=Vector(H @ x_0 + rng.normal(0, 1, m), obs_space),
        prior=Vector(x_0, state_space),
        forward=Operator(sp.csr_array(H), obs_space, state_space),
        prior_cov=Covariance(S_0, state_space),
        obs_cov=Covariance.diagonal(rng.uniform(0.2, 0.5), obs_space),
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
def test_cov_and_averaging_kernel_of_an_aggregation(sparse, shuffle, analytic_form):
    """``cov(W)`` is ``W S_hat W^T`` and ``averaging_kernel(W)`` is ``W A W*``, labeled."""
    problem, state = _labeled()
    post = fips.solve(problem, gamma=0.8)
    W, Wa = _aggregation(state, sparse=sparse, shuffle=shuffle)
    assert isinstance(W.data, SparseOperator) == sparse
    S_hat = post.cov().dense()
    A = post.averaging_kernel().dense()

    cov = post.cov(W)
    assert isinstance(cov, Covariance)
    assert cov.rows.index.equals(W.rows.index) and cov.cols.index.equals(W.rows.index)
    assert cov.name == "posterior_cov"
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
    """Without ``W``, ``cov`` and ``averaging_kernel`` are formed once: the same object each time."""
    problem, _ = _labeled()
    post = fips.solve(problem)
    assert post.cov() is post.cov()
    assert post.averaging_kernel() is post.averaging_kernel()


def test_variances_without_the_covariance(analytic_form):
    """``variances`` is the diagonal of ``covariance``, read without forming it."""
    problem, state = _labeled()
    post = fips.solve(problem)
    variances = post.variances
    assert "_cov" not in vars(post)
    assert variances.index.equals(state) and variances.name == "variance"
    _close(variances.values, np.diag(post.cov().dense()), "variances", 1e-12)


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
        match=r"The columns of the aggregation and the state do not have the same "
        r"labels: labels of block 'flux' missing from the columns of the aggregation",
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
        match=r"labels of block 'flux' only in the columns of the aggregation",
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


def test_aggregation_across_blocks(analytic_form):
    """A multi-block state (NaN-padded levels) aggregates with an exact join."""
    rng = np.random.default_rng(4)
    flux = pd.MultiIndex.from_product(
        [["flux"], ["a", "b", "c"]], names=["block", "cell"]
    )
    bias = pd.MultiIndex.from_product([["bias"], ["s1", "s2"]], names=["block", "site"])
    obs = pd.MultiIndex.from_product([["conc"], range(6)], names=["block", "obs_id"])
    prior = Vector.from_blocks(
        {
            "flux": pd.Series(rng.normal(size=3), index=flux.droplevel(0)),
            "bias": pd.Series([0.1, -0.1], index=bias.droplevel(0)),
        }
    )
    state = prior.index
    H = pd.DataFrame(rng.uniform(0, 1, (6, 5)), index=obs, columns=state)
    problem = Problem(
        obs=Vector.from_series(pd.Series(rng.normal(size=6), index=obs)),
        prior=prior,
        forward=Operator.from_frame(H),
        prior_cov=Covariance.diagonal(1.0, prior.space),
        obs_cov=Covariance.diagonal(0.5, Space.from_index(obs)),
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
        Wa @ post.cov().dense() @ Wa.T,
        "cov across blocks",
    )


@pytest.mark.parametrize("gamma", [0.25, 2.0, 7.5])
def test_posterior_with_gamma(gamma, analytic_form):
    """``Posterior.with_gamma`` is the posterior of a fresh solve with that gamma."""
    problem, state = _labeled(seed=3)
    post = fips.solve(problem)
    swept = post.with_gamma(gamma)
    fresh = fips.solve(problem, gamma=gamma)
    assert "estimate" in vars(swept)
    assert swept._solution.eigenvalues is post.with_gamma(1.0)._solution.eigenvalues
    assert post.options.gamma == 1.0
    W, _ = _aggregation(state)
    pairs = [
        (swept.estimate, fresh.estimate),
        (swept.variances, fresh.variances),
        (swept.cov(), fresh.cov()),
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
    state_space, obs_space = Space.from_index(state), Space.from_index(obs)
    if prior == "Kronecker":  # a model, which the problem builds on the state
        S_0 = Kronecker(
            Exponential("time", "5D"),
            Exponential(["lat", "lon"], 20.0, metric="haversine"),
        ).with_std(np.sqrt(0.5))
    else:
        S_0 = Covariance(DiagonalOperator(rng.uniform(0.5, 1.0, n)), state_space)
    problem = Problem(
        obs=Vector(rng.normal(size=m), obs_space),
        prior=Vector(rng.normal(size=n), state_space),
        forward=Operator(H, obs_space, state_space),
        prior_cov=S_0,
        obs_cov=Covariance.diagonal(0.1, obs_space),
    )
    if prior == "Kronecker":
        assert isinstance(problem.prior_cov.data, KroneckerOperator)
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
    held = {**vars(post), **vars(post._solution)}
    big = [k for k, v in held.items() if getattr(v, "size", 0) >= n * n]
    assert not big, big


# ---------------------------------------------------------------------------
# Pickle, refused inputs, logged clips
# ---------------------------------------------------------------------------


def test_pickle_keeps_the_factors_and_drops_n_by_n(analytic_form):
    """A posterior pickles its problem, options and factors, not ``S_hat``, ``A`` or ``K``, and sweeps gamma after loading."""
    problem, _ = _labeled()
    post = fips.solve(problem)
    _ = post.cov(), post.averaging_kernel(), post.kalman_gain, post.variances
    state = post.__getstate__()
    assert set(state) == {"problem", "options", "solution"}
    assert isinstance(state["solution"], analytic_form)
    loaded = pickle.loads(pickle.dumps(post))
    assert not {"_cov", "_averaging_kernel", "kalman_gain"} & set(vars(loaded))
    np.testing.assert_array_equal(loaded.variances.values, post.variances.values)
    np.testing.assert_array_equal(loaded.cov().dense(), post.cov().dense())
    assert loaded.problem.prior.space == problem.prior.space
    assert isinstance(loaded.problem.prior_cov.data, KroneckerOperator)
    assert loaded._solution.problem is loaded.problem  # stored once
    _close(
        loaded.with_gamma(3.0).estimate.values,
        fips.solve(problem, gamma=3.0).estimate.values,
        "with_gamma after a pickle",
    )


def test_singular_mismatch_raises(analytic_form):
    """``S_z`` whitens the observations, so it must be positive-definite."""
    z, x_0, H, S_0, _, c = _arrays("dense", 0)
    m = len(z)
    with pytest.raises(LinAlgError, match="not positive-definite"):
        fips.solve(_problem(z, x_0, H, S_0, np.ones((m, m)), c))
    zero = sp.diags(np.r_[0.0, np.ones(m - 1)]).tocsr()
    with pytest.raises(LinAlgError, match="not positive"):
        fips.solve(_problem(z, x_0, H, S_0, zero, c))


def test_negative_eigenvalues_are_clipped_and_logged(caplog):
    """A prior error that is not positive semidefinite gives eigenvalues below zero: clipped, logged."""
    z, x_0, H, _, S_z, c = _arrays("dense", 0, n_time=1, n_cell=3, m=3)
    S_0 = np.diag([1.0, 1.0, -0.5])
    with caplog.at_level(logging.WARNING, logger="fips.methods.analytic"):
        post = fips.solve(_problem(z, x_0, np.eye(3), S_0, S_z, c, check=False))
    assert any("negative eigenvalue" in r.message for r in caplog.records)
    assert post._solution.eigenvalues is not None  # eigendecomposed, not a Cholesky
    assert (post._solution.eigenvalues >= 0).all()


def test_roundoff_is_clipped_quietly(caplog):
    """Eigenvalues negative only by roundoff (a near-singular prior) are clipped without a warning."""
    rng = np.random.default_rng(0)
    t = np.arange(10)
    T = np.exp(-((t[:, None] - t[None, :]) ** 2) / 200.0)  # near rank-deficient
    S_0 = np.kron(T, np.eye(3))
    H = rng.uniform(0, 1, (40, 30))
    with caplog.at_level(logging.WARNING, logger="fips.methods.analytic"):
        post = fips.solve(
            _problem(
                rng.normal(size=40), np.zeros(30), H, S_0, np.eye(40), np.zeros(40)
            )
        )
    assert not caplog.records
    assert np.isfinite(post.variances.values).all()


def _spy_eigh(monkeypatch, *, force=None, no_memory_for=None):
    """
    Replace the method's ``eigh`` with one that records the drivers it is called with.

    `force` makes every call use that driver. A call whose driver is
    `no_memory_for` raises ``MemoryError``, as SciPy does when LAPACK's
    workspace cannot be allocated, before it touches the matrix.
    """
    from fips.methods import analytic

    drivers = []

    def eigh(a, **kwargs):
        drivers.append(kwargs["driver"])
        if force is not None:
            kwargs["driver"] = force
        if kwargs["driver"] == no_memory_for:
            raise MemoryError
        return scipy_eigh(a, **kwargs)

    monkeypatch.setattr(analytic, "eigh", eigh)
    return drivers


def _logged(caplog):
    """The records the analytic method logged (``fips.solve`` logs its own start)."""
    return [r for r in caplog.records if r.name == "fips.methods.analytic"]


def test_only_the_first_with_gamma_eigendecomposes(analytic_form, monkeypatch, caplog):
    """A solve and every output at its gamma never eigendecompose; the first ``with_gamma`` does, once, with ``evd``, and logs nothing about memory."""
    drivers = _spy_eigh(monkeypatch)
    with caplog.at_level(logging.INFO, logger="fips.methods.analytic"):
        post = fips.solve(_problem(*_arrays("dense", 0)))
        _outputs(post)
        _outputs(post.with_data(post.problem.obs))
        assert drivers == []
        swept = [post.with_gamma(g) for g in GAMMAS]
        assert drivers == ["evd"]
        _ = [_outputs(s.with_gamma(3.0)) for s in swept]
        _outputs(swept[0].with_data(post.problem.obs).with_gamma(0.2))
    assert drivers == ["evd"]
    assert not _logged(caplog)


def test_no_memory_for_evd_redoes_the_eigendecomposition_with_evr(
    analytic_form, monkeypatch, caplog
):
    """A ``MemoryError`` from ``evd`` is met with ``evr`` on the same matrix: the same result, and one INFO line."""
    args = _arrays("dense", 0)
    _spy_eigh(monkeypatch, force="evr")
    expected = _outputs(fips.solve(_problem(*args)).with_gamma(2.0))
    drivers = _spy_eigh(monkeypatch, no_memory_for="evd")
    with caplog.at_level(logging.INFO, logger="fips.methods.analytic"):
        post = fips.solve(_problem(*args)).with_gamma(2.0)
    assert drivers == ["evd", "evr"]
    (record,) = _logged(caplog)
    assert record.levelno == logging.INFO
    assert "'evd'" in record.message and "'evr'" in record.message
    for name, value in _outputs(post).items():
        np.testing.assert_array_equal(value, expected[name], err_msg=name)


# ---------------------------------------------------------------------------
# One gamma by Cholesky, any gamma by the eigendecomposition
# ---------------------------------------------------------------------------

MISMATCHES = ["diagonal", "sparse", "block diagonal", "dense", "Kronecker"]


def _both(problem, form, gamma):
    """The posteriors of `problem` at `gamma` in `form`: from the Cholesky a solve does, and from the eigendecomposition."""
    options = AnalyticOptions(gamma=gamma)
    cholesky = Posterior(problem, form.factor(problem, gamma), options)
    eigen = form.factor(problem, gamma, eigendecompose=True)
    assert cholesky._solution.eigenvalues is None
    assert eigen.eigenvalues is not None
    return cholesky, Posterior(problem, eigen, options)


def _every_output(post) -> dict:
    """:func:`_outputs`, and the covariance and averaging kernel of two aggregates."""
    state = post.problem.prior.space
    n = len(post.problem.prior)
    rows = Space({"total": pd.Index(["a", "b"], name="total")})
    W = Operator(np.random.default_rng(0).uniform(0, 1, (2, n)), rows, state)
    return {
        **_outputs(post),
        "cov(W)": post.cov(W).dense(),
        "averaging_kernel(W)": post.averaging_kernel(W).dense(),
    }


@pytest.mark.parametrize(
    "form", [Whitened, Information], ids=["whitened", "information"]
)
@pytest.mark.parametrize("prior", FORMS)
@pytest.mark.parametrize("mismatch", MISMATCHES)
@pytest.mark.parametrize("shape", [(3, 4, 8), (1, 3, 8)], ids=["n > m", "n < m"])
@pytest.mark.parametrize("gamma", [0.05, 1.0, 30.0])
@pytest.mark.parametrize("seed", range(2))
def test_the_cholesky_gives_what_the_eigendecomposition_gives(
    form, prior, mismatch, shape, gamma, seed
):
    """Every output of a one-gamma solve, at 1e-10 of the eigendecomposition's at that gamma and of the closed form."""
    z, x_0, H, S_0, _, c = _arrays(prior, seed, *shape)
    S_z = _mismatch(mismatch, shape[2], np.random.default_rng(seed + 10))
    problem = _problem(z, x_0, H, S_0, S_z, c)
    cholesky, eigen = _both(problem, form, gamma)
    got, expected = _every_output(cholesky), _every_output(eigen)
    for name, value in got.items():
        _close(value, expected[name], f"{name} against the eigendecomposition")
    for name, value in _closed_form(z, x_0, H, S_0, _dense(S_z), c, gamma).items():
        _close(got[name], value, f"{name} against the closed form")


@pytest.mark.parametrize(
    "form, n, m",
    [(Whitened, 300, 200), (Information, 60, 200)],
    ids=["whitened", "information"],
)
@pytest.mark.parametrize("gamma", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("weak", [False, True], ids=["strong", "weak"])
def test_the_cholesky_matches_at_size(form, n, m, gamma, weak):
    """
    At a few hundred observations, strongly or weakly observed, the Cholesky's outputs are the eigendecomposition's at 1e-10.

    Weakly observed (a mismatch a thousand times the prior's projection), DOFS
    is much smaller than ``k``, where ``k - ||R^-1||^2 / gamma`` would lose
    digits; ``tr(R^-1 A R^-T)`` does not.
    """
    rng = np.random.default_rng(int(gamma * 100) + weak)
    H = sp.random(m, n, density=0.3, random_state=rng, format="csr")
    times = np.arange(n)
    S_0 = np.exp(-np.abs(times[:, None] - times[None, :]) / 20.0)
    scale = 1e3 if weak else 1e-2
    S_z = sp.diags(rng.uniform(0.5, 2.0, m) * scale * n * 0.3).tocsr()
    x_0 = rng.normal(size=n)
    z = np.asarray(H @ x_0) + rng.normal(size=m)
    problem = _problem(z, x_0, H, S_0, S_z, np.zeros(m))
    cholesky, eigen = _both(problem, form, gamma)
    got, expected = _every_output(cholesky), _every_output(eigen)
    for name, value in got.items():
        # Desroziers multiplies the residual by the innovation. Strongly observed
        # at gamma 100 the residual is 5 % of the innovation, and each path's is
        # 1e-10 of its size from an extended-precision solve (measured): the two
        # agree to 1e-9 there (macOS's BLAS showed 2e-9 at one observation)
        _close(value, expected[name], name, rtol=1e-9 if name == "desroziers" else RTOL)
    if weak:
        assert got["dofs"] < 0.05 * n


@pytest.mark.parametrize("smallest", [-0.5, -1e-4, -1e-9, 0.0, 1e-9])
def test_a_prior_that_is_not_positive_semidefinite_warns_as_before(smallest, caplog):
    """
    A one-gamma solve warns about a negative eigenvalue for the same prior errors as the eigendecomposition, and gives its posterior.

    The prior error covariance has a positive diagonal and one eigenvalue
    `smallest` (relative to the largest), so nothing but its eigenvalues shows
    it. A Cholesky at gamma cannot see an eigenvalue above ``-1/gamma``: the
    solve's check sends such a prior to the eigendecomposition, which clips
    and warns.
    """
    rng = np.random.default_rng(3)
    k = 6
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    S_0 = Q @ np.diag([1.0, 0.8, 0.5, 0.3, 0.1, smallest]) @ Q.T
    S_0 = 0.5 * (S_0 + S_0.T)
    assert (np.diag(S_0) > 0).all()
    H = rng.standard_normal((k, k))
    S_z = sp.diags(rng.uniform(0.5, 1.5, k)).tocsr()
    z, x_0 = rng.normal(size=k), np.zeros(k)
    problem = _problem(z, x_0, H, S_0, S_z, np.zeros(k), check=False)

    def warned(factor):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="fips.methods.analytic"):
            result = factor()
        return result, any("negative eigenvalue" in r.message for r in caplog.records)

    post, solve_warned = warned(lambda: fips.solve(problem))
    eigen, eigen_warned = warned(
        lambda: Whitened.factor(problem, 1.0, eigendecompose=True)
    )
    assert type(post._solution) is Whitened  # S_0 has no square root, or n = m
    assert solve_warned == eigen_warned == (smallest < 0)
    assert (post._solution.eigenvalues is not None) == (smallest < 0)
    expected = Posterior(problem, eigen, post.options)

    def core(p):
        """Not the outputs that take a square root: the variances can sum below zero."""
        return {
            "estimate": p.estimate.values,
            "variances": p.variances.values,
            "kalman_gain": p.kalman_gain.dense(),
            "dofs": p.dofs,
            "chi2": p.chi2,
        }

    want = core(expected)
    for name, value in core(post).items():
        _close(value, want[name], name)


def test_a_solution_pickled_before_the_cholesky_loads(analytic_form):
    """A solution pickled by fips 0.2.0b1 (an eigendecomposition, with no ``dofs_at_gamma``) loads, reads the same, and sweeps gamma."""
    post = fips.solve(_problem(*_arrays("dense", 4))).with_gamma(1.5)
    state = post._solution.__getstate__()
    del state["dofs_at_gamma"]
    old = analytic_form.__new__(analytic_form)
    old.__setstate__(state)
    loaded = pickle.loads(pickle.dumps(Posterior(post.problem, old, post.options)))
    expected = _outputs(post)
    for name, value in _outputs(loaded).items():
        np.testing.assert_array_equal(value, expected[name], err_msg=name)
    assert loaded.with_gamma(3.0)._solution.eigenvalues is loaded._solution.eigenvalues


def test_a_pickle_leaves_out_the_kept_eigendecomposition(analytic_form):
    """The eigendecomposition the first ``with_gamma`` keeps is not pickled or copied; after loading, ``with_gamma`` redoes it."""
    problem = _problem(*_arrays("dense", 5))
    post = fips.solve(problem)
    swept = post.with_gamma(2.0)
    assert vars(post._solution)["_eigendecomposed"].eigenvalues is not None
    for loaded in (pickle.loads(pickle.dumps(post)), copy.deepcopy(post)):
        assert "_eigendecomposed" not in vars(loaded._solution)
        assert loaded._solution.eigenvalues is None
        _close(
            loaded.with_gamma(2.0).estimate.values, swept.estimate.values, "estimate"
        )


# ---------------------------------------------------------------------------
# The two forms: which one factors, and what each holds
# ---------------------------------------------------------------------------


def _mismatch(kind: str, m: int, rng) -> object:
    """A model-data mismatch of ``m`` observations in one form."""
    if kind == "diagonal":
        return sp.diags(rng.uniform(0.5, 2.0, m)).tocsr()
    if kind == "sparse":
        return sp.csr_matrix(
            np.diag(rng.uniform(1, 3, m))
            + np.diag(np.full(m - 1, 0.2), 1)
            + np.diag(np.full(m - 1, 0.2), -1)
        )
    if kind == "block diagonal":
        k = m // 2
        return BlockDiagonalOperator([_spd(rng, k), DiagonalOperator(np.ones(m - k))])
    if kind == "Kronecker":
        return KroneckerOperator([_spd(rng, 2), _spd(rng, m // 2)])
    return _spd(rng, m)  # dense


@pytest.mark.parametrize(
    "n_time, n_cell, m, mismatch, chosen",
    [
        (1, 3, 8, "diagonal", Information),
        (1, 3, 8, "sparse", Information),
        (1, 3, 8, "block diagonal", Information),
        (1, 3, 8, "dense", Information),
        (1, 3, 8, "Kronecker", Information),
        (2, 4, 8, "dense", Whitened),  # n = m
        (3, 4, 7, "dense", Whitened),  # n > m
        (2, 4, 8, "Kronecker", Whitened),  # n = m
        (3, 4, 8, "Kronecker", Whitened),  # n > m
        (2, 4, 8, "diagonal", Whitened),  # n = m
        (3, 4, 7, "diagonal", Whitened),  # n > m
    ],
)
def test_the_form_is_chosen_by_size(n_time, n_cell, m, mismatch, chosen, caplog):
    """The information form when ``n < m`` and ``S_z`` is dense, diagonal, sparse, block diagonal or Kronecker; else the whitened form. Logged at DEBUG."""
    z, x_0, H, S_0, _, c = _arrays("dense", 0, n_time=n_time, n_cell=n_cell, m=m)
    S_z = _mismatch(mismatch, m, np.random.default_rng(1))
    with caplog.at_level(logging.DEBUG, logger="fips.methods.analytic"):
        post = fips.solve(_problem(z, x_0, H, S_0, S_z, c), gamma=0.7)
    assert type(post._solution) is chosen
    name = "information" if chosen is Information else "whitened"
    assert any(
        r.levelno == logging.DEBUG and f"uses the {name} form: " in r.message
        for r in caplog.records
    )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    expected = _closed_form(z, x_0, H, S_0, _dense(S_z), c, 0.7)
    for name, value in _outputs(post).items():
        if name in expected:
            _close(value, expected[name], name)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("seed", range(3))
def test_a_dense_mismatch_with_fewer_states_than_observations(gamma, seed):
    """``n < m`` and a dense ``S_z`` (not chosen by a fixture): the information form, with the closed form's outputs and the same ones as the whitened form."""
    args = _arrays("dense", seed, n_time=2, n_cell=3, m=20)
    problem = _problem(*args)
    post = fips.solve(problem, gamma=gamma)
    assert type(post._solution) is Information
    outputs = _outputs(post)
    for name, expected in _closed_form(*args, gamma).items():
        _close(outputs[name], expected, f"{name}, gamma {gamma}")
    whitened = _outputs(
        Posterior(problem, Whitened.factor(problem, gamma), post.options)
    )
    for name, value in outputs.items():
        _close(value, whitened[name], f"{name} against the whitened form")


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("seed", range(3))
def test_a_kronecker_mismatch_with_fewer_states_than_observations(gamma, seed):
    """``n < m`` and a Kronecker ``S_z`` (with standard deviations, not chosen by a fixture): the information form, with the closed form's outputs and the same ones as the whitened form."""
    rng = np.random.default_rng(100 + seed)
    z, x_0, H, S_0, _, c = _arrays("dense", seed, n_time=2, n_cell=3, m=20)
    S_z = KroneckerOperator([_spd(rng, 4), _spd(rng, 5)], std=rng.uniform(0.5, 2.0, 20))
    problem = _problem(z, x_0, H, S_0, S_z, c)
    post = fips.solve(problem, gamma=gamma)
    assert type(post._solution) is Information
    outputs = _outputs(post)
    for name, expected in _closed_form(z, x_0, H, S_0, S_z, c, gamma).items():
        _close(outputs[name], expected, f"{name}, gamma {gamma}")
    whitened = _outputs(
        Posterior(problem, Whitened.factor(problem, gamma), post.options)
    )
    for name, value in outputs.items():
        _close(value, whitened[name], f"{name} against the whitened form")
    # with_gamma and with_data share the factors and give the same answers
    other = _outputs(post.with_gamma(3.0))
    for name, expected in _closed_form(z, x_0, H, S_0, S_z, c, 3.0).items():
        _close(other[name], expected, f"{name}, with_gamma")
    z_new = z + rng.standard_normal(len(z))
    new = _problem(z_new, x_0, H, S_0, S_z, c)
    again = _outputs(post.with_data(new.obs))
    for name, expected in _closed_form(z_new, x_0, H, S_0, S_z, c, gamma).items():
        _close(again[name], expected, f"{name}, with_data")


def test_a_kronecker_mismatch_that_is_not_positive_definite_raises():
    """A Kronecker ``S_z`` with a factor that is not positive-definite (only semidefinite) gives the clear error, not a wrong answer."""
    z, x_0, H, S_0, _, c = _arrays("dense", 0, n_time=1, n_cell=3, m=8)
    rng = np.random.default_rng(0)
    S_z = KroneckerOperator([np.array([[1.0, 1.0], [1.0, 1.0]]), _spd(rng, 4)])
    with pytest.raises(
        LinAlgError, match="model-data mismatch is not positive-definite"
    ):
        fips.solve(_problem(z, x_0, H, S_0, S_z, c))


def test_a_dense_mismatch_that_is_not_positive_definite_raises():
    """The information form applies ``S_z^-1``, so a dense ``S_z`` it is chosen for must be positive-definite: a clear error, not a wrong answer."""
    z, x_0, H, S_0, _, c = _arrays("dense", 0, n_time=1, n_cell=3, m=8)
    with pytest.raises(
        LinAlgError, match="model-data mismatch is not positive-definite"
    ):
        fips.solve(_problem(z, x_0, H, S_0, np.ones((8, 8)), c))


def test_a_prior_without_a_square_root_takes_the_whitened_form(caplog):
    """A prior error that is only semidefinite has no Cholesky factor: the whitened form, which needs none, solves."""
    z, x_0, H, _, _, c = _arrays("dense", 0, n_time=1, n_cell=3, m=8)
    S_0 = np.diag([1.0, 2.0, 0.0])  # the third element is known exactly
    S_z = sp.diags(np.linspace(0.5, 1.5, 8)).tocsr()
    with caplog.at_level(logging.DEBUG, logger="fips.methods.analytic"):
        post = fips.solve(_problem(z, x_0, H, S_0, S_z, c))
    assert type(post._solution) is Whitened
    assert any("has no square root" in r.message for r in caplog.records)
    # The gain form of the posterior, which needs no inverse of S_0
    K = S_0 @ H.T @ np.linalg.inv(H @ S_0 @ H.T + S_z.toarray())
    _close(post.estimate.values, x_0 + K @ (z - H @ x_0 - c), "estimate")
    assert post.estimate.values[2] == x_0[2]


def test_a_large_whitened_solve_warns(caplog, monkeypatch):
    """Only a whitened solve above ``_LARGE_M`` observations and more than twice ``n`` warns, saying why the information form was not used."""
    monkeypatch.setattr("fips.methods.analytic._LARGE_M", 5)
    rng = np.random.default_rng(2)
    z, x_0, H, _, S_z, c = _arrays("dense", 0, n_time=1, n_cell=3, m=8)
    S_0 = np.diag([1.0, 2.0, 0.0])  # no square root: the one reason left
    with caplog.at_level(logging.WARNING, logger="fips.methods.analytic"):
        fips.solve(_problem(z, x_0, H, S_0, S_z, c))
    (record,) = caplog.records
    assert "m = 8 observations, more than twice the n = 3 state elements" in (
        record.message
    )
    assert "the prior error covariance has no square root" in record.message
    quiet = {
        "information form, diagonal": (
            _arrays("dense", 0, 1, 3, 8),
            _mismatch("diagonal", 8, rng),
        ),
        "information form, dense": (_arrays("dense", 0, 1, 3, 8), None),
        "information form, Kronecker": (
            _arrays("dense", 0, 1, 3, 8),
            _mismatch("Kronecker", 8, rng),
        ),
    }
    for what, ((z, x_0, H, S_0, S_z, c), other) in quiet.items():
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="fips.methods.analytic"):
            fips.solve(_problem(z, x_0, H, S_0, S_z if other is None else other, c))
        assert not caplog.records, what
    # the whitened form, but m = 2 n
    z, x_0, H, _, S_z, c = _arrays("dense", 0, n_time=1, n_cell=4, m=8)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="fips.methods.analytic"):
        fips.solve(_problem(z, x_0, H, np.diag([1.0, 2.0, 3.0, 0.0]), S_z, c))
    assert not caplog.records, "m = 2 n"
    monkeypatch.undo()
    caplog.clear()
    z, x_0, H, _, S_z, c = _arrays("dense", 0, n_time=1, n_cell=3, m=8)
    with caplog.at_level(logging.WARNING, logger="fips.methods.analytic"):
        fips.solve(_problem(z, x_0, H, np.diag([1.0, 2.0, 0.0]), S_z, c))
    assert not caplog.records, "below the default _LARGE_M"


@pytest.mark.parametrize("mismatch", ["diagonal", "Kronecker"])
def test_the_information_form_allocates_nothing_m_by_m(mismatch):
    """
    With ``n = 30`` and ``m = 2,400``, the information form is chosen, and solving and reading every output but the gain never allocates ``m x m``.

    One ``m x m`` array is 46 MB; the information form needs ``m x n`` (0.6
    MB for ``H R``) and holds ``n x n``. The traced peak must stay under half
    of one ``m x m`` array.
    """
    rng = np.random.default_rng(0)
    n, m = 30, 2400
    state = Space({"flux": pd.Index(range(n), name="cell")})
    obs = Space({"conc": pd.Index(range(m), name="obs_id")})
    problem = Problem(
        obs=Vector(rng.normal(size=m), obs),
        prior=Vector(rng.normal(size=n), state),
        forward=Operator(
            sp.random(m, n, density=0.2, random_state=1, format="csr"), obs, state
        ),
        prior_cov=Covariance(_spd(rng, n), state),
        obs_cov=Covariance(
            DiagonalOperator(rng.uniform(0.1, 0.5, m))
            if mismatch == "diagonal"
            else KroneckerOperator(
                [_spd(rng, 40), _spd(rng, 60)], std=rng.uniform(0.5, 2.0, m)
            ),
            obs,
        ),
    )
    m_by_m = m * m * 8
    tracemalloc.start()
    try:
        post = fips.solve(problem)
        _ = post.variances, post.dofs, post.reduced_chi2, post.cov()
        _ = post.with_gamma(2.0).variances, post.with_data(problem.obs).estimate
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert type(post._solution) is Information
    assert peak < m_by_m / 2, (
        f"peak {peak / 1e6:.1f} MB; one m x m is {m_by_m / 1e6:.1f} MB"
    )
    for solution in (post._solution, post.with_gamma(2.0)._solution):
        held = {k: v for k, v in vars(solution).items() if isinstance(v, np.ndarray)}
        shapes = {"G": (n, n), "innovation": (n,)}
        if solution.eigenvalues is not None:
            shapes["eigenvalues"] = (n,)
        assert {k: v.shape for k, v in held.items()} == shapes
