"""
Oracle tests: the posterior against formulas written in this file.

The other posterior tests check properties: shapes, labels, signs, the
direction in which ``gamma`` moves a diagnostic, and ``K`` against its own
equation. The tests here pin the numbers. Each output of the
:class:`fips.Posterior` that :func:`fips.solve` returns is compared with the
closed-form Gaussian posterior, written below with ``numpy.linalg`` and
nothing from fips. They are parametrized over the registered methods, so a
method registered later is held to them too.

Also here: a permutation test (the labeled posterior does not depend on the
order in which any input is given) and dot tests (the adjoint of each linear
map fips applies is its transpose).
"""

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import fips
from fips import (
    Covariance,
    Independent,
    Operator,
    Posterior,
    Problem,
    Space,
    Vector,
    available_methods,
    solve,
)
from fips.kronecker import KroneckerOperator
from fips.linalg import DiagonalOperator, SparseOperator
from fips.solve import _METHODS

METHODS = available_methods()

RTOL = 1e-10
"""
Relative tolerance of every comparison with the closed form.

An absolute floor goes with it: ``RTOL`` times the size of the terms the
quantity is computed from (see :func:`assert_close`). Roundoff in a sum of
products is a multiple of machine epsilon times the sum of the absolute values
of its terms, not of the result, so an entry that is near zero by cancellation
(an off-diagonal of the posterior covariance, or ``x_hat - x_0`` where the data
barely move the state) can only be held to ``1e-16`` of those terms. The test
matrices are well conditioned (condition numbers of order 10), so both sides
are good to about ``1e-14`` of that scale, four orders inside the tolerance,
while a sign error or a transposed matrix is wrong at order one.
"""

SEEDS = range(20)
SIZES = [(n, m) for n in (1, 3, 7) for m in (1, 2, 10)]


# ---------------------------------------------------------------------------
# The closed form, written independently of fips
# ---------------------------------------------------------------------------


@dataclass
class ClosedForm:
    """The Gaussian posterior and its diagnostics, from the formulas."""

    x_hat: np.ndarray
    S_hat_cov: np.ndarray
    S_hat_info: np.ndarray
    K: np.ndarray
    A: np.ndarray
    DOFS: float
    y_hat: np.ndarray
    y_0: np.ndarray
    G: np.ndarray
    reduced_chi2: float
    scale: dict[str, float]


def closed_form(z, x_0, H, S_0, S_z, c) -> ClosedForm:
    """
    Return the posterior of ``z = H x + c + e`` with ``x ~ N(x_0, S_0)``, ``e ~ N(0, S_z)``.

    Written with explicit inverses, the formulas as they are printed, so that
    the reference does not share an algorithm with the solve (which whitens
    and eigendecomposes).
    """
    inv = np.linalg.inv
    G = H @ S_0 @ H.T + S_z
    K = S_0 @ H.T @ inv(G)
    r_0 = z - H @ x_0 - c
    x_hat = x_0 + S_0 @ H.T @ inv(G) @ r_0
    S_hat_cov = S_0 - S_0 @ H.T @ inv(G) @ H @ S_0
    S_hat_info = inv(H.T @ inv(S_z) @ H + inv(S_0))
    A = K @ H
    r = z - H @ x_hat - c
    d = x_hat - x_0
    chi2 = (r @ inv(S_z) @ r + d @ inv(S_0) @ d) / len(z)

    a = np.abs
    scale = {
        "x_hat": max(a(x_0).max(), (a(K) @ a(r_0)).max()),
        "S_hat": max(a(S_0).max(), (a(K) @ a(H @ S_0)).max()),
        "K": (a(S_0 @ H.T) @ a(inv(G))).max(),
        "A": (a(K) @ a(H)).max(),
        "DOFS": np.trace(a(K) @ a(H)),
        "y": (a(H) @ a(x_hat)).max() + a(c).max(),
        "G": (a(H) @ a(S_0) @ a(H.T)).max() + a(S_z).max(),
        "chi2": chi2,
    }
    return ClosedForm(
        x_hat=x_hat,
        S_hat_cov=S_hat_cov,
        S_hat_info=S_hat_info,
        K=K,
        A=A,
        DOFS=float(np.trace(A)),
        y_hat=H @ x_hat + c,
        y_0=H @ x_0 + c,
        G=H @ S_0 @ H.T + S_z,
        reduced_chi2=float(chi2),
        scale=scale,
    )


def assert_close(actual, expected, scale: float, what: str) -> None:
    """Compare at ``RTOL``, with an absolute floor of ``RTOL * scale``."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=RTOL,
        atol=RTOL * scale,
        err_msg=what,
    )


# ---------------------------------------------------------------------------
# Random problems
# ---------------------------------------------------------------------------


def random_spd(rng: np.random.Generator, k: int) -> np.ndarray:
    """Return a dense, well-conditioned SPD matrix with a random scale."""
    B = rng.standard_normal((k, k))
    return (B @ B.T + k * np.eye(k)) * 10 ** rng.uniform(-1, 1)


@dataclass
class Case:
    """A random single-block problem: the arrays and the labels that go with them."""

    z: np.ndarray
    x_0: np.ndarray
    H: np.ndarray
    S_0: np.ndarray
    S_z: np.ndarray
    c: np.ndarray
    offset: float | Vector
    state_index: pd.MultiIndex
    obs_index: pd.MultiIndex

    @classmethod
    def random(cls, seed: int, n: int, m: int, constant: str = "vector") -> "Case":
        """
        Draw a problem with ``n`` states and ``m`` observations.

        Labels are strings, in a random order.
        """
        rng = np.random.default_rng(seed)
        state_index = pd.MultiIndex.from_arrays(
            [["flux"] * n, [f"c{k}" for k in rng.permutation(n)]],
            names=["block", "cell"],
        )
        obs_index = pd.MultiIndex.from_arrays(
            [["conc"] * m, [f"o{k}" for k in rng.permutation(m)]],
            names=["block", "obs_id"],
        )
        x_0 = rng.normal(5.0, 2.0, n)
        H = rng.standard_normal((m, n))
        S_0 = random_spd(rng, n)
        S_z = random_spd(rng, m)
        z = H @ x_0 + rng.standard_normal(m) * 3.0
        if constant == "scalar":
            value = float(rng.normal(1.0, 0.5))
            c, const = np.full(m, value), value
        else:
            c = rng.normal(1.0, 0.5, m)
            const = Vector.from_series(pd.Series(c, index=obs_index))
        return cls(z, x_0, H, S_0, S_z, c, const, state_index, obs_index)

    def problem(self, **inputs) -> Problem:
        """Build the labeled problem; keyword arguments replace an input."""
        args = {
            "obs": Vector.from_series(pd.Series(self.z, index=self.obs_index)),
            "prior": Vector.from_series(pd.Series(self.x_0, index=self.state_index)),
            "forward": Operator.from_frame(
                pd.DataFrame(self.H, index=self.obs_index, columns=self.state_index)
            ),
            "prior_cov": Covariance.from_frame(
                pd.DataFrame(self.S_0, index=self.state_index, columns=self.state_index)
            ),
            "obs_cov": Covariance.from_frame(
                pd.DataFrame(self.S_z, index=self.obs_index, columns=self.obs_index)
            ),
            "offset": self.offset,
        }
        args.update(inputs)
        return Problem(**args)

    @property
    def state_space(self) -> Space:
        """The state's labels, in the order of the arrays."""
        return Space.from_index(self.state_index)

    @property
    def obs_space(self) -> Space:
        """The observations' labels, in the order of the arrays."""
        return Space.from_index(self.obs_index)

    def sparse_inputs(self) -> dict[str, Operator]:
        """Return ``H``, ``S_0`` and ``S_z`` stored sparse, as inputs of :meth:`problem`."""
        s, o = self.state_space, self.obs_space
        # SparseOperator itself: a 1 x 1 sparse matrix would read as a diagonal
        return {
            "forward": Operator(SparseOperator(sp.csr_array(self.H)), o, s),
            "prior_cov": Covariance(SparseOperator(sp.csr_array(self.S_0)), s),
            "obs_cov": Covariance(SparseOperator(sp.csr_array(self.S_z)), o),
        }


# ---------------------------------------------------------------------------
# Reading the labeled outputs
# ---------------------------------------------------------------------------


def vector_on(v: Vector, index: pd.Index, what: str) -> np.ndarray:
    """Return the values of `v` after checking that it is labeled by `index`, in order."""
    assert isinstance(v, Vector), what
    assert v.index.equals(index), f"{what}: wrong index"
    assert list(v.index.names) == list(index.names), f"{what}: wrong level names"
    return v.values


def matrix_on(M: Operator, index: pd.Index, columns: pd.Index, what: str) -> np.ndarray:
    """Return the values of `M` after checking its labels on both axes, in order."""
    assert isinstance(M, Operator), what
    rows, cols = M.rows.index, M.cols.index
    assert rows.equals(index), f"{what}: wrong row index"
    assert cols.equals(columns), f"{what}: wrong column index"
    assert list(rows.names) == list(index.names), f"{what}: wrong row level names"
    assert list(cols.names) == list(columns.names), f"{what}: wrong column level names"
    return M.dense()


def check_posterior(
    post: Posterior,
    expected: ClosedForm,
    state_index: pd.Index,
    obs_index: pd.Index,
    what: str,
) -> None:
    """Compare every output of a posterior with the closed form, by label."""
    s, o, sc = state_index, obs_index, expected.scale

    def at(name):
        return f"{what}: {name}"

    assert isinstance(post, Posterior), what
    x_hat = vector_on(post.estimate, s, at("estimate"))
    assert_close(x_hat, expected.x_hat, sc["x_hat"], at("estimate"))
    assert post.mean is post.estimate
    S_hat = matrix_on(post.cov(), s, s, at("cov()"))
    assert_close(S_hat, expected.S_hat_cov, sc["S_hat"], at("S_hat, covariance form"))
    assert_close(S_hat, expected.S_hat_info, sc["S_hat"], at("S_hat, information form"))
    assert_close(
        vector_on(post.variances, s, at("variances")),
        np.diag(expected.S_hat_cov),
        sc["S_hat"],
        at("variances"),
    )
    assert_close(
        matrix_on(post.kalman_gain, s, o, at("kalman_gain")),
        expected.K,
        sc["K"],
        at("kalman_gain"),
    )
    assert_close(
        matrix_on(post.averaging_kernel(), s, s, at("averaging_kernel")),
        expected.A,
        sc["A"],
        at("averaging_kernel"),
    )
    assert_close(post.dofs, expected.DOFS, sc["DOFS"], at("dofs"))
    assert_close(
        vector_on(post.modeled_obs, o, at("modeled_obs")),
        expected.y_hat,
        sc["y"],
        at("modeled_obs"),
    )
    assert_close(
        vector_on(post.problem.predict(post.problem.prior), o, at("predict(prior)")),
        expected.y_0,
        sc["y"],
        at("predict(prior)"),
    )
    assert_close(
        matrix_on(post.innovation_cov, o, o, at("innovation_cov")),
        expected.G,
        sc["G"],
        at("innovation_cov"),
    )
    assert_close(
        post.reduced_chi2, expected.reduced_chi2, sc["chi2"], at("reduced_chi2")
    )
    assert_close(
        post.chi2,
        expected.reduced_chi2 * len(o),
        sc["chi2"] * len(o),
        at("chi2"),
    )


def takes_gamma(method: str) -> bool:
    """Return True if the registered method has a ``gamma`` option."""
    return "gamma" in [f.name for f in fields(_METHODS[method].options)]


# ---------------------------------------------------------------------------
# 1. The closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n, m", SIZES)
def test_closed_form(method, n, m, sparse):
    """
    The posterior equals the closed form, in both covariance forms, for 20 seeds.

    The offset is a scalar for the even seeds and a vector for the odd ones,
    so each size sees both forms without solving every problem twice. With
    ``sparse``, ``H``, ``S_0`` and ``S_z`` are stored sparse (#79).
    """
    for seed in SEEDS:
        case = Case.random(seed, n, m, "scalar" if seed % 2 == 0 else "vector")
        expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
        # The reference itself: the two forms of S_hat, and A = K H, agree
        assert_close(
            expected.S_hat_cov,
            expected.S_hat_info,
            expected.scale["S_hat"],
            f"seed {seed}: the covariance and information forms",
        )
        problem = case.problem(**case.sparse_inputs() if sparse else {})
        check_posterior(
            solve(problem, method),
            expected,
            case.state_index,
            case.obs_index,
            f"seed {seed}",
        )


@pytest.mark.parametrize("method", [m for m in METHODS if takes_gamma(m)])
@pytest.mark.parametrize("gamma", [0.25, 3.0])
@pytest.mark.parametrize("n, m", [(3, 10), (7, 2)])
def test_gamma_scales_mismatch(method, gamma, n, m):
    """``gamma`` gives the closed form with the model-data mismatch divided by ``gamma``."""
    for seed in range(5):
        case = Case.random(seed, n, m)
        expected = closed_form(
            case.z, case.x_0, case.H, case.S_0, case.S_z / gamma, case.c
        )
        post = solve(case.problem(), method, gamma=gamma)
        check_posterior(
            post, expected, case.state_index, case.obs_index, f"seed {seed}"
        )
        # and from another gamma's factorization
        post = solve(case.problem(), method, gamma=1.0).with_gamma(gamma)
        check_posterior(
            post, expected, case.state_index, case.obs_index, f"seed {seed}, with_gamma"
        )


def _diagonal_inputs(case: Case, form: str, v_0: np.ndarray, v_z: np.ndarray):
    """Return the prior error and mismatch given as variances, in one of several forms."""
    s, o = case.state_index, case.obs_index
    if form == "vector":
        return (
            Covariance.diagonal(Vector.from_series(pd.Series(v_0, index=s))),
            Covariance.diagonal(Vector.from_series(pd.Series(v_z, index=o))),
        )
    if form == "model":
        return (
            Independent().with_std(np.sqrt(v_0)),
            Independent().with_std(np.sqrt(v_z)),
        )
    if form == "std_vector":
        return (
            Independent().with_std(
                Vector.from_series(pd.Series(np.sqrt(v_0), index=s))
            ),
            Independent().with_std(
                Vector.from_series(pd.Series(np.sqrt(v_z), index=o))
            ),
        )
    if form == "scalar":
        return (
            Independent().with_std(np.sqrt(v_0[0])),
            Independent().with_std(np.sqrt(v_z[0])),
        )
    if form == "diagonal":
        S_0 = Covariance.diagonal(v_0, case.state_space)
        S_z = Covariance.diagonal(v_z, case.obs_space)
        assert isinstance(S_0.data, DiagonalOperator)
        return S_0, S_z
    if form == "sparse":
        return (
            Covariance(SparseOperator(sp.diags_array(v_0)), case.state_space),
            Covariance(SparseOperator(sp.diags_array(v_z)), case.obs_space),
        )
    raise ValueError(form)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "form",
    [
        "vector",
        "model",
        "std_vector",
        "scalar",
        "diagonal",
        "sparse",
    ],
)
def test_diagonal_given_as_variances(method, form):
    """A diagonal covariance, given as variances or as a model, solves as the full matrix does."""
    for seed in range(5):
        case = Case.random(seed, 7, 10)
        rng = np.random.default_rng(seed)
        if form == "scalar":
            v_0, v_z = np.full(7, 2.5), np.full(10, 0.4)
        else:
            v_0, v_z = rng.uniform(0.5, 4.0, 7), rng.uniform(0.1, 1.0, 10)
        S_0, S_z = np.diag(v_0), np.diag(v_z)
        expected = closed_form(case.z, case.x_0, case.H, S_0, S_z, case.c)
        prior_cov, obs_cov = _diagonal_inputs(case, form, v_0, v_z)
        post = solve(case.problem(prior_cov=prior_cov, obs_cov=obs_cov), method)
        check_posterior(
            post, expected, case.state_index, case.obs_index, f"seed {seed}"
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("rows", ["product", "permuted", "ragged"])
def test_kronecker_prior_error(method, rows):
    """
    A prior error kept as Kronecker factors gives the closed form of its matrix.

    The matrix for the closed form is ``np.kron`` of the factors, with the rows
    picked and scaled here: the whole product in order, a permutation of it, or
    9 of its 12 rows.
    """
    sizes = (3, 4)
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        chosen = {
            "product": np.arange(12),
            "permuted": rng.permutation(12),
            "ragged": rng.permutation(12)[:9],
        }[rows]
        case = Case.random(seed, len(chosen), 7)
        factors = [random_spd(rng, k) for k in sizes]
        std = rng.uniform(0.5, 2.0, len(chosen))
        S_0 = std[:, None] * np.kron(*factors)[np.ix_(chosen, chosen)] * std[None, :]
        expected = closed_form(case.z, case.x_0, case.H, S_0, case.S_z, case.c)
        op = KroneckerOperator(
            factors, std=std, positions=np.unravel_index(chosen, sizes)
        )
        problem = case.problem(prior_cov=Covariance(op, case.state_space))
        assert isinstance(problem.prior_cov.data, KroneckerOperator)
        post = solve(problem, method)
        check_posterior(
            post, expected, case.state_index, case.obs_index, f"seed {seed}"
        )


@pytest.mark.parametrize("method", METHODS)
def test_sparse_forward_operator(method):
    """A sparse forward operator gives the closed form, as the dense one does."""
    for seed in range(5):
        case = Case.random(seed, 7, 10)
        case.H[np.abs(case.H) < 0.8] = 0.0  # make it sparse
        expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
        H = Operator(sp.csr_array(case.H), case.obs_space, case.state_space)
        assert isinstance(H.data, SparseOperator)
        post = solve(case.problem(forward=H), method)
        check_posterior(
            post, expected, case.state_index, case.obs_index, f"seed {seed}"
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "given", ["sparse tuple", "pandas-sparse frame", "dense frame"]
)
def test_jacobian_on_part_of_the_state_is_placed(method, given):
    """
    A labeled Jacobian of some cells, placed by ``Operator.from_blocks``, gives the closed form.

    The columns it lacks are cells no observation sees: zero columns of the
    ``H`` the closed form uses. Its rows and columns come in their own order.
    """
    rng = np.random.default_rng(7)
    for seed in range(5):
        case = Case.random(seed, 7, 10)
        seen = np.sort(rng.choice(7, 4, replace=False))
        case.H[:, np.setdiff1d(np.arange(7), seen)] = 0.0
        expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)

        cells = case.state_space.select("flux")
        obs_ids = case.obs_space.select("conc")
        rows, cols = rng.permutation(10), seen[rng.permutation(4)]
        J = case.H[np.ix_(rows, cols)]
        if given == "sparse tuple":
            block = (sp.csr_array(J), obs_ids[rows], cells[cols])
        else:
            block = pd.DataFrame(J, index=obs_ids[rows], columns=cells[cols])
            if given == "pandas-sparse frame":
                block = block.astype(pd.SparseDtype(float, 0.0))
        forward = Operator.from_blocks(
            {("conc", "flux"): block}, rows=case.obs_space, cols=case.state_space
        )
        assert isinstance(forward.data, SparseOperator) == (given != "dense frame")
        post = solve(case.problem(forward=forward), method)
        check_posterior(
            post, expected, case.state_index, case.obs_index, f"seed {seed}"
        )
        # and the placed operator's adjoint is its transpose
        form, adjoint = forward.data, forward.T.data
        dot_test(lambda v: form @ v, lambda v: adjoint @ v, 7, 10, rng)  # noqa: B023


# ---------------------------------------------------------------------------
# 2. Several blocks
# ---------------------------------------------------------------------------


@dataclass
class MultiBlockCase:
    """
    Two state blocks and two observation blocks, each with its own index levels.

    The arrays are the assembled problem, with the blocks in the order
    ``flux, bias`` (state) and ``co2, ch4`` (observations).
    """

    state: dict[str, pd.Index]
    obs: dict[str, pd.Index]
    z: np.ndarray
    x_0: np.ndarray
    H: np.ndarray
    S_0: np.ndarray
    S_z: np.ndarray
    c: np.ndarray

    @classmethod
    def random(cls, seed: int) -> "MultiBlockCase":
        """Draw the arrays; the ``(co2, bias)`` block of ``H`` is left out (zero)."""
        rng = np.random.default_rng(seed)
        state = {
            "flux": pd.MultiIndex.from_product([[0, 1, 2], [0, 1]], names=["x", "y"]),
            "bias": pd.Index(["north", "south"], name="site"),
        }
        obs = {
            "co2": pd.Index([f"a{k}" for k in rng.permutation(5)], name="obs_id"),
            "ch4": pd.MultiIndex.from_product(
                [["north", "south"], [6, 18]], names=["site", "hour"]
            ),
        }
        n, m = 8, 9
        H = rng.standard_normal((m, n))
        H[:5, 6:] = 0.0  # no (co2, bias) block
        x_0 = rng.normal(5.0, 2.0, n)
        z = H @ x_0 + rng.standard_normal(m)
        c = rng.normal(1.0, 0.5, m)
        S_0, S_z = random_spd(rng, n), random_spd(rng, m)
        return cls(state, obs, z, x_0, H, S_0, S_z, c)

    @staticmethod
    def _slices(blocks: dict[str, pd.Index]) -> dict[str, slice]:
        """Return where each block sits in the assembled arrays."""
        out, start = {}, 0
        for name, idx in blocks.items():
            out[name] = slice(start, start + len(idx))
            start += len(idx)
        return out

    @property
    def state_slices(self) -> dict[str, slice]:
        """Where each state block sits in the assembled state arrays."""
        return self._slices(self.state)

    @property
    def obs_slices(self) -> dict[str, slice]:
        """Where each observation block sits in the assembled observation arrays."""
        return self._slices(self.obs)

    def problem(self) -> Problem:
        """Build the problem from vectors and operators assembled from their blocks."""
        ss, os_ = self.state_slices, self.obs_slices

        def vector(values, blocks, slices):
            return Vector.from_blocks(
                {
                    b: pd.Series(values[slices[b]], index=idx)
                    for b, idx in blocks.items()
                }
            )

        def blocks_2d(values, rows, cols, rs, cs, skip=(), cls=Operator):
            return cls.from_blocks(
                {
                    (r, c): pd.DataFrame(values[rs[r], cs[c]], index=ri, columns=ci)
                    for r, ri in rows.items()
                    for c, ci in cols.items()
                    if (r, c) not in skip
                }
            )

        return Problem(
            obs=vector(self.z, self.obs, os_),
            prior=vector(self.x_0, self.state, ss),
            forward=blocks_2d(
                self.H, self.obs, self.state, os_, ss, skip={("co2", "bias")}
            ),
            prior_cov=blocks_2d(
                self.S_0, self.state, self.state, ss, ss, cls=Covariance
            ),
            obs_cov=blocks_2d(self.S_z, self.obs, self.obs, os_, os_, cls=Covariance),
            offset=vector(self.c, self.obs, os_),
        )


def _block_values(structure, key, index, columns=None) -> np.ndarray:
    """Read one block of a solved output by its name(s), checking its labels."""
    if columns is None:
        vector = structure[key]
        labels = vector.space.select(key)
        assert labels.equals(index), f"{key}: wrong index"
        assert list(labels.names) == list(index.names), f"{key}: level names"
        return vector.values
    op = structure[key]
    rows, cols = op.rows.select(key[0]), op.cols.select(key[1])
    assert rows.equals(index), f"{key}: wrong row index"
    assert cols.equals(columns), f"{key}: wrong column index"
    assert list(rows.names) == list(index.names), f"{key}: row level names"
    assert list(cols.names) == list(columns.names), f"{key}: column level names"
    return op.dense()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(5))
def test_multi_block(method, seed):
    """Block by block, read back by label, the posterior equals the closed form."""
    case = MultiBlockCase.random(seed)
    expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
    post = solve(case.problem(), method)
    sc = expected.scale
    ss, os_ = case.state_slices, case.obs_slices

    for b, idx in case.state.items():
        what = f"seed {seed}: estimate[{b}]"
        got = _block_values(post.estimate, b, idx)
        assert_close(got, expected.x_hat[ss[b]], sc["x_hat"], what)
        for b2, idx2 in case.state.items():
            what = f"seed {seed}: [{b}, {b2}]"
            got = _block_values(post.cov(), (b, b2), idx, idx2)
            assert_close(got, expected.S_hat_cov[ss[b], ss[b2]], sc["S_hat"], what)
            got = _block_values(post.averaging_kernel(), (b, b2), idx, idx2)
            assert_close(got, expected.A[ss[b], ss[b2]], sc["A"], what)
        for o, oidx in case.obs.items():
            what = f"seed {seed}: kalman_gain[{b}, {o}]"
            got = _block_values(post.kalman_gain, (b, o), idx, oidx)
            assert_close(got, expected.K[ss[b], os_[o]], sc["K"], what)

    for o, oidx in case.obs.items():
        what = f"seed {seed}: modeled_obs[{o}]"
        got = _block_values(post.modeled_obs, o, oidx)
        assert_close(got, expected.y_hat[os_[o]], sc["y"], what)
        for o2, oidx2 in case.obs.items():
            what = f"seed {seed}: innovation_cov[{o}, {o2}]"
            got = _block_values(post.innovation_cov, (o, o2), oidx, oidx2)
            assert_close(got, expected.G[os_[o], os_[o2]], sc["G"], what)

    assert_close(post.dofs, expected.DOFS, sc["DOFS"], "dofs")
    assert_close(post.reduced_chi2, expected.reduced_chi2, sc["chi2"], "chi2")


# ---------------------------------------------------------------------------
# 3. Permutation
# ---------------------------------------------------------------------------


def _shuffled(obj, rng: np.random.Generator):
    """Return a Series in a random row order, or a DataFrame with rows and columns each in their own."""
    if isinstance(obj, pd.Series):
        return obj.iloc[rng.permutation(len(obj))]
    return obj.iloc[rng.permutation(obj.shape[0]), rng.permutation(obj.shape[1])]


def _aligned(structure, index, columns=None) -> np.ndarray:
    """Return a structure's values in the order of `index` (and `columns`), by label."""
    data = (
        structure.to_frame()
        if isinstance(structure, Operator)
        else structure.to_series()
    )
    assert data.index.sort_values().equals(index.sort_values()), "row labels differ"
    if columns is None:
        return data.reindex(index).to_numpy()
    assert data.columns.sort_values().equals(columns.sort_values()), (
        "column labels differ"
    )
    return data.reindex(index=index, columns=columns).to_numpy()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(5))
def test_permutation(method, seed):
    """Each input in its own random order gives the same labeled posterior."""
    case = MultiBlockCase.random(seed)
    ref = solve(case.problem(), method)
    p = ref.problem
    rng = np.random.default_rng(1000 + seed)

    # The assembled inputs, with block levels, each shuffled on its own
    shuffled = solve(
        Problem(
            obs=Vector.from_series(_shuffled(p.obs.to_series(), rng)),
            prior=Vector.from_series(_shuffled(p.prior.to_series(), rng)),
            forward=Operator.from_frame(_shuffled(p.forward.to_frame(), rng)),
            prior_cov=Covariance.from_frame(_shuffled(p.prior_cov.to_frame(), rng)),
            obs_cov=Covariance.from_frame(_shuffled(p.obs_cov.to_frame(), rng)),
            offset=Vector.from_series(_shuffled(p.offset.to_series(), rng)),
        ),
        method,
    )

    s, o = p.prior.index, p.obs.index
    expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
    sc = expected.scale
    outputs = [
        ("estimate", lambda post: post.estimate, s, None, "x_hat"),
        ("cov()", lambda post: post.cov(), s, s, "S_hat"),
        ("averaging_kernel()", lambda post: post.averaging_kernel(), s, s, "A"),
        ("kalman_gain", lambda post: post.kalman_gain, s, o, "K"),
        ("modeled_obs", lambda post: post.modeled_obs, o, None, "y"),
        ("innovation_cov", lambda post: post.innovation_cov, o, o, "G"),
    ]
    for name, output, rows, cols, key in outputs:
        want = output(ref)
        assert_close(
            _aligned(output(shuffled), rows, cols),
            want.dense() if cols is not None else want.values,
            sc[key],
            f"seed {seed}: {name}",
        )
    for diag in ["dofs", "chi2", "reduced_chi2", "r2", "rmse", "uncertainty_reduction"]:
        got = getattr(shuffled, diag)
        want = getattr(ref, diag)
        assert_close(got, want, abs(want), f"seed {seed}: {diag}")


# ---------------------------------------------------------------------------
# 4. Adjoint (dot) tests
# ---------------------------------------------------------------------------


def dot_test(
    forward: Callable[[np.ndarray], np.ndarray],
    adjoint: Callable[[np.ndarray], np.ndarray],
    n: int,
    m: int,
    rng: np.random.Generator,
    rtol: float = 1e-12,
    repeats: int = 3,
) -> None:
    """
    Check that `adjoint` is the adjoint of `forward`, a linear map from R^n to R^m.

    For random ``x`` and ``y``, ``<A x, y>`` must equal ``<x, A^T y>`` to within
    `rtol` of ``|A x| |y| + |x| |A^T y|``, the size of the terms the two inner
    products are summed from.
    """
    for _ in range(repeats):
        x, y = rng.standard_normal(n), rng.standard_normal(m)
        Ax, ATy = np.asarray(forward(x)), np.asarray(adjoint(y))
        assert Ax.shape == (m,), f"forward returned shape {Ax.shape}, not ({m},)"
        assert ATy.shape == (n,), f"adjoint returned shape {ATy.shape}, not ({n},)"
        lhs, rhs = Ax @ y, x @ ATy
        bound = np.linalg.norm(Ax) * np.linalg.norm(y)
        bound += np.linalg.norm(x) * np.linalg.norm(ATy)
        assert abs(lhs - rhs) <= rtol * bound, (
            f"<Ax, y> = {lhs!r} but <x, A^T y> = {rhs!r}"
        )


def test_dot_test_catches_a_wrong_adjoint():
    """The dot test fails for a map paired with something that is not its adjoint."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 3))
    dot_test(lambda x: A @ x, lambda y: A.T @ y, 3, 4, rng)
    with pytest.raises(AssertionError):
        dot_test(lambda x: A @ x, lambda y: 1.01 * A.T @ y, 3, 4, rng)
    B = rng.standard_normal((3, 3))
    with pytest.raises(AssertionError):
        dot_test(lambda x: B @ x, lambda y: B @ y, 3, 3, rng)


def _labeled_map(frame: pd.DataFrame, rng: np.random.Generator):
    """
    Return the map ``x -> frame @ x`` through ``Operator @ Vector``.

    ``x`` is in the order of ``frame.columns`` and the result in the order of
    ``frame.index``; the state is given to the product in a random order, so
    it goes through the label alignment.
    """
    H = Operator.from_frame(frame)

    def apply(x: np.ndarray) -> np.ndarray:
        perm = rng.permutation(len(x))
        state = pd.Series(x[perm], index=frame.columns[perm], name="x")
        y = H @ Vector.from_series(state)
        assert isinstance(y, Vector)
        assert y.index.equals(frame.index)
        return y.values

    return apply


def _matrix_structure(kind: str, sparse: bool, rng: np.random.Generator):
    """Return a two-block operator of class `kind`, dense or sparse."""
    flux = pd.MultiIndex.from_product([[0, 1, 2], [0, 1]], names=["x", "y"])
    bias = pd.Index(["north", "south"], name="site")
    co2 = pd.Index([f"a{k}" for k in range(5)], name="obs_id")
    ch4 = pd.Index(["north", "south"], name="site")

    def block(rows, cols):
        values = rng.standard_normal((len(rows), len(cols)))
        values[np.abs(values) < 0.7] = 0.0
        if sparse:
            return sp.csr_array(values)
        return values

    if kind == "Covariance":
        blocks = {
            ("flux", "flux"): block(flux, flux),
            ("flux", "bias"): block(flux, bias),
            ("bias", "bias"): block(bias, bias),
        }
        space = Space({"flux": flux, "bias": bias})
        return Covariance.from_blocks(blocks, rows=space, cols=space)
    blocks = {
        ("co2", "flux"): block(co2, flux),
        ("co2", "bias"): block(co2, bias),
        ("ch4", "flux"): block(ch4, flux),
    }
    return Operator.from_blocks(
        blocks,
        rows=Space({"co2": co2, "ch4": ch4}),
        cols=Space({"flux": flux, "bias": bias}),
    )


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("kind", ["Operator", "Covariance"])
def test_dot_test_structures(kind, sparse):
    """An operator and its transpose, applied to vectors by label, are adjoint."""
    rng = np.random.default_rng(1)
    M = _matrix_structure(kind, sparse, rng)
    assert isinstance(M.data, SparseOperator) == sparse
    assert isinstance(M, Covariance) == (kind == "Covariance")
    frame = M.to_frame()
    m, n = frame.shape
    dot_test(_labeled_map(frame, rng), _labeled_map(frame.T, rng), n, m, rng)
    # And the map is the matrix the operator holds, and its transpose's
    x = rng.standard_normal(n)
    np.testing.assert_allclose(_labeled_map(frame, rng)(x), M.dense() @ x, rtol=1e-12)
    dot_test(lambda v: M.data @ v, lambda v: M.T.data @ v, n, m, rng)


def _obs_index(blocks: list[str], seed: int) -> pd.MultiIndex:
    """Six-hourly observations at two sites over three days, in a random order."""
    times = pd.date_range("2024-01-01", periods=12, freq="6h")
    idx = pd.MultiIndex.from_product(
        [blocks, ["north", "south"], times], names=["block", "site", "time"]
    )
    return idx[np.random.default_rng(seed).permutation(len(idx))]


_DAILY = pd.Grouper(level="time", freq="D")
_DAYS = pd.IntervalIndex.from_breaks(
    pd.date_range("2024-01-01", periods=4, freq="D"), closed="left", name="time"
)


def _groups_by_hand(
    index: pd.MultiIndex, aggregated: list[str], by: list[str], passthrough: bool
):
    """
    The group of each label, written with pandas alone: (block, site and/or day).

    A label of a block not aggregated is its own group if it passes through,
    and in none (None) if not.
    """
    day = index.get_level_values("time").floor("D")
    site = index.get_level_values("site")
    block = index.get_level_values("block")
    keys = []
    for k in range(len(index)):
        if block[k] in aggregated:
            parts = {"site": site[k], "day": day[k]}
            keys.append((block[k], *(parts[name] for name in by)))
        else:
            keys.append(("pass", *index[k]) if passthrough else None)
    return keys


def _expected_aggregation(index, aggregated, by, average, passthrough):
    """W written by hand: rows are the groups (in any order), columns the labels of `index`."""
    keys = _groups_by_hand(index, aggregated, by, passthrough)
    groups = sorted({key for key in keys if key is not None}, key=repr)
    W = np.zeros((len(groups), len(index)))
    for j, key in enumerate(keys):
        if key is not None:
            W[groups.index(key), j] = 1.0
    if average:
        W /= W.sum(axis=1, keepdims=True)
    return W


AGGREGATIONS = {
    "daily-mean": (
        lambda s: fips.aggregate.mean(s, by=["site", _DAILY]),
        ["conc"],
        ["conc"],
        ["site", "day"],
        True,
        True,
    ),
    "site-sum": (
        lambda s: fips.aggregate.sum(s, by="site"),
        ["conc"],
        ["conc"],
        ["site"],
        False,
        True,
    ),
    "one-block": (
        lambda s: fips.aggregate.mean(s, by=["site", _DAYS], blocks="co2"),
        ["co2", "ch4"],
        ["co2"],
        ["site", "day"],
        True,
        True,
    ),
    "offsets": (
        lambda s: fips.offset_block(s, by=_DAILY, blocks="co2")[1].T,
        ["co2", "ch4"],
        ["co2"],
        ["day"],
        False,
        False,
    ),
}


@pytest.mark.parametrize("name", list(AGGREGATIONS))
def test_dot_test_aggregation(name):
    """
    An aggregation and its transpose, applied to vectors by label, are adjoint, and the matrix is the one written by hand.

    ``W z`` comes from ``W @ Vector`` with the observations in a random
    order; ``W^T y`` from ``W.T @ Vector`` with the aggregates in another.
    The matrix is compared with one built from the labels with pandas alone
    (rows matched by their groups). The offsets' forward block is the
    transpose of a sum, so its transpose is checked as one.
    """
    make, blocks, aggregated, by, average, passthrough = AGGREGATIONS[name]
    rng = np.random.default_rng(2)
    obs_index = _obs_index(blocks, seed=3)
    space = Space.from_index(obs_index)
    W = make(space)
    m, k = len(space), W.shape[0]
    assert W.cols == space
    assert k < m

    def forward(x):
        perm = rng.permutation(m)
        z = Vector.from_series(pd.Series(x[perm], index=space.index[perm]))
        Wz = W @ z
        assert Wz.space == W.rows
        return Wz.values

    def adjoint(y):
        perm = rng.permutation(k)
        y_agg = Vector.from_series(pd.Series(y[perm], index=W.rows.index[perm]))
        WTy = W.T @ y_agg
        assert WTy.space == space
        return WTy.values

    dot_test(forward, adjoint, m, k, rng)

    expected = _expected_aggregation(space.index, aggregated, by, average, passthrough)
    got = W.dense()
    # Rows in fips's order: match each to the hand-written row with its pattern
    assert expected.shape == got.shape
    order = [int(np.flatnonzero((expected == row).all(axis=1))[0]) for row in got]
    assert sorted(order) == list(range(k)), "a group is missing or doubled"
    np.testing.assert_array_equal(got, expected[order])


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("name", ["daily-mean", "site-sum", "one-block"])
def test_problem_aggregate_is_one_map(name, sparse):
    """
    ``Problem.aggregate(W)`` puts the observations, ``H``, ``S_z`` and the offset through one ``W``.

    Each input is given in its own random order, so the result is right only
    if each is aligned by label: ``W z``, ``W H``, ``W S_z W^T`` and ``W c``
    against the matrices by position.
    """
    make, blocks = AGGREGATIONS[name][:2]
    rng = np.random.default_rng(4)
    obs_index = _obs_index(blocks, seed=5)
    m, n = len(obs_index), 3
    state_index = pd.MultiIndex.from_arrays(
        [["flux"] * n, [f"c{j}" for j in range(n)]], names=["block", "cell"]
    )
    z, c = rng.standard_normal(m), rng.standard_normal(m)
    H, S_z = rng.standard_normal((m, n)), random_spd(rng, m)
    forward = Operator.from_frame(
        _shuffled(pd.DataFrame(H, index=obs_index, columns=state_index), rng)
    )
    obs_cov = Covariance.from_frame(
        _shuffled(pd.DataFrame(S_z, index=obs_index, columns=obs_index), rng)
    )
    if sparse:
        forward = Operator(
            SparseOperator(sp.csr_array(forward.dense())), forward.rows, forward.cols
        )
        obs_cov = Covariance(
            SparseOperator(sp.csr_array(obs_cov.dense())), obs_cov.space
        )
    problem = Problem(
        prior=Vector(np.zeros(n), Space.from_index(state_index)),
        prior_cov=Independent().with_std(1.0),
        obs=Vector.from_series(_shuffled(pd.Series(z, index=obs_index), rng)),
        obs_cov=obs_cov,
        forward=forward,
        offset=Vector.from_series(_shuffled(pd.Series(c, index=obs_index), rng)),
    )
    W = make(Space.from_index(_shuffled(pd.Series(0.0, index=obs_index), rng).index))
    agg = problem.aggregate(W)
    W_pos = W.to_frame().reindex(columns=obs_index).to_numpy()
    rows = W.rows.index
    scale = np.abs(W_pos).sum(axis=1).max()
    assert_close(_aligned(agg.obs, rows), W_pos @ z, scale * np.abs(z).max(), "W z")
    assert_close(_aligned(agg.offset, rows), W_pos @ c, scale * np.abs(c).max(), "W c")
    assert_close(
        _aligned(agg.forward, rows, state_index),
        W_pos @ H,
        scale * np.abs(H).max(),
        "W H",
    )
    assert_close(
        _aligned(agg.obs_cov, rows, rows),
        W_pos @ S_z @ W_pos.T,
        scale**2 * np.abs(S_z).max(),
        "W S_z W^T",
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("average", [False, True], ids=["sum", "mean"])
def test_posterior_of_state_aggregates(method, average):
    """
    ``post.cov(W)`` and ``post.averaging_kernel(W)`` with an aggregation of the state are the closed form's.

    The state's cells are mapped to regions (a ``Series`` key) and summed or
    averaged: ``W S_hat W^T`` and ``W A W^T (W W^T)^-1``.
    """
    case = Case.random(seed=11, n=7, m=10)
    post = solve(case.problem(), method)
    cells = case.state_index.get_level_values("cell")
    regions = pd.Series(
        ["east", "west", "east", "north", "west", "east", "north"],
        index=pd.Index([f"c{j}" for j in range(7)], name="cell"),
        name="region",
    )
    aggregate = fips.aggregate.mean if average else fips.aggregate.sum
    W = aggregate(case.state_space, by=regions)
    W_pos = W.to_frame().reindex(columns=case.state_index).to_numpy()
    by_hand = np.zeros_like(W_pos)
    names = list(W.rows.select("flux"))
    for j, cell in enumerate(cells):
        by_hand[names.index(regions[cell]), j] = 1.0
    if average:
        by_hand /= by_hand.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(W_pos, by_hand)

    expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
    sc = expected.scale
    rows = W.rows.index
    assert_close(
        _aligned(post.cov(W), rows, rows),
        W_pos @ expected.S_hat_cov @ W_pos.T,
        sc["S_hat"] * 7,
        "W S_hat W^T",
    )
    pinv = W_pos.T @ np.linalg.inv(W_pos @ W_pos.T)
    assert_close(
        _aligned(post.averaging_kernel(W), rows, rows),
        W_pos @ expected.A @ pinv,
        sc["A"] * 7,
        "W A W*",
    )
