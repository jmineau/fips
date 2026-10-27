"""
Tests for ``Covariance.sample``: a draw from ``N(0, S)`` made without forming ``S`` where a form has a cheap root.

Each form (diagonal, block diagonal, dense, sparse with independent groups,
Kronecker on its whole grid and on a strict subset, and a semi-definite
``Constant()`` block) is held to the same checks: its root times its transpose
is ``S`` (so a draw has the right covariance), 20,000 seeded draws have
``S`` as their sample covariance to a stated tolerance, a seed repeats and a
generator advances, and the root a draw builds is not pickled. A Kronecker
covariance on a subset of its grid and a sparse one with many groups never
allocate an ``n x n`` array, and the roundoff and the clearly negative
eigenvalue of a semi-definite covariance are clipped and refused.
"""

import logging
import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from numpy.linalg import LinAlgError

from fips import Constant, Covariance, Exponential, Kronecker, Space, Vector
from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    SparseOperator,
)

N_DRAWS = 20_000


def _spd(rng, k):
    """A well-conditioned symmetric positive-definite matrix."""
    B = rng.standard_normal((k, k))
    return B @ B.T + k * np.eye(k)


def _block_diagonal_csr(blocks):
    """The CSR array with these dense blocks on its diagonal."""
    return sp.block_diag([sp.csr_array(b) for b in blocks], format="csr")


def _correlation(rng, k):
    """A correlation matrix (unit diagonal) that is positive-definite."""
    C = _spd(rng, k)
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def _diagonal():
    return DiagonalOperator(np.random.default_rng(1).uniform(0.5, 2.0, 12))


def _dense():
    return DenseOperator(_spd(np.random.default_rng(2), 12))


def _kronecker_whole():
    """The whole 4 x 5 grid, with the rows in a shuffled order."""
    rng = np.random.default_rng(3)
    order = rng.permutation(20)
    positions = np.unravel_index(order, (4, 5))
    return KroneckerOperator(
        [_correlation(rng, 4), _correlation(rng, 5)],
        std=rng.uniform(0.5, 2.0, 20),
        positions=positions,
    )


def _kronecker_subset(keep=0.6, sizes=(4, 5, 3), seed=4):
    """A random strict subset of a 4 x 5 x 3 grid (three factors), in a random order."""
    rng = np.random.default_rng(seed)
    total = int(np.prod(sizes))
    rows = rng.permutation(total)[: int(keep * total)]
    return KroneckerOperator(
        [_correlation(rng, k) for k in sizes],
        std=rng.uniform(0.5, 2.0, len(rows)),
        positions=np.unravel_index(rows, sizes),
    )


def _sparse_groups():
    """Independent groups of 1 to 5 rows, scattered through the matrix by a permutation."""
    rng = np.random.default_rng(5)
    blocks = [_spd(rng, k) for k in (1, 3, 1, 4, 2, 5, 1, 3)]
    A = _block_diagonal_csr(blocks)
    p = rng.permutation(A.shape[0])
    return SparseOperator(A[p][:, p])


def _block_diagonal():
    """A dense block, a diagonal one and a Kronecker one on a subset of its grid."""
    rng = np.random.default_rng(6)
    return BlockDiagonalOperator(
        [
            DenseOperator(_spd(rng, 6)),
            DiagonalOperator([1.0, 2.0, 3.0]),
            _kronecker_subset(),
        ]
    )


def _constant_block():
    """Independent errors with a ``Constant()`` block of four: rank one, so semi-definite."""
    flux = Space({"flux": pd.Index(list("abc"), name="cell")})
    bias = Space({"bias": pd.Index(list("ABCD"), name="site")})
    cov = Covariance.block_diag(
        Covariance.diagonal([1.0, 2.0, 0.5], flux),
        bias=Constant().with_std(0.7).build(bias),
    )
    assert isinstance(cov.data, BlockDiagonalOperator)
    return cov.data


FORMS = {
    "diagonal": _diagonal,
    "block_diagonal": _block_diagonal,
    "dense": _dense,
    "sparse_groups": _sparse_groups,
    "kronecker_whole": _kronecker_whole,
    "kronecker_subset": _kronecker_subset,
    "constant_semidefinite": _constant_block,
}


def _covariance(form):
    """The form as a Covariance on a space of one block, labeled by an integer."""
    n = form.shape[0]
    space = Space({"x": pd.Index(np.arange(n) * 2 + 10, name="i")})
    return Covariance(form, space)


@pytest.fixture(params=list(FORMS))
def cov(request):
    """A covariance in each form."""
    return _covariance(FORMS[request.param]())


def _dense_root(form):
    """The root of a form as a dense ``n x r`` array."""
    root = form._root
    return root.matmat(np.eye(root.shape[1]))


# ---------------------------------------------------------------------------
# The root, the draws' covariance, the randomness
# ---------------------------------------------------------------------------


def test_the_root_times_its_transpose_is_the_covariance(cov):
    """``R R^T = S`` to 1e-10 of its largest entry, for every form."""
    S = cov.data.dense()
    R = _dense_root(cov.data)
    assert R.shape[0] == len(S)
    np.testing.assert_allclose(R @ R.T, S, rtol=0, atol=1e-10 * np.abs(S).max())


def test_a_kronecker_root_is_n_by_the_whole_grid():
    """A strict subset's root has a column for each point of the whole grid, not for each row."""
    form = _kronecker_subset()
    assert form._root.shape == (form.shape[0], 60)
    assert form.shape[0] < 60
    whole = _kronecker_whole()
    assert whole._root.shape == (20, 20)


def test_the_sample_covariance_of_20000_draws_is_the_covariance(cov):
    """
    Each entry of the sample covariance is within 5 standard errors of ``S``.

    The estimate of ``S_ij`` has standard deviation ``sqrt((S_ii S_jj + S_ij^2) /
    N)``, at most ``sqrt(2 / N) sqrt(S_ii S_jj)``, so in units of
    ``sqrt(S_ii S_jj)`` the error of every entry stays below ``5 sqrt(2 / N)``
    (0.05 for N = 20,000). The generator is seeded, so this is deterministic.
    """
    rng = np.random.default_rng(10)
    draws = np.array([cov.sample(rng).values for _ in range(N_DRAWS)])
    sample = draws.T @ draws / N_DRAWS  # the mean is 0 by construction
    S = cov.data.dense()
    scale = np.sqrt(np.outer(np.diag(S), np.diag(S)))
    assert np.max(np.abs(sample - S) / scale) < 5 * np.sqrt(2 / N_DRAWS)
    assert np.max(np.abs(draws.mean(axis=0)) / np.sqrt(np.diag(S) / N_DRAWS)) < 5


def test_a_draw_is_a_vector_on_the_covariance_space(cov):
    """The draw is a Vector on the space, in its order, and is the root times the normals."""
    draw = cov.sample(rng=3)
    assert isinstance(draw, Vector)
    assert draw.space == cov.space
    root = cov.data._root
    w = np.random.default_rng(3).standard_normal(root.shape[1])
    np.testing.assert_allclose(draw.values, _dense_root(cov.data) @ w, atol=1e-12)


def test_a_seed_repeats_and_a_generator_advances(cov):
    """A seed gives the same draw again; a generator is used as given, so its draws differ."""
    first = cov.sample(rng=7).values
    np.testing.assert_array_equal(first, cov.sample(rng=7).values)
    np.testing.assert_array_equal(
        first, cov.sample(rng=np.random.default_rng(7)).values
    )
    assert not np.array_equal(first, cov.sample(rng=8).values)

    g, h = np.random.default_rng(7), np.random.default_rng(7)
    a, b = cov.sample(g).values, cov.sample(g).values
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, cov.sample(h).values)  # the same stream
    np.testing.assert_array_equal(b, cov.sample(h).values)
    assert not np.array_equal(
        cov.sample().values, cov.sample().values
    )  # None: new entropy


def test_a_zero_variance_draws_zero():
    """A variance of zero is a draw of zero, in the diagonal form and in a sparse group."""
    space = Space({"x": pd.Index(["a", "b", "c"], name="i")})
    diagonal = Covariance.diagonal([0.0, 1.0, 4.0], space)
    np.testing.assert_array_equal(diagonal.sample(1).values[0], 0.0)
    sparse = Covariance(
        SparseOperator(sp.csr_array(np.diag([1.0, 0.0, 4.0]) + 0.0)), space
    )
    assert sparse.sample(1).values[1] == 0.0


def test_a_model_on_a_subset_of_its_grid_draws_from_its_factors():
    """The model a user writes, built on a strict subset of the product, is a Kronecker form drawn through its factors."""
    times = pd.date_range("2024-01-01", periods=6, freq="D")
    cells = pd.MultiIndex.from_product(
        [[40.0, 40.5, 41.0], [-112.0, -111.5]], names=["lat", "lon"]
    )
    full = pd.MultiIndex.from_tuples(
        [("flux", t, *c) for t in times for c in cells],
        names=["block", "time", "lat", "lon"],
    )
    space = Space.from_index(full[np.r_[0:17, 20:36]])  # a gap in the grid
    S = (
        Kronecker(
            Exponential("time", "2D"),
            Exponential(["lat", "lon"], 50.0, metric="haversine"),
        )
        .with_std(0.5)
        .build(space)
    )
    assert isinstance(S.data, KroneckerOperator)
    assert S.data.shape[0] < S.data._size
    R = _dense_root(S.data)
    np.testing.assert_allclose(R @ R.T, S.data.dense(), atol=1e-12)
    assert S.sample(rng=0).values.shape == (S.data.shape[0],)


# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------


def test_the_cached_root_is_not_pickled(cov):
    """A draw keeps its root on the form, but a pickle after it is no bigger and the loaded covariance draws again."""
    before = len(pickle.dumps(cov))
    expected = cov.sample(rng=2).values
    assert "_root" in vars(cov.data)  # kept, so a loop factors once
    assert len(pickle.dumps(cov)) <= before
    loaded = pickle.loads(pickle.dumps(cov))
    assert "_root" not in vars(loaded.data)
    np.testing.assert_array_equal(loaded.sample(rng=2).values, expected)


def test_the_root_is_built_once(cov):
    """The second draw reuses the first one's root."""
    cov.sample(rng=0)
    root = cov.data._root
    cov.sample(rng=1)
    assert cov.data._root is root


# ---------------------------------------------------------------------------
# Memory: nothing n x n
# ---------------------------------------------------------------------------


def test_a_subset_kronecker_draw_allocates_nothing_n_by_n():
    """
    A draw on a strict subset of a Kronecker grid never allocates ``n x n``.

    The grid is 60 x 60 and the form keeps 3,000 of its 3,600 rows, so one
    ``n x n`` array is 72 MB while the draw works on the ``N = 3,600`` normals
    (29 KB) and the factors (29 KB each). The traced peak, with the roots of
    the factors built in it, must stay under half of one ``n x n`` array.
    """
    rng = np.random.default_rng(0)
    rows = np.sort(rng.permutation(3600)[:3000])
    form = KroneckerOperator(
        [_correlation(rng, 60), _correlation(rng, 60)],
        std=rng.uniform(0.5, 2.0, 3000),
        positions=np.unravel_index(rows, (60, 60)),
    )
    n = form.shape[0]
    cov = _covariance(form)
    tracemalloc.start()
    try:
        draw = cov.sample(rng=1)
        cov.sample(rng=2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert draw.values.shape == (n,)
    assert peak < n * n * 8 / 2, (
        f"peak {peak / 1e6:.1f} MB; n x n is {n * n * 8 / 1e6:.0f} MB"
    )
    kept = vars(form._root)
    big = [k for k, v in kept.items() if isinstance(v, np.ndarray) and v.size >= n * n]
    big += [i for i, L in enumerate(form._root._roots) if L.size >= n * n]
    assert not big, big  # the root holds its factors, not a matrix


def test_a_sparse_draw_with_many_groups_stays_sparse():
    """
    A sparse covariance of 2,000 independent groups of five is drawn group by group.

    ``n = 10,000``, so one ``n x n`` array is 800 MB; the root is as sparse as
    the groups (30,000 entries). The traced peak must stay under 5% of one.
    """
    rng = np.random.default_rng(0)
    blocks = [_spd(rng, 5) for _ in range(2000)]
    p = rng.permutation(10_000)
    form = SparseOperator(_block_diagonal_csr(blocks)[p][:, p])
    cov = _covariance(form)
    n = form.shape[0]
    tracemalloc.start()
    try:
        cov.sample(rng=1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert isinstance(form._root, SparseOperator)
    assert form._root.matrix.nnz == 2000 * 15
    assert peak < n * n * 8 * 0.05, f"peak {peak / 1e6:.1f} MB"
    # a draw of one group has that group's covariance: the root squares back to it
    R = form._root.matrix
    np.testing.assert_allclose(
        (R @ R.T)[:5, :5].toarray(), form.matrix[:5, :5].toarray(), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Semi-definite covariances
# ---------------------------------------------------------------------------


def _info_lines(caplog):
    """The INFO records fips logged."""
    return [
        r
        for r in caplog.records
        if r.name.startswith("fips") and r.levelno == logging.INFO
    ]


def test_a_semidefinite_block_is_drawn_from_its_eigendecomposition(caplog):
    """A ``Constant()`` block has no Cholesky factor; it is drawn by ``eigh`` with one INFO line, and the draw is constant over the block."""
    cov = _covariance(_constant_block())
    with caplog.at_level(logging.INFO, logger="fips"):
        draw = cov.sample(rng=0).values
        cov.sample(rng=1)  # the root is kept: nothing more is logged
    assert len(_info_lines(caplog)) == 1
    assert "semi-definite" in _info_lines(caplog)[0].getMessage()
    bias = draw[3:]
    np.testing.assert_allclose(bias, bias[0], atol=1e-12)  # fully correlated, std 0.7


def test_a_roundoff_negative_eigenvalue_is_clipped(caplog):
    """An eigenvalue negative by roundoff (-5e-17) is set to zero, with one INFO line."""
    A = np.array([[1.0, 1.0, 0.0], [1.0, 1.0 - 1.1e-16, 0.0], [0.0, 0.0, 2.0]])
    cov = Covariance(DenseOperator(A), Space({"x": pd.Index(list("abc"), name="i")}))
    with caplog.at_level(logging.INFO, logger="fips"):
        draw = cov.sample(rng=0).values
    assert len(_info_lines(caplog)) == 1
    R = _dense_root(cov.data)
    np.testing.assert_allclose(R @ R.T, A, atol=1e-12)
    assert np.isfinite(draw).all()


def test_a_sparse_group_that_is_semidefinite_logs_once(caplog):
    """Two groups of ones in a sparse covariance are drawn by ``eigh`` and logged as one line."""
    A = _block_diagonal_csr(
        [np.ones((2, 2)), _spd(np.random.default_rng(0), 3), np.ones((3, 3))]
    )
    cov = _covariance(SparseOperator(A))
    with caplog.at_level(logging.INFO, logger="fips"):
        cov.sample(rng=0)
    assert len(_info_lines(caplog)) == 1
    R = cov.data._root.matrix.toarray()
    np.testing.assert_allclose(R @ R.T, A.toarray(), atol=1e-12)


def test_a_roundoff_negative_variance_is_clipped(caplog):
    """A diagonal entry of -1e-20 beside entries near one is drawn as zero, with one INFO line."""
    space = Space({"x": pd.Index(list("abc"), name="i")})
    cov = Covariance.diagonal([1.0, -1e-20, 2.0], space)
    with caplog.at_level(logging.INFO, logger="fips"):
        draw = cov.sample(rng=0).values
    assert draw[1] == 0.0
    assert len(_info_lines(caplog)) == 1


@pytest.mark.parametrize("form", ["dense", "diagonal", "sparse", "block"])
def test_a_clearly_negative_eigenvalue_raises(form):
    """A covariance with an eigenvalue of -1 is not a covariance: the draw raises."""
    dense = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    forms = {
        "dense": DenseOperator(dense),
        "diagonal": DiagonalOperator([1.0, -1.0]),
        "sparse": SparseOperator(sp.csr_array(dense)),
        "block": BlockDiagonalOperator([DiagonalOperator([1.0]), DenseOperator(dense)]),
    }
    cov = _covariance(forms[form])
    with pytest.raises(LinAlgError, match="not positive semi-definite"):
        cov.sample(rng=0)
