"""
Tests for Kronecker covariances kept as factors (#6).

``KroneckerOperator`` against the dense matrix it stands for (``KroneckerError.build``
on the product, and the formula evaluated pair by pair on any other row order or
subset), its products and solves, structures built on an operator
(``from_operator``), and a solve from the factors against the solve from the dense
matrix.
"""

import logging
import pickle
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import fips
from fips import (
    CovarianceMatrix,
    ForwardOperator,
    InverseProblem,
    Matrix,
    MatrixBlock,
    available_estimators,
)
from fips.covariance import KroneckerError, KroneckerOperator
from fips.kernels import GridSpatialDecay, GridTimeDecay
from fips.operator import (
    DenseOperator,
    DiagonalOperator,
    Operator,
    SparseOperator,
    as_operator,
    dottest,
)

TIMES = pd.date_range("2024-01-01", periods=4, freq="D")
LATS = [40.0, 40.5, 41.0]
LONS = [-112.0, -111.5]


def _line_kernel(dim, scale):
    """Exponential decay along one numeric level."""

    def kernel(coords):
        x = coords[dim].to_numpy(dtype=float)
        return np.exp(-np.abs(x[:, None] - x[None, :]) / scale)

    return kernel


#: Two marginals: time, and space as one marginal of (lat, lon).
KERNELS_2 = [
    ("time", GridTimeDecay("1D")),
    (["lat", "lon"], GridSpatialDecay("lat", "lon", 50.0)),
]
#: Three marginals: time, lat and lon separately.
KERNELS_3 = [
    ("time", GridTimeDecay("1D")),
    ("lat", _line_kernel("lat", 0.7)),
    ("lon", _line_kernel("lon", 0.4)),
]
KERNELS = pytest.mark.parametrize("kernels", [KERNELS_2, KERNELS_3], ids=["2", "3"])


def _product(block: bool = True) -> pd.MultiIndex:
    """The full product of the coordinates, time slowest, with a ``block`` level."""
    if not block:
        return pd.MultiIndex.from_product(
            [TIMES, LATS, LONS], names=["time", "lat", "lon"]
        )
    return pd.MultiIndex.from_product(
        [["flux"], TIMES, LATS, LONS], names=["block", "time", "lat", "lon"]
    )


def _variances(index: pd.Index, seed: int = 0) -> pd.Series:
    """Random variances, one per row of `index`."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(0.5, 2.0, len(index)), index=index)


def _direct_formula(kernels, variances: pd.Series, index: pd.Index) -> np.ndarray:
    """``std_i std_j prod_k C_k(row i, row j)``, one pair of rows at a time."""
    frame = index.to_frame(index=False)
    std = np.sqrt(variances.reindex(index).to_numpy())
    n = len(index)
    expected = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            corr = 1.0
            for dims, kernel in kernels:
                dims = [dims] if isinstance(dims, str) else dims
                pair = frame.loc[[i, j], dims].reset_index(drop=True)
                corr *= np.asarray(kernel(pair))[0, 1]
            expected[i, j] = std[i] * std[j] * corr
    return expected


def _ragged(index: pd.MultiIndex) -> pd.MultiIndex:
    """`index` without one cell at one time and another cell at two times."""
    drop = {
        (TIMES[1], 40.5, -112.0),
        (TIMES[0], 41.0, -111.5),
        (TIMES[2], 41.0, -111.5),
    }
    keep = [
        (t, la, lo) not in drop
        for t, la, lo in zip(
            index.get_level_values("time"),
            index.get_level_values("lat"),
            index.get_level_values("lon"),
            strict=True,
        )
    ]
    return index[keep]


def _spd_factor(rng, k: int) -> np.ndarray:
    """A random correlation-like SPD matrix of size `k`."""
    B = rng.standard_normal((k, k))
    A = B @ B.T + k * np.eye(k)
    d = np.sqrt(np.diag(A))
    return A / d[:, None] / d[None, :]


# ---------------------------------------------------------------------------
# The operator against the dense matrix
# ---------------------------------------------------------------------------


class TestAgainstBuild:
    """``KroneckerError.operator(index).dense()`` is the covariance ``build`` gives."""

    @KERNELS
    @pytest.mark.parametrize("variances", ["series", "scalar"])
    def test_product_order_is_build_exactly(self, kernels, variances):
        """On the product in the kernels' order, the same numbers as ``build``, bit for bit."""
        index = _product()
        var = _variances(index) if variances == "series" else 1.7
        error = KroneckerError("prior", var, kernels)
        op = error.operator(index)
        assert isinstance(op, KroneckerOperator)
        assert op._rows is None  # the whole grid, in order: no scatter
        np.testing.assert_array_equal(op.dense(), error.build(index).to_numpy())

    @KERNELS
    def test_same_cells_dropped_at_every_time(self, kernels):
        """A product of the cells that remain (what slv builds) is still a whole grid."""
        index = _product()
        lat, lon = index.get_level_values("lat"), index.get_level_values("lon")
        subset = index[~((lat == 40.5) & (lon == -112.0))]
        var = _variances(index)
        error = KroneckerError("prior", var, kernels)
        op = error.operator(subset)
        if kernels is KERNELS_2:
            assert op._rows is None
            np.testing.assert_array_equal(op.dense(), error.build(subset).to_numpy())
        np.testing.assert_allclose(
            op.dense(), _direct_formula(kernels, var, subset), rtol=1e-14, atol=0
        )

    @KERNELS
    @pytest.mark.parametrize("seed", range(3))
    def test_permuted_rows(self, kernels, seed):
        """Shuffled rows give the same entries, label by label."""
        index = _product()
        var = _variances(index)
        error = KroneckerError("prior", var, kernels)
        expected = pd.DataFrame(
            error.build(index).to_numpy(), index=index, columns=index
        )
        shuffled = index[np.random.default_rng(seed).permutation(len(index))]
        op = error.operator(shuffled)
        np.testing.assert_allclose(
            op.dense(),
            expected.reindex(index=shuffled, columns=shuffled).to_numpy(),
            rtol=0,
            atol=1e-15,
        )

    @KERNELS
    def test_space_first_order(self, kernels):
        """An index with its levels and rows ordered by space first."""
        index = _product(block=False)
        order = ["lat", "lon", "time"]
        var = _variances(index)
        var.index = var.index.reorder_levels(order)
        other = index.reorder_levels(order).sort_values()
        op = KroneckerError("prior", var, kernels).operator(other)
        np.testing.assert_allclose(
            op.dense(), _direct_formula(kernels, var, other), rtol=1e-14, atol=0
        )

    @KERNELS
    def test_ragged_subset(self, kernels):
        """Cells missing at some times only: a strict subset of the product."""
        index = _ragged(_product())
        var = _variances(_product())
        op = KroneckerError("prior", var, kernels).operator(index)
        assert op._rows is not None and op.shape == (len(index), len(index))
        np.testing.assert_allclose(
            op.dense(), _direct_formula(kernels, var, index), rtol=1e-14, atol=0
        )

    def test_variances_on_some_levels(self):
        """A variance Series indexed by (lat, lon) is aligned onto every time."""
        cells = pd.MultiIndex.from_product([LATS, LONS], names=["lat", "lon"])
        cell_var = pd.Series(np.arange(1.0, 7.0), index=cells, name="var")
        index = _product(block=False)
        error = KroneckerError("prior", cell_var, KERNELS_2)
        np.testing.assert_array_equal(
            error.operator(index).dense(), error.build(index).to_numpy()
        )

    def test_diagonal_is_the_dense_diagonal(self):
        """``diagonal`` reads the variances without the matrix, the same numbers."""
        index = _ragged(_product())
        op = KroneckerError("prior", _variances(_product()), KERNELS_3).operator(index)
        np.testing.assert_array_equal(op.diagonal(), np.diag(op.dense()))
        assert op.trace() == pytest.approx(np.trace(op.dense()), rel=1e-15)

    def test_repeated_coordinates_raise(self):
        """Two rows on the same coordinates would be perfectly correlated."""
        index = pd.MultiIndex.from_product(
            [["a", "b"], TIMES, LATS, LONS], names=["block", "time", "lat", "lon"]
        )
        with pytest.raises(ValueError, match="repeat"):
            KroneckerError("prior", 1.0, KERNELS_2).operator(index)

    def test_kernel_of_wrong_shape_raises(self):
        """A kernel must return one row per unique coordinate."""
        kernels = [("time", GridTimeDecay("1D")), (["lat", "lon"], lambda c: np.eye(4))]
        with pytest.raises(ValueError, match="shape"):
            KroneckerError("prior", 1.0, kernels).operator(_product())

    def test_no_kernel_raises(self):
        """At least one marginal kernel."""
        with pytest.raises(ValueError, match="at least one"):
            KroneckerError("prior", 1.0, []).operator(_product())


class TestConstructor:
    """``KroneckerOperator`` refuses what it cannot stand for."""

    def test_full_product_by_default(self):
        """Without positions, the rows are the product in ``np.kron``'s order."""
        rng = np.random.default_rng(0)
        A, B = _spd_factor(rng, 3), _spd_factor(rng, 4)
        std = rng.uniform(0.5, 2, 12)
        op = KroneckerOperator([A, B], std=std)
        np.testing.assert_allclose(
            op.dense(), std[:, None] * np.kron(A, B) * std[None, :], rtol=1e-15
        )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"factors": []}, "at least one factor"),
            ({"factors": [np.ones((2, 3))]}, "square"),
            ({"factors": [np.array([[1.0, np.nan], [np.nan, 1.0]])]}, "finite"),
            ({"factors": [np.eye(2)], "std": [1.0, 2.0, 3.0]}, "entries"),
            ({"factors": [np.eye(2)], "std": -1.0}, "not negative"),
            ({"factors": [np.eye(2)], "positions": [[0], [1]]}, "positions for"),
            ({"factors": [np.eye(2)], "positions": [[0, 2]]}, "outside"),
            ({"factors": [np.eye(2)], "positions": [[0.0, 1.0]]}, "integers"),
            (
                {"factors": [np.eye(2), np.eye(2)], "positions": [[0, 1], [0]]},
                "one length",
            ),
            ({"factors": [np.eye(2)], "positions": [[1, 1]]}, "perfectly correlated"),
        ],
    )
    def test_invalid(self, kwargs, match):
        """Each invalid input raises a ValueError that says what is wrong."""
        with pytest.raises(ValueError, match=match):
            KroneckerOperator(**kwargs)


# ---------------------------------------------------------------------------
# Products, solves, and the rest of the Operator interface
# ---------------------------------------------------------------------------


def _operators():
    """A Kronecker operator on the whole product, on a permutation, and on a ragged subset."""
    rng = np.random.default_rng(3)
    factors = [_spd_factor(rng, 4), _spd_factor(rng, 3), _spd_factor(rng, 2)]
    sizes = (4, 3, 2)
    whole = np.arange(24)
    permuted = rng.permutation(24)
    ragged = np.sort(rng.permutation(24)[:19])
    out = {}
    for name, rows in [("whole", whole), ("permuted", permuted), ("ragged", ragged)]:
        std = rng.uniform(0.5, 2.0, len(rows))
        out[name] = KroneckerOperator(
            factors, std=std, positions=np.unravel_index(rows, sizes)
        )
    return out


OPS = _operators()
FORMS = pytest.mark.parametrize("which", list(OPS))


@FORMS
def test_products_with_arrays(which):
    """``matvec``, ``matmat``, ``rmatvec``, ``rmatmat`` and ``@`` from both sides."""
    op = OPS[which]
    S = op.dense()
    n = op.shape[0]
    rng = np.random.default_rng(1)
    x, X = rng.standard_normal(n), rng.standard_normal((n, 5))
    np.testing.assert_allclose(op.matvec(x), S @ x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(op.matmat(X), S @ X, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(op.rmatvec(x), S.T @ x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(op.rmatmat(X), S.T @ X, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(op @ X, S @ X, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(X.T @ op, X.T @ S, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(op.T.dense(), S.T, rtol=0, atol=1e-15)
    assert dottest(op)


@FORMS
def test_products_in_column_blocks(which, monkeypatch):
    """A product with more columns than one block holds runs block by block."""
    monkeypatch.setattr("fips.covariance._CHUNK", 50)  # two columns at a time
    op = OPS[which]
    X = np.random.default_rng(2).standard_normal((op.shape[0], 7))
    np.testing.assert_allclose(op @ X, op.dense() @ X, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(op.solve(X), np.linalg.solve(op.dense(), X), rtol=1e-10)


@FORMS
@pytest.mark.parametrize("rhs", ["vector", "matrix", "sparse", "operator"])
def test_solve(which, rhs):
    """``solve`` equals the dense solve at 1e-10, a whole grid or a ragged subset."""
    op = OPS[which]
    S = op.dense()
    n = op.shape[0]
    rng = np.random.default_rng(4)
    B = {
        "vector": rng.standard_normal(n),
        "matrix": rng.standard_normal((n, 3)),
        "sparse": sp.random(n, 3, density=0.4, random_state=5, format="csr"),
        "operator": DenseOperator(rng.standard_normal((n, 2))),
    }[rhs]
    expected = np.linalg.solve(S, np.asarray(B.todense() if sp.issparse(B) else B))
    actual = op.solve(B)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_solve_keeps_the_factorization():
    """Each factor is factored once, whatever the number of solves."""
    op = KroneckerOperator(
        OPS["ragged"].factors, OPS["ragged"].std, OPS["ragged"].positions
    )
    op.solve(np.ones(op.shape[0]))
    inverses, complement = op._inverses, op._complement
    op.solve(np.ones(op.shape[0]))
    assert op._inverses is inverses and op._complement is complement


def test_solve_of_a_thin_subset_is_dense(caplog):
    """When the rows fill too little of the product, the dense solve is used, and logged."""
    rng = np.random.default_rng(6)
    factors = [_spd_factor(rng, 6), _spd_factor(rng, 5)]
    rows = rng.permutation(30)[:10]  # N d = 30 * 20 > n^2 = 100
    op = KroneckerOperator(factors, std=2.0, positions=np.unravel_index(rows, (6, 5)))
    b = rng.standard_normal(10)
    with caplog.at_level(logging.INFO, logger="fips.covariance"):
        x = op.solve(b)
    assert "densely" in caplog.text
    assert "_complement" not in vars(op)
    np.testing.assert_allclose(x, np.linalg.solve(op.dense(), b), rtol=1e-10)


def _gauss(n: int, scale: float) -> np.ndarray:
    """A Gaussian correlation on a line, badly conditioned for a wide scale."""
    t = np.arange(n, dtype=float)
    return np.exp(-(((t[:, None] - t[None, :]) / scale) ** 2)) + 1e-8 * np.eye(n)


def _backward_error(S: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """The normwise backward error of `x` as a solution of ``S x = b``."""
    r = np.linalg.norm(S @ x - b)
    return r / (np.linalg.norm(S, 2) * np.linalg.norm(x) + np.linalg.norm(b))


def test_subset_solve_on_ill_conditioned_factors_falls_back(caplog):
    """When the Schur complement loses accuracy, the residual check solves the dense matrix."""
    rng = np.random.default_rng(0)
    keep = np.sort(rng.permutation(900)[:700])
    op = KroneckerOperator(
        [_gauss(30, 3.0), _gauss(30, 3.0)], positions=np.unravel_index(keep, (30, 30))
    )
    S, b = op.dense(), rng.standard_normal(700)
    with caplog.at_level(logging.WARNING, logger="fips.covariance"):
        x = op.solve(b)
    assert "lost accuracy" in caplog.text
    eps = np.finfo(float).eps
    assert _backward_error(S, x, b) < 700 * eps
    # The same factors on the whole grid, and a well-conditioned subset: no fallback
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="fips.covariance"):
        KroneckerOperator([_gauss(30, 3.0), _gauss(30, 3.0)]).solve(np.ones(900))
        good = KroneckerOperator(
            [_gauss(30, 1.0), _gauss(30, 1.0)],
            positions=np.unravel_index(keep, (30, 30)),
        )
        x = good.solve(b)
    assert "lost accuracy" not in caplog.text
    np.testing.assert_allclose(x, np.linalg.solve(good.dense(), b), rtol=1e-10)


def test_solve_with_a_zero_std_raises():
    """A zero standard deviation makes the operator singular."""
    op = KroneckerOperator([np.eye(2)], std=[1.0, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="zero"):
        op.solve(np.ones(2))


def test_solve_of_a_factor_off_positive_definite_falls_back(caplog):
    """A factor that roundoff tips off positive-definite is solved by LDL^T, with a warning."""
    C = np.array([[1.0, 1.0 + 1e-9], [1.0 + 1e-9, 1.0]])  # indefinite, invertible
    op = KroneckerOperator([C, np.eye(2)])
    b = np.arange(1.0, 5.0)
    with caplog.at_level(logging.WARNING, logger="fips.operator"):
        x = op.solve(b)
    assert "non-positive-definite" in caplog.text
    np.testing.assert_allclose(op @ x, b, rtol=0, atol=1e-5)


@FORMS
def test_scalar_arithmetic(which):
    """A positive scalar keeps the factors; a negative one gives the dense matrix."""
    op = OPS[which]
    S = op.dense()
    for result, expected in [
        (op * 4.0, 4.0 * S),
        (2.0 * op, 2.0 * S),
        (op / 4.0, S / 4.0),
    ]:
        assert isinstance(result, KroneckerOperator)
        np.testing.assert_allclose(result.dense(), expected, rtol=1e-14)
    assert isinstance(-op, DenseOperator)
    np.testing.assert_array_equal((-op).dense(), -S)


def test_to_frame_and_pickle():
    """``to_frame`` is the dense matrix labeled; a pickle keeps the factors, not the caches."""
    op = OPS["ragged"]
    op.solve(np.ones(op.shape[0]))  # fill the caches
    idx = pd.RangeIndex(op.shape[0], name="row")
    frame = op.to_frame(idx)
    np.testing.assert_array_equal(frame.to_numpy(), op.dense())
    assert frame.index.equals(idx) and frame.columns.equals(idx)
    state = op.__getstate__()
    assert "_inverses" not in state and "_complement" not in state
    loaded = pickle.loads(pickle.dumps(op))
    assert isinstance(loaded, KroneckerOperator)
    np.testing.assert_array_equal(loaded.dense(), op.dense())
    np.testing.assert_array_equal(
        loaded.solve(np.ones(op.shape[0])), op.solve(np.ones(op.shape[0]))
    )


def _others(n: int, m: int = 5):
    """A dense, a sparse and a diagonal operator to multiply and add with."""
    rng = np.random.default_rng(7)
    H = rng.standard_normal((m, n))
    H[np.abs(H) < 1.0] = 0.0
    return {
        "dense H": DenseOperator(H),
        "sparse H": SparseOperator(sp.csr_matrix(H)),
        "diagonal": DiagonalOperator(rng.uniform(1, 2, n)),
        "dense square": DenseOperator(rng.standard_normal((n, n))),
        "kronecker": OPS["whole"],
    }


@pytest.mark.parametrize("other", ["dense H", "sparse H"])
@FORMS
def test_forward_products(which, other):
    """``H @ S_0`` and ``H @ S_0 @ H.T`` are right, dense, and never form ``S_0``."""
    op = OPS[which]
    H = _others(op.shape[0])[other]
    S, Hd = op.dense(), H.dense()
    HS = H @ op
    assert isinstance(HS, DenseOperator)
    np.testing.assert_allclose(HS.dense(), Hd @ S, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        (HS @ H.T).dense(), Hd @ S @ Hd.T, rtol=1e-12, atol=1e-12
    )
    SH = op @ H.T
    assert isinstance(SH, DenseOperator)
    np.testing.assert_allclose(SH.dense(), S @ Hd.T, rtol=1e-13, atol=1e-13)


def test_forward_products_never_form_the_matrix(monkeypatch):
    """The products with ``H`` go through the factors."""
    op = OPS["ragged"]
    S = op.dense()
    H = _others(op.shape[0])
    monkeypatch.setattr(KroneckerOperator, "dense", _refuse)
    for name in ("dense H", "sparse H"):
        np.testing.assert_allclose(
            (H[name] @ op).dense(), H[name].dense() @ S, rtol=1e-13, atol=1e-13
        )


@pytest.mark.parametrize("other", ["diagonal", "dense square", "kronecker"])
def test_products_with_square_forms(other):
    """With another ``n x n`` form the product is dense, and right from both sides."""
    op = OPS["whole"]
    B = _others(op.shape[0])[other]
    for result, expected in [
        (op @ B, op.dense() @ B.dense()),
        (B @ op, B.dense() @ op.dense()),
    ]:
        assert isinstance(result, DenseOperator)
        np.testing.assert_allclose(result.dense(), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("other", ["diagonal", "dense square", "kronecker"])
def test_sums(other):
    """A sum with a Kronecker side is dense: no form holds such a sum."""
    op = OPS["whole"]
    B = _others(op.shape[0])[other]
    for result, expected in [
        (op + B, op.dense() + B.dense()),
        (B + op, B.dense() + op.dense()),
        (op - B, op.dense() - B.dense()),
    ]:
        assert isinstance(result, DenseOperator)
        np.testing.assert_allclose(result.dense(), expected, rtol=1e-14)


@FORMS
def test_select(which):
    """The same rows and columns keep the factors; others give the entries."""
    op = OPS[which]
    S = op.dense()
    rng = np.random.default_rng(8)
    rows = rng.permutation(op.shape[0])[:9]
    picked = op._select(rows, rows)
    assert isinstance(picked, KroneckerOperator)
    np.testing.assert_array_equal(picked.dense(), S[np.ix_(rows, rows)])
    cols = rows[::-1]
    np.testing.assert_array_equal(op._select(rows, cols).dense(), S[np.ix_(rows, cols)])


@pytest.mark.parametrize("form", ["dense", "sparse", "diagonal"])
def test_select_other_forms(form):
    """``_select`` keeps a sparse or diagonal operator sparse or diagonal."""
    rng = np.random.default_rng(9)
    values = rng.uniform(1, 2, 5)
    A = np.diag(values)
    A[0, 3] = A[3, 0] = 0.5 if form != "diagonal" else 0.0
    op = {
        "dense": DenseOperator(A),
        "sparse": SparseOperator(sp.csr_matrix(A)),
        "diagonal": DiagonalOperator(values),
    }[form]
    rows = np.array([4, 0, 3])
    picked = op._select(rows, rows)
    assert type(picked) is type(op)
    np.testing.assert_array_equal(picked.dense(), A[np.ix_(rows, rows)])


# ---------------------------------------------------------------------------
# A structure built on an operator
# ---------------------------------------------------------------------------


def _refuse(*args, **kwargs):
    """Stand in for a method that would form the n x n matrix."""
    raise AssertionError("the dense matrix was formed")


@pytest.fixture
def no_dense(monkeypatch):
    """Fail if anything forms a Kronecker operator's dense matrix (or a frame from it)."""
    for name in ("dense", "to_frame", "__array__"):
        monkeypatch.setattr(KroneckerOperator, name, _refuse)


def _kron_cov(index=None, kernels=KERNELS_2, var=None, name="prior_error"):
    """A covariance built on the operator of a Kronecker error on `index`."""
    index = _product() if index is None else index
    var = _variances(_product()) if var is None else var
    op = KroneckerError(name, var, kernels).operator(index)
    return CovarianceMatrix.from_operator(op, index, name=name), op


class TestFromOperator:
    """``Structure2D.from_operator`` keeps the operator until the DataFrame is asked for."""

    def test_labels_shape_and_operator(self, no_dense):
        """The labels, shape, operator and variances, without the matrix."""
        S, op = _kron_cov()
        idx = _product()
        assert S.index.equals(idx) and S.columns.equals(idx)
        assert S.shape == (len(idx), len(idx)) and len(S) == len(idx)
        assert S.operator is op
        assert not S.is_sparse
        assert S.name == "prior_error"
        repr(S)
        np.testing.assert_array_equal(S.variances.values, op.diagonal())

    def test_data_forms_the_frame_once(self, monkeypatch):
        """``data`` forms the frame (a warning past the size, once), and the structure then holds it."""
        S, op = _kron_cov()
        monkeypatch.setattr("fips.base._FRAME_WARN_SIZE", 10)
        with pytest.warns(UserWarning, match="Forming the 24 x 24 DataFrame"):
            frame = S.data
        np.testing.assert_array_equal(frame.to_numpy(), op.dense())
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no second warning
            assert S.data is frame  # held
        assert isinstance(S.operator, DenseOperator)
        np.testing.assert_array_equal(S.values, op.dense())

    def test_everything_that_reads_data_works(self):
        """``values``, ``xs``, a block, ``to_frame``, ``force_symmetry``, a sum: as for a frame."""
        S, op = _kron_cov()
        D = op.dense()
        ref = CovarianceMatrix(pd.DataFrame(D, index=_product(), columns=_product()))
        np.testing.assert_array_equal(_kron_cov()[0].values, D)
        pd.testing.assert_frame_equal(
            _kron_cov()[0]["flux", "flux"], ref["flux", "flux"]
        )
        pd.testing.assert_frame_equal(_kron_cov()[0].to_frame(), ref.to_frame())
        np.testing.assert_allclose(
            _kron_cov()[0].force_symmetry().values, ref.force_symmetry().values
        )
        total = S + ref
        np.testing.assert_allclose(total.values, 2 * D)

    def test_labels_are_checked(self):
        """An unnamed level, a missing ``block`` level and a shape mismatch raise."""
        op = OPS["whole"]
        unnamed = pd.MultiIndex.from_product(
            [["flux"], range(24)], names=["block", None]
        )
        with pytest.raises(ValueError, match="must be named"):
            CovarianceMatrix.from_operator(op, unnamed)
        no_block = pd.Index(range(24), name="cell")
        with pytest.raises(ValueError, match="'block' level"):
            CovarianceMatrix.from_operator(op, no_block)
        short = pd.MultiIndex.from_product([["flux"], range(23)], names=["block", "c"])
        with pytest.raises(ValueError, match="does not match"):
            CovarianceMatrix.from_operator(op, short)

    def test_constructor_points_to_from_operator(self):
        """An operator given to a constructor raises and names ``from_operator``."""
        with pytest.raises(TypeError, match="from_operator"):
            CovarianceMatrix(OPS["whole"])

    def test_labels_are_sanitized(self, no_dense):
        """The ``block`` level moves first and numeric strings become numbers, as for a frame."""
        idx = pd.MultiIndex.from_product(
            [["1", "2"], ["flux"]], names=["cell", "block"]
        )
        S = CovarianceMatrix.from_operator(KroneckerOperator([np.eye(2)]), idx)
        assert list(S.index.names) == ["block", "cell"]
        assert list(S.index.get_level_values("cell")) == [1, 2]
        assert S.columns.equals(S.index)

    def test_reindex_and_round_keep_the_operator(self, no_dense):
        """A permutation or a subset of the labels picks rows of the operator, labels rounded."""
        S, op = _kron_cov()
        idx = _product()
        target = idx[np.random.default_rng(0).permutation(len(idx))][:20]
        R = S.round_index(6, axis="both").reindex(
            index=target, columns=target, fill_value=0.0, verify_overlap=True
        )
        assert isinstance(R.operator, KroneckerOperator)
        assert R.index.equals(target) and R.columns.equals(target)
        same = S.reindex(index=idx, columns=idx)
        assert same.operator is op

    def test_reindex_values(self):
        """The reindexed operator holds the reindexed entries."""
        S, op = _kron_cov()
        idx = _product()
        target = idx[np.random.default_rng(1).permutation(len(idx))][:20]
        R = S.reindex(index=target, columns=target)
        expected = pd.DataFrame(op.dense(), index=idx, columns=idx).reindex(
            index=target, columns=target
        )
        np.testing.assert_array_equal(R.operator.dense(), expected.to_numpy())

    def test_reindex_with_levels_in_another_order(self):
        """A target with the levels in another order is matched label by label, overlap checked."""
        S, op = _kron_cov()
        target = (
            _product().reorder_levels(["block", "lat", "lon", "time"]).sort_values()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # full overlap: no warning
            R = S.reindex(index=target, columns=target, verify_overlap=True)
        assert isinstance(R.operator, KroneckerOperator) and R._data is None
        assert R.index.equals(target)
        ref = CovarianceMatrix(
            pd.DataFrame(op.dense(), index=_product(), columns=_product())
        )
        expected = ref.reindex(index=target, columns=target, verify_overlap=True)
        np.testing.assert_array_equal(R.values, expected.values)

    def test_problem_with_levels_in_another_order(self):
        """``InverseProblem`` takes a factored covariance whose levels are in another order than the prior's."""
        S, op = _kron_cov()
        inputs = _problem_inputs(_product(), sparse_h=True)
        order = ["block", "lat", "lon", "time"]
        S_other = CovarianceMatrix.from_operator(op, _product().reorder_levels(order))
        dense = CovarianceMatrix(
            pd.DataFrame(op.dense(), index=_product(), columns=_product())
        )
        post = fips.solve(InverseProblem(prior_error=S_other, **inputs))
        ref = fips.solve(InverseProblem(prior_error=dense, **inputs))
        assert isinstance(post.estimator._S_0_op, KroneckerOperator)
        _close(post.estimate.values, ref.estimate.values, "estimate")

    def test_reindex_with_a_new_label_forms_the_frame(self):
        """A label the operator does not have: the DataFrame path, which fills zeros and warns."""
        S, op = _kron_cov()
        idx = _product()
        extra = pd.MultiIndex.from_tuples(
            [("flux", TIMES[0], 39.0, -112.0)], names=idx.names
        )
        target = idx.append(extra)
        with pytest.warns(UserWarning, match="Partial overlap"):
            R = S.reindex(
                index=target, columns=target, fill_value=0.0, verify_overlap=True
            )
        assert isinstance(R.operator, DenseOperator)
        np.testing.assert_array_equal(R.values[:-1, :-1], op.dense())
        assert not R.values[-1].any()

    def test_reindex_level_mismatch_raises(self):
        """A target with other levels raises, as for a frame."""
        S, _ = _kron_cov()
        other = pd.MultiIndex.from_product([["flux"], range(24)], names=["block", "x"])
        with pytest.raises(ValueError, match="Cannot reindex"):
            S.reindex(index=other, columns=other)

    def test_inplace(self, no_dense):
        """``reindex`` and ``round_index`` in place keep the operator."""
        S, op = _kron_cov()
        target = _product()[::-1]
        assert S.reindex(index=target, columns=target, inplace=True) is S
        assert S.index.equals(target) and isinstance(S.operator, KroneckerOperator)
        assert S.round_index(3, axis="both", inplace=True) is S
        assert S._op is not None

    def test_copy_scale_and_matrix_of_it(self, no_dense):
        """``copy``, ``scale`` and ``CovarianceMatrix(S)`` keep the operator."""
        S, op = _kron_cov()
        assert isinstance(S.copy().operator, KroneckerOperator)
        assert S.copy(deep=False).operator is op
        assert isinstance(S.scale(2.0).operator, KroneckerOperator)
        np.testing.assert_allclose(S.scale(2.0).variances.values, 2 * op.diagonal())
        again = CovarianceMatrix(S, name="again")
        assert again.operator is op and again.name == "again"
        assert Matrix(S).operator is op

    def test_pickle(self, no_dense, tmp_path):
        """A pickle keeps the operator and the labels, not a frame."""
        S, op = _kron_cov()
        loaded = pickle.loads(pickle.dumps(S))
        assert isinstance(loaded.operator, KroneckerOperator)
        assert loaded.index.equals(S.index) and loaded.name == S.name
        path = tmp_path / "S.pkl"
        S.to_file(path)
        assert CovarianceMatrix.from_file(path).index.equals(S.index)

    def test_pickle_of_a_frame_is_unchanged(self):
        """A structure that holds a frame pickles as before: data and name only."""
        S = CovarianceMatrix.diagonal(1.0, index=_product())
        assert set(S.__getstate__()) == {"data", "name"}

    def test_matrix_block(self, no_dense):
        """A block is built on an operator too, with its block names."""
        idx = _product(block=False)
        op = KroneckerError("prior", 1.0, KERNELS_2).operator(idx)
        B = MatrixBlock.from_operator(op, idx, row_block="flux", col_block="flux")
        assert B.row_block == "flux" and B.operator is op
        assert MatrixBlock(B).operator is op
        loaded = pickle.loads(pickle.dumps(B))
        assert loaded.row_block == "flux" and loaded.operator is not None
        with pytest.raises(ValueError, match="'block' level"):
            MatrixBlock.from_operator(
                OPS["whole"], _product()[:24], row_block="flux", col_block="flux"
            )

    def test_repr_does_not_form_the_frame(self, no_dense):
        """``repr`` of a block or a matrix built on an operator shows the operator, not a frame."""
        idx = _product(block=False)
        op = KroneckerError("prior", 1.0, KERNELS_2).operator(idx)
        B = MatrixBlock.from_operator(op, idx, row_block="flux", col_block="flux")
        text = repr(B)
        assert "KroneckerOperator" in text and "shape=(24, 24)" in text
        assert B._data is None and B.operator is op
        S, _ = _kron_cov()
        repr(S)
        assert S._data is None

    @pytest.mark.parametrize("form", ["array", "dense", "sparse", "diagonal"])
    def test_nan_raises_as_for_a_frame(self, form):
        """An operator that stores a NaN is refused, as a frame with one is."""
        idx = pd.MultiIndex.from_product([["flux"], [0, 1]], names=["block", "cell"])
        A = np.array([[1.0, np.nan], [np.nan, 1.0]])
        op = {
            "array": A,
            "dense": DenseOperator(A),
            "sparse": SparseOperator(sp.csr_matrix(A)),
            "diagonal": DiagonalOperator([1.0, np.nan]),
        }[form]
        with pytest.raises(ValueError, match="NaN"):
            CovarianceMatrix.from_operator(op, idx)
        with pytest.raises(ValueError, match="NaN"):
            CovarianceMatrix(pd.DataFrame(A, index=idx, columns=idx))

    def test_sparse_operator_behind_labels(self):
        """A sparse operator stays sparse, and its frame is a sparse frame."""
        idx = _product()
        D = DiagonalOperator(np.arange(1.0, 25.0))
        S = CovarianceMatrix.from_operator(D, idx)
        assert S.is_sparse and S.operator is D
        assert S.data.dtypes.map(lambda d: isinstance(d, pd.SparseDtype)).all()
        np.testing.assert_array_equal(S.values.toarray(), np.diag(np.arange(1.0, 25.0)))


# ---------------------------------------------------------------------------
# The solve, from the factors against the dense matrix
# ---------------------------------------------------------------------------

ESTIMATOR_OUTPUTS = [
    "x_hat", "S_hat", "K", "A", "y_hat", "y_0", "S_y0", "desroziers", "U_red",
    "DOFS", "reduced_chi2", "R2", "RMSE", "uncertainty_reduction",
]  # fmt: skip
POSTERIOR_OUTPUTS = [
    "estimate", "mean", "covariance", "variances", "modeled_obs", "prior_modeled_obs",
    "prior_obs_covariance", "kalman_gain", "averaging_kernel", "desroziers",
    "dofs", "reduced_chi2", "r2", "rmse", "uncertainty_reduction",
]  # fmt: skip


def _close(actual, expected, what):
    """Compare at 1e-10 relative, with an absolute floor of 1e-10 of the largest entry."""
    expected = np.asarray(expected, dtype=float)
    floor = 1e-10 * max(np.abs(expected).max(), 1.0)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), expected, rtol=1e-10, atol=floor, err_msg=what
    )


def _problem_inputs(state_index, sparse_h: bool, seed: int = 11):
    """The obs, prior, H, S_z and constant of a small problem on `state_index`."""
    rng = np.random.default_rng(seed)
    n, m = len(state_index), 15
    obs_index = pd.MultiIndex.from_product(
        [["conc"], [f"o{k}" for k in range(m)]], names=["block", "obs"]
    )
    H = np.zeros((m, n))
    for i in range(m):
        H[i, rng.choice(n, 4, replace=False)] = rng.uniform(0.5, 2.0, 4)
    x_0 = rng.normal(5.0, 1.0, n)
    z = H @ x_0 + rng.normal(0, 1.0, m)
    return {
        "obs": pd.Series(z, index=obs_index),
        "prior": pd.Series(x_0, index=state_index),
        "forward_operator": ForwardOperator(
            pd.DataFrame(H, index=obs_index, columns=state_index), sparse=sparse_h
        ),
        "modeldata_mismatch": CovarianceMatrix.diagonal(
            rng.uniform(0.1, 0.4, m), index=obs_index
        ),
        "constant": pd.Series(rng.normal(0.5, 0.1, m), index=obs_index),
    }


def _compare(post, ref):
    """Every estimator and Posterior output of `post` against `ref`, at 1e-10."""
    for name in ESTIMATOR_OUTPUTS:
        _close(getattr(post.estimator, name), getattr(ref.estimator, name), name)
    for name in POSTERIOR_OUTPUTS:
        value, expected = getattr(post, name), getattr(ref, name)
        if callable(value):  # averaging_kernel, with no aggregation
            value, expected = value(), expected()
        if hasattr(value, "to_dense"):  # a labeled structure
            assert value.index.equals(expected.index), name
            value, expected = value.to_dense().values, expected.to_dense().values
        elif hasattr(value, "index"):
            assert value.index.equals(expected.index), name
            value, expected = value.values, expected.values
        _close(value, expected, f"posterior {name}")
    x = np.linspace(1.0, 2.0, post.estimator.n_x)
    _close(post.estimator.cost(x), ref.estimator.cost(x), "cost")


@pytest.mark.parametrize("estimator", available_estimators())
@pytest.mark.parametrize("sparse_h", [False, True], ids=["dense H", "sparse H"])
@pytest.mark.parametrize("shape", ["product", "permuted", "ragged"])
@KERNELS
def test_solve_matches_the_dense_build(estimator, sparse_h, shape, kernels):
    """
    A problem whose prior error is built on the factors solves as the dense one does.

    ``product``: the prior error from ``build`` is the reference. ``permuted``: the
    covariance is built on the product but the prior comes in a shuffled order, so
    the problem reorders the operator. ``ragged``: cells missing at some times
    (the dense reference is the operator's own matrix, which ``build`` cannot
    give on this index).
    """
    index = _product()
    var = _variances(index)
    error = KroneckerError("prior_error", var, kernels)
    if shape == "ragged":
        index = _ragged(index)
    op = error.operator(index)
    dense = op.dense() if shape == "ragged" else error.build(index).to_numpy()
    if shape == "product":
        np.testing.assert_array_equal(op.dense(), dense)
    state = index
    if shape == "permuted":
        state = index[np.random.default_rng(5).permutation(len(index))]
    inputs = _problem_inputs(state, sparse_h)

    S_factored = CovarianceMatrix.from_operator(op, index)
    S_dense = CovarianceMatrix(pd.DataFrame(dense, index=index, columns=index))
    options = {"gamma": 0.7} if estimator == "bayesian" else {}
    post = fips.solve(
        InverseProblem(prior_error=S_factored, **inputs), estimator, **options
    )
    ref = fips.solve(
        InverseProblem(prior_error=S_dense, **inputs), estimator, **options
    )
    assert isinstance(post.estimator._S_0_op, KroneckerOperator)
    _compare(post, ref)


def test_operator_given_to_the_problem():
    """An operator as ``prior_error`` is labeled by the prior's index."""
    index = _product()
    op = KroneckerError("prior_error", _variances(index), KERNELS_2).operator(index)
    inputs = _problem_inputs(index, sparse_h=True)
    problem = InverseProblem(prior_error=op, **inputs)
    assert problem.prior_error.operator is op
    assert problem.prior_error.index.equals(problem.prior.index)
    ref = InverseProblem(
        prior_error=CovarianceMatrix.from_operator(op, index), **inputs
    )
    np.testing.assert_array_equal(
        fips.solve(problem).estimate.values, fips.solve(ref).estimate.values
    )


def test_solve_never_forms_the_prior_error(no_dense):
    """
    Building the problem and solving it never forms the ``n x n`` prior error.

    The prior comes in another row order with float labels, so the problem
    rounds and reorders the covariance. Every output but the full posterior
    covariance is read, the variances and an aggregated covariance included:
    the whitened solve gives them from its factors.
    """
    index = _product()
    S, op = _kron_cov(index)
    state = index[np.random.default_rng(2).permutation(len(index))]
    problem = InverseProblem(prior_error=S, **_problem_inputs(state, sparse_h=True))
    assert isinstance(problem.prior_error.operator, KroneckerOperator)
    assert problem.prior_error._data is None
    post = fips.solve(problem)
    for name in (
        "estimate", "modeled_obs", "prior_modeled_obs", "prior_obs_covariance",
        "kalman_gain", "dofs", "reduced_chi2", "r2", "rmse", "desroziers",
        "variances", "uncertainty_reduction",
    ):  # fmt: skip
        getattr(post, name)
    post.averaging_kernel()
    rows = pd.MultiIndex.from_tuples([("total", "all")], names=["block", "region"])
    post.cov(Matrix(pd.DataFrame(1.0, index=rows, columns=state)))
    post.estimator.cost(np.ones(len(index)))
    pickle.loads(pickle.dumps(problem.solve("bayesian")))
    assert problem.prior_error._data is None


def test_solved_problem_pickles():
    """A solved problem on factors pickles, and its posterior reads back."""
    index = _product()
    S, op = _kron_cov(index)
    problem = InverseProblem(prior_error=S, **_problem_inputs(index, sparse_h=True))
    problem.solve("bayesian")
    loaded = pickle.loads(pickle.dumps(problem))
    assert isinstance(loaded.prior_error.operator, KroneckerOperator)
    assert isinstance(loaded.estimator._S_0_op, KroneckerOperator)
    np.testing.assert_array_equal(loaded.posterior.values, problem.posterior.values)
    # The public attribute is the dense matrix, formed when read
    S_0 = problem.estimator.S_0
    assert type(S_0) is np.ndarray
    np.testing.assert_array_equal(S_0, op.dense())
    np.testing.assert_array_equal(
        loaded.estimator.reduced_chi2, problem.estimator.reduced_chi2
    )


def test_kron_reduce_matches_product_order():
    """The default positions lay the rows out as ``reduce(np.kron, factors)`` does."""
    rng = np.random.default_rng(12)
    factors = [_spd_factor(rng, k) for k in (2, 3, 2)]
    np.testing.assert_allclose(
        KroneckerOperator(factors).dense(), reduce(np.kron, factors), rtol=1e-15
    )
    assert isinstance(as_operator(KroneckerOperator(factors)), Operator)
