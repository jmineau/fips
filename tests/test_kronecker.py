"""
Tests for Kronecker covariances kept as factors (#6).

``KroneckerOperator`` against the dense matrix it stands for (``np.kron`` of
the factors on the product, and the formula evaluated pair by pair on any other
row order or subset), its products, solves and log-determinant, a
``Covariance`` that holds it behind labels, and a solve from the factors
against the solve from the dense matrix. The operators come from
``Kronecker`` models, as a user builds them.
"""

import logging
import pickle
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp

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
from fips.correlation import CorrelationModel
from fips.kronecker import KroneckerOperator
from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    SparseOperator,
    as_linear_operator,
    dottest,
)
from fips.metrics import haversine_matrix

TIMES = pd.date_range("2024-01-01", periods=4, freq="D")
LATS = [40.0, 40.5, 41.0]
LONS = [-112.0, -111.5]
DAY = pd.Timedelta("1D")


def _time(a, b):
    """The time factor's correlation of two rows: ``exp(-|dt| / 1 day)``."""
    return np.exp(-abs(a["time"] - b["time"]) / DAY)


def _cell(a, b):
    """The space factor's correlation: ``exp(-d / 50 km)``, great-circle."""
    d = haversine_matrix([a["lat"], b["lat"]], [a["lon"], b["lon"]])[0, 1]
    return np.exp(-d / 50.0)


def _line(level, scale):
    """Exponential decay along one numeric level."""
    return lambda a, b: np.exp(-abs(a[level] - b[level]) / scale)


#: Two factors: time, and space as one factor of (lat, lon). With each factor's
#: correlation of a pair of rows, written out, for the reference formula.
KERNELS_2 = Kronecker(
    Exponential("time", "1D"), Exponential(["lat", "lon"], 50.0, metric="haversine")
)
#: Three factors: time, lat and lon separately.
KERNELS_3 = Kronecker(
    Exponential("time", "1D"), Exponential("lat", 0.7), Exponential("lon", 0.4)
)
PAIRS = {
    KERNELS_2: [_time, _cell],
    KERNELS_3: [_time, _line("lat", 0.7), _line("lon", 0.4)],
}
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


def _space(index: pd.MultiIndex) -> Space:
    """The space of `index`: its blocks, or one block ``flux`` without a ``block`` level."""
    return Space.from_index(index) if "block" in index.names else Space({"flux": index})


def _std(variances, index: pd.Index) -> np.ndarray | float:
    """The standard deviations of `index`'s rows, from variances by label (or one)."""
    if isinstance(variances, pd.Series):
        return np.sqrt(variances.reindex(index).to_numpy())
    return float(np.sqrt(variances))


def _operator(model, variances, index: pd.MultiIndex) -> KroneckerOperator:
    """The form of ``model.with_std(std)`` on `index`."""
    S = model.with_std(_std(variances, index)).build(_space(index))
    assert isinstance(S.data, KroneckerOperator)
    return S.data


def _direct_formula(kernels, variances, index: pd.Index) -> np.ndarray:
    """``std_i std_j prod_k C_k(row i, row j)``, one pair of rows at a time."""
    frame = index.to_frame(index=False)
    std = _std(variances, index) * np.ones(len(index))
    n = len(index)
    expected = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            a, b = frame.iloc[i], frame.iloc[j]
            corr = np.prod([pair(a, b) for pair in PAIRS[kernels]])
            expected[i, j] = std[i] * std[j] * corr
    return expected


def _kron_formula(kernels, variances, index: pd.Index) -> np.ndarray:
    """``np.kron`` of the factors' matrices, scaled by std, for an index that is the product."""
    frame = index.to_frame(index=False)
    matrices = []
    for factor in kernels.factors:
        unique = frame[list(factor.levels)].drop_duplicates().reset_index(drop=True)
        matrices.append(factor.correlation(unique))
    std = _std(variances, index) * np.ones(len(index))
    return std[:, None] * reduce(np.kron, matrices) * std[None, :]


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


@dataclass(frozen=True)
class _WrongSize(CorrelationModel):
    """A model that returns a matrix of the wrong size."""

    levels: tuple[str, ...] = ("lat", "lon")

    def correlation(self, coords):
        """Return a 4 x 4 identity whatever the coordinates."""
        return np.eye(4)


# ---------------------------------------------------------------------------
# The operator against the dense matrix
# ---------------------------------------------------------------------------


class TestAgainstTheFormula:
    """A ``Kronecker`` model's operator is the covariance its formula gives."""

    @KERNELS
    @pytest.mark.parametrize("variances", ["series", "scalar"])
    def test_product_order_is_kron(self, kernels, variances):
        """On the product in the factors' order, ``np.kron`` of the factors, bit for bit."""
        index = _product()
        var = _variances(index) if variances == "series" else 1.7
        op = _operator(kernels, var, index)
        assert op._rows is None  # the whole grid, in order: no scatter
        np.testing.assert_array_equal(op.dense(), _kron_formula(kernels, var, index))
        np.testing.assert_allclose(
            op.dense(), _direct_formula(kernels, var, index), rtol=1e-14, atol=0
        )

    @KERNELS
    def test_same_cells_dropped_at_every_time(self, kernels):
        """A product of the cells that remain (what slv builds) is still a whole grid."""
        index = _product()
        lat, lon = index.get_level_values("lat"), index.get_level_values("lon")
        subset = index[~((lat == 40.5) & (lon == -112.0))]
        var = _variances(index)
        op = _operator(kernels, var, subset)
        if kernels is KERNELS_2:
            assert op._rows is None
            np.testing.assert_array_equal(
                op.dense(), _kron_formula(kernels, var, subset)
            )
        np.testing.assert_allclose(
            op.dense(), _direct_formula(kernels, var, subset), rtol=1e-14, atol=0
        )

    @KERNELS
    @pytest.mark.parametrize("seed", range(3))
    def test_permuted_rows(self, kernels, seed):
        """Shuffled rows give the same entries, label by label."""
        index = _product()
        var = _variances(index)
        expected = pd.DataFrame(
            _operator(kernels, var, index).dense(), index=index, columns=index
        )
        shuffled = index[np.random.default_rng(seed).permutation(len(index))]
        op = _operator(kernels, var, shuffled)
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
        op = _operator(kernels, var, other)
        np.testing.assert_allclose(
            op.dense(), _direct_formula(kernels, var, other), rtol=1e-14, atol=0
        )

    @KERNELS
    def test_ragged_subset(self, kernels):
        """Cells missing at some times only: a strict subset of the product."""
        index = _ragged(_product())
        var = _variances(_product())
        op = _operator(kernels, var, index)
        assert op._rows is not None and op.shape == (len(index), len(index))
        np.testing.assert_allclose(
            op.dense(), _direct_formula(kernels, var, index), rtol=1e-14, atol=0
        )

    def test_std_from_variances_on_some_levels(self):
        """Variances by (lat, lon), put on every time by the caller, give each row its own."""
        cells = pd.MultiIndex.from_product([LATS, LONS], names=["lat", "lon"])
        cell_var = pd.Series(np.arange(1.0, 7.0), index=cells, name="var")
        index = _product(block=False)
        row_var = cell_var.reindex(index.droplevel("time"))
        row_var.index = index
        np.testing.assert_allclose(
            _operator(KERNELS_2, row_var, index).dense(),
            _direct_formula(KERNELS_2, row_var, index),
            rtol=1e-14,
            atol=0,
        )

    def test_diagonal_is_the_dense_diagonal(self):
        """``diagonal`` reads the variances without the matrix, the same numbers."""
        index = _ragged(_product())
        op = _operator(KERNELS_3, _variances(_product()), index)
        np.testing.assert_array_equal(op.diagonal(), np.diag(op.dense()))
        assert op.trace() == pytest.approx(np.trace(op.dense()), rel=1e-15)

    def test_factors_on_unique_coordinates_in_first_appearance_order(self):
        """Each factor is evaluated once, on its levels' unique coordinates as they first appear."""
        index = _product(block=False).reorder_levels(["lat", "lon", "time"])
        op = _operator(KERNELS_2, 1.0, index.sort_values())
        times = pd.DataFrame({"time": TIMES})
        np.testing.assert_array_equal(
            op.factors[0], Exponential("time", "1D").correlation(times)
        )
        assert op.factors[1].shape == (len(LATS) * len(LONS),) * 2

    def test_repeated_coordinates_raise(self):
        """Two rows on the same coordinates would be perfectly correlated."""
        index = pd.MultiIndex.from_product(
            [["a", "b"], TIMES, LATS, LONS], names=["source", "time", "lat", "lon"]
        )
        with pytest.raises(ValueError, match="repeat"):
            KERNELS_2.with_std(1.0).build(Space({"flux": index}))

    def test_factor_of_wrong_size_raises(self):
        """A factor must give one row per unique coordinate."""
        model = Kronecker(Exponential("time", "1D"), _WrongSize())
        with pytest.raises(ValueError, match="shape"):
            model.with_std(1.0).build(_space(_product()))


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
    monkeypatch.setattr("fips.kronecker._CHUNK", 50)  # two columns at a time
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
    with caplog.at_level(logging.INFO, logger="fips.kronecker"):
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
    with caplog.at_level(logging.WARNING, logger="fips.kronecker"):
        x = op.solve(b)
    assert "lost accuracy" in caplog.text
    eps = np.finfo(float).eps
    assert _backward_error(S, x, b) < 700 * eps
    # The same factors on the whole grid, and a well-conditioned subset: no fallback
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="fips.kronecker"):
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
    with caplog.at_level(logging.WARNING, logger="fips.linalg"):
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


def test_pickle_keeps_the_factors_not_the_caches():
    """A pickle keeps the factors, not the cached inverses."""
    op = OPS["ragged"]
    op.solve(np.ones(op.shape[0]))  # fill the caches
    state = op.__getstate__()
    assert not {"_inverses", "_complement", "_norm"} & set(state)
    loaded = pickle.loads(pickle.dumps(op))
    assert isinstance(loaded, KroneckerOperator)
    np.testing.assert_array_equal(loaded.dense(), op.dense())
    np.testing.assert_array_equal(
        loaded.solve(np.ones(op.shape[0])), op.solve(np.ones(op.shape[0]))
    )


class TestLogdet:
    """``KroneckerOperator.logdet`` through the factors, against ``np.linalg.slogdet``."""

    @pytest.mark.parametrize("which", ["whole", "permuted"])
    def test_full_product_in_any_order(self, which, monkeypatch):
        """The whole product, in any row order, from the factors without the matrix."""
        op = OPS[which]
        sign, expected = np.linalg.slogdet(op.dense())
        assert sign == 1.0
        monkeypatch.setattr(KroneckerOperator, "dense", _refuse)
        assert op.logdet() == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("seed", range(3))
    def test_random_factors_and_std(self, seed):
        """Factors of other sizes, a random permutation and random deviations."""
        rng = np.random.default_rng(seed)
        sizes = (3, 2, 4)
        factors = [_spd_factor(rng, k) for k in sizes]
        rows = rng.permutation(24)
        op = KroneckerOperator(
            factors,
            std=rng.uniform(0.2, 3.0, 24),
            positions=np.unravel_index(rows, sizes),
        )
        assert op.logdet() == pytest.approx(np.linalg.slogdet(op.dense())[1], rel=1e-12)

    def test_strict_subset_falls_back_to_dense(self, monkeypatch):
        """A strict subset of the product factors the dense matrix."""
        op = OPS["ragged"]
        expected = np.linalg.slogdet(op.dense())[1]
        formed = []
        dense = KroneckerOperator.dense

        def spy(self):
            """Record that the dense matrix was formed, and form it."""
            formed.append(self.shape)
            return dense(self)

        monkeypatch.setattr(KroneckerOperator, "dense", spy)
        assert op.logdet() == pytest.approx(expected, rel=1e-12)
        assert formed == [op.shape]

    def test_zero_std_raises(self):
        """A zero standard deviation makes the operator singular."""
        op = KroneckerOperator([np.eye(2), np.eye(2)], std=[1.0, 0.0, 1.0, 1.0])
        with pytest.raises(np.linalg.LinAlgError, match="zero"):
            op.logdet()

    def test_indefinite_factor_raises(self):
        """A factor that is not positive-definite has no log-determinant."""
        op = KroneckerOperator([np.array([[1.0, 2.0], [2.0, 1.0]]), np.eye(2)])
        with pytest.raises(np.linalg.LinAlgError):
            op.logdet()


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
# A covariance that holds the factors behind labels
# ---------------------------------------------------------------------------


def _refuse(*args, **kwargs):
    """Stand in for a method that would form the n x n matrix."""
    raise AssertionError("the dense matrix was formed")


@pytest.fixture
def no_dense(monkeypatch):
    """Fail if anything forms a Kronecker operator's dense matrix."""
    for name in ("dense", "__array__"):
        monkeypatch.setattr(KroneckerOperator, name, _refuse)


def _kron_cov(index=None, kernels=KERNELS_2, var=None, name="prior_error"):
    """A covariance on the operator of a Kronecker error on `index`."""
    index = _product() if index is None else index
    var = _variances(_product()) if var is None else var
    op = _operator(kernels, var, index)
    return Covariance(op, Space.from_index(index), name=name), op


def _dense_cov(op: KroneckerOperator, index: pd.MultiIndex) -> Covariance:
    """The same covariance as a dense matrix, read from a labeled frame."""
    return Covariance.from_frame(pd.DataFrame(op.dense(), index=index, columns=index))


def _positions_by_label(source: pd.MultiIndex, target: pd.MultiIndex) -> np.ndarray:
    """Where each label of `target` is in `source`, matched by level name in a dict."""
    names = list(source.names)
    where = {label: i for i, label in enumerate(source)}
    return np.array([where[label] for label in target.reorder_levels(names)])


class TestCovarianceOnFactors:
    """``Covariance(op, space)`` keeps the factors; only ``dense`` forms the matrix."""

    def test_labels_shape_and_form(self, no_dense):
        """The labels, shape, form and variances, without the matrix."""
        S, op = _kron_cov()
        idx = _product()
        assert S.rows.index.equals(idx) and S.cols is S.rows
        assert S.shape == (len(idx), len(idx))
        assert S.data is op
        assert S.name == "prior_error"
        text = repr(S)
        assert "KroneckerOperator" in text and "shape=(24, 24)" in text
        np.testing.assert_array_equal(S.variances.values, op.diagonal())

    def test_dense_outputs(self):
        """``dense``, ``np.asarray``, ``to_frame``, a block and a sum, as for a dense matrix."""
        S, op = _kron_cov()
        D = op.dense()
        ref = _dense_cov(op, _product())
        assert isinstance(ref.data, DenseOperator)
        np.testing.assert_array_equal(S.dense(), D)
        np.testing.assert_array_equal(np.asarray(S), D)
        pd.testing.assert_frame_equal(S.to_frame(), ref.to_frame())
        np.testing.assert_array_equal(S["flux", "flux"].dense(), D)
        total = S + ref
        assert isinstance(total, Covariance)
        np.testing.assert_allclose(total.dense(), 2 * D)

    def test_diagonal_block_keeps_the_factors(self, no_dense):
        """``cov[b, b]`` is a covariance on that block, still factors."""
        S, op = _kron_cov()
        block = S["flux", "flux"]
        assert isinstance(block, Covariance)
        assert isinstance(block.data, KroneckerOperator)
        assert block.space == S.space

    def test_labels_are_checked(self):
        """An unnamed level, a missing ``block`` level and a shape mismatch raise."""
        op = OPS["whole"]
        unnamed = pd.MultiIndex.from_product(
            [["flux"], range(24)], names=["block", None]
        )
        with pytest.raises(ValueError, match="must be named"):
            Covariance(op, Space.from_index(unnamed))
        no_block = pd.Index(range(24), name="cell")
        with pytest.raises(ValueError, match="'block' level"):
            Covariance(op, Space.from_index(no_block))
        short = pd.MultiIndex.from_product([["flux"], range(23)], names=["block", "c"])
        with pytest.raises(ValueError, match="shape"):
            Covariance(op, Space.from_index(short))

    def test_frame_refused(self):
        """A DataFrame given to the constructor raises and names ``from_frame``."""
        S, op = _kron_cov()
        with pytest.raises(TypeError, match="from_frame"):
            Covariance(S.to_frame(), S.space)

    def test_block_level_put_first(self, no_dense):
        """The ``block`` level comes first in the labels, which are kept as given."""
        idx = pd.MultiIndex.from_product(
            [["1", "2"], ["flux"]], names=["cell", "block"]
        )
        S = Covariance(KroneckerOperator([np.eye(2)]), Space.from_index(idx))
        assert list(S.rows.index.names) == ["block", "cell"]
        assert list(S.space.select("flux")) == ["1", "2"]
        assert S.cols is S.rows

    def test_align_and_round_keep_the_factors(self, no_dense):
        """A permutation of the labels picks rows of the factors; rounding keeps them."""
        S, op = _kron_cov()
        idx = _product()
        target = Space.from_index(idx[np.random.default_rng(0).permutation(len(idx))])
        R = S.round_labels(6).align(target, target)
        assert isinstance(R, Covariance) and isinstance(R.data, KroneckerOperator)
        assert R.space is target
        assert S.round_labels(6).data is op
        assert S.align(S.space, S.space).data is op

    def test_align_values(self):
        """The aligned covariance holds the entries of the labels it is put on."""
        S, op = _kron_cov()
        idx = _product()
        shuffled = idx[np.random.default_rng(1).permutation(len(idx))]
        target = Space.from_index(shuffled)
        R = S.align(target, target)
        expected = (
            pd.DataFrame(op.dense(), index=idx, columns=idx)
            .reindex(index=shuffled, columns=shuffled)
            .to_numpy()
        )
        np.testing.assert_array_equal(R.dense(), expected)

    def test_align_to_a_subset_raises(self):
        """``align`` is an exact join: a subset of the labels raises."""
        S, _ = _kron_cov()
        subset = Space.from_index(_product()[:20])
        with pytest.raises(ValueError, match="only in the other space"):
            S.align(subset, subset)

    def test_align_with_levels_in_another_order(self):
        """Labels with the levels in another order are matched label by label."""
        S, op = _kron_cov()
        target_index = (
            _product().reorder_levels(["block", "lat", "lon", "time"]).sort_values()
        )
        target = Space.from_index(target_index)
        R = S.align(target, target)
        assert isinstance(R.data, KroneckerOperator)
        assert R.rows.index.equals(target_index)
        p = _positions_by_label(_product(), target_index)
        np.testing.assert_array_equal(R.dense(), op.dense()[np.ix_(p, p)])

    def test_align_to_other_levels_raises(self):
        """A space with other levels raises."""
        S, _ = _kron_cov()
        other = Space.from_index(
            pd.MultiIndex.from_product([["flux"], range(24)], names=["block", "x"])
        )
        with pytest.raises(ValueError, match="levels"):
            S.align(other, other)

    def test_problem_with_levels_in_another_order(self):
        """``Problem`` takes a factored covariance whose levels are in another order than the prior's."""
        _, op = _kron_cov()
        inputs = _problem_inputs(_product(), sparse_h=True)
        order = ["block", "lat", "lon", "time"]
        S_other = Covariance(op, Space.from_index(_product().reorder_levels(order)))
        problem = Problem(prior_cov=S_other, **inputs)
        assert isinstance(problem.prior_cov.data, KroneckerOperator)
        post = fips.solve(problem)
        ref = fips.solve(Problem(prior_cov=_dense_cov(op, _product()), **inputs))
        _close(post.estimate.values, ref.estimate.values, "estimate")

    def test_problem_with_a_new_label_raises(self):
        """A state label the covariance does not have raises, naming it: no zero fill."""
        S, _ = _kron_cov()
        idx = _product()
        extra = pd.MultiIndex.from_tuples(
            [("flux", TIMES[0], 39.0, -112.0)], names=idx.names
        )
        inputs = _problem_inputs(idx.append(extra), sparse_h=True)
        with pytest.raises(
            ValueError,
            match=r"labels of block 'flux' missing from the rows of prior_cov",
        ):
            Problem(prior_cov=S, **inputs)

    def test_scalars_and_relabeling_keep_the_factors(self, no_dense):
        """A positive scalar, and the form put behind labels again, keep the factors."""
        S, op = _kron_cov()
        for scaled in (S * 2.0, 2.0 * S, S / 0.5):
            assert isinstance(scaled, Covariance)
            assert isinstance(scaled.data, KroneckerOperator)
            np.testing.assert_allclose(scaled.variances.values, 2 * op.diagonal())
        again = Covariance(S.data, S.space, name="again")
        assert again.data is op and again.name == "again"
        assert Operator(S.data, S.rows, S.cols).data is op

    def test_pickle(self, no_dense):
        """A pickle keeps the factors, the labels and the name."""
        S, op = _kron_cov()
        loaded = pickle.loads(pickle.dumps(S))
        assert isinstance(loaded, Covariance)
        assert isinstance(loaded.data, KroneckerOperator)
        assert loaded.space == S.space and loaded.name == S.name

    def test_one_block_without_a_block_level(self, no_dense):
        """An index without a ``block`` level is one named block; ``from_blocks`` keeps the form."""
        idx = _product(block=False)
        op = _operator(KERNELS_2, 1.0, idx)
        space = Space({"flux": idx})
        S = Covariance(op, space)
        assert S.space.names == ("flux",) and S.data is op
        B = Covariance.from_blocks({("flux", "flux"): op}, rows=space, cols=space)
        assert isinstance(B, Covariance) and B.data is op
        loaded = pickle.loads(pickle.dumps(B))
        assert isinstance(loaded.data, KroneckerOperator)

    @pytest.mark.parametrize("form", ["array", "dense", "sparse", "diagonal"])
    def test_nan_raises_as_for_a_frame(self, form):
        """A form that stores a NaN is refused, as a frame with one is."""
        idx = pd.MultiIndex.from_product([["flux"], [0, 1]], names=["block", "cell"])
        A = np.array([[1.0, np.nan], [np.nan, 1.0]])
        data = {
            "array": A,
            "dense": DenseOperator(A),
            "sparse": SparseOperator(sp.csr_matrix(A)),
            "diagonal": DiagonalOperator([1.0, np.nan]),
        }[form]
        with pytest.raises(ValueError, match="NaN"):
            Covariance(data, Space.from_index(idx))
        with pytest.raises(ValueError, match="NaN"):
            Covariance.from_frame(pd.DataFrame(A, index=idx, columns=idx))

    def test_diagonal_form_behind_labels(self):
        """A diagonal form stays diagonal behind labels."""
        idx = _product()
        D = DiagonalOperator(np.arange(1.0, 25.0))
        S = Covariance(D, Space.from_index(idx))
        assert S.data is D
        np.testing.assert_array_equal(np.asarray(S), np.diag(np.arange(1.0, 25.0)))
        np.testing.assert_array_equal(S.variances.values, np.arange(1.0, 25.0))


# ---------------------------------------------------------------------------
# The solve, from the factors against the dense matrix
# ---------------------------------------------------------------------------

POSTERIOR_OUTPUTS = [
    "estimate", "mean", "cov", "variances", "modeled_obs", "prior_modeled_obs",
    "prior_obs_cov", "kalman_gain", "averaging_kernel", "desroziers", "dofs", "chi2",
    "reduced_chi2", "r2", "rmse", "uncertainty_reduction",
]  # fmt: skip


def _close(actual, expected, what):
    """Compare at 1e-10 relative, with an absolute floor of 1e-10 of the largest entry."""
    expected = np.asarray(expected, dtype=float)
    floor = 1e-10 * max(np.abs(expected).max(), 1.0)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), expected, rtol=1e-10, atol=floor, err_msg=what
    )


def _problem_inputs(state_index, sparse_h: bool, seed: int = 11):
    """The obs, prior, H, S_z and offset of a small problem on `state_index`."""
    rng = np.random.default_rng(seed)
    n, m = len(state_index), 15
    obs_index = pd.MultiIndex.from_product(
        [["conc"], [f"o{k}" for k in range(m)]], names=["block", "obs"]
    )
    obs_space = Space.from_index(obs_index)
    H = np.zeros((m, n))
    for i in range(m):
        H[i, rng.choice(n, 4, replace=False)] = rng.uniform(0.5, 2.0, 4)
    x_0 = rng.normal(5.0, 1.0, n)
    z = H @ x_0 + rng.normal(0, 1.0, m)
    state_space = Space.from_index(state_index)
    return {
        "obs": Vector(z, obs_space),
        "prior": Vector(x_0, state_space),
        "forward": Operator(sp.csr_array(H) if sparse_h else H, obs_space, state_space),
        "obs_cov": Covariance.diagonal(rng.uniform(0.1, 0.4, m), obs_space),
        "offset": Vector(rng.normal(0.5, 0.1, m), obs_space),
    }


def _compare(post, ref):
    """Every Posterior output of `post` against `ref`, at 1e-10."""
    for name in POSTERIOR_OUTPUTS:
        value, expected = getattr(post, name), getattr(ref, name)
        if callable(value):  # cov, averaging_kernel, desroziers: with no argument
            value, expected = value(), expected()
        if isinstance(value, pd.DataFrame):  # desroziers
            assert value.index.equals(expected.index), name
            value, expected = value.to_numpy(), expected.to_numpy()
        if isinstance(value, Operator):  # a labeled matrix
            assert value.rows == expected.rows and value.cols == expected.cols, name
            value, expected = value.dense(), expected.dense()
        elif hasattr(value, "index"):
            assert value.index.equals(expected.index), name
            value, expected = value.values, expected.values
        _close(value, expected, f"posterior {name}")


@pytest.mark.parametrize("sparse_h", [False, True], ids=["dense H", "sparse H"])
@pytest.mark.parametrize("shape", ["product", "permuted", "ragged"])
@KERNELS
def test_solve_matches_the_dense_build(sparse_h, shape, kernels):
    """
    A problem whose prior error is built on the factors solves as the dense one does.

    ``product``: ``np.kron`` of the factors is the reference. ``permuted``: the
    covariance is built on the product but the prior comes in a shuffled order, so
    the problem reorders the operator. ``ragged``: cells missing at some times
    (the dense reference is the operator's own matrix, which ``np.kron`` cannot
    give on this index).
    """
    index = _product()
    var = _variances(index)
    if shape == "ragged":
        index = _ragged(index)
    op = _operator(kernels, var, index)
    dense = op.dense() if shape == "ragged" else _kron_formula(kernels, var, index)
    if shape == "product":
        np.testing.assert_array_equal(op.dense(), dense)
    state = index
    if shape == "permuted":
        state = index[np.random.default_rng(5).permutation(len(index))]
    inputs = _problem_inputs(state, sparse_h)

    S_factored = Covariance(op, Space.from_index(index))
    S_dense = Covariance.from_frame(pd.DataFrame(dense, index=index, columns=index))
    problem = Problem(prior_cov=S_factored, **inputs)
    assert isinstance(problem.prior_cov.data, KroneckerOperator)
    post = fips.solve(problem, gamma=0.7)
    ref = fips.solve(Problem(prior_cov=S_dense, **inputs), gamma=0.7)
    _compare(post, ref)


def _flux_and_bias(seed: int = 3):
    """A state of a Kronecker flux block and a diagonal bias block, its covariance, and the problem's other inputs."""
    rng = np.random.default_rng(seed)
    cells = _ragged(_product(block=False))
    op = _operator(KERNELS_3, _variances(cells), cells)
    sites = pd.Index(["A", "B", "C"], name="site")
    S_0 = Covariance.block_diag(
        Covariance(op, Space({"flux": cells})),
        Covariance.diagonal(rng.uniform(0.5, 2.0, 3), Space({"bias": sites})),
        name="prior_error",
    )
    prior = Vector(rng.normal(5.0, 1.0, len(S_0.space)), S_0.space, name="prior")
    m = 12
    obs_space = Space({"conc": pd.Index([f"o{k}" for k in range(m)], name="obs")})
    H = np.zeros((m, len(S_0.space)))
    for i in range(m):
        H[i, rng.choice(len(cells), 4, replace=False)] = rng.uniform(0.5, 2.0, 4)
        H[i, len(cells) + i % 3] = 1.0  # each observation sees one site's bias
    z = H @ prior.values + rng.normal(0, 1.0, m)
    inputs = {
        "obs": Vector(z, obs_space, name="obs"),
        "forward": Operator(sp.csr_array(H), obs_space, S_0.space),
        "obs_cov": Covariance.diagonal(rng.uniform(0.1, 0.4, m), obs_space),
    }
    return S_0, prior, inputs


def test_block_diagonal_prior_error_solves_as_the_dense_one():
    """
    A Kronecker flux block and a diagonal bias block, kept as forms, solve as the dense matrix does.

    The prior error is ``Covariance.block_diag`` of the two; the reference is
    the same covariance as one dense matrix.
    """
    S_0, prior, inputs = _flux_and_bias()
    assert isinstance(S_0.data, BlockDiagonalOperator)
    flux, bias = S_0.data.blocks
    assert isinstance(flux, KroneckerOperator) and isinstance(bias, DiagonalOperator)
    np.testing.assert_array_equal(
        S_0.dense(), scipy.linalg.block_diag(flux.dense(), bias.dense())
    )
    S_dense = Covariance(S_0.dense(), S_0.space)
    problem = Problem(prior=prior, prior_cov=S_0, **inputs)
    post = fips.solve(problem, gamma=0.7)
    ref = fips.solve(Problem(prior=prior, prior_cov=S_dense, **inputs), gamma=0.7)
    assert isinstance(problem.prior_cov.data, BlockDiagonalOperator)
    assert isinstance(problem.prior_cov.data.blocks[0], KroneckerOperator)
    _compare(post, ref)
    flux_cov = problem.prior_cov["flux", "flux"]
    assert isinstance(flux_cov, Covariance)
    np.testing.assert_array_equal(flux_cov.dense(), flux.dense())


def test_block_of_a_block_diagonal_keeps_the_factors(no_dense):
    """The flux block of a block diagonal covariance is a covariance that still solves through the factors."""
    S_0, _, _ = _flux_and_bias()
    flux = S_0["flux", "flux"]
    assert isinstance(flux, Covariance)
    assert flux.space.select("flux").equals(S_0.space.select("flux"))
    b = np.random.default_rng(0).standard_normal(len(flux.space))
    x = flux.solve(Vector(b, flux.space))
    np.testing.assert_allclose(S_0.data.blocks[0] @ x.values, b, rtol=1e-10, atol=1e-10)


def test_block_diagonal_in_another_block_order_keeps_the_factors():
    """Putting a block diagonal covariance on its blocks in another order keeps each block's form."""
    S_0, _, _ = _flux_and_bias()
    other = Space({"bias": S_0.space.select("bias"), "flux": S_0.space.select("flux")})
    R = S_0.align(other, other)
    np.testing.assert_array_equal(
        R["flux", "flux"].dense(), S_0["flux", "flux"].dense()
    )
    assert isinstance(R.data, BlockDiagonalOperator)
    assert isinstance(R.data.blocks[1], KroneckerOperator)


def test_a_model_given_to_the_problem_is_kept_as_factors():
    """A Kronecker covariance model is built on the prior's space as a ``KroneckerOperator``."""
    index = _product()
    model = Kronecker(
        Exponential("time", "30D"),
        Exponential(["lat", "lon"], 50.0, metric="haversine"),
    ).with_std(0.5)
    problem = Problem(prior_cov=model, **_problem_inputs(index, sparse_h=True))
    assert isinstance(problem.prior_cov.data, KroneckerOperator)
    assert problem.prior_cov.space is problem.prior.space
    np.testing.assert_array_equal(
        problem.prior_cov.dense(), model.build(Space.from_index(index)).dense()
    )


def test_solve_never_forms_the_prior_error(no_dense):
    """
    Building the problem and solving it never forms the ``n x n`` prior error.

    The prior comes in another row order, so the problem reorders the
    covariance. Every output but the full posterior covariance is read, the
    variances and an aggregated covariance included: the whitened solve gives
    them from its factors.
    """
    index = _product()
    S, op = _kron_cov(index)
    state = index[np.random.default_rng(2).permutation(len(index))]
    problem = Problem(prior_cov=S, **_problem_inputs(state, sparse_h=True))
    assert isinstance(problem.prior_cov.data, KroneckerOperator)
    post = fips.solve(problem)
    for name in (
        "estimate", "modeled_obs", "prior_modeled_obs", "prior_obs_cov",
        "kalman_gain", "dofs", "chi2", "reduced_chi2", "r2", "rmse",
        "variances", "uncertainty_reduction",
    ):  # fmt: skip
        getattr(post, name)
    post.averaging_kernel()
    post.desroziers()
    post.leverage(post.estimate)
    rows = pd.MultiIndex.from_tuples([("total", "all")], names=["block", "region"])
    post.cov(Operator.from_frame(pd.DataFrame(1.0, index=rows, columns=state)))
    pickle.loads(pickle.dumps(post.with_gamma(2.0)))
    assert isinstance(problem.prior_cov.data, KroneckerOperator)


def test_posterior_on_factors_pickles():
    """A posterior whose prior error is factors pickles, and keeps the factors."""
    index = _product()
    S, _ = _kron_cov(index)
    post = fips.solve(Problem(prior_cov=S, **_problem_inputs(index, sparse_h=True)))
    loaded = pickle.loads(pickle.dumps(post))
    assert isinstance(loaded.problem.prior_cov, Covariance)
    assert isinstance(loaded.problem.prior_cov.data, KroneckerOperator)
    np.testing.assert_array_equal(loaded.estimate.values, post.estimate.values)
    assert loaded.reduced_chi2 == post.reduced_chi2


def test_kron_reduce_matches_product_order():
    """The default positions lay the rows out as ``reduce(np.kron, factors)`` does."""
    rng = np.random.default_rng(12)
    factors = [_spd_factor(rng, k) for k in (2, 3, 2)]
    np.testing.assert_allclose(
        KroneckerOperator(factors).dense(), reduce(np.kron, factors), rtol=1e-15
    )
    op = KroneckerOperator(factors)
    assert as_linear_operator(op) is op and isinstance(op, LinearOperator)
