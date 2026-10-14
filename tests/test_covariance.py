"""
Tests for the covariance module.

``Covariance`` (a symmetric operator on one space, kept in its cheapest form)
against NumPy on small random positive-definite matrices, in each form; then
the error components and ``CovarianceBuilder`` that build a covariance's
matrix as a DataFrame.
"""

import pickle
from functools import reduce

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp

from fips import Covariance, Field, Operator, Space
from fips.covariance import (
    BlockDecayError,
    CovarianceBuilder,
    DiagonalError,
    KroneckerError,
)
from fips.kernels import GridSpatialDecay, GridTimeDecay
from fips.kronecker import KroneckerOperator
from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    SparseOperator,
)

# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

CELLS = Space({"flux": pd.Index(list("abcdef"), name="cell")})
SITES = Space({"bias": pd.Index(["A", "B"], name="site")})
OTHER = Space({"other": pd.Index([1, 2], name="id")})


def _spd(rng, n: int) -> np.ndarray:
    """A random symmetric positive-definite matrix of size `n`, well conditioned."""
    B = rng.standard_normal((n, n))
    return B @ B.T + n * np.eye(n)


def _corr(rng, n: int) -> np.ndarray:
    """A random correlation matrix of size `n`."""
    A = _spd(rng, n)
    d = np.sqrt(np.diag(A))
    return A / d[:, None] / d[None, :]


def _covariances() -> dict[str, Covariance]:
    """A covariance in each form, on six cells (eight labels for the block diagonal)."""
    rng = np.random.default_rng(0)
    tridiagonal = sp.diags_array(
        [np.full(5, 0.4), rng.uniform(1.0, 2.0, 6), np.full(5, 0.4)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    kron = KroneckerOperator(
        [_corr(rng, 2), _corr(rng, 3)], std=rng.uniform(0.5, 2.0, 6)
    )
    flux = Covariance(kron, CELLS)
    bias = Covariance.diagonal([0.5, 2.0], SITES)
    return {
        "dense": Covariance(_spd(rng, 6), CELLS, name="prior_error"),
        "sparse": Covariance(tridiagonal, CELLS, name="prior_error"),
        "diagonal": Covariance.diagonal(rng.uniform(0.5, 2.0, 6), CELLS),
        "kronecker": flux,
        "block": Covariance.block_diag(flux, bias, name="prior_error"),
    }


COVS = _covariances()
FORM_TYPES = {
    "dense": DenseOperator,
    "sparse": SparseOperator,
    "diagonal": DiagonalOperator,
    "kronecker": KroneckerOperator,
    "block": BlockDiagonalOperator,
}
EACH_FORM = pytest.mark.parametrize("form", list(COVS))


def _shuffled(space: Space, seed: int = 0) -> Space:
    """`space` with the labels of each block in a random order (and the blocks reversed)."""
    rng = np.random.default_rng(seed)
    return Space(
        {
            name: space.select(name)[rng.permutation(len(space.select(name)))]
            for name in reversed(space.names)
        }
    )


def _keys(space: Space) -> list[tuple]:
    """Each label of `space` as ``(block, label)``, in order (no NaN padding)."""
    return [(name, label) for name in space.names for label in space.select(name)]


def _values_on(values: np.ndarray, source: Space, target: Space) -> np.ndarray:
    """`values` (in the order of `source`) in the order of `target`, matched by label in a dict."""
    where = {key: i for i, key in enumerate(_keys(source))}
    return values[[where[key] for key in _keys(target)]]


def _frame(A: np.ndarray, rows: pd.Index, cols: pd.Index | None = None) -> pd.DataFrame:
    """`A` labeled by `rows` and `cols` (`rows` on both axes by default)."""
    return pd.DataFrame(A, index=rows, columns=rows if cols is None else cols)


class TestConstruction:
    """A covariance is square, with the same labels on both axes."""

    def test_one_space_on_both_axes(self):
        """The space labels the rows and the columns; the form is kept."""
        A = _spd(np.random.default_rng(1), 6)
        S = Covariance(A, CELLS, name="prior_error")
        assert isinstance(S, Operator)
        assert S.space is CELLS and S.rows is CELLS and S.cols is CELLS
        assert S.shape == (6, 6)
        assert S.name == "prior_error"
        assert isinstance(S.data, DenseOperator)
        np.testing.assert_array_equal(S.dense(), A)
        assert "Covariance" in repr(S) and "space" in repr(S)

    def test_wrong_shape_raises(self):
        """A matrix with other than one row and one column per label raises."""
        with pytest.raises(ValueError, match="shape"):
            Covariance(np.eye(5), CELLS)
        with pytest.raises(ValueError, match="shape"):
            Covariance(np.ones((6, 5)), CELLS)

    def test_frame_and_nan_refused(self):
        """A DataFrame is refused (its labels would be ignored), and so is a NaN."""
        with pytest.raises(TypeError, match="from_frame"):
            Covariance(_frame(np.eye(6), CELLS.index), CELLS)
        A = np.eye(6)
        A[0, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            Covariance(A, CELLS)
        with pytest.raises(TypeError, match="Space"):
            Covariance(np.eye(6), CELLS.index)

    def test_from_frame_puts_the_columns_in_the_rows_order(self):
        """``from_frame`` reads the labels and reorders the columns to the rows."""
        A = _spd(np.random.default_rng(2), 6)
        labels = CELLS.select("flux")
        order = np.array([3, 0, 5, 1, 4, 2])
        frame = _frame(A, labels).iloc[:, order]
        S = Covariance.from_frame(frame, row_block="flux", col_block="flux")
        assert isinstance(S, Covariance)
        assert S.rows == CELLS and S.cols is S.rows
        np.testing.assert_array_equal(S.dense(), A)
        pd.testing.assert_frame_equal(S.to_frame(), _frame(A, CELLS.index))

    def test_from_frame_with_block_levels(self):
        """A frame with a ``block`` level on both axes, blocks in another order on the columns."""
        rng = np.random.default_rng(3)
        space = Space({"flux": CELLS.select("flux"), "bias": SITES.select("bias")})
        A = _spd(rng, 8)
        frame = _frame(A, space.index)
        columns = np.r_[6, 7, rng.permutation(6)]
        S = Covariance.from_frame(frame.iloc[:, columns])
        assert isinstance(S, Covariance) and S.space == space
        np.testing.assert_array_equal(S.dense(), A)

    def test_from_frame_with_other_labels_raises(self):
        """Columns with labels the rows do not have raise."""
        labels = CELLS.select("flux")
        other = pd.Index(list("abcdeg"), name="cell")
        with pytest.raises(ValueError, match="same labels on both axes"):
            Covariance.from_frame(
                _frame(np.eye(6), labels, other), row_block="flux", col_block="flux"
            )
        with pytest.raises(ValueError, match="same labels on both axes"):
            Covariance.from_frame(
                _frame(np.eye(6), labels), row_block="flux", col_block="other"
            )

    def test_from_blocks(self):
        """Blocks on the diagonal, zero between; the columns put in the rows' order."""
        rng = np.random.default_rng(4)
        A, B = _spd(rng, 6), _spd(rng, 2)
        flux, bias = CELLS.select("flux"), SITES.select("bias")
        blocks = {("flux", "flux"): _frame(A, flux), ("bias", "bias"): _frame(B, bias)}
        S = Covariance.from_blocks(blocks)
        assert isinstance(S, Covariance)
        assert S.space.names == ("flux", "bias")
        np.testing.assert_array_equal(S.dense(), scipy.linalg.block_diag(A, B))
        rows = Space({"flux": flux, "bias": bias})
        cols = Space({"bias": bias, "flux": flux[::-1]})
        T = Covariance.from_blocks(blocks, rows=rows, cols=cols)
        assert T.cols is T.rows and T.rows == rows
        np.testing.assert_array_equal(T.dense(), S.dense())

    def test_from_blocks_on_other_blocks_raises(self):
        """Rows and columns on different blocks are not a covariance."""
        frame = _frame(np.eye(2), SITES.select("bias"))
        with pytest.raises(ValueError, match="same labels on both axes"):
            Covariance.from_blocks({("bias", "other"): frame})

    def test_one_block_keeps_its_form(self):
        """A single form given as the one block stays that form."""
        op = COVS["kronecker"].data
        S = Covariance.from_blocks({("flux", "flux"): op}, rows=CELLS, cols=CELLS)
        assert isinstance(S, Covariance) and S.data is op


class TestDiagonal:
    """``Covariance.diagonal`` keeps the variances as a diagonal."""

    def test_scalar(self):
        """A scalar is the same variance for every label."""
        S = Covariance.diagonal(0.25, CELLS, name="obs_error")
        assert isinstance(S.data, DiagonalOperator)
        assert S.space is CELLS and S.name == "obs_error"
        np.testing.assert_array_equal(S.dense(), 0.25 * np.eye(6))

    def test_array(self):
        """An array gives a variance for each label, in the space's order."""
        values = np.arange(1.0, 7.0)
        S = Covariance.diagonal(values, CELLS)
        assert isinstance(S.data, DiagonalOperator)
        np.testing.assert_array_equal(S.variances.values, values)

    def test_field(self):
        """A field brings its own space."""
        field = Field(np.arange(1.0, 7.0), CELLS)
        S = Covariance.diagonal(field)
        assert S.space is CELLS
        np.testing.assert_array_equal(S.dense(), np.diag(field.values))

    def test_field_on_a_space_in_another_order(self):
        """A field and a space in another order: the variances follow the space."""
        values = np.arange(1.0, 7.0)
        other = _shuffled(CELLS)
        S = Covariance.diagonal(Field(values, CELLS), other)
        assert S.space is other
        np.testing.assert_array_equal(
            S.variances.values, _values_on(values, CELLS, other)
        )

    @pytest.mark.parametrize(
        "values, space, match",
        [
            (1.0, None, "needs `space`"),
            (np.ones(6), None, "needs `space`"),
            (np.ones((6, 1)), CELLS, "scalar or 1D"),
            (np.ones(5), CELLS, "Got 5 variances"),
            (Field(np.ones(6), CELLS), SITES, "same labels"),
        ],
        ids=["scalar", "array", "2D", "length", "field"],
    )
    def test_errors(self, values, space, match):
        """No space for a scalar or array, a 2D array, a wrong length, a field on other labels."""
        with pytest.raises(ValueError, match=match):
            Covariance.diagonal(values, space)


class TestBlockDiag:
    """``Covariance.block_diag`` puts independent covariances side by side, each in its form."""

    def test_all_diagonal_is_diagonal(self):
        """Diagonal parts give one diagonal."""
        S = Covariance.block_diag(
            Covariance.diagonal(2.0, CELLS), Covariance.diagonal([0.5, 3.0], SITES)
        )
        assert isinstance(S.data, DiagonalOperator)
        assert S.space.names == ("flux", "bias")
        assert S.space.select("bias").equals(SITES.select("bias"))
        np.testing.assert_array_equal(S.variances.values, [2.0] * 6 + [0.5, 3.0])

    def test_a_kronecker_part_is_kept(self):
        """A Kronecker part stays factors inside a block diagonal form."""
        flux = COVS["kronecker"]
        bias = Covariance.diagonal([0.5, 3.0], SITES)
        S = Covariance.block_diag(flux, bias, name="prior_error")
        assert isinstance(S, Covariance) and S.name == "prior_error"
        assert isinstance(S.data, BlockDiagonalOperator)
        assert S.data.blocks[0] is flux.data and S.data.blocks[1] is bias.data
        np.testing.assert_array_equal(
            S.dense(), scipy.linalg.block_diag(flux.dense(), bias.dense())
        )

    def test_one_part_keeps_its_form(self):
        """A single covariance is returned in its form, on its space."""
        S = Covariance.block_diag(COVS["dense"])
        assert S.data is COVS["dense"].data and S.space == CELLS

    def test_errors(self):
        """No part, a part that is not a covariance, or two blocks of one name raise."""
        with pytest.raises(ValueError, match="at least one"):
            Covariance.block_diag()
        with pytest.raises(TypeError, match="Covariances"):
            Covariance.block_diag(COVS["dense"], Operator(np.eye(2), SITES, SITES))
        with pytest.raises(ValueError, match="Two blocks are named 'flux'"):
            Covariance.block_diag(COVS["dense"], COVS["diagonal"])


class TestAgainstNumPy:
    """Each method of each form against the same computation on the dense matrix."""

    @EACH_FORM
    def test_form(self, form):
        """Each covariance holds the form it was built in."""
        assert isinstance(COVS[form].data, FORM_TYPES[form])

    @EACH_FORM
    def test_variances(self, form):
        """The variances are the diagonal, as a field on the space."""
        S = COVS[form]
        variances = S.variances
        assert isinstance(variances, Field) and variances.space is S.space
        assert variances.name == "variance"
        np.testing.assert_allclose(variances.values, np.diag(S.dense()), rtol=1e-14)

    @EACH_FORM
    def test_solve_a_field(self, form):
        """``solve`` of a field is ``np.linalg.solve``, on the covariance's space."""
        S = COVS[form]
        b = np.random.default_rng(5).standard_normal(len(S.space))
        x = S.solve(Field(b, S.space))
        assert isinstance(x, Field) and x.space is S.space
        np.testing.assert_allclose(x.values, np.linalg.solve(S.dense(), b), rtol=1e-10)

    @EACH_FORM
    def test_solve_a_field_in_another_order(self, form):
        """A right-hand side in another order is matched by label."""
        S = COVS[form]
        b = np.random.default_rng(6).standard_normal(len(S.space))
        other = _shuffled(S.space)
        x = S.solve(Field(_values_on(b, S.space, other), other))
        assert x.space is S.space
        np.testing.assert_allclose(x.values, np.linalg.solve(S.dense(), b), rtol=1e-10)

    @EACH_FORM
    def test_solve_an_operator(self, form):
        """An operator right-hand side, rows in any order, gives an operator on its columns."""
        S = COVS[form]
        n = len(S.space)
        B = np.random.default_rng(7).standard_normal((n, 3))
        obs = Space({"co2": pd.Index(["o1", "o2", "o3"], name="obs_id")})
        other = _shuffled(S.space)
        rhs = Operator(_values_on(B, S.space, other), other, obs)
        X = S.solve(rhs)
        assert type(X) is Operator
        assert X.rows is S.space and X.cols is obs
        np.testing.assert_allclose(X.dense(), np.linalg.solve(S.dense(), B), rtol=1e-10)

    @EACH_FORM
    def test_solve_refuses_arrays_and_other_labels(self, form):
        """An array has no labels to check; a field on other labels raises."""
        S = COVS[form]
        with pytest.raises(TypeError, match="no labels"):
            S.solve(np.ones(len(S.space)))
        with pytest.raises(ValueError, match="same labels"):
            S.solve(Field(np.ones(2), OTHER))

    @EACH_FORM
    def test_logdet(self, form):
        """``logdet`` is ``np.linalg.slogdet``'s."""
        S = COVS[form]
        sign, expected = np.linalg.slogdet(S.dense())
        assert sign == 1.0
        assert S.logdet() == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        "form",
        [
            "dense",
            "diagonal",
            "sparse",
        ],
    )
    def test_logdet_of_an_indefinite_covariance_raises(self, form):
        """A covariance that is not positive-definite has no log-determinant."""
        A = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
        data = {
            "dense": A,
            "diagonal": DiagonalOperator([1.0, -1.0]),
            "sparse": sp.csr_array(A),
        }[form]
        S = Covariance(data, SITES)
        assert isinstance(S.data, FORM_TYPES[form])
        with pytest.raises(np.linalg.LinAlgError):
            S.logdet()

    @EACH_FORM
    def test_quadratic_form(self, form):
        """``x^T S^-1 x``, with `x` in any order."""
        S = COVS[form]
        x = np.random.default_rng(8).standard_normal(len(S.space))
        expected = x @ np.linalg.solve(S.dense(), x)
        assert S.quadratic_form(Field(x, S.space)) == pytest.approx(expected, rel=1e-10)
        other = _shuffled(S.space)
        shuffled = Field(_values_on(x, S.space, other), other)
        assert S.quadratic_form(shuffled) == pytest.approx(expected, rel=1e-10)

    @EACH_FORM
    def test_sqrt(self, form):
        """``L L^T = S``; a diagonal or block diagonal covariance keeps its form."""
        S = COVS[form]
        L = S.sqrt()
        assert type(L) is Operator
        assert L.rows is S.space and L.cols is S.space
        Ld = L.dense()
        np.testing.assert_allclose(Ld @ Ld.T, S.dense(), rtol=1e-12, atol=1e-12)
        if form in ("diagonal", "block"):
            assert isinstance(L.data, FORM_TYPES[form])

    def test_sqrt_of_an_indefinite_covariance_raises(self):
        """A covariance that is not positive-definite has no square root."""
        with pytest.raises(np.linalg.LinAlgError):
            Covariance(np.array([[1.0, 2.0], [2.0, 1.0]]), SITES).sqrt()


class TestArithmeticAndBlocks:
    """What stays a covariance: sums of covariances, scalars, diagonal blocks."""

    @EACH_FORM
    def test_sum_of_covariances(self, form):
        """The sum or difference of two covariances, in any order, is a covariance."""
        S = COVS[form]
        other = _shuffled(S.space)
        T = Covariance.diagonal(1.0, other)
        total = S + T
        assert isinstance(total, Covariance) and total.space is S.space
        assert total.name is None  # the names differ
        np.testing.assert_allclose(total.dense(), S.dense() + np.eye(len(S.space)))
        difference = S - T
        assert isinstance(difference, Covariance)
        np.testing.assert_allclose(difference.dense(), S.dense() - np.eye(len(S.space)))

    def test_sum_with_an_operator_is_an_operator(self):
        """A covariance plus an operator that is not one is an operator."""
        S = COVS["dense"]
        total = S + Operator(np.eye(6), CELLS, CELLS)
        assert type(total) is Operator

    def test_name_kept_when_both_have_it(self):
        """A sum keeps the name both parts share."""
        S = COVS["dense"]
        assert (S + S).name == "prior_error"

    @EACH_FORM
    def test_scalars_keep_a_covariance(self, form):
        """``*``, ``/`` and unary ``-`` give a covariance in the same form."""
        S = COVS[form]
        for result, expected in [
            (S * 2.0, 2.0 * S.dense()),
            (3.0 * S, 3.0 * S.dense()),
            (S / 4.0, S.dense() / 4.0),
        ]:
            assert isinstance(result, Covariance) and result.space is S.space
            assert isinstance(result.data, FORM_TYPES[form])
            np.testing.assert_allclose(result.dense(), expected, rtol=1e-14)
        assert isinstance(-S, Covariance)

    def test_diagonal_block_is_a_covariance(self):
        """``cov[b, b]`` is a covariance on that block; ``cov[a, b]`` an operator."""
        S = COVS["block"]
        flux = S["flux", "flux"]
        assert isinstance(flux, Covariance) and flux.space == CELLS
        np.testing.assert_array_equal(flux.dense(), S.dense()[:6, :6])
        bias = S["bias", "bias"]
        assert isinstance(bias, Covariance) and bias.space == SITES
        np.testing.assert_array_equal(bias.dense(), np.diag([0.5, 2.0]))
        cross = S["flux", "bias"]
        assert type(cross) is Operator
        assert cross.shape == (6, 2) and not cross.dense().any()

    def test_diagonal_block_of_a_dense_covariance(self):
        """A block of a dense covariance on two blocks is the dense block."""
        space = Space({"flux": CELLS.select("flux"), "bias": SITES.select("bias")})
        A = _spd(np.random.default_rng(9), 8)
        S = Covariance(A, space)
        bias = S["bias", "bias"]
        assert isinstance(bias, Covariance)
        np.testing.assert_array_equal(bias.dense(), A[6:, 6:])
        with pytest.raises(KeyError, match="No block named 'other'"):
            S["other", "other"]

    @EACH_FORM
    def test_align(self, form):
        """``align`` on the same space in another order is a covariance, matched by label."""
        S = COVS[form]
        other = _shuffled(S.space)
        T = S.align(other, other)
        assert isinstance(T, Covariance) and T.space is other
        p = _values_on(np.arange(len(S.space)), S.space, other)
        np.testing.assert_array_equal(T.dense(), S.dense()[np.ix_(p, p)])

    @EACH_FORM
    def test_pickle(self, form):
        """A pickle keeps the form, the space and the name."""
        S = COVS[form]
        S.solve(Field(np.ones(len(S.space)), S.space))  # fill any cache
        loaded = pickle.loads(pickle.dumps(S))
        assert isinstance(loaded, Covariance)
        assert isinstance(loaded.data, FORM_TYPES[form])
        assert loaded.space == S.space and loaded.name == S.name
        assert loaded.cols is loaded.rows
        np.testing.assert_array_equal(loaded.dense(), S.dense())


class TestFromBuilder:
    """A builder's frame is read by ``Covariance.from_frame``."""

    def test_builder_frame_becomes_a_covariance(self):
        """``build`` returns a dense frame; ``from_frame`` labels it."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        frame = (DiagonalError("e1", 1.0) + DiagonalError("e2", 0.5)).build(idx)
        assert not any(isinstance(d, pd.SparseDtype) for d in frame.dtypes)
        S = Covariance.from_frame(frame)
        assert S.space == Space.from_index(idx)
        np.testing.assert_array_equal(S.variances.values, [1.5, 1.5, 1.5])


# ---------------------------------------------------------------------------
# The error components and the builder
# ---------------------------------------------------------------------------


class TestDiagonalError:
    """Tests for DiagonalError component."""

    def test_diagonal_error_with_series(self):
        """Test DiagonalError with pd.Series variances."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        variances = pd.Series([1.0, 2.0, 3.0], index=idx)
        component = DiagonalError("test_error", variances)

        result = component.build(idx)

        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result.values), [1.0, 2.0, 3.0])
        # Should be diagonal
        off_diag = result.values - np.diag(np.diag(result.values))
        assert np.allclose(off_diag, 0.0)

    def test_diagonal_error_with_scalar(self):
        """Test DiagonalError with scalar variance."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        component = DiagonalError("test_error", 5.0)

        result = component.build(idx)

        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result.values), 5.0)

    def test_diagonal_error_with_multiindex(self):
        """Test DiagonalError with MultiIndex."""
        idx = pd.MultiIndex.from_product(
            [["b1", "b2"], ["x", "y"], [1, 2]], names=["block", "loc", "time"]
        )
        variances = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=idx)
        component = DiagonalError("test_error", variances)

        result = component.build(idx)

        assert result.shape == (8, 8)
        assert np.allclose(
            np.diag(result.values), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )

    def test_diagonal_error_labels_both_axes_with_the_index(self):
        """The built frame is the dense diagonal on the index it was asked for."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "c"), ("b1", "a"), ("b1", "b")], names=["block", "x"]
        )

        result = DiagonalError("test_error", 5.0).build(idx)

        expected = pd.DataFrame(5.0 * np.eye(3), index=idx, columns=idx)
        pd.testing.assert_frame_equal(result, expected)

    def test_diagonal_error_name(self):
        """Test DiagonalError name attribute."""
        component = DiagonalError("my_error", 1.0)
        assert component.name == "my_error"


class TestBlockDecayError:
    """Tests for BlockDecayError component."""

    def test_block_decay_error_simple(self):
        """Test BlockDecayError with simple block structure."""
        # Create data with blocks
        idx = pd.MultiIndex.from_product(
            [["block1"], ["block1", "block2"], [0, 1, 2]],
            names=["outer", "block", "idx"],
        )
        variances = pd.Series(np.ones(6), index=idx)

        # Correlation function that creates exponential decay
        def corr_func(coords):
            n = len(coords)
            C = np.eye(n)
            # Add some off-diagonal correlations
            for i in range(n):
                for j in range(n):
                    if i != j:
                        C[i, j] = np.exp(-0.5 * abs(i - j))
            return C

        component = BlockDecayError("block_decay", variances, ["block"], corr_func)

        result = component.build(idx)

        assert result.shape == (6, 6)
        # Check symmetry
        assert np.allclose(result.values, result.values.T)
        # Check that it's positive definite (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(result.values)
        assert np.all(eigenvalues > -1e-10)

    def test_block_decay_error_multiple_blocks(self):
        """Test BlockDecayError with multiple block columns."""
        idx = pd.MultiIndex.from_product(
            [["b1"], ["A", "B"], ["x", "y"], [1, 2]],
            names=["block", "region", "type", "time"],
        )
        variances = pd.Series(np.ones(8), index=idx)

        def simple_corr(coords):
            return np.eye(len(coords))

        component = BlockDecayError(
            "multi_block", variances, ["region", "type"], simple_corr
        )

        result = component.build(idx)

        assert result.shape == (8, 8)
        assert np.allclose(np.diag(result.values), 1.0)


class TestKroneckerError:
    """Tests for KroneckerError component."""

    def test_kronecker_error_basic(self):
        """Test KroneckerError with basic 2D grid."""
        idx = pd.MultiIndex.from_product(
            [["b1"], [0.0, 1.0], [0.0, 1.0]], names=["block", "x", "y"]
        )
        variances = pd.Series(np.ones(4), index=idx)

        # Simple exponential kernel
        def kernel(coords):
            n = len(coords)
            C = np.eye(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        C[i, j] = np.exp(
                            -0.5 * abs(coords.iloc[i, 0] - coords.iloc[j, 0])
                        )
            return C

        marginal_kernels = [("x", kernel), ("y", kernel)]
        component = KroneckerError("kronecker", variances, marginal_kernels)

        result = component.build(idx)

        assert result.shape == (4, 4)
        # Check symmetry
        assert np.allclose(result.values, result.values.T)

    def test_kronecker_error_3d_grid(self):
        """Test KroneckerError with 3D grid."""
        idx = pd.MultiIndex.from_product(
            [["b1"], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            names=["block", "x", "y", "time"],
        )
        variances = pd.Series(np.ones(8), index=idx)

        def identity_kernel(coords):
            return np.eye(len(coords))

        marginal_kernels = [
            ("x", identity_kernel),
            ("y", identity_kernel),
            ("time", identity_kernel),
        ]
        component = KroneckerError("3d_kronecker", variances, marginal_kernels)

        result = component.build(idx)

        assert result.shape == (8, 8)
        # With identity kernels and unit variances, should be identity matrix
        assert np.allclose(result.values, np.eye(8))

    def test_kronecker_error_with_string_dims(self):
        """Test KroneckerError with string dimension names."""
        idx = pd.MultiIndex.from_product(
            [["b1"], [0, 1], [0, 1]], names=["block", "x", "y"]
        )
        variances = pd.Series(np.ones(4), index=idx)

        def identity_kernel(coords):
            return np.eye(len(coords))

        # Test with single string instead of list
        marginal_kernels = [("x", identity_kernel), ("y", identity_kernel)]
        component = KroneckerError("test", variances, marginal_kernels)

        result = component.build(idx)

        assert result.shape == (4, 4)


TIMES = pd.date_range("2024-01-01", periods=3, freq="D")
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


def _product():
    return pd.MultiIndex.from_product([TIMES, LATS, LONS], names=["time", "lat", "lon"])


def _variances(index, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(0.5, 2.0, len(index)), index=index)


def _build(kernels, variances, index):
    return KroneckerError("prior", variances, kernels).build(index)


def _old_formula(kernels, variances, index):
    """What build computed before #76: np.kron of the marginals, scaled by std."""
    frame = index.to_frame(index=False)
    marginals = []
    for dims, kernel in kernels:
        dims = [dims] if isinstance(dims, str) else dims
        marginals.append(kernel(frame[dims].drop_duplicates()))
    std = np.sqrt(variances.reindex(index).to_numpy())
    return std[:, None] * reduce(np.kron, marginals) * std[None, :]


def _direct_formula(kernels, variances, index):
    """std_i std_j prod_k C_k(row i, row j), one pair of rows at a time."""
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


def _relabel(S, order):
    """S with its index levels put in ``order`` (the same entries, by label)."""
    index = S.index.reorder_levels(order)
    return pd.DataFrame(S.to_numpy(), index=index, columns=index)


class TestKroneckerErrorIndexOrder:
    """KroneckerError.build is right for any row order and subset (#76)."""

    @pytest.mark.parametrize("kernels", [KERNELS_2, KERNELS_3], ids=["2", "3"])
    def test_product_order_matches_kron(self, kernels):
        """On the product in the kernels' order the result is np.kron's."""
        index = _product()
        variances = _variances(index)
        result = _build(kernels, variances, index)
        np.testing.assert_allclose(
            result.to_numpy(),
            _old_formula(kernels, variances, index),
            rtol=0,
            atol=1e-15,
        )
        assert result.index.equals(index)
        assert result.columns.equals(index)

    @pytest.mark.parametrize("kernels", [KERNELS_2, KERNELS_3], ids=["2", "3"])
    @pytest.mark.parametrize("seed", range(4))
    def test_any_row_permutation(self, kernels, seed):
        """Shuffled rows give the same entries, label by label."""
        index = _product()
        variances = _variances(index)
        expected = _build(kernels, variances, index)

        rng = np.random.default_rng(seed)
        shuffled = index[rng.permutation(len(index))]
        result = _build(kernels, variances, shuffled)

        assert result.index.equals(shuffled)
        np.testing.assert_allclose(
            result.reindex(index=index, columns=index).to_numpy(),
            expected.to_numpy(),
            rtol=0,
            atol=1e-15,
        )

    @pytest.mark.parametrize("kernels", [KERNELS_2, KERNELS_3], ids=["2", "3"])
    @pytest.mark.parametrize("order", [["lat", "lon", "time"], ["lon", "lat", "time"]])
    def test_space_first_orders(self, kernels, order):
        """An index sorted by space first, as #76 reports, gives the same entries."""
        index = _product()
        variances = _variances(index)
        expected = _relabel(_build(kernels, variances, index), order)

        other = index.reorder_levels(order).sort_values()
        result = _build(kernels, variances, other)

        np.testing.assert_allclose(
            result.to_numpy(),
            expected.reindex(index=other, columns=other).to_numpy(),
            rtol=0,
            atol=1e-15,
        )

    @pytest.mark.parametrize("kernels", [KERNELS_2, KERNELS_3], ids=["2", "3"])
    def test_same_cells_dropped_at_every_time(self, kernels):
        """A product with the same cells left out at every time, as slv builds."""
        index = _product()
        lat = index.get_level_values("lat")
        lon = index.get_level_values("lon")
        subset = index[~((lat == 40.5) & (lon == -112.0))]
        variances = _variances(index)

        result = _build(kernels, variances, subset).to_numpy()

        np.testing.assert_allclose(
            result, _direct_formula(kernels, variances, subset), rtol=1e-14, atol=0
        )
        if kernels is KERNELS_2:
            # Time by the remaining cells is still a product, and the result is
            # what it was before #76.
            np.testing.assert_allclose(
                result, _old_formula(kernels, variances, subset), rtol=0, atol=1e-15
            )

    @pytest.mark.parametrize("kernels", [KERNELS_2, KERNELS_3], ids=["2", "3"])
    def test_ragged_subset(self, kernels):
        """A cell missing at one time only (this raised before #76)."""
        index = _product()
        missing = (TIMES[1], 40.5, -112.0)
        subset = index[[label != missing for label in index]]
        variances = _variances(index)

        result = _build(kernels, variances, subset)

        assert result.shape == (len(index) - 1, len(index) - 1)
        np.testing.assert_allclose(
            result.to_numpy(),
            _direct_formula(kernels, variances, subset),
            rtol=1e-14,
            atol=0,
        )

    def test_ragged_shuffled_subset_with_scalar_variance(self):
        """A ragged subset in shuffled order, with one variance for every row."""
        index = _product()
        rng = np.random.default_rng(7)
        subset = index[rng.permutation(len(index))[:-4]]
        variances = pd.Series(2.5, index=index)

        result = _build(KERNELS_2, 2.5, subset)

        np.testing.assert_allclose(
            result.to_numpy(),
            _direct_formula(KERNELS_2, variances, subset),
            rtol=1e-14,
            atol=0,
        )

    def test_variances_on_some_levels(self):
        """A variance Series indexed by (lat, lon) is aligned onto every time."""
        cells = pd.MultiIndex.from_product([LATS, LONS], names=["lat", "lon"])
        cell_variances = pd.Series(np.arange(1.0, 7.0), index=cells, name="var")
        other = _product().reorder_levels(["lon", "lat", "time"]).sort_values()

        result = _build(KERNELS_2, cell_variances, other)

        row_variances = pd.Series(
            [cell_variances[(lat, lon)] for lon, lat, _ in other], index=other
        )
        np.testing.assert_allclose(
            result.to_numpy(),
            _direct_formula(KERNELS_2, row_variances, other),
            rtol=1e-14,
            atol=0,
        )

    def test_kernel_receives_unique_coords_in_first_appearance_order(self):
        """Each kernel gets a DataFrame of the unique coordinates of its levels."""
        seen = []

        def recording_kernel(coords):
            seen.append(coords.copy())
            return np.eye(len(coords))

        index = _product().reorder_levels(["lat", "lon", "time"]).sort_values()
        KroneckerError(
            "prior",
            1.0,
            [("time", recording_kernel), (["lat", "lon"], recording_kernel)],
        ).build(index)

        times, cells = seen
        assert list(times.columns) == ["time"]
        assert list(times["time"]) == list(TIMES)
        assert list(cells.columns) == ["lat", "lon"]
        assert list(cells.itertuples(index=False, name=None)) == [
            (lat, lon) for lat in LATS for lon in LONS
        ]

    def test_repeated_coordinates_raise(self):
        """Two blocks on the same grid would be perfectly correlated: raise."""
        index = pd.MultiIndex.from_product(
            [["a", "b"], TIMES, LATS, LONS], names=["block", "time", "lat", "lon"]
        )
        with pytest.raises(ValueError, match="its own combination"):
            _build(KERNELS_2, 1.0, index)

    def test_kernel_with_wrong_shape_raises(self):
        """A kernel must return one row and column per unique coordinate."""
        kernels = [
            ("time", lambda coords: np.eye(2)),
            ("lat", _line_kernel("lat", 1.0)),
        ]
        index = pd.MultiIndex.from_product([TIMES, LATS], names=["time", "lat"])
        with pytest.raises(ValueError, match="expected \\(3, 3\\)"):
            _build(kernels, 1.0, index)

    def test_no_kernels_raise(self):
        """At least one marginal kernel is needed."""
        with pytest.raises(ValueError, match="at least one marginal kernel"):
            _build([], 1.0, _product())


class TestErrorComponentAddition:
    """Tests for ErrorComponent addition and composition."""

    def test_add_two_error_components(self):
        """Test adding two ErrorComponent instances."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        var1 = pd.Series([1.0, 1.0, 1.0], index=idx)
        var2 = pd.Series([2.0, 2.0, 2.0], index=idx)

        comp1 = DiagonalError("error1", var1)
        comp2 = DiagonalError("error2", var2)

        result = comp1 + comp2

        assert isinstance(result, CovarianceBuilder)
        assert len(result.components) == 2

    def test_add_component_to_builder(self):
        """Test adding a component to a CovarianceBuilder."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        var1 = pd.Series([1.0, 1.0, 1.0], index=idx)
        var2 = pd.Series([2.0, 2.0, 2.0], index=idx)

        comp1 = DiagonalError("error1", var1)
        comp2 = DiagonalError("error2", var2)

        builder = comp1 + comp2
        result = builder + DiagonalError("error3", var1)

        assert isinstance(result, CovarianceBuilder)
        assert len(result.components) == 3

    def test_add_invalid_type_raises(self):
        """Test that adding invalid type raises error."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        var = pd.Series([1.0, 1.0, 1.0], index=idx)
        comp = DiagonalError("error", var)

        with pytest.raises(TypeError):
            comp + "invalid"


class TestCovarianceBuilder:
    """Tests for CovarianceBuilder class."""

    def test_builder_with_single_component(self):
        """Test CovarianceBuilder with single component."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        var = pd.Series([1.0, 2.0, 3.0], index=idx)
        component = DiagonalError("error", var)

        builder = CovarianceBuilder([component])
        result = builder.build(idx)

        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result.values), [1.0, 2.0, 3.0])

    def test_builder_with_multiple_components(self):
        """Test CovarianceBuilder with multiple components."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        var1 = pd.Series([1.0, 1.0, 1.0], index=idx)
        var2 = pd.Series([2.0, 2.0, 2.0], index=idx)

        comp1 = DiagonalError("error1", var1)
        comp2 = DiagonalError("error2", var2)

        builder = CovarianceBuilder([comp1, comp2])
        result = builder.build(idx)

        # Sum of diagonal elements should be [3, 3, 3]
        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result.values), [3.0, 3.0, 3.0])

    def test_builder_no_components_raises(self):
        """Test that empty CovarianceBuilder raises error."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        builder = CovarianceBuilder([])

        with pytest.raises(ValueError, match="No components"):
            builder.build(idx)

    def test_builder_composition_via_addition(self):
        """Test building CovarianceBuilder through addition."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        var1 = pd.Series([1.0, 1.0, 1.0], index=idx)
        var2 = pd.Series([0.5, 0.5, 0.5], index=idx)

        comp1 = DiagonalError("error1", var1)
        comp2 = DiagonalError("error2", var2)

        # Build through addition
        builder = comp1 + comp2
        result = builder.build(idx)

        # Diagonals should be summed
        assert np.allclose(np.diag(result.values), [1.5, 1.5, 1.5])

    def test_builder_mixed_components(self):
        """Test CovarianceBuilder with different component types."""
        idx = pd.MultiIndex.from_product(
            [["b1"], [0.0, 1.0], [0, 1]], names=["block", "x", "time"]
        )

        var = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
        diag_comp = DiagonalError("diagonal", var)

        def identity_kernel(coords):
            return np.eye(len(coords))

        kron_comp = KroneckerError(
            "kronecker", 1.0, [("x", identity_kernel), ("time", identity_kernel)]
        )

        builder = CovarianceBuilder([diag_comp, kron_comp])
        result = builder.build(idx)

        assert result.shape == (4, 4)
        # Result should be symmetric
        assert np.allclose(result.values, result.values.T)


class TestVarianceAlignmentInComponents:
    """Tests for variance alignment in ErrorComponent._align_variances."""

    def test_align_variances_with_matching_series(self):
        """Test alignment with matching Series index."""
        idx = pd.MultiIndex.from_product(
            [["b1"], ["a", "b"], [1, 2]], names=["block", "x", "y"]
        )
        variances = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        component = DiagonalError("test", variances)

        aligned = component._align_variances(idx)

        assert isinstance(aligned, pd.Series)
        assert len(aligned) == 4
        assert aligned.isna().sum() == 0  # No NaNs

    def test_align_variances_with_partial_alignment(self):
        """Test alignment when Series has subset of target index."""
        target_idx = pd.MultiIndex.from_product(
            [["b1"], ["a", "b"], [1, 2], ["x", "y"]], names=["block", "x", "y", "z"]
        )
        partial_idx = pd.MultiIndex.from_product(
            [["b1"], ["a", "b"], [1, 2]], names=["block", "x", "y"]
        )
        variances = pd.Series(2.0, index=partial_idx)

        component = DiagonalError("test", variances)
        aligned = component._align_variances(target_idx)

        assert len(aligned) == len(target_idx)

    def test_align_variances_scalar(self):
        """Test alignment with scalar variance."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        component = DiagonalError("test", 5.0)

        aligned = component._align_variances(idx)

        assert len(aligned) == 3
        assert np.allclose(aligned.to_numpy(), 5.0)

    def test_align_variances_with_missing_names_raises(self):
        """Test that unnamed index levels raise error."""
        idx = pd.MultiIndex.from_product(
            [["b1"], ["a", "b"], [1, 2]], names=[None, "y", None]
        )
        variances = pd.Series([1.0, 2.0], index=idx.droplevel([0, 2]).drop_duplicates())

        component = DiagonalError("test", variances)

        with pytest.raises(ValueError, match="All levels.*must be named"):
            component._align_variances(idx)


class TestCovarianceBuilderArithmetic:
    """Tests for CovarianceBuilder arithmetic operators."""

    def _make_idx(self):
        return pd.MultiIndex.from_tuples(
            [("b", "a"), ("b", "b"), ("b", "c")], names=["block", "x"]
        )

    def test_add_two_components_returns_builder(self):
        """Test that adding two error components returns a CovarianceBuilder."""
        c1 = DiagonalError("c1", 1.0)
        c2 = DiagonalError("c2", 2.0)
        result = c1 + c2
        assert isinstance(result, CovarianceBuilder)
        assert len(result.components) == 2

    def test_builder_add_component(self):
        """Test adding a component to a builder."""
        c1 = DiagonalError("c1", 1.0)
        c2 = DiagonalError("c2", 2.0)
        c3 = DiagonalError("c3", 3.0)
        builder = c1 + c2
        result = builder + c3
        assert isinstance(result, CovarianceBuilder)
        assert len(result.components) == 3

    def test_builder_radd_component(self):
        """Test right-adding a component to a builder."""
        c1 = DiagonalError("c1", 1.0)
        c2 = DiagonalError("c2", 2.0)
        c3 = DiagonalError("c3", 3.0)
        builder = c2 + c3
        result = c1 + builder  # c1.__add__(builder) → builder prepend
        assert isinstance(result, CovarianceBuilder)
        assert len(result.components) == 3

    def test_builder_add_builder(self):
        """Test adding two builders together."""
        c1 = DiagonalError("c1", 1.0)
        c2 = DiagonalError("c2", 2.0)
        c3 = DiagonalError("c3", 3.0)
        c4 = DiagonalError("c4", 4.0)
        b1 = c1 + c2
        b2 = c3 + c4
        result = b1 + b2
        assert isinstance(result, CovarianceBuilder)
        assert len(result.components) == 4

    def test_combined_builder_builds_correctly(self):
        """Test that combined builder builds correctly."""
        idx = self._make_idx()
        c1 = DiagonalError("c1", 1.0)
        c2 = DiagonalError("c2", 2.0)
        builder = c1 + c2
        mat = builder.build(idx)
        assert mat.shape == (3, 3)
        np.testing.assert_allclose(np.diag(mat.values), 3.0)


class TestCovarianceRepr:
    """Tests for __repr__ on covariance builder and error components."""

    def test_diagonal_error_repr(self):
        """Test string representation of DiagonalError."""
        c = DiagonalError("sigma", 1.0)
        assert repr(c) == "DiagonalError(name='sigma')"

    def test_covariance_builder_repr(self):
        """Test string representation of CovarianceBuilder."""
        c1 = DiagonalError("c1", 1.0)
        c2 = DiagonalError("c2", 2.0)
        builder = c1 + c2
        r = repr(builder)
        assert "CovarianceBuilder" in r
        assert "n_components=2" in r
        assert "c1" in r
        assert "c2" in r
