"""
Tests for the covariance module.

``Covariance`` (a symmetric operator on one space, kept in its cheapest form)
against NumPy on small random positive-definite matrices, in each form;
``block_diag`` with covariances and models by block; and the covariance models
pinned, bit for bit, to the matrices the error components and
``CovarianceBuilder`` built before the models replaced them.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp

from fips import (
    Constant,
    Covariance,
    Exponential,
    Independent,
    Kronecker,
    Operator,
    Space,
    Vector,
)
from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
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

    def test_from_blocks_with_a_label_missing_raises(self):
        """A label a block's own pair lacks would get variance 0: it raises, naming the labels."""
        flux = CELLS.select("flux")
        part = _frame(np.eye(4), flux[:4])
        with pytest.raises(
            ValueError, match=r"2 of the 6 labels of block 'flux': 'e', 'f'"
        ):
            Covariance.from_blocks({("flux", "flux"): part}, rows=CELLS, cols=CELLS)
        # a label in the columns but not in the rows is missing too
        wide = _frame(np.eye(6)[:4], flux[:4], flux)
        with pytest.raises(ValueError, match="'e', 'f'"):
            Covariance.from_blocks({("flux", "flux"): wide}, rows=CELLS, cols=CELLS)

    def test_from_blocks_names_a_few_labels_then_a_count(self):
        """A long list of missing labels is cut short with a count."""
        labels = pd.Index(range(20), name="cell")
        space = Space({"flux": labels})
        part = _frame(np.eye(2), labels[:2])
        with pytest.raises(ValueError, match=r"18 of the 20 labels.*and \d+ more"):
            Covariance.from_blocks({("flux", "flux"): part}, rows=space, cols=space)

    def test_from_blocks_with_a_block_lacking_its_own_pair_raises(self):
        """A block with only cross pairs has no variances at all."""
        flux, bias = CELLS.select("flux"), SITES.select("bias")
        space = Space({"flux": flux, "bias": bias})
        blocks = {
            ("flux", "flux"): _frame(np.eye(6), flux),
            ("bias", "flux"): _frame(np.zeros((2, 6)), bias, flux),
            ("flux", "bias"): _frame(np.zeros((6, 2)), flux, bias),
        }
        with pytest.raises(
            ValueError, match=r"Block 'bias' has no \('bias', 'bias'\) pair"
        ):
            Covariance.from_blocks(blocks, rows=space, cols=space)

    def test_from_blocks_with_labels_in_another_order_is_fine(self):
        """Every label present, in any order, is not missing."""
        rng = np.random.default_rng(5)
        flux = CELLS.select("flux")
        A = _spd(rng, 6)
        order = rng.permutation(6)
        S = Covariance.from_blocks(
            {("flux", "flux"): _frame(A[np.ix_(order, order)], flux[order])},
            rows=CELLS,
            cols=CELLS,
        )
        np.testing.assert_allclose(S.dense(), A)

    def test_one_block_keeps_its_form(self):
        """A single form given as the one block stays that form."""
        op = COVS["kronecker"].data
        S = Covariance.from_blocks({("flux", "flux"): op}, rows=CELLS, cols=CELLS)
        assert isinstance(S, Covariance) and S.data is op


class TestOwnsItsData:
    """A covariance copies its matrix and makes the copy read-only, as an Operator does."""

    def test_editing_the_input_changes_nothing(self):
        """Writing into the matrix handed over leaves the covariance as it was."""
        A = _spd(np.random.default_rng(1), 6)
        S = Covariance(A, CELLS)
        before = S.dense()
        A.fill(-1.0)
        np.testing.assert_array_equal(S.dense(), before)
        buffer = np.asarray(S)
        assert not buffer.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            buffer[0, 0] = 99.0
        copy = np.array(S)
        copy[0, 0] = 99.0
        np.testing.assert_array_equal(S.dense(), before)

    def test_sparse_input_is_copied(self):
        """A sparse matrix handed over is copied, and the copy is read-only."""
        M = sp.diags_array(
            [np.full(5, 0.4), np.full(6, 2.0), np.full(5, 0.4)],
            offsets=[-1, 0, 1],
            format="csr",
        )
        S = Covariance(M, CELLS)
        before = S.dense()
        M.data.fill(-1.0)
        np.testing.assert_array_equal(S.dense(), before)
        assert not S.data.matrix.data.flags.writeable

    def test_diagonal_copies_its_variances(self):
        """``Covariance.diagonal`` copies an array of variances; a Vector's are read-only already."""
        variances = np.arange(1.0, 7.0)
        S = Covariance.diagonal(variances, CELLS)
        variances.fill(-1.0)
        np.testing.assert_array_equal(S.variances.values, np.arange(1.0, 7.0))
        assert not S.data.values.flags.writeable

    def test_block_diag_and_models_hold_read_only_arrays(self):
        """Covariances put side by side, and a model's build, hold read-only arrays."""
        flux = Covariance(_spd(np.random.default_rng(2), 6), CELLS)
        bias = Covariance.diagonal([1.0, 2.0], SITES)
        both = Covariance.block_diag(flux, bias)
        assert not both.data.blocks[0].array.flags.writeable
        assert not both.data.blocks[1].values.flags.writeable
        assert not Independent().with_std(0.5).build(CELLS).data.values.flags.writeable
        assert not Constant().with_std(0.5).build(CELLS).data.array.flags.writeable


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

    def test_vector(self):
        """A vector brings its own space."""
        vector = Vector(np.arange(1.0, 7.0), CELLS)
        S = Covariance.diagonal(vector)
        assert S.space is CELLS
        np.testing.assert_array_equal(S.dense(), np.diag(vector.values))

    def test_vector_on_a_space_in_another_order(self):
        """A vector and a space in another order: the variances follow the space."""
        values = np.arange(1.0, 7.0)
        other = _shuffled(CELLS)
        S = Covariance.diagonal(Vector(values, CELLS), other)
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
            (Vector(np.ones(6), CELLS), SITES, "same labels"),
        ],
        ids=["scalar", "array", "2D", "length", "vector"],
    )
    def test_errors(self, values, space, match):
        """No space for a scalar or array, a 2D array, a wrong length, a vector on other labels."""
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
        """The variances are the diagonal, as a vector on the space."""
        S = COVS[form]
        variances = S.variances
        assert isinstance(variances, Vector) and variances.space is S.space
        assert variances.name == "variance"
        np.testing.assert_allclose(variances.values, np.diag(S.dense()), rtol=1e-14)

    @EACH_FORM
    def test_solve_a_field(self, form):
        """``solve`` of a vector is ``np.linalg.solve``, on the covariance's space."""
        S = COVS[form]
        b = np.random.default_rng(5).standard_normal(len(S.space))
        x = S.solve(Vector(b, S.space))
        assert isinstance(x, Vector) and x.space is S.space
        np.testing.assert_allclose(x.values, np.linalg.solve(S.dense(), b), rtol=1e-10)

    @EACH_FORM
    def test_solve_a_field_in_another_order(self, form):
        """A right-hand side in another order is matched by label."""
        S = COVS[form]
        b = np.random.default_rng(6).standard_normal(len(S.space))
        other = _shuffled(S.space)
        x = S.solve(Vector(_values_on(b, S.space, other), other))
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
        """An array has no labels to check; a vector on other labels raises."""
        S = COVS[form]
        with pytest.raises(TypeError, match="no labels"):
            S.solve(np.ones(len(S.space)))
        with pytest.raises(ValueError, match="same labels"):
            S.solve(Vector(np.ones(2), OTHER))

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
        assert S.quadratic_form(Vector(x, S.space)) == pytest.approx(
            expected, rel=1e-10
        )
        other = _shuffled(S.space)
        shuffled = Vector(_values_on(x, S.space, other), other)
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
        S.solve(Vector(np.ones(len(S.space)), S.space))  # fill any cache
        loaded = pickle.loads(pickle.dumps(S))
        assert isinstance(loaded, Covariance)
        assert isinstance(loaded.data, FORM_TYPES[form])
        assert loaded.space == S.space and loaded.name == S.name
        assert loaded.cols is loaded.rows
        np.testing.assert_array_equal(loaded.dense(), S.dense())


class TestBlockDiagByName:
    """``Covariance.block_diag`` with parts named by block: covariances and models."""

    def test_covariances_by_name(self):
        """A covariance given as ``block=`` is on that one block."""
        S = Covariance.block_diag(
            flux=COVS["kronecker"], bias=COVS["block"]["bias", "bias"]
        )
        assert isinstance(S, Covariance) and S.space.names == ("flux", "bias")
        assert isinstance(S.data, BlockDiagonalOperator)
        np.testing.assert_array_equal(S.dense(), COVS["block"].dense())

    def test_covariance_on_another_block_raises(self):
        """The name must be the covariance's block."""
        with pytest.raises(ValueError, match="given as bias="):
            Covariance.block_diag(bias=COVS["dense"])

    def test_a_model_part_waits_for_the_space(self):
        """With a model, the result is a model that ``build`` puts on a space, in its order."""
        model = Covariance.block_diag(
            COVS["kronecker"], bias=Independent().with_std(2.0), name="prior_error"
        )
        assert not isinstance(model, Covariance)
        space = Space({"bias": SITES.select("bias"), "flux": CELLS.select("flux")})
        S = model.build(space)
        assert S.name == "prior_error" and S.space == space
        assert isinstance(S.data, BlockDiagonalOperator)
        assert isinstance(S.data.blocks[1], KroneckerOperator)
        np.testing.assert_array_equal(
            S.dense(),
            scipy.linalg.block_diag(4.0 * np.eye(2), COVS["kronecker"].dense()),
        )

    def test_a_covariance_part_is_aligned_by_label(self):
        """A covariance part is put in the order of the space's labels."""
        model = Covariance.block_diag(
            flux=COVS["dense"], bias=Independent().with_std(1.0)
        )
        flux = _shuffled(CELLS)
        S = model.build(
            Space({"flux": flux.select("flux"), "bias": SITES.select("bias")})
        )
        np.testing.assert_array_equal(
            S["flux", "flux"].dense(), COVS["dense"].align(flux, flux).dense()
        )

    def test_errors(self):
        """Blocks the space lacks or has beyond the parts, and non-models, raise."""
        model = Covariance.block_diag(flux=Independent().with_std(1.0))
        with pytest.raises(ValueError, match="same blocks"):
            model.build(COVS["block"].space)
        with pytest.raises(ValueError, match="same blocks"):
            Covariance.block_diag(
                flux=Independent().with_std(1.0), other=Independent().with_std(1.0)
            ).build(CELLS)
        with pytest.raises(TypeError, match="as bias="):
            Covariance.block_diag(flux=Independent().with_std(1.0), bias=np.eye(2))
        with pytest.raises(TypeError, match="in order"):
            Covariance.block_diag(Independent().with_std(1.0))
        with pytest.raises(ValueError, match="block named 'flux'"):
            Covariance.block_diag(COVS["dense"], flux=Independent().with_std(1.0))

    def test_a_part_given_as_name_raises(self):
        """``name=`` is the covariance's name: a part for a block called 'name' raises and says how."""
        with pytest.raises(TypeError, match="block called 'name'"):
            Covariance.block_diag(
                name=Independent().with_std(1.0), x=Independent().with_std(2.0)
            )
        with pytest.raises(TypeError, match="block called 'name'"):
            Covariance.block_diag(COVS["dense"], name=COVS["diagonal"])

    def test_a_block_called_name_given_in_order(self):
        """The way the message gives: build the part on its block, give it in order."""
        space = Space({"name": pd.Index([1], name="i"), "x": pd.Index([2], name="i")})
        part = Independent().with_std(1.0).build(Space({"name": space.select("name")}))
        S = Covariance.block_diag(part, x=Independent().with_std(2.0)).build(space)
        np.testing.assert_array_equal(S.variances.values, [1.0, 4.0])


class TestSymmetry:
    """A covariance refuses a matrix that is not symmetric, in every way in."""

    def _upper(self) -> np.ndarray:
        """A symmetric positive-definite matrix given by its upper triangle only."""
        return np.triu(_spd(np.random.default_rng(7), 6))

    def test_dense_upper_triangle_raises(self):
        """An upper triangle is not the covariance; the message names the entries' labels."""
        with pytest.raises(ValueError, match=r"must be symmetric.*'flux'.*1e-10"):
            Covariance(self._upper(), CELLS)

    def test_sparse_raises(self):
        """A sparse matrix is checked as sparse."""
        with pytest.raises(ValueError, match="must be symmetric"):
            Covariance(sp.csr_array(self._upper()), CELLS)

    def test_from_frame_and_from_blocks_raise(self):
        """Both labeled ways in check too."""
        labels = CELLS.select("flux")
        with pytest.raises(ValueError, match="must be symmetric"):
            Covariance.from_frame(
                _frame(self._upper(), labels), row_block="flux", col_block="flux"
            )
        with pytest.raises(ValueError, match="must be symmetric"):
            Covariance.from_blocks({("flux", "flux"): _frame(self._upper(), labels)})

    def test_a_block_of_a_block_diagonal_form_raises(self):
        """A block diagonal form is checked block by block, labels at the right place."""
        form = BlockDiagonalOperator([np.eye(2), np.array([[1.0, 0.5], [0.0, 1.0]])])
        space = Space({"bias": SITES.select("bias"), "other": OTHER.select("other")})
        with pytest.raises(
            ValueError, match=r"\('other', nan, 1\).*\('other', nan, 2\)"
        ):
            Covariance(form, space)

    def test_roundoff_is_accepted(self):
        """A difference below 1e-10 of the largest entry is roundoff, not an error."""
        A = _spd(np.random.default_rng(8), 6)
        A[0, 1] += 1e-12 * np.abs(A).max()
        np.testing.assert_array_equal(Covariance(A, CELLS).dense(), A)

    def test_the_solve_the_review_found(self):
        """A non-symmetric prior covariance gave a solve that was not its closed form: it raises now."""
        S0 = np.array([[1.0, 0.9, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        space = Space({"flux": pd.Index(["a", "b", "c"], name="cell")})
        with pytest.raises(ValueError, match="0.9 and the one at"):
            Covariance(S0, space)

    def test_what_fips_computes_is_not_rechecked(self, monkeypatch):
        """
        Sums, reorderings, blocks and block diagonals of covariances skip the check.

        They are symmetric up to roundoff, which in a computed covariance (a
        posterior, ``W S W^T``) can pass 1e-10 of the largest entry.
        """
        import fips.covariance.covariance as module

        def refuse(form, space, start=0):
            raise AssertionError("checked again")

        monkeypatch.setattr(module, "_check_symmetric", refuse)
        dense = COVS["dense"]
        assert isinstance(dense + dense, Covariance)
        assert isinstance(2.0 * dense, Covariance)
        assert isinstance(dense.align(_shuffled(CELLS), _shuffled(CELLS)), Covariance)
        assert isinstance(COVS["block"]["bias", "bias"], Covariance)
        assert isinstance(Covariance.block_diag(COVS["block"]), Covariance)
        model = Independent().with_std(1.0) + dense
        assert isinstance(model.build(CELLS), Covariance)

    def test_a_covariance_is_refused_as_data(self):
        """``Covariance(cov, space)`` would ignore the labels: it points to align and data."""
        with pytest.raises(TypeError, match=r"align\(rows, cols\).*\.data"):
            Covariance(COVS["sparse"], CELLS)


# ---------------------------------------------------------------------------
# The models, pinned to the builders they replaced
# ---------------------------------------------------------------------------

#: The matrices ``DiagonalError``, ``BlockDecayError``, ``KroneckerError`` and
#: ``CovarianceBuilder`` built on the labels of ``_pinned_inputs``, with the code
#: of refactor/operator-spaces at 2cf777e, the last before the models replaced
#: them (variances given as ``std**2``; ``same_day`` grouped by block, site and
#: the date of ``obs_time``).
OLD = np.load(Path(__file__).parent / "data" / "old_builders.npz")


def _pinned_inputs():
    """The labels of the pinned cases: observations at two sites across midnight, and a grid."""
    times = pd.date_range("2024-01-01 20:00", periods=6, freq="h")
    obs = pd.MultiIndex.from_product(
        [["co2"], ["A", "B"], times], names=["block", "site", "obs_time"]
    )
    grid = pd.MultiIndex.from_product(
        [
            ["flux"],
            pd.date_range("2024-01-01", periods=4, freq="D"),
            [40.0, 40.5, 41.0],
            [-112.0, -111.5],
        ],
        names=["block", "time", "lat", "lon"],
    )
    lat, lon = grid.get_level_values("lat"), grid.get_level_values("lon")
    coverage = grid[~((lat == 40.5) & (lon == -112.0))]
    ragged = coverage[[0, 5, 3, 19, 7, 2, 11, 13, 1, 17, 8, 4, 16, 9, 12]]
    return obs, coverage, ragged


OBS, COVERAGE, RAGGED = _pinned_inputs()
STD = pd.Series(np.linspace(0.1, 0.6, len(OBS)), index=OBS.droplevel("block"))
SAME_DAY = ["site", "obs_time.date"]
MDM = {
    "diagonal_scalar": Independent().with_std(0.3),
    "diagonal_series": Independent().with_std(STD),
    "block_constant": Constant(within=SAME_DAY).with_std(0.0257),
    "block_decay": Exponential("obs_time", "2.8h", within=SAME_DAY).with_std(STD),
    "block_decay_interday": Exponential("obs_time", "7D", within="site").with_std(
        0.011
    ),
}
PRIOR = Kronecker(
    Exponential("time", "14D"), Exponential(["lat", "lon"], 20.0, metric="haversine")
)


def _same(actual: np.ndarray, expected: np.ndarray) -> None:
    """
    Equal to one rounding of the last bit (1e-14 relative).

    The fixture was written with pandas 3, whose time differences are in
    microseconds; pandas 2 divides nanoseconds, which can move the last bit.
    On one pandas version the models and the builders agree bit for bit.
    """
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)


class TestPinnedToTheOldBuilders:
    """Each covariance the old builders made, from a model: the same numbers."""

    @pytest.mark.parametrize("case", list(MDM))
    def test_mdm_components(self, case):
        """``DiagonalError`` is ``Independent``; ``BlockDecayError`` a kernel ``within`` groups."""
        S = MDM[case].build(Space.from_index(OBS))
        _same(S.dense(), OLD[case])

    def test_the_sum(self):
        """``CovarianceBuilder`` is a sum of models."""
        terms = list(MDM.values())
        total = terms[0] + terms[1] + terms[2] + terms[3] + terms[4]
        _same(total.build(Space.from_index(OBS)).dense(), OLD["builder_sum"])

    @pytest.mark.parametrize("case", ["kronecker_coverage", "kronecker_ragged"])
    def test_kronecker(self, case):
        """``KroneckerError`` is ``Kronecker``, kept as factors, on any row order and subset."""
        index = COVERAGE if case == "kronecker_coverage" else RAGGED
        std = np.linspace(0.5, 1.5, len(index))
        S = PRIOR.with_std(pd.Series(std, index=index.droplevel("block"))).build(
            Space.from_index(index)
        )
        assert isinstance(S.data, KroneckerOperator)
        _same(S.dense(), OLD[case])

    def test_independent_is_diagonal(self):
        """``Independent().with_std(std)`` is kept as its variances."""
        S = MDM["diagonal_series"].build(Space.from_index(OBS))
        assert isinstance(S.data, DiagonalOperator)
