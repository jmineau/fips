"""Test suite for fips matrix types (structures, covariance, operators)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from fips.covariance import CovarianceMatrix
from fips.matrix import Matrix, MatrixBlock
from fips.operators import ForwardOperator
from fips.vector import Block, Vector

# A scalar in each form a caller might pass: Python float and int, a NumPy scalar, a 0-d array
SCALARS = [5.0, 5, np.float64(5.0), np.array(5.0)]

MATRIX_TYPES = [Matrix, CovarianceMatrix, ForwardOperator]


@pytest.fixture
def index():
    """A block-level index whose labels are not in sorted order."""
    return pd.MultiIndex.from_tuples(
        [("state", "c"), ("state", "a"), ("state", "b")],
        names=["block", "site"],
    )


@pytest.fixture
def sites():
    """The same labels without a block level, as a MatrixBlock takes them."""
    return pd.Index(["c", "a", "b"], name="site")


def dense(structure):
    """The values of a 2D structure as a dense array."""
    return structure.values.toarray() if structure.is_sparse else structure.values


class TestDiagonal:
    """Tests for the `diagonal` constructor of the 2D structures."""

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    @pytest.mark.parametrize("sparse", [False, True])
    def test_scalar(self, matrix_type, scalar, sparse, index):
        """A scalar or 0-d array puts its value on the diagonal and zeros elsewhere."""
        matrix = matrix_type.diagonal(scalar, index=index, sparse=sparse)

        assert isinstance(matrix, matrix_type)
        assert matrix.is_sparse is sparse
        pd.testing.assert_index_equal(matrix.index, index)
        pd.testing.assert_index_equal(matrix.columns, index)
        np.testing.assert_array_equal(dense(matrix), 5.0 * np.eye(3))

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize("container", [list, tuple, np.array])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_values_use_supplied_index(self, matrix_type, container, sparse, index):
        """A 1D array, list or tuple is the diagonal and keeps its labels on both axes."""
        values = container([1.0, 2.0, 3.0])

        matrix = matrix_type.diagonal(values, index=index, sparse=sparse)

        assert isinstance(matrix, matrix_type)
        assert matrix.is_sparse is sparse
        pd.testing.assert_index_equal(matrix.index, index)
        pd.testing.assert_index_equal(matrix.columns, index)
        np.testing.assert_array_equal(dense(matrix), np.diag([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize("wrap", [lambda s: s, Vector], ids=["series", "vector"])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_labeled_values_supply_their_index(self, wrap, sparse, index):
        """A Series or a Vector is the diagonal, labeled by its own index."""
        series = pd.Series([1.0, 2.0, 3.0], index=index, name="variance")

        matrix = CovarianceMatrix.diagonal(wrap(series), sparse=sparse)

        pd.testing.assert_index_equal(matrix.index, index)
        pd.testing.assert_index_equal(matrix.columns, index)
        np.testing.assert_array_equal(dense(matrix), np.diag([1.0, 2.0, 3.0]))

    def test_labeled_values_accept_their_own_index(self, index):
        """Passing the index a Series already has changes nothing."""
        series = pd.Series([1.0, 2.0, 3.0], index=index)

        with_index = Matrix.diagonal(series, index=index)

        pd.testing.assert_frame_equal(with_index.data, Matrix.diagonal(series).data)

    def test_labeled_values_refuse_another_index(self, index):
        """A Series is not silently relabeled or reordered by a different index."""
        series = pd.Series([1.0, 2.0, 3.0], index=index)

        with pytest.raises(ValueError, match="carry their own index"):
            Matrix.diagonal(series, index=index[::-1])

    @pytest.mark.parametrize("size", [1, 3])
    def test_scalar_matches_values_of_the_same_number(self, size):
        """A scalar builds what 1D values of that number on every entry build."""
        index = pd.MultiIndex.from_product(
            [["state"], range(size)], names=["block", "i"]
        )

        from_scalar = Matrix.diagonal(5.0, index=index)
        from_list = Matrix.diagonal([5.0] * size, index=index)

        pd.testing.assert_frame_equal(from_scalar.data, from_list.data)

    @pytest.mark.parametrize("values", [5.0, [0.0, 2.0, 3.0]], ids=["scalar", "list"])
    def test_sparse_equals_dense(self, values, index):
        """The sparse diagonal holds the numbers of the dense one, zeros included."""
        sparse = CovarianceMatrix.diagonal(values, index=index, sparse=True)
        dense_ = CovarianceMatrix.diagonal(values, index=index)

        assert sparse.is_sparse and not dense_.is_sparse
        np.testing.assert_array_equal(sparse.values.toarray(), dense_.values)
        pd.testing.assert_frame_equal(sparse.to_dense().data, dense_.data.astype(float))

    def test_sparse_stores_only_the_diagonal(self):
        """A sparse diagonal stores n values, not n squared."""
        n = 500
        index = pd.MultiIndex.from_product([["state"], range(n)], names=["block", "i"])

        matrix = CovarianceMatrix.diagonal(2.0, index=index, sparse=True)

        assert matrix.values.nnz == n
        assert matrix.data.sparse.density == pytest.approx(1 / n)

    def test_variances_give_back_the_diagonal(self, index):
        """`diagonal` is the inverse of `CovarianceMatrix.variances`."""
        cov = CovarianceMatrix.diagonal([1.0, 2.0, 3.0], index=index)

        again = CovarianceMatrix.diagonal(cov.variances)

        pd.testing.assert_frame_equal(again.data, cov.data)

    def test_name_goes_to_the_constructor(self, index):
        """Keyword arguments other than `index` and `sparse` reach the constructor."""
        matrix = Matrix.diagonal(5.0, index=index, name="prior_error")

        assert matrix.name == "prior_error"

    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    def test_scalar_requires_an_index(self, scalar):
        """An unlabeled scalar says that it needs an index."""
        with pytest.raises(ValueError, match="requires an index"):
            Matrix.diagonal(scalar)

    @pytest.mark.parametrize("container", [list, tuple, np.array])
    def test_values_require_an_index(self, container):
        """Unlabeled 1D values say that they need an index."""
        with pytest.raises(ValueError, match="requires an index"):
            Matrix.diagonal(container([1.0, 2.0, 3.0]))

    def test_values_must_match_the_index_length(self, index):
        """Too few or too many values raise, naming both lengths."""
        with pytest.raises(ValueError, match="2 diagonal values.*length 3"):
            Matrix.diagonal([1.0, 2.0], index=index)

    def test_2d_values_are_refused(self, index):
        """A 2D array is a matrix, not a diagonal."""
        with pytest.raises(ValueError, match="scalar or 1D"):
            Matrix.diagonal(np.eye(3), index=index)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_nan_is_rejected(self, sparse, index):
        """A NaN value raises the NaN error rather than building a matrix."""
        with pytest.raises(ValueError, match="NaN"):
            Matrix.diagonal([1.0, np.nan, 3.0], index=index, sparse=sparse)

    def test_matrix_needs_a_block_level(self, sites):
        """A Matrix diagonal is validated like any Matrix."""
        with pytest.raises(ValueError, match="'block' level"):
            Matrix.diagonal(5.0, index=sites)

    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    @pytest.mark.parametrize("sparse", [False, True])
    def test_block_from_scalar(self, scalar, sparse, sites):
        """A MatrixBlock diagonal takes the block names its constructor takes."""
        block = MatrixBlock.diagonal(
            scalar, index=sites, row_block="state", col_block="state", sparse=sparse
        )

        assert isinstance(block, MatrixBlock)
        assert (block.row_block, block.col_block) == ("state", "state")
        assert block.name == "state_state"
        pd.testing.assert_index_equal(block.index, sites)
        pd.testing.assert_index_equal(block.columns, sites)
        np.testing.assert_array_equal(dense(block), 5.0 * np.eye(3))

    @pytest.mark.parametrize(
        "wrap", [lambda s: s, lambda s: Block(s)], ids=["series", "block"]
    )
    def test_block_from_labeled_values(self, wrap, sites):
        """A Series or a Block is the diagonal of a block, labeled by its own index."""
        series = pd.Series([1.0, 2.0, 3.0], index=sites, name="var")

        block = MatrixBlock.diagonal(wrap(series), row_block="state", col_block="state")

        pd.testing.assert_index_equal(block.index, sites)
        pd.testing.assert_index_equal(block.columns, sites)
        np.testing.assert_array_equal(block.values, np.diag([1.0, 2.0, 3.0]))
        assert block.name == "state_state"

    def test_block_needs_its_block_names(self, sites):
        """A MatrixBlock diagonal without block names fails as its constructor does."""
        with pytest.raises(ValueError, match="row_block and col_block"):
            MatrixBlock.diagonal(5.0, index=sites)

    def test_diagonal_block_assembles_into_a_matrix(self):
        """A diagonal block in a Matrix is the diagonal of that block's rows and columns."""
        flux = pd.Index(["a", "b"], name="site")
        bias = pd.Index(["x"], name="instrument")

        matrix = CovarianceMatrix(
            [
                MatrixBlock(np.full((2, 2), 0.5), "flux", "flux", index=flux),
                MatrixBlock.diagonal(
                    2.0, index=bias, row_block="bias", col_block="bias"
                ),
            ]
        )

        np.testing.assert_array_equal(
            matrix.blocks["bias", "bias"].values, 2.0 * np.eye(1)
        )
        np.testing.assert_array_equal(
            matrix.blocks["flux", "flux"].values, np.full((2, 2), 0.5)
        )
        np.testing.assert_array_equal(
            matrix["flux", "bias"].to_numpy(), np.zeros((2, 1))
        )


class TestConstructorsTake2DData:
    """A scalar or 1D input to a 2D structure raises and names `diagonal`."""

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    def test_matrix_refuses_a_scalar(self, matrix_type, scalar, index):
        """A scalar is not a matrix: the error names the class and both ways out."""
        name = matrix_type.__name__
        with pytest.raises(
            ValueError,
            match=rf"{name} takes 2D data, got a scalar.*{name}\.diagonal.*np\.full",
        ):
            matrix_type(scalar, index=index)

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize("container", [list, tuple, np.array])
    def test_matrix_refuses_1d_values(self, matrix_type, container, index):
        """A 1D array, list or tuple is not a matrix."""
        name = matrix_type.__name__
        with pytest.raises(
            ValueError, match=rf"{name} takes 2D data, got 1D data.*{name}\.diagonal"
        ):
            matrix_type(container([1.0, 2.0, 3.0]), index=index)

    @pytest.mark.parametrize("wrap", [lambda s: s, Vector], ids=["series", "vector"])
    def test_matrix_refuses_labeled_values(self, wrap, index):
        """A Series or a Vector is 1D data, with or without an index argument."""
        series = pd.Series([1.0, 2.0, 3.0], index=index)

        with pytest.raises(ValueError, match="takes 2D data, got 1D data"):
            CovarianceMatrix(wrap(series))

    @pytest.mark.parametrize(
        "data",
        [
            5.0,
            np.array(5.0),
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            pd.Series([1.0, 2.0, 3.0], index=pd.Index(["c", "a", "b"], name="site")),
        ],
        ids=["float", "0-d", "list", "array", "series"],
    )
    @pytest.mark.parametrize("with_columns", [False, True])
    def test_block_refuses_scalar_and_1d(self, data, with_columns, sites):
        """A MatrixBlock refuses them too, whether or not `columns` is given."""
        columns = pd.Index(["u", "v"], name="other") if with_columns else None

        with pytest.raises(
            ValueError, match=r"MatrixBlock takes 2D data.*MatrixBlock\.diagonal"
        ):
            MatrixBlock(data, "r", "c", index=sites, columns=columns)

    def test_dict_values_are_2d_too(self, sites):
        """The dict shorthand builds blocks, so a scalar value is refused there as well."""
        with pytest.raises(ValueError, match="MatrixBlock takes 2D data, got a scalar"):
            Matrix({("state", "state"): 5.0})

    def test_matrix_of_one_value_from_an_array(self, index):
        """The way to a matrix of one value is a 2D array."""
        matrix = Matrix(np.full((3, 3), 5.0), index=index)

        np.testing.assert_array_equal(matrix.values, np.full((3, 3), 5.0))

    def test_rectangular_block_of_one_value_from_an_array(self, sites):
        """A rectangular block of one value is a 2D array with its own columns."""
        columns = pd.Index(["u", "v"], name="other")

        block = MatrixBlock(
            np.zeros((3, 2)), "obs", "state", index=sites, columns=columns
        )

        assert block.shape == (3, 2)
        pd.testing.assert_index_equal(block.columns, columns)

    def test_one_by_one_array_is_2d(self):
        """A 1-by-1 array is 2D data, not a scalar."""
        index = pd.MultiIndex.from_product([["state"], [0]], names=["block", "i"])

        matrix = Matrix(np.array([[5.0]]), index=index)

        assert matrix.shape == (1, 1)


class TestMatrix:
    """Tests for base Matrix class."""

    def test_matrix_creation(self):
        """Test basic Matrix creation from MatrixBlock."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["a", "b"], name="idx_row"),
            columns=pd.Index(["x", "y"], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        assert matrix.shape == (2, 2)
        assert isinstance(matrix.data, pd.DataFrame)
        assert "block" in matrix.data.index.names

    def test_matrix_values_property(self):
        """Test values property returns numpy array."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="idx_row"),
            columns=pd.Index([0, 1], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        values = matrix.values
        assert isinstance(values, np.ndarray)
        assert values.shape == (2, 2)
        assert np.array_equal(values, np.array([[1, 2], [3, 4]]))

    def test_matrix_index_property(self):
        """Test index property."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["a", "b"], name="idx_row"),
            columns=pd.Index(["x", "y"], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        assert isinstance(matrix.index, pd.MultiIndex)
        assert "block" in matrix.index.names

    def test_matrix_columns_property(self):
        """Test columns property."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.Index(["x", "y"], name="idx_col"),
            index=pd.Index(["a", "b"], name="idx_row"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        assert isinstance(matrix.columns, pd.MultiIndex)
        assert "block" in matrix.columns.names

    def test_matrix_shape_property(self):
        """Test shape property."""
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=pd.Index(range(5), name="idx_row"),
            columns=pd.Index(range(3), name="idx_col"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        assert matrix.shape == (5, 3)

    def test_matrix_copies_data(self):
        """Test that Matrix creates a copy of input data."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="idx_row"),
            columns=pd.Index([0, 1], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        # Modify original
        data.iloc[0, 0] = 999

        # Matrix data should not change
        assert matrix.data.iloc[0, 0] == 1

    def test_matrix_repr(self):
        """Test string representation."""
        data = pd.DataFrame(
            np.zeros((5, 3)),
            index=pd.Index(range(5), name="idx_row"),
            columns=pd.Index(range(3), name="idx_col"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        repr_str = repr(matrix)
        assert "Matrix" in repr_str
        assert "5" in repr_str
        assert "3" in repr_str

    def test_matrix_with_multiindex(self):
        """Test Matrix with MultiIndex."""
        idx = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["x", "y"])
        data = pd.DataFrame(np.random.randn(4, 4), index=idx, columns=idx)
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        assert isinstance(matrix.index, pd.MultiIndex)
        assert "block" in matrix.index.names

    def test_matrix_blocks_accessor(self):
        """Test that matrix.blocks[row, col] wraps the frame matrix[row, col] returns."""
        rows = pd.Index(["a", "b"], name="x")
        flux = MatrixBlock(
            np.arange(6.0).reshape(2, 3),
            row_block="obs",
            col_block="flux",
            index=rows,
            columns=pd.Index([1, 2, 3], name="y"),
        )
        bias = MatrixBlock(
            np.eye(2),
            row_block="obs",
            col_block="bias",
            index=rows,
            columns=pd.Index([10, 11], name="y"),
        )
        matrix = Matrix([flux, bias])

        block = matrix.blocks["obs", "flux"]

        assert isinstance(block, MatrixBlock)
        assert (block.row_block, block.col_block) == ("obs", "flux")
        pd.testing.assert_frame_equal(block.data, matrix["obs", "flux"])
        pd.testing.assert_frame_equal(block.data, flux.data)


class TestMatrixSelection:
    """Selecting a block back out of a Matrix gives it the levels it went in with."""

    # Rows and columns that differ in their levels, so that assembling the
    # Matrix fills the missing levels with NaN.
    ROWS = {
        "o1": pd.MultiIndex.from_product([[0, 1], ["x"]], names=["t", "site"]),
        "o2": pd.Index([10, 11], name="k"),
    }
    COLUMNS = {
        "s1": pd.MultiIndex.from_product([[0.5, 1.5], [7]], names=["lat", "lon"]),
        "s2": pd.Index(["p", "q"], name="param"),
    }

    def _blocks(self):
        """The four (row, column) blocks, each filled with its own value."""
        return {
            (row, col): MatrixBlock(
                np.full((2, 2), 1.0 + 2 * i + j),
                row_block=row,
                col_block=col,
                index=self.ROWS[row],
                columns=self.COLUMNS[col],
            )
            for i, row in enumerate(self.ROWS)
            for j, col in enumerate(self.COLUMNS)
        }

    @pytest.mark.parametrize("Klass", [Matrix, ForwardOperator])
    @pytest.mark.parametrize("row", ["o1", "o2"])
    @pytest.mark.parametrize("col", ["s1", "s2"])
    def test_block_keeps_its_own_levels(self, Klass, row, col):
        """Rows and columns lose the levels that are NaN-only for the block."""
        block = self._blocks()[row, col]
        matrix = Klass(list(self._blocks().values()))
        # block, then every level of the other blocks, NaN-filled
        assert matrix.index.nlevels == matrix.columns.nlevels == 4

        for frame in (matrix[row, col], matrix.blocks[row, col].data):
            assert frame.index.names == block.index.names
            assert frame.columns.names == block.columns.names
            assert frame.index.to_list() == block.index.to_list()
            assert frame.columns.to_list() == block.columns.to_list()
            assert (frame.to_numpy() == block.to_numpy()).all()

        taken = matrix.blocks[row, col]
        assert (taken.row_block, taken.col_block) == (row, col)

    def test_levels_shared_by_every_block_are_kept(self):
        """With no NaN-only level to drop, the selection is the block as it went in."""
        rows = pd.MultiIndex.from_product([[0, 1], ["x"]], names=["t", "site"])
        columns = pd.MultiIndex.from_product([[0.5, 1.5], [7.5]], names=["lat", "lon"])
        blocks = [
            MatrixBlock(
                np.full((2, 2), float(value)),
                row_block="obs",
                col_block=col,
                index=rows,
                columns=columns,
            )
            for value, col in enumerate(["flux", "bias"], start=1)
        ]
        matrix = Matrix(blocks)

        for block in blocks:
            selected = matrix["obs", block.col_block]
            pd.testing.assert_frame_equal(selected, block.data)
            pd.testing.assert_frame_equal(
                matrix.blocks["obs", block.col_block].data, block.data
            )

    def test_covariance_matrix_block_keeps_its_own_levels(self):
        """CovarianceMatrix selects through the same code, on rows and columns."""
        flux_idx = pd.MultiIndex.from_product([[0.5, 1.5], [7]], names=["lat", "lon"])
        bias_idx = pd.Index(["a", "b"], name="site")
        cov = CovarianceMatrix(
            [
                MatrixBlock(np.eye(2), "flux", "flux", index=flux_idx),
                MatrixBlock(np.eye(2) * 2, "bias", "bias", index=bias_idx),
            ]
        )

        flux = cov["flux", "flux"]
        assert flux.index.names == ["lat", "lon"]
        assert flux.columns.names == ["lat", "lon"]
        bias = cov.blocks["bias", "bias"].data
        assert bias.index.names == ["site"]
        assert bias.columns.names == ["site"]
        assert (bias.to_numpy() == np.eye(2) * 2).all()


class TestCovarianceMatrix:
    """Tests for CovarianceMatrix class."""

    def test_covariance_matrix_creation(self):
        """Test basic CovarianceMatrix creation."""
        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[1, 0.5], [0.5, 2]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        assert cov.shape == (2, 2)
        assert isinstance(cov, Matrix)

    def test_covariance_matrix_is_symmetric(self):
        """Test symmetric covariance matrix properties."""
        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[2, 0.5], [0.5, 3]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        # Check symmetry
        assert np.allclose(cov.values, cov.values.T)


class TestCovarianceMatrixAddition:
    """Tests for CovarianceMatrix addition."""

    def test_add_two_covariance_matrices(self):
        """Test adding two CovarianceMatrices."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b"), ("b1", "c")], names=["block", "x"]
        )
        cov1_data = pd.DataFrame(np.eye(3), index=idx, columns=idx)
        cov2_data = pd.DataFrame(np.eye(3) * 2, index=idx, columns=idx)

        cov1 = CovarianceMatrix(cov1_data)
        cov2 = CovarianceMatrix(cov2_data)

        result = cov1 + cov2

        expected = np.eye(3) * 3
        assert np.allclose(result.to_numpy(), expected)
        assert isinstance(result, CovarianceMatrix)

    def test_add_nondiagonal_covariances(self):
        """Test addition preserves off-diagonal elements."""
        idx = pd.MultiIndex.from_tuples(
            [("b1", "a"), ("b1", "b")], names=["block", "x"]
        )
        cov1_data = pd.DataFrame([[1, 0.5], [0.5, 2]], index=idx, columns=idx)
        cov2_data = pd.DataFrame([[0.5, 0.2], [0.2, 0.5]], index=idx, columns=idx)

        cov1 = CovarianceMatrix(cov1_data)
        cov2 = CovarianceMatrix(cov2_data)

        result = cov1 + cov2

        expected = np.array([[1.5, 0.7], [0.7, 2.5]])
        assert np.allclose(result.to_numpy(), expected)

    def test_add_preserves_multiindex(self):
        """Test that addition preserves MultiIndex structure."""
        uidx = pd.MultiIndex.from_product(
            [["b1", "b2"], [1, 2]], names=["block", "num"]
        )
        cov1_data = pd.DataFrame(np.eye(4), index=uidx, columns=uidx)
        cov2_data = pd.DataFrame(np.eye(4) * 2, index=uidx, columns=uidx)

        cov1 = CovarianceMatrix(cov1_data)
        cov2 = CovarianceMatrix(cov2_data)

        result = cov1 + cov2

        assert isinstance(result.index, pd.MultiIndex)


class TestMatrixName:
    """The name of a 2D structure: a string or None, and what `scale` and a sum make of it."""

    @staticmethod
    def make(matrix_type, name, index, sparse=False, scale=1.0):
        """A 3 x 3 diagonal matrix of ``matrix_type``, named ``name`` (not passed when "unset")."""
        kwargs = {} if name == "unset" else {"name": name}
        return matrix_type.diagonal(scale, index=index, sparse=sparse, **kwargs)

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize("sparse", [False, True])
    def test_scale_keeps_the_name(self, matrix_type, sparse, index):
        """A scaled matrix has the name of the original, which keeps it too."""
        matrix = self.make(matrix_type, "prior_error", index, sparse, scale=2.0)

        scaled = matrix.scale(3.0)

        assert isinstance(scaled, matrix_type)
        assert scaled.name == "prior_error"
        assert matrix.name == "prior_error"
        assert scaled.is_sparse is sparse
        np.testing.assert_array_equal(dense(scaled), 6.0 * np.eye(3))

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize("sparse", [False, True])
    def test_scale_of_an_unnamed_matrix_is_unnamed(self, matrix_type, sparse, index):
        """Scaling does not give a matrix without a name the string "None"."""
        matrix = self.make(matrix_type, "unset", index, sparse)

        assert matrix.scale(3.0).name is None

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    def test_sum_with_the_same_name_keeps_it(self, matrix_type, index):
        """Two matrices with one name add up to a matrix with that name."""
        a = self.make(matrix_type, "S", index, scale=1.0)
        b = self.make(matrix_type, "S", index, scale=2.0)

        total = a + b

        assert isinstance(total, matrix_type)
        assert total.name == "S"
        np.testing.assert_array_equal(dense(total), 3.0 * np.eye(3))
        assert (a.name, b.name) == ("S", "S")

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize(
        ("left", "right"),
        [("S_a", "S_b"), ("S", "unset"), ("unset", "S"), (None, "S"), ("S", None)],
        ids=["different", "right_unnamed", "left_unnamed", "left_none", "right_none"],
    )
    def test_sum_with_different_names_is_unnamed(self, matrix_type, left, right, index):
        """Names that differ, or a name on one side only, leave the sum unnamed."""
        a = self.make(matrix_type, left, index)
        b = self.make(matrix_type, right, index)

        total = a + b

        assert total.name is None
        np.testing.assert_array_equal(dense(total), 2.0 * np.eye(3))

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    def test_sum_of_unnamed_matrices_is_unnamed(self, matrix_type, index):
        """Two unnamed matrices have an unnamed sum, not the name "None"."""
        a = self.make(matrix_type, "unset", index)

        assert (a + a).name is None

    @pytest.mark.parametrize(("sparse_a", "sparse_b"), [(True, False), (True, True)])
    def test_sum_keeps_the_name_of_sparse_matrices(self, sparse_a, sparse_b, index):
        """A sparse operand is added dense, and the name rule is the same."""
        a = self.make(CovarianceMatrix, "S", index, sparse_a)
        b = self.make(CovarianceMatrix, "S", index, sparse_b)
        c = self.make(CovarianceMatrix, "T", index, sparse_b)

        assert (a + b).name == "S"
        assert (a + c).name is None

    def test_sum_compares_names_as_strings(self, index):
        """Names are strings from construction, so 5 and "5" are one name."""
        a = self.make(Matrix, 5, index)
        b = self.make(Matrix, "5", index)

        assert (a + b).name == "5"

    @pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("x", "x"),
            (5, "5"),
            (0, "0"),
            (2020, "2020"),
            (1.5, "1.5"),
            ((1, 2), "(1, 2)"),
            (None, None),
            ("unset", None),
        ],
        ids=["str", "int", "zero", "year", "float", "tuple", "None", "unset"],
    )
    def test_name_is_a_string_or_none(self, matrix_type, name, expected, index):
        """A name that is not None becomes a string at construction, as in a Block."""
        matrix = self.make(matrix_type, name, index)

        assert matrix.name == expected
        assert type(matrix.name) is type(expected)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("x", "x"), (5, "5"), (2020, "2020"), ((1, 2), "(1, 2)"), (None, "r_c")],
        ids=["str", "int", "year", "tuple", "None"],
    )
    def test_block_name_is_a_string(self, name, expected, sites):
        """A MatrixBlock's name becomes a string too, and None is its default name."""
        block = MatrixBlock(np.eye(3), "r", "c", name=name, index=sites)

        assert block.name == expected
        assert type(block.name) is str

    def test_block_given_a_block_keeps_its_string_name(self, sites):
        """A block built from a block has the name the first block has."""
        block = MatrixBlock(np.eye(3), "r", "c", name=5, index=sites)

        assert MatrixBlock(block).name == "5"

    def test_blocks_named_alike_after_conversion_are_duplicates(self, sites):
        """Blocks named 1 and "1" have one name, so they cannot be assembled together."""
        frame = pd.DataFrame(np.eye(3), index=sites, columns=sites)
        one = MatrixBlock(frame, "a", "b", name=1)
        other = MatrixBlock(frame, "c", "d", name="1")

        with pytest.raises(ValueError, match="Duplicate block name '1'"):
            Matrix([one, other])

    def test_given_name_is_not_changed_on_the_input(self, index):
        """Converting the name does not touch the object it came from."""
        name = (1, 2)

        Matrix(np.eye(3), name=name, index=index)

        assert name == (1, 2)

    @pytest.mark.parametrize("name", ["S", 5, None])
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize(
        "rebuild",
        [
            lambda m: m.copy(),
            lambda m: m.copy(deep=False),
            lambda m: m.reindex(m.index, columns=m.columns, fill_value=0.0),
            lambda m: m.round_index(1, axis="both"),
            lambda m: m.to_dense(),
            lambda m: m.to_sparse(),
            lambda m: m.scale(2.0),
            lambda m: m + m,
            lambda m: pickle.loads(pickle.dumps(m)),
        ],
        ids=[
            "copy",
            "shallow_copy",
            "reindex",
            "round_index",
            "to_dense",
            "to_sparse",
            "scale",
            "sum_with_itself",
            "pickle",
        ],
    )
    def test_name_survives_rebuilding(self, name, sparse, rebuild, index):
        """Every way of making a matrix from a matrix keeps its string name, or its lack of one."""
        matrix = self.make(CovarianceMatrix, name, index, sparse)
        expected = None if name is None else str(name)

        result = rebuild(matrix)

        assert isinstance(result, CovarianceMatrix)
        assert result.name == expected
        assert type(result.name) is type(expected)

    def test_force_symmetry_keeps_the_name(self, index):
        """A covariance made symmetric is the same covariance, with its name."""
        cov = self.make(CovarianceMatrix, 5, index)

        assert cov.force_symmetry().name == "5"


class TestForwardOperator:
    """Tests for ForwardOperator class."""

    def test_forward_operator_creation(self):
        """Test basic ForwardOperator creation."""
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=pd.Index([f"obs_{i}" for i in range(5)], name="obs"),
            columns=pd.Index([f"state_{i}" for i in range(3)], name="state"),
        )
        block = MatrixBlock(data, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        assert forward_op.shape == (5, 3)

    def test_forward_operator_is_matrix_subclass(self):
        """Test that ForwardOperator is a Matrix subclass."""
        data = pd.DataFrame(
            np.ones((3, 3)),
            index=pd.Index(range(3), name="obs"),
            columns=pd.Index(range(3), name="state"),
        )
        block = MatrixBlock(data, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        assert isinstance(forward_op, Matrix)

    def test_forward_operator_rectangular(self):
        """Test ForwardOperator with rectangular matrix."""
        data = pd.DataFrame(
            np.random.randn(10, 5),
            index=pd.Index(range(10), name="obs"),
            columns=pd.Index(range(5), name="state"),
        )
        block = MatrixBlock(data, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        assert forward_op.shape[0] > forward_op.shape[1]


class TestMatrixBlock:
    """Tests for MatrixBlock class."""

    def test_matrix_block_creation(self):
        """Test basic MatrixBlock creation."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["a", "b"], name="idx_row"),
            columns=pd.Index(["x", "y"], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="state", col_block="obs")

        assert block.shape == (2, 2)
        assert block.row_block == "state"
        assert block.col_block == "obs"
        assert isinstance(block.data, pd.DataFrame)

    def test_matrix_block_with_name(self):
        """Test MatrixBlock with custom name."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="idx_row"),
            columns=pd.Index([0, 1], name="idx_col"),
        )
        block = MatrixBlock(
            data, row_block="state", col_block="obs", name="custom_name"
        )

        assert block.name == "custom_name"

    def test_matrix_block_default_name(self):
        """Test MatrixBlock generates default name."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="idx_row"),
            columns=pd.Index([0, 1], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="state", col_block="obs")

        assert block.name == "state_obs"

    def test_matrix_block_repr(self):
        """Test MatrixBlock string representation."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="idx_row"),
            columns=pd.Index([0, 1], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="state", col_block="obs")

        repr_str = repr(block)
        assert "MatrixBlock" in repr_str
        assert "state" in repr_str
        assert "obs" in repr_str

    def test_matrix_block_to_frame(self):
        """Test converting MatrixBlock to frame without block level."""
        data = pd.DataFrame(
            [[1, 2]],
            index=pd.Index(["a"], name="idx_row"),
            columns=pd.Index(["x", "y"], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="state", col_block="obs")

        df = block.to_frame(add_block_level=False)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 2)

    def test_matrix_block_to_frame_with_blocks(self):
        """Test converting MatrixBlock to frame with block level."""
        data = pd.DataFrame(
            [[1, 2]],
            index=pd.Index(["a"], name="idx_row"),
            columns=pd.Index(["x", "y"], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="state", col_block="obs")

        df = block.to_frame(add_block_level=True)
        assert isinstance(df.index, pd.MultiIndex)
        assert isinstance(df.columns, pd.MultiIndex)
        assert "state" in df.index.get_level_values(0)
        assert "obs" in df.columns.get_level_values(0)

    @pytest.fixture(params=[False, True], ids=["dense", "sparse"])
    def block(self, request):
        """A rectangular block with a name of its own, dense and sparse."""
        data = pd.DataFrame(
            [[1.0, 2.0, 0.0], [0.0, 4.0, 5.0]],
            index=pd.Index([0.11, 0.26], name="obs_id"),
            columns=pd.Index(["s0", "s1", "s2"], name="state_id"),
        )
        return MatrixBlock(
            data, row_block="obs", col_block="state", name="H", sparse=request.param
        )

    @pytest.mark.parametrize(
        "rebuild",
        [
            lambda block: block.reindex(block.index, fill_value=0.0),
            lambda block: block.round_index(1),
            lambda block: block.copy(),
            lambda block: block.copy(deep=False),
            lambda block: block.to_dense(),
            lambda block: block.to_sparse(),
        ],
        ids=["reindex", "round_index", "copy", "shallow_copy", "to_dense", "to_sparse"],
    )
    def test_rebuilt_block_keeps_its_names(self, block, rebuild):
        """A new block made from this one has its block names and its name."""
        result = rebuild(block)
        assert isinstance(result, MatrixBlock)
        assert (result.row_block, result.col_block, result.name) == (
            "obs",
            "state",
            "H",
        )

    def test_reindex_block(self, block):
        """New rows are filled, and storage stays as it was."""
        new_rows = pd.Index([0.11, 0.26, 0.5], name="obs_id")
        result = block.reindex(new_rows, fill_value=0.0)
        assert result.is_sparse == block.is_sparse
        pd.testing.assert_index_equal(result.index, new_rows)
        np.testing.assert_array_equal(
            dense(result), [[1.0, 2.0, 0.0], [0.0, 4.0, 5.0], [0.0, 0.0, 0.0]]
        )
        np.testing.assert_array_equal(dense(block), [[1.0, 2.0, 0.0], [0.0, 4.0, 5.0]])

    def test_reindex_block_columns(self, block):
        """Rows and columns are both reindexed."""
        new_cols = pd.Index(["s2", "s0", "s3"], name="state_id")
        result = block.reindex(block.index, columns=new_cols, fill_value=0.0)
        assert (result.row_block, result.col_block) == ("obs", "state")
        np.testing.assert_array_equal(dense(result), [[0.0, 1.0, 0.0], [5.0, 0.0, 0.0]])

    def test_round_index_block(self, block):
        """The rounded labels are on the new block only."""
        result = block.round_index(1)
        pd.testing.assert_index_equal(result.index, pd.Index([0.1, 0.3], name="obs_id"))
        pd.testing.assert_index_equal(
            block.index, pd.Index([0.11, 0.26], name="obs_id")
        )

    def test_matrix_block_pickle_support(self):
        """Test MatrixBlock pickle serialization."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["a", "b"], name="idx_row"),
            columns=pd.Index(["x", "y"], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="state", col_block="obs", name="test_block")

        # Get state for pickling
        state = block.__getstate__()
        assert "data" in state
        assert "name" in state
        assert "row_block" in state
        assert "col_block" in state

        # Create new block and restore state
        new_block = MatrixBlock(
            pd.DataFrame(
                [[0]],
                index=pd.Index([0], name="temp"),
                columns=pd.Index([0], name="temp"),
            ),
            "temp",
            "temp",
        )
        new_block.__setstate__(state)

        assert new_block.row_block == "state"
        assert new_block.col_block == "obs"
        assert new_block.name == "test_block"


class TestMatrixOperations:
    """Tests for Matrix utility operations."""

    def test_matrix_reindex(self):
        """Test reindexing a matrix with blocks."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["a", "b"], name="idx_row"),
            columns=pd.Index(["x", "y"], name="idx_col"),
        )
        block = MatrixBlock(data, row_block="rows", col_block="cols")
        matrix = Matrix([block])

        reindexed = matrix.data.reindex(
            index=pd.MultiIndex.from_tuples(
                [("rows", "a"), ("rows", "b"), ("rows", "c")],
                names=["block", "idx_row"],
            ),
            fill_value=0,
        )

        assert reindexed.shape == (3, 2)

    def test_large_covariance_matrix(self):
        """Test creating a large covariance matrix."""
        size = 100
        block_idx = pd.MultiIndex.from_product(
            [["b1"], range(size)], names=["block", "idx"]
        )
        data = pd.DataFrame(np.eye(size), index=block_idx, columns=block_idx)
        cov = CovarianceMatrix(data)

        assert cov.shape == (size, size)
        assert np.allclose(cov.values, np.eye(size))

    def test_forward_operator_multiplication_shape(self):
        """Test that forward operator has correct multiplication properties."""
        H = pd.DataFrame(
            np.random.randn(5, 3),
            index=pd.Index(range(5), name="obs"),
            columns=pd.Index(range(3), name="state"),
        )
        block = MatrixBlock(H, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        x = np.array([1, 2, 3])
        result = forward_op.values @ x

        assert result.shape == (5,)


class TestConvolve:
    """Tests for convolve function and ForwardOperator.convolve method."""

    def test_convolve_basic(self):
        """Test basic convolution with vector and forward operator."""
        from fips.vector import Block, Vector

        idx = pd.Index([0, 1, 2], name="idx")
        state_series = pd.Series([1.0, 2.0, 3.0], index=idx, name="state")
        state = Vector(data=[Block(state_series)], name="state")
        H_idx = pd.Index([0, 1, 2], name="idx")
        H = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=H_idx,
            columns=H_idx,
        )
        block = MatrixBlock(H, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        result = forward_op.convolve(state)

        assert isinstance(result, pd.Series)
        assert result.shape == (3,)

    def test_convolve_with_dataframe_operator(self):
        """Test convolution with DataFrame forward operator."""
        from fips.vector import Block, Vector

        idx = pd.Index([0, 1], name="idx")
        state_series = pd.Series([1.0, 2.0], index=idx, name="state")
        state = Vector(data=[Block(state_series)], name="state")
        H_idx = pd.Index([0, 1], name="idx")
        H = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=H_idx, columns=H_idx)
        block = MatrixBlock(H, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        result = forward_op.convolve(state)

        assert isinstance(result, pd.Series)
        assert result.shape == (2,)

    def test_convolve_with_forward_operator_object(self):
        """Test convolution with ForwardOperator object."""
        from fips.vector import Block, Vector

        idx = pd.Index([0, 1], name="idx")
        state_series = pd.Series([1.0, 2.0], index=idx, name="state")
        state = Vector(data=[Block(state_series)], name="state")
        H_idx = pd.Index([0, 1], name="idx")
        H_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=H_idx, columns=H_idx)
        block = MatrixBlock(H_df, row_block="obs", col_block="state")
        H = ForwardOperator([block])

        result = H.convolve(state)

        assert isinstance(result, pd.Series)

    def test_convolve_result_values(self):
        """Test that convolution produces correct values."""
        from fips.vector import Block, Vector

        state_values = np.array([1.0, 2.0])
        idx = pd.Index([0, 1], name="idx")
        state_series = pd.Series(state_values, index=idx, name="state")
        state = Vector(data=[Block(state_series)], name="state")
        H_idx = pd.Index([0, 1], name="idx")
        H = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=H_idx, columns=H_idx)
        block = MatrixBlock(H, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        result = forward_op.convolve(state)

        # For identity matrix, result should match input
        assert np.allclose(result.to_numpy(), state_values)

    def test_forward_operator_convolve_method(self):
        """Test ForwardOperator.convolve method directly."""
        from fips.vector import Block, Vector

        state_values = np.array([1.0, 2.0, 3.0])
        idx = pd.Index([0, 1, 2], name="idx")
        state_series = pd.Series(state_values, index=idx, name="state")
        state = Vector(data=[Block(state_series)], name="state")
        H_idx = pd.Index([0, 1, 2], name="idx")
        H_df = pd.DataFrame(np.eye(3), index=H_idx, columns=H_idx)
        block = MatrixBlock(H_df, row_block="obs", col_block="state")
        H = ForwardOperator([block])

        result = H.convolve(state)

        assert isinstance(result, pd.Series)
        assert np.allclose(result.to_numpy(), state_values)

    def test_convolve_with_series(self):
        """Test convolution with pandas Series state."""
        from fips.vector import Block, Vector

        idx = pd.Index([0, 1, 2], name="idx")
        state_series = pd.Series([1.0, 2.0, 3.0], index=idx, name="state")
        state = Vector(data=[Block(state_series)], name="state")
        H_idx = pd.Index([0, 1, 2], name="idx")
        H = pd.DataFrame(np.eye(3), index=H_idx, columns=H_idx)
        block = MatrixBlock(H, row_block="obs", col_block="state")
        forward_op = ForwardOperator([block])

        result = forward_op.convolve(state)

        assert isinstance(result, pd.Series)
