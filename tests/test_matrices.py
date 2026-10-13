"""Test suite for fips matrix types (structures, covariance, operators)."""

import numpy as np
import pandas as pd
import pytest

from fips.covariance import CovarianceMatrix
from fips.matrix import Matrix, MatrixBlock
from fips.operators import ForwardOperator

# A scalar in each form a caller might pass: Python float and int, a NumPy scalar, a 0-d array
SCALARS = [5.0, 5, np.float64(5.0), np.array(5.0)]


class TestMatrix:
    """Tests for base Matrix class."""

    @pytest.mark.parametrize("matrix_type", [Matrix, CovarianceMatrix])
    @pytest.mark.parametrize("container", [list, np.array])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_diagonal_values_use_supplied_index(self, matrix_type, container, sparse):
        """Diagonal inputs preserve their labels in dense and sparse matrices."""
        values = container([1.0, 2.0, 3.0])
        index = pd.MultiIndex.from_tuples(
            [("state", "c"), ("state", "a"), ("state", "b")],
            names=["block", "site"],
        )

        matrix = matrix_type(values, index=index, sparse=sparse)

        pd.testing.assert_index_equal(matrix.index, index)
        pd.testing.assert_index_equal(matrix.columns, index)
        actual = matrix.values.toarray() if sparse else matrix.values
        np.testing.assert_array_equal(actual, np.diag(values))
        assert isinstance(matrix, matrix_type)

    @pytest.mark.parametrize("matrix_type", [Matrix, CovarianceMatrix])
    @pytest.mark.parametrize("container", [list, np.array])
    def test_diagonal_values_require_an_index(self, matrix_type, container):
        """Unlabeled diagonal values explain the required row index."""
        with pytest.raises(ValueError, match="1D data requires an index"):
            matrix_type(container([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize("matrix_type", [Matrix, CovarianceMatrix])
    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    @pytest.mark.parametrize("sparse", [False, True])
    def test_scalar_builds_a_diagonal(self, matrix_type, scalar, sparse):
        """A scalar or 0-d array puts its value on the diagonal and zeros elsewhere."""
        index = pd.MultiIndex.from_tuples(
            [("state", "c"), ("state", "a"), ("state", "b")],
            names=["block", "site"],
        )

        matrix = matrix_type(scalar, index=index, sparse=sparse)

        pd.testing.assert_index_equal(matrix.index, index)
        pd.testing.assert_index_equal(matrix.columns, index)
        actual = matrix.values.toarray() if sparse else matrix.values
        np.testing.assert_array_equal(actual, 5.0 * np.eye(3))
        assert isinstance(matrix, matrix_type)

    @pytest.mark.parametrize("matrix_type", [Matrix, CovarianceMatrix])
    @pytest.mark.parametrize("size", [1, 3])
    def test_scalar_matches_1d_input_of_the_same_value(self, matrix_type, size):
        """A scalar builds what a 1D input of that value on every entry builds."""
        index = pd.MultiIndex.from_product(
            [["state"], range(size)], names=["block", "i"]
        )

        from_scalar = matrix_type(5.0, index=index)
        from_list = matrix_type([5.0] * size, index=index)

        pd.testing.assert_frame_equal(from_scalar.data, from_list.data)

    def test_scalar_labels_both_axes_with_index(self):
        """Columns are not used for a scalar, as for a 1D input: the matrix is square."""
        index = pd.MultiIndex.from_product([["state"], [0, 1, 2]], names=["block", "i"])
        columns = pd.MultiIndex.from_product([["state"], [0, 1]], names=["block", "i"])

        matrix = Matrix(5.0, index=index, columns=columns)

        assert matrix.shape == (3, 3)
        pd.testing.assert_index_equal(matrix.columns, index)

    @pytest.mark.parametrize("matrix_type", [Matrix, CovarianceMatrix])
    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    def test_scalar_requires_an_index(self, matrix_type, scalar):
        """An unlabeled scalar explains the required row index, as a 1D input does."""
        with pytest.raises(ValueError, match="Scalar data requires an index"):
            matrix_type(scalar)

    def test_nan_scalar_is_rejected(self):
        """A NaN scalar raises the NaN error rather than building a matrix."""
        index = pd.MultiIndex.from_product([["state"], [0, 1]], names=["block", "i"])
        with pytest.raises(ValueError, match="NaN"):
            Matrix(np.nan, index=index)

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

    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    @pytest.mark.parametrize("sparse", [False, True])
    def test_scalar_builds_a_diagonal(self, scalar, sparse):
        """A scalar or 0-d array puts its value on the diagonal and zeros elsewhere."""
        index = pd.Index(["c", "a", "b"], name="site")

        block = MatrixBlock(
            scalar, row_block="state", col_block="state", index=index, sparse=sparse
        )

        pd.testing.assert_index_equal(block.index, index)
        pd.testing.assert_index_equal(block.columns, index)
        actual = block.values.toarray() if sparse else block.values
        np.testing.assert_array_equal(actual, 5.0 * np.eye(3))
        assert (block.row_block, block.col_block) == ("state", "state")
        assert block.name == "state_state"

    @pytest.mark.parametrize("container", [list, np.array])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_diagonal_values_use_supplied_index(self, container, sparse):
        """A 1D array or list is the diagonal and keeps its labels on both axes."""
        values = container([1.0, 2.0, 3.0])
        index = pd.Index(["c", "a", "b"], name="site")

        block = MatrixBlock(
            values, row_block="state", col_block="state", index=index, sparse=sparse
        )

        pd.testing.assert_index_equal(block.index, index)
        pd.testing.assert_index_equal(block.columns, index)
        actual = block.values.toarray() if sparse else block.values
        np.testing.assert_array_equal(actual, np.diag([1.0, 2.0, 3.0]))

    def test_series_is_the_diagonal_with_its_own_index(self):
        """A Series supplies the diagonal values and its index labels both axes."""
        series = pd.Series(
            [1.0, 2.0, 3.0], index=pd.Index(["c", "a", "b"], name="site"), name="var"
        )

        block = MatrixBlock(series, row_block="state", col_block="state")

        pd.testing.assert_index_equal(block.index, series.index)
        pd.testing.assert_index_equal(block.columns, series.index)
        np.testing.assert_array_equal(block.values, np.diag([1.0, 2.0, 3.0]))
        assert block.name == "state_state"

    @pytest.mark.parametrize("size", [1, 3])
    def test_scalar_matches_1d_input_of_the_same_value(self, size):
        """A scalar builds what a 1D input of that value on every entry builds."""
        index = pd.RangeIndex(size, name="i")

        from_scalar = MatrixBlock(5.0, row_block="r", col_block="c", index=index)
        from_list = MatrixBlock([5.0] * size, row_block="r", col_block="c", index=index)

        pd.testing.assert_frame_equal(from_scalar.data, from_list.data)

    def test_diagonal_block_assembles_into_a_matrix(self):
        """A diagonal block in a Matrix is the diagonal of that block's rows and columns."""
        index = pd.Index(["a", "b"], name="site")

        block = MatrixBlock(2.0, row_block="state", col_block="state", index=index)
        matrix = Matrix([block])

        np.testing.assert_array_equal(matrix.values, 2.0 * np.eye(2))
        np.testing.assert_array_equal(
            matrix.blocks["state", "state"].values, 2.0 * np.eye(2)
        )

    @pytest.mark.parametrize("scalar", SCALARS, ids=lambda s: type(s).__name__)
    def test_scalar_requires_an_index(self, scalar):
        """An unlabeled scalar explains the required row index, as a Matrix does."""
        with pytest.raises(ValueError, match="Scalar data requires an index"):
            MatrixBlock(scalar, row_block="r", col_block="c")

    @pytest.mark.parametrize("container", [list, np.array])
    def test_diagonal_values_require_an_index(self, container):
        """Unlabeled diagonal values explain the required row index, as a Matrix does."""
        with pytest.raises(ValueError, match="1D data requires an index"):
            MatrixBlock(container([1.0, 2.0, 3.0]), row_block="r", col_block="c")

    @pytest.mark.parametrize(
        "data",
        [
            5.0,
            np.array(5.0),
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            pd.Series([1.0, 2.0, 3.0], index=pd.Index(["a", "b", "c"], name="site")),
        ],
        ids=["float", "0-d", "list", "array", "series"],
    )
    def test_columns_cannot_be_given_with_a_diagonal(self, data):
        """A diagonal is square: a scalar or 1D input with columns raises, naming the fix."""
        index = pd.Index(["a", "b", "c"], name="site")
        columns = pd.Index(["u", "v"], name="other")

        with pytest.raises(ValueError, match=r"`columns` cannot be given.*np\.full"):
            MatrixBlock(
                data, row_block="r", col_block="c", index=index, columns=columns
            )

    def test_rectangular_block_of_one_value_from_an_array(self):
        """The way to a rectangular block of one value is a 2D array."""
        index = pd.Index(["a", "b", "c"], name="site")
        columns = pd.Index(["u", "v"], name="other")

        block = MatrixBlock(
            np.full((3, 2), 5.0),
            row_block="r",
            col_block="c",
            index=index,
            columns=columns,
        )

        assert block.shape == (3, 2)
        pd.testing.assert_index_equal(block.columns, columns)
        np.testing.assert_array_equal(block.values, np.full((3, 2), 5.0))

    @pytest.mark.parametrize("container", [list, np.array])
    def test_matrix_does_not_use_columns_for_diagonal_values(self, container):
        """A Matrix ignores `columns` for a 1D input, as for a scalar."""
        index = pd.MultiIndex.from_product([["state"], [0, 1, 2]], names=["block", "i"])
        columns = pd.MultiIndex.from_product([["state"], [0, 1]], names=["block", "i"])

        matrix = Matrix(container([1.0, 2.0, 3.0]), index=index, columns=columns)

        assert matrix.shape == (3, 3)
        pd.testing.assert_index_equal(matrix.columns, index)


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
