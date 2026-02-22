"""Test suite for fips matrix types (structures, covariance, operators)."""

import numpy as np
import pandas as pd

from fips.covariance import CovarianceMatrix
from fips.matrix import Matrix, MatrixBlock
from fips.operators import ForwardOperator


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
        assert np.allclose(result.values, expected)
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
        assert np.allclose(result.values, expected)

    def test_add_preserves_multiindex(self):
        """Test that addition preserves MultiIndex structure."""
        idx = pd.MultiIndex.from_product(
            [["b1", "b2"], [1, 2], ["block"]], names=["block", "num", "extra"]
        )
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
        assert np.allclose(result.values, state_values)

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
        assert np.allclose(result.values, state_values)

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
