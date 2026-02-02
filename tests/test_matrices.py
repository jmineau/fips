"""Test suite for fips matrix types (structures, covariance, operators)."""

import numpy as np
import pandas as pd

from fips.covariance import CovarianceMatrix
from fips.operators import ForwardOperator, convolve
from fips.structures import Matrix, prepare_matrix


class TestMatrix:
    """Tests for base Matrix class."""

    def test_matrix_creation(self):
        """Test basic Matrix creation."""
        data = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        matrix = Matrix(data)

        assert matrix.shape == (2, 2)
        assert isinstance(matrix.data, pd.DataFrame)

    def test_matrix_values_property(self):
        """Test values property returns numpy array."""
        data = pd.DataFrame([[1, 2], [3, 4]])
        matrix = Matrix(data)

        values = matrix.values
        assert isinstance(values, np.ndarray)
        assert values.shape == (2, 2)
        assert np.array_equal(values, np.array([[1, 2], [3, 4]]))

    def test_matrix_index_property(self):
        """Test index property."""
        data = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"])
        matrix = Matrix(data)

        assert isinstance(matrix.index, pd.Index)
        assert matrix.index.tolist() == ["a", "b"]

    def test_matrix_columns_property(self):
        """Test columns property."""
        data = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "y"])
        matrix = Matrix(data)

        assert isinstance(matrix.columns, pd.Index)
        assert matrix.columns.tolist() == ["x", "y"]

    def test_matrix_shape_property(self):
        """Test shape property."""
        data = pd.DataFrame(np.random.randn(5, 3))
        matrix = Matrix(data)

        assert matrix.shape == (5, 3)

    def test_matrix_copies_data(self):
        """Test that Matrix creates a copy of input data."""
        data = pd.DataFrame([[1, 2], [3, 4]])
        matrix = Matrix(data)

        # Modify original
        data.iloc[0, 0] = 999

        # Matrix data should not change
        assert matrix.data.iloc[0, 0] == 1

    def test_matrix_repr(self):
        """Test string representation."""
        data = pd.DataFrame(np.zeros((5, 3)))
        matrix = Matrix(data)

        repr_str = repr(matrix)
        assert "Matrix" in repr_str
        assert "5" in repr_str
        assert "3" in repr_str

    def test_matrix_with_multiindex(self):
        """Test Matrix with MultiIndex."""
        idx = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["x", "y"])
        data = pd.DataFrame(np.random.randn(4, 4), index=idx, columns=idx)
        matrix = Matrix(data)

        assert isinstance(matrix.index, pd.MultiIndex)
        assert matrix.index.nlevels == 2


class TestCovarianceMatrix:
    """Tests for CovarianceMatrix class."""

    def test_covariance_matrix_creation(self):
        """Test basic CovarianceMatrix creation."""
        data = pd.DataFrame([[1, 0.5], [0.5, 2]], index=["a", "b"], columns=["a", "b"])
        cov = CovarianceMatrix(data)

        assert cov.shape == (2, 2)
        assert isinstance(cov, Matrix)

    def test_covariance_matrix_is_symmetric(self):
        """Test symmetric covariance matrix properties."""
        data = pd.DataFrame([[2, 0.5], [0.5, 3]], index=["a", "b"], columns=["a", "b"])
        cov = CovarianceMatrix(data)

        # Check symmetry
        assert np.allclose(cov.values, cov.values.T)

    def test_covariance_matrix_from_variances(self):
        """Test creating CovarianceMatrix from variances."""
        variances = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"], name="variance")

        cov = CovarianceMatrix.from_variances(variances)

        assert cov.shape == (3, 3)
        # Should be diagonal with variances on diagonal
        assert np.allclose(np.diag(cov.values), [1.0, 2.0, 3.0])
        assert np.allclose(
            cov.values - np.diag(np.diag(cov.values)), 0
        )  # Off-diagonal is zero

    def test_covariance_matrix_from_variances_multiindex(self):
        """Test CovarianceMatrix.from_variances with MultiIndex."""
        idx = pd.MultiIndex.from_product([["b1", "b2"], [0, 1]], names=["block", "idx"])
        variances = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="variance")

        cov = CovarianceMatrix.from_variances(variances)

        assert cov.shape == (4, 4)
        assert "block" in cov.index.names
        assert np.allclose(np.diag(cov.values), [1.0, 2.0, 3.0, 4.0])


class TestCovarianceMatrixAddition:
    """Tests for CovarianceMatrix addition."""

    def test_add_two_covariance_matrices(self):
        """Test adding two CovarianceMatrices."""
        idx = ["a", "b", "c"]
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
        cov1_data = pd.DataFrame([[1, 0.5], [0.5, 2]])
        cov2_data = pd.DataFrame([[0.5, 0.2], [0.2, 0.5]])

        cov1 = CovarianceMatrix(cov1_data)
        cov2 = CovarianceMatrix(cov2_data)

        result = cov1 + cov2

        expected = np.array([[1.5, 0.7], [0.7, 2.5]])
        assert np.allclose(result.values, expected)

    def test_add_preserves_multiindex(self):
        """Test that addition preserves MultiIndex structure."""
        idx = pd.MultiIndex.from_product([["b1", "b2"], [1, 2]])
        cov1_data = pd.DataFrame(np.eye(4), index=idx, columns=idx)
        cov2_data = pd.DataFrame(np.eye(4) * 2, index=idx, columns=idx)

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
            index=[f"obs_{i}" for i in range(5)],
            columns=[f"state_{i}" for i in range(3)],
        )
        forward_op = ForwardOperator(data)

        assert forward_op.shape == (5, 3)

    def test_forward_operator_is_matrix_subclass(self):
        """Test that ForwardOperator is a Matrix subclass."""
        data = pd.DataFrame(np.ones((3, 3)))
        forward_op = ForwardOperator(data)

        assert isinstance(forward_op, Matrix)

    def test_forward_operator_rectangular(self):
        """Test ForwardOperator with rectangular matrix."""
        data = pd.DataFrame(np.random.randn(10, 5), index=range(10), columns=range(5))
        forward_op = ForwardOperator(data)

        assert forward_op.shape[0] > forward_op.shape[1]


class TestPrepareMatrix:
    """Tests for prepare_matrix function."""

    def test_prepare_matrix_basic(self):
        """Test basic prepare_matrix functionality."""
        data = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        row_index = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "state"]
        )
        col_index = pd.MultiIndex.from_tuples(
            [("blockA", "x"), ("blockA", "y")], names=["block", "obs"]
        )
        result = prepare_matrix(
            data,
            matrix_class=Matrix,
            row_index=row_index,
            col_index=col_index,
            float_precision=None,
        )

        assert isinstance(result, Matrix)
        assert result.shape == (2, 2)

    def test_prepare_matrix_sanitizes_indices(self):
        """Test that prepare_matrix sanitizes indices."""
        data = pd.DataFrame(
            [[1, 2], [3, 4]], index=["1.0", "2.0"], columns=["3.0", "4.0"]
        )
        row_index = pd.MultiIndex.from_tuples(
            [("block1", "1.0"), ("block1", "2.0")], names=["block", "state"]
        )
        col_index = pd.MultiIndex.from_tuples(
            [("blockA", "3.0"), ("blockA", "4.0")], names=["block", "obs"]
        )
        result = prepare_matrix(
            data,
            matrix_class=Matrix,
            row_index=row_index,
            col_index=col_index,
            float_precision=None,
        )

        assert pd.api.types.is_float_dtype(result.data.index.get_level_values("state"))
        assert pd.api.types.is_float_dtype(result.data.columns.get_level_values("obs"))

    def test_prepare_matrix_promotion(self):
        """Test prepare_matrix with row promotion."""
        data = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        row_index = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "state"]
        )
        col_index = pd.MultiIndex.from_tuples(
            [("blockA", "x"), ("blockA", "y")], names=["block", "obs"]
        )
        result = prepare_matrix(
            data,
            matrix_class=Matrix,
            row_index=row_index,
            col_index=col_index,
            float_precision=None,
        )

        # Should have block level in index
        assert isinstance(result.index, pd.MultiIndex)
        assert "block" in result.index.names
        assert "block" in result.columns.names

    def test_prepare_matrix_copies_data(self):
        """Test that prepare_matrix creates a copy."""
        data = pd.DataFrame([[1, 2], [3, 4]])
        row_index = pd.MultiIndex.from_tuples(
            [("block1", 0), ("block1", 1)], names=["block", "state"]
        )
        col_index = pd.MultiIndex.from_tuples(
            [("blockA", 0), ("blockA", 1)], names=["block", "obs"]
        )
        result = prepare_matrix(
            data,
            matrix_class=Matrix,
            row_index=row_index,
            col_index=col_index,
            float_precision=None,
        )

        # Modify original
        data.iloc[0, 0] = 999

        # Result should not change
        assert result.data.iloc[0, 0] == 1


class TestMatrixOperations:
    """Tests for Matrix utility operations."""

    def test_matrix_reindex(self):
        """Test reindexing a matrix."""
        data = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["x", "y"])
        matrix = Matrix(data)

        reindexed = matrix.data.reindex(index=["a", "b", "c"], fill_value=0)

        assert reindexed.shape == (3, 2)
        assert reindexed.loc["c"].tolist() == [0, 0]

    def test_large_covariance_matrix(self):
        """Test creating a large covariance matrix."""
        size = 100
        data = pd.DataFrame(np.eye(size))
        cov = CovarianceMatrix(data)

        assert cov.shape == (size, size)
        assert np.allclose(cov.values, np.eye(size))

    def test_forward_operator_multiplication_shape(self):
        """Test that forward operator has correct multiplication properties."""
        H = pd.DataFrame(np.random.randn(5, 3))
        x = np.array([1, 2, 3])

        forward_op = ForwardOperator(H)

        result = forward_op.values @ x

        assert result.shape == (5,)


class TestConvolve:
    """Tests for convolve function and ForwardOperator.convolve method."""

    def test_convolve_basic(self):
        """Test basic convolution with vector and forward operator."""

        state = np.array([1.0, 2.0, 3.0])
        H = pd.DataFrame([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        result = convolve(state, H)

        assert isinstance(result, pd.Series)
        assert result.shape == (3,)

    def test_convolve_with_dataframe_operator(self):
        """Test convolution with DataFrame forward operator."""

        state = np.array([1.0, 2.0])
        H = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])

        result = convolve(state, H)

        assert isinstance(result, pd.Series)
        assert result.shape == (2,)

    def test_convolve_with_forward_operator_object(self):
        """Test convolution with ForwardOperator object."""

        state = np.array([1.0, 2.0])
        H_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        H = ForwardOperator(H_df)

        result = convolve(state, H)

        assert isinstance(result, pd.Series)

    def test_convolve_result_values(self):
        """Test that convolution produces correct values."""

        state = np.array([1.0, 2.0])
        H = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]])

        result = convolve(state, H)

        # For identity matrix, result should match input
        assert np.allclose(result.values, state)

    def test_forward_operator_convolve_method(self):
        """Test ForwardOperator.convolve method directly."""
        state = np.array([1.0, 2.0, 3.0])
        H_df = pd.DataFrame(np.eye(3))
        H = ForwardOperator(H_df)

        result = H.convolve(state)

        assert isinstance(result, pd.Series)
        assert np.allclose(result.values, state)

    def test_convolve_with_series(self):
        """Test convolution with pandas Series state."""

        idx = pd.Index([0, 1, 2], name="x")
        state = pd.Series([1.0, 2.0, 3.0], index=idx)
        H = pd.DataFrame(np.eye(3))

        result = convolve(state, H)

        assert isinstance(result, pd.Series)

    def test_convolve_float_precision(self):
        """Test convolution with float precision parameter."""

        state = np.array([1.123456, 2.654321])
        H = pd.DataFrame(np.eye(2))

        result = convolve(state, H, float_precision=2)

        assert isinstance(result, pd.Series)
