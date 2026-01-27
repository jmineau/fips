"""Test suite for fips.matrices module."""

import pytest
import pandas as pd
import numpy as np

from fips.matrices import Matrix, CovarianceMatrix, ForwardOperator, prepare_matrix
from fips.vectors import Vector, Block


class TestMatrix:
    """Tests for base Matrix class."""

    def test_matrix_creation(self):
        """Test basic Matrix creation."""
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           index=['a', 'b'], 
                           columns=['x', 'y'])
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
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           index=['a', 'b'])
        matrix = Matrix(data)
        
        assert isinstance(matrix.index, pd.Index)
        assert matrix.index.tolist() == ['a', 'b']

    def test_matrix_columns_property(self):
        """Test columns property."""
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           columns=['x', 'y'])
        matrix = Matrix(data)
        
        assert isinstance(matrix.columns, pd.Index)
        assert matrix.columns.tolist() == ['x', 'y']

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
        assert 'Matrix' in repr_str
        assert '5' in repr_str
        assert '3' in repr_str

    def test_matrix_with_multiindex(self):
        """Test Matrix with MultiIndex."""
        idx = pd.MultiIndex.from_product([['a', 'b'], [1, 2]], names=['x', 'y'])
        data = pd.DataFrame(np.random.randn(4, 4), index=idx, columns=idx)
        matrix = Matrix(data)
        
        assert isinstance(matrix.index, pd.MultiIndex)
        assert matrix.index.nlevels == 2


class TestCovarianceMatrix:
    """Tests for CovarianceMatrix class."""

    def test_covariance_matrix_creation(self):
        """Test basic CovarianceMatrix creation."""
        data = pd.DataFrame([[1, 0.5], [0.5, 2]], 
                           index=['a', 'b'], 
                           columns=['a', 'b'])
        cov = CovarianceMatrix(data)
        
        assert cov.shape == (2, 2)
        assert isinstance(cov, Matrix)

    def test_covariance_matrix_is_symmetric(self):
        """Test symmetric covariance matrix properties."""
        data = pd.DataFrame([[2, 0.5], [0.5, 3]], 
                           index=['a', 'b'], 
                           columns=['a', 'b'])
        cov = CovarianceMatrix(data)
        
        # Check symmetry
        assert np.allclose(cov.values, cov.values.T)

    def test_covariance_matrix_from_vector(self):
        """Test creating CovarianceMatrix from Vector."""
        block = Block('b1', pd.Series([1, 2, 3], index=['a', 'b', 'c']))
        vector = Vector([block])
        
        cov = CovarianceMatrix.from_vector(vector)
        
        assert cov.shape == (3, 3)
        # Should be zero-initialized
        assert np.allclose(cov.values, 0)

    def test_covariance_matrix_from_vector_multiple_blocks(self):
        """Test CovarianceMatrix.from_vector with multiple blocks."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4, 5], index=['a', 'b', 'c']))
        vector = Vector([block1, block2])
        
        cov = CovarianceMatrix.from_vector(vector)
        
        assert cov.shape == (5, 5)
        assert 'block' in cov.index.names


class TestCovarianceMatrixSetBlock:
    """Tests for CovarianceMatrix.set_block() method."""

    def test_set_block_scalar_sigma(self):
        """Test set_block with scalar sigma (identity-like covariance)."""
        block = Block('state', pd.Series([1, 2, 3], index=['a', 'b', 'c']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        cov.set_block('state', sigma=2.0)
        
        # Should be diagonal with 2.0^2 = 4.0 on diagonal
        expected = np.eye(3) * 4.0
        assert np.allclose(cov.values, expected)

    def test_set_block_array_sigma(self):
        """Test set_block with array sigma (heterogeneous variances)."""
        block = Block('state', pd.Series([1, 2, 3], index=['a', 'b', 'c']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        sigmas = np.array([1.0, 2.0, 3.0])
        cov.set_block('state', sigma=sigmas)
        
        # Diagonal should be sigmas^2
        expected = np.diag([1.0, 4.0, 9.0])
        assert np.allclose(cov.values, expected)

    def test_set_block_with_kernel(self):
        """Test set_block with correlation kernel."""
        block = Block('state', pd.Series([1, 2, 3], index=[0.0, 1.0, 2.0]))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        # Exponential decay kernel
        def kernel(idx1, idx2):
            x1 = idx1.values[:, None]
            x2 = idx2.values[None, :]
            dists = np.abs(x1 - x2)
            return np.exp(-dists / 1.0)  # correlation length = 1.0
        
        cov.set_block('state', sigma=1.0, kernel=kernel)
        
        # Should be symmetric and have 1 on diagonal
        assert np.allclose(np.diag(cov.values), [1.0, 1.0, 1.0])
        assert np.allclose(cov.values, cov.values.T)

    def test_set_block_from_array_covariance(self):
        """Test set_block with explicit covariance array."""
        block = Block('state', pd.Series([1, 2], index=['a', 'b']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        cov_array = np.array([[1.0, 0.5], [0.5, 2.0]])
        cov.set_block('state', covariance=cov_array)
        
        assert np.allclose(cov.values, cov_array)

    def test_set_block_from_dataframe_covariance(self):
        """Test set_block with DataFrame covariance."""
        block = Block('state', pd.Series([1, 2], index=['a', 'b']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        cov_df = pd.DataFrame([[1.0, 0.5], [0.5, 2.0]], 
                             index=['a', 'b'], columns=['a', 'b'])
        cov.set_block('state', covariance=cov_df)
        
        assert np.allclose(cov.values, cov_df.values)

    def test_set_block_from_callable_covariance(self):
        """Test set_block with callable covariance."""
        block = Block('state', pd.Series([1, 2, 3], index=['a', 'b', 'c']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        def covariance_func(idx_row, idx_col):
            # Simple function that returns ones (all fully correlated)
            return np.ones((len(idx_row), len(idx_col)))
        
        cov.set_block('state', covariance=covariance_func)
        
        expected = np.ones((3, 3))
        assert np.allclose(cov.values, expected)

    def test_set_block_multiblock_vector(self):
        """Test set_block on one block of a multi-block vector."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4, 5], index=['a', 'b', 'c']))
        vector = Vector([block1, block2])
        cov = CovarianceMatrix.from_vector(vector)
        
        # Set first block only
        cov.set_block('b1', sigma=2.0)
        
        # Block 1 (rows/cols 0-1) should have 4.0 on diagonal
        assert np.allclose(cov.values[0, 0], 4.0)
        assert np.allclose(cov.values[1, 1], 4.0)
        # Block 2 (rows/cols 2-4) should still be zero
        assert np.allclose(cov.values[2:, 2:], 0.0)

    def test_set_block_chaining(self):
        """Test that set_block returns self for method chaining."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4], index=['a', 'b']))
        vector = Vector([block1, block2])
        cov = CovarianceMatrix.from_vector(vector)
        
        result = cov.set_block('b1', sigma=1.0).set_block('b2', sigma=2.0)
        
        assert result is cov
        assert np.allclose(cov.values[0, 0], 1.0)
        assert np.allclose(cov.values[2, 2], 4.0)

    def test_set_block_cannot_specify_both_sigma_covariance(self):
        """Test that specifying both sigma and covariance raises error."""
        block = Block('state', pd.Series([1, 2], index=['a', 'b']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        with pytest.raises(ValueError, match="Specify 'covariance' OR 'sigma'"):
            cov.set_block('state', sigma=1.0, covariance=np.eye(2))

    def test_set_block_must_specify_sigma_or_covariance(self):
        """Test that must specify either sigma or covariance."""
        block = Block('state', pd.Series([1, 2], index=['a', 'b']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        with pytest.raises(ValueError, match="Must specify either 'covariance' or 'sigma'"):
            cov.set_block('state')


class TestCovarianceMatrixSetInteraction:
    """Tests for CovarianceMatrix.set_interaction() method."""

    def test_set_interaction_scalar_sigma(self):
        """Test set_interaction with scalar sigma for cross-covariance."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4, 5], index=['a', 'b', 'c']))
        vector = Vector([block1, block2])
        cov = CovarianceMatrix.from_vector(vector)
        
        cov.set_interaction('b1', 'b2', sigma=0.5)
        
        # Cross block should have 0.25 (0.5^2)
        assert np.allclose(cov.values[0, 2], 0.25)
        # Should be symmetric
        assert np.allclose(cov.values[2, 0], 0.25)

    def test_set_interaction_tuple_sigma(self):
        """Test set_interaction with tuple of sigmas."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4], index=['a', 'b']))
        vector = Vector([block1, block2])
        cov = CovarianceMatrix.from_vector(vector)
        
        cov.set_interaction('b1', 'b2', sigma=(1.0, 2.0))
        
        # Should be 1.0 * 2.0 = 2.0
        assert np.allclose(cov.values[0, 2], 2.0)
        assert np.allclose(cov.values[2, 0], 2.0)

    def test_set_interaction_from_array_covariance(self):
        """Test set_interaction with explicit covariance array."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4], index=['a', 'b']))
        vector = Vector([block1, block2])
        cov = CovarianceMatrix.from_vector(vector)
        
        cross_cov = np.array([[0.1, 0.2], [0.3, 0.4]])
        cov.set_interaction('b1', 'b2', covariance=cross_cov)
        
        # Check both blocks and symmetry
        assert np.allclose(cov.values[0:2, 2:4], cross_cov)
        assert np.allclose(cov.values[2:4, 0:2], cross_cov.T)

    def test_set_interaction_with_kernel(self):
        """Test set_interaction with kernel function."""
        idx1 = pd.Index([0.0, 1.0], name='x')
        idx2 = pd.Index([2.0, 3.0], name='y')
        
        block1 = Block('b1', pd.Series([1, 2], index=idx1))
        block2 = Block('b2', pd.Series([3, 4], index=idx2))
        vector = Vector([block1, block2])
        cov = CovarianceMatrix.from_vector(vector)
        
        def distance_kernel(idx_row, idx_col):
            x = idx_row.values[:, None]
            y = idx_col.values[None, :]
            dists = np.abs(x - y)
            return np.exp(-dists / 1.0)
        
        cov.set_interaction('b1', 'b2', sigma=1.0, kernel=distance_kernel)
        
        # Should be symmetric
        assert np.allclose(cov.values[0:2, 2:4], cov.values[2:4, 0:2].T)

    def test_set_interaction_same_block_calls_set_block(self):
        """Test that set_interaction with same block calls set_block."""
        block = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        vector = Vector([block])
        cov = CovarianceMatrix.from_vector(vector)
        
        cov.set_interaction('b1', 'b1', sigma=2.0)
        
        # Should behave like set_block
        expected = np.eye(2) * 4.0
        assert np.allclose(cov.values, expected)

    def test_set_interaction_three_blocks(self):
        """Test set_interaction with three blocks."""
        block1 = Block('b1', pd.Series([1], index=['a']))
        block2 = Block('b2', pd.Series([2], index=['b']))
        block3 = Block('b3', pd.Series([3], index=['c']))
        vector = Vector([block1, block2, block3])
        cov = CovarianceMatrix.from_vector(vector)
        
        # Set all interactions
        cov.set_block('b1', sigma=1.0)
        cov.set_block('b2', sigma=2.0)
        cov.set_block('b3', sigma=3.0)
        cov.set_interaction('b1', 'b2', sigma=0.5)
        cov.set_interaction('b1', 'b3', sigma=0.5)
        cov.set_interaction('b2', 'b3', sigma=0.5)
        
        # Check structure
        assert cov.shape == (3, 3)
        assert np.allclose(cov.values[0, 0], 1.0)
        assert np.allclose(cov.values[1, 1], 4.0)
        assert np.allclose(cov.values[2, 2], 9.0)
        assert np.allclose(cov.values[0, 1], 0.25)
        assert np.allclose(cov.values[1, 0], 0.25)

    def test_set_interaction_cannot_have_kernel_with_covariance(self):
        """Test that kernel cannot be used with covariance."""
        block1 = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        block2 = Block('b2', pd.Series([3, 4], index=['a', 'b']))
        vector = Vector([block1, block2])
        cov = CovarianceMatrix.from_vector(vector)
        
        def kernel_func(idx1, idx2):
            return np.ones((len(idx1), len(idx2)))
        
        with pytest.raises(ValueError, match="Cannot apply 'kernel' to 'covariance'"):
            cov.set_interaction('b1', 'b2', covariance=np.eye(2), kernel=kernel_func)


class TestCovarianceMatrixAddition:
    """Tests for CovarianceMatrix addition."""

    def test_add_two_covariance_matrices(self):
        """Test adding two CovarianceMatrices."""
        idx = ['a', 'b', 'c']
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
        idx = pd.MultiIndex.from_product([['b1', 'b2'], [1, 2]])
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
        data = pd.DataFrame(np.random.randn(5, 3), 
                           index=[f'obs_{i}' for i in range(5)],
                           columns=[f'state_{i}' for i in range(3)])
        forward_op = ForwardOperator(data)
        
        assert forward_op.shape == (5, 3)

    def test_forward_operator_is_matrix_subclass(self):
        """Test that ForwardOperator is a Matrix subclass."""
        data = pd.DataFrame(np.ones((3, 3)))
        forward_op = ForwardOperator(data)
        
        assert isinstance(forward_op, Matrix)

    def test_forward_operator_rectangular(self):
        """Test ForwardOperator with rectangular matrix."""
        data = pd.DataFrame(np.random.randn(10, 5), 
                           index=range(10),
                           columns=range(5))
        forward_op = ForwardOperator(data)
        
        assert forward_op.shape[0] > forward_op.shape[1]


class TestPrepareMatrix:
    """Tests for prepare_matrix function."""

    def test_prepare_matrix_basic(self):
        """Test basic prepare_matrix functionality."""
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           index=['a', 'b'], 
                           columns=['x', 'y'])
        result = prepare_matrix(data, False, None, False, None, float_precision=None)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_prepare_matrix_sanitizes_indices(self):
        """Test that prepare_matrix sanitizes indices."""
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           index=['1.0', '2.0'], 
                           columns=['3.0', '4.0'])
        result = prepare_matrix(data, False, None, False, None, float_precision=2)
        
        # Indices should be converted to numeric
        assert pd.api.types.is_numeric_dtype(result.index) or isinstance(result.index[0], (int, float, np.number))

    def test_prepare_matrix_with_row_promotion(self):
        """Test prepare_matrix with row promotion."""
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           index=['a', 'b'], 
                           columns=['x', 'y'])
        
        block = Block('b1', pd.Series([1, 2], index=['a', 'b']))
        vector = Vector([block])
        
        result = prepare_matrix(data, row_promote=True, row_asm=vector, 
                               col_promote=False, col_asm=None,
                               float_precision=None)
        
        # Should have block level in index
        if isinstance(result.index, pd.MultiIndex):
            assert 'block' in result.index.names

    def test_prepare_matrix_with_col_promotion(self):
        """Test prepare_matrix with column promotion."""
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           index=['a', 'b'], 
                           columns=['x', 'y'])
        
        block = Block('b1', pd.Series([1, 2], index=['x', 'y']))
        vector = Vector([block])
        
        result = prepare_matrix(data, row_promote=False, row_asm=None,
                               col_promote=True, col_asm=vector,
                               float_precision=None)
        
        # Should have block level in columns
        if isinstance(result.columns, pd.MultiIndex):
            assert 'block' in result.columns.names

    def test_prepare_matrix_copies_data(self):
        """Test that prepare_matrix creates a copy."""
        data = pd.DataFrame([[1, 2], [3, 4]])
        result = prepare_matrix(data, False, None, False, None, float_precision=None)
        
        # Modify original
        data.iloc[0, 0] = 999
        
        # Result should not change
        assert result.iloc[0, 0] == 1


class TestMatrixOperations:
    """Tests for Matrix utility operations."""

    def test_matrix_reindex(self):
        """Test reindexing a matrix."""
        data = pd.DataFrame([[1, 2], [3, 4]], 
                           index=['a', 'b'], 
                           columns=['x', 'y'])
        matrix = Matrix(data)
        
        reindexed = matrix.data.reindex(index=['a', 'b', 'c'], fill_value=0)
        
        assert reindexed.shape == (3, 2)
        assert reindexed.loc['c'].tolist() == [0, 0]

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
