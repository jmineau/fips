"""Tests for covariance module."""

import numpy as np
import pandas as pd
import pytest

from fips.covariance import (
    BlockDecayError,
    CovarianceBuilder,
    CovarianceMatrix,
    DiagonalError,
    KroneckerError,
)


class TestCovarianceMatrix:
    """Tests for CovarianceMatrix class."""

    def test_covariance_matrix_creation(self):
        """Test basic CovarianceMatrix creation."""
        # Create MultiIndex with block level
        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[1, 0.5], [0.5, 2]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        assert cov.shape == (2, 2)
        assert isinstance(cov.data, pd.DataFrame)

    def test_covariance_matrix_is_matrix(self):
        """Test that CovarianceMatrix is a Matrix subclass."""
        from fips.matrix import Matrix

        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[1, 0.5], [0.5, 2]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        assert isinstance(cov, Matrix)

    def test_covariance_matrix_variances_property(self):
        """Test variances property returns diagonal elements."""
        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[2.0, 0.5], [0.5, 3.0]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        variances = cov.variances
        assert isinstance(variances.data, pd.Series) or isinstance(variances, pd.Series)
        assert np.allclose(
            variances.values if hasattr(variances, "values") else variances, [2.0, 3.0]
        )

    def test_covariance_matrix_force_symmetry_lower(self):
        """Test force_symmetry method with lower triangle."""
        # Create asymmetric matrix with block level
        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[1.0, 0.3], [0.5, 2.0]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        sym = cov.force_symmetry(keep="lower")

        # Check that result is symmetric
        assert np.allclose(sym.values, sym.values.T)
        # Check that diagonal is preserved
        assert np.allclose(np.diag(sym.values), [1.0, 2.0])
        # Check that lower triangle was used
        assert np.allclose(sym.values[1, 0], 0.5)

    def test_covariance_matrix_force_symmetry_upper(self):
        """Test force_symmetry method with upper triangle."""
        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[1.0, 0.3], [0.5, 2.0]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        sym = cov.force_symmetry(keep="upper")

        # Check that result is symmetric
        assert np.allclose(sym.values, sym.values.T)
        # Check that upper triangle was used
        assert np.allclose(sym.values[1, 0], 0.3)

    def test_covariance_matrix_force_symmetry_invalid_keep(self):
        """Test force_symmetry with invalid keep parameter."""
        idx = pd.MultiIndex.from_tuples(
            [("block1", "a"), ("block1", "b")], names=["block", "x"]
        )
        data = pd.DataFrame([[1.0, 0.5], [0.5, 2.0]], index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        with pytest.raises(ValueError, match="keep must be"):
            cov.force_symmetry(keep="invalid")

    def test_covariance_matrix_with_multiindex(self):
        """Test CovarianceMatrix with MultiIndex."""
        idx = pd.MultiIndex.from_product(
            [["b1", "b2"], ["a", "b"], [1, 2]], names=["block", "x", "y"]
        )
        data = pd.DataFrame(np.eye(8), index=idx, columns=idx)
        cov = CovarianceMatrix(data)

        assert isinstance(cov.index, pd.MultiIndex)
        assert cov.shape == (8, 8)


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
        assert np.allclose(aligned.values, 5.0)

    def test_align_variances_with_missing_names_raises(self):
        """Test that unnamed index levels raise error."""
        idx = pd.MultiIndex.from_product(
            [["b1"], ["a", "b"], [1, 2]], names=[None, "y", None]
        )
        variances = pd.Series([1.0, 2.0], index=idx.droplevel([0, 2]).drop_duplicates())

        component = DiagonalError("test", variances)

        with pytest.raises(ValueError, match="All levels.*must be named"):
            component._align_variances(idx)
