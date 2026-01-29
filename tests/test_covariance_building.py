"""Test covariance matrix building methods with small, verifiable datasets."""

import numpy as np
import pandas as pd

from fips import Block, CovarianceMatrix, Vector
from fips.kernels import exponential_decay_kernel


def test_diagonal_from_scalar():
    """Test creating diagonal covariance from scalar sigma."""
    # Create simple state vector
    index = pd.Index([0, 1, 2], name="location")
    flux_block = Block(pd.Series([1.0, 1.0, 1.0], index=index), name="flux")
    state = Vector("state", [flux_block])

    # Build covariance with scalar sigma
    cov = CovarianceMatrix.from_vector(state)
    cov.set_block("flux", sigma=2.0)

    # Expected: diagonal matrix with sigma^2 = 4.0 on diagonal
    expected = np.eye(3) * 4.0

    np.testing.assert_allclose(cov.values, expected)
    print("✓ diagonal from scalar sigma works correctly")


def test_diagonal_from_array():
    """Test creating diagonal covariance from array of sigmas."""
    # Create simple state vector
    index = pd.Index([0, 1, 2], name="location")
    flux_block = Block(pd.Series([1.0, 1.0, 1.0], index=index), name="flux")
    state = Vector("state", [flux_block])

    # Build covariance with array of sigmas
    cov = CovarianceMatrix.from_vector(state)
    sigmas = np.array([1.0, 2.0, 3.0])
    cov.set_block("flux", sigma=sigmas)

    # Expected: diagonal with sigma^2 values
    expected = np.diag([1.0, 4.0, 9.0])

    np.testing.assert_allclose(cov.values, expected)
    print("✓ diagonal from array of sigmas works correctly")


def test_set_block_with_kernel():
    """Test applying correlation kernel to a single block."""
    # Create simple temporal data
    times = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    index = pd.Index(times, name="time")
    flux_block = Block(pd.Series([1.0, 1.0, 1.0], index=index), name="flux")
    state = Vector("state", [flux_block])

    # Start with diagonal covariance
    cov = CovarianceMatrix.from_vector(state)

    # Apply exponential decay kernel (1-day e-folding)
    kernel = exponential_decay_kernel(times, length_scale=1.0)
    cov.set_block("flux", sigma=1.0, kernel=lambda idx_row, idx_col: kernel)

    # Expected: exponential decay pattern
    # Day 0-1: exp(-1) ≈ 0.368
    # Day 0-2: exp(-2) ≈ 0.135
    # Day 1-2: exp(-1) ≈ 0.368
    expected = np.array(
        [
            [1.0, np.exp(-1), np.exp(-2)],
            [np.exp(-1), 1.0, np.exp(-1)],
            [np.exp(-2), np.exp(-1), 1.0],
        ]
    )

    np.testing.assert_allclose(cov.values, expected, rtol=1e-5)
    print("✓ set_block with kernel works correctly")


def test_multiblock_set_block():
    """Test applying kernel to only one block in multi-block covariance."""
    # Create multi-block state
    times = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    flux_index = pd.Index(times, name="time")
    bias_index = pd.Index([0], name="location")

    flux = Block(pd.Series([1.0, 1.0], index=flux_index), name="flux")
    bias = Block(pd.Series([2.0], index=bias_index), name="bias")
    state = Vector("state", [flux, bias])

    # Build diagonal covariance
    cov = CovarianceMatrix.from_vector(state)
    cov.set_block("bias", sigma=np.sqrt(2.0))

    # Apply kernel only to flux block
    kernel = exponential_decay_kernel(times, length_scale=1.0)
    cov.set_block("flux", sigma=1.0, kernel=lambda idx_row, idx_col: kernel)

    # Expected: flux block is correlated, bias remains diagonal
    expected = np.array(
        [[1.0, np.exp(-1), 0.0], [np.exp(-1), 1.0, 0.0], [0.0, 0.0, 2.0]]
    )

    np.testing.assert_allclose(cov.values, expected, rtol=1e-5)
    print("✓ set_block on multi-block covariance works correctly")


def test_covariance_addition():
    """Test adding two covariance matrices together."""
    # Create simple state
    index = pd.Index([0, 1], name="location")
    flux = Block(pd.Series([1.0, 1.0], index=index), name="flux")
    state = Vector("state", [flux])

    # Build two covariance matrices
    cov1 = CovarianceMatrix.from_vector(state)
    cov1.set_block("flux", sigma=1.0)

    cov2 = CovarianceMatrix.from_vector(state)
    cov2.set_block("flux", sigma=np.sqrt(2.0))  # sigma=sqrt(2) -> variance=2

    # Add them
    cov_sum = cov1 + cov2

    # Expected: sum of the two matrices
    expected = np.array([[3.0, 0.0], [0.0, 3.0]])

    np.testing.assert_allclose(cov_sum.values, expected)
    print("✓ covariance addition works correctly")


def test_multicomponent_covariance_composition():
    """Test composing multi-component covariance (realistic flux use case)."""
    # Create simple spatial state
    locations = pd.Index([0, 1], name="location")
    flux = Block(pd.Series([1.0, 1.0], index=locations), name="flux")
    state = Vector("state", [flux])

    # Component 1: Spatially uncorrelated background error
    background = CovarianceMatrix.from_vector(state)
    background.set_block("flux", sigma=np.sqrt(0.5))

    # Component 2: Spatially correlated transport error
    transport = CovarianceMatrix.from_vector(state)
    # Apply full correlation (for simplicity)
    kernel = np.array([[1.0, 0.8], [0.8, 1.0]])
    transport.set_block("flux", sigma=np.sqrt(0.3), kernel=lambda r, c: kernel)

    # Combine components
    total = background + transport

    # Expected: sum of background (diagonal) and transport (correlated)
    expected = np.array([[0.5 + 0.3, 0.3 * 0.8], [0.3 * 0.8, 0.5 + 0.3]])

    np.testing.assert_allclose(total.values, expected, rtol=1e-5)
    print("✓ multi-component covariance composition works correctly")


def test_block_structure_preservation():
    """Verify that block structure is preserved through operations."""
    # Create multi-block state
    flux_index = pd.Index([0, 1], name="location")
    bias_index = pd.Index([10], name="location")

    flux = Block(pd.Series([1.0, 1.0], index=flux_index), name="flux")
    bias = Block(pd.Series([2.0], index=bias_index), name="bias")
    state = Vector("state", [flux, bias])

    # Build covariance
    cov = CovarianceMatrix.from_vector(state)
    cov.set_block("flux", sigma=1.0)
    cov.set_block("bias", sigma=np.sqrt(2.0))

    # Verify index structure matches input
    assert cov.index.equals(state.index)
    assert cov.columns.equals(state.index)

    # Verify block names are accessible
    assert "flux" in state.blocks
    assert "bias" in state.blocks

    print("✓ block structure preservation works correctly")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Covariance Matrix Building Methods")
    print("=" * 60 + "\n")

    print("SINGLE BLOCK TESTS:")
    test_diagonal_from_scalar()
    test_diagonal_from_array()
    test_set_block_with_kernel()
    test_covariance_addition()
    test_multicomponent_covariance_composition()

    print("\nMULTI-BLOCK TESTS:")
    test_multiblock_set_block()
    test_block_structure_preservation()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")
