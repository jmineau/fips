"""Tests for covariance module."""

from functools import reduce

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
from fips.kernels import GridSpatialDecay, GridTimeDecay


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
