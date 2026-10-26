"""
Tests for the covariance models (``fips.covariance.models``).

``with_std`` (a Vector gives a Covariance at once, a number or a Series on a
block's labels a model; a plain array is refused),
``build`` on a space (the levels read, the order of the space, the form kept),
sums of models and covariances, and pickling.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from fips import (
    Constant,
    Covariance,
    Exponential,
    Independent,
    Kronecker,
    Space,
    Vector,
)
from fips.covariance import CovarianceModel
from fips.linalg import (
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    SparseOperator,
)

X = pd.Index([0.0, 1.0, 2.0, 4.0], name="x")
SPACE = Space({"flux": X})
SITES = pd.MultiIndex.from_product(
    [[f"s{k}" for k in range(10)], pd.date_range("2024-01-01", periods=3, freq="h")],
    names=["site", "obs_time"],
)
OBS = Space({"co2": SITES})


def _scaled(C: np.ndarray, std) -> np.ndarray:
    """``std_i C_ij std_j``."""
    std = np.broadcast_to(np.asarray(std, dtype=float), (len(C),))
    return std[:, None] * C * std[None, :]


class TestWithStd:
    """``with_std`` makes a covariance from a correlation."""

    def test_number_gives_a_model(self):
        """The model waits for a space; built, it is ``std^2 C``."""
        C = Exponential("x", 1.5).correlation(X.to_frame(index=False))
        for value in (0.7, np.float32(0.7), np.array(0.7), 2):
            model = Exponential("x", 1.5).with_std(value)
            assert isinstance(model, CovarianceModel)
            S = model.build(SPACE)
            assert isinstance(S, Covariance) and S.space == SPACE
            np.testing.assert_allclose(S.dense(), _scaled(C, float(value)), rtol=1e-7)

    def test_series_gives_a_model_lined_up_by_label(self):
        """A Series on the block's labels is read by label, in any order."""
        std = np.array([1.0, 2.0, 0.5, 0.0])
        C = Exponential("x", 1.5).correlation(X.to_frame(index=False))
        in_order = pd.Series(std, index=X)
        for series in (in_order, in_order.iloc[[3, 1, 0, 2]]):
            model = Exponential("x", 1.5).with_std(series)
            assert isinstance(model, CovarianceModel)
            S = model.build(SPACE)
            assert S.space == SPACE
            np.testing.assert_array_equal(S.dense(), _scaled(C, std))

    def test_series_on_a_multiindex_in_another_level_order(self):
        """The levels of the Series' index may come in any order, as in ``Space.align``."""
        std = np.linspace(0.1, 3.0, len(SITES))
        series = pd.Series(std, index=SITES).reorder_levels(["obs_time", "site"])
        S = Independent().with_std(series.sample(frac=1.0, random_state=0)).build(OBS)
        np.testing.assert_array_equal(S.variances.values, std**2)

    def test_series_equals_the_vector_on_the_same_labels(self):
        """A Series and a Vector of the same standard deviations give the same covariance."""
        std = pd.Series([1.0, 2.0, 0.5, 3.0], index=X)
        model = Exponential("x", 1.5)
        from_series = model.with_std(std).build(SPACE)
        from_vector = model.with_std(Vector(std.to_numpy(), SPACE))
        np.testing.assert_array_equal(from_series.dense(), from_vector.dense())

    def test_vector_gives_a_covariance_on_its_space(self):
        """A Vector brings its labels: the covariance is made at once."""
        std = Vector([1.0, 2.0, 3.0, 4.0], SPACE)
        S = Independent().with_std(std)
        assert isinstance(S, Covariance) and S.space is SPACE
        np.testing.assert_array_equal(S.variances.values, [1.0, 4.0, 9.0, 16.0])

    @pytest.mark.parametrize(
        "std",
        [np.ones(4), [1.0, 1.0, 1.0, 1.0], (1.0, 1.0), pd.Index([1.0, 2.0]), [[1.0]]],
        ids=["array", "list", "tuple", "index", "nested"],
    )
    def test_plain_array_raises(self, std):
        """An array has no labels, so it is refused, and the message says what to pass."""
        with pytest.raises(TypeError, match="Pass a Series") as raised:
            Independent().with_std(std)
        assert "Vector" in str(raised.value)

    @pytest.mark.parametrize("std", ["0.5", None, {"x": 1.0}])
    def test_other_types_raise(self, std):
        """Only a number, a Series or a Vector is a standard deviation."""
        with pytest.raises(TypeError, match="with_std takes a number"):
            Independent().with_std(std)  # pyrefly: ignore[no-matching-overload]

    @pytest.mark.parametrize(
        ("std", "match"),
        [
            (-1.0, "not negative"),
            (np.inf, "finite"),
            (pd.Series([1.0, np.nan, 1.0, 1.0], index=X), "finite"),
            (pd.Series([1.0, -2.0, 1.0, 1.0], index=X), "not negative"),
        ],
    )
    def test_bad_std_raises(self, std, match):
        """Standard deviations are finite and not negative."""
        with pytest.raises(ValueError, match=match):
            Independent().with_std(std)

    @pytest.mark.parametrize(
        ("index", "match"),
        [
            (pd.Index([0.0, 1.0, 1.0, 4.0], name="x"), "label twice: 1.0"),
            (pd.Index([0.0, 1.0, 2.0, 4.0]), "must be named"),
            (SPACE.index, "'block' level"),
        ],
        ids=["duplicate", "unnamed", "block level"],
    )
    def test_series_with_bad_labels_raises(self, index, match):
        """A Series' labels are unique, named, and one block's, without a ``block`` level."""
        with pytest.raises(ValueError, match=match):
            Independent().with_std(pd.Series(1.0, index=index))

    @pytest.mark.parametrize(
        ("labels", "match"),
        [
            ([0.0, 1.0, 2.0], "missing from the standard deviations: 4.0"),
            ([0.0, 1.0, 2.0, 4.0, 5.0], "only in the standard deviations: 5.0"),
        ],
        ids=["missing", "extra"],
    )
    def test_series_on_other_labels_raises_when_built(self, labels, match):
        """A label of the space with no standard deviation, or one too many, raises, naming it."""
        model = Independent().with_std(pd.Series(1.0, index=pd.Index(labels, name="x")))
        with pytest.raises(ValueError, match=match):
            model.build(SPACE)

    def test_series_on_other_levels_raises_when_built(self):
        """A Series on another level is not the block's labels."""
        model = Independent().with_std(pd.Series(1.0, index=X.rename("y")))
        with pytest.raises(ValueError, match=r"levels \['x'\] here and \['y'\]"):
            model.build(SPACE)

    def test_series_on_a_space_of_two_blocks_raises(self):
        """A Series is one block's: on two blocks, the message points to ``block_diag``."""
        two = Space({"flux": X, "bias": pd.Index(["A"], name="site")})
        model = Independent().with_std(pd.Series(1.0, index=X))
        with pytest.raises(ValueError, match=r"block_diag\(flux=\.\.\., bias=\.\.\.\)"):
            model.build(two)
        S = Covariance.block_diag(flux=model, bias=Independent().with_std(2.0)).build(
            two
        )
        np.testing.assert_array_equal(S.variances.values, [1.0, 1.0, 1.0, 1.0, 4.0])

    def test_std_series_is_kept_as_given(self):
        """The model holds its own read-only copy of the values."""
        std = pd.Series(np.ones(4), index=X)
        model = Independent().with_std(std)
        std.iloc[0] = 5.0
        np.testing.assert_array_equal(model.build(SPACE).variances.values, np.ones(4))


class TestBuild:
    """``build`` puts a model on a space."""

    @pytest.mark.parametrize(
        ("model", "form"),
        [
            (Independent(), DiagonalOperator),
            (Exponential("obs_time", "1h"), DenseOperator),
            (Exponential("obs_time", "1h", within="site"), SparseOperator),
            (Constant(within="site"), SparseOperator),
            (
                Kronecker(Exponential("obs_time", "1h"), Constant(within="site")),
                KroneckerOperator,
            ),
        ],
    )
    def test_forms(self, model, form):
        """Each model keeps its cheapest form once scaled."""
        std = np.linspace(0.5, 1.5, len(OBS))
        S = model.with_std(pd.Series(std, index=SITES)).build(OBS)
        assert isinstance(S.data, form)
        C = model.correlation(SITES.to_frame(index=False))
        np.testing.assert_allclose(S.dense(), _scaled(C, std), rtol=1e-15)

    def test_reads_the_labels_in_the_space_order(self):
        """The rows follow the space, whatever order its labels are in."""
        shuffled = Space({"flux": X[[2, 0, 3, 1]]})
        S = Exponential("x", 1.0).with_std(1.0).build(shuffled)
        x = shuffled.select("flux").to_numpy()
        np.testing.assert_allclose(
            S.dense(), np.exp(-np.abs(x[:, None] - x[None, :])), rtol=1e-15
        )

    def test_block_level_can_be_read(self):
        """``block`` is a level like any other: independent blocks with ``within``."""
        space = Space({"a": X, "b": X})
        S = Exponential("x", 1.0, within="block").with_std(1.0).build(space)
        np.testing.assert_allclose(
            S["a", "b"].dense(), np.zeros((4, 4)), rtol=0, atol=0
        )
        A = Exponential("x", 1.0).with_std(1.0).build(space)
        assert (A["a", "b"].dense() > 0).all()

    def test_errors(self):
        """Not a space, a level the space lacks, or one a block lacks."""
        model = Exponential("x", 1.0).with_std(1.0)
        with pytest.raises(TypeError, match="Space"):
            model.build(X)  # pyrefly: ignore[bad-argument-type]
        with pytest.raises(KeyError, match="'y'"):
            Exponential("y", 1.0).with_std(1.0).build(SPACE)
        two = Space({"flux": X, "bias": pd.Index(["A"], name="site")})
        with pytest.raises(ValueError, match="'bias'"):
            model.build(two)

    def test_name(self):
        """``build`` names the covariance."""
        assert (
            Independent().with_std(1.0).build(SPACE, name="obs_cov").name == "obs_cov"
        )


class TestSums:
    """A sum of covariance models (and covariances) is built term by term."""

    def test_sum_of_models(self):
        """The sum of the built terms, in the forms the terms allow."""
        a = Independent().with_std(0.1)
        b = Exponential("obs_time", "2h", within="site").with_std(0.5)
        total = a + b
        assert isinstance(total, CovarianceModel)
        S = total.build(OBS)
        assert isinstance(S.data, SparseOperator)
        np.testing.assert_allclose(
            S.dense(), a.build(OBS).dense() + b.build(OBS).dense(), rtol=1e-15
        )

    def test_covariance_on_either_side(self):
        """A covariance in a sum is aligned to the space by label."""
        C = Covariance.diagonal([1.0, 2.0, 3.0, 4.0], SPACE)
        m = Exponential("x", 1.0).with_std(1.0)
        shuffled = Space({"flux": X[[3, 1, 0, 2]]})
        left, right = (C + m).build(shuffled), (m + C).build(shuffled)
        np.testing.assert_array_equal(left.dense(), right.dense())
        np.testing.assert_array_equal(left.variances.values, [5.0, 3.0, 2.0, 4.0])

    def test_terms_are_flat(self):
        """``a + b + c`` holds three terms."""
        a = Independent().with_std(1.0)
        total = a + a + Constant().with_std(1.0)
        assert len(total.terms) == 3  # pyrefly: ignore[missing-attribute]
        np.testing.assert_array_equal(
            total.build(SPACE).dense(), 2 * np.eye(4) + np.ones((4, 4))
        )

    def test_other_operands_raise(self):
        """A number or an array is not a covariance."""
        with pytest.raises(TypeError):
            Independent().with_std(1.0) + 1.0  # pyrefly: ignore[unsupported-operation]
        with pytest.raises(TypeError):
            Covariance.diagonal(1.0, SPACE) + 1.0  # pyrefly: ignore[unsupported-operation]


def test_models_pickle_and_print():
    """Each covariance model pickles and prints how it was made."""
    models = [
        Exponential("x", 1.0).with_std(pd.Series(np.ones(4), index=X)),
        Independent().with_std(1.0) + Covariance.diagonal(1.0, SPACE),
        Covariance.block_diag(flux=Independent().with_std(2.0)),
    ]
    for model in models:
        loaded = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(
            loaded.build(SPACE).dense(), model.build(SPACE).dense()
        )
    assert repr(models[0]).endswith(".with_std(<Series of 4 values>)")
    assert " + " in repr(models[1])
    assert repr(models[2]).startswith("Covariance.block_diag(flux=Independent()")
