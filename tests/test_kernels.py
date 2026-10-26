"""
Tests for the correlation models (``fips.covariance.models`` and ``fips.covariance.kernels``).

Each kernel against its formula written out, on numeric, datetime and
geographic levels; ``within`` groups (the date cut at midnight, the
hour-of-day mask); the parameters each model refuses; ``Independent`` and
``Kronecker``; and the extension point, a model subclass.
"""

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from fips import (
    Constant,
    Exponential,
    Gaussian,
    Independent,
    Kronecker,
    Matern,
    Space,
)
from fips.covariance import CorrelationModel
from fips.covariance.distances import chordal_matrix, haversine_matrix
from fips.linalg import (
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    SparseOperator,
)

X = pd.DataFrame({"x": [0.0, 0.5, 1.5, 4.0, 9.0]})
D = np.abs(X["x"].to_numpy()[:, None] - X["x"].to_numpy()[None, :])
HOURS = pd.date_range("2024-01-01 20:00", periods=8, freq="h")


def _coords(**levels) -> pd.DataFrame:
    """A frame of coordinates, one column per level."""
    return pd.DataFrame(levels)


# ---------------------------------------------------------------------------
# The kernels against their formulas
# ---------------------------------------------------------------------------


class TestFormulas:
    """Each kernel is its formula of the distance divided by the scale."""

    def test_exponential(self):
        """``exp(-d / l)``."""
        np.testing.assert_allclose(
            Exponential("x", 2.0).correlation(X), np.exp(-D / 2.0), rtol=1e-15
        )

    def test_gaussian(self):
        """``exp(-d**2 / (2 l**2))``."""
        np.testing.assert_allclose(
            Gaussian("x", 2.0).correlation(X), np.exp(-(D**2) / 8.0), rtol=1e-15
        )

    @pytest.mark.parametrize(
        ("nu", "formula"),
        [
            (0.5, lambda r: np.exp(-r)),
            (1.5, lambda r: (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)),
            (
                2.5,
                lambda r: (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r),
            ),
        ],
    )
    def test_matern_half_integers(self, nu, formula):
        """The closed forms for nu = 1/2, 3/2, 5/2 (r = d / l)."""
        np.testing.assert_allclose(
            Matern("x", 2.0, nu).correlation(X), formula(D / 2.0), rtol=1e-12
        )

    def test_matern_tends_to_gaussian(self):
        """The largest nu is close to the Gaussian; near and far apart stay finite."""
        np.testing.assert_allclose(
            Matern("x", 2.0, 50.0).correlation(X),
            Gaussian("x", 2.0).correlation(X),
            atol=5e-3,
        )
        c = _coords(x=[0.0, 1e-9, 1e4])
        C = Matern("x", 1.0, 50.0).correlation(c)
        np.testing.assert_allclose(C, [[1, 1, 0], [1, 1, 0], [0, 0, 1]], atol=1e-15)

    def test_time(self):
        """A datetime level: the time apart divided by a time scale."""
        hours = ((HOURS - HOURS[0]) / pd.Timedelta("1h")).to_numpy()
        expected = np.exp(-np.abs(hours[:, None] - hours[None, :]) / 3)
        np.testing.assert_allclose(
            Exponential("t", "3h").correlation(_coords(t=HOURS)), expected, rtol=1e-15
        )

    def test_timedelta(self):
        """A timedelta level (lags) is a time, as a datetime level is."""
        lags = pd.to_timedelta([0.0, 1.5, 3.0, 10.0], unit="h")
        hours = np.array([0.0, 1.5, 3.0, 10.0])
        expected = np.exp(-np.abs(hours[:, None] - hours[None, :]) / 2)
        np.testing.assert_allclose(
            Exponential("lag", "2h").correlation(_coords(lag=lags)),
            expected,
            rtol=1e-15,
        )
        space = Space({"co2": pd.TimedeltaIndex(["0h", "1h", "2h"], name="lag")})
        C = Exponential("lag", "1h").with_std(1.0).build(space).dense()
        k = np.arange(3.0)
        np.testing.assert_allclose(
            C, np.exp(-np.abs(k[:, None] - k[None, :])), rtol=1e-15
        )

    def test_euclidean_along_two_levels(self):
        """Two numeric levels: the straight-line distance."""
        c = _coords(x=[0.0, 3.0, 0.0], y=[0.0, 4.0, 1.0])
        d = np.array([[0, 5, 1], [5, 0, np.sqrt(18)], [1, np.sqrt(18), 0]])
        np.testing.assert_allclose(
            Exponential(["x", "y"], 5.0).correlation(c), np.exp(-d / 5), rtol=1e-15
        )

    @pytest.mark.parametrize("metric", ["haversine", "chordal"])
    def test_on_the_earth(self, metric):
        """Latitude and longitude in degrees, the distance in km."""
        c = _coords(lat=[40.0, 40.5, 41.0], lon=[-112.0, -111.5, -112.0])
        measure = haversine_matrix if metric == "haversine" else chordal_matrix
        d = measure(c["lat"], c["lon"])
        np.testing.assert_array_equal(
            Exponential(["lat", "lon"], 20.0, metric=metric).correlation(c),
            np.exp(-(d / 20.0)),
        )

    def test_chordal_is_the_chord_of_the_great_circle(self):
        """``2 R sin(d / 2R)`` for a great-circle distance ``d``."""
        lats, lons = [0.0, 10.0, -45.0, 80.0], [0.0, 100.0, -170.0, 30.0]
        great = haversine_matrix(lats, lons)
        np.testing.assert_allclose(
            chordal_matrix(lats, lons),
            2 * 6371.0 * np.sin(great / (2 * 6371.0)),
            rtol=1e-12,
            atol=1e-9,
        )

    @pytest.mark.parametrize("nu", [2.5, None], ids=["matern", "gaussian"])
    def test_chordal_keeps_smooth_kernels_positive_definite(self, nu):
        """Points around the globe: positive-definite with the chordal distance."""
        rng = np.random.default_rng(0)
        c = _coords(lat=rng.uniform(-80, 80, 40), lon=rng.uniform(-180, 180, 40))
        levels = ["lat", "lon"]
        model = (
            Gaussian(levels, 3000.0, metric="chordal")
            if nu is None
            else Matern(levels, 3000.0, nu, metric="chordal")
        )
        assert np.linalg.eigvalsh(model.correlation(c)).min() > 0

    def test_constant(self):
        """One at any distance."""
        np.testing.assert_array_equal(Constant().correlation(X), np.ones((5, 5)))


# ---------------------------------------------------------------------------
# within
# ---------------------------------------------------------------------------


class TestWithin:
    """``within`` keeps a kernel inside groups of labels: zero between them."""

    def test_level(self):
        """Labels of different sites are independent."""
        c = _coords(site=["A", "A", "B", "B", "A"], x=[0.0, 1.0, 0.0, 1.0, 2.0])
        C = Exponential("x", 1.0, within="site").correlation(c)
        same = c["site"].to_numpy()[:, None] == c["site"].to_numpy()[None, :]
        np.testing.assert_allclose(
            C, np.where(same, Exponential("x", 1.0).correlation(c), 0.0), rtol=1e-15
        )

    def test_date_cuts_at_midnight(self):
        """``"t.date"``: correlated within a day, not across midnight."""
        C = Exponential("t", "2h", within="t.date").correlation(_coords(t=HOURS))
        day = np.asarray(HOURS.date)
        assert (C[day[:, None] != day[None, :]] == 0).all()
        assert C[3, 4] == 0 and C[4, 5] > 0  # 23:00 and 00:00 are apart

    def test_hour_of_day_mask(self):
        """``"t.hour"``: only the same hour of day correlates (Kunik's mask)."""
        t = pd.date_range("2024-01-01", periods=72, freq="6h")
        C = Exponential("t", "5D", within="t.hour").correlation(_coords(t=t))
        days = ((t - t[0]) / pd.Timedelta("1D")).to_numpy()
        expected = np.exp(-np.abs(days[:, None] - days[None, :]) / 5.0)
        hour = t.hour.to_numpy()
        expected[hour[:, None] != hour[None, :]] = 0.0
        np.testing.assert_allclose(C, expected, rtol=1e-14, atol=0)

    def test_constant_within_groups(self):
        """``Constant(within=...)``: one inside a group, zero between groups."""
        c = _coords(site=["A", "A", "B"], t=HOURS[:3])
        np.testing.assert_array_equal(
            Constant(within=["site", "t.date"]).correlation(c),
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

    def test_form(self):
        """Many groups: sparse; a few large ones: dense; one label each: diagonal."""
        c = _coords(site=np.repeat(np.arange(20), 3), x=np.tile([0.0, 1.0, 2.0], 20))
        assert isinstance(Exponential("x", 1.0, within="site")._form(c), SparseOperator)
        assert isinstance(Exponential("x", 1.0)._form(c), DenseOperator)
        two = _coords(site=[0, 0, 1, 1], x=[0.0, 1.0, 0.0, 1.0])
        assert isinstance(
            Exponential("x", 1.0, within="site")._form(two), DenseOperator
        )
        alone = _coords(site=np.arange(10), x=np.zeros(10))
        assert isinstance(Constant(within="site")._form(alone), DiagonalOperator)

    def test_dotted_name_needs_datetimes(self):
        """An attribute of a level that does not hold datetimes raises when read."""
        with pytest.raises(ValueError, match="does not hold datetimes"):
            Exponential("x", 1.0, within="x.date").correlation(X)

    def test_missing_key_is_a_group(self):
        """Labels that all lack the key are one group, as any other value is."""
        c = _coords(site=["A", None, "A", None], x=[0.0, 1.0, 2.0, 3.0])
        C = Exponential("x", 1.0, within="site").correlation(c)
        np.testing.assert_array_equal(C == 0, [[0, 1, 0, 1], [1, 0, 1, 0]] * 2)

    @pytest.mark.parametrize("n", [0, 1])
    def test_no_label_and_one_label(self, n):
        """No labels give an empty matrix, one label its own variance."""
        c = _coords(site=["A"] * n, x=np.zeros(n))
        np.testing.assert_array_equal(
            Exponential("x", 1.0, within="site").correlation(c), np.ones((n, n))
        )


def _shuffled_times(tz) -> pd.DatetimeIndex:
    """Times 37 minutes apart in a time zone, in a random order."""
    t = pd.date_range("2024-03-01", periods=300, freq="37min", tz=tz)
    return t[np.random.default_rng(1).permutation(len(t))]


def _within_data() -> pd.DataFrame:
    """Coordinates of 300 labels on every kind of level, three sites over eight days."""
    rng = np.random.default_rng(0)
    n = 300
    return _coords(
        site=rng.choice(["A", "B", "C"], size=n),
        t=_shuffled_times(None),
        utc=_shuffled_times("UTC"),
        denver=_shuffled_times("America/Denver"),
        lag=pd.to_timedelta(rng.integers(0, 10**5, n), unit="s"),
        x=rng.uniform(0, 10, n),
        y=rng.uniform(0, 10, n),
        lat=rng.uniform(40, 41, n),
        lon=rng.uniform(-112, -111, n),
    )


WITHIN_COORDS = _within_data()


def _within_cases():
    """Yield each kernel with ``within`` (site and day) and the same one without."""
    by = ["site", "t.date"]
    yield pytest.param(Constant(within=by), Constant(), id="Constant")
    for levels, scale, extra in [
        ("t", "2.8h", {}),
        ("utc", "2.8h", {}),
        ("denver", "2.8h", {}),
        ("lag", "3h", {}),
        ("x", 2.0, {}),
        (["x", "y"], 2.0, {}),
        (["lat", "lon"], 20.0, {"metric": "haversine"}),
        (["lat", "lon"], 20.0, {"metric": "chordal"}),
    ]:
        for cls, args in [(Exponential, ()), (Matern, (1.5,)), (Gaussian, ())]:
            if cls is not Exponential and extra.get("metric") == "haversine":
                continue  # not positive-definite, so refused
            name = f"{cls.__name__}-{'-'.join(np.atleast_1d(levels))}-{extra.get('metric', '')}"
            yield pytest.param(
                cls(levels, scale, *args, within=by, **extra),
                cls(levels, scale, *args, **extra),
                id=name,
            )


class TestWithinBlocks:
    """The ``within`` matrix is, bit for bit, the kernel on each group alone."""

    @pytest.mark.parametrize(("grouped", "plain"), list(_within_cases()))
    def test_each_group_alone(self, grouped, plain):
        """Every block is the plain kernel on that group's rows, the rest zero."""
        c = WITHIN_COORDS
        days = c["t"].dt.date
        groups = c.groupby([c["site"], days]).indices.values()
        expected = np.zeros((len(c), len(c)))
        for members in groups:
            expected[np.ix_(members, members)] = plain.correlation(c.iloc[members])
        np.testing.assert_array_equal(grouped.correlation(c), expected)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Exponential("x", 0.0), "positive"),
        (lambda: Exponential("x", -1.0), "positive"),
        (lambda: Exponential("x", np.inf), "positive"),
        (lambda: Exponential("t", "-1D"), "positive"),
        (lambda: Exponential([], 1.0), "at least one level"),
        (lambda: Exponential(["x", "x"], 1.0), "twice"),
        (lambda: Exponential(["x", 3], 1.0), "level names"),
        (lambda: Exponential("x", 1.0, metric="manhattan"), "metric must be"),
        (lambda: Exponential("lat", 1.0, metric="haversine"), "two levels"),
        (lambda: Exponential(["lat", "lon"], "1D", metric="chordal"), "not a time"),
        (lambda: Gaussian(["lat", "lon"], 1.0, metric="haversine"), "chordal"),
        (lambda: Matern(["lat", "lon"], 1.0, 1.5, metric="haversine"), "chordal"),
        (lambda: Matern("x", 1.0, 0.0), "nu must be"),
        (lambda: Matern("x", 1.0, np.inf), "nu must be"),
        (lambda: Matern("x", 1.0, 51.0), "nu must be"),
        (lambda: Exponential("t", "1D", within="t.normalize"), "dotted name"),
        (lambda: Exponential("t", "1D", within="t.nope"), "dotted name"),
        (lambda: Constant(within=["site", "site"]), "twice"),
    ],
    ids=[
        "zero scale",
        "negative scale",
        "infinite scale",
        "negative time",
        "no level",
        "level twice",
        "level not text",
        "unknown metric",
        "one level on the earth",
        "time on the earth",
        "gaussian great circle",
        "matern great circle",
        "nu zero",
        "nu infinite",
        "nu above 50",
        "method attribute",
        "unknown attribute",
        "within twice",
    ],
)
def test_bad_parameters_raise(make, match):
    """Each parameter is checked when the model is made."""
    with pytest.raises(ValueError, match=match):
        make()


def test_matern_of_half_on_the_great_circle():
    """Nu <= 1/2 is positive-definite with the great-circle distance: allowed."""
    model = Matern(["lat", "lon"], 10.0, 0.5, metric="haversine")
    assert model.nu == 0.5 and model.metric == "haversine"


@pytest.mark.parametrize(
    ("model", "coords", "match"),
    [
        (Exponential("t", 1.0), _coords(t=HOURS), "scale that is a time"),
        (Exponential("x", "1D"), X, "do not hold datetimes or timedeltas"),
        (
            Exponential("t", 1.0),
            _coords(t=pd.to_timedelta([0, 1], unit="h")),
            "holds timedeltas",
        ),
        (Exponential(["t", "u"], "1D"), _coords(t=HOURS, u=HOURS), "one level"),
    ],
)
def test_scale_must_fit_the_level(model, coords, match):
    """A time scale goes with one datetime level, a number with numeric levels."""
    with pytest.raises(ValueError, match=match):
        model.correlation(coords)


def test_scale_given_as_text_or_timedelta():
    """``"1D"``, ``pd.Timedelta`` and ``np.timedelta64`` are one scale."""
    a = Exponential("t", "1D")
    assert a == Exponential("t", pd.Timedelta("24h"))
    assert a == Exponential("t", np.timedelta64(1, "D"))
    assert hash(a) == hash(Exponential("t", "24h"))
    assert Exponential("x", 2) == Exponential("x", 2.0)


@pytest.mark.parametrize(
    "model",
    [
        Exponential("x", 2.0),
        Constant(),
        Exponential("x", 2.0, within="x"),
        Independent(),
    ],
    ids=["dense", "constant", "within", "independent"],
)
def test_correlation_is_a_new_writable_array(model):
    """``correlation`` returns a matrix of its own, not the read-only array of the form it is built in."""
    first = model.correlation(X)
    assert first.flags.writeable
    expected = first.copy()
    first[0, 0] = 99.0
    np.testing.assert_array_equal(model.correlation(X), expected)


def test_models_are_frozen_and_pickle():
    """A model cannot be changed, and pickles as it is."""
    model = Kronecker(
        Exponential("t", "1D", within="t.hour"), Matern(["lat", "lon"], 5.0, 2.5)
    )
    with pytest.raises(AttributeError):
        model.factors = ()  # pyrefly: ignore[read-only]
    assert pickle.loads(pickle.dumps(model)) == model
    assert "within=('t.hour',)" in repr(model)


# ---------------------------------------------------------------------------
# Independent, Kronecker and a new model
# ---------------------------------------------------------------------------


def test_independent_is_by_label():
    """Two labels at the same coordinates are still independent (a nugget by index)."""
    c = _coords(x=[0.0, 0.0, 1.0])
    np.testing.assert_array_equal(Independent().correlation(c), np.eye(3))
    form = Independent()._form(c)
    assert isinstance(form, DiagonalOperator)
    np.testing.assert_array_equal(form.values, np.ones(3))


class TestKronecker:
    """The product of models on separate levels, kept as factors."""

    def test_product_of_the_factors(self):
        """Each entry is the product of the factors' correlations of that pair."""
        full = pd.MultiIndex.from_product(
            [HOURS[:3], [0.0, 1.0], ["A", "B"]], names=["t", "x", "site"]
        ).to_frame(index=False)
        c = full.iloc[np.random.default_rng(1).permutation(len(full))[:10]]
        model = Kronecker(Exponential("t", "2h"), Exponential("x", 1.0, within="site"))
        expected = model.factors[0].correlation(c) * model.factors[1].correlation(c)
        np.testing.assert_allclose(model.correlation(c), expected, rtol=1e-15)
        assert isinstance(model._form(c), KroneckerOperator)
        assert model.levels == ("t", "x", "site")

    def test_hour_of_day_mask_as_a_factor(self):
        """The time factor masked by hour of day, times a spatial factor."""
        t = pd.date_range("2024-01-01", periods=8, freq="12h")
        index = pd.MultiIndex.from_product([t, [0.0, 1.0]], names=["time", "x"])
        time = Exponential("time", "2D", within="time.hour")
        S = (
            Kronecker(time, Exponential("x", 1.0))
            .with_std(1.0)
            .build(Space({"flux": index}))
        )
        assert isinstance(S.data, KroneckerOperator)
        c = index.to_frame(index=False)
        expected = time.correlation(c) * Exponential("x", 1.0).correlation(c)
        np.testing.assert_allclose(S.dense(), expected, rtol=1e-15)

    def test_nested_factors_are_flattened(self):
        """A Kronecker factor brings its own factors."""
        a, b, c = Exponential("t", "1D"), Exponential("x", 1.0), Constant(within="s")
        assert Kronecker(Kronecker(a, b), c).factors == (a, b, c)

    @pytest.mark.parametrize(
        ("factors", "error", "match"),
        [
            ((Exponential("t", "1D"),), ValueError, "at least two"),
            (
                (Exponential("t", "1D"), Exponential("t", "2D")),
                ValueError,
                "two factors",
            ),
            (
                (Exponential("t", "1D"), Exponential("x", 1.0, within="t.date")),
                ValueError,
                "two factors",
            ),
            ((Exponential("t", "1D"), Independent()), ValueError, "reads none"),
            ((Exponential("t", "1D"), Constant()), ValueError, "reads none"),
            ((Exponential("t", "1D"), "x"), TypeError, "correlation models"),
        ],
    )
    def test_refused(self, factors, error, match):
        """Two or more factors, each reading its own levels."""
        with pytest.raises(error, match=match):
            Kronecker(*factors)


@dataclass(frozen=True)
class _Steps(CorrelationModel):
    """A new model: 1/2 between neighbors on a numeric level, 0 further."""

    levels: tuple[str, ...] = ("x",)

    def correlation(self, coords):
        """Return the step correlation."""
        x = coords["x"].to_numpy()
        d = np.abs(x[:, None] - x[None, :])
        return np.where(d == 0, 1.0, np.where(d == 1, 0.5, 0.0))


def test_a_new_model_is_a_subclass():
    """A subclass with ``levels`` and ``correlation`` scales and multiplies like the rest."""
    index = pd.MultiIndex.from_product([HOURS[:2], [0, 1, 2]], names=["t", "x"])
    S = (
        Kronecker(Exponential("t", "1h"), _Steps())
        .with_std(2.0)
        .build(Space({"flux": index}))
    )
    assert isinstance(S.data, KroneckerOperator)
    time = np.array([[1.0, np.exp(-1.0)], [np.exp(-1.0), 1.0]])
    steps = _Steps().correlation(pd.DataFrame({"x": [0, 1, 2]}))
    np.testing.assert_allclose(S.dense(), 4.0 * np.kron(time, steps), rtol=1e-15)
