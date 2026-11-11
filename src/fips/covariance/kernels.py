"""
Kernels: correlations of the labels' coordinates.

:class:`Exponential`, :class:`Gaussian` and :class:`Matern` fall from one
with the distance along some levels (a time, a line, a distance on the
Earth); :class:`Constant` is one at any distance. ``within=`` keeps any of
them inside groups of labels (the same site and day, say): zero between groups.

A smooth kernel on close points is nearly singular: its smallest eigenvalues
are roundoff, and a solve with it as the observation covariance keeps few digits.
``rtol=`` raises its smallest eigenvalues so that the condition number stays
near ``1 / rtol``.
"""

import logging
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.special import gammaln, kve

from fips.covariance.distances import (
    _chord,
    _great_circle,
    _half_angle_pairs,
    _half_angle_term,
)
from fips.covariance.models import CorrelationModel, _names
from fips.linalg import DenseOperator, LinearOperator
from fips.linalg.forms import _DENSE_FILL, _adopt

logger = logging.getLogger(__name__)

_METRICS = ("euclidean", "haversine", "chordal")
"""The distances a kernel can measure: a straight line, and two on the Earth (km)."""

_LARGE_SHRINK = 0.01
"""The fraction by which ``rtol`` may shrink a correlation without a warning."""


def _split(name: str) -> tuple[str, str | None]:
    """Return a ``within`` entry as its level and its datetime attribute (or None)."""
    level, _, attribute = name.partition(".")
    return level, (attribute or None)


def _key(coords: pd.DataFrame, name: str) -> np.ndarray:
    """Return each row's label for one ``within`` entry."""
    level, attribute = _split(name)
    values = coords[level]
    if attribute is None:
        return values.to_numpy()
    if not pd.api.types.is_datetime64_any_dtype(values):
        raise ValueError(
            f"within={name!r} reads the {attribute!r} of level {level!r}, "
            f"which does not hold datetimes ({values.dtype})."
        )
    return np.asarray(getattr(pd.DatetimeIndex(values), attribute))


def _time_kind(values: pd.Series) -> str | None:
    """Return ``"timedeltas"`` or ``"datetimes"`` for a level that holds times, else None."""
    if pd.api.types.is_timedelta64_dtype(values):
        return "timedeltas"
    if pd.api.types.is_datetime64_any_dtype(values):
        return "datetimes"
    return None


def _time_error(level: str, kind: str) -> ValueError:
    """Return the error for a level of times that is not measured alone with a scale that is a time."""
    return ValueError(
        f"Level {level!r} holds {kind}: a model measures time along that one "
        "level, with a scale that is a time (as '30D'), and takes no other "
        "level or metric with it."
    )


def _ends(values: np.ndarray, rows: np.ndarray | None, cols: np.ndarray | None):
    """Return the two sides of the differences between rows: every two rows (as a column and a row), or the pairs `rows` and `cols` name."""
    if rows is None or cols is None:
        return values[:, None], values[None, :]
    return values[rows], values[cols]


def _within_pairs(
    coords: pd.DataFrame, within: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the positions of every pair of rows in the same group, as rows and columns, and the size of each group.

    A group is the rows that agree on every `within` entry. The pairs come group
    by group (groups in order of first appearance, the order of the sizes), and
    in each group row by row: the entries of one dense block of the group's
    members, in order.
    """
    keys = pd.DataFrame({name: _key(coords, name) for name in within})
    groups = keys.groupby(list(within), sort=False, dropna=False).ngroup().to_numpy()
    order = np.argsort(groups, kind="stable")  # the members of each group, together
    sizes = np.bincount(groups)
    size = np.repeat(sizes, sizes)  # the size of each member's group
    begin = np.repeat(
        np.cumsum(sizes) - sizes, sizes
    )  # where its group begins in `order`
    start = np.cumsum(size) - size  # where each member's run of pairs begins
    rows = np.repeat(order, size)  # each member, once for each member of its group
    # in each run, the group's members in order: the position within the run, from the group's begin
    cols = order[np.arange(len(rows)) - np.repeat(start - begin, size)]
    return rows, cols, sizes


def _groups(data: np.ndarray, sizes: np.ndarray) -> list[np.ndarray]:
    """Return each group's dense block of the pair values `data`, in the order :func:`_within_pairs` gives them."""
    if not len(sizes):  # no labels
        return []
    ends = np.cumsum(sizes * sizes)[:-1]
    return [
        block.reshape(size, size)
        for block, size in zip(np.split(data, ends), sizes, strict=True)
    ]


def _floored(C: np.ndarray, rtol: float) -> tuple[np.ndarray, int, float]:
    """
    Return the correlation `C` with its eigenvalues below `rtol` times the largest raised to that, how many were, and the shrink.

    The matrix is rebuilt from its eigendecomposition and scaled back to ones
    on the diagonal, so it stays a correlation (the scaling moves each
    eigenvalue by a factor of at most ``1 + rtol * max(w)``). The shrink is
    the largest fraction by which that scaling reduces an entry of the
    rebuilt matrix. A `C` with no eigenvalue below the floor is returned as
    it is.
    """
    w, V = (np.asarray(a) for a in eigh(C))  # SciPy is not annotated
    floor = rtol * w[-1]
    raised = int((w < floor).sum())
    if not raised:
        return C, 0, 0.0
    A = (V * np.maximum(w, floor)) @ V.T
    diagonal = np.diag(A)
    shrink = float(1.0 - 1.0 / diagonal.max())
    d = 1.0 / np.sqrt(diagonal)
    A = d[:, None] * A * d[None, :]
    A = 0.5 * (A + A.T)  # exactly symmetric
    np.fill_diagonal(A, 1.0)
    return A, raised, shrink


@dataclass(frozen=True, init=False)
class _Kernel(CorrelationModel):
    """A correlation that is a function of the labels' coordinates, within groups of labels."""

    within: tuple[str, ...]
    """The levels on which two labels must be the same to be correlated (``"time.hour"`` is an attribute of a datetime level)."""
    rtol: float | None
    """Eigenvalues of the correlation below this times the largest are raised to it; None keeps the correlation as the kernel gives it."""

    def _set_rtol(self, rtol: float | None) -> None:
        """Check and set :attr:`rtol`."""
        if rtol is not None:
            value = float(rtol)
            if not 0 < value < 1:
                raise ValueError(f"rtol must be in (0, 1), got {rtol!r}.")
            rtol = value
        object.__setattr__(self, "rtol", rtol)

    def _set_within(self, within: str | Sequence[str] | None) -> None:
        """Check and set :attr:`within`."""
        names = _names(within, "within")
        for name in names:
            level, attribute = _split(name)
            if attribute is None:
                continue
            value = getattr(pd.DatetimeIndex([]), attribute, None)
            if not level or value is None or callable(value):
                raise ValueError(
                    f"within={name!r}: a dotted name is a level and an attribute "
                    "of its datetimes, as 'obs_time.date' or 'time.hour'."
                )
        object.__setattr__(self, "within", names)

    def _reads(self) -> tuple[str, ...]:
        """Return :attr:`levels` and the levels of :attr:`within`."""
        grouped = (_split(name)[0] for name in self.within)
        return tuple(dict.fromkeys((*self.levels, *grouped)))

    @abstractmethod
    def _unmasked(self, coords: pd.DataFrame) -> np.ndarray:
        """Return the correlation between every two rows of `coords`, ignoring ``within``."""

    @abstractmethod
    def _paired(
        self, coords: pd.DataFrame, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        """Return the correlation between row ``rows[p]`` and row ``cols[p]`` of `coords` for each p, ignoring ``within``."""

    def correlation(self, coords: pd.DataFrame) -> np.ndarray:
        """Return the correlation between the rows of `coords`, zero between groups (see :meth:`CorrelationModel.correlation`)."""
        return np.array(self._form(coords))

    def _conditioned(self, blocks: list[np.ndarray]) -> list[np.ndarray]:
        """
        Return the correlation's blocks (the whole matrix, or each group's) with :attr:`rtol` applied.

        Each block's eigenvalues below :attr:`rtol` times the largest are
        raised to that. One INFO line says how many; a warning says when the
        rescaling shrank the correlations by more than ``_LARGE_SHRINK``.
        """
        rtol = self.rtol
        if rtol is None:
            return blocks
        out, raised, shrink = [], 0, 0.0
        for C in blocks:
            if len(C) > 1:
                C, count, by = _floored(C, rtol)
                raised += count
                shrink = max(shrink, by)
            out.append(C)
        if raised:
            logger.info(
                "%r raised %d of the %d eigenvalues of its correlation to "
                "rtol times the largest.",
                self,
                raised,
                sum(len(C) for C in blocks),
            )
        if shrink > _LARGE_SHRINK:
            logger.warning(
                "rtol=%g is large for the correlation of %r: scaling it back to "
                "ones on the diagonal shrank its correlations by up to %.0f%%, "
                "which changes the model. A small rtol, such as 1e-8, keeps it "
                "positive-definite without that.",
                rtol,
                self,
                100 * shrink,
            )
        return out

    def _form(self, coords: pd.DataFrame) -> LinearOperator:
        """Return the correlation, conditioned (see :meth:`_conditioned`); with ``within``, sparse, with every pair of a group built at once."""
        if not self.within:
            (matrix,) = self._conditioned([self._unmasked(coords)])
            return DenseOperator._adopt(matrix)
        rows, cols, sizes = _within_pairs(coords, self.within)
        n = len(coords)
        data = self._paired(coords, rows, cols)
        if self.rtol is not None and len(sizes):
            blocks = self._conditioned(_groups(data, sizes))
            data = np.concatenate([C.ravel() for C in blocks])
        matrix = sp.csr_array((data, (rows, cols)), shape=(n, n))
        if matrix.nnz > _DENSE_FILL * n * n:
            return DenseOperator._adopt(matrix.toarray())
        return _adopt(matrix)


@dataclass(frozen=True, init=False)
class Constant(_Kernel):
    """
    A correlation of one between every two labels (of a group, with `within`).

    Parameters
    ----------
    within : str or sequence of str, optional
        Correlate only labels that are the same on these levels; labels of
        different groups are independent. ``"level.attribute"`` groups by an
        attribute of a datetime level (``"obs_time.date"``).
    rtol : float, optional
        Raise the eigenvalues of each group's correlation that are below
        `rtol` times the largest to that value, and scale it back to ones on
        the diagonal, which makes each group's rank-one block
        positive-definite (see :class:`Exponential`). By default the blocks
        are rank one, which is what the model says.

    Raises
    ------
    ValueError
        If `rtol` is not in (0, 1).

    Examples
    --------
    Errors shared by every observation of a site on one day:

    >>> import pandas as pd
    >>> from fips import Constant, Space
    >>> times = pd.to_datetime(
    ...     ["2024-01-01 10:00", "2024-01-01 14:00", "2024-01-02 10:00"]
    ... )
    >>> obs = pd.MultiIndex.from_product([["A"], times], names=["site", "obs_time"])
    >>> model = Constant(within=["site", "obs_time.date"])
    >>> model.with_std(1.0).build(Space({"co2": obs})).dense()
    array([[1., 1., 0.],
           [1., 1., 0.],
           [0., 0., 1.]])
    """

    levels: tuple[str, ...] = field(default=(), repr=False)
    """No levels, since the correlation is one at any distance."""

    def __init__(
        self,
        *,
        within: str | Sequence[str] | None = None,
        rtol: float | None = None,
    ):
        self._set_within(within)
        self._set_rtol(rtol)

    def _unmasked(self, coords: pd.DataFrame) -> np.ndarray:
        """Return a matrix of ones."""
        return np.ones((len(coords), len(coords)))

    def _paired(
        self, coords: pd.DataFrame, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        """Return a one for each pair."""
        return np.ones(len(rows))


@dataclass(frozen=True, init=False)
class _Decay(_Kernel):
    """A correlation that decays with the distance along some levels."""

    scale: "float | pd.Timedelta"
    """The length scale, a time for a datetime or timedelta level, else in the levels' units (km on the Earth)."""
    metric: str
    """How the distance is measured, ``"euclidean"``, ``"haversine"`` or ``"chordal"``."""

    def __init__(
        self,
        levels: str | Sequence[str],
        scale: "float | str | pd.Timedelta",
        *,
        within: str | Sequence[str] | None = None,
        metric: str = "euclidean",
        rtol: float | None = None,
    ):
        names = _names(levels, "levels")
        if not names:
            raise ValueError(f"{type(self).__name__} needs at least one level.")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}.")
        if metric != "euclidean" and len(names) != 2:
            raise ValueError(
                f"metric={metric!r} reads two levels, latitude then longitude; "
                f"got {list(names)}."
            )
        if metric == "haversine" and not self._valid_on_sphere():
            raise ValueError(
                f"A {type(self).__name__} correlation of the great-circle distance "
                "is not positive-definite; use metric='chordal'."
            )
        if isinstance(scale, (str, pd.Timedelta, np.timedelta64)):
            value = pd.Timedelta(scale)
            positive = value > pd.Timedelta(0)
            if metric != "euclidean":
                raise ValueError(f"metric={metric!r} takes a scale in km, not a time.")
        else:
            value = float(scale)
            positive = 0 < value < np.inf
        if not positive:
            raise ValueError(f"The scale must be positive and finite, got {scale!r}.")
        object.__setattr__(self, "levels", names)
        object.__setattr__(self, "scale", value)
        object.__setattr__(self, "metric", metric)
        self._set_within(within)
        self._set_rtol(rtol)

    def _valid_on_sphere(self) -> bool:
        """Return whether a correlation of the great-circle distance is positive-definite."""
        return True

    def _numbers(self, coords: pd.DataFrame) -> np.ndarray:
        """Return the levels of `coords` as float columns, or raise naming a level that holds no numbers."""
        columns = []
        for level in self.levels:
            if (kind := _time_kind(coords[level])) is not None:
                raise _time_error(level, kind)
            try:
                columns.append(coords[level].to_numpy(dtype=float))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Level {level!r} holds {coords[level].dtype} values, which "
                    f"{type(self).__name__} cannot measure a distance along: "
                    "a covariance model needs levels of numbers (coordinates) "
                    "or of times."
                ) from None
        return np.column_stack(columns)

    def _distance(
        self,
        coords: pd.DataFrame,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return the distance between rows, divided by the scale.

        A matrix between every two rows of `coords`; with `rows` and `cols`, a
        vector with the distance from row ``rows[p]`` to row ``cols[p]``.
        """
        if self.metric != "euclidean":
            lat, lon = self._numbers(coords).T
            if rows is None or cols is None:
                term = _half_angle_term(lat, lon)
            else:
                term = _half_angle_pairs(lat, lon, rows, cols)
            measure = _great_circle if self.metric == "haversine" else _chord
            return measure(term) / self.scale
        times = {lv: k for lv in self.levels if (k := _time_kind(coords[lv]))}
        if times:
            level, kind = next(iter(times.items()))
            if self.levels != (level,) or not isinstance(self.scale, pd.Timedelta):
                raise _time_error(level, kind)
            first = coords[level]
            if kind == "timedeltas":
                t = first.to_numpy()
            else:  # an instant is the same instant in any time zone
                instants = pd.DatetimeIndex(first)
                t = (instants.tz_convert(None) if instants.tz else instants).to_numpy()
            a, b = _ends(t, rows, cols)
            return np.abs(a - b) / self.scale.to_timedelta64()
        if isinstance(self.scale, pd.Timedelta):
            raise ValueError(
                f"The scale is a time, but the levels {list(self.levels)} do not "
                "hold datetimes or timedeltas."
            )
        x = self._numbers(coords)
        if x.shape[1] == 1:
            a, b = _ends(x[:, 0], rows, cols)
            distance = np.abs(a - b)
        else:
            a, b = _ends(x, rows, cols)
            distance = np.sqrt(((a - b) ** 2).sum(axis=-1))
        return distance / self.scale

    @abstractmethod
    def _decay(self, distance: np.ndarray) -> np.ndarray:
        """Return the correlation at a distance (divided by the scale), for any shape of array."""

    def _unmasked(self, coords: pd.DataFrame) -> np.ndarray:
        """Return the correlation between every two rows of `coords`, ignoring ``within``."""
        return self._decay(self._distance(coords))

    def _paired(
        self, coords: pd.DataFrame, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        """Return the correlation of each pair of rows, ignoring ``within``."""
        return self._decay(self._distance(coords, rows, cols))


@dataclass(frozen=True, init=False)
class Exponential(_Decay):
    r"""
    An exponential correlation, :math:`\exp(-d / \ell)` at a distance :math:`d`.

    Parameters
    ----------
    levels : str or sequence of str
        The levels the distance is measured along: one datetime or timedelta
        level (a time), numeric levels (a straight line), or latitude and
        longitude with ``metric="haversine"`` or ``"chordal"``.
    scale : float, str or pd.Timedelta
        The length scale :math:`\ell`, positive: a time (``"30D"``) for a
        datetime or timedelta level, else a number in the levels' units (km
        on the Earth).
    within : str or sequence of str, optional
        Correlate only labels that are the same on these levels (the matrix
        is then sparse). ``"level.attribute"`` is an attribute of a datetime
        level: ``"obs_time.date"`` cuts the correlation at midnight, and
        ``"time.hour"`` keeps it to the same hour of day.
    metric : {"euclidean", "haversine", "chordal"}, default "euclidean"
        How the distance is measured. ``"haversine"`` (great-circle) and
        ``"chordal"`` (through the Earth) read the levels as latitude and
        longitude in degrees, in that order, and give kilometers.
    rtol : float, optional
        Raise the eigenvalues of the correlation that are below `rtol` times
        the largest to that value, then scale it back to ones on the
        diagonal, so that its condition number is at most about
        ``1 / rtol`` (``rtol=1e-8``: about ``1e8``). The correlation is each
        group's with `within`, and the factor's (on the factor's own
        coordinates) in a :class:`~fips.Kronecker`. The scaling divides each
        eigenvalue by at most ``1 + rtol * max(w)``, with ``max(w)`` the
        largest, and reduces the correlations by about as much, so keep
        `rtol` small: a warning is logged when they shrink by more than 1%.
        The floor follows the largest eigenvalue, so the same two labels can
        get slightly different correlations among different labels. It costs
        one eigendecomposition of each such matrix. By default the
        correlation is the formula's.

    Raises
    ------
    ValueError
        If the scale is not positive, a level is named twice, the metric
        is unknown or on the Earth with other than two levels or a time, or
        `rtol` is not in (0, 1).

    Notes
    -----
    A correlation that decays smoothly, on points close together for its
    scale, is nearly singular: its smallest eigenvalues are roundoff, and a
    solve with it loses about as many digits as its condition number has (a
    Cholesky factorization fails past about ``1e16``). A prior
    covariance may be singular; an observation covariance must be
    positive-definite, and the analytic method logs a warning when the correlation of
    the observation covariance has a condition number above ``1e10``. Either of two things
    fixes it: `rtol`, in the correlation, or an independent error added to
    the covariance (``+ Independent().with_std(...)``).

    Examples
    --------
    A prior covariance with correlations over 30 days in time and 20 km in
    space, kept as the two factors:

    >>> from fips import Exponential, Kronecker
    >>> model = Kronecker(
    ...     Exponential("time", "30D"),
    ...     Exponential(["lat", "lon"], 20.0, metric="haversine"),
    ... )

    Errors correlated over 2.8 hours at one site, never across midnight:

    >>> mdm = Exponential("obs_time", "2.8h", within=["site", "obs_time.date"])
    """

    def _decay(self, distance: np.ndarray) -> np.ndarray:
        """Return ``exp(-d / scale)``."""
        return np.exp(-distance)


@dataclass(frozen=True, init=False)
class Gaussian(_Decay):
    r"""
    A Gaussian correlation, :math:`\exp(-d^2 / (2 \ell^2))` at a distance :math:`d`.

    The limit of :class:`Matern` as :math:`\nu \to \infty`: very smooth. On
    the Earth it takes ``metric="chordal"`` (with the great-circle distance it
    is not positive-definite).

    Parameters
    ----------
    levels : str or sequence of str
        The levels the distance is measured along (see :class:`Exponential`).
    scale : float, str or pd.Timedelta
        The length scale :math:`\ell`, positive.
    within : str or sequence of str, optional
        Correlate only labels the same on these levels (see :class:`Exponential`).
    metric : {"euclidean", "chordal"}, default "euclidean"
        How the distance is measured (see :class:`Exponential`).
    rtol : float, optional
        Raise the correlation's eigenvalues below `rtol` times the largest to
        that value (see :class:`Exponential`).

    Raises
    ------
    ValueError
        As :class:`Exponential`, and for ``metric="haversine"``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fips import Gaussian
    >>> Gaussian("x", 1.0).correlation(pd.DataFrame({"x": [0.0, 1.0]})).round(4)
    array([[1.    , 0.6065],
           [0.6065, 1.    ]])

    On points close together for the scale it is numerically singular, and
    `rtol` keeps its condition number near ``1 / rtol``:

    >>> import numpy as np
    >>> coords = pd.DataFrame({"x": np.arange(100.0)})
    >>> C = Gaussian("x", 10.0, rtol=1e-8).correlation(coords)
    >>> bool(np.linalg.cond(C) < 1.01e8)
    True
    """

    def _valid_on_sphere(self) -> bool:
        """Return False: not positive-definite on the great-circle distance."""
        return False

    def _decay(self, distance: np.ndarray) -> np.ndarray:
        """Return ``exp(-d**2 / (2 scale**2))``."""
        return np.exp(-0.5 * distance**2)


@dataclass(frozen=True, init=False)
class Matern(_Decay):
    r"""
    A Matérn correlation, of smoothness :math:`\nu`.

    .. math::

        C(d) = \frac{2^{1-\nu}}{\Gamma(\nu)} \, r^\nu K_\nu(r), \qquad
        r = \sqrt{2\nu} \, d / \ell,

    where :math:`K_\nu` is the modified Bessel function of the second kind
    (Rasmussen and Williams 2006, eq. 4.14). :math:`\nu = 1/2` is
    :class:`Exponential`, and :math:`\nu \to \infty` is :class:`Gaussian`.

    Parameters
    ----------
    levels : str or sequence of str
        The levels the distance is measured along (see :class:`Exponential`).
    scale : float, str or pd.Timedelta
        The length scale :math:`\ell`, positive.
    nu : float, default 1.5
        The smoothness :math:`\nu`, in (0, 50]: past that, use :class:`Gaussian`.
    within : str or sequence of str, optional
        Correlate only labels the same on these levels (see :class:`Exponential`).
    metric : {"euclidean", "haversine", "chordal"}, default "euclidean"
        How the distance is measured (see :class:`Exponential`);
        ``"haversine"`` only for :math:`\nu \le 1/2` (positive-definite there).
    rtol : float, optional
        Raise the correlation's eigenvalues below `rtol` times the largest to
        that value (see :class:`Exponential`).

    Raises
    ------
    ValueError
        As :class:`Exponential`, and if `nu` is not in (0, 50].

    Examples
    --------
    >>> import pandas as pd
    >>> from fips import Matern
    >>> coords = pd.DataFrame({"x": [0.0, 1.0]})
    >>> Matern("x", 1.0, nu=0.5).correlation(coords).round(4)
    array([[1.    , 0.3679],
           [0.3679, 1.    ]])
    """

    nu: float
    """The smoothness, 1/2 for the exponential and smoother as it grows."""

    def __init__(
        self,
        levels: str | Sequence[str],
        scale: "float | str | pd.Timedelta",
        nu: float = 1.5,
        *,
        within: str | Sequence[str] | None = None,
        metric: str = "euclidean",
        rtol: float | None = None,
    ):
        nu = float(nu)
        if not 0 < nu <= 50:
            raise ValueError(
                f"nu must be in (0, 50] (Gaussian is the limit), got {nu}."
            )
        object.__setattr__(self, "nu", nu)
        super().__init__(levels, scale, within=within, metric=metric, rtol=rtol)

    def _valid_on_sphere(self) -> bool:
        """Return whether nu <= 1/2, where the great-circle distance is allowed."""
        return self.nu <= 0.5

    def _decay(self, distance: np.ndarray) -> np.ndarray:
        """Return the Matérn correlation, through logarithms so that no term overflows."""
        r = np.sqrt(2.0 * self.nu) * distance
        out = np.ones_like(r)
        apart = r > 0
        x = r[apart]
        k = kve(self.nu, x)
        with np.errstate(divide="ignore"):  # K_nu underflows to 0 far away
            log_c = (1.0 - self.nu) * np.log(2.0) - gammaln(self.nu)
            c = np.exp(log_c + self.nu * np.log(x) + np.log(k) - x)
        # K_nu overflows only for a large nu at a tiny r, where C = 1 - r^2 / (4 (nu - 1))
        big = np.isinf(k)
        c[big] = 1.0 - x[big] ** 2 / (4.0 * (self.nu - 1.0))
        out[apart] = c
        return out
