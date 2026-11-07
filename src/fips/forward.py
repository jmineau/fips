r"""
Forward models: a map from the state to the observations that is not linear.

A :class:`~fips.Problem` takes its forward map as an :class:`~fips.Operator`,
the matrix :math:`H` of a linear map, or as a :class:`ForwardModel`, a class
the caller writes with two methods: the model itself, :math:`F(x)`, and its
Jacobian, :math:`J(x) = \partial F / \partial x` at a state, as an
:class:`~fips.Operator`. The observations are then :math:`z = F(x) + c + e`.
fips builds no Jacobian: it calls the caller's. :meth:`Problem.linearized
<fips.Problem.linearized>` is the linear problem at a state, and
``fips.solve(problem, "gauss_newton")`` finds the mode of the posterior by
solving it again at each estimate. :func:`taylortest` checks a model's
Jacobian against the model.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from fips.operator import Operator
from fips.space import Space
from fips.vector import Vector

logger = logging.getLogger(__name__)


class ForwardModel(ABC):
    r"""
    A forward model: the map :math:`F` from the state to the observations, and its Jacobian.

    Subclass it and write two methods, then pass an instance to
    :class:`~fips.Problem` as ``forward=``:

    - ``__call__(x)``: the observations the state `x` models, without the
      offset, :math:`F(x)`, a :class:`~fips.Vector` on :attr:`rows`;
    - ``jacobian(x)``: the derivatives of :math:`F` at `x`, :math:`J_{ij} =
      \partial F_i / \partial x_j`, an :class:`~fips.Operator` with
      :attr:`rows` as its rows and :attr:`cols` as its columns (in any
      order, and in any form: dense, sparse, the product of a matrix and a
      diagonal).

    Call ``super().__init__(rows, cols)`` with the two spaces in the
    subclass's ``__init__``. fips calls both methods with a vector on
    :attr:`cols` itself, so ``self.H @ x`` with ``self.H`` on the same columns
    needs no reordering. A model with a linear map is an
    :class:`~fips.Operator`, and needs no class.

    Parameters
    ----------
    rows : Space
        The labels of the observations the model makes: those of the
        problem's observations, in any order.
    cols : Space
        The labels of the state it takes: those of the problem's prior, in
        any order.

    Raises
    ------
    TypeError
        If `rows` or `cols` is not a Space.

    See Also
    --------
    taylortest : Check the Jacobian against the model.
    fips.Problem.linearized : The linear problem at a state.

    Notes
    -----
    A problem with a model pickles when its model does: a class defined at
    the top level of a module, holding data that pickles (an
    :class:`~fips.Operator`, arrays). A class defined inside a function, or
    one holding an open file, does not, and ``pickle`` says which.

    Examples
    --------
    The observations of the exponential of the state, :math:`F(x) = H
    e^{x}`, whose Jacobian is :math:`H \operatorname{diag}(e^{x})`:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> from fips.linalg import DiagonalOperator
    >>> class Exp(fips.ForwardModel):
    ...     def __init__(self, H):
    ...         super().__init__(H.rows, H.cols)
    ...         self.H = H
    ...
    ...     def __call__(self, x):
    ...         return self.H @ np.exp(x)
    ...
    ...     def jacobian(self, x):
    ...         scale = fips.Operator(
    ...             DiagonalOperator(np.exp(x.values)), x.space, x.space
    ...         )
    ...         return self.H @ scale
    >>> state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
    >>> observed = fips.Space({"co2": pd.RangeIndex(3, name="obs_id")})
    >>> H = fips.Operator(
    ...     np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), observed, state
    ... )
    >>> model = Exp(H)
    >>> model(fips.Vector(np.zeros(2), state)).values
    array([1., 1., 2.])
    >>> fips.taylortest(model, fips.Vector(np.zeros(2), state))
    True
    """

    def __init__(self, rows: Space, cols: Space):
        for space, what in ((rows, "rows"), (cols, "cols")):
            if not isinstance(space, Space):
                raise TypeError(f"{what} must be a Space, got {type(space).__name__}.")
        self._rows = rows
        self._cols = cols

    @property
    def rows(self) -> Space:
        """The labels of the observations the model makes."""
        return self._rows

    @property
    def cols(self) -> Space:
        """The labels of the state the model takes."""
        return self._cols

    @abstractmethod
    def __call__(self, x: Vector) -> Vector:
        r"""
        Return the observations the state `x` models, without the offset: :math:`F(x)`.

        Parameters
        ----------
        x : Vector
            A state on :attr:`cols`.

        Returns
        -------
        Vector
            :math:`F(x)`, on :attr:`rows` (its labels in any order).
        """

    @abstractmethod
    def jacobian(self, x: Vector) -> Operator:
        r"""
        Return the Jacobian of the model at `x`: :math:`J_{ij} = \partial F_i / \partial x_j`.

        Parameters
        ----------
        x : Vector
            A state on :attr:`cols`.

        Returns
        -------
        Operator
            :math:`J(x)`, its rows the labels of :attr:`rows` and its columns
            those of :attr:`cols` (in any order).
        """

    def _gradient(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""
        Return :math:`J(x)^T v`, for the state values `x` in the order of :attr:`cols` and `v` in the order of :attr:`rows`, in the order of :attr:`cols`.

        A product with the transpose of the Jacobian, for a sampler's
        gradient. This one forms the Jacobian; a model with a cheaper product
        overrides it.
        """
        return np.asarray(_differentiated(self, x).data._transpose() @ v, dtype=float)

    def _curvature(self, x: np.ndarray, v: np.ndarray) -> np.ndarray | None:
        r"""
        Return the diagonal of :math:`\sum_k v_k \nabla^2 F_k(x)` when that matrix is diagonal, else None; `x` and the result in the order of :attr:`cols`, `v` in that of :attr:`rows`.

        The second-order term of the cost's Hessian, which Gauss-Newton leaves
        out: with it, ``"gauss_newton"`` takes Newton steps. A model whose
        second derivatives are not diagonal (any model fips did not write)
        returns None, and its steps are Gauss-Newton's.
        """
        return None

    def __repr__(self) -> str:
        """Return the class and the two spaces."""
        return f"{type(self).__name__}(rows: {self._rows!r}, cols: {self._cols!r})"


class _Aggregated(ForwardModel):
    r"""
    A model followed by an aggregation of its observations, :math:`W F(x)`; its Jacobian is :math:`W J(x)`.

    :meth:`Problem.aggregate <fips.Problem.aggregate>` makes one from a
    problem whose forward is a model. It pickles when the model does.
    """

    def __init__(self, model: ForwardModel, aggregation: Operator):
        super().__init__(aggregation.rows, model.cols)
        self.model = model
        """The model whose observations are aggregated."""
        self.aggregation = aggregation
        """The aggregation :math:`W`, its columns on the model's rows."""

    def __call__(self, x: Vector) -> Vector:
        r"""Return :math:`W F(x)`."""
        modeled = self.model(x)
        W = self.aggregation
        form = W._on(W.rows, modeled.space, "the aggregation")
        return Vector(form @ modeled.values, W.rows)

    def jacobian(self, x: Vector) -> Operator:
        r"""Return :math:`W J(x)`."""
        J = self.model.jacobian(x)
        W = self.aggregation
        form = W._on(W.rows, J.rows, "the aggregation")
        return Operator(form @ J.data, W.rows, J.cols, name=J.name)

    def _curvature(self, x: np.ndarray, v: np.ndarray) -> np.ndarray | None:
        r"""Return the model's curvature with the weights :math:`W^T v`: :math:`\sum_k v_k \nabla^2 (W F)_k = \sum_j (W^T v)_j \nabla^2 F_j`."""
        W = self.aggregation
        form = W._on(W.rows, self.model.rows, "the aggregation")
        return self.model._curvature(x, np.asarray(form._transpose() @ v, dtype=float))

    def __repr__(self) -> str:
        """Return the model and the aggregation's rows."""
        return f"_Aggregated({self.model!r}, rows: {self._rows!r})"


def _evaluated(model: ForwardModel, values: np.ndarray) -> np.ndarray:
    """
    Return the model's observations of the state `values` (in the order of its columns), in the order of its rows.

    Raises
    ------
    TypeError
        If the model does not return a Vector.
    ValueError
        If the vector it returns does not have exactly the labels of its rows.
    """
    modeled = model(Vector(values, model.cols))
    if not isinstance(modeled, Vector):
        raise TypeError(
            f"{type(model).__name__}.__call__ must return a Vector on its rows, "
            f"got {type(modeled).__name__}."
        )
    if modeled.space is model.rows:
        return modeled.values
    positions = model.rows._align(
        modeled.space,
        "what the model returns",
        f"What {type(model).__name__}.__call__ returns and its rows",
    )
    return modeled.values[positions]


def _differentiated(model: ForwardModel, values: np.ndarray) -> Operator:
    """
    Return the model's Jacobian at the state `values` (in the order of its columns), on its rows and columns.

    Raises
    ------
    TypeError
        If the model's ``jacobian`` does not return an Operator.
    ValueError
        If that operator does not have exactly the labels of the model's rows
        and columns.
    """
    J = model.jacobian(Vector(values, model.cols))
    if not isinstance(J, Operator):
        raise TypeError(
            f"{type(model).__name__}.jacobian must return an Operator, got "
            f"{type(J).__name__}."
        )
    form = J._on(
        model.rows,
        model.cols,
        f"the Jacobian of {type(model).__name__}",
        ("the model's rows", "the model's columns"),
    )
    return Operator(form, model.rows, model.cols, name=J.name)


_STEPS = 30
"""The number of halvings of the step :func:`taylortest` takes, from 1/2 to about 1e-9."""

_ORDER = 1.8
"""The order of the remainder :func:`taylortest` asks for at its smallest steps: 2 for a right Jacobian, 1 for a wrong one."""


def taylortest(
    model: ForwardModel, x: Vector, dx: Vector | None = None, seed: int | None = 0
) -> bool:
    r"""
    Check that a model's Jacobian is the derivative of the model, by a Taylor test.

    For a step :math:`\epsilon` along a direction :math:`\delta x`, the
    remainder :math:`r(\epsilon) = \lVert F(x + \epsilon\, \delta x) - F(x) -
    \epsilon J(x)\, \delta x \rVert` falls as :math:`\epsilon^2` when
    :math:`J` is the Jacobian of :math:`F`, and only as :math:`\epsilon` when
    it is not (a sign flipped, a term missing). The test halves
    :math:`\epsilon` from 1/2 to about :math:`10^{-9}` and passes when the
    remainder falls by at least :math:`2^{1.8}` per halving at its three
    smallest steps above roundoff (:math:`10^3` times the machine epsilon of
    :math:`\lVert F \rVert`). A remainder at roundoff from the start (a model
    whose Jacobian is exact, such as a linear one) passes. The orders are
    logged at DEBUG.

    Parameters
    ----------
    model : ForwardModel
        The model to check.
    x : Vector
        The state to check it at, on the model's columns (in any order): the
        prior, an estimate.
    dx : Vector, optional
        The direction of the steps, on the model's columns. By default normal
        draws, each scaled by its element's size in `x` (where an element is
        zero, by the mean size of `x`; 1 if `x` is zero).
    seed : int, optional
        The seed of the default direction.

    Returns
    -------
    bool
        True if the remainder falls as the square of the step.

    Raises
    ------
    TypeError
        If `model` is not a ForwardModel, or it returns other than a Vector
        and an Operator.
    ValueError
        If `x` or `dx` does not have exactly the model's columns' labels, or
        what the model returns does not have exactly its labels.

    See Also
    --------
    fips.linalg.dottest : Check that an operator's transpose is its adjoint.

    Notes
    -----
    The test checks the Jacobian in one direction. A term wrong in some
    elements only shows when the direction moves them, which a random one
    does. It cannot see an error of the Jacobian below about the square root
    of the roundoff of the model (a relative error near :math:`10^{-6}`).

    Examples
    --------
    :math:`F(x) = x^2`, with a right and a wrong Jacobian:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> from fips.linalg import DiagonalOperator
    >>> class Square(fips.ForwardModel):
    ...     def __init__(self, space, factor=2.0):
    ...         super().__init__(space, space)
    ...         self.factor = factor
    ...
    ...     def __call__(self, x):
    ...         return x * x
    ...
    ...     def jacobian(self, x):
    ...         form = DiagonalOperator(self.factor * x.values)
    ...         return fips.Operator(form, x.space, x.space)
    >>> space = fips.Space({"s": pd.RangeIndex(3, name="i")})
    >>> x = fips.Vector([1.0, -2.0, 0.5], space)
    >>> fips.taylortest(Square(space), x)
    True
    >>> fips.taylortest(Square(space, factor=1.0), x)
    False
    """
    if not isinstance(model, ForwardModel):
        raise TypeError(
            f"model must be a ForwardModel, got {type(model).__name__}; an "
            "Operator's transpose is checked with fips.linalg.dottest."
        )
    cols = model.cols

    def on_cols(v: Vector, what: str) -> np.ndarray:
        """Return the values of `v` in the order of the model's columns."""
        if not isinstance(v, Vector):
            raise TypeError(f"{what} must be a Vector, got {type(v).__name__}.")
        if v.space is cols:
            return v.values
        return v.values[cols._align(v.space, what, f"{what} and the model's columns")]

    point = on_cols(x, "x")
    if dx is None:
        size = np.abs(point)
        size[size == 0] = size.mean() if size.any() else 1.0
        direction = np.random.default_rng(seed).standard_normal(len(point)) * size
    else:
        direction = on_cols(dx, "dx")
    F0 = _evaluated(model, point)
    Jdx = np.asarray(_differentiated(model, point).data @ direction, dtype=float)
    eps = np.finfo(float).eps
    steps = 0.5 ** np.arange(1, _STEPS + 1)
    remainders, floors = [], []
    for step in steps:
        F = _evaluated(model, point + step * direction)
        remainders.append(float(np.linalg.norm(F - F0 - step * Jdx)))
        floors.append(1e3 * eps * float(np.linalg.norm(F) + np.linalg.norm(F0)))
    r, floor = np.array(remainders), np.array(floors)
    above = np.isfinite(r) & (r > floor)
    pairs = np.flatnonzero(above[:-1] & above[1:])
    if not len(pairs):
        exact = bool(np.all(r[np.isfinite(r)] <= floor[np.isfinite(r)]))
        logger.debug("Taylor test: the remainder is at roundoff (%s).", exact)
        return exact
    orders = np.log2(r[pairs] / r[pairs + 1])
    logger.debug(
        "Taylor test: the remainder's order at each halving of the step, from the "
        "largest step: %s.",
        np.array2string(orders, precision=2),
    )
    return bool(np.all(orders[-3:] > _ORDER))
