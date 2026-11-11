r"""
Forward models: a map from the state to the observations that is not linear.

A :class:`~fips.Problem` takes its forward map (a :class:`Forward`) as an
:class:`~fips.Operator`, the matrix :math:`H` of a linear map and its own
Jacobian, or as a :class:`ForwardModel`, a class
the caller writes with two methods: the model itself, :math:`F(x)`, and its
Jacobian, :math:`J(x) = \partial F / \partial x` at a state, as an
:class:`~fips.Operator` (or, for a model with an adjoint, its products
:math:`J v` and :math:`J^T w`, :meth:`ForwardModel.jvp` and
:meth:`ForwardModel.vjp`). The observations are then :math:`z = F(x) + c + e`.
For a model the caller writes, fips builds no Jacobian: it calls the
caller's (:class:`~fips.LogNormalForward` and :class:`~fips.SoftplusForward`
bring their own). :meth:`Problem.linearized
<fips.Problem.linearized>` is the linear problem at a state, and
``fips.solve(problem, "gauss_newton")`` finds the mode of the posterior by
solving it again at each estimate (``"variational"`` does the same with
products only). :func:`taylortest` checks a model's Jacobian against the
model, and its transpose against the Jacobian.
"""

import logging
from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from fips.linalg import LinearOperator, MatrixFreeOperator, dottest
from fips.operator import Operator
from fips.space import Space
from fips.vector import Vector

logger = logging.getLogger(__name__)


class Forward(Protocol):
    r"""
    A problem's forward map, linear or not: what :attr:`Problem.forward <fips.Problem.forward>` is.

    Two kinds of object answer it. An :class:`~fips.Operator` :math:`H` is a
    forward model whose Jacobian is constant: ``H(x)`` is ``H @ x`` and
    ``H.jacobian(x)`` is ``H``. A :class:`ForwardModel` :math:`F` is one the
    caller writes. Code that reads a problem's forward through these four
    members holds for both; what needs the matrix (``@``, ``.data``,
    ``.dense()``) is an operator's own, and :class:`~fips.Problem` takes
    only these two kinds as ``forward=``.

    See Also
    --------
    ForwardModel : A forward map that is not linear, written by the caller.
    fips.Operator : A linear one.
    """

    @property
    def rows(self) -> Space:
        """The labels of the observations the map makes."""
        ...

    @property
    def cols(self) -> Space:
        """The labels of the state it takes."""
        ...

    def __call__(self, x: Vector) -> Vector:
        r"""Return the observations the state `x` models, without the offset: :math:`F(x)`, or :math:`H x`, on :attr:`rows`."""
        ...

    def jacobian(self, x: Vector) -> Operator:
        r"""Return the Jacobian at `x`, :math:`J(x)`, an Operator on :attr:`rows` and :attr:`cols` (for an operator, itself)."""
        ...


class ForwardModel(ABC):
    r"""
    A forward model: the map :math:`F` from the state to the observations, and its Jacobian.

    Subclass it and write two methods (only the first for the ensemble
    method, ``fips.solve(problem, "ensemble", ...)``, which runs the model
    and needs no Jacobian), then pass an instance to :class:`~fips.Problem`
    as ``forward=``:

    - ``__call__(x)``: the observations the state `x` models, without the
      offset, :math:`F(x)`, a :class:`~fips.Vector` on :attr:`rows`;
    - ``jacobian(x)``: the derivatives of :math:`F` at `x`, :math:`J_{ij} =
      \partial F_i / \partial x_j`, an :class:`~fips.Operator` with
      :attr:`rows` as its rows and :attr:`cols` as its columns (in any
      order, and in any form: dense, sparse, the product of a matrix and a
      diagonal).

    A model whose Jacobian is too large to hold, but which has a tangent
    linear and an adjoint (a model run forward and backward),
    writes :meth:`jvp` and :meth:`vjp` instead of ``jacobian``: its Jacobian
    is then those products (a :class:`~fips.linalg.MatrixFreeOperator`), and
    ``fips.solve(problem, "variational")`` solves with them alone.

    Call ``super().__init__(rows, cols)`` with the two spaces in the
    subclass's ``__init__``. fips calls both methods with a vector on
    :attr:`cols` itself, so ``self.H @ x`` with ``self.H`` on the same columns
    needs no reordering. A model with a linear map is an
    :class:`~fips.Operator`, and needs no class: an operator is a forward
    model whose Jacobian is itself (both answer :class:`Forward`).

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

    def jacobian(self, x: Vector) -> Operator:
        r"""
        Return the Jacobian of the model at `x`: :math:`J_{ij} = \partial F_i / \partial x_j`.

        ``"gauss_newton"``, ``"variational"`` and :func:`taylortest` need it;
        the ensemble method only runs the model, so a model solved with it
        need not write one. This one gives the products of :meth:`jvp` and
        :meth:`vjp` at `x`, a :class:`~fips.linalg.MatrixFreeOperator`, when
        the subclass writes both, and raises otherwise.

        Parameters
        ----------
        x : Vector
            A state on :attr:`cols`.

        Returns
        -------
        Operator
            :math:`J(x)`, its rows the labels of :attr:`rows` and its columns
            those of :attr:`cols` (in any order).

        Raises
        ------
        NotImplementedError
            Unless the subclass writes it, or both :meth:`jvp` and :meth:`vjp`.
        """
        cls = type(self)
        if cls.jvp is not ForwardModel.jvp and cls.vjp is not ForwardModel.vjp:
            values = _on(self.cols, x, "x", "the model's columns")
            return Operator(_Linearization(self, values), self.rows, self.cols)
        raise NotImplementedError(
            f"{cls.__name__} has no jacobian: write one (or both jvp and vjp, its "
            'products) to solve with "gauss_newton" or "variational" or check it '
            'with taylortest, or solve with "ensemble", which only runs the model.'
        )

    def jvp(self, x: Vector, v: Vector) -> Vector:
        r"""
        Return the Jacobian at `x` times `v`, :math:`J(x)\, v`: a run of the tangent linear model.

        Write it, with :meth:`vjp`, for a model whose Jacobian is known only
        by its products. This one multiplies by :meth:`jacobian`.

        Parameters
        ----------
        x : Vector
            A state on :attr:`cols`, where the model is linearized.
        v : Vector
            A change of the state, on :attr:`cols`.

        Returns
        -------
        Vector
            :math:`J(x)\, v`, on :attr:`rows` (its labels in any order).

        Raises
        ------
        NotImplementedError
            If the subclass writes neither this nor :meth:`jacobian`.
        """
        if type(self).jacobian is ForwardModel.jacobian:
            raise NotImplementedError(
                f"{type(self).__name__} writes neither jacobian nor jvp."
            )
        J = _differentiated(self, _on(self.cols, x, "x", "the model's columns"))
        values = _on(self.cols, v, "v", "the model's columns")
        return Vector(np.asarray(J.data @ values, dtype=float), self.rows)

    def vjp(self, x: Vector, w: Vector) -> Vector:
        r"""
        Return the transpose of the Jacobian at `x` times `w`, :math:`J(x)^T w`: a run of the adjoint model.

        Write it, with :meth:`jvp`, for a model whose Jacobian is known only
        by its products. This one multiplies by the transpose of
        :meth:`jacobian` (or by a cheaper product the model has, as a
        transform does).

        Parameters
        ----------
        x : Vector
            A state on :attr:`cols`, where the model is linearized.
        w : Vector
            Weights of the observations, on :attr:`rows`.

        Returns
        -------
        Vector
            :math:`J(x)^T w`, on :attr:`cols` (its labels in any order).

        Raises
        ------
        NotImplementedError
            If the subclass writes neither this nor :meth:`jacobian`.
        """
        if type(self).jacobian is ForwardModel.jacobian:
            raise NotImplementedError(
                f"{type(self).__name__} writes neither jacobian nor vjp."
            )
        values = _on(self.cols, x, "x", "the model's columns")
        weights = _on(self.rows, w, "w", "the model's rows")
        return Vector(self._gradient(values, weights), self.cols)

    def _gradient(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""
        Return :math:`J(x)^T v`, for the state values `x` in the order of :attr:`cols` and `v` in the order of :attr:`rows`, in the order of :attr:`cols`.

        A product with the transpose of the Jacobian, for a sampler's
        gradient. This one forms the Jacobian (the products of :meth:`vjp`,
        for a model that writes them); a model with a cheaper product
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


class _Linearization(MatrixFreeOperator):
    r"""
    The Jacobian of a model at a state as its products (:meth:`ForwardModel.jvp`, and :meth:`ForwardModel.vjp` for its transpose), on the model's rows and columns in their order.

    A product with a block of vectors is one call of the model's product per
    column. It pickles when the model does.
    """

    def __init__(
        self, model: ForwardModel, values: np.ndarray, transposed: bool = False
    ):
        self._model = model
        self._values = values
        self._transposed = transposed
        shape = (len(model.rows), len(model.cols))
        LinearOperator.__init__(self, shape[::-1] if transposed else shape)

    def _matmat(self, X):
        """Return ``J @ X`` (``J^T @ X`` for the transpose), a column at a time."""
        model = self._model
        x = Vector(self._values, model.cols)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)
        out = np.empty((self.shape[0], X.shape[1]))
        for j in range(X.shape[1]):
            if self._transposed:
                product = model.vjp(x, Vector(X[:, j], model.rows))
                out[:, j] = _returned(model, product, model.cols, "vjp")
            else:
                product = model.jvp(x, Vector(X[:, j], model.cols))
                out[:, j] = _returned(model, product, model.rows, "jvp")
        return out

    def _transpose(self):
        """Return the transpose: the other product."""
        return _Linearization(self._model, self._values, not self._transposed)


def _on(space: Space, v: Vector, what: str, where: str) -> np.ndarray:
    """Return the values of the Vector `v` in the order of `space`, which must have exactly its labels."""
    if not isinstance(v, Vector):
        raise TypeError(f"{what} must be a Vector, got {type(v).__name__}.")
    if v.space is space:
        return v.values
    return v.values[
        space._align(v.space, what, f"{what[:1].upper()}{what[1:]} and {where}")
    ]


def _returned(model: ForwardModel, v: Vector, space: Space, method: str) -> np.ndarray:
    """Return what the model's `method` returned, a Vector on `space`, in the order of `space`."""
    if not isinstance(v, Vector):
        raise TypeError(
            f"{type(model).__name__}.{method} must return a Vector, got "
            f"{type(v).__name__}."
        )
    return _on(space, v, f"what {type(model).__name__}.{method} returns", "its labels")


def _evaluated(model: Forward, values: np.ndarray) -> np.ndarray:
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

_SEEN = 1e6
"""How many times its roundoff the model's output must change at :func:`taylortest`'s largest step for a remainder at roundoff to pass: a Jacobian off by a millionth of the change or more would have shown."""

_ADJOINT = 1e-10
r"""How far :math:`\langle J v, w \rangle` and :math:`\langle v, J^T w \rangle` may differ in :func:`taylortest`, relative to the size of their terms: roundoff of an exact discrete adjoint, far below a wrong one's."""


def taylortest(
    model: ForwardModel,
    x: Vector,
    dx: Vector | None = None,
    rng: int | np.random.Generator | None = 0,
) -> bool:
    r"""
    Check that a model's Jacobian is the derivative of the model, by a Taylor test, and that its transpose is its adjoint.

    For a step :math:`\epsilon` along a direction :math:`\delta x`, the
    remainder :math:`r(\epsilon) = \lVert F(x + \epsilon\, \delta x) - F(x) -
    \epsilon J(x)\, \delta x \rVert` falls as :math:`\epsilon^2` when
    :math:`J` is the Jacobian of :math:`F`, and only as :math:`\epsilon` when
    it is not (a sign flipped, a term missing). The test halves
    :math:`\epsilon` from 1/2 to about :math:`10^{-9}` and passes when the
    remainder falls by at least :math:`2^{1.8}` per halving at its three
    smallest steps above roundoff (:math:`10^3` times the machine epsilon of
    :math:`\lVert F \rVert`). A remainder at roundoff from the start (a model
    whose Jacobian is exact, such as a linear one) passes, if the output moved
    by a million times its roundoff at the largest step; with a `dx` too small
    for that, any Jacobian would pass, and the test raises. The orders are
    logged at DEBUG.

    The Jacobian is the one fips solves with: ``jacobian(x)``, or for a model
    that writes only its products, :meth:`~ForwardModel.jvp` and
    :meth:`~ForwardModel.vjp`. Its transpose must be its adjoint too:
    :math:`\langle J v, w \rangle = \langle v, J^T w \rangle` for random
    :math:`v` and :math:`w`, to within :math:`10^{-10}` of the size of the
    terms (:func:`fips.linalg.dottest`), which an adjoint model with a term
    missing or transposed wrong fails at order one. A failure of either
    check is logged at INFO.

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
    rng : int, numpy.random.Generator or None, default 0
        A seed, or the generator to draw from (used as given, so a generator
        passed on is advanced; None draws from fresh entropy), as
        :meth:`fips.Posterior.sample` takes it: the default direction, then
        the vectors of the adjoint check. A solve has a ``seed``, a draw takes
        an ``rng``; this check draws.

    Returns
    -------
    bool
        True if the remainder falls as the square of the step and the
        transpose is the adjoint.

    Raises
    ------
    TypeError
        If `model` is not a ForwardModel, or it returns other than a Vector
        and an Operator.
    ValueError
        If `x` or `dx` does not have exactly the model's columns' labels, or
        what the model returns does not have exactly its labels; `dx` is zero
        everywhere or not finite; or the remainder is at roundoff while the
        model's output changes by less than a million times roundoff at the
        largest step along `dx` (a step far too small, or a direction the
        observations do not see), where any Jacobian would pass.

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
    draw = np.random.default_rng(rng)
    if dx is None:
        size = np.abs(point)
        size[size == 0] = size.mean() if size.any() else 1.0
        direction = draw.standard_normal(len(point)) * size
    else:
        direction = on_cols(dx, "dx")
        if not np.isfinite(direction).all() or not direction.any():
            raise ValueError(
                "dx must be finite and not zero everywhere: it is the direction "
                "of the steps, and a zero step leaves nothing to test."
            )
    F0 = _evaluated(model, point)
    J = _differentiated(model, point).data
    if not dottest(J, rtol=_ADJOINT, rng=draw):
        logger.info(
            "Taylor test: the transpose of %s's Jacobian is not its adjoint (vjp "
            "is not the transpose of jvp).",
            type(model).__name__,
        )
        return False
    Jdx = np.asarray(J @ direction, dtype=float)
    eps = np.finfo(float).eps
    steps = 0.5 ** np.arange(1, _STEPS + 1)
    remainders, floors = [], []
    changes = []
    for step in steps:
        F = _evaluated(model, point + step * direction)
        remainders.append(float(np.linalg.norm(F - F0 - step * Jdx)))
        floors.append(1e3 * eps * float(np.linalg.norm(F) + np.linalg.norm(F0)))
        changes.append(float(np.linalg.norm(F - F0)))
    r, floor = np.array(remainders), np.array(floors)
    above = np.isfinite(r) & (r > floor)
    pairs = np.flatnonzero(above[:-1] & above[1:])
    if not len(pairs):
        exact = bool(np.all(r[np.isfinite(r)] <= floor[np.isfinite(r)]))
        logger.debug("Taylor test: the remainder is at roundoff (%s).", exact)
        if not exact:
            logger.info("Taylor test: the remainder is not at roundoff.")
            return False
        # a remainder at roundoff says the Jacobian is right only if the
        # output moved well above roundoff, where a wrong one would show
        finite = np.isfinite(r)
        first = int(np.argmax(finite))
        if not (finite.any() and changes[first] > _SEEN * floor[first]):
            raise ValueError(
                "The model's output barely changes along dx, even at the largest "
                f"step that it models (by {changes[first]:.3g}, against a roundoff "
                f"of {floor[first]:.3g}), so the remainder is at roundoff whatever "
                "the Jacobian: the test cannot tell a right one from a wrong one. "
                "Give a dx that moves the observations, such as one of the size of "
                "x (the default)."
            )
        return True
    orders = np.log2(r[pairs] / r[pairs + 1])
    logger.debug(
        "Taylor test: the remainder's order at each halving of the step, from the "
        "largest step: %s.",
        np.array2string(orders, precision=2),
    )
    passed = bool(np.all(orders[-3:] > _ORDER))
    if not passed:
        logger.info(
            "Taylor test: the remainder falls as the step to the power %s at the "
            "smallest steps, not 2: the Jacobian is not the derivative of the model.",
            np.array2string(orders[-3:], precision=2),
        )
    return passed
