"""
Inverse problems: the inputs of a solve, on their labels.

A :class:`Problem` holds the prior, the observations, the forward operator,
the two covariances and an optional offset of a linear inverse problem,
``obs = forward @ state + offset + error``. It checks their labels once,
when it is built, and puts every input on the prior's and the observations'
spaces: :func:`fips.solve` then computes by position.

The problem is the question, and the :class:`~fips.Posterior` that
:func:`fips.solve` returns is the answer. The problem holds what needs no
solve and no ``m x m`` matrix (:meth:`Problem.predict`), and makes other
problems from itself (:meth:`Problem.aggregate`, :meth:`Problem.restrict`);
the posterior holds what depends on the solve or on the ``m x m`` matrix it
factors.
"""

import logging
from typing import Any

import numpy as np

from fips.covariance import Covariance, CovarianceModel
from fips.linalg import LinearOperator
from fips.operator import Operator
from fips.space import Space
from fips.vector import Vector

logger = logging.getLogger(__name__)


def _vector(value: Any, what: str) -> Vector:
    """Return `value` if it is a Vector; raise a TypeError naming the way in otherwise."""
    if not isinstance(value, Vector):
        raise TypeError(
            f"{what} must be a Vector, got {type(value).__name__}. Build one with "
            "Vector.from_series(series, block=...) or Vector.from_blocks({...})."
        )
    return value


def _on(
    op: Operator, rows: Space, cols: Space, what: str, of: tuple[str, str]
) -> LinearOperator:
    """
    Return the form of `op` with its rows on `rows` and its columns on `cols`.

    Both are exact joins: a label on one side and not the other raises, and
    the message names it (:meth:`Space.align <fips.Space.align>`). `of` names
    what `rows` and `cols` are the labels of, for the message.
    """
    r = rows._align(op.rows, f"the rows of {what}", f"The rows of {what} and {of[0]}")
    if cols is rows and op.cols is op.rows:
        c = r
    else:
        c = cols._align(
            op.cols, f"the columns of {what}", f"The columns of {what} and {of[1]}"
        )
    if np.array_equal(r, np.arange(len(r))) and np.array_equal(c, np.arange(len(c))):
        return op.data
    return op.data._select(r, c)


def _covariance(value: Any, space: Space, what: str, of: str) -> Covariance:
    """
    Return `value` as a covariance on `space`.

    A :class:`~fips.covariance.CovarianceModel` is built there; a
    :class:`~fips.Covariance` must have exactly the space's labels.
    """
    if isinstance(value, CovarianceModel):
        return value.build(space, name=what)
    if not isinstance(value, Covariance):
        raise TypeError(
            f"{what} must be a Covariance or a covariance model, got "
            f"{type(value).__name__}. Make one with a model "
            "(Independent().with_std(std)), Covariance.diagonal(variances, space) "
            "or Covariance.from_frame(frame)."
        )
    return Covariance(_on(value, space, space, what, (of, of)), space, name=value.name)


class Problem:
    r"""
    A linear inverse problem: what is known before the observations, and the observations.

    The state is :math:`x \sim N(x_0, S_0)` (the prior), and the observations
    are :math:`z = H x + c + e` with :math:`e \sim N(0, S_z)`. Every input is
    labeled, and the labels are checked once, here, by an exact join: the
    forward operator, the covariances and the offset must have exactly the
    labels of the prior or of the observations, in any order. Each is then put
    in the order of the prior's and the observations' spaces. A problem is not
    changed after it is built; :func:`fips.solve` solves it.

    The problem is the question: it holds what needs no solve and no
    ``m x m`` matrix, such as :meth:`predict`, and it makes other questions
    from itself (:meth:`aggregate`, :meth:`restrict`). What depends on the
    solve, or on the ``m x m`` matrix the solve factors, is on the
    :class:`~fips.Posterior`.

    Parameters
    ----------
    prior : Vector
        The prior state :math:`x_0`. Its space labels the state.
    prior_cov : Covariance or CovarianceModel
        The prior error covariance :math:`S_0`. A covariance model (such as
        ``Kronecker(...).with_std(std)``) is built on the prior's space.
    obs : Vector
        The observations :math:`z`. Its space labels the observations.
    obs_cov : Covariance or CovarianceModel
        The model-data mismatch :math:`S_z`, the covariance of the
        observation error; a model is built on the observations' space. It
        must be positive-definite.
    forward : Operator
        The forward operator :math:`H`, from the state (its columns) to the
        observations (its rows).
    offset : Vector or float, optional
        The part of the observations the state does not explain, :math:`c`
        (a background): a vector on the observations' labels, or one value
        for all of them. None (the default) is no offset.

    Raises
    ------
    TypeError
        If an input is not of a type named above.
    ValueError
        If the forward operator, a covariance or the offset does not have
        exactly the labels of the space it is put on: a block or a label on
        one side and not the other. The message names them.

    See Also
    --------
    fips.solve : Solve a problem; it returns a :class:`~fips.Posterior`.

    Notes
    -----
    A forward operator that covers only part of the state (a transport
    Jacobian has columns only for the cells a footprint touched) is placed on
    the whole state by :meth:`Operator.from_blocks <fips.Operator.from_blocks>`,
    which puts a labeled block by its labels: a cell the Jacobian lacks is a
    column of zeros, and a label the space lacks raises. With ``J`` the
    labeled Jacobian (a DataFrame, dense or pandas-sparse, or a tuple
    ``(matrix, row_labels, col_labels)``)::

        forward = Operator.from_blocks(
            {("co2", "flux"): J}, rows=obs.space, cols=prior.space
        )

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> cells = pd.Index(["a", "b"], name="cell")
    >>> obs_id = pd.Index(["o1", "o2", "o3"], name="obs_id")
    >>> prior = fips.Vector.from_series(pd.Series(0.0, index=cells), block="flux")
    >>> obs = fips.Vector.from_series(
    ...     pd.Series([1.0, 2.0, 3.0], index=obs_id), block="co2"
    ... )
    >>> problem = fips.Problem(
    ...     prior=prior,
    ...     prior_cov=fips.Independent().with_std(1.0),
    ...     obs=obs,
    ...     obs_cov=fips.Independent().with_std(np.sqrt(0.1)),
    ...     forward=fips.Operator(
    ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), obs.space, prior.space
    ...     ),
    ... )
    >>> problem
    Problem(state: Space(flux: 2 x ['cell']), obs: Space(co2: 3 x ['obs_id']))
    >>> fips.solve(problem).estimate.values.round(3)
    array([0.997, 1.906])
    """

    def __init__(
        self,
        prior: Vector,
        prior_cov: "Covariance | CovarianceModel",
        obs: Vector,
        obs_cov: "Covariance | CovarianceModel",
        forward: Operator,
        offset: "Vector | float | None" = None,
    ):
        self._prior = _vector(prior, "prior")
        self._obs = _vector(obs, "obs")
        state, observed = self._prior.space, self._obs.space

        if not isinstance(forward, Operator):
            raise TypeError(
                f"forward must be an Operator, got {type(forward).__name__}. Build "
                "one with Operator(matrix, rows, cols), Operator.from_blocks or "
                "Operator.from_frame."
            )
        self._forward = Operator(
            _on(forward, observed, state, "forward", ("the observations", "the prior")),
            observed,
            state,
            name=forward.name,
        )
        self._prior_cov = _covariance(prior_cov, state, "prior_cov", "the prior")
        self._obs_cov = _covariance(obs_cov, observed, "obs_cov", "the observations")

        if offset is None:
            self._offset = None
        elif isinstance(offset, Vector):
            positions = observed._align(
                offset.space, "the offset", "The offset and the observations"
            )
            self._offset = Vector(offset.values[positions], observed, name=offset.name)
        elif isinstance(offset, (int, float, np.integer, np.floating)):
            self._offset = Vector(np.full(len(observed), float(offset)), observed)
        else:
            raise TypeError(
                f"offset must be a Vector, a number or None, got {type(offset).__name__}."
            )

    @property
    def prior(self) -> Vector:
        """The prior state, :math:`x_0`; its space labels the state."""
        return self._prior

    @property
    def prior_cov(self) -> Covariance:
        """The prior error covariance, :math:`S_0`, on the prior's space."""
        return self._prior_cov

    @property
    def obs(self) -> Vector:
        """The observations, :math:`z`; their space labels the observations."""
        return self._obs

    @property
    def obs_cov(self) -> Covariance:
        """The model-data mismatch, :math:`S_z`, on the observations' space."""
        return self._obs_cov

    @property
    def forward(self) -> Operator:
        """The forward operator, :math:`H`: rows on the observations, columns on the prior."""
        return self._forward

    @property
    def offset(self) -> Vector | None:
        """The offset, :math:`c`, on the observations' space, or None."""
        return self._offset

    def predict(self, x: Vector) -> Vector:
        r"""
        Return the observations the state `x` models, :math:`H x + c`.

        Parameters
        ----------
        x : Vector
            A state on the prior's labels, in any order: the prior, a
            posterior estimate, a truth to test with.

        Returns
        -------
        Vector
            :math:`H x + c` (with the offset :math:`c`, if any) on the
            observations' space.

        Raises
        ------
        TypeError
            If `x` is not a Vector.
        ValueError
            If `x` does not have exactly the prior's labels; the message names
            those on one side only.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> cells = pd.Index(["a", "b"], name="cell")
        >>> obs_id = pd.Index(["o1", "o2", "o3"], name="obs_id")
        >>> prior = fips.Vector.from_series(pd.Series(1.0, index=cells), block="flux")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0, 3.0], index=obs_id), block="co2"
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(0.3),
        ...     forward=fips.Operator(
        ...         np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ...         obs.space,
        ...         prior.space,
        ...     ),
        ...     offset=0.5,
        ... )
        >>> problem.predict(problem.prior).values
        array([1.5, 1.5, 2.5])
        """
        state = self._prior.space
        values = _vector(x, "x").values
        if x.space is not state:
            values = values[state._align(x.space, "x", "x and the prior")]
        predicted = self._forward.data @ values + self._offset_values()
        return Vector(predicted, self._obs.space)

    def restrict(self, keep: Space) -> "Problem":
        r"""
        Return the problem on part of the state, the rest held at its prior.

        Use it to solve for some of the state only, such as the cells the
        observations see well, and keep every other element at its prior
        value. Which elements to keep is your choice: for example, the cells
        whose total sensitivity is above a percentile of all cells'
        (:func:`fips.aggregate.sum` of the forward operator's absolute column
        sums, grouped by cell, gives that total).

        The dropped elements still add to the observations, at their prior
        values: :math:`H_r x_{0,r}` (the forward operator's columns for them
        times their prior) is added to the offset. The prior, the prior
        covariance and the forward operator are cut to the kept labels; the
        covariance keeps its form, so a Kronecker covariance stays as its
        factors. The dropped elements' prior uncertainty is not added to the
        model-data mismatch: they are held exactly at the prior.

        After solving, ``post.estimate.combine_first(problem.prior)`` puts
        the estimate back on the whole state, with the prior for the dropped
        elements (:meth:`Vector.combine_first`).

        Parameters
        ----------
        keep : Space
            The state of the new problem: blocks of the prior's space, each
            with some of its labels (in any order; the new problem has them
            in this order). A block of the prior not in `keep` is held at the
            prior whole.

        Returns
        -------
        Problem
            The restricted problem (of the same class). The observations and
            their covariance are unchanged; its offset is this problem's
            offset (zero if none) plus the contribution of the elements left
            out.

        Raises
        ------
        ValueError
            If `keep` has a block or a label that the prior lacks.

        See Also
        --------
        Vector.combine_first : The estimate put back on the whole state.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> cells = pd.Index(["a", "b", "c"], name="cell")
        >>> obs_id = pd.Index(["o1", "o2"], name="obs_id")
        >>> prior = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0, 3.0], index=cells), block="flux"
        ... )
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([4.0, 5.0], index=obs_id), block="co2"
        ... )
        >>> H = np.array([[1.0, 1.0, 1e-6], [0.0, 1.0, 1e-6]])
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(0.5),
        ...     forward=fips.Operator(H, obs.space, prior.space),
        ... )
        >>> small = problem.restrict(fips.Space({"flux": cells[:2]}))
        >>> small.offset.values
        array([3.e-06, 3.e-06])
        >>> post = fips.solve(small)
        >>> post.estimate.combine_first(problem.prior).values.round(3)
        array([0.034, 4.207, 3.   ])
        """
        if not isinstance(keep, Space):
            raise TypeError(f"keep must be a Space, got {type(keep).__name__}.")
        state = self._prior.space
        kept = state._subset(keep, "The space to keep")
        left_out = np.ones(len(state), dtype=bool)
        left_out[kept] = False
        H = self._forward.data
        rows = np.arange(H.shape[0])

        offset = self._offset
        if left_out.any():
            removed = np.flatnonzero(left_out)
            contribution = H._select(rows, removed) @ self._prior.values[removed]
            name = None if offset is None else offset.name
            values = self._offset_values() + contribution
            offset = Vector(values, self._obs.space, name=name)

        prior_cov = self._prior_cov
        return type(self)(
            prior=Vector(self._prior.values[kept], keep, name=self._prior.name),
            prior_cov=Covariance(
                prior_cov.data._select(kept, kept), keep, name=prior_cov.name
            ),
            obs=self._obs,
            obs_cov=self._obs_cov,
            forward=Operator(
                H._select(rows, kept), self._obs.space, keep, name=self._forward.name
            ),
            offset=offset,
        )

    def aggregate(self, W: Operator) -> "Problem":
        r"""
        Return the problem of aggregated observations.

        Each input on the observations goes through the same aggregation
        :math:`W`: the observations :math:`W z`, the forward operator
        :math:`W H`, the model-data mismatch :math:`W S_z W^T` and the offset
        :math:`W c`. Each keeps the cheapest form (a sparse :math:`W` and a
        diagonal :math:`S_z` give a sparse mismatch).

        Parameters
        ----------
        W : Operator
            The aggregation, such as :func:`fips.aggregate.mean`'s: its
            columns the observations' labels (in any order), its rows the
            aggregated observations.

        Returns
        -------
        Problem
            The problem (of the same class) on the aggregated observations,
            with the same prior and prior covariance.

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If the columns of `W` do not have exactly the observations'
            labels.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> times = pd.date_range("2020-01-01", periods=4, freq="12h", name="obs_time")
        >>> obs = fips.Vector.from_series(
        ...     pd.Series([1.0, 2.0, 3.0, 4.0], index=times), block="co2"
        ... )
        >>> prior = fips.Vector.from_series(
        ...     pd.Series(0.0, index=pd.Index(["a"], name="cell")), block="flux"
        ... )
        >>> problem = fips.Problem(
        ...     prior=prior,
        ...     prior_cov=fips.Independent().with_std(1.0),
        ...     obs=obs,
        ...     obs_cov=fips.Independent().with_std(1.0),
        ...     forward=fips.Operator(np.ones((4, 1)), obs.space, prior.space),
        ... )
        >>> W = fips.aggregate.mean(
        ...     obs.space, by=pd.Grouper(level="obs_time", freq="1D")
        ... )
        >>> daily = problem.aggregate(W)
        >>> daily.obs.values, daily.obs_cov.variances.values
        (array([1.5, 3.5]), array([0.5, 0.5]))
        """
        if not isinstance(W, Operator):
            raise TypeError(
                f"An aggregation is an Operator, not {type(W).__name__}: build one "
                "with fips.aggregate.mean or fips.aggregate.sum."
            )
        observed = self._obs.space
        form = _on(
            W, W.rows, observed, "the aggregation", ("its rows", "the observations")
        )
        rows = W.rows
        offset = self._offset
        if offset is not None:
            offset = Vector(form @ offset.values, rows, name=offset.name)
        obs_cov = self._obs_cov
        return type(self)(
            prior=self._prior,
            prior_cov=self._prior_cov,
            obs=Vector(form @ self._obs.values, rows, name=self._obs.name),
            obs_cov=Covariance(
                (form @ obs_cov.data) @ form._transpose(), rows, name=obs_cov.name
            ),
            forward=Operator(
                form @ self._forward.data,
                rows,
                self._prior.space,
                name=self._forward.name,
            ),
            offset=offset,
        )

    def _offset_values(self) -> np.ndarray:
        """Return the offset's values, zeros where there is none."""
        if self._offset is None:
            return np.zeros(len(self._obs))
        return self._offset.values

    def __repr__(self) -> str:
        """Return the two spaces."""
        return f"Problem(state: {self._prior.space!r}, obs: {self._obs.space!r})"

    def __getstate__(self) -> dict[str, Any]:
        """Return the inputs, already on their spaces."""
        return {
            "prior": self._prior,
            "prior_cov": self._prior_cov,
            "obs": self._obs,
            "obs_cov": self._obs_cov,
            "forward": self._forward,
            "offset": self._offset,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`; the labels were checked when it was built."""
        self._prior = state["prior"]
        self._prior_cov = state["prior_cov"]
        self._obs = state["obs"]
        self._obs_cov = state["obs_cov"]
        self._forward = state["forward"]
        self._offset = state["offset"]
