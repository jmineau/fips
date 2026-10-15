"""
Inverse problem framework for state estimation.

This module provides the core InverseProblem class which combines observations,
prior estimates, forward operators, and error covariances into a unified framework
for state estimation.
"""

import logging
from typing import Self

import numpy as np
import pandas as pd

from fips import indexes
from fips.base import Pickleable
from fips.covariance import Covariance
from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.linalg import LinearOperator
from fips.operator import Operator, _check_overlap
from fips.posterior import Posterior
from fips.space import _SHOWN, Space, _first_duplicates, _listed, _positions
from fips.vector import Vector

logger = logging.getLogger(__name__)


def _check_rounded_labels(
    index: pd.Index, unrounded: pd.Index, space: str, decimals: int
) -> None:
    """
    Raise a ValueError naming the labels of ``unrounded`` that rounding merged in ``index``.

    Parameters
    ----------
    index : pd.Index
        The state or observation index, after ``round_index``.
    unrounded : pd.Index
        ``index`` before ``round_index`` rounded it.
    space : str
        What the index labels ("state" or "observation"), for the message.
    decimals : int
        The ``round_index`` that was used, for the message.
    """
    firsts = _first_duplicates(index)
    if len(firsts) == 0:
        return
    merged = []
    for first in firsts[:_SHOWN]:
        label = index[[first]]
        sources = unrounded[index.get_indexer_for(label)]
        merged.append(f"{_listed(list(sources))} become {_listed(list(label))}")
    more = f"; {len(firsts) - _SHOWN} more labels" if len(firsts) > _SHOWN else ""
    raise ValueError(
        f"round_index={decimals} merges distinct labels of the {space} index: "
        f"{'; '.join(merged)}{more}. Pass round_index=None (the default), or more "
        "decimals, to keep them apart."
    )


def _place(op: Operator, rows: Space, cols: Space, what: str) -> Operator:
    """
    Return `op` on `rows` and `cols`, zero where it has no entry for a label.

    A block of `op` that `rows` or `cols` does not have is a typo or a stale
    name, and placing would silently drop its values, so it raises. Labels
    missing within a known block are filled with zeros, with a warning; labels
    `op` has that the spaces do not are dropped.
    """
    for axis, mine, target in (("row", op.rows, rows), ("column", op.cols, cols)):
        unknown = [b for b in mine.names if b not in target]
        if unknown:
            raise ValueError(
                f"{what} has {axis} blocks that match no block of the space "
                f"it is placed on: {_listed(unknown)}. The known blocks are "
                f"{_listed(target.names)}."
            )
    r, c = _positions(rows, op.rows), _positions(cols, op.cols)
    _check_overlap(r, rows, what)
    _check_overlap(c, cols, what)
    same = np.array_equal(r, np.arange(len(r))) and np.array_equal(c, np.arange(len(c)))
    data = op.data if same else op.data._select(r, c)
    return op._like(data, rows, cols)


def _as_vector(data: "Vector | pd.Series", what: str) -> Vector:
    """
    Return `data` as a Vector.

    A Series with a ``block`` level has those blocks; one without is one
    block, named by the Series' name.
    """
    if isinstance(data, Vector):
        return data
    if isinstance(data, pd.Series):
        if "block" in data.index.names:
            return Vector.from_series(data)
        if data.name is None:
            raise ValueError(
                f"{what} has no 'block' level and no name to name its one block."
            )
        return Vector.from_series(data, block=str(data.name))
    raise TypeError(f"{what} must be a Vector or a Series, got {type(data).__name__}.")


class InverseProblem(Pickleable):
    """
    Inverse problem combining observations, priors, and forward model.

    Organizes state vectors, observations, forward operators, and error covariances
    into a unified framework for solving inverse problems via different estimators.

    Parameters
    ----------
    obs : Vector or pd.Series
        Observation vector. A Series with a ``block`` level has those blocks;
        one without is one block, named by the Series' name.
    prior : Vector or pd.Series
        Prior state vector, given as `obs` is.
    forward_operator : Operator or pd.DataFrame
        Forward operator mapping state space to observation space. A
        DataFrame is read by :meth:`Operator.from_frame
        <fips.Operator.from_frame>`; an axis without a ``block`` level takes
        the one block of the observations (rows) or the state (columns).
    modeldata_mismatch : Covariance, Operator, pd.DataFrame, Vector, float or LinearOperator
        Covariance matrix representing model-data mismatch (observation
        error). A scalar is one variance for every observation, and a Series
        (with a ``block`` level) or a Vector gives a variance for each: both
        build a diagonal covariance. A form of :mod:`fips.linalg` (such as a
        :class:`~fips.kronecker.KroneckerOperator`) is labeled by the
        observations, in their order, without forming the matrix.
    prior_error : Covariance, Operator, pd.DataFrame, Vector, float or LinearOperator
        Covariance matrix representing prior error, given in any of the ways
        `modeldata_mismatch` is; a form of :mod:`fips.linalg` is labeled by
        the prior, in its order.
    constant : Vector, pd.Series or float, optional
        Optional constant term added to the forward model (e.g., background or bias).
    round_index : int, optional
        Number of decimal places to round every float label of every input
        to. By default the labels are kept as given: fips does not round them
        unless asked. Labels that rounding makes equal raise a ValueError
        rather than being merged.

    Raises
    ------
    ValueError
        If the state or observation index has duplicate labels, before or
        after rounding, or if a block of `forward_operator`, `prior_error` or
        `modeldata_mismatch` matches no block of the space it is placed on.
        Labels missing within a known block are filled with zeros, with a
        warning.

    Notes
    -----
    The observations and the prior are put in the order of their
    :class:`~fips.Space`: each block's rows together, the blocks in the order
    they first appear. Every matrix is put on those spaces.
    """

    obs: Vector
    """Observation vector."""
    prior: Vector
    """Prior state vector."""
    forward_operator: Operator
    """Forward operator mapping state space to observation space."""
    prior_error: Covariance
    """Covariance matrix representing prior error."""
    modeldata_mismatch: Covariance
    """Covariance matrix representing model-data mismatch (observation error)."""
    constant: Vector | None
    """Optional constant term added to the forward model (e.g., background or bias)."""

    def __init__(
        self,
        obs: "Vector | pd.Series",
        prior: "Vector | pd.Series",
        forward_operator: "Operator | pd.DataFrame",
        modeldata_mismatch: "Operator | pd.DataFrame | Vector | float | LinearOperator",
        prior_error: "Operator | pd.DataFrame | Vector | float | LinearOperator",
        constant: "Vector | pd.Series | float | None" = None,
        round_index: int | None = None,
    ):
        obs = _as_vector(obs, "obs")
        prior = _as_vector(prior, "prior")

        # Duplicate labels raised when obs and prior became vectors. Rounding is
        # opt-in; the rounded labels are checked before the vectors are rebuilt
        # on them, so that labels rounding merges are named as such.
        if round_index is not None:
            _check_rounded_labels(
                indexes.round_index(obs.index, decimals=round_index),
                obs.index,
                "observation",
                round_index,
            )
            _check_rounded_labels(
                indexes.round_index(prior.index, decimals=round_index),
                prior.index,
                "state",
                round_index,
            )
            obs = obs.round_labels(round_index)
            prior = prior.round_labels(round_index)
        obs_space = obs.space
        state_space = prior.space

        def operator(data, rows: Space, cols: Space, what: str) -> Operator:
            """Return `data` as an operator; a frame's axis without blocks takes the space's one block."""
            if isinstance(data, Operator):
                return data
            if isinstance(data, pd.DataFrame):
                blocks = []
                for labels, space, axis in (
                    (data.index, rows, "rows"),
                    (data.columns, cols, "columns"),
                ):
                    if "block" in labels.names:
                        blocks.append(None)
                    elif len(space.names) == 1:
                        blocks.append(space.names[0])
                    else:
                        raise ValueError(
                            f"The {axis} of {what} have no 'block' level, but the "
                            f"space they are placed on has blocks {_listed(space.names)}."
                        )
                return Operator.from_frame(
                    data, row_block=blocks[0], col_block=blocks[1]
                )
            more = ""
            if rows is cols:  # a covariance
                more = (
                    "; a covariance may also be a scalar, a Series or Vector of "
                    "variances, or a fips.linalg form"
                )
            raise TypeError(
                f"{what} must be an Operator or a DataFrame{more}, got "
                f"{type(data).__name__}."
            )

        def covariance(data, space: Space, what: str) -> Operator:
            """Return `data` as a covariance; a scalar, Series or Vector is its diagonal."""
            if isinstance(data, LinearOperator):
                return Covariance(data, space)
            if isinstance(data, Vector):
                return Covariance.diagonal(data)
            if isinstance(data, pd.Series):
                if "block" in data.index.names:
                    variances = Vector.from_series(data)
                elif len(space.names) == 1:
                    variances = Vector.from_series(data, block=space.names[0])
                else:
                    raise ValueError(
                        f"The variances of {what} have no 'block' level, but the "
                        f"space they are placed on has blocks {_listed(space.names)}."
                    )
                return Covariance.diagonal(variances)
            if isinstance(data, (int, float, np.number)):
                return Covariance.diagonal(float(data), space)
            return operator(data, space, space, what)

        forward_operator = operator(
            forward_operator, obs_space, state_space, "forward_operator"
        )
        prior_error = covariance(prior_error, state_space, "prior_error")
        modeldata_mismatch = covariance(
            modeldata_mismatch, obs_space, "modeldata_mismatch"
        )
        if round_index is not None:
            forward_operator = forward_operator.round_labels(round_index)
            prior_error = prior_error.round_labels(round_index)
            modeldata_mismatch = modeldata_mismatch.round_labels(round_index)

        forward_operator = _place(
            forward_operator, obs_space, state_space, "forward_operator"
        )
        prior_error = _place(prior_error, state_space, state_space, "prior_error")
        modeldata_mismatch = _place(
            modeldata_mismatch, obs_space, obs_space, "modeldata_mismatch"
        )
        if not isinstance(prior_error, Covariance):
            prior_error = Covariance(prior_error.data, state_space, prior_error.name)
        if not isinstance(modeldata_mismatch, Covariance):
            modeldata_mismatch = Covariance(
                modeldata_mismatch.data, obs_space, modeldata_mismatch.name
            )

        if isinstance(constant, (int, float)):
            constant = Vector(np.full(len(obs_space), float(constant)), obs_space)
        elif constant is not None:
            constant = _as_vector(constant, "constant")
            if round_index is not None:
                constant = constant.round_labels(round_index)
            found = _positions(obs_space, constant.space)
            _check_overlap(found, obs_space, "constant")
            values = np.where(found >= 0, constant.values[np.maximum(found, 0)], 0.0)
            constant = Vector(values, obs_space, name=constant.name)

        self.obs = obs
        self.prior = prior
        self.forward_operator = forward_operator
        self.modeldata_mismatch = modeldata_mismatch
        self.prior_error = prior_error
        self.constant = constant
        self._result: Posterior | None = None  # set by solve

    @property
    def state_index(self) -> pd.Index:
        """Return the state space index."""
        return self.prior.index

    @property
    def obs_index(self) -> pd.Index:
        """Return the observation space index."""
        return self.obs.index

    @property
    def n_state(self) -> int:
        """Return number of state variables."""
        return len(self.state_index)

    @property
    def n_obs(self) -> int:
        """Return number of observations."""
        return len(self.obs_index)

    @property
    def result(self) -> Posterior:
        """
        The posterior from the last :meth:`solve` (raises if not solved).

        ``posterior``, ``posterior_error``, ``posterior_obs``, ``prior_obs``,
        ``prior_obs_error``, ``kalman_gain``, ``averaging_kernel`` and
        ``desroziers`` are its outputs, under the names they have on the
        problem: ``problem.posterior`` is ``problem.result.estimate``.

        See Also
        --------
        fips.solve : Solve a problem and return its posterior.
        """
        if self._result is None:
            raise RuntimeError("Problem has not been solved. Call .solve() first.")
        return self._result

    @property
    def estimator(self) -> Estimator:
        """Return the fitted estimator (raises if not solved)."""
        return self.result.estimator

    @property
    def _estimator(self) -> Estimator | None:
        """The fitted estimator, or None before :meth:`solve`."""
        return None if self._result is None else self._result.estimator

    def get_block(
        self, component: str, block: str, crossblock: str | None = None
    ) -> "Operator | Vector":
        """
        Get block from a component (Vector or Operator).

        Parameters
        ----------
        component : str
            Name of the component ('obs', 'prior', 'forward_operator', 'modeldata_mismatch', 'prior_error', or 'constant').
        block : str
            Name of the block to retrieve.
        crossblock : str, optional
            For matrices, the name of the cross block (e.g., 'state' for forward_operator). If None, defaults to the same as 'block'.

        Returns
        -------
        Vector or Operator
            The requested block: a :class:`~fips.Vector` of that block for a
            vector, an :class:`~fips.Operator` (on the two blocks) for a
            matrix.
        """
        obj = getattr(self, component)

        if isinstance(obj, Vector):
            return obj[block]
        elif isinstance(obj, Operator):
            if crossblock is None:
                crossblock = block
            return obj[block, crossblock]
        else:
            raise TypeError(
                f"Object '{component}' is neither a Vector nor an Operator."
            )

    def solve(self, estimator: str | type[Estimator], **kwargs) -> Self:
        """
        Solve the inverse problem using the specified estimator.

        Calls :func:`fips.solve` and keeps the :class:`~fips.Posterior` it
        returns as :attr:`result`.

        Parameters
        ----------
        estimator : str or type[Estimator]
            Estimator to use for solving the inverse problem. Can be a string key for registered estimators or a subclass of Estimator.
        **kwargs
            Additional keyword arguments to pass to the estimator.

        Returns
        -------
        Self
            The InverseProblem instance with the estimator fitted.
        """
        # The module's solve function, not this method.
        self._result = solve(self, estimator, **kwargs)
        return self

    def __getstate__(self):
        """Explicit pickle support: return state as dict."""
        return {
            "prior": self.prior,
            "obs": self.obs,
            "forward_operator": self.forward_operator,
            "prior_error": self.prior_error,
            "modeldata_mismatch": self.modeldata_mismatch,
            "constant": self.constant,
            # The estimator, not the posterior, so a pickle reads as it did
            # before the posterior existed; the posterior is rebuilt from it.
            "_estimator": self._estimator,
        }

    def __setstate__(self, state):
        """Explicit pickle support: restore state from dict."""
        self.prior = state["prior"]
        self.obs = state["obs"]
        self.forward_operator = state["forward_operator"]
        self.prior_error = state["prior_error"]
        self.modeldata_mismatch = state["modeldata_mismatch"]
        self.constant = state["constant"]
        estimator = state.get("_estimator", None)
        self._result = (
            None
            if estimator is None
            else Posterior(estimator, self.state_index, self.obs_index)
        )

    def __repr__(self) -> str:
        """Return string representation."""
        solved = self._estimator is not None
        return (
            f"{self.__class__.__name__}("
            f"n_obs={self.n_obs}, n_state={self.n_state}, solved={solved})"
        )

    @property
    def posterior(self) -> Vector:
        """Posterior state estimate (:attr:`~fips.Posterior.estimate`)."""
        return self.result.estimate

    @property
    def posterior_error(self) -> Covariance:
        """Posterior error covariance (:attr:`~fips.Posterior.covariance`)."""
        return self.result.covariance

    @property
    def posterior_obs(self) -> Vector:
        """Modeled observations, H @ posterior + c (:attr:`~fips.Posterior.modeled_obs`)."""
        return self.result.modeled_obs

    @property
    def prior_obs(self) -> Vector:
        """Modeled observations, H @ prior + c (:attr:`~fips.Posterior.prior_modeled_obs`)."""
        return self.result.prior_modeled_obs

    @property
    def prior_obs_error(self) -> Covariance:
        """
        Prior error covariance in observation space (H @ S_0 @ H.T).

        The uncertainty of ``prior_obs`` that the prior error implies, on the
        observation index. Its diagonal, square-rooted, is the prior's share
        of each observation's error budget; add the model-data mismatch in
        quadrature for the total the solve expects
        (:attr:`~fips.Posterior.prior_obs_covariance`).
        """
        return self.result.prior_obs_covariance

    @property
    def kalman_gain(self) -> Operator:
        """Kalman gain matrix, K (:attr:`~fips.Posterior.kalman_gain`)."""
        return self.result.kalman_gain

    @property
    def averaging_kernel(self) -> Operator:
        """Averaging kernel matrix, A (:meth:`~fips.Posterior.averaging_kernel`)."""
        return self.result.averaging_kernel()

    @property
    def desroziers(self) -> Covariance:
        """Desroziers et al. (2005) diagnosed observation error covariance (:attr:`~fips.Posterior.desroziers`)."""
        return self.result.desroziers


def solve(
    problem: InverseProblem,
    estimator: str | type[Estimator] = "bayesian",
    **options,
) -> Posterior:
    """
    Solve an inverse problem and return its posterior.

    Fits the estimator to the problem's inputs and labels what it computes
    with the problem's indexes. The problem is not changed;
    :meth:`~fips.InverseProblem.solve` calls this and keeps the result.

    Parameters
    ----------
    problem : InverseProblem
        The problem to solve.
    estimator : str or type[Estimator], default "bayesian"
        A registered estimator's name (see :func:`~fips.available_estimators`) or
        an :class:`~fips.Estimator` subclass.
    **options
        Passed to the estimator, such as ``gamma`` for ``"bayesian"``.

    Returns
    -------
    Posterior
        The labeled solution. Its estimate is computed before it returns.

    Raises
    ------
    ValueError
        If ``estimator`` is a name that is not registered.
    TypeError
        If ``estimator`` is neither a name nor an Estimator subclass.

    See Also
    --------
    fips.Posterior : What it returns.
    fips.InverseProblem.solve : Solve in place and keep the posterior.

    Examples
    --------
    >>> import pandas as pd
    >>> import fips
    >>> state = pd.Index(["a", "b"], name="cell")
    >>> obs = pd.Index(["o1", "o2", "o3"], name="obs_id")
    >>> problem = fips.InverseProblem(
    ...     obs=pd.Series([1.0, 2.0, 3.0], index=obs, name="obs"),
    ...     prior=pd.Series([0.0, 0.0], index=state, name="prior"),
    ...     forward_operator=pd.DataFrame(
    ...         [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=obs, columns=state
    ...     ),
    ...     prior_error=1.0,
    ...     modeldata_mismatch=0.1,
    ... )
    >>> post = fips.solve(problem, gamma=0.5)
    >>> post.estimator.gamma
    0.5
    """
    if isinstance(estimator, str):
        if estimator not in ESTIMATOR_REGISTRY:
            raise ValueError(f"Estimator '{estimator}' is not registered.")
        estimator_cls = ESTIMATOR_REGISTRY[estimator]
    elif isinstance(estimator, type) and issubclass(estimator, Estimator):
        estimator_cls = estimator
    else:
        raise TypeError("Estimator must be a string or a subclass of Estimator.")

    logger.info(f"Solving using {estimator_cls.__name__}...")
    fitted = estimator_cls(
        z=problem.obs.values,
        x_0=problem.prior.values,
        H=problem.forward_operator.data,
        S_0=problem.prior_error.data,
        S_z=problem.modeldata_mismatch.data,
        c=getattr(problem.constant, "values", None),
        **options,
    )
    posterior = Posterior(fitted, problem.state_index, problem.obs_index)
    _ = posterior.estimate  # solve now, as InverseProblem.solve always has
    return posterior
