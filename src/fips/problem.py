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

from fips.base import Pickleable, Structure1D
from fips.covariance import CovarianceMatrix
from fips.estimators import ESTIMATOR_REGISTRY, Estimator
from fips.matrix import Matrix, MatrixBlock, MatrixLike
from fips.operators import ForwardOperator
from fips.posterior import Posterior
from fips.vector import Vector, VectorLike

logger = logging.getLogger(__name__)

_SHOWN = 5  # how many labels an error message lists


def _plain(label):
    """Return ``label`` with NumPy scalars as Python ones, so it prints as typed."""
    if isinstance(label, tuple):
        return tuple(_plain(part) for part in label)
    return label.item() if isinstance(label, np.generic) else label


def _listed(labels: list) -> str:
    """Return the first few of ``labels`` as text, saying how many were left out."""
    text = ", ".join(repr(_plain(label)) for label in labels[:_SHOWN])
    if len(labels) > _SHOWN:
        text += f" and {len(labels) - _SHOWN} more"
    return text


def _check_unique_labels(
    index: pd.Index,
    space: str,
    unrounded: pd.Index | None = None,
    decimals: int | None = None,
) -> None:
    """
    Raise a ValueError naming the duplicate labels of ``index``.

    Parameters
    ----------
    index : pd.Index
        The state or observation index.
    space : str
        What the index labels ("state" or "observation"), for the message.
    unrounded : pd.Index, optional
        ``index`` before ``round_index`` rounded it. When given, the message
        names the distinct labels that rounding merged.
    decimals : int, optional
        The ``round_index`` that was used, for the message.
    """
    # The first position of each label that appears more than once
    firsts = np.flatnonzero(
        index.duplicated(keep=False) & ~index.duplicated(keep="first")
    )
    if len(firsts) == 0:
        return
    if unrounded is None:
        raise ValueError(
            f"The {space} index has duplicate labels: "
            f"{_listed(list(index[firsts]))}. Each label must be unique."
        )
    merged = []
    for first in firsts[:_SHOWN]:
        label = index[[first]]
        sources = unrounded[index.get_indexer_for(label)]
        merged.append(f"{_listed(list(sources))} become {_listed(list(label))}")
    more = f"; {len(firsts) - _SHOWN} more labels" if len(firsts) > _SHOWN else ""
    raise ValueError(
        f"round_index={decimals} merges distinct labels of the {space} index: "
        f"{'; '.join(merged)}{more}. Pass round_index=None, or more decimals, "
        "to keep them apart."
    )


def _check_block_names(
    name: str, matrix: Matrix, rows: pd.Index, columns: pd.Index
) -> None:
    """
    Raise a ValueError if a block of ``matrix`` matches no block of its space.

    ``rows`` and ``columns`` are the indexes ``matrix`` is placed on. A block
    they have that ``matrix`` lacks is fine (it is filled with zeros), as are
    labels missing within a block; a block name they do not have is a typo or
    a stale name, and reindexing would silently drop that block's values.
    """
    for axis, labels, target, argument in (
        ("row", matrix.index, rows, "row_block"),
        ("column", matrix.columns, columns, "col_block"),
    ):
        known = list(target.unique(level="block"))
        unknown = [b for b in labels.unique(level="block") if b not in known]
        if unknown:
            raise ValueError(
                f"{name} has {axis} blocks that match no block of the space "
                f"it is placed on: {_listed(unknown)}. The known blocks are "
                f"{_listed(known)}. Check the {argument} of each MatrixBlock."
            )


class InverseProblem(Pickleable):
    """
    Inverse problem combining observations, priors, and forward model.

    Organizes state vectors, observations, forward operators, and error covariances
    into a unified framework for solving inverse problems via different estimators.

    Parameters
    ----------
    obs : VectorLike
        Observation vector.
    prior : VectorLike
        Prior state vector.
    forward_operator : MatrixLike
        Forward operator mapping state space to observation space.
    modeldata_mismatch : MatrixLike or Vector or float
        Covariance matrix representing model-data mismatch (observation error).
        A scalar is one variance for every observation, and a Series or Vector
        (with a ``block`` level) gives a variance for each: both build a
        diagonal covariance.
    prior_error : MatrixLike or Vector or float
        Covariance matrix representing prior error. A scalar, a Series or a
        Vector builds a diagonal covariance, as for `modeldata_mismatch`.
    constant : VectorLike or float, optional
        Optional constant term added to the forward model (e.g., background or bias).
    round_index : int, optional
        Number of decimal places to round float labels to. If None, no
        rounding is performed. Labels that rounding makes equal raise a
        ValueError rather than being merged.

    Raises
    ------
    ValueError
        If the state or observation index has duplicate labels, before or
        after rounding, or if a block name of `forward_operator`,
        `prior_error` or `modeldata_mismatch` (a MatrixBlock's `row_block` or
        `col_block`) matches no block of the space it is placed on. Labels
        missing within a known block are filled with zeros, with a warning.
    """

    obs: Vector
    """Observation vector."""
    prior: Vector
    """Prior state vector."""
    forward_operator: ForwardOperator
    """Forward operator mapping state space to observation space."""
    prior_error: CovarianceMatrix
    """Covariance matrix representing prior error."""
    modeldata_mismatch: CovarianceMatrix
    """Covariance matrix representing model-data mismatch (observation error)."""
    constant: Vector | None
    """Optional constant term added to the forward model (e.g., background or bias)."""

    def __init__(
        self,
        obs: VectorLike,
        prior: VectorLike,
        forward_operator: MatrixLike,
        modeldata_mismatch: "MatrixLike | Vector",
        prior_error: "MatrixLike | Vector",
        constant: "VectorLike | float | None" = None,
        round_index: int | None = 6,
    ):
        obs = Vector(obs)
        prior = Vector(prior)

        def promote_2d(
            data, target_index: pd.MultiIndex, target_columns: pd.MultiIndex
        ):
            """Promote 2D DataFrame to MatrixBlock, using block levels from target index/columns if needed."""
            if isinstance(data, pd.DataFrame):
                has_row = "block" in data.axes[0].names
                has_col = "block" in data.axes[1].names

                if not has_row:
                    row_blks = target_index.get_level_values("block").unique()
                    if len(row_blks) > 1:
                        raise ValueError(
                            "Data has no row 'block' level but target index has multiple blocks."
                        )
                    row_block = str(row_blks[0]) if len(row_blks) == 1 else None
                else:
                    row_block = None

                if not has_col:
                    col_blks = target_columns.get_level_values("block").unique()
                    if len(col_blks) > 1:
                        raise ValueError(
                            "Data has no col 'block' level but target columns has multiple blocks."
                        )
                    col_block = str(col_blks[0]) if len(col_blks) == 1 else None
                else:
                    col_block = None

                if not has_row or not has_col:
                    # Wrap naked DataFrame in a MatrixBlock
                    return MatrixBlock(data, row_block=row_block, col_block=col_block)
            return data

        forward_operator = ForwardOperator(
            promote_2d(
                forward_operator, target_index=obs.index, target_columns=prior.index
            )
        )

        def covariance(data, target_index: pd.MultiIndex) -> CovarianceMatrix:
            """Build a covariance; a scalar, Series or Vector is its diagonal."""
            if isinstance(data, (pd.Series, Structure1D)):
                return CovarianceMatrix.diagonal(data)
            if isinstance(data, (int, float, np.number)):
                return CovarianceMatrix.diagonal(data, index=target_index)
            return CovarianceMatrix(
                promote_2d(data, target_index=target_index, target_columns=target_index)
            )

        prior_error = covariance(prior_error, prior.index)
        modeldata_mismatch = covariance(modeldata_mismatch, obs.index)

        if constant is not None:
            constant_index = obs.index if isinstance(constant, (int, float)) else None
            constant = Vector(constant, index=constant_index)

        _check_unique_labels(obs.index, "observation")
        _check_unique_labels(prior.index, "state")

        # Round to specified precision if provided
        if round_index is not None:
            unrounded_obs, unrounded_prior = obs.index, prior.index
            obs = obs.round_index(round_index)
            prior = prior.round_index(round_index)
            _check_unique_labels(obs.index, "observation", unrounded_obs, round_index)
            _check_unique_labels(prior.index, "state", unrounded_prior, round_index)
            forward_operator = forward_operator.round_index(round_index, axis="both")
            modeldata_mismatch = modeldata_mismatch.round_index(
                round_index, axis="both"
            )
            prior_error = prior_error.round_index(round_index, axis="both")
            if constant is not None:
                constant = constant.round_index(round_index)

        # A block name that matches no block would be dropped by the reindex
        _check_block_names("forward_operator", forward_operator, obs.index, prior.index)
        _check_block_names(
            "modeldata_mismatch", modeldata_mismatch, obs.index, obs.index
        )
        _check_block_names("prior_error", prior_error, prior.index, prior.index)

        # Reindex matrices to obs and prior (state) indices
        def reindex(matrix, row_idx, col_idx):
            """Reindex ``matrix`` onto the given labels, filling gaps with zeros."""
            return matrix.reindex(
                index=row_idx, columns=col_idx, fill_value=0.0, verify_overlap=True
            )

        forward_operator = reindex(
            forward_operator, row_idx=obs.index, col_idx=prior.index
        )
        modeldata_mismatch = reindex(
            modeldata_mismatch, row_idx=obs.index, col_idx=obs.index
        )
        prior_error = reindex(prior_error, row_idx=prior.index, col_idx=prior.index)
        if constant is not None:
            constant = constant.reindex(
                index=obs.index, fill_value=0.0, verify_overlap=True
            )

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
    ) -> pd.DataFrame | pd.Series:
        """
        Get block from a component (Vector or Matrix).

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
        pd.Series or pd.DataFrame
            The requested block of data.
        """
        obj = getattr(self, component)

        if isinstance(obj, Vector):
            return obj[block]
        elif isinstance(obj, Matrix):
            if crossblock is None:
                crossblock = block
            return obj[block, crossblock]
        else:
            raise TypeError(f"Object '{component}' is neither a Vector nor a Matrix.")

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
    def posterior_error(self) -> CovarianceMatrix:
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
    def prior_obs_error(self) -> CovarianceMatrix:
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
    def kalman_gain(self) -> Matrix:
        """Kalman gain matrix, K (:attr:`~fips.Posterior.kalman_gain`)."""
        return self.result.kalman_gain

    @property
    def averaging_kernel(self) -> Matrix:
        """Averaging kernel matrix, A (:attr:`~fips.Posterior.averaging_kernel`)."""
        return self.result.averaging_kernel

    @property
    def desroziers(self) -> CovarianceMatrix:
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
        H=problem.forward_operator.operator,
        S_0=problem.prior_error.operator,
        S_z=problem.modeldata_mismatch.operator,
        c=getattr(problem.constant, "values", None),
        **options,
    )
    posterior = Posterior(fitted, problem.state_index, problem.obs_index)
    _ = posterior.estimate  # solve now, as InverseProblem.solve always has
    return posterior
