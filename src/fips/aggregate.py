"""
Aggregation: operators that sum or average the labels of a space by group.

:func:`sum` and :func:`mean` build an aggregation :math:`W`, an
:class:`~fips.Operator` from a :class:`~fips.Space` (its columns) to the groups
of its labels (its rows). The same operator works on both sides of a problem:

- on the observations, ``problem.aggregate(W)`` gives the problem of the
  aggregated observations, :math:`W z`, :math:`W H`, :math:`W S_z W^T` and
  :math:`W c` (hourly observations to daily means, say);
- on the state, ``post.cov(W)`` and ``post.averaging_kernel(W)`` give the
  posterior of the aggregates (a total per month, the mean over a region).

A sum is right for extensive quantities (a total, the columns of a Jacobian),
a mean for intensive ones (a flux, a concentration). A weighted sum, such as
fluxes per area to a total mass, is the sum times a diagonal of the weights:
``fips.aggregate.sum(space, by="time") @ Operator(scipy.sparse.diags(w),
space, space)``.

:func:`offset_block` uses the same groups for a block of offsets in the state:
one unknown offset per group of observations (a site's background in a month),
with the forward block that adds each to its observations.

The groups are given by ``by``, one key or a list of keys, each of which gives
every label of a block a group label on one level:

- a level name: the labels' values on that level;
- a :class:`pandas.Grouper` with a ``level`` and a ``freq``: that (datetime)
  level binned at the frequency, labeled by the start of each bin, as pandas
  bins it (``pd.Grouper(level="obs_time", freq="1D")``);
- a :class:`pandas.IntervalIndex` named after a level: that level binned into
  the intervals, labeled by the left edge of each;
- a :class:`pandas.Series` that maps the values of a level (its index, named
  after the level; an IntervalIndex maps the intervals) to group labels (its
  values), on a level named after the Series (or, unnamed, after the level it
  maps): sites to networks, cells to regions.

The group labels of every key together label the groups. A label a key cannot
place (outside every interval, missing from a mapping) raises.
"""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fips.linalg.base import _adopt
from fips.operator import Operator
from fips.space import Space, _listed, _sorted
from fips.vector import Vector

__all__ = ["sum", "mean", "offset_block"]


def _level(index: pd.Index, level: Hashable, block: str) -> pd.Index:
    """Return the values of `level` in the labels of `block`, or raise naming its levels."""
    if level not in index.names:
        raise ValueError(
            f"Block {block!r} has no level {level!r}; its levels are "
            f"{list(index.names)}."
        )
    return index.get_level_values(list(index.names).index(level))


def _key_labels(index: pd.Index, key: Any, block: str) -> pd.Index:
    """
    Return the group label one key gives each label of a block, as an Index named after its level.

    Raises
    ------
    TypeError
        If `key` is not one of the kinds the module docstring lists.
    ValueError
        If the block lacks the key's level, or the key cannot place a label.
    """
    if isinstance(key, str):
        return _level(index, key, block)
    if isinstance(key, pd.Grouper):
        level = key.level
        if not isinstance(level, str) or key.freq is None:
            raise ValueError(
                "A pandas Grouper in `by` needs a `level` (by name) and a "
                "`freq`; a level on its own is its name."
            )
        values = _level(index, level, block)
        # pandas numbers the bins of one key in time order, empty bins
        # included (as size() lists them), whatever the order of the labels:
        # bin the labels sorted, then put the bins back in the labels' order
        order = np.argsort(values, kind="stable")
        groups = pd.Series(np.arange(len(values)), index=values[order]).groupby(key)
        bins = groups.size().index
        labels = np.empty(len(values), dtype=np.intp)
        labels[order] = groups.ngroup().to_numpy()
        return pd.Index(bins.take(labels), name=level)
    if isinstance(key, pd.IntervalIndex):
        if key.name is None:
            raise ValueError(
                "An IntervalIndex in `by` must be named after the level it bins "
                '(bins.rename("obs_time")).'
            )
        return _mapped(index, key, pd.Index(key.left), key.name, block)
    if isinstance(key, pd.Series):
        if key.index.name is None:
            raise ValueError(
                "A Series in `by` maps the values of a level: name its index "
                "after that level."
            )
        if not key.index.is_unique:
            raise ValueError(
                f"The Series in `by` that maps level {key.index.name!r} has "
                "duplicate entries in its index."
            )
        name = key.index.name if key.name is None else key.name
        return _mapped(index, key.index, pd.Index(key.array), name, block)
    raise TypeError(
        "Each key in `by` is a level name, a pandas Grouper, an IntervalIndex "
        f"or a Series, not {type(key).__name__}."
    )


def _mapped(
    index: pd.Index, lookup: pd.Index, to: pd.Index, name: Hashable, block: str
) -> pd.Index:
    """
    Return the entry of `to` at the position of each label's value in `lookup`.

    `lookup`'s name is the level read; the result is named `name`.

    Raises
    ------
    ValueError
        If a value is not in `lookup` (outside every interval of an
        IntervalIndex).
    """
    values = _level(index, lookup.name, block)
    if isinstance(lookup, pd.IntervalIndex):
        subtype = getattr(lookup.dtype, "subtype", None)
        # pandas 2 cannot look up datetimes of one unit in intervals of another
        if (
            subtype is not None
            and values.dtype != subtype
            and values.dtype.kind == "M"
            and np.dtype(subtype).kind == "M"
        ):
            values = values.astype(subtype)
    positions = lookup.get_indexer(values)
    if (positions < 0).any():
        where = (
            "outside every interval"
            if isinstance(lookup, pd.IntervalIndex)
            else "not in the mapping"
        )
        raise ValueError(
            f"Labels of block {block!r} with a {lookup.name!r} {where}: "
            f"{_listed(values[positions < 0].unique())}."
        )
    return pd.Index(to.take(positions), name=name)


def _groups(
    blocks: Sequence[tuple[str, pd.Index]], by: Any
) -> tuple[np.ndarray, pd.Index]:
    """
    Return the group of each label of `blocks` (their labels in order) and the groups, sorted.

    Raises
    ------
    ValueError
        If `by` is empty, two keys label the same level, or a key cannot
        place a label.
    """
    keys = list(by) if isinstance(by, (list, tuple)) else [by]
    if not keys:
        raise ValueError("`by` needs at least one key.")
    columns = []
    for key in keys:
        parts = [_key_labels(index, key, name) for name, index in blocks]
        columns.append(parts[0].append(parts[1:]) if len(parts) > 1 else parts[0])
    names = [column.name for column in columns]
    if len(set(names)) < len(names):
        raise ValueError(f"Two keys in `by` label the same level: {names}.")
    labels = columns[0] if len(columns) == 1 else pd.MultiIndex.from_arrays(columns)
    groups = _sorted(labels.unique())
    return groups.get_indexer(labels), groups


def _chosen(space: Space, blocks: "str | Sequence[str] | None") -> list[str]:
    """Return the names of `blocks` (every block of `space` by default), checked."""
    if blocks is None:
        return list(space.names)
    chosen = [blocks] if isinstance(blocks, str) else list(blocks)
    unknown = [name for name in chosen if name not in space]
    if unknown:
        raise ValueError(
            f"No block named {_listed(unknown)}; the blocks are {_listed(space.names)}."
        )
    if not chosen:
        raise ValueError("`blocks` names no block.")
    return chosen


def _aggregation(
    space: Space, by: Any, blocks: "str | Sequence[str] | None", average: bool
) -> Operator:
    """Return the sum (or the mean, if `average`) of each group of labels, block by block."""
    if not isinstance(space, Space):
        raise TypeError(f"space must be a Space, got {type(space).__name__}.")
    chosen = _chosen(space, blocks)
    rows, cols, weights = [], [], []
    labels: dict[str, pd.Index] = {}
    start = 0
    for name in space.names:
        index = space.select(name)
        if name in chosen:
            codes, groups = _groups([(name, index)], by)
            counts = np.bincount(codes, minlength=len(groups))
            weight = 1.0 / counts[codes] if average else np.ones(len(codes))
        else:  # passed through, one row per label
            codes, groups, weight = np.arange(len(index)), index, np.ones(len(index))
        rows.append(start + codes)
        cols.append(space.slice(name).start + np.arange(len(index)))
        weights.append(weight)
        labels[name] = groups
        start += len(groups)
    matrix = sp.csr_array(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(start, len(space)),
    )
    return Operator(_adopt(matrix), Space(labels), space)


def sum(space: Space, by: Any, blocks: "str | Sequence[str] | None" = None) -> Operator:
    """
    Return the operator that sums the labels of `space` by group.

    Parameters
    ----------
    space : Space
        The labels to sum: the state's or the observations'.
    by : key or list of keys
        The groups: level names, a :class:`pandas.Grouper` with a ``level`` and
        a ``freq``, an :class:`pandas.IntervalIndex` named after a level, or a
        :class:`pandas.Series` that maps a level's values to groups (see
        :mod:`fips.aggregate`). Every block summed must have the levels the
        keys read.
    blocks : str or list of str, optional
        The blocks to sum, each on its own; every block by default. A block
        not named is passed through, one row for each of its labels.

    Returns
    -------
    Operator
        :math:`W`, sparse, with one row per group (ones where a label is in
        the group) and `space` as its columns. Its rows are a space of the
        same blocks, each labeled by its groups in sorted order on the
        levels the keys name (a block passed through keeps its labels).

    Raises
    ------
    TypeError
        If `space` is not a Space, or a key is of another kind.
    ValueError
        If `by` is empty, a block summed lacks a level a key reads, a key
        cannot place a label (outside every interval, missing from a
        mapping), two keys label one level, or `blocks` names no block of
        `space`.

    See Also
    --------
    mean : The operator that averages by group.
    fips.Posterior.cov : The posterior covariance of aggregates of the state.
    fips.Problem.aggregate : The problem of aggregated observations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> cells = pd.MultiIndex.from_product(
    ...     [pd.date_range("2020-01-01", periods=2, freq="MS"), ["a", "b", "c"]],
    ...     names=["time", "cell"],
    ... )
    >>> flux = fips.Vector.from_series(
    ...     pd.Series(np.arange(6.0), index=cells), block="flux"
    ... )
    >>> W = fips.aggregate.sum(flux.space, by="time")
    >>> W.rows
    Space(flux: 2 x ['time'])
    >>> (W @ flux).values
    array([ 3., 12.])
    """
    return _aggregation(space, by, blocks, average=False)


def mean(
    space: Space, by: Any, blocks: "str | Sequence[str] | None" = None
) -> Operator:
    """
    Return the operator that averages the labels of `space` by group.

    Parameters
    ----------
    space : Space
        The labels to average: the observations' or the state's.
    by : key or list of keys
        The groups, as :func:`sum` takes them.
    blocks : str or list of str, optional
        The blocks to average, each on its own; every block by default. A
        block not named is passed through, one row for each of its labels.

    Returns
    -------
    Operator
        :math:`W`, sparse, with one row per group (``1/n`` for each of its
        ``n`` labels) and `space` as its columns, its rows labeled as
        :func:`sum` labels them. :math:`W S W^T` divides the variance of
        independent errors by ``n``.

    Raises
    ------
    TypeError
        If `space` is not a Space, or a key is of another kind.
    ValueError
        As :func:`sum`.

    See Also
    --------
    sum : The operator that sums by group.
    fips.Problem.aggregate : The problem of aggregated observations.

    Examples
    --------
    Hourly observations at two sites, to daily means at each:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> labels = pd.MultiIndex.from_product(
    ...     [["A", "B"], pd.date_range("2020-01-01", periods=4, freq="12h")],
    ...     names=["site", "obs_time"],
    ... )
    >>> obs = fips.Vector.from_series(
    ...     pd.Series(np.arange(8.0), index=labels), block="co2"
    ... )
    >>> W = fips.aggregate.mean(
    ...     obs.space, by=["site", pd.Grouper(level="obs_time", freq="1D")]
    ... )
    >>> W.rows.select("co2")
    MultiIndex([('A', '2020-01-01'),
                ('A', '2020-01-02'),
                ('B', '2020-01-01'),
                ('B', '2020-01-02')],
               names=['site', 'obs_time'])
    >>> (W @ obs).values
    array([0.5, 2.5, 4.5, 6.5])
    """
    return _aggregation(space, by, blocks, average=True)


def offset_block(
    obs_space: Space,
    by: Any,
    block: str = "offset",
    blocks: "str | Sequence[str] | None" = None,
) -> tuple[Vector, Operator]:
    """
    Return a block of offsets for the state: one per group of observations, and the map to them.

    An offset is an unknown that adds to each observation of its group (the
    background at a site in a month, a calibration bias of a network): it is
    solved for with the state, as a block of its own, instead of being fixed
    (the problem's ``offset``) or left in the model-data mismatch. Its prior
    and its covariance are the caller's: the offsets' labels are the groups,
    so a covariance model on their levels correlates them (an
    ``Exponential("obs_time", "30D")`` between the months of one site).

    Parameters
    ----------
    obs_space : Space
        The observations.
    by : key or list of keys
        The groups, as :func:`sum` takes them: one offset for each group of
        observations that has any (``[bins, "site"]`` for one per interval and
        site, with ``bins`` an IntervalIndex named after the time level).
    block : str, default "offset"
        The name of the offsets' block in the state.
    blocks : str or list of str, optional
        The blocks of observations the offsets apply to, every block by
        default; their observations share an offset when their groups have
        the same labels. Observations of another block get none. For separate
        offsets per block, call this once per block with another `block`
        name.

    Returns
    -------
    prior : Vector
        Zeros on the offsets' space: one block, `block`, labeled by the
        groups (sorted) on the levels the keys name.
    forward : Operator
        The forward block, from the offsets (its columns) to `obs_space` (its
        rows): 1 where an observation has that offset, 0 elsewhere. Sparse.
        It is the transpose of :func:`sum`'s operator on the same groups.

    Raises
    ------
    TypeError
        If `obs_space` is not a Space, or a key is of another kind.
    ValueError
        As :func:`sum`.

    See Also
    --------
    sum : The aggregation on the same groups.

    Examples
    --------
    One offset per site and month, for hourly observations at two sites:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> labels = pd.MultiIndex.from_product(
    ...     [["A", "B"], pd.to_datetime(["2020-01-10", "2020-01-20", "2020-02-05"])],
    ...     names=["site", "obs_time"],
    ... )
    >>> obs = fips.Vector.from_series(pd.Series(400.0, index=labels), block="co2")
    >>> months = pd.interval_range(
    ...     pd.Timestamp("2020-01-01"),
    ...     periods=2,
    ...     freq="MS",
    ...     closed="left",
    ...     name="obs_time",
    ... )
    >>> prior, forward = fips.offset_block(obs.space, by=[months, "site"])
    >>> prior.space
    Space(offset: 4 x ['obs_time', 'site'])
    >>> forward.dense()
    array([[1., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 1., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 0., 1.]])

    The offsets go in the state beside the fluxes, with a prior covariance of
    their own:

    .. code-block:: python

        prior = fips.Vector.from_blocks({"flux": flux_prior, "offset": prior})
        H = fips.Operator.from_blocks(
            {("co2", "flux"): H_flux, ("co2", "offset"): forward["co2", "offset"]}
        )
        prior_cov = fips.Covariance.block_diag(
            flux=flux_cov, offset=fips.Independent().with_std(0.5)
        )
    """
    if not isinstance(obs_space, Space):
        raise TypeError(f"obs_space must be a Space, got {type(obs_space).__name__}.")
    chosen = _chosen(obs_space, blocks)
    codes, groups = _groups([(name, obs_space.select(name)) for name in chosen], by)
    rows = np.concatenate(
        [
            obs_space.slice(name).start + np.arange(len(obs_space.select(name)))
            for name in chosen
        ]
    )
    offsets = Space({block: groups})
    matrix = sp.csr_array(
        (np.ones(len(rows)), (rows, codes)), shape=(len(obs_space), len(groups))
    )
    prior = Vector(np.zeros(len(groups)), offsets)
    return prior, Operator(_adopt(matrix), obs_space, offsets)
