"""
Spaces: the labels of a state, of the observations, or of a matrix's axis.

A :class:`Space` is an ordered set of named blocks, each labeled by its own
pandas index. Together they make one :class:`pandas.MultiIndex` with a
``block`` level first. Blocks may have different levels: a level a block does
not have is missing (NaN) on that block's rows. This module is the one place
that pads those levels and strips them again.
"""

from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from functools import cached_property

import numpy as np
import pandas as pd

_SHOWN = 5
"""How many labels an error message lists."""


def _rounded(index: pd.Index, decimals: int) -> pd.Index:
    """Return `index` with each float level rounded to `decimals` places; other levels as they are."""

    def level(values: pd.Index) -> pd.Index:
        """Return one level, rounded if it holds floats."""
        if pd.api.types.is_float_dtype(values):
            return pd.Index(np.round(values.to_numpy(), decimals), name=values.name)
        return values

    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_arrays(
            [level(index.get_level_values(i)) for i in range(index.nlevels)],
            names=index.names,
        )
    return level(index)


def _plain(label):
    """Return ``label`` with NumPy scalars as Python ones, so it prints as typed."""
    if isinstance(label, tuple):
        return tuple(_plain(part) for part in label)
    return label.item() if isinstance(label, np.generic) else label


def _listed(labels: Iterable) -> str:
    """Return the first few of ``labels`` as text, saying how many were left out."""
    labels = list(labels)
    text = ", ".join(repr(_plain(label)) for label in labels[:_SHOWN])
    if len(labels) > _SHOWN:
        text += f" and {len(labels) - _SHOWN} more"
    return text


def _first_duplicates(index: pd.Index) -> np.ndarray:
    """Return the first position of each label that appears more than once in ``index``."""
    return np.flatnonzero(
        index.duplicated(keep=False) & ~index.duplicated(keep="first")
    )


def _has_missing(index: pd.Index) -> bool:
    """Return True if any label of ``index`` (on any level) is missing."""
    if isinstance(index, pd.MultiIndex):
        return any((codes == -1).any() for codes in index.codes)
    return bool(index.hasnans)


def _sorted(values: pd.Index) -> pd.Index:
    """Return the unique ``values`` sorted, or as they are when they do not sort (mixed types)."""
    try:
        return values.sort_values()
    except TypeError:
        return values


def _checked_block(name: str, index: pd.Index) -> pd.Index:
    """Return ``index``, after checking that it can label the block ``name``."""
    if not isinstance(index, pd.Index):
        raise TypeError(
            f"The labels of block {name!r} must be a pandas Index, got "
            f"{type(index).__name__}."
        )
    if isinstance(index, pd.MultiIndex) and index.nlevels == 1:
        index = index.get_level_values(
            0
        )  # one level is a plain Index, as from_index gives
    if len(index) == 0:
        raise ValueError(f"Block {name!r} has no labels.")
    if None in index.names:
        raise ValueError(f"Every level of block {name!r} must be named.")
    if "block" in index.names:
        raise ValueError(
            f"Block {name!r} has a 'block' level; the block's name is its key."
        )
    if _has_missing(index):
        raise ValueError(
            f"Block {name!r} has missing (NaN) labels. A level a block does not "
            "have is missing on its rows, so a block's own labels cannot be."
        )
    firsts = _first_duplicates(index)
    if len(firsts):
        raise ValueError(
            f"Block {name!r} has duplicate labels: {_listed(index[firsts])}. "
            "Each label must be unique."
        )
    return index


class Space:
    """
    An ordered set of named blocks, each labeled by its own pandas index.

    A space labels a state vector, the observations, or one axis of a
    matrix. Each block keeps the index it was given, with its own levels and
    dtypes; :attr:`index` puts them together as one
    :class:`pandas.MultiIndex` with a ``block`` level first. A level that one
    block has and another does not is missing (NaN) on the other block's rows.
    The labels are never changed: they are not converted to numbers and not
    rounded (see :meth:`round_labels`).

    Parameters
    ----------
    blocks : mapping of str to pd.Index
        The blocks in order, each name mapped to its labels. A name that is
        not a string is converted to one. Every level must be named, no level
        may be called ``block``, and a block's labels must be unique and not
        missing.
    levels : sequence of str, optional
        The order of the levels after ``block`` in :attr:`index`: every level
        of every block, once. By default, the order in which they first
        appear.

    Raises
    ------
    ValueError
        If two blocks have the same name, a block has no labels, an unnamed
        level, a ``block`` level, a missing label or a label twice, or
        `levels` does not name each level once.

    See Also
    --------
    Vector : Values on a space.

    Examples
    --------
    >>> import pandas as pd
    >>> from fips import Space
    >>> space = Space(
    ...     {
    ...         "flux": pd.Index([1, 2], name="cell"),
    ...         "bias": pd.Index(["A"], name="site"),
    ...     }
    ... )
    >>> space.index
    MultiIndex([('flux', 1.0, nan),
                ('flux', 2.0, nan),
                ('bias', nan, 'A')],
               names=['block', 'cell', 'site'])

    The padding shows ``cell`` as floats, but the level keeps its integers,
    and a block comes back as it was given:

    >>> space.index.levels[1]
    Index([1, 2], dtype='int64', name='cell')
    >>> space.select("flux")
    Index([1, 2], dtype='int64', name='cell')
    >>> space.slice("bias")
    slice(2, 3, None)
    """

    def __init__(
        self,
        blocks: Mapping[str, pd.Index],
        levels: Sequence[Hashable] | None = None,
    ):
        checked: dict[str, pd.Index] = {}
        for key, index in blocks.items():
            name = str(key)
            if name in checked:
                raise ValueError(f"Two blocks are named {name!r}.")
            checked[name] = _checked_block(name, index)
        if not checked:
            raise ValueError("A space needs at least one block.")
        self._blocks = checked

        seen: list[Hashable] = list(
            dict.fromkeys(n for index in checked.values() for n in index.names)
        )
        order = seen if levels is None else list(levels)
        if len(order) != len(seen) or set(order) != set(seen):
            raise ValueError(
                f"levels must name each level of the blocks once: {seen}, got {order}."
            )
        self._levels = tuple(order)

    @classmethod
    def from_index(cls, index: pd.Index) -> "Space":
        """
        Build the space an index with a ``block`` level describes.

        Each block gets the levels it has labels on; a level that is missing
        (NaN) on every row of a block is padding and is dropped. A block's
        labels keep the order of its rows in `index`, and the blocks the
        order in which they first appear. The levels keep their order in
        `index`.

        Parameters
        ----------
        index : pd.MultiIndex
            Labels with a ``block`` level, such as :attr:`index` or the index
            of a :class:`~fips.Vector`.

        Returns
        -------
        Space
            The space whose blocks the index holds.

        Raises
        ------
        ValueError
            If `index` has no ``block`` level, a row has no block name, a
            block has no other level, or a block has labels on a level for
            some of its rows and not for others.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Space
        >>> index = pd.MultiIndex.from_tuples(
        ...     [("flux", 1.0, None), ("flux", 2.0, None), ("bias", None, "A")],
        ...     names=["block", "cell", "site"],
        ... )
        >>> Space.from_index(index).select("bias").tolist()
        ['A']
        """
        if not isinstance(index, pd.MultiIndex) or "block" not in index.names:
            raise ValueError("The index must be a MultiIndex with a 'block' level.")
        at = list(index.names).index("block")
        block_codes = index.codes[at]
        if (block_codes == -1).any():
            raise ValueError("Every label must name its block; some have none.")
        others = [i for i in range(index.nlevels) if i != at]

        _, firsts = np.unique(block_codes, return_index=True)
        blocks: dict[str, pd.Index] = {}
        used: set = set()
        for code in block_codes[np.sort(firsts)]:
            name = str(index.levels[at][code])
            if name in blocks:  # two block labels with one string, as 1 and "1"
                raise ValueError(f"Two blocks are named {name!r}.")
            rows = np.flatnonzero(block_codes == code)
            arrays, names = [], []
            for i in others:
                codes = index.codes[i][rows]
                missing = codes == -1
                if missing.all():  # padding: a level this block does not have
                    continue
                if missing.any():
                    raise ValueError(
                        f"Block {name!r} has labels on level {index.names[i]!r} "
                        f"for some of its rows and not for others "
                        f"({int(missing.sum())} of {len(rows)} missing)."
                    )
                arrays.append(index.levels[i].take(codes))
                names.append(index.names[i])
                used.add(index.names[i])
            if not arrays:
                raise ValueError(f"Block {name!r} has no level besides 'block'.")
            if len(arrays) == 1:
                blocks[name] = pd.Index(arrays[0], name=names[0])
            else:
                blocks[name] = pd.MultiIndex.from_arrays(arrays, names=names)
        levels = [index.names[i] for i in others if index.names[i] in used]
        return cls(blocks, levels=levels)

    @classmethod
    def _regroup(cls, index: pd.Index) -> tuple["Space", np.ndarray]:
        """Return the space of `index` (:meth:`from_index`), and the order that puts the rows of each block together."""
        space = cls.from_index(index)
        rank, _ = pd.factorize(index.get_level_values("block").astype(str))
        return space, np.argsort(rank, kind="stable")

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the blocks, in order."""
        return tuple(self._blocks)

    @property
    def levels(self) -> tuple[Hashable, ...]:
        """The names of the levels after ``block`` in :attr:`index`, in order."""
        return self._levels

    @cached_property
    def offsets(self) -> np.ndarray:
        """
        Where each block starts in :attr:`index`, and the total length last.

        Block ``k`` is ``offsets[k]:offsets[k + 1]``.
        """
        sizes = [len(index) for index in self._blocks.values()]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(np.intp)

    def slice(self, block: str) -> slice:
        """
        Return the positions of `block` in :attr:`index`, as a slice.

        Parameters
        ----------
        block : str
            The name of a block; a name that is not a string is compared as
            one, as the space keeps it (``0`` is block ``"0"``).

        Returns
        -------
        slice
            ``index[space.slice(block)]`` are the block's labels.
        """
        k = self._position(block)
        return slice(int(self.offsets[k]), int(self.offsets[k + 1]))

    def select(self, block: str) -> pd.Index:
        """
        Return the labels of `block`, as the block was given.

        Parameters
        ----------
        block : str
            The name of a block; a name that is not a string is compared as
            one (``0`` is block ``"0"``).

        Returns
        -------
        pd.Index
            The block's labels, with its own levels and dtypes and no
            ``block`` level.

        Raises
        ------
        KeyError
            If the space has no block of that name.
        """
        return self._blocks[self.names[self._position(block)]]

    @cached_property
    def index(self) -> pd.MultiIndex:
        """
        The labels of every block as one MultiIndex, with the ``block`` level first.

        A level a block does not have is missing (NaN) on its rows; each
        level keeps its dtype (an integer level stays integer).
        """
        # Built from levels and codes, so that a block lacking a level gets
        # code -1 (missing) and the level keeps its dtype. Each level is
        # sorted, as MultiIndex.from_arrays makes it: pandas 2.0's sort_index
        # turns a missing code into an invalid cast when it re-sorts a level.
        blocks = list(self._blocks.values())
        in_order = pd.Index(list(self._blocks), name="block")
        names = _sorted(in_order)
        level_values = [names]
        codes = [np.repeat(names.get_indexer(in_order), [len(b) for b in blocks])]
        for name in self._levels:
            # Each block's labels on this level, by block position (None: lacks it)
            labels = [
                b.get_level_values(list(b.names).index(name))
                if name in b.names
                else None
                for b in blocks
            ]
            having = [values for values in labels if values is not None]
            values = _sorted(having[0].append(having[1:]).unique())
            level_codes = np.full(len(self), -1, dtype=np.intp)
            for k, block_labels in enumerate(labels):
                if block_labels is not None:
                    start, stop = self.offsets[k], self.offsets[k + 1]
                    level_codes[start:stop] = values.get_indexer(block_labels)
            level_values.append(values)
            codes.append(level_codes)
        return pd.MultiIndex(
            # pandas-stubs take levels as sequences that are not str (an Index
            # does not match that protocol) and codes as int sequences, not
            # arrays; pandas takes both
            # pyrefly: ignore[bad-argument-type]
            levels=level_values,
            # pyrefly: ignore[bad-argument-type]
            codes=codes,
            names=["block", *self._levels],
            verify_integrity=False,
        )

    def __len__(self) -> int:
        """Return the number of labels, over every block."""
        return int(self.offsets[-1])

    def __iter__(self) -> Iterator[str]:
        """Iterate over the block names."""
        return iter(self._blocks)

    def __contains__(self, block: object) -> bool:
        """Return True if the space has a block of that name (compared as a string, as the names are kept)."""
        return str(block) in self._blocks

    def __eq__(self, other: object) -> bool:
        """Return True if `other` has the same blocks, in the same order, with equal labels."""
        if not isinstance(other, Space):
            return NotImplemented
        return self.names == other.names and all(
            mine.equals(other._blocks[name]) and mine.names == other._blocks[name].names
            for name, mine in self._blocks.items()
        )

    def __repr__(self) -> str:
        """Return the blocks with their sizes and levels."""
        blocks = ", ".join(
            f"{name}: {len(index)} x {list(index.names)}"
            for name, index in self._blocks.items()
        )
        return f"Space({blocks})"

    def __getstate__(self) -> dict:
        """Pickle the blocks and the level order (the cached index is rebuilt)."""
        return {"blocks": self._blocks, "levels": self._levels}

    def __setstate__(self, state: dict) -> None:
        """Restore from :meth:`__getstate__`."""
        self._blocks = state["blocks"]
        self._levels = state["levels"]

    def _position(self, block: Hashable) -> int:
        """
        Return the position of `block` among the blocks, or raise a KeyError naming them.

        The name is compared as a string, as the space keeps it: block ``0``
        is block ``"0"``.
        """
        try:
            return self.names.index(str(block))
        except ValueError:
            raise KeyError(
                f"No block named {block!r}; the blocks are {_listed(self.names)}."
            ) from None

    def align(self, other: "Space") -> np.ndarray:
        """
        Return where each label of this space is in `other`: an exact join.

        The two spaces must have the same blocks and, in each block, the same
        labels; their order may differ, and so may the order of a block's
        levels. ``values_on_other[space.align(other)]`` puts values that
        follow `other` in this space's order.

        Parameters
        ----------
        other : Space
            The space to look the labels up in.

        Returns
        -------
        np.ndarray
            For each label of this space, in order, its position in
            ``other.index``.

        Raises
        ------
        ValueError
            If a block is in one space and not the other, a block has other
            levels in `other`, or a label is in one and not the other. The
            message names them (the first few of each).

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Space
        >>> a = Space({"flux": pd.Index([1, 2, 3], name="cell")})
        >>> b = Space({"flux": pd.Index([3, 1, 2], name="cell")})
        >>> a.align(b)
        array([1, 2, 0])
        """
        return self._align(other, "the other space", "The spaces")

    def _align(self, other: "Space", called: str, pair: str) -> np.ndarray:
        """
        Return :meth:`align`'s positions; the message calls `other` `called` and the two `pair`.

        Raises
        ------
        ValueError
            As :meth:`align` does, naming the spaces as given.
        """
        problems = []
        missing = [b for b in self.names if b not in other]
        extra = [b for b in other.names if b not in self]
        if missing:
            problems.append(f"blocks missing from {called}: {_listed(missing)}")
        if extra:
            problems.append(f"blocks only in {called}: {_listed(extra)}")

        positions = []
        for name in self.names:
            if name not in other:
                continue
            mine, theirs = self._blocks[name], other._blocks[name]
            if set(mine.names) != set(theirs.names):
                problems.append(
                    f"block {name!r} has levels {list(mine.names)} here and "
                    f"{list(theirs.names)} in {called}"
                )
                continue
            if isinstance(theirs, pd.MultiIndex) and theirs.names != mine.names:
                theirs = theirs.reorder_levels(list(mine.names))
            found = theirs.get_indexer(mine)
            if (found < 0).any():
                problems.append(
                    f"labels of block {name!r} missing from {called}: "
                    f"{_listed(mine[found < 0])}"
                )
            if len(theirs) > (found >= 0).sum():
                unmatched = np.ones(len(theirs), dtype=bool)
                unmatched[found[found >= 0]] = False
                problems.append(
                    f"labels of block {name!r} only in {called}: "
                    f"{_listed(theirs[unmatched])}"
                )
            positions.append(found + other.slice(name).start)
        if problems:
            raise ValueError(
                f"{pair} do not have the same labels: " + "; ".join(problems) + "."
            )
        return np.concatenate(positions)

    def _subset(self, sub: "Space", called: str) -> np.ndarray:
        """
        Return where each label of `sub` is in this space; `sub` must hold some of its labels.

        Every block of `sub` must be a block here, with the same levels (in
        any order), and every label of it must be among that block's. The
        message calls `sub` `called`.

        Raises
        ------
        ValueError
            If a block or a label of `sub` is not here, naming them.
        """
        problems, positions = [], []
        unknown = [b for b in sub.names if b not in self]
        if unknown:
            problems.append(
                f"blocks not in the space: {_listed(unknown)} "
                f"(its blocks are {_listed(self.names)})"
            )
        for name in sub.names:
            if name not in self:
                continue
            mine, theirs = self._blocks[name], sub._blocks[name]
            if set(mine.names) != set(theirs.names):
                problems.append(
                    f"block {name!r} has levels {list(theirs.names)} there and "
                    f"{list(mine.names)} in the space"
                )
                continue
            if isinstance(mine, pd.MultiIndex) and mine.names != theirs.names:
                mine = mine.reorder_levels(list(theirs.names))
            found = mine.get_indexer(theirs)
            if (found < 0).any():
                problems.append(
                    f"labels of block {name!r} not in the space: "
                    f"{_listed(theirs[found < 0])}"
                )
            positions.append(found + self.slice(name).start)
        if problems:
            raise ValueError(
                f"{called} is not part of the space: " + "; ".join(problems) + "."
            )
        return np.concatenate(positions)

    def round_labels(self, decimals: int) -> "Space":
        """
        Return the space with every float level rounded to `decimals` places.

        fips never rounds labels on its own. Round them on purpose when two
        sources give the same coordinate with different float noise (a grid
        written by two programs, say), so that their labels match.

        Parameters
        ----------
        decimals : int
            Number of decimal places. Levels that are not floats are kept.

        Returns
        -------
        Space
            The space with rounded labels.

        Raises
        ------
        ValueError
            If rounding makes two labels of a block equal. The message names
            the labels that merged.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Space
        >>> space = Space({"flux": pd.Index([40.7600001, 40.77], name="lat")})
        >>> space.round_labels(4).select("flux")
        Index([40.76, 40.77], dtype='float64', name='lat')
        """
        blocks = {}
        for name, index in self._blocks.items():
            rounded = _rounded(index, decimals)
            firsts = _first_duplicates(rounded)
            if len(firsts):
                merged = []
                for first in firsts[:_SHOWN]:
                    label = rounded[[first]]
                    sources = index[rounded.get_indexer_for(label)]
                    merged.append(f"{_listed(sources)} become {_listed(label)}")
                more = f"; {len(firsts) - _SHOWN} more" if len(firsts) > _SHOWN else ""
                raise ValueError(
                    f"Rounding to {decimals} decimals merges labels of block "
                    f"{name!r}: {'; '.join(merged)}{more}."
                )
            blocks[name] = rounded
        return Space(blocks, levels=self._levels)
