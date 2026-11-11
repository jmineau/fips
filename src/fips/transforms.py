r"""
Forward models that keep one block of the state positive: lognormal and softplus.

A flux that cannot be negative is solved as a hidden variable :math:`u` that
can be anything, and a transform :math:`T` that makes it positive:
the observations are :math:`z = H\, [T(u); \text{the other blocks}] + c + e`,
where :math:`H` is the forward operator on the whole state. The other blocks
(a bias, a background) pass through as they are. The state, its prior and its
covariance are in :math:`u`'s units, written by the user;
:meth:`LognormalForward.prior_for` and :meth:`SoftplusForward.prior_for`
give the prior on :math:`u` whose flux has a mean and a standard deviation
asked for. ``fips.solve(problem, "gauss_newton")`` finds the mode of
:math:`u`; :meth:`~LognormalForward.transform` maps it, or draws of the
posterior (:meth:`Posterior.sample <fips.Posterior.sample>`), to the flux.

- :class:`LognormalForward`: :math:`T(u) = r\, e^{u}`, a scale factor of a
  reference :math:`r` (an inventory): the uncertainty is a factor.
- :class:`SoftplusForward`: :math:`T(u) = s \log(1 + e^{u/s})`: the flux
  itself where it is large (:math:`T(u) \approx u` for :math:`u \gg s`),
  bending to stay above zero within about :math:`s` of it.
"""

import contextlib
import math
import numbers
from abc import abstractmethod
from functools import cached_property
from typing import overload

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from numpy.polynomial.legendre import leggauss
from scipy.special import expit

from fips.forward import ForwardModel
from fips.linalg import DiagonalOperator, LinearOperator
from fips.operator import Operator
from fips.space import Space, _listed
from fips.vector import Vector

_NEAR = leggauss(64)
"""Gauss-Legendre nodes and weights on [-1, 1] for the panel next to the bend, in the logarithm of the distance to it."""

_FAR = leggauss(128)
"""Gauss-Legendre nodes and weights on [-1, 1] for the rest of each side."""

_SPAN = 12.0
"""The standard normal is integrated over [-_SPAN, _SPAN] (its mass outside is 4e-33)."""

_NEAREST = 1e-14
"""The near panel starts this fraction of its width from the bend (what it leaves out is below roundoff)."""

_CHUNK = 2048
"""The cells whose softplus moments are computed at once (each holds 384 nodes)."""

_MATCHED = 1e-12
"""The relative error of the mean and the standard deviation at which the softplus prior stops."""

_ACCEPTED = 1e-9
"""The relative error above which the softplus prior raises instead."""


def _on_block(
    value: "Vector | pd.Series | float", labels: Space, what: str
) -> np.ndarray:
    """
    Return `value` on the block `labels` (a space of one block), in its order: a number for every label, a Series by its labels, a Vector by its space.

    Raises
    ------
    TypeError
        If `value` is an array or anything else without labels.
    ValueError
        If its labels are not exactly the block's, or a value is not finite.
    """
    (block,) = labels.names
    if isinstance(value, pd.Series):
        if "block" in value.index.names:
            value = Vector.from_series(value)
        else:
            value = Vector.from_series(value, block=block)
    if isinstance(value, Vector):
        values = value.values
        if value.space is not labels:
            values = values[
                labels._align(value.space, what, f"{what} and block {block!r}")
            ]
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        values = np.full(len(labels), float(value))
    else:
        raise TypeError(
            f"{what} is a number, a Series on the labels of block {block!r} or a "
            f"Vector on that block, not {type(value).__name__}: an array has no "
            "labels to say which value is whose."
        )
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(
            f"{what} is not finite at {int(bad.sum())} labels: "
            f"{_listed(labels.index[bad])}."
        )
    return values


def _negative(values: np.ndarray, labels: Space, what: str) -> None:
    """Raise if a value is negative, naming its labels."""
    bad = values < 0
    if bad.any():
        raise ValueError(
            f"{what} is negative at {int(bad.sum())} labels: "
            f"{_listed(labels.index[bad])}."
        )


class _Transformed(ForwardModel):
    r"""
    The machinery of a model :math:`F(x) = H\, [T(u); \text{the other blocks}]`, with :math:`T` element-wise on one block.

    A subclass gives :meth:`_values` (:math:`T` on the block's values),
    :meth:`_slopes` (:math:`T'`) and :meth:`_second_derivatives`
    (:math:`T''`, for Newton's steps), and its public methods.
    """

    def __init__(self, H: Operator, block: str | None):
        if not isinstance(H, Operator):
            raise TypeError(
                f"H is the forward operator on the whole state, an Operator, got "
                f"{type(H).__name__}."
            )
        cols = H.cols
        if block is None:
            if len(cols.names) != 1:
                raise ValueError(
                    f"The state has the blocks {_listed(cols.names)}: name the one "
                    "the transform is on with block=."
                )
            block = cols.names[0]
        block = cols.names[cols._position(block)]
        super().__init__(H.rows, cols)
        self._H = H
        self._block = block
        self._slice = cols.slice(block)
        self._labels = Space({block: cols.select(block)})

    @property
    def H(self) -> Operator:
        """The forward operator on the whole state, the transformed block and the others."""
        return self._H

    @property
    def block(self) -> str:
        """The name of the block the transform is on."""
        return self._block

    @abstractmethod
    def _values(self, u: np.ndarray) -> np.ndarray:
        """Return the transform of the block's values `u` (one row per label, in the order of the block)."""

    @abstractmethod
    def _slopes(self, u: np.ndarray) -> np.ndarray:
        """Return the derivative of the transform at the block's values `u`."""

    @abstractmethod
    def _second_derivatives(self, u: np.ndarray) -> np.ndarray:
        """Return the second derivative of the transform at the block's values `u`."""

    def _on_cols(self, x: Vector) -> np.ndarray:
        """Return the values of the state `x` in the order of the model's columns."""
        if not isinstance(x, Vector):
            raise TypeError(f"x must be a Vector, got {type(x).__name__}.")
        cols = self.cols
        if x.space is cols:
            return x.values
        return x.values[cols._align(x.space, "x", "x and the model's columns")]

    def _finite(self, u: np.ndarray, transformed: np.ndarray) -> np.ndarray:
        """
        Return `transformed`, the transform (or its slope) of the block's values `u`, if it is finite.

        Raises
        ------
        ValueError
            If it is not finite (it overflows) at some label, naming them.
        """
        bad = ~np.isfinite(transformed)
        if bad.any():
            # the labels' positions (of a state, or of any of its draws)
            rows = np.flatnonzero(bad if bad.ndim == 1 else bad.any(axis=1))
            raise ValueError(
                f"The transform of block {self._block!r} overflows at "
                f"{len(rows)} labels, where u is up to "
                f"{float(np.max(u[bad])):.4g}: {_listed(self._labels.index[rows])}. "
                "A state this large is far outside any prior on u; check the "
                "prior's units."
            )
        return transformed

    def _mapped(self, values: np.ndarray) -> np.ndarray:
        """
        Return the state `values` (in the columns' order; one column per draw if 2D) with the block transformed.

        Raises
        ------
        ValueError
            If the transform overflows at some label.
        """
        mapped = np.array(values, dtype=float)
        u = mapped[self._slice]
        mapped[self._slice] = self._finite(u, self._values(u))
        return mapped

    def _forward(self, x: Vector) -> Vector:
        """Return :math:`F(x)`."""
        values = self._mapped(self._on_cols(x))
        return Vector._adopt(np.asarray(self._H.data @ values, dtype=float), self.rows)

    def _jacobian(self, x: Vector) -> Operator:
        """Return :math:`H` times the diagonal of the slopes (1 off the block), in the form of :math:`H`."""
        values = self._on_cols(x)
        u = values[self._slice]
        slopes = np.ones(len(values))
        slopes[self._slice] = self._finite(u, self._slopes(u))
        form = self._H.data @ DiagonalOperator._adopt(slopes)
        return Operator(form, self.rows, self.cols, name=self._H.name)

    @cached_property
    def _H_T(self) -> LinearOperator:
        """The transpose of :math:`H`'s form, made once."""
        return self._H.data._transpose()

    def _gradient(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""Return :math:`J(x)^T v = T'(u) \circ (H^T v)` on the block (:math:`H^T v` off it), without forming the Jacobian."""
        gradient = np.asarray(self._H_T @ v, dtype=float)
        u = x[self._slice]
        gradient[self._slice] *= self._finite(u, self._slopes(u))
        return gradient

    def _curvature(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""
        Return the diagonal of :math:`\sum_k v_k \nabla^2 F_k(x)`: :math:`T''(u) \circ (H^T v)` on the block, zero off it.

        :math:`F_k = \sum_i H_{ki} T(u_i)` plus the other blocks' linear
        terms, so each :math:`\nabla^2 F_k` is the diagonal
        :math:`H_{ki} T''(u_i)` on the block.
        """
        curvature = np.zeros(len(x))
        u = x[self._slice]
        weighted = np.asarray(self._H_T @ v, dtype=float)[self._slice]
        curvature[self._slice] = self._finite(u, self._second_derivatives(u)) * weighted
        return curvature

    def _transform(self, x: "Vector | pd.DataFrame") -> "Vector | pd.DataFrame":
        """Return `x` (a state, or draws of it, one column each) with the block transformed, on its own labels."""
        cols = self.cols
        if isinstance(x, Vector):
            if x.space is cols:
                return Vector._adopt(self._mapped(x.values), cols, x.name)
            # cols[i] is x.space[positions[i]]
            positions = cols._align(x.space, "x", "x and the model's columns")
            mapped = np.empty(len(x))
            mapped[positions] = self._mapped(x.values[positions])
            return Vector._adopt(mapped, x.space, x.name)
        if isinstance(x, pd.DataFrame):
            # the rows in the order of their space (each block's together),
            # then in the order of the model's columns
            space, order = Space._regroup(x.index)
            positions = order[
                cols._align(
                    space, "the draws' rows", "The draws' rows and the model's columns"
                )
            ]
            mapped = np.empty(x.shape)
            mapped[positions] = self._mapped(x.to_numpy(dtype=float)[positions])
            return pd.DataFrame(mapped, index=x.index, columns=x.columns)
        raise TypeError(
            "x is a Vector (a state, such as post.estimate) or a DataFrame of "
            f"draws with the state's labels as rows (post.sample), not "
            f"{type(x).__name__}."
        )

    def __repr__(self) -> str:
        """Return the class, the block and the two spaces."""
        return (
            f"{type(self).__name__}(block: {self._block!r}, rows: {self.rows!r}, "
            f"cols: {self.cols!r})"
        )


class LognormalForward(_Transformed):
    r"""
    A forward model whose state on one block is the logarithm of a scale factor of a reference: the flux :math:`r\, e^{u}`.

    The observations are :math:`F(x) = H\, [r \circ e^{u}; \text{the other
    blocks}]`: :math:`H` is the forward operator on the whole state, the
    state on `block` is :math:`u`, the log of the factor that scales the
    reference :math:`r` (an inventory), and every other block (a bias)
    passes through as it is. The Jacobian is :math:`H` with each column of
    the block scaled by its flux, :math:`H \operatorname{diag}(r \circ
    e^{u}, 1)`, in the form of :math:`H` (sparse stays sparse).

    The prior and its covariance on the block are in log units: a standard
    deviation of 0.5 is a factor of :math:`e^{0.5} = 1.65` either way. A cell
    whose reference is zero has a zero flux whatever :math:`u` is, and its
    column of the Jacobian is zero: no observation moves it, so a source
    the reference has not got cannot appear (:class:`SoftplusForward` or the
    ``"bounded"`` method can). :meth:`prior_for` gives the prior on
    :math:`u` whose flux has a mean and a standard deviation;
    :meth:`transform` maps the estimate, or draws of the posterior, to the
    flux.

    Parameters
    ----------
    H : Operator
        The forward operator: its rows the observations' labels, its columns
        the whole state's (the block and any other).
    reference : Vector, Series or float
        The reference flux :math:`r` of each cell of the block, not negative:
        a Vector on the block (``inventory["flux"]``), a Series on its labels
        (without a ``block`` level), or one number for every cell.
    block : str, optional
        The block of the state that is :math:`u`. By default the state's one
        block.

    Raises
    ------
    TypeError
        If `H` is not an Operator, or `reference` is an array (it has no
        labels).
    ValueError
        If the state has several blocks and `block` is not given, `reference`
        does not have exactly the block's labels, or a reference is negative
        or not finite.
    KeyError
        If `block` is not a block of the state.

    See Also
    --------
    SoftplusForward : A flux that is additive away from zero.
    fips.ForwardModel : A forward model written by hand.

    Notes
    -----
    The mode :math:`\hat{u}` is the mean and the median of the Gaussian at
    the mode, so :math:`r\, e^{\hat{u}}` is the median of each cell's flux
    under it, below its mean :math:`r\, e^{\hat{u} + \sigma^2/2}` (with
    :math:`\sigma^2` the variance at the mode). A sum of lognormals has no
    closed form: draws of the total are ``model.transform(post.sample(n))``
    summed by draw. Where the data constrain sums of cells whose prior is
    wide, the Gaussian at the mode lies along the tangent of the curved set
    of states they allow, and the draws' totals run high: the sum of the flux
    at the mode is the better total (the user guide's "Fluxes that stay
    positive" measures both).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
    >>> observed = fips.Space({"ch4": pd.RangeIndex(3, name="obs_id")})
    >>> H = fips.Operator(
    ...     np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), observed, state
    ... )
    >>> inventory = pd.Series([2.0, 0.0], index=state.select("flux"))
    >>> model = fips.LognormalForward(H, inventory, block="flux")
    >>> u = fips.Vector([np.log(1.5), 3.0], state)
    >>> model(u).values  # cell b has no reference, so no flux
    array([3., 0., 3.])
    >>> model.transform(u).values
    array([3., 0.])
    >>> fips.taylortest(model, u)
    True
    """

    def __init__(
        self,
        H: Operator,
        reference: "Vector | pd.Series | float",
        block: str | None = None,
    ):
        super().__init__(H, block)
        values = _on_block(reference, self._labels, "reference")
        _negative(values, self._labels, "reference")
        values.flags.writeable = False
        self._reference = values

    @property
    def reference(self) -> Vector:
        r"""The reference flux :math:`r` of each cell of the block, on the block's labels."""
        return Vector._adopt(self._reference, self._labels, "reference")

    def _values(self, u: np.ndarray) -> np.ndarray:
        """Return :math:`r e^u`, zero where the reference is zero (whatever `u`)."""
        r = self._reference.reshape(-1, *([1] * (u.ndim - 1)))
        with np.errstate(over="ignore", invalid="ignore"):
            flux = r * np.exp(u)
        return np.where(r > 0, flux, 0.0)

    def _slopes(self, u: np.ndarray) -> np.ndarray:
        """Return the derivative of :math:`r e^u`, which is itself."""
        return self._values(u)

    def _second_derivatives(self, u: np.ndarray) -> np.ndarray:
        """Return the second derivative of :math:`r e^u`, itself again."""
        return self._values(u)

    def __call__(self, x: Vector) -> Vector:
        r"""
        Return the observations of the state `x`, without the offset: :math:`H\, [r \circ e^{u}; \text{the other blocks}]`.

        Parameters
        ----------
        x : Vector
            A state on the model's columns (in any order).

        Returns
        -------
        Vector
            :math:`F(x)`, on :attr:`rows`.

        Raises
        ------
        ValueError
            If `x` does not have the model's columns' labels, or the flux
            overflows (:math:`u` above about 700).
        """
        return self._forward(x)

    def jacobian(self, x: Vector) -> Operator:
        r"""
        Return the Jacobian at `x`: :math:`H \operatorname{diag}(r \circ e^{u}, 1)`.

        Each column of the block is scaled by its flux; the columns of the
        other blocks are those of :math:`H`. The form is that of :math:`H`.

        Parameters
        ----------
        x : Vector
            A state on the model's columns (in any order).

        Returns
        -------
        Operator
            :math:`J(x)`, on :attr:`rows` and :attr:`cols`.
        """
        return self._jacobian(x)

    @overload
    def transform(self, x: Vector) -> Vector:  # noqa: D418
        """Of a state: a Vector on its labels."""

    @overload
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: D418
        """Of draws: a DataFrame on their labels."""

    def transform(self, x: "Vector | pd.DataFrame") -> "Vector | pd.DataFrame":
        r"""
        Return the state with the block as the flux, :math:`r\, e^{u}`: of a state, or of each draw.

        The other blocks are as they are. For the estimate, the flux at the
        mode, the median of each cell's flux under the Gaussian at the mode
        (not its mean). For draws of the posterior (:meth:`Posterior.sample
        <fips.Posterior.sample>`, one column each), draws of the flux: the
        total of each column is a draw of the total (see the class's Notes
        for when they run high). Aggregating ``u`` first
        (``post.sample(n, W=W)``) and transforming that is not a total of
        the flux.

        Parameters
        ----------
        x : Vector or DataFrame
            A state on the model's columns (in any order), or draws with the
            state's labels as rows (the index of :attr:`Posterior.estimate
            <fips.Posterior.estimate>`) and one column per draw.

        Returns
        -------
        Vector or DataFrame
            The same labels (and columns) as `x`, the block transformed.

        Raises
        ------
        TypeError
            If `x` is neither a Vector nor a DataFrame.
        ValueError
            If its labels are not the model's columns', or the flux
            overflows.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
        >>> H = fips.Operator(
        ...     np.ones((1, 2)), fips.Space({"ch4": pd.RangeIndex(1, name="i")}), state
        ... )
        >>> model = fips.LognormalForward(H, 2.0)
        >>> draws = pd.DataFrame(
        ...     [[0.0, np.log(2.0)], [np.log(0.5), 0.0]], index=state.index
        ... )
        >>> model.transform(draws).sum().tolist()  # the total of each draw
        [3.0, 6.0]
        """
        return self._transform(x)

    def prior_for(
        self, mean: "Vector | pd.Series | float", std: "Vector | pd.Series | float"
    ) -> tuple[Vector, Vector]:
        r"""
        Return the prior on :math:`u` (its mean and standard deviation, each cell's) whose flux :math:`r\, e^{u}` has the mean and the standard deviation asked for.

        The flux is lognormal, with :math:`u \sim N(\mu, \sigma^2)`:
        :math:`\sigma^2 = \log(1 + s^2 / m^2)` and :math:`\mu = \log(m / r) -
        \sigma^2 / 2`, so that its mean is :math:`m` and its standard
        deviation :math:`s` exactly. With the reference as the mean (an
        inventory), :math:`\mu = -\sigma^2 / 2`: the prior median of the flux,
        :math:`r\, e^{\mu}`, is below the inventory, and its mean is the
        inventory. A prior of :math:`\mu = 0` instead makes the inventory the
        median, and the prior's total :math:`e^{\sigma^2/2}` times the
        inventory's (1.65 times for :math:`\sigma = 1`).

        The correlations are those of :math:`u`, not of the flux: build the
        covariance with them (``Exponential(...).with_std(sigma)``). Two cells
        whose :math:`u` correlate by :math:`\rho` have fluxes that correlate
        by :math:`(e^{\rho \sigma_i \sigma_j} - 1) / \sqrt{(e^{\sigma_i^2} -
        1)(e^{\sigma_j^2} - 1)}`, less than :math:`\rho` (0.38 for
        :math:`\rho = 0.5` and :math:`\sigma = 1`).

        Parameters
        ----------
        mean : Vector, Series or float
            The prior mean :math:`m` of each cell's flux, on the block's labels
            (or one number). Positive where the reference is; zero where it
            is zero.
        std : Vector, Series or float
            The prior standard deviation :math:`s` of each cell's flux, not
            negative; zero where the reference is zero.

        Returns
        -------
        mean, std : Vector
            :math:`\mu` and :math:`\sigma`, on the block's labels. A cell whose
            reference is zero gets 0 and 0: its flux is zero whatever its
            :math:`u`, and any prior there gives the other cells the same
            posterior.

        Raises
        ------
        TypeError
            If `mean` or `std` is an array.
        ValueError
            If their labels are not the block's, a value is negative or not
            finite, a mean is zero where the reference is not, or a mean or a
            standard deviation is not zero where the reference is.

        Examples
        --------
        A flux of mean 2 and standard deviation 1 in each cell, the reference
        its mean:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
        >>> H = fips.Operator(
        ...     np.ones((1, 2)), fips.Space({"ch4": pd.RangeIndex(1, name="i")}), state
        ... )
        >>> model = fips.LognormalForward(H, 2.0)
        >>> mu, sigma = model.prior_for(mean=2.0, std=1.0)
        >>> sigma.values.round(4)  # sqrt(log(1.25))
        array([0.4724, 0.4724])
        >>> mu.values.round(4)  # -sigma^2 / 2
        array([-0.1116, -0.1116])
        """
        labels = self._labels
        m = _on_block(mean, labels, "mean")
        s = _on_block(std, labels, "std")
        _negative(m, labels, "mean")
        _negative(s, labels, "std")
        r = self._reference
        zero = r == 0
        for values, what in ((m, "mean"), (s, "std")):
            bad = zero & (values != 0)
            if bad.any():
                raise ValueError(
                    f"The {what} is not zero at {int(bad.sum())} labels where the "
                    "reference is zero, whose flux is zero whatever u is: "
                    f"{_listed(labels.index[bad])}. A source the reference has "
                    "not got needs a reference there, or SoftplusForward."
                )
        bad = ~zero & (m == 0)
        if bad.any():
            raise ValueError(
                f"The mean is zero at {int(bad.sum())} labels whose reference is "
                f"not, and a lognormal flux is never zero: "
                f"{_listed(labels.index[bad])}."
            )
        safe = np.where(zero, 1.0, m)
        variance = np.where(zero, 0.0, np.log1p((s / safe) ** 2))
        mu = np.where(zero, 0.0, np.log(safe / np.where(zero, 1.0, r)) - variance / 2)
        return (
            Vector._adopt(mu, labels, "prior"),
            Vector._adopt(np.sqrt(variance), labels, "prior_std"),
        )


def _softplus_rule(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Return nodes :math:`\xi` and weights (one row per cell) for :math:`E[f(a + b \xi)]`, :math:`\xi \sim N(0, 1)`, for a function that bends at :math:`a + b \xi = 0`.

    Each side of the bend, :math:`\xi_0 = -a/b`, is one panel of
    Gauss-Legendre nodes in the logarithm of the distance to it, up to 1 (the
    bend's width, :math:`1/b`, can be far smaller), and one in the distance
    beyond, both within :math:`[-12, 12]`; the weights carry the normal
    density. Accurate to about 1e-14 for a smooth function on each side
    (the tests hold it to SciPy's adaptive ``quad``).
    """
    xi_0 = np.clip(-a / b, -_SPAN, _SPAN)[:, None]
    nodes, weights = [], []
    for side in (1.0, -1.0):
        width = (_SPAN - side * xi_0).ravel()  # each side's length, at least 0
        empty = width <= 0
        width = np.where(empty, 1.0, width)
        near = np.minimum(width, 1.0)[:, None]
        # the near panel: log s from log(near * _NEAREST) to log(near)
        x, w = _NEAR
        low, high = np.log(near * _NEAREST), np.log(near)
        s = np.exp((high + low) / 2 + (high - low) / 2 * x)
        nodes.append(xi_0 + side * s)
        weights.append((high - low) / 2 * w * s)
        # the far panel: s from near to the side's length
        x, w = _FAR
        far = width[:, None]
        s = (far + near) / 2 + (far - near) / 2 * x
        nodes.append(xi_0 + side * s)
        weights.append((far - near) / 2 * w * np.ones_like(s))
        for k in (-2, -1):
            weights[k] = np.where(empty[:, None], 0.0, weights[k])
    xi = np.concatenate(nodes, axis=1)
    weight = (
        np.concatenate(weights, axis=1) * np.exp(-(xi**2) / 2) / math.sqrt(2 * math.pi)
    )
    return xi, weight


def _softplus_moments(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Return the mean and standard deviation of :math:`\log(1 + e^t)`, :math:`t \sim N(a, b^2)`, and their derivatives by :math:`a` and :math:`b` (``k x 2 x 2``).

    The deviations are taken from :math:`f(a)` as :math:`f(t) - f(a) =
    \log(1 + \sigma(a)(e^{t - a} - 1))` (with :math:`\sigma` the logistic
    function), so a standard deviation far below the mean loses no digits.
    The cells are taken ``_CHUNK`` at a time.
    """
    parts = [
        _softplus_chunk(a[k : k + _CHUNK], b[k : k + _CHUNK])
        for k in range(0, len(a), _CHUNK)
    ] or [_softplus_chunk(a, b)]
    mean, std, jac = zip(*parts, strict=True)
    return np.concatenate(mean), np.concatenate(std), np.concatenate(jac)


def _softplus_chunk(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return :func:`_softplus_moments` of a few cells."""
    xi, w = _softplus_rule(a, b)
    step = b[:, None] * xi
    t = a[:, None] + step
    # f(t) - f(a) = log(1 + expit(a) (e^step - 1)): exact for a small step, and
    # wherever expit(a) <= 1/2; else step + g(t) - g(a), g(x) = log(1 + e^-x)
    near = (step <= 700.0) & ((a[:, None] <= 0.0) | (step >= -1.0))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        d = np.log1p(expit(a)[:, None] * np.expm1(np.minimum(step, 700.0)))
    d = np.where(
        near,
        d,
        step + np.logaddexp(0.0, -t) - np.logaddexp(0.0, -a)[:, None],
    )
    shift = (d * w).sum(axis=1)
    mean = np.logaddexp(0.0, a) + shift
    dev = d - shift[:, None]
    std = np.sqrt((dev**2 * w).sum(axis=1))
    g = expit(t)
    jac = np.empty((len(a), 2, 2))
    jac[:, 0, 0] = (g * w).sum(axis=1)
    jac[:, 0, 1] = (g * xi * w).sum(axis=1)
    safe = np.where(std > 0, std, 1.0)
    jac[:, 1, 0] = (dev * g * w).sum(axis=1) / safe
    jac[:, 1, 1] = (dev * g * xi * w).sum(axis=1) / safe
    return mean, std, jac


def _steps(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the solution of each cell's 2 x 2 system ``A x = b`` (``k x 2``), NaN for a cell whose matrix is singular."""
    try:
        return np.linalg.solve(A, b[:, :, None])[:, :, 0]
    except LinAlgError:  # some cell is singular: the others one by one
        x = np.full(b.shape, np.nan)
        for k in range(len(b)):
            with contextlib.suppress(LinAlgError):
                x[k] = np.linalg.solve(A[k], b[k])
        return x


def _softplus_match(
    m: np.ndarray, s: np.ndarray, cells: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Return :math:`a, b` with :math:`\log(1 + e^t)`, :math:`t \sim N(a, b^2)`, of mean `m` and standard deviation `s` (both positive).

    Newton's method on the logarithms of the two moments, in :math:`\alpha =
    a/b` and :math:`\beta = \log b` (where the flux is far above the bend,
    each moment is :math:`b` times a function of :math:`\alpha`, so the
    equations are nearly linear), from the better of the lognormal answer
    (right for :math:`m \ll 1`) and the additive one (:math:`a = m, b = s`,
    right for :math:`m \gg 1`). Each cell's step is halved until its error
    falls.

    Raises
    ------
    RuntimeError
        If some cell does not reach a relative error of ``_ACCEPTED`` (its
        Newton matrix singular, or its steps stalled), naming `cells`, the
        labels of the cells.
    """
    log_m, log_s = np.log(m), np.log(s)

    def errors(
        alpha: np.ndarray, beta: np.ndarray, k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the errors of the log moments of the cells `k`, and their derivatives by alpha and beta."""
        b = np.exp(beta)
        a = alpha * b
        mean, std, J = _softplus_moments(a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.stack([np.log(mean) - log_m[k], np.log(std) - log_s[k]], axis=1)
            # d/dalpha = b d/da; d/dbeta = a d/da + b d/db; divided by the moment
            dr = np.stack(
                [
                    J[:, :, 0] * b[:, None],
                    J[:, :, 0] * a[:, None] + J[:, :, 1] * b[:, None],
                ],
                axis=2,
            )
            dr = dr / np.stack([mean, std], axis=1)[:, :, None]
        r[~np.isfinite(r)] = np.inf
        return r, dr

    every = np.arange(len(m))
    variance = np.log1p((s / m) ** 2)
    lognormal = (np.log(m) - variance / 2) / np.sqrt(variance), 0.5 * np.log(variance)
    additive = m / s, np.log(s)
    r_log, dr_log = errors(*lognormal, every)
    r_add, dr_add = errors(*additive, every)
    use = np.abs(r_add).max(axis=1) < np.abs(r_log).max(axis=1)
    alpha = np.where(use, additive[0], lognormal[0])
    beta = np.where(use, additive[1], lognormal[1])
    r = np.where(use[:, None], r_add, r_log)
    dr = np.where(use[:, None, None], dr_add, dr_log)
    for _ in range(100):
        size = np.abs(r).max(axis=1)
        pending = np.flatnonzero(size > _MATCHED)
        if not len(pending):
            break
        with np.errstate(invalid="ignore"):
            step = _steps(dr[pending], -r[pending])
        step[~np.isfinite(step)] = 0.0
        fraction, moved = 1.0, np.zeros(len(pending), dtype=bool)
        for _ in range(40):  # halve each cell's step until its error falls
            left = np.flatnonzero(~moved)
            k = pending[left]
            new_alpha = alpha[k] + fraction * step[left, 0]
            new_beta = beta[k] + fraction * step[left, 1]
            new_r, new_dr = errors(new_alpha, new_beta, k)
            better = np.abs(new_r).max(axis=1) < size[k]
            alpha[k[better]], beta[k[better]] = new_alpha[better], new_beta[better]
            r[k[better]], dr[k[better]] = new_r[better], new_dr[better]
            moved[left[better]] = True
            if moved.all():
                break
            fraction /= 2
        if not moved.any():  # at roundoff
            break
    worst = np.abs(r).max(axis=1)
    failed = ~(worst <= _ACCEPTED)
    if failed.any():
        raise RuntimeError(
            f"The softplus prior did not match the mean and standard deviation in "
            f"{int(failed.sum())} cells (relative error up to {worst.max():.2g}), "
            f"such as a mean of {m[failed][0]:.4g} and a standard deviation of "
            f"{s[failed][0]:.4g} times the scale: {_listed(cells[failed])}."
        )
    b = np.exp(beta)
    return alpha * b, b


class SoftplusForward(_Transformed):
    r"""
    A forward model whose state on one block is a flux that bends to stay above zero: :math:`s \log(1 + e^{u/s})`.

    The observations are :math:`F(x) = H\, [s \log(1 + e^{u/s}); \text{the
    other blocks}]`: :math:`H` is the forward operator on the whole state,
    the state on `block` is :math:`u`, in the flux's units, and every other
    block passes through as it is. Where :math:`u` is several `scale` above
    zero the flux is :math:`u` (the problem is the linear one); where it is
    several below, the flux is :math:`s\, e^{u/s}`, near zero and positive.
    The Jacobian is :math:`H \operatorname{diag}(\operatorname{expit}(u/s),
    1)`, the logistic function on the block, in the form of :math:`H`. Both
    are computed without overflow for any :math:`u` (``np.logaddexp`` and
    ``scipy.special.expit``).

    `scale` is in the flux's units and says where the bend is: the transform
    is not the same in other units (:math:`\log(1 + e^u)` bends within 1 of
    zero, whatever 1 is), so it has no default. A scale near the prior's standard deviation of
    the flux in the cells near zero bends where the prior says the flux
    could be zero. :meth:`prior_for` gives the prior on :math:`u` whose flux
    has a mean and a standard deviation; :meth:`transform` maps the
    estimate, or draws of the posterior, to the flux.

    Parameters
    ----------
    H : Operator
        The forward operator: its rows the observations' labels, its columns
        the whole state's (the block and any other).
    scale : float
        The scale :math:`s` of the bend, positive, in the units of the
        transformed value: within a few :math:`s` of zero the value bends
        away from zero, and above that it is :math:`u`. It has no default,
        since no number is right in every unit.
    block : str, optional
        The block of the state that is :math:`u`. By default the state's one
        block.

    Raises
    ------
    TypeError
        If `H` is not an Operator.
    ValueError
        If the state has several blocks and `block` is not given, or `scale`
        is not a positive, finite number.
    KeyError
        If `block` is not a block of the state.

    See Also
    --------
    LognormalForward : A flux whose uncertainty is a factor of a reference.

    Notes
    -----
    The transform is monotone, so the mode :math:`\hat{u}` (the median of
    the Gaussian at the mode) maps to the median of each cell's flux under
    it, :meth:`transform` of the estimate. Draws of a total are
    ``model.transform(post.sample(n))`` summed by draw; near the bend they run
    high, as :class:`LognormalForward`'s do (see its Notes).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> state = fips.Space({"flux": pd.Index(["a", "b", "c"], name="cell")})
    >>> observed = fips.Space({"ch4": pd.RangeIndex(1, name="obs_id")})
    >>> H = fips.Operator(np.ones((1, 3)), observed, state)
    >>> model = fips.SoftplusForward(H, scale=0.1)
    >>> u = fips.Vector([5.0, 0.0, -1.0], state)
    >>> model.transform(u).values.round(5)  # u far above the scale is the flux
    array([5.     , 0.06931, 0.     ])
    >>> fips.taylortest(model, u)
    True
    """

    def __init__(self, H: Operator, scale: float, block: str | None = None):
        super().__init__(H, block)
        try:
            value = float(scale)
        except (TypeError, ValueError):
            raise ValueError(f"scale must be a number, got {scale!r}.") from None
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"scale must be positive and finite, got {value}.")
        self._scale = value

    @property
    def scale(self) -> float:
        """The scale of the bend, in the flux's units: the flux is ``scale * log(1 + exp(u / scale))``."""
        return self._scale

    def _values(self, u: np.ndarray) -> np.ndarray:
        r"""Return :math:`s \log(1 + e^{u/s})`."""
        s = self._scale
        return s * np.logaddexp(0.0, u / s)

    def _slopes(self, u: np.ndarray) -> np.ndarray:
        """Return the logistic function of :math:`u/s`."""
        return expit(u / self._scale)

    def _second_derivatives(self, u: np.ndarray) -> np.ndarray:
        r"""Return the derivative of the logistic function of :math:`u/s`, :math:`\sigma (1 - \sigma) / s`."""
        s = self._scale
        return expit(u / s) * expit(-u / s) / s  # no cancellation in 1 - sigma

    def __call__(self, x: Vector) -> Vector:
        r"""
        Return the observations of the state `x`, without the offset: :math:`H\, [s \log(1 + e^{u/s}); \text{the other blocks}]`.

        Parameters
        ----------
        x : Vector
            A state on the model's columns (in any order).

        Returns
        -------
        Vector
            :math:`F(x)`, on :attr:`rows`.

        Raises
        ------
        ValueError
            If `x` does not have the model's columns' labels.
        """
        return self._forward(x)

    def jacobian(self, x: Vector) -> Operator:
        r"""
        Return the Jacobian at `x`: :math:`H \operatorname{diag}(\operatorname{expit}(u/s), 1)`.

        Each column of the block is scaled by the slope of the transform,
        between 0 (far below zero) and 1 (far above); the columns of the other
        blocks are those of :math:`H`. The form is that of :math:`H`.

        Parameters
        ----------
        x : Vector
            A state on the model's columns (in any order).

        Returns
        -------
        Operator
            :math:`J(x)`, on :attr:`rows` and :attr:`cols`.
        """
        return self._jacobian(x)

    @overload
    def transform(self, x: Vector) -> Vector:  # noqa: D418
        """Of a state: a Vector on its labels."""

    @overload
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: D418
        """Of draws: a DataFrame on their labels."""

    def transform(self, x: "Vector | pd.DataFrame") -> "Vector | pd.DataFrame":
        r"""
        Return the state with the block as the flux, :math:`s \log(1 + e^{u/s})`: of a state, or of each draw.

        The other blocks are as they are. For the estimate, the flux at the
        mode, the median of each cell's flux under the Gaussian at the mode.
        For draws of the posterior (:meth:`Posterior.sample
        <fips.Posterior.sample>`, one column each), draws of the flux: the
        total of each column is a draw of the total (see the class's Notes
        for when they run high).

        Parameters
        ----------
        x : Vector or DataFrame
            A state on the model's columns (in any order), or draws with the
            state's labels as rows and one column per draw.

        Returns
        -------
        Vector or DataFrame
            The same labels (and columns) as `x`, the block transformed.

        Raises
        ------
        TypeError
            If `x` is neither a Vector nor a DataFrame.
        ValueError
            If its labels are not the model's columns'.
        """
        return self._transform(x)

    def prior_for(
        self, mean: "Vector | pd.Series | float", std: "Vector | pd.Series | float"
    ) -> tuple[Vector, Vector]:
        r"""
        Return the prior on :math:`u` (its mean and standard deviation, each cell's) whose flux :math:`s \log(1 + e^{u/s})` has the mean and the standard deviation asked for.

        There is no closed form: for :math:`u \sim N(\mu, \sigma^2)` the mean
        and the standard deviation of the flux are integrals, computed by
        quadrature (to about 1e-13), and :math:`\mu, \sigma` are found by
        Newton's method until both match to 1e-12, relative. Far above the
        scale (:math:`m \gg s`) the answer is :math:`\mu = m`, :math:`\sigma
        = s` (the flux is :math:`u`); far below, the lognormal's. A cell with
        no spread asked (a standard deviation of zero) gets :math:`\sigma =
        0` and the :math:`\mu` whose flux is the mean.

        The correlations are those of :math:`u`: where the flux is far above
        the scale they are the flux's too; near zero the flux's are weaker.

        Parameters
        ----------
        mean : Vector, Series or float
            The prior mean of each cell's flux, positive (the flux is never
            zero), on the block's labels or one number.
        std : Vector, Series or float
            The prior standard deviation of each cell's flux, not negative.

        Returns
        -------
        mean, std : Vector
            :math:`\mu` and :math:`\sigma`, in the flux's units, on the block's
            labels.

        Raises
        ------
        TypeError
            If `mean` or `std` is an array.
        ValueError
            If their labels are not the block's, a mean is not positive, or a
            standard deviation is negative or not finite.
        RuntimeError
            If Newton's method does not match some cell (it has not failed in
            the tests, over means from 1e-6 to 1e6 scales and standard
            deviations from 1e-6 to 1e3 times the mean).

        Examples
        --------
        A flux of mean 1 and standard deviation 1, with the bend at 1: the
        prior on :math:`u` is centered below the mean, and wider:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import fips
        >>> state = fips.Space({"flux": pd.Index(["a"], name="cell")})
        >>> H = fips.Operator(
        ...     np.ones((1, 1)), fips.Space({"ch4": pd.RangeIndex(1, name="i")}), state
        ... )
        >>> mu, sigma = fips.SoftplusForward(H, scale=1.0).prior_for(mean=1.0, std=1.0)
        >>> mu.values.round(3), sigma.values.round(3)
        (array([-0.059]), array([1.872]))
        """
        labels = self._labels
        m = _on_block(mean, labels, "mean")
        s_flux = _on_block(std, labels, "std")
        _negative(s_flux, labels, "std")
        bad = m <= 0
        if bad.any():
            raise ValueError(
                f"The mean is not positive at {int(bad.sum())} labels, and a "
                f"softplus flux is always above zero: {_listed(labels.index[bad])}."
            )
        scale = self._scale
        m, s_flux = m / scale, s_flux / scale  # in units of the scale
        mu, sigma = np.empty(len(m)), np.zeros(len(m))
        fixed = s_flux == 0
        # no spread: the u whose flux is the mean, log(e^m - 1)
        mu[fixed] = m[fixed] + np.log(-np.expm1(-m[fixed]))
        if (~fixed).any():
            mu[~fixed], sigma[~fixed] = _softplus_match(
                m[~fixed], s_flux[~fixed], labels.index[~fixed]
            )
        return (
            Vector._adopt(mu * scale, labels, "prior"),
            Vector._adopt(sigma * scale, labels, "prior_std"),
        )
