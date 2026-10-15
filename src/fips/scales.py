r"""
Scales for the model-data mismatch: how big each observation's signal is.

A model-data mismatch that grows with the signal (a transport error of 30 %
of the enhancement, say) is a standard deviation of ``fraction * scale`` for
each observation. Each function here gives a scale, one value per
observation, as a :class:`~fips.Vector` on the observations' space; make the
covariance with a model, ``fips.Independent().with_std(fips.Vector(0.3 *
scale.values, scale.space))``, and add a floor as another term.

=================  ==========================  ===========================================
Function           Scale                       What it measures
=================  ==========================  ===========================================
:func:`footprint`  :math:`|H| \mathbf{1}\,\bar{x}_0`  how much state each observation sees,
                                               in observation units; it depends on neither
                                               the observations nor the prior's pattern
:func:`modeled`    :math:`|H x_0|`             the signal the prior models; small where the
                                               prior is low
:func:`observed`   :math:`|z - c|`             the signal observed above the offset
=================  ==========================  ===========================================

:math:`|H|` is the forward operator's entries in absolute value and
:math:`\bar{x}_0` the mean of the prior. Pass the part of the state the
signal comes from: ``problem.forward["co2", "flux"]`` and
``problem.prior["flux"]`` leave a block of offsets out.
"""

import numpy as np

from fips.operator import Operator
from fips.vector import Vector

__all__ = ["footprint", "modeled", "observed"]


def _check(forward: object, prior: object) -> None:
    """Raise a TypeError unless `forward` is an Operator and `prior` a Vector."""
    if not isinstance(forward, Operator):
        raise TypeError(f"forward must be an Operator, got {type(forward).__name__}.")
    if not isinstance(prior, Vector):
        raise TypeError(f"prior must be a Vector, got {type(prior).__name__}.")


def footprint(forward: Operator, prior: Vector) -> Vector:
    r"""
    Return each observation's total sensitivity times the mean prior: :math:`|H| \mathbf{1}\,\bar{x}_0`.

    The sum of the absolute entries of each row of :math:`H` is how much of
    the state an observation sees; times the mean of the prior it is in the
    observations' units. It scales an error by the strength of the footprint,
    whatever the observations are (no circularity) and wherever the prior puts
    its mass.

    Parameters
    ----------
    forward : Operator
        The forward operator, or the block of it the signal comes from.
    prior : Vector
        The prior on `forward`'s columns: exactly its labels, in any order.

    Returns
    -------
    Vector
        One scale per observation, on `forward`'s rows, named ``"footprint"``.

    Raises
    ------
    TypeError
        If `forward` is not an Operator or `prior` not a Vector.
    ValueError
        If `prior` does not have exactly the labels of `forward`'s columns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
    >>> obs = fips.Space({"co2": pd.Index(["o1", "o2"], name="obs_id")})
    >>> H = fips.Operator(np.array([[1.0, -2.0], [0.5, 0.0]]), obs, state)
    >>> prior = fips.Vector([1.0, 3.0], state)
    >>> fips.scales.footprint(H, prior).values
    array([6., 1.])
    """
    _check(forward, prior)
    x = prior.align(forward.cols)
    csr = forward.data._csr()
    if csr is None:
        row_abs = np.abs(forward.data.dense()).sum(axis=1)
    else:
        row_abs = np.asarray(abs(csr).sum(axis=1)).ravel()
    return Vector(row_abs * float(np.mean(x.values)), forward.rows, name="footprint")


def modeled(forward: Operator, prior: Vector) -> Vector:
    r"""
    Return the size of the signal the prior models: :math:`|H x_0|`.

    Parameters
    ----------
    forward : Operator
        The forward operator, or the block of it the signal comes from.
    prior : Vector
        The prior on `forward`'s columns: exactly its labels, in any order
        (they are matched by label).

    Returns
    -------
    Vector
        One scale per observation, on `forward`'s rows, named ``"modeled"``.

    Raises
    ------
    TypeError
        If `forward` is not an Operator or `prior` not a Vector.
    ValueError
        If `prior` does not have exactly the labels of `forward`'s columns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
    >>> obs = fips.Space({"co2": pd.Index(["o1", "o2"], name="obs_id")})
    >>> H = fips.Operator(np.array([[1.0, -2.0], [0.5, 0.0]]), obs, state)
    >>> fips.scales.modeled(H, fips.Vector([1.0, 3.0], state)).values
    array([5. , 0.5])
    """
    _check(forward, prior)
    values = forward.data @ prior.align(forward.cols).values
    return Vector(np.abs(values), forward.rows, name="modeled")


def observed(obs: Vector, offset: "Vector | float | None" = None) -> Vector:
    r"""
    Return the size of the signal observed above the offset: :math:`|z - c|`.

    Parameters
    ----------
    obs : Vector
        The observations, :math:`z`.
    offset : Vector or float, optional
        The offset (a background), :math:`c`: a vector on exactly the
        observations' labels (in any order), or one value for all. None is
        zero.

    Returns
    -------
    Vector
        One scale per observation, on `obs`'s space, named ``"observed"``.

    Raises
    ------
    TypeError
        If `obs` is not a Vector.
    ValueError
        If `offset` does not have exactly the observations' labels.

    Examples
    --------
    >>> import pandas as pd
    >>> import fips
    >>> obs = fips.Vector.from_series(
    ...     pd.Series([2.1, 1.8], index=pd.Index(["o1", "o2"], name="obs_id")),
    ...     block="ch4",
    ... )
    >>> fips.scales.observed(obs, 1.9).values.round(2)
    array([0.2, 0.1])
    """
    if not isinstance(obs, Vector):
        raise TypeError(f"obs must be a Vector, got {type(obs).__name__}.")
    if offset is None:
        values = obs.values
    elif isinstance(offset, Vector):
        values = obs.values - offset.align(obs.space).values
    else:
        values = obs.values - float(offset)
    return Vector(np.abs(values), obs.space, name="observed")
