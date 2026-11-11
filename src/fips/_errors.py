"""
fips's warning class, and the one helper that logs a warning and issues it.

A warning that changes how far an answer can be trusted goes to both places:
the ``fips`` logger, for a program that configures logging, and Python's
:mod:`warnings`, so that a user who never does still sees it. :func:`warn` does
both, so the two cannot drift apart.
"""

import inspect
import logging
import os
import warnings

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep
"""The folder fips is imported from: a frame whose file is in it is fips's own."""


class FipsWarning(UserWarning):
    """
    Warned when fips finds something that changes how far an answer can be trusted.

    A covariance close to singular, a negative variance, a prior wide against
    its bounds, draws that mix slowly, iterations that converge slowly, a
    ``rank`` too small for what the data inform: each says what it found and
    what to do. Python shows a warning once for each line of your code that
    leads to it; :func:`warnings.filterwarnings` with
    ``category=fips.FipsWarning`` turns them into errors or hides them. fips
    also logs each one, at WARNING, under the ``fips`` logger.

    See Also
    --------
    warnings.filterwarnings : Choose what Python does with a warning.

    Examples
    --------
    A rank too small for what the data inform makes the variances too large:

    >>> import warnings
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> state = fips.Space({"x": pd.RangeIndex(10, name="i")})
    >>> observed = fips.Space({"z": pd.RangeIndex(6, name="j")})
    >>> H = np.random.default_rng(0).normal(size=(6, 10))
    >>> problem = fips.Problem(
    ...     prior=fips.Vector(np.zeros(10), state),
    ...     obs=fips.Vector(np.ones(6), observed),
    ...     forward=fips.Operator(H, observed, state),
    ...     prior_cov=fips.Independent().with_std(1.0),
    ...     obs_cov=fips.Independent().with_std(0.1),
    ... )
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always", fips.FipsWarning)
    ...     post = fips.solve(problem, "variational", rank=2)
    >>> [issubclass(w.category, fips.FipsWarning) for w in caught]
    [True]

    To make every fips warning an error (in a test suite, say):

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error", fips.FipsWarning)
    ...     fips.solve(
    ...         problem, "variational", rank=2
    ...     )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    fips.FipsWarning: The last of the 2 pairs of the low-rank posterior still has ...
    """


def warn(logger: logging.Logger, message: str, *args: object) -> None:
    """
    Log `message` at WARNING to `logger` and issue it as a :class:`FipsWarning`, at the first line outside fips.

    `message` and `args` are formatted as :mod:`logging` formats them
    (``message % args``). The warning points at the user's call that led to
    it, not at the line in fips that found it, so Python's once-per-line rule
    and a traceback under ``-W error`` name the user's code.
    """
    logger.warning(message, *args)
    frame, level = inspect.currentframe(), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message % args if args else message, FipsWarning, stacklevel=level)
