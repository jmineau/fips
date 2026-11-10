r"""
The Monte Carlo error of what a sampling method reads from its kept draws.

``"bounded"`` and ``"gauss_newton"`` report the variances and covariances of
a chain of draws of their posterior. The draws are exact, but each starts
from the last, so a chain of ``N`` holds fewer than ``N`` independent ones,
and a standard deviation read from it has a larger error than ``N``
independent draws would give. :func:`_std_errors` estimates that error from
the chain itself, and :func:`_warn_if_imprecise` warns when it passes
:data:`_IMPRECISE`.
"""

import logging
import math

import numpy as np

_IMPRECISE = 0.1
"""The relative error of a standard deviation from the kept draws above which a solve warns."""

_ROWS = 256
"""The rows whose autocorrelation is computed at once."""


def _autocorrelation_times(draws: np.ndarray) -> np.ndarray:
    r"""
    Return the integrated autocorrelation time :math:`\tau` of each row of `draws` (a chain, one column per draw), by Sokal's adaptive window.

    :math:`\tau = 1 + 2 \sum_{k=1}^{w} \rho_k`, with :math:`\rho_k` the
    autocorrelation at lag :math:`k` (by FFT) and the window :math:`w` the
    first at least five times its :math:`\tau`. A chain has
    :math:`N / \tau` effective draws. A row that never moves has
    :math:`\tau = 1`. It reads one chain, so a chain that rarely visits a
    tail gives too small a :math:`\tau`: a diagnostic, not a guarantee.
    """
    n, N = draws.shape
    tau = np.ones(n)
    if N < 3:
        return tau
    windows = np.arange(1, N)
    for first in range(0, n, _ROWS):
        rows = slice(first, min(n, first + _ROWS))
        x = draws[rows] - draws[rows].mean(axis=1, keepdims=True)
        f = np.fft.rfft(x, 2 * N, axis=1)
        acf = np.fft.irfft(f * np.conj(f), 2 * N, axis=1)[:, :N]
        variance = acf[:, 0]
        moves = variance > 0
        if not moves.any():
            continue
        taus = 1.0 + 2.0 * np.cumsum(acf[moves, 1:] / variance[moves, None], axis=1)
        stop = windows[None, :] >= 5.0 * taus
        window = np.where(stop.any(axis=1), stop.argmax(axis=1), N - 2)
        block = tau[rows]
        block[moves] = np.maximum(taus[np.arange(len(window)), window], 1.0)
        tau[rows] = block
    return tau


def _std_errors(
    draws: np.ndarray, independent: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Return the relative error of the standard deviation of each row of the chain `draws`, and the autocorrelation time it counts.

    A variance from the chain is the mean of :math:`y = (x - \bar{x})^2`, so
    its error is :math:`\sqrt{\tau_y \operatorname{var}(y) / N}`, with
    :math:`\tau_y` the autocorrelation time of :math:`y`
    (:func:`_autocorrelation_times`), and a standard deviation's relative
    error half the variance's:

    .. math::

        e = \frac{1}{2 \bar{y}} \sqrt{\frac{\tau_y \operatorname{var}(y)}{N}}.

    For independent Gaussian draws it is :math:`1/\sqrt{2N}`. It counts the
    shape of the distribution as well as the correlation of the chain: an
    element piled against its bound (a Gaussian truncated one to a few of its
    standard deviations inside the box) has a larger :math:`\operatorname{var}(y)
    / \bar{y}^2` than a Gaussian and mixes slower, and both raise :math:`e`.
    A row that never moves has an error of 0. Draws `independent` by
    construction (not a chain) have :math:`\tau_y = 1`, not an estimate of it.
    """
    N = draws.shape[1]
    y = (draws - draws.mean(axis=1, keepdims=True)) ** 2
    tau = np.ones(len(y)) if independent else _autocorrelation_times(y)
    mean = y.mean(axis=1)
    moves = mean > 0
    error = np.zeros(len(mean))
    error[moves] = 0.5 * np.sqrt(tau[moves] * y[moves].var(axis=1) / N) / mean[moves]
    return error, tau


def _warn_if_imprecise(
    draws: np.ndarray, logger: logging.Logger, why: str, independent: bool = False
) -> None:
    """
    Log a warning to `logger` when a standard deviation from the kept `draws` is uncertain by more than :data:`_IMPRECISE` (:func:`_std_errors`), saying how many draws would do; `why` is the method's sentence on the cause and the remedy.

    Draws `independent` by construction are not a chain: their error is
    counted with no autocorrelation, and the warning says they are too few
    rather than that they mix slowly.

    The draws needed scale as the square of the error: ``N (e / 0.1)^2``.
    """
    error, tau = _std_errors(draws, independent)
    slow = error > _IMPRECISE
    if not slow.any():
        return
    N = draws.shape[1]
    worst = int(np.argmax(error))
    needed = math.ceil(N * (float(error[worst]) / _IMPRECISE) ** 2)
    if independent:
        what, chain = "are too few", ""
    else:
        what = "mix slowly"
        chain = f" (its variance has {float(tau[worst]):.0f} draws for each independent one)"
    logger.warning(
        "The posterior's draws %s: %d state elements have a standard deviation "
        "from the %d kept draws uncertain by more than %d%%, the worst (element "
        "%d) by about %d%%%s; about %d draws would bring it to %d%%. %s",
        what,
        int(slow.sum()),
        N,
        round(100 * _IMPRECISE),
        worst,
        round(100 * float(error[worst])),
        chain,
        needed,
        round(100 * _IMPRECISE),
        why,
    )
