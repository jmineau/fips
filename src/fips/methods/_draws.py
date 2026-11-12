r"""
The Monte Carlo error of what a sampling method reads from its kept draws.

``"bounded"`` and ``"gauss_newton"`` report the variances and covariances of
chains of draws of their posterior. The draws are exact, but each starts
from the last, so a chain of ``N`` holds fewer than ``N`` independent ones,
and a standard deviation read from it has a larger error than ``N``
independent draws would give. :func:`_std_errors` estimates that error from
the chains themselves, and :func:`_warn_if_imprecise` warns when it passes
:data:`_IMPRECISE`.

A chain that never enters a tail cannot see that it missed it: its error
estimate is as wrong as its variance. Several chains from spread starts can
(``"gauss_newton"`` runs them): :func:`_split_rhat` compares the spread
between them with the spread within each, and :func:`_warn_if_unmixed` warns
when they disagree by more than :data:`_UNMIXED` on at least
:data:`_UNMIXED_SHARE` of the elements.
"""

import logging
import math

import numpy as np

from fips._errors import warn

_IMPRECISE = 0.1
"""The relative error of a standard deviation from the kept draws above which a solve warns."""

_UNMIXED = 1.05
"""The split R-hat of an element above which its chains disagree."""

_UNMIXED_SHARE = 0.05
"""
The share of the state elements whose chains must disagree for a solve to warn; fewer are logged at INFO.

At 20 elements or fewer, one element is enough. With many, a few elements
of a curved posterior mix slowly at the default draws whatever else does
(seen at 400 elements, 2026-11-12: a median of 5 above 1.05 in a solve,
while totals of the elements covered as claimed). Warning of them would warn
of nearly every solve. A sampler that missed part of the posterior leaves
many more: 2 of 2 elements, and up to 6 of 10.
"""

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
    draws: np.ndarray, independent: bool = False, lengths: tuple[int, ...] = ()
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
    Draws of several chains side by side, of the `lengths` given, have
    :math:`y` about the mean of them all, and :math:`\tau_y` the mean of the
    chains' own, weighted by their lengths.
    """
    N = draws.shape[1]
    y = (draws - draws.mean(axis=1, keepdims=True)) ** 2
    tau = np.ones(len(y))
    if not independent:
        tau = np.zeros(len(y))
        for len_, chain in zip(lengths or (N,), _chains(y, lengths), strict=True):
            tau += (len_ / N) * _autocorrelation_times(chain)
    mean = y.mean(axis=1)
    moves = mean > 0
    error = np.zeros(len(mean))
    error[moves] = 0.5 * np.sqrt(tau[moves] * y[moves].var(axis=1) / N) / mean[moves]
    return error, tau


def _warn_if_imprecise(
    draws: np.ndarray,
    logger: logging.Logger,
    why: str,
    independent: bool = False,
    lengths: tuple[int, ...] = (),
) -> None:
    """
    Warn (`logger`, and a :class:`~fips.FipsWarning`) when a standard deviation from the kept `draws` is uncertain by more than :data:`_IMPRECISE` (:func:`_std_errors`), saying how many draws would do; `why` is the method's sentence on the cause and the remedy.

    Draws `independent` by construction are not a chain: their error is
    counted with no autocorrelation, and the warning says they are too few
    rather than that they mix slowly. Draws of several chains side by side
    have their `lengths`.

    The draws needed scale as the square of the error: ``N (e / 0.1)^2``.
    """
    error, tau = _std_errors(draws, independent, lengths)
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
    warn(
        logger,
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


def _chains(draws: np.ndarray, lengths: tuple[int, ...]) -> list[np.ndarray]:
    """Return the chains held side by side in `draws` (one column per draw), of the `lengths` given; no lengths is one chain."""
    if not lengths:
        return [draws]
    ends = np.cumsum(lengths)
    return [draws[:, end - len_ : end] for len_, end in zip(lengths, ends, strict=True)]


def _split_rhat(draws: np.ndarray, lengths: tuple[int, ...]) -> np.ndarray:
    r"""
    Return the split R-hat of each row of the chains held side by side in `draws`, of the `lengths` given: the larger of the location's and the spread's.

    Each chain is cut in halves (a chain that drifts disagrees with itself),
    and every half to the length :math:`L` of the shortest. With :math:`W`
    the mean of the halves' variances and :math:`B / L` the variance of
    their means,

    .. math::

        \hat{R} = \sqrt{\frac{(L - 1) W / L + B / L}{W}},

    which is near 1 when the chains draw the same distribution and above it
    when one has stayed where another has not (Gelman et al., 2013, section
    11.4). It is computed of the draws, for the location, and of their
    distance from the median of them all, for the spread (Vehtari et al.,
    2021, the folded R-hat): a chain held out of a tail has a smaller spread
    than the others, often with nearly the same mean. A row that never moves
    has :math:`\hat{R} = 1`.

    References
    ----------
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and
    Rubin, D. B. (2013). *Bayesian Data Analysis*, third edition. CRC Press.

    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C.
    (2021). Rank-normalization, folding, and localization: an improved
    R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2),
    667-718. https://doi.org/10.1214/20-BA1221
    """
    folded = np.abs(draws - np.median(draws, axis=1, keepdims=True))
    half = min(lengths) // 2
    rhat = np.ones(len(draws))
    if half < 2:
        return rhat
    for values in (draws, folded):
        halves = np.stack(
            [
                part
                for chain in _chains(values, lengths)
                for part in (chain[:, :half], chain[:, -half:])
            ]
        )  # halves x rows x L
        W = halves.var(axis=2, ddof=1).mean(axis=0)
        B = half * halves.mean(axis=2).var(axis=0, ddof=1)
        moves = W > 0
        pooled = (half - 1) / half * W[moves] + B[moves] / half
        rhat[moves] = np.maximum(rhat[moves], np.sqrt(pooled / W[moves]))
    return rhat


def _warn_if_unmixed(
    draws: np.ndarray, lengths: tuple[int, ...], logger: logging.Logger, why: str
) -> None:
    """
    Warn (`logger`, and a :class:`~fips.FipsWarning`) when the chains held side by side in `draws`, of the `lengths` given, disagree: a split R-hat above :data:`_UNMIXED` (:func:`_split_rhat`) on at least :data:`_UNMIXED_SHARE` of the elements, naming the worst element and the least and the most of its chains' standard deviations; `why` is the method's sentence on the cause and the remedy.

    Fewer elements are logged at INFO, with the same names. One chain cannot
    give this warning: a chain that never entered a tail reports a small
    spread and a small error together.
    """
    if len(lengths) < 2:
        return
    rhat = _split_rhat(draws, lengths)
    unmixed = rhat > _UNMIXED
    if not unmixed.any():
        return
    count, n = int(unmixed.sum()), len(rhat)
    worst = int(np.argmax(rhat))
    stds = [float(chain[worst].std(ddof=1)) for chain in _chains(draws, lengths)]
    found = (
        len(lengths),
        count,
        n,
        _UNMIXED,
        worst,
        float(rhat[worst]),
        min(stds),
        max(stds),
    )
    message = (
        "The posterior's %d chains disagree: %d of the %d state elements have a "
        "split R-hat above %.2f, the worst (element %d) %.2f, its standard "
        "deviation %.3g in one chain and %.3g in another."
    )
    if count < _UNMIXED_SHARE * n:
        logger.info(
            message + " Those elements' variances are not yet the posterior's; "
            "too few elements to warn of (aggregates of many mix faster).",
            *found,
        )
        return
    warn(
        logger,
        message + " Their variances are not yet the posterior's. %s",
        *found,
        why,
    )
