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

An aggregate of the elements, :math:`W x`, depends on how they vary
together, so its chains can disagree where few elements' do, and agree where
many do. :func:`_warn_if_aggregates_unmixed` compares the chains on the
aggregates that are read, and warns when they disagree by more than
:data:`_UNMIXED_AGGREGATE` on any of them.

:func:`_diagnostics` gives the numbers a user of PyMC or Stan reads after a
run, as ArviZ computes them (Vehtari et al., 2021): the rank-normalized split
R-hat, the effective number of draws in the bulk and in the tails, and the
Monte Carlo errors of the mean and of the standard deviation.
:attr:`Posterior.diagnostics <fips.Posterior.diagnostics>` labels them. An
interval of a sampled answer is two quantiles of its draws, so its ends are
only as good as the draws in the tails: :func:`_warn_if_tails_thin` warns
when those are fewer than :data:`_THIN_TAILS` for each chain. A path whose
energy rose by more than :data:`_DIVERGENT` diverged (Stan's rule):
:func:`_log_divergent` logs how many of the kept draws came from one, at INFO.

References
----------
Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C.
(2021). Rank-normalization, folding, and localization: an improved R-hat for
assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
https://doi.org/10.1214/20-BA1221

Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
Science*, 7(4), 473-483.
"""

import logging
import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

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

_UNMIXED_AGGREGATE = 1.1
"""
The split R-hat of an aggregate that is read above which its chains disagree, and the read warns.

With four chains it is a standard deviation known to about 8% or worse.
Measured over 736 default solves (2000 draws) of 23 problems of 100 to 900
elements, each read through eight aggregates against two long solves
(2026-11-14): every aggregate whose standard deviation was more than 20% off
was above it (38 of 38), a third of those above it were more than 10% off,
and 3% of those within 10% were above it. At :data:`_UNMIXED`, the elements'
cutoff, 11% of those within 10% were, and two solves in five warned. An
aggregate 10% to 20% off is not told from one that is right at any cutoff:
that is the scatter of a standard deviation from 2000 draws, which more
draws lower and no comparison of chains sees.
"""

_NAMED = 8
"""The rows of an aggregation a warning names at most; it counts the rest."""

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


def _warn_if_aggregates_unmixed(
    draws: np.ndarray, lengths: tuple[int, ...], logger: logging.Logger, why: str
) -> None:
    """
    Warn (`logger`, and a :class:`~fips.FipsWarning`) when the chains disagree on an aggregate that is read: `draws` are the chains' draws of :math:`W x` side by side (one row per row of :math:`W`), of the `lengths` given, and a row warns with a split R-hat above :data:`_UNMIXED_AGGREGATE` (:func:`_split_rhat`).

    It names the rows (their positions in :math:`W`, from 0; the first
    :data:`_NAMED` of them), the worst one and the least and the most of its
    chains' standard deviations; `why` is the method's sentence on the cause
    and the remedy. Any one row warns, however many rows there are: each is
    an aggregate the caller asked for. One chain cannot give this warning.
    """
    if len(lengths) < 2:
        return
    rhat = _split_rhat(draws, lengths)
    unmixed = np.flatnonzero(rhat > _UNMIXED_AGGREGATE)
    if not len(unmixed):
        return
    worst = int(np.argmax(rhat))
    stds = [float(chain[worst].std(ddof=1)) for chain in _chains(draws, lengths)]
    named = ("rows " if len(unmixed) > 1 else "row ") + ", ".join(
        str(int(row)) for row in unmixed[:_NAMED]
    )
    if len(unmixed) > _NAMED:
        named += f" and {len(unmixed) - _NAMED} more"
    warn(
        logger,
        "The posterior's %d chains disagree on %d of the %d aggregates read "
        "(%s of the aggregation, counted from 0): a split R-hat above %.2f, "
        "the worst (row %d) %.2f, its standard deviation %.3g in one chain and "
        "%.3g in another. Their variances and intervals are not yet the "
        "posterior's. %s",
        len(lengths),
        len(unmixed),
        len(rhat),
        named,
        _UNMIXED_AGGREGATE,
        worst,
        float(rhat[worst]),
        min(stds),
        max(stds),
        why,
    )


_THIN_TAILS = 100
"""
The effective draws in the tails, for each chain, below which an interval read from the draws warns.

It is the floor Vehtari et al. (2021) give for every effective number of
draws (100 for each chain), applied to the tail's (:func:`_ess_tail`): an
interval's ends are the 2.5% and 97.5% quantiles of the draws, which only the
draws out there decide. Measured over 90 default solves (2000 draws, four
chains) of 23 problems of 100 to 900 elements, both transforms
(2026-11-16): the elements' warning (at least :data:`_UNMIXED_SHARE` of
them) fired on 4, all in the settings whose chains mix slowest; one of
them was also the one solve with 5% of its elements' standard deviations
more than 10% off a long solve's.
"""

_DIVERGENT = 1000.0
"""The rise of the energy along a path above which the path diverged: the sampler could not follow the posterior there (Stan's rule)."""

_TAIL = 0.05
"""The probability of each tail whose quantile :func:`_ess_tail` reads, as ArviZ does: the 5% and the 95% quantiles."""


def _equal_chains(draws: np.ndarray, lengths: tuple[int, ...]) -> np.ndarray:
    """Return the chains held side by side in `draws` (one column per draw), of the `lengths` given (no lengths: one chain), each cut to the shortest: ``rows x chains x L``."""
    chains = _chains(draws, lengths)
    shortest = min(chain.shape[1] for chain in chains)
    return np.stack([chain[:, :shortest] for chain in chains], axis=1)


def _halves(chains: np.ndarray) -> np.ndarray:
    """Return each chain cut in a first and a last half (the middle draw of an odd chain left out), ``rows x 2 chains x L // 2``: a chain that drifts disagrees with itself."""
    half = chains.shape[2] // 2
    return np.concatenate([chains[:, :, :half], chains[:, :, -half:]], axis=1)


def _z_scale(values: np.ndarray) -> np.ndarray:
    r"""
    Return `values` (``rows x chains x L``) rank-normalized, each row over all its draws.

    A draw of rank :math:`r` (ties averaged) among the :math:`S` of its row
    becomes :math:`\Phi^{-1}((r - 3/8) / (S + 1/4))` (Blom's offset): the
    draws' order kept, their scale made normal, so a heavy tail does not
    decide what is computed from them.
    """
    rows = values.reshape(len(values), -1)
    ranks = rankdata(rows, method="average", axis=1)
    size = rows.shape[1]
    return ndtri((ranks - 0.375) / (size + 0.25)).reshape(values.shape)


def _rhat_of(halves: np.ndarray) -> np.ndarray:
    r"""Return the R-hat of each row of `halves` (``rows x chains x L``): :math:`\sqrt{(B / W + L - 1) / L}`, with :math:`W` the mean of the chains' variances and :math:`B / L` the variance of their means."""
    L = halves.shape[2]
    W = halves.var(axis=2, ddof=1).mean(axis=1)
    B = L * halves.mean(axis=2).var(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((B / W + L - 1) / L)


def _rhat(draws: np.ndarray, lengths: tuple[int, ...]) -> np.ndarray:
    """
    Return the rank-normalized split R-hat of each row of the chains held side by side in `draws`, of the `lengths` given (Vehtari et al., 2021): the larger of the bulk's and the tails'.

    The bulk's is the split R-hat of the rank-normalized draws, the tails' of
    their rank-normalized distance from the median of the row: a chain held
    out of a tail has a smaller spread than the others, often with nearly the
    same location. It is near 1 when the chains draw the same distribution.
    It is NaN with one chain, or fewer than four draws in each, as in ArviZ;
    a row that never moves is 1.
    """
    rhat = np.full(len(draws), np.nan)
    if len(lengths) < 2 or min(lengths) < 4:
        return rhat
    halves = _halves(_equal_chains(draws, lengths))
    moves = np.ptp(halves.reshape(len(halves), -1), axis=1) > 0
    rhat[~moves] = 1.0
    if not moves.any():
        return rhat
    halves = halves[moves]
    folded = np.abs(
        halves - np.median(halves.reshape(len(halves), -1), axis=1)[:, None, None]
    )
    rhat[moves] = np.maximum(_rhat_of(_z_scale(halves)), _rhat_of(_z_scale(folded)))
    return rhat


def _ess_of(chains: np.ndarray) -> np.ndarray:
    r"""
    Return the effective number of draws of each row of `chains` (``rows x M x N``), by Geyer's initial monotone sequence over the chains' autocorrelations (Vehtari et al., 2021, section 3.2; as Stan and ArviZ compute it).

    With :math:`\rho_t = 1 - (W - \bar{c}_t) / \widehat{\operatorname{var}}^+`
    (:math:`\bar{c}_t` the chains' mean autocovariance at lag :math:`t`,
    :math:`W` their mean variance), the pairs :math:`\rho_{2k} +
    \rho_{2k+1}` are summed while they are positive, each made no larger than
    the one before, and :math:`\tau = -1 + 2 \sum_k (\rho_{2k} +
    \rho_{2k+1}) + \rho_{2K}`, at least :math:`1 / \log_{10}(M N)`; the
    effective number is :math:`M N / \tau`. A row that never moves has
    :math:`M N`. Each row is centered and scaled to at most 1 first (the
    number does not change), so that draws far out do not overflow the FFT.
    """
    rows, M, N = chains.shape
    size = M * N
    ess = np.full(rows, float(size))
    pairs = N // 2
    stops = 2 * np.arange(pairs) + 1 >= N - 3  # the last pair Geyer's loop reaches
    for first in range(0, rows, _ROWS):
        block = chains[first : first + _ROWS]
        moves = (
            np.ptp(block.reshape(len(block), -1), axis=1) >= np.finfo(float).resolution
        )
        if not moves.any():
            continue
        x = block[moves]
        with np.errstate(all="ignore"):  # the number is the same at any scale
            x = x - x.mean(axis=(1, 2), keepdims=True)
            x = x / np.max(np.abs(x), axis=(1, 2), keepdims=True)  # no overflow
        means = x.mean(axis=2)
        x = x - means[:, :, None]
        f = np.fft.rfft(x, 2 * N, axis=2)
        acov = np.fft.irfft(f * np.conj(f), 2 * N, axis=2)[:, :, :N] / N
        mean_acov = acov.mean(axis=1)  # rows x N
        mean_var = mean_acov[:, 0] * N / (N - 1.0)
        var_plus = mean_var * (N - 1.0) / N
        if M > 1:
            var_plus = var_plus + means.var(axis=1, ddof=1)
        rho = 1.0 - (mean_var[:, None] - mean_acov) / var_plus[:, None]
        rho[:, 0] = 1.0
        sums = rho[:, 0 : 2 * pairs : 2] + rho[:, 1 : 2 * pairs : 2]  # rows x pairs
        end = np.argmax((sums <= 0.0) | stops[None, :], axis=1)  # the last pair reached
        reached = np.arange(pairs)[None, :] < end[:, None]
        kept = np.where(
            reached, np.minimum.accumulate(np.where(reached, sums, np.inf), axis=1), 0.0
        )
        last_even = rho[np.arange(len(end)), 2 * end]
        last = np.where(
            (last_even > 0.0) | (sums[np.arange(len(end)), end] >= 0.0), last_even, 0.0
        )
        tau = -1.0 + 2.0 * kept.sum(axis=1) + last
        tau = np.maximum(tau, 1.0 / math.log10(size))
        values = size / tau
        values[~np.isfinite(rho).all(axis=1)] = np.nan
        part = ess[first : first + _ROWS]
        part[moves] = values
        ess[first : first + _ROWS] = part
    return ess


def _ess_bulk(draws: np.ndarray, lengths: tuple[int, ...]) -> np.ndarray:
    """Return the effective number of draws in the bulk of each row: of the rank-normalized split chains (Vehtari et al., 2021). NaN with fewer than four draws in a chain."""
    if min(lengths or (draws.shape[1],)) < 4:
        return np.full(len(draws), np.nan)
    return _ess_of(_z_scale(_halves(_equal_chains(draws, lengths))))


def _ess_quantile(chains: np.ndarray, prob: float) -> np.ndarray:
    """Return the effective number of draws of each row of `chains` (``rows x M x N``, equal lengths) for its `prob` quantile: of the split chains of the indicator of a draw at or below it."""
    rows = chains.reshape(len(chains), -1)
    quantile = np.quantile(rows, prob, axis=1)
    below = (chains <= quantile[:, None, None]).astype(float)
    return _ess_of(_halves(below))


def _ess_tail(draws: np.ndarray, lengths: tuple[int, ...]) -> np.ndarray:
    """Return the effective number of draws in the tails of each row: the smaller of those for its :data:`_TAIL` and ``1 -`` :data:`_TAIL` quantiles (Vehtari et al., 2021). NaN with fewer than four draws in a chain."""
    if min(lengths or (draws.shape[1],)) < 4:
        return np.full(len(draws), np.nan)
    chains = _equal_chains(draws, lengths)
    return np.minimum(_ess_quantile(chains, _TAIL), _ess_quantile(chains, 1.0 - _TAIL))


def _diagnostics(draws: np.ndarray, lengths: tuple[int, ...]) -> dict[str, np.ndarray]:
    r"""
    Return, for each row of the chains held side by side in `draws` (of the `lengths` given; none: one chain), the numbers ArviZ's summary gives.

    ``mean`` and ``sd`` of all the draws; ``mcse_mean``, the standard
    deviation over the root of the effective draws of the split chains;
    ``mcse_sd``, :math:`\sqrt{(\overline{y^2} - \bar{y}^2) / (4\, \bar{y}\,
    \text{ESS}_y)}` with :math:`y` the squared distance from the mean (the
    error of a variance, carried to the standard deviation); ``ess_bulk`` and
    ``ess_tail`` (:func:`_ess_bulk`, :func:`_ess_tail`); and ``r_hat``
    (:func:`_rhat`). The chains are cut to the shortest for all but the mean
    and the standard deviation. A statistic that needs more draws than there
    are is NaN.
    """
    lengths = lengths or (draws.shape[1],)
    mean = draws.mean(axis=1)
    sd = draws.std(axis=1, ddof=1)
    nan = np.full(len(draws), np.nan)
    if min(lengths) < 4:
        return {
            "mean": mean,
            "sd": sd,
            "mcse_mean": nan,
            "mcse_sd": nan,
            "ess_bulk": nan,
            "ess_tail": nan,
            "r_hat": _rhat(draws, lengths),
        }
    chains = _equal_chains(draws, lengths)
    ess_mean = _ess_of(_halves(chains))
    rows = chains.reshape(len(chains), -1)
    centered = rows - rows.mean(axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):  # draws far out: inf, NaN
        y = centered**2
        ess_y = _ess_of(_halves(y.reshape(chains.shape)))
        y_mean = y.mean(axis=1)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mcse_mean = rows.std(axis=1, ddof=1) / np.sqrt(ess_mean)
        mcse_sd = np.sqrt(((y**2).mean(axis=1) - y_mean**2) / ess_y / y_mean / 4.0)
    return {
        "mean": mean,
        "sd": sd,
        "mcse_mean": mcse_mean,
        "mcse_sd": mcse_sd,
        "ess_bulk": _ess_bulk(draws, lengths),
        "ess_tail": _ess_tail(draws, lengths),
        "r_hat": _rhat(draws, lengths),
    }


def _warn_if_tails_thin(
    draws: np.ndarray,
    lengths: tuple[int, ...],
    logger: logging.Logger,
    aggregates: bool,
) -> None:
    """
    Warn (`logger`, and a :class:`~fips.FipsWarning`) when an interval is read from draws with fewer than :data:`_THIN_TAILS` effective draws in the tails for each chain (:func:`_ess_tail`).

    `draws` are the rows the interval is read for, the chains side by side,
    of the `lengths` given (none: one chain). Of `aggregates` (rows of
    :math:`W x`), any one row warns, each being an aggregate the caller asked
    for; of the state's elements, at least :data:`_UNMIXED_SHARE` of them
    (any one, at 20 or fewer), as for the chains' disagreement; fewer are
    logged at INFO. It names the worst row (counted from 0) and the draws
    that would bring it to the floor: the effective draws grow with the
    draws.
    """
    lengths = lengths or (draws.shape[1],)
    if min(lengths) < 4:
        return
    ess = _ess_tail(draws, lengths)
    floor = _THIN_TAILS * len(lengths)
    thin = ess < floor
    if not thin.any():
        return
    count, n = int(thin.sum()), len(ess)
    worst = int(np.nanargmin(ess))
    N = draws.shape[1]
    needed = math.ceil(N * floor / max(float(ess[worst]), 1.0))
    what = "aggregates" if aggregates else "state elements"
    found = (count, n, what, floor, len(lengths), worst, float(ess[worst]), N, needed)
    message = (
        "%d of the %d %s have fewer than %d effective draws in their tails "
        "(%d chains), which their intervals' ends are read from: the worst "
        "(row %d, counted from 0) has %.0f of the %d draws; about %d draws "
        "would bring it to the floor."
    )
    if not aggregates and count < _UNMIXED_SHARE * n and n > 20:
        logger.info(message + " Too few elements to warn of.", *found)
        return
    warn(
        logger,
        message + " The interval's ends are not yet the posterior's quantiles; "
        "raise draws=.",
        *found,
    )


def _log_divergent(divergent: np.ndarray, logger: logging.Logger) -> None:
    """
    Log at INFO how many of the kept draws came from a path that diverged (`divergent`, one boolean a draw): its energy rose by more than :data:`_DIVERGENT`, or the model could not be run along it.

    A divergent path is rejected, so the draws stay exact, but it marks a
    part of the posterior the sampler could not follow there. Stan warns of
    any; fips logs them, since for its sampler they do not mark the solves
    that are off: over 90 default solves of 23 problems of 100 to 900
    elements (2026-11-16), 20 had a divergent draw (1 to 22 of 2000), none of
    them with 5% of its elements' standard deviations more than 10% off a
    long solve's, and the four solves that were off had 0 to 6.
    """
    count = int(np.count_nonzero(divergent))
    if not count:
        return
    logger.info(
        "%d of the %d kept draws came from a path that diverged (its energy "
        "rose by more than %.0f, or the model failed along it): the sampler "
        "could not follow the posterior there. Many, with chains that disagree, "
        "point at where the model bends most or overflows.",
        count,
        len(divergent),
        _DIVERGENT,
    )
