r"""
The covariance hyperparameters as unknowns: their priors, their posterior, and the state over them.

Every problem rests on hyperparameters someone chose (the prior's standard
deviation and correlation length, the observation error's standard
deviation). :func:`hierarchical` gives each hyperparameter a prior, draws the
hyperparameters from their posterior,

.. math::

    p(s \mid z) \propto p(s) \, p(z \mid s),

with :math:`p(z \mid s)` the exact evidence of the problem built at
:math:`s` (:attr:`Posterior.log_evidence <fips.Posterior.log_evidence>`),
and draws the state exactly from :math:`x \mid s, z` at each kept draw of the
hyperparameters (:meth:`Posterior.sample <fips.Posterior.sample>`). The draws of the
state are then draws of :math:`p(x \mid z)`, a mixture of the Gaussian
posteriors of every hyperparameter, and their intervals hold the uncertainty of the
hyperparameters. :class:`Hierarchical` holds them.

The hyperparameters' priors are the distributions of :mod:`fips.distributions`
(:class:`~fips.distributions.LogNormal`,
:class:`~fips.distributions.HalfNormal` and
:class:`~fips.distributions.Uniform`): each has a log density and a map to unbounded coordinates
(the log of a positive hyperparameter, the logit of a bounded one), where the
sampler works. The sampler is a random-walk Metropolis whose proposal is
Laplace's approximation of the hyperparameters' posterior at its mode (a numerical
Hessian), scaled during the burn-in towards an acceptance of 0.3 (Gelman,
Roberts and Gilks, 1996): a few hyperparameters need no gradient, and each step costs
one solve.

:func:`sweep` solves the problem at every combination of a grid of the
hyperparameters' values instead, with no prior on them: :class:`Sweep` holds
each member's diagnostics, estimate and posterior variances, and the mean
and spread of the estimates of the members a filter of its table keeps.
"""

import logging
import math
import numbers
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.optimize import minimize

from fips._errors import warn
from fips.covariance import Covariance
from fips.distributions import _LOG_2PI, _Distribution
from fips.linalg.forms import _adopt
from fips.methods import solve
from fips.methods.analytic import AnalyticOptions, Information, Whitened
from fips.operator import Operator
from fips.posterior import Posterior, UnavailableError
from fips.problem import Problem
from fips.space import Space
from fips.vector import Vector

logger = logging.getLogger(__name__)

_SCALES = ("prior_scale", "obs_scale", "gamma")
"""The hyperparameters the scales-only fast path takes, when :func:`hierarchical` is given a problem."""

_ACCEPTANCE = 0.3
"""The acceptance the burn-in steers the proposal's scale towards."""


# -- what one step of the chain solves ---------------------------------------


def _scaled(problem: Problem, prior_factor: float, obs_factor: float) -> Problem:
    """Return `problem` with its prior covariance times `prior_factor` and its observation covariance times `obs_factor`."""
    prior_cov, obs_cov = problem.prior_cov * prior_factor, problem.obs_cov * obs_factor
    assert isinstance(prior_cov, Covariance) and isinstance(obs_cov, Covariance)
    return Problem(
        prior=problem.prior,
        prior_cov=prior_cov,
        obs=problem.obs,
        obs_cov=obs_cov,
        forward=problem.forward,
        offset=problem.offset,
    )


class _Built:
    """The general path: the caller's function from the hyperparameters to a problem (or a posterior), solved at each step."""

    def __init__(self, build: Callable[[dict[str, float]], Any]):
        self.build = build
        self.space: Space | None = None
        """The state's labels of the first problem built (at the priors' medians), which the draws are put in the order of."""

    def posterior(self, values: dict[str, float]) -> Posterior:
        """Return the posterior at `values`: the built problem solved, or the posterior the function gave."""
        made = self.build(dict(values))
        if isinstance(made, Problem):
            return solve(made)
        if isinstance(made, Posterior):
            if not isinstance(made.options, AnalyticOptions):
                raise TypeError(
                    "hierarchical needs the exact evidence and exact draws of the "
                    "analytic method, and the function gave a posterior of "
                    f"another method ({type(made.options).__name__}). Return the "
                    "Problem, or fips.solve(problem, gamma=...)."
                )
            return made
        raise TypeError(
            "The function hierarchical takes builds a Problem (or its analytic "
            f"Posterior) from a dict of hyperparameters; it returned {type(made).__name__}."
        )

    def evaluate(self, values: dict[str, float]) -> tuple[float, Any]:
        """Return the log evidence at `values` and the posterior there (``-inf`` and None where a covariance is not positive-definite)."""
        try:
            post = self.posterior(values)
            if self.space is None:
                self.space = post.problem.prior.space
            return post.log_evidence, post
        except LinAlgError as err:
            logger.debug("The problem at %s did not solve (%s).", values, err)
            return -math.inf, None

    def keep(self, post: Posterior, rng: np.random.Generator) -> np.ndarray:
        """
        Return one exact draw of the state from a kept step's posterior, in the order of the first problem built.

        Drawn at once, so the chain holds no more than two posteriors. A
        posterior whose prior is in another order is reordered; one with
        other labels raises.
        """
        space = self.space
        assert space is not None  # the search for the mode built a problem
        draw = post._optional("sample", "draws from the posterior")(1, rng)[:, 0]
        other = post.problem.prior.space
        if other is not space and other != space:
            draw = draw[
                space._align(
                    other,
                    "the state of a built problem",
                    "The state of the problems built at two hyperparameters",
                )
            ]
        return draw

    def stack(
        self, kept: list[Any], rng: np.random.Generator
    ) -> tuple[Space, np.ndarray]:
        """Return the state's space and the kept draws, ``n x N``."""
        assert self.space is not None  # the search for the mode built a problem
        return self.space, np.column_stack(kept)


class _Scales:
    r"""
    The fast path: the hyperparameters scale the problem's prior std (``prior_scale``) and its observation covariance (``obs_scale``, or ``gamma``).

    At :math:`s` = ``prior_scale`` and :math:`c` = ``obs_scale**2 /
    gamma``, the innovation covariance is :math:`s^2 H S_0 H^T + c S_z = s^2
    (H S_0 H^T + S_z / \gamma')` with :math:`\gamma' = s^2 / c`: the
    evidence is the problem's at :math:`\gamma'`, with its chi-squared over
    :math:`s^2`, and :math:`m \ln c` in place of :math:`- m \ln \gamma'`. One
    eigendecomposition gives it at every step, in :math:`O(k)`
    (``_evidence_terms``), and the draws of the state at every kept step in
    one product (``_draws_at``).
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        solution = solve(problem)._solution  # raises for a forward model
        assert isinstance(solution, (Whitened, Information))  # the analytic method's
        self.factors = solution._eigendecomposition()
        self.obs_logdet = problem._obs_logdet

    @staticmethod
    def scales(values: dict[str, float]) -> tuple[float, float]:
        """Return the prior std's scale and the observation covariance's, :math:`(s, c)`."""
        obs_scale = values.get("obs_scale", 1.0)
        return values.get("prior_scale", 1.0), obs_scale * obs_scale / values.get(
            "gamma", 1.0
        )

    def evaluate(self, values: dict[str, float]) -> tuple[float, Any]:
        r"""Return the log evidence at `values` and :math:`(\gamma', s)`, what a draw at them needs."""
        s, c = self.scales(values)
        if not (0.0 < s < math.inf and 0.0 < c < math.inf):
            return -math.inf, None
        gamma = s * s / c
        if not 0.0 < gamma < math.inf:
            return -math.inf, None
        with np.errstate(all="ignore"):  # a scale far out in the tails: no density
            chi2, information = self.factors._evidence_terms(np.array([gamma]))
            m = len(self.problem.obs)
            log_evidence = -0.5 * float(
                chi2[0] / (s * s)
                + 2.0 * information[0]
                + self.obs_logdet
                + m * math.log(c)
                + m * _LOG_2PI
            )
        if not math.isfinite(log_evidence):
            return -math.inf, None
        return log_evidence, (gamma, s)

    def posterior(self, values: dict[str, float]) -> Posterior:
        """Return the posterior of the problem with its covariances scaled, solved."""
        s, c = self.scales(values)
        return solve(_scaled(self.problem, s * s, c))

    def keep(self, payload: Any, rng: np.random.Generator) -> Any:
        r"""Return what a kept step's draw needs, :math:`(\gamma', s)`: the draws are made together, in :meth:`stack`."""
        return payload

    def stack(
        self, kept: list[Any], rng: np.random.Generator
    ) -> tuple[Space, np.ndarray]:
        r"""Return the state's space and one exact draw of the state at each kept :math:`(\gamma', s)`, in one product (``n x N``)."""
        gammas, scales = np.array(kept, dtype=float).T
        return self.problem.prior.space, self.factors._draws_at(gammas, scales, rng)


# -- the sampler ----------------------------------------------------------------


def _accept(log_ratio: float, rng: np.random.Generator) -> bool:
    """Return whether Metropolis accepts a proposal whose log density is `log_ratio` above the current one; one uniform each call."""
    return bool(rng.random() < math.exp(min(0.0, log_ratio)))


def _hessian(f: Callable[[np.ndarray], float], u: np.ndarray, h: float) -> np.ndarray:
    """Return the Hessian of `f` at `u` by central differences of step `h`."""
    p = len(u)
    H = np.empty((p, p))
    f0 = f(u)
    steps = h * np.eye(p)
    for i in range(p):
        H[i, i] = (f(u + steps[i]) - 2.0 * f0 + f(u - steps[i])) / (h * h)
        for j in range(i):
            H[i, j] = H[j, i] = (
                f(u + steps[i] + steps[j])
                - f(u + steps[i] - steps[j])
                - f(u - steps[i] + steps[j])
                + f(u - steps[i] - steps[j])
            ) / (4.0 * h * h)
    return H


def _laplace(
    log_density: Callable[[np.ndarray], float], start: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the mode of the hyperparameters' log posterior in unbounded coordinates and the covariance of Laplace's approximation there.

    The mode by L-BFGS-B from `start` (the priors' medians); the covariance
    is the inverse of the negative log density's Hessian, by central
    differences. A direction the Hessian does not curve up in (a flat or
    bent posterior) gets a variance of 1 in its unbounded coordinate (a
    factor of :math:`e` in a positive hyperparameter), logged; the burn-in then
    scales the proposal.
    """

    def cost(u: np.ndarray) -> float:
        """Return the negative log density, large and finite where it has none (L-BFGS-B steps back from it)."""
        value = log_density(u)
        return -value if math.isfinite(value) else 1e300

    if not math.isfinite(log_density(start)):
        raise ValueError(
            "The hyperparameters' log posterior is not finite at the priors' medians "
            f"({dict(zip(names, start, strict=True))} in unbounded coordinates): the problem "
            "there did not solve, or a prior gives it no density."
        )
    found = minimize(cost, start, method="L-BFGS-B")
    if not found.success:
        logger.info(
            "The search for the hyperparameters' mode stopped: %s", found.message
        )
    mode = np.asarray(found.x, dtype=float)
    curvature = _hessian(cost, mode, 1e-3)
    curvature = 0.5 * (curvature + curvature.T)
    eigenvalues, vectors = np.linalg.eigh(curvature)
    flat = ~(eigenvalues > 0) | ~np.isfinite(eigenvalues)
    if flat.any():
        warn(
            logger,
            "The hyperparameters' log posterior does not curve down in %d of its %d "
            "directions at its mode (eigenvalues of the negative Hessian %s): "
            "the proposal takes a variance of 1 in them.",
            int(flat.sum()),
            len(eigenvalues),
            np.array2string(eigenvalues, precision=3),
        )
        eigenvalues = np.where(flat, 1.0, eigenvalues)
    return mode, (vectors / eigenvalues) @ vectors.T


def _effective_size(x: np.ndarray) -> float:
    """
    Return the effective sample size of the chain `x`, by Geyer's initial monotone sequence.

    The autocorrelations, by FFT, are summed in pairs while the pairs are
    positive, each pair at most the one before (Geyer, 1992); the size is
    the number of draws over the integrated autocorrelation time.
    """
    n = len(x)
    with np.errstate(all="ignore"):
        x = np.asarray(x, dtype=float) - np.mean(x)
        x = x / np.max(np.abs(x), initial=0.0)  # the FFT does not overflow
    if not np.all(np.isfinite(x)):
        return math.nan
    sum_sq = float(x @ x)
    if n < 4 or not sum_sq > 0:
        return float(n)
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), 2 * n)[:n] / sum_sq
    pairs = acf[0 : n - 1 : 2] + acf[1:n:2]
    negative = np.flatnonzero(pairs <= 0)
    pairs = pairs[: negative[0] if len(negative) else len(pairs)]
    tau = -1.0 + 2.0 * float(np.minimum.accumulate(pairs).sum())
    return n / max(tau, 1.0 / n)


def _count(value: Any, what: str, least: int) -> int:
    """Return `value` as an int, or raise unless it is an integer of at least `least`."""
    if (
        not isinstance(value, numbers.Integral)
        or isinstance(value, bool)
        or value < least
    ):
        raise ValueError(
            f"{what} must be an integer of at least {least}, got {value!r}."
        )
    return int(value)


def hierarchical(
    build: "Callable[[dict[str, float]], Problem | Posterior] | Problem",
    priors: Mapping[str, _Distribution],
    *,
    draws: int = 2000,
    burn_in: int = 500,
    seed: int = 0,
) -> "Hierarchical":
    r"""
    Draw the covariance hyperparameters from their posterior, and the state at each draw: the posterior with the hyperparameters' uncertainty.

    The hyperparameters get priors (`priors`), and their posterior is
    :math:`p(s \mid z) \propto p(s) \, p(z \mid s)`, with :math:`p(z \mid s)`
    the exact evidence of the problem `build` makes at :math:`s`
    (:attr:`Posterior.log_evidence <fips.Posterior.log_evidence>`, the
    analytic method's). The sampler is a random-walk Metropolis in unbounded
    coordinates (the log of a positive hyperparameter, the logit of a bounded one):
    it starts at the posterior mode of the hyperparameters (L-BFGS-B from the
    priors' medians), proposes from Laplace's approximation there (the
    inverse of a numerical Hessian), and during the `burn_in` scales that
    proposal towards an acceptance of 0.3 (a Robbins-Monro step on the log of
    its scale; fixed after, so the kept chain is a Markov chain of the exact
    posterior). Each step solves one problem. At each of the `draws` kept
    steps, it draws the state exactly from the posterior at the chain's
    hyperparameters (:meth:`Posterior.sample <fips.Posterior.sample>`), so the
    state's draws are draws of :math:`p(x \mid z)`, the Gaussian posteriors
    mixed over the hyperparameters' posterior.

    **The fast path.** When the only hyperparameters are a factor on the prior's
    standard deviation and one on the observation error's, pass the problem
    at factors of 1 as `build`, and name the hyperparameters ``"prior_scale"`` (the
    prior std times it) and ``"obs_scale"`` (the observation error's std times it), or
    ``"gamma"`` instead of ``"obs_scale"`` (the observation covariance over it).
    One eigendecomposition then gives the evidence at every step, in
    :math:`O(m)` (or :math:`O(n)`) with no solve, and the state's draws at
    every kept step in one product: a run costs about one eigendecomposition
    of the problem and one product for the draws of the state.

    Parameters
    ----------
    build : callable or Problem
        The function from the hyperparameters (a dict of floats, by name) to the
        :class:`~fips.Problem` they make, or to its analytic
        :class:`~fips.Posterior` (``fips.solve(problem, gamma=s["gamma"])``,
        to make gamma a hyperparameter). Or, for the fast path, the problem itself.
    priors : mapping of str to a distribution
        The prior of each hyperparameter, by the name `build` reads it by: a
        :class:`~fips.distributions.LogNormal`,
        :class:`~fips.distributions.HalfNormal` or
        :class:`~fips.distributions.Uniform`. Its
        order is the columns' of :attr:`Hierarchical.hyperparameters`.
    draws : int, default 2000
        The number of kept steps, each with one draw of the state.
    burn_in : int, default 500
        The steps before them, which tune the proposal and are not kept.
    seed : int, default 0
        The seed of the chain and of the state's draws, so that a run is
        reproducible. The two have their own streams: the chain is the same
        whatever the state's draws are.

    Returns
    -------
    Hierarchical
        The draws of the hyperparameters and of the state, and the diagnostics of
        the chain.

    Raises
    ------
    TypeError
        If a hyperparameter's prior is not one of :mod:`fips.distributions`, `build`
        is neither callable nor a problem, or it makes neither a problem nor
        an analytic posterior.
    ValueError
        If `priors` is empty, the fast path is given other names than its
        three (or both ``"obs_scale"`` and ``"gamma"``), `draws`, `burn_in` or
        `seed` is not an integer in range, or the log posterior is not finite
        at the priors' medians.

    See Also
    --------
    fips.Posterior.log_evidence : The evidence of one problem, which the chain reads.
    fips.Posterior.sample : The state's draws at one hyperparameter.

    References
    ----------
    Gelman, A., Roberts, G. O., and Gilks, W. R. (1996). Efficient
    Metropolis jumping rules. *Bayesian Statistics 5*, 599-607.

    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
    Science*, 7(4), 473-483.

    Examples
    --------
    The fast path: a factor on the prior std and one on the observation error's, on a
    problem of 60 observations of 20 cells, whose truth has a std of 2 and
    whose observations an error of 0.5:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> rng = np.random.default_rng(2)
    >>> cells = fips.Space({"flux": pd.RangeIndex(20, name="cell")})
    >>> observed = fips.Space({"ch4": pd.RangeIndex(60, name="obs_id")})
    >>> H = fips.Operator(rng.uniform(0.0, 1.0, (60, 20)), observed, cells)
    >>> truth = fips.Vector(2.0 * rng.standard_normal(20), cells)
    >>> obs = H @ truth + fips.Vector(0.5 * rng.standard_normal(60), observed)
    >>> problem = fips.Problem(
    ...     prior=fips.Vector(np.zeros(20), cells),
    ...     prior_cov=fips.Independent().with_std(1.0),
    ...     obs=obs,
    ...     obs_cov=fips.Independent().with_std(1.0),
    ...     forward=H,
    ... )
    >>> result = fips.hierarchical(
    ...     problem,
    ...     {
    ...         "prior_scale": fips.distributions.LogNormal(0.0, 1.0),
    ...         "obs_scale": fips.distributions.LogNormal(0.0, 1.0),
    ...     },
    ...     draws=1000,
    ...     burn_in=300,
    ... )
    >>> result.hyperparameters.shape
    (1000, 2)
    >>> low, high = result.hyperparameters.quantile([0.05, 0.95]).values
    >>> bool(np.all((low < [2.0, 0.5]) & (high > [2.0, 0.5])))  # 90% intervals
    True

    The general path is a function from the hyperparameters to the problem; here
    the same problem, so the same chain:

    >>> def build(s):
    ...     return fips.Problem(
    ...         prior=problem.prior,
    ...         prior_cov=fips.Independent().with_std(s["prior_std"]),
    ...         obs=obs,
    ...         obs_cov=fips.Independent().with_std(s["obs_std"]),
    ...         forward=H,
    ...     )
    >>> general = fips.hierarchical(
    ...     build,
    ...     {
    ...         "prior_std": fips.distributions.LogNormal(0.0, 1.0),
    ...         "obs_std": fips.distributions.LogNormal(0.0, 1.0),
    ...     },
    ...     draws=1000,
    ...     burn_in=300,
    ... )
    >>> bool(
    ...     np.allclose(
    ...         general.hyperparameters.values, result.hyperparameters.values, rtol=1e-3
    ...     )
    ... )
    True
    """
    if not isinstance(priors, Mapping) or not priors:
        raise ValueError(
            "priors maps each hyperparameter's name to its prior, such as "
            '{"prior_std": fips.distributions.LogNormal(0.0, 0.5)}; it is empty or not a mapping.'
        )
    names = list(priors)
    distributions = [priors[name] for name in names]
    for name, prior in zip(names, distributions, strict=True):
        if not isinstance(name, str):
            raise TypeError(f"A hyperparameter's name is a str, got {name!r}.")
        if not isinstance(prior, _Distribution):
            raise TypeError(
                f"The prior of hyperparameter {name!r} is a fips.distributions.LogNormal, "
                f"HalfNormal or Uniform, got {type(prior).__name__}."
            )
    draws = _count(draws, "draws", 1)
    burn_in = _count(burn_in, "burn_in", 0)
    seed = _count(seed, "seed", 0)
    target: _Built | _Scales
    if isinstance(build, Problem):
        unknown = [name for name in names if name not in _SCALES]
        if unknown:
            raise ValueError(
                f"Given a problem, hierarchical samples only the scales of its "
                f"covariances, named {', '.join(map(repr, _SCALES))}; it has no "
                f"{', '.join(map(repr, unknown))}. For other hyperparameters, pass a "
                "function that builds the problem from them."
            )
        if "obs_scale" in names and "gamma" in names:
            raise ValueError(
                "obs_scale and gamma both scale the observation covariance: give one."
            )
        target = _Scales(build)
    elif callable(build):
        target = _Built(build)
    else:
        raise TypeError(
            "build is a function from the hyperparameters to a Problem, or a Problem "
            f"(the fast path), got {type(build).__name__}."
        )

    def evaluate(u: np.ndarray) -> tuple[float, Any]:
        """Return the hyperparameters' log posterior at unbounded coordinates `u`, and what a draw of the state there needs."""
        log_prior = sum(
            p._log_density_unbounded(float(x))
            for p, x in zip(distributions, u, strict=True)
        )
        if not math.isfinite(log_prior):
            return -math.inf, None
        values = {
            n: p._from_unbounded(float(x))
            for n, p, x in zip(names, distributions, u, strict=True)
        }
        log_evidence, payload = target.evaluate(values)
        return log_prior + log_evidence, payload

    chain_rng, state_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )
    start = np.array([p._to_unbounded(p.median) for p in distributions])
    mode, laplace = _laplace(lambda u: evaluate(u)[0], start, names)
    root = np.linalg.cholesky(laplace)
    log_scale = math.log(2.38 / math.sqrt(len(names)))
    u = mode
    log_density, payload = evaluate(u)
    chain = np.empty((draws, len(names)))
    kept: list[Any] = []
    accepted = 0
    for step in range(burn_in + draws):
        proposal = u + math.exp(log_scale) * (root @ chain_rng.standard_normal(len(u)))
        proposed, proposed_payload = evaluate(proposal)
        log_ratio = proposed - log_density if math.isfinite(proposed) else -math.inf
        accept = _accept(log_ratio, chain_rng)
        if accept:
            u, log_density, payload = proposal, proposed, proposed_payload
        if step < burn_in:
            probability = math.exp(min(0.0, log_ratio))
            log_scale += (probability - _ACCEPTANCE) / (step + 1) ** 0.6
        else:
            chain[step - burn_in] = u
            kept.append(target.keep(payload, state_rng))
            accepted += accept
    values = np.array(
        [
            [
                p._from_unbounded(float(x))
                for p, x in zip(distributions, row, strict=True)
            ]
            for row in chain
        ]
    )
    index = pd.RangeIndex(draws, name="draw")
    space, states = target.stack(kept, state_rng)
    result = Hierarchical(
        target=target,
        names=names,
        hyperparameters=pd.DataFrame(
            values, index=index, columns=pd.Index(names, name="hyperparameter")
        ),
        space=space,
        states=states,
        mode=pd.Series(
            [
                p._from_unbounded(float(x))
                for p, x in zip(distributions, mode, strict=True)
            ],
            index=pd.Index(names, name="hyperparameter"),
            name="mode",
        ),
        acceptance=accepted / draws,
        proposal_scale=math.exp(log_scale),
    )
    logger.info(
        "hierarchical: %d draws after %d burn-in steps; acceptance %.2f; "
        "effective sample sizes %s.",
        draws,
        burn_in,
        result.acceptance,
        result.ess.round(0).to_dict(),
    )
    return result


# -- the result -----------------------------------------------------------------


class Hierarchical:
    r"""
    Draws of the covariance hyperparameters from their posterior, and of the state at each: the posterior with the hyperparameters' uncertainty.

    :func:`hierarchical` builds it. It is not a :class:`~fips.Posterior`: the
    posterior of the state is a mixture of Gaussians, one for each draw of the
    hyperparameters, so its estimate, variances and covariances are those of the
    draws of the state (``draws`` of them: a Monte Carlo estimate, whose
    error falls as the effective number of draws, :attr:`ess`, grows). What is defined only for one Gaussian (the gain,
    DOFS, the chi-squared tests, the evidence) is on the posterior at one
    hyperparameter, :meth:`at`. The draws of the state are held as a ``state x
    draws`` array.

    Pickles with :mod:`pickle` when the function that builds the problem does
    (a function defined at the top level of a module).
    """

    def __init__(
        self,
        *,
        target: "_Built | _Scales",
        names: list[str],
        hyperparameters: pd.DataFrame,
        space: Space,
        states: np.ndarray,
        mode: pd.Series,
        acceptance: float,
        proposal_scale: float,
    ):
        self._target = target
        self._names = names
        self._hyperparameters = hyperparameters
        self._space = space
        self._states = states
        self._states.flags.writeable = False
        self._mode = mode
        self._acceptance = acceptance
        self._proposal_scale = proposal_scale

    def __repr__(self) -> str:
        """Return the hyperparameters, the number of draws and the acceptance."""
        return (
            f"Hierarchical(hyperparameters={self._names}, draws={len(self._hyperparameters)}, "
            f"acceptance={self._acceptance:.2f})"
        )

    # -- the hyperparameters ----------------------------------------------------------

    @property
    def hyperparameters(self) -> pd.DataFrame:
        """The kept draws of the hyperparameters: one row a draw (``draw``), one column a hyperparameter."""
        return self._hyperparameters

    @property
    def mode(self) -> pd.Series:
        """The posterior mode of the hyperparameters, where the chain started and Laplace's approximation was taken."""
        return self._mode

    @property
    def acceptance(self) -> float:
        """The fraction of the kept steps whose proposal was accepted (near 0.3 when the burn-in tuned the proposal well)."""
        return self._acceptance

    @property
    def ess(self) -> pd.Series:
        """
        The effective sample size of each hyperparameter's draws: about as many independent draws as they are worth.

        By Geyer's initial monotone sequence of autocorrelations. A Monte
        Carlo standard error of a hyperparameter's mean is its standard deviation
        over the root of this.
        """
        return pd.Series(
            [
                _effective_size(self._hyperparameters[name].to_numpy(dtype=float))
                for name in self._names
            ],
            index=self._hyperparameters.columns,
            name="ess",
        )

    def at(self, hyperparameters: "Mapping[str, float] | pd.Series") -> Posterior:
        """
        Return the posterior at one hyperparameter of every hyperparameter: the ordinary :class:`~fips.Posterior`, solved.

        For what the mixture does not define: the gain, DOFS, the chi-squared
        tests, the evidence. ``result.at(result.hyperparameters.median())``.

        Parameters
        ----------
        hyperparameters : mapping or pandas.Series
            A value for each hyperparameter, by name.

        Raises
        ------
        ValueError
            If a hyperparameter is missing or unknown.
        """
        values = {str(k): float(v) for k, v in dict(hyperparameters).items()}
        missing = [name for name in self._names if name not in values]
        unknown = [name for name in values if name not in self._names]
        if missing or unknown:
            raise ValueError(
                f"at takes a value for each of the hyperparameters {self._names}; "
                f"missing {missing}, unknown {unknown}."
            )
        return self._target.posterior({name: values[name] for name in self._names})

    # -- the state -------------------------------------------------------------

    def _aggregation(self, W: Operator) -> Any:
        """Return an aggregation's matrix with its columns in the state's order (as :class:`~fips.Posterior` checks it)."""
        if not isinstance(W, Operator):
            raise TypeError(
                f"An aggregation is an Operator, not {type(W).__name__}: build one "
                "with Operator.from_frame or Operator.from_blocks."
            )
        return W._on(W.rows, self._space, "the aggregation", ("its rows", "the state"))

    def _draws(self, W: Operator | None) -> tuple[np.ndarray, pd.Index]:
        """Return the draws of the state (or of `W` x) and their row labels."""
        if W is None:
            return self._states, self._space.index
        return np.asarray(self._aggregation(W) @ self._states), W.rows.index

    @property
    def estimate(self) -> Vector:
        r"""The posterior mean of the state, the mean of its draws: :math:`E[\hat{x}(s)]` over the hyperparameters' posterior."""
        return Vector(self._states.mean(axis=1), self._space, name="estimate")

    @property
    def variances(self) -> Vector:
        r"""
        The posterior variances of the state, those of its draws (about their mean).

        By the law of total variance they are :math:`E[\operatorname{diag}
        \hat{S}(s)] + \operatorname{var}[\hat{x}(s)]` over the hyperparameters'
        posterior: the mean of each hyperparameter's variances, and the spread of
        its estimates.
        """
        return Vector(self._states.var(axis=1, ddof=1), self._space, name="variance")

    def cov(self, W: Operator | None = None) -> Covariance:
        """
        Return the posterior covariance of the state, or of aggregates of it, from the draws.

        Parameters
        ----------
        W : Operator, optional
            An aggregation, as :meth:`Posterior.cov <fips.Posterior.cov>`
            takes it: one row per aggregate, the state's labels as its
            columns (in any order).

        Returns
        -------
        Covariance
            The covariance of the draws of :math:`x` (``n x n``, dense) or of
            :math:`W x` (``k x k``).

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If the columns of `W` do not have exactly the state's labels.
        """
        values, _ = self._draws(W)
        space = self._space if W is None else W.rows
        covariance = np.atleast_2d(np.cov(values, ddof=1))
        return Covariance._computed(_adopt(covariance), space, name="posterior_cov")

    def uncertainty(
        self, W: Operator | None = None, level: float = 0.95
    ) -> pd.DataFrame:
        """
        Return the mean, standard deviation and central interval of each element or aggregate, from the draws.

        Parameters
        ----------
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it. By default, one row per
            state element.
        level : float, default 0.95
            The probability of the interval, between 0 and 1: the interval
            is from the ``(1 - level) / 2`` quantile of the draws to the
            ``(1 + level) / 2`` one.

        Returns
        -------
        pandas.DataFrame
            One row per element or aggregate: ``estimate`` (the mean of the
            draws), ``posterior_std``, ``lower`` and ``upper``. There is no
            prior column: the prior of the state depends on the hyperparameters.

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If `level` is not between 0 and 1, or the columns of `W` do not
            have exactly the state's labels.
        """
        if not (isinstance(level, numbers.Real) and 0.0 < level < 1.0):
            raise ValueError(f"level is a probability between 0 and 1, got {level!r}.")
        values, index = self._draws(W)
        lower, upper = np.quantile(
            values, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=1
        )
        return pd.DataFrame(
            {
                "estimate": values.mean(axis=1),
                "posterior_std": values.std(axis=1, ddof=1),
                "lower": lower,
                "upper": upper,
            },
            index=index,
        )

    def sample(
        self,
        n: int,
        rng: "int | np.random.Generator | None" = None,
        *,
        W: Operator | None = None,
    ) -> pd.DataFrame:
        """
        Return `n` of the kept draws of the state, mixed over the hyperparameters' draws: one column each.

        The draws are those made at the kept steps of the chain, one at each
        step's hyperparameters, so a column's draw belongs to the row of
        :attr:`hyperparameters` its label names. They are chosen at random,
        without repeats, and listed in the order of the chain; all of them,
        when `n` is the number kept.

        Parameters
        ----------
        n : int
            The number of draws, from 1 to the number kept.
        rng : int or numpy.random.Generator, optional
            Chooses which draws, as in :meth:`Posterior.sample
            <fips.Posterior.sample>`: a seed or a generator, or none for fresh
            entropy.
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it: the draws are then those
            of :math:`W x`.

        Returns
        -------
        pandas.DataFrame
            One row per state element, labeled as the :attr:`estimate` is (or
            one per row of `W`), and one column per draw, labeled by its row
            of :attr:`hyperparameters` (``draw``).

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If `n` is not an integer from 1 to the number of kept draws, or the
            columns of `W` do not have exactly the state's labels.
        """
        kept = self._states.shape[1]
        if (
            not isinstance(n, numbers.Integral)
            or isinstance(n, bool)
            or not 1 <= int(n) <= kept
        ):
            raise ValueError(
                f"n is a number of draws from 1 to the {kept} kept, got {n!r}; "
                "for more, run hierarchical with more draws."
            )
        values, index = self._draws(W)
        chosen = np.sort(
            np.random.default_rng(rng).choice(kept, size=int(n), replace=False)
        )
        return pd.DataFrame(
            values[:, chosen],
            index=index,
            columns=pd.Index(self._hyperparameters.index[chosen], name="draw"),
        )


# -- a grid of hyperparameters -----------------------------------------------------

_COLUMNS = (
    "gamma",
    "dofs",
    "information_content",
    "chi2",
    "reduced_chi2",
    "data_misfit",
    "state_misfit",
    "log_evidence",
    "r2",
    "rmse",
    "uncertainty_reduction",
)
"""The columns of :attr:`Sweep.table`: each a scalar of the member's :class:`~fips.Posterior`."""


def _diagnostic(post: Posterior, name: str) -> float:
    """Return the posterior's scalar `name`, or NaN where its method cannot give it."""
    try:
        return float(post._gamma if name == "gamma" else getattr(post, name))
    except UnavailableError:
        return math.nan


def _grid(grid: Any) -> pd.Index:
    """Return the members of `grid`, every combination of its values, as an index named by the hyperparameters (plain for one)."""
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError(
            "grid maps each hyperparameter's name to its values, such as "
            '{"prior_scale": [0.5, 1.0, 2.0]}; it is empty or not a mapping.'
        )
    names, levels = [], []
    for name, values in grid.items():
        if not isinstance(name, str):
            raise TypeError(f"A hyperparameter's name is a str, got {name!r}.")
        if isinstance(values, (str, bytes)) or not isinstance(
            values, (Sequence, np.ndarray, pd.Index, pd.Series)
        ):
            raise TypeError(
                f"The values of hyperparameter {name!r} are a list, got {type(values).__name__}."
            )
        level = pd.Index(list(values))
        if level.empty:
            raise ValueError(f"Hyperparameter {name!r} has no values.")
        if level.has_duplicates:
            raise ValueError(
                f"Hyperparameter {name!r} repeats values "
                f"{level[level.duplicated()].unique().tolist()}: each member is labeled by its values."
            )
        names.append(name)
        levels.append(level)
    members = pd.MultiIndex.from_product(levels, names=names)
    return members.get_level_values(0) if len(names) == 1 else members


def _names(members: pd.Index) -> list[str]:
    """Return the hyperparameters' names, the levels of a grid's index (each a str, as :func:`_grid` checked)."""
    return [str(name) for name in members.names]


def _values(names: list[str], label: Any) -> dict[str, Any]:
    """Return a member's hyperparameters by name, from its label (a tuple, or the value itself for one hyperparameter)."""
    return dict(zip(names, label if len(names) > 1 else (label,), strict=True))


def _scales_of(values: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return the prior std's factor, the observation error std's factor and gamma of one member of a grid on a problem."""
    return (
        float(values.get("prior_scale", 1.0)),
        float(values.get("obs_scale", 1.0)),
        float(values.get("gamma", 1.0)),
    )


def _check_scales(members: pd.Index) -> None:
    """Raise unless a grid on a problem names only the scales, not both obs_scale and gamma, each a positive number."""
    names = _names(members)
    unknown = [name for name in names if name not in _SCALES]
    if unknown:
        raise ValueError(
            f"Given a problem, sweep varies only the scales of its covariances, "
            f"named {', '.join(map(repr, _SCALES))}; it has no "
            f"{', '.join(map(repr, unknown))}. For other hyperparameters, pass a "
            "function that builds the problem from them."
        )
    if "obs_scale" in names and "gamma" in names:
        raise ValueError(
            "obs_scale and gamma both scale the observation covariance: give one."
        )
    for name in names:
        values = members.get_level_values(name).unique()
        bad = [
            v
            for v in values
            if not isinstance(v, numbers.Real) or not 0.0 < float(v) < math.inf
        ]
        if bad:
            raise ValueError(f"{name} takes positive numbers, got {bad}.")


def _member_posterior(build: Any, values: dict[str, Any]) -> Posterior:
    """Return the posterior of one member: the problem `build` makes, solved, or the posterior it gives."""
    if isinstance(build, Problem):
        prior_scale, obs_scale, gamma = _scales_of(values)
        problem = _scaled(build, prior_scale**2, obs_scale**2)
        return solve(problem, gamma=gamma)
    made = build(dict(values))
    if isinstance(made, Problem):
        return solve(made)
    if isinstance(made, Posterior):
        return made
    raise TypeError(
        "The function sweep takes builds a Problem (or its Posterior) from a dict "
        f"of hyperparameters; it returned {type(made).__name__}."
    )


def _posteriors(build: Any, members: pd.Index) -> "Iterator[tuple[Any, Posterior]]":
    """
    Yield each member's label and posterior.

    On a problem, the members that differ only in gamma share one solve:
    the first is solved at its gamma, the others are its
    :meth:`~fips.Posterior.with_gamma`, which eigendecomposes once.
    """
    names = _names(members)
    if not isinstance(build, Problem):
        for label in members:
            yield label, _member_posterior(build, _values(names, label))
        return
    groups: dict[tuple[float, float], list[tuple[Any, float]]] = {}
    for label in members:
        prior_scale, obs_scale, gamma = _scales_of(_values(names, label))
        groups.setdefault((prior_scale, obs_scale), []).append((label, gamma))
    for (prior_scale, obs_scale), group in groups.items():
        problem = _scaled(build, prior_scale**2, obs_scale**2)
        post: Posterior | None = None
        for label, gamma in group:
            post = (
                solve(problem, gamma=gamma) if post is None else post.with_gamma(gamma)
            )
            yield label, post


def sweep(
    build: "Callable[[dict[str, Any]], Problem | Posterior] | Problem",
    grid: Mapping[str, Any],
) -> "Sweep":
    r"""
    Solve the problem at every combination of the hyperparameters' values: a grid of posteriors, their diagnostics, and the spread of their estimates.

    `grid` gives each hyperparameter a list of values; every combination of
    them is a *member*, and the member's posterior is the problem `build`
    makes from them, solved. The result holds a table of each member's
    diagnostics (the reduced chi-squared, the misfits, the log evidence, the
    DOFS), its estimate and its posterior variances, and nothing larger:
    :meth:`Sweep.at` solves one member again for the rest.

    Which members to keep is a filter on :attr:`Sweep.table`: the member
    of highest evidence, those within a few units of log evidence of it,
    those whose reduced chi-squared is near 1, or whose state misfit per
    state element is in a window (as the Integrated Methane Inversion does,
    Varon et al., 2022). :meth:`Sweep.uncertainty` gives the mean of the kept
    members' estimates, their posterior standard deviation, and the spread
    of their estimates, a second uncertainty: how much the answer moves
    with hyperparameters the data do not decide between.

    **On a problem.** When the hyperparameters are a factor on the prior's standard
    deviation and one on the observation error's, pass the problem at factors of 1
    as `build`, and name them ``"prior_scale"`` and ``"obs_scale"``, or
    ``"gamma"`` (the observation covariance over it) instead of
    ``"obs_scale"``, as :func:`hierarchical` does. The members that differ only
    in gamma then share one solve (:meth:`Posterior.with_gamma
    <fips.Posterior.with_gamma>`): a grid over gamma alone costs about two solves.

    Parameters
    ----------
    build : callable or Problem
        The function from the hyperparameters (a dict, by name) to the
        :class:`~fips.Problem` they make, or to its :class:`~fips.Posterior`
        (``fips.solve(problem, gamma=s["gamma"])``, or another method's). Or
        the problem itself, with a grid of its scales.
    grid : mapping of str to list
        The values of each hyperparameter, by the name `build` reads it by.
        Its order is the order of the levels of :attr:`Sweep.table`'s index,
        and the last varies fastest.

    Returns
    -------
    Sweep
        The members' diagnostics, estimates and variances.

    Raises
    ------
    TypeError
        If a name is not a str, a hyperparameter's values are not a list, or
        `build` makes neither a problem nor a posterior.
    ValueError
        If `grid` is empty, a hyperparameter has no values or repeats one, or,
        given a problem, a name is not one of its three scales, both
        ``"obs_scale"`` and ``"gamma"`` are given, or a scale is not a positive number.

    See Also
    --------
    hierarchical : The hyperparameters drawn from their posterior instead, each with its weight.
    fips.Posterior.with_gamma : The posterior at another gamma.
    fips.Posterior.summary : One posterior's diagnostics as a table row.

    References
    ----------
    Varon, D. J., Jacob, D. J., Sulprizio, M., et al. (2022). Integrated
    Methane Inversion (IMI 1.0). *Geoscientific Model Development*, 15,
    5787-5805. https://doi.org/10.5194/gmd-15-5787-2022

    Examples
    --------
    A problem of 60 observations of 20 cells, whose truth has a std of 2 and
    whose observations an error of 0.5, solved with a prior std of 1 and an
    observation error of 1, and a grid of factors on each:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import fips
    >>> rng = np.random.default_rng(2)
    >>> cells = fips.Space({"flux": pd.RangeIndex(20, name="cell")})
    >>> observed = fips.Space({"ch4": pd.RangeIndex(60, name="obs_id")})
    >>> H = fips.Operator(rng.uniform(0.0, 1.0, (60, 20)), observed, cells)
    >>> truth = fips.Vector(2.0 * rng.standard_normal(20), cells)
    >>> obs = H @ truth + fips.Vector(0.5 * rng.standard_normal(60), observed)
    >>> problem = fips.Problem(
    ...     prior=fips.Vector(np.zeros(20), cells),
    ...     prior_cov=fips.Independent().with_std(1.0),
    ...     obs=obs,
    ...     obs_cov=fips.Independent().with_std(1.0),
    ...     forward=H,
    ... )
    >>> result = fips.sweep(
    ...     problem, {"prior_scale": [1.0, 2.0, 4.0], "obs_scale": [0.25, 0.5, 1.0]}
    ... )
    >>> result.table[["reduced_chi2", "log_evidence"]].round(
    ...     2
    ... )  # doctest: +NORMALIZE_WHITESPACE
                           reduced_chi2  log_evidence
    prior_scale obs_scale
    1.0         0.25               4.14       -140.51
                0.50               1.98       -103.99
                1.00               1.23       -110.74
    2.0         0.25               3.11       -123.32
                0.50               1.03        -88.96
                1.00               0.50       -101.00
    4.0         0.25               2.85       -129.24
                0.50               0.78        -94.97
                1.00               0.26       -107.27

    The member of highest evidence has the true factors, 2 and 0.5. A
    reduced chi-squared near 1 keeps three members, with priors a factor of
    four apart; the spread of their estimates is the answer's dependence on
    that choice:

    >>> kept = result.table.index[abs(result.table["reduced_chi2"] - 1.0) < 0.3]
    >>> result.uncertainty(kept).head(3).round(3)  # doctest: +NORMALIZE_WHITESPACE
                estimate  posterior_std  spread
    block cell
    flux  0        0.562          0.333   0.282
          1        1.165          0.313   0.198
          2       -0.672          0.326   0.146
    """
    members = _grid(grid)
    if isinstance(build, Problem):
        _check_scales(members)
    elif not callable(build):
        raise TypeError(
            "build is a function from the hyperparameters to a Problem, or a Problem "
            f"(a grid of its scales), got {type(build).__name__}."
        )
    rows: dict[Any, list[float]] = {}
    estimates: dict[Any, np.ndarray] = {}
    variances: dict[Any, np.ndarray] = {}
    space: Space | None = None
    for label, post in _posteriors(build, members):
        rows[label] = [_diagnostic(post, name) for name in _COLUMNS]
        own = post.problem.prior.space
        if space is None:
            space = own
        order = (
            slice(None)
            if own is space or own == space
            else space._align(
                own,
                "the state of a member's problem",
                "The state of the problems built at two members",
            )
        )
        estimates[label] = post.estimate.values[order]
        variances[label] = post.variances.values[order]
    assert space is not None  # a grid has a member
    labels = list(members)
    result = Sweep(
        build=build,
        table=pd.DataFrame(
            [rows[label] for label in labels], index=members, columns=list(_COLUMNS)
        ),
        estimates=pd.DataFrame(
            np.column_stack([estimates[label] for label in labels]),
            index=space.index,
            columns=members,
        ),
        variances=pd.DataFrame(
            np.column_stack([variances[label] for label in labels]),
            index=space.index,
            columns=members,
        ),
    )
    logger.info("sweep: %d members of %s.", len(members), list(members.names))
    return result


class Sweep:
    """
    The posteriors of a grid of hyperparameters: each member's diagnostics, estimate and variances.

    :func:`sweep` builds it. A member is one combination of the
    hyperparameters' values, labeled by them (a tuple, or the value itself for
    one hyperparameter) in :attr:`table`'s index and in the columns of
    :attr:`estimates` and :attr:`variances`. Choose members by filtering
    :attr:`table`, then pass their labels to :meth:`uncertainty`, or one
    label to :meth:`at` for that member's whole :class:`~fips.Posterior`.

    Pickles with :mod:`pickle` when the function that builds the problem does
    (a function defined at the top level of a module).
    """

    def __init__(
        self,
        *,
        build: Any,
        table: pd.DataFrame,
        estimates: pd.DataFrame,
        variances: pd.DataFrame,
    ):
        self._build = build
        self._table = table
        self._estimates = estimates
        self._variances = variances

    def __repr__(self) -> str:
        """Return the hyperparameters and the number of members."""
        return f"Sweep(hyperparameters={list(self._table.index.names)}, members={len(self._table)})"

    @property
    def table(self) -> pd.DataFrame:
        """
        The diagnostics of each member: one row a member, labeled by its hyperparameters.

        The columns are the scalars of its :class:`~fips.Posterior`:
        ``gamma``, ``dofs``, ``information_content``, ``chi2``,
        ``reduced_chi2``, ``data_misfit``, ``state_misfit``,
        ``log_evidence``, ``r2``, ``rmse`` and ``uncertainty_reduction``. A
        scalar the member's method cannot give is NaN (the log evidence of an
        ensemble solve).
        """
        return self._table

    @property
    def estimates(self) -> pd.DataFrame:
        """The estimate of each member: one row a state element, one column a member."""
        return self._estimates

    @property
    def variances(self) -> pd.DataFrame:
        """The posterior variances of each member: one row a state element, one column a member."""
        return self._variances

    def _positions(self, members: Any) -> np.ndarray:
        """Return the positions of the members labeled `members` (all of them for None), or raise naming those the grid has not."""
        index = self._table.index
        if members is None:
            return np.arange(len(index))
        if isinstance(index, pd.MultiIndex) and isinstance(members, tuple):
            members = [members]
        labels = (
            members
            if isinstance(members, (pd.Index, list))
            else list(np.atleast_1d(members))
        )
        positions = np.asarray(index.get_indexer(pd.Index(labels)))
        if positions.size == 0 or (positions < 0).any():
            missing = [
                label for label, at in zip(labels, positions, strict=True) if at < 0
            ]
            raise ValueError(
                "members are labels of the sweep's table, such as "
                f"{index[0]!r}; it has no {missing if missing else 'members given'}."
            )
        return positions

    def uncertainty(self, members: Any = None) -> pd.DataFrame:
        r"""
        Return the mean of the members' estimates, their posterior standard deviation, and the spread of their estimates.

        Parameters
        ----------
        members : list or pandas.Index, optional
            The labels of the members to keep, from :attr:`table`'s index
            (``result.table.index[mask]``). By default, every member.

        Returns
        -------
        pandas.DataFrame
            One row per state element: ``estimate`` (the mean of the
            members' estimates), ``posterior_std`` (the root of the mean of
            their posterior variances) and ``spread`` (the standard deviation
            of their estimates, over the members as a whole set, so 0 for one).
            The members mixed with equal weights have a variance of
            ``posterior_std**2 + spread**2`` (the law of total variance).

        Raises
        ------
        ValueError
            If a label is not a member's, or none is given.
        """
        at = self._positions(members)
        estimates = self._estimates.to_numpy()[:, at]
        return pd.DataFrame(
            {
                "estimate": estimates.mean(axis=1),
                "posterior_std": np.sqrt(
                    self._variances.to_numpy()[:, at].mean(axis=1)
                ),
                "spread": estimates.std(axis=1),
            },
            index=self._estimates.index,
        )

    def at(self, member: Any) -> Posterior:
        """
        Return the posterior of one member, solved again: for what the table does not hold.

        ``result.at(result.table["log_evidence"].idxmax())``.

        Parameters
        ----------
        member : tuple or scalar
            The member's label in :attr:`table`'s index: a tuple of the
            hyperparameters' values, or the value itself for one hyperparameter.

        Raises
        ------
        ValueError
            If `member` is not a member's label.
        """
        index = self._table.index
        position = int(
            self._positions(member if isinstance(member, tuple) else [member])[0]
        )
        return _member_posterior(self._build, _values(_names(index), index[position]))
