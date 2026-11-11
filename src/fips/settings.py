r"""
The covariance settings as unknowns: their priors, their posterior, and the state over them.

Every problem rests on settings someone chose (the prior's standard
deviation and correlation length, the observation error's standard
deviation). :func:`hierarchical` gives each setting a prior, draws the
settings from their posterior,

.. math::

    p(s \mid z) \propto p(s) \, p(z \mid s),

with :math:`p(z \mid s)` the exact evidence of the problem built at
:math:`s` (:attr:`Posterior.log_evidence <fips.Posterior.log_evidence>`),
and draws the state exactly from :math:`x \mid s, z` at each kept draw of the
settings (:meth:`Posterior.sample <fips.Posterior.sample>`). The draws of the
state are then draws of :math:`p(x \mid z)`, a mixture of the Gaussian
posteriors of every setting, and their intervals hold the uncertainty of the
settings. :class:`Hierarchical` holds them.

The settings' priors are the distributions of :mod:`fips.distributions`
(:class:`~fips.distributions.LogNormal`,
:class:`~fips.distributions.HalfNormal` and
:class:`~fips.distributions.Uniform`): each has a log density and a map to unbounded coordinates
(the log of a positive setting, the logit of a bounded one), where the
sampler works. The sampler is a random-walk Metropolis whose proposal is
Laplace's approximation of the settings' posterior at its mode (a numerical
Hessian), scaled during the burn-in towards an acceptance of 0.3 (Gelman,
Roberts and Gilks, 1996): a few settings need no gradient, and each step costs
one solve.
"""

import logging
import math
import numbers
from collections.abc import Callable, Mapping
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
from fips.posterior import Posterior
from fips.problem import Problem
from fips.space import Space
from fips.vector import Vector

logger = logging.getLogger(__name__)

_SCALES = ("prior_scale", "obs_scale", "gamma")
"""The settings the scales-only fast path takes, when :func:`hierarchical` is given a problem."""

_ACCEPTANCE = 0.3
"""The acceptance the burn-in steers the proposal's scale towards."""


# -- what one step of the chain solves ---------------------------------------


class _Built:
    """The general path: the caller's function from the settings to a problem (or a posterior), solved at each step."""

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
            f"Posterior) from a dict of settings; it returned {type(made).__name__}."
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
                    "The state of the problems built at two settings",
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
    The fast path: the settings scale the problem's prior std (``prior_scale``) and its observation covariance (``obs_scale``, or ``gamma``).

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
        problem = self.problem
        prior_cov, obs_cov = problem.prior_cov * (s * s), problem.obs_cov * c
        assert isinstance(prior_cov, Covariance) and isinstance(obs_cov, Covariance)
        return solve(
            Problem(
                prior=problem.prior,
                prior_cov=prior_cov,
                obs=problem.obs,
                obs_cov=obs_cov,
                forward=problem.forward,
                offset=problem.offset,
            )
        )

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
    Return the mode of the settings' log posterior in unbounded coordinates and the covariance of Laplace's approximation there.

    The mode by L-BFGS-B from `start` (the priors' medians); the covariance
    is the inverse of the negative log density's Hessian, by central
    differences. A direction the Hessian does not curve up in (a flat or
    bent posterior) gets a variance of 1 in its unbounded coordinate (a
    factor of :math:`e` in a positive setting), logged; the burn-in then
    scales the proposal.
    """

    def cost(u: np.ndarray) -> float:
        """Return the negative log density, large and finite where it has none (L-BFGS-B steps back from it)."""
        value = log_density(u)
        return -value if math.isfinite(value) else 1e300

    if not math.isfinite(log_density(start)):
        raise ValueError(
            "The settings' log posterior is not finite at the priors' medians "
            f"({dict(zip(names, start, strict=True))} in unbounded coordinates): the problem "
            "there did not solve, or a prior gives it no density."
        )
    found = minimize(cost, start, method="L-BFGS-B")
    if not found.success:
        logger.info("The search for the settings' mode stopped: %s", found.message)
    mode = np.asarray(found.x, dtype=float)
    curvature = _hessian(cost, mode, 1e-3)
    curvature = 0.5 * (curvature + curvature.T)
    eigenvalues, vectors = np.linalg.eigh(curvature)
    flat = ~(eigenvalues > 0) | ~np.isfinite(eigenvalues)
    if flat.any():
        warn(
            logger,
            "The settings' log posterior does not curve down in %d of its %d "
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
    settings: Mapping[str, _Distribution],
    *,
    draws: int = 2000,
    burn_in: int = 500,
    seed: int = 0,
) -> "Hierarchical":
    r"""
    Draw the covariance settings from their posterior, and the state at each draw: the posterior with the settings' uncertainty.

    The settings get priors (`settings`), and their posterior is
    :math:`p(s \mid z) \propto p(s) \, p(z \mid s)`, with :math:`p(z \mid s)`
    the exact evidence of the problem `build` makes at :math:`s`
    (:attr:`Posterior.log_evidence <fips.Posterior.log_evidence>`, the
    analytic method's). The sampler is a random-walk Metropolis in unbounded
    coordinates (the log of a positive setting, the logit of a bounded one):
    it starts at the posterior mode of the settings (L-BFGS-B from the
    priors' medians), proposes from Laplace's approximation there (the
    inverse of a numerical Hessian), and during the `burn_in` scales that
    proposal towards an acceptance of 0.3 (a Robbins-Monro step on the log of
    its scale; fixed after, so the kept chain is a Markov chain of the exact
    posterior). Each step solves one problem. At each of the `draws` kept
    steps, it draws the state exactly from the posterior at the chain's
    settings (:meth:`Posterior.sample <fips.Posterior.sample>`), so the
    state's draws are draws of :math:`p(x \mid z)`, the Gaussian posteriors
    mixed over the settings' posterior.

    **The fast path.** When the only settings are a factor on the prior's
    standard deviation and one on the observation error's, pass the problem
    at factors of 1 as `build`, and name the settings ``"prior_scale"`` (the
    prior std times it) and ``"obs_scale"`` (the observation error's std times it), or
    ``"gamma"`` instead of ``"obs_scale"`` (the observation covariance over it).
    One eigendecomposition then gives the evidence at every step, in
    :math:`O(m)` (or :math:`O(n)`) with no solve, and the state's draws at
    every kept step in one product: a run costs about one eigendecomposition
    of the problem and one product for the draws of the state.

    Parameters
    ----------
    build : callable or Problem
        The function from the settings (a dict of floats, by name) to the
        :class:`~fips.Problem` they make, or to its analytic
        :class:`~fips.Posterior` (``fips.solve(problem, gamma=s["gamma"])``,
        to make gamma a setting). Or, for the fast path, the problem itself.
    settings : mapping of str to a distribution
        The prior of each setting, by the name `build` reads it by: a
        :class:`~fips.distributions.LogNormal`,
        :class:`~fips.distributions.HalfNormal` or
        :class:`~fips.distributions.Uniform`. Its
        order is the columns' of :attr:`Hierarchical.settings`.
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
        The draws of the settings and of the state, and the diagnostics of
        the chain.

    Raises
    ------
    TypeError
        If a setting's prior is not one of :mod:`fips.distributions`, `build`
        is neither callable nor a problem, or it makes neither a problem nor
        an analytic posterior.
    ValueError
        If `settings` is empty, the fast path is given other names than its
        three (or both ``"obs_scale"`` and ``"gamma"``), `draws`, `burn_in` or
        `seed` is not an integer in range, or the log posterior is not finite
        at the priors' medians.

    See Also
    --------
    fips.Posterior.log_evidence : The evidence of one problem, which the chain reads.
    fips.Posterior.sample : The state's draws at one setting.

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
    >>> result.settings.shape
    (1000, 2)
    >>> low, high = result.settings.quantile([0.05, 0.95]).values
    >>> bool(np.all((low < [2.0, 0.5]) & (high > [2.0, 0.5])))  # 90% intervals
    True

    The general path is a function from the settings to the problem; here
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
    >>> bool(np.allclose(general.settings.values, result.settings.values, rtol=1e-3))
    True
    """
    if not isinstance(settings, Mapping) or not settings:
        raise ValueError(
            "settings maps each setting's name to its prior, such as "
            '{"prior_std": fips.distributions.LogNormal(0.0, 0.5)}; it is empty or not a mapping.'
        )
    names = list(settings)
    priors = [settings[name] for name in names]
    for name, prior in zip(names, priors, strict=True):
        if not isinstance(name, str):
            raise TypeError(f"A setting's name is a str, got {name!r}.")
        if not isinstance(prior, _Distribution):
            raise TypeError(
                f"The prior of setting {name!r} is a fips.distributions.LogNormal, "
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
                f"{', '.join(map(repr, unknown))}. For other settings, pass a "
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
            "build is a function from the settings to a Problem, or a Problem "
            f"(the fast path), got {type(build).__name__}."
        )

    def evaluate(u: np.ndarray) -> tuple[float, Any]:
        """Return the settings' log posterior at unbounded coordinates `u`, and what a draw of the state there needs."""
        log_prior = sum(
            p._log_density_unbounded(float(x)) for p, x in zip(priors, u, strict=True)
        )
        if not math.isfinite(log_prior):
            return -math.inf, None
        values = {
            n: p._from_unbounded(float(x))
            for n, p, x in zip(names, priors, u, strict=True)
        }
        log_evidence, payload = target.evaluate(values)
        return log_prior + log_evidence, payload

    chain_rng, state_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )
    start = np.array([p._to_unbounded(p.median) for p in priors])
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
            [p._from_unbounded(float(x)) for p, x in zip(priors, row, strict=True)]
            for row in chain
        ]
    )
    index = pd.RangeIndex(draws, name="draw")
    space, states = target.stack(kept, state_rng)
    result = Hierarchical(
        target=target,
        names=names,
        settings=pd.DataFrame(
            values, index=index, columns=pd.Index(names, name="setting")
        ),
        space=space,
        states=states,
        mode=pd.Series(
            [p._from_unbounded(float(x)) for p, x in zip(priors, mode, strict=True)],
            index=pd.Index(names, name="setting"),
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
    Draws of the covariance settings from their posterior, and of the state at each: the posterior with the settings' uncertainty.

    :func:`hierarchical` builds it. It is not a :class:`~fips.Posterior`: the
    posterior of the state is a mixture of Gaussians, one for each draw of the
    settings, so its estimate, variances and covariances are those of the
    draws of the state (``draws`` of them: a Monte Carlo estimate, whose
    error falls as the effective number of draws, :attr:`ess`, grows). What is defined only for one Gaussian (the gain,
    DOFS, the chi-squared tests, the evidence) is on the posterior at one
    setting, :meth:`at`. The draws of the state are held as a ``state x
    draws`` array.

    Pickles with :mod:`pickle` when the function that builds the problem does
    (a function defined at the top level of a module).
    """

    def __init__(
        self,
        *,
        target: "_Built | _Scales",
        names: list[str],
        settings: pd.DataFrame,
        space: Space,
        states: np.ndarray,
        mode: pd.Series,
        acceptance: float,
        proposal_scale: float,
    ):
        self._target = target
        self._names = names
        self._settings = settings
        self._space = space
        self._states = states
        self._states.flags.writeable = False
        self._mode = mode
        self._acceptance = acceptance
        self._proposal_scale = proposal_scale

    def __repr__(self) -> str:
        """Return the settings, the number of draws and the acceptance."""
        return (
            f"Hierarchical(settings={self._names}, draws={len(self._settings)}, "
            f"acceptance={self._acceptance:.2f})"
        )

    # -- the settings ----------------------------------------------------------

    @property
    def settings(self) -> pd.DataFrame:
        """The kept draws of the settings: one row a draw (``draw``), one column a setting."""
        return self._settings

    @property
    def mode(self) -> pd.Series:
        """The posterior mode of the settings, where the chain started and Laplace's approximation was taken."""
        return self._mode

    @property
    def acceptance(self) -> float:
        """The fraction of the kept steps whose proposal was accepted (near 0.3 when the burn-in tuned the proposal well)."""
        return self._acceptance

    @property
    def ess(self) -> pd.Series:
        """
        The effective sample size of each setting's draws: about as many independent draws as they are worth.

        By Geyer's initial monotone sequence of autocorrelations. A Monte
        Carlo standard error of a setting's mean is its standard deviation
        over the root of this.
        """
        return pd.Series(
            [
                _effective_size(self._settings[name].to_numpy(dtype=float))
                for name in self._names
            ],
            index=self._settings.columns,
            name="ess",
        )

    def at(self, settings: "Mapping[str, float] | pd.Series") -> Posterior:
        """
        Return the posterior at one setting of every setting: the ordinary :class:`~fips.Posterior`, solved.

        For what the mixture does not define: the gain, DOFS, the chi-squared
        tests, the evidence. ``result.at(result.settings.median())``.

        Parameters
        ----------
        settings : mapping or pandas.Series
            A value for each setting, by name.

        Raises
        ------
        ValueError
            If a setting is missing or unknown.
        """
        values = {str(k): float(v) for k, v in dict(settings).items()}
        missing = [name for name in self._names if name not in values]
        unknown = [name for name in values if name not in self._names]
        if missing or unknown:
            raise ValueError(
                f"at takes a value for each of the settings {self._names}; "
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
        r"""The posterior mean of the state, the mean of its draws: :math:`E[\hat{x}(s)]` over the settings' posterior."""
        return Vector(self._states.mean(axis=1), self._space, name="estimate")

    @property
    def variances(self) -> Vector:
        r"""
        The posterior variances of the state, those of its draws (about their mean).

        By the law of total variance they are :math:`E[\operatorname{diag}
        \hat{S}(s)] + \operatorname{var}[\hat{x}(s)]` over the settings'
        posterior: the mean of each setting's variances, and the spread of
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
            prior column: the prior of the state depends on the settings.

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
        self, n: int | None = None, *, W: Operator | None = None
    ) -> pd.DataFrame:
        """
        Return draws of the state from its posterior, mixed over the settings' draws: one column each.

        The draws are those made at the kept steps of the chain, one at each
        step's settings, so a column's draw belongs to the row of
        :attr:`settings` its label names. Ask for fewer than were kept and
        they are spread evenly along the chain, which takes them farther
        apart in it.

        Parameters
        ----------
        n : int, optional
            The number of draws, at most the number kept; all by default.
        W : Operator, optional
            An aggregation, as :meth:`cov` takes it: the draws are then those
            of :math:`W x`.

        Returns
        -------
        pandas.DataFrame
            One row per state element, labeled as the :attr:`estimate` is (or
            one per row of `W`), and one column per draw, labeled by its row
            of :attr:`settings` (``draw``).

        Raises
        ------
        TypeError
            If `W` is not an Operator.
        ValueError
            If `n` is not a positive integer of at most the kept draws, or the
            columns of `W` do not have exactly the state's labels.
        """
        kept = self._states.shape[1]
        if n is None:
            n = kept
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
        chosen = np.round(np.linspace(0, kept - 1, int(n))).astype(int)
        return pd.DataFrame(
            values[:, chosen],
            index=index,
            columns=pd.Index(self._settings.index[chosen], name="draw"),
        )
