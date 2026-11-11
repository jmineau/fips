r"""
The distributions :func:`fips.hierarchical` accepts as the prior of a covariance setting.

Three: :class:`LogNormal`, :class:`HalfNormal` and :class:`Uniform`.
Each has a log density, draws, a median (where the search for the mode
starts) and a map to unbounded coordinates (the log of a positive setting,
the logit of a bounded one), where the sampler of :func:`fips.hierarchical`
works. This is not a general library of distributions: they are the priors
of the settings (a standard deviation, a correlation length, the observation
error's scale), not of the state, whose prior is a :class:`~fips.Vector` and a
:class:`~fips.Covariance`.

>>> import numpy as np
>>> from fips.distributions import LogNormal
>>> round(LogNormal(np.log(2.0), 0.5).median, 3)
2.0
"""

import math
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import log_expit, logit, ndtri

__all__ = ["HalfNormal", "LogNormal", "Uniform"]

_LOG_2PI = math.log(2.0 * math.pi)


class _Distribution(ABC):
    """
    The prior of one setting: its log density, and a map to unbounded coordinates, where the sampler works.

    :class:`LogNormal`, :class:`HalfNormal` and :class:`Uniform` are the
    ones fips has. A new one is a subclass with :meth:`logpdf`,
    :meth:`sample`, and the three private members of the map
    (``_to_unbounded``, ``_from_unbounded``, ``_log_density_unbounded``).
    """

    @abstractmethod
    def logpdf(self, x: float) -> float:
        """Return the log density at `x` (``-inf`` outside the support)."""

    @abstractmethod
    def sample(
        self, n: int, rng: "int | np.random.Generator | None" = None
    ) -> np.ndarray:
        """Return `n` draws, from a seed or a generator (as :meth:`fips.Posterior.sample` takes it)."""

    @abstractmethod
    def _to_unbounded(self, x: float) -> float:
        """Return the unbounded coordinate of `x`."""

    @abstractmethod
    def _from_unbounded(self, u: float) -> float:
        """Return the setting at the unbounded coordinate `u`."""

    @abstractmethod
    def _log_density_unbounded(self, u: float) -> float:
        r"""Return the log density of the unbounded coordinate, :math:`\ln p(x(u)) + \ln \lvert dx/du \rvert`."""

    @property
    @abstractmethod
    def median(self) -> float:
        """The prior median, where the search for the mode starts."""


def _positive(value: Any, what: str) -> float:
    """Return `value` as a float, or raise unless it is positive and finite."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{what} is a number, got {type(value).__name__}.")
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{what} must be positive and finite, got {value}.")
    return value


def _finite(value: Any, what: str) -> float:
    """Return `value` as a float, or raise unless it is finite."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{what} is a number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value}.")
    return value


class _Positive(_Distribution):
    """A prior of a positive setting, sampled in its log."""

    def _to_unbounded(self, x: float) -> float:
        r"""Return :math:`\ln x`."""
        return math.log(x)

    def _from_unbounded(self, u: float) -> float:
        """Return :math:`e^u` (``inf`` past the largest float)."""
        return math.exp(u) if u < 709.0 else math.inf

    def _log_density_unbounded(self, u: float) -> float:
        r"""Return :math:`\ln p(e^u) + u`."""
        x = self._from_unbounded(u)
        return self.logpdf(x) + u if 0.0 < x < math.inf else -math.inf


@dataclass(frozen=True)
class LogNormal(_Positive):
    r"""
    A lognormal prior: the log of the setting is normal, :math:`\ln s \sim N(\mu, \sigma^2)`.

    The prior of a standard deviation or a length known to a factor:
    ``LogNormal(np.log(2.0), 0.5)`` has its median at 2 and a 95% interval
    from 0.75 to 5.3 (a factor :math:`e^{1.96 \sigma}` either side).

    Parameters
    ----------
    mu : float
        The mean of the log, the log of the median.
    sigma : float
        The standard deviation of the log, positive.

    Raises
    ------
    ValueError
        If `mu` is not finite or `sigma` is not positive and finite.

    Examples
    --------
    >>> import numpy as np
    >>> import fips
    >>> prior = fips.distributions.LogNormal(np.log(2.0), 0.5)
    >>> round(prior.median, 3)
    2.0
    >>> round(prior.logpdf(2.0), 4)
    -0.9189
    """

    mu: float = 0.0
    """The mean of the log of the setting."""
    sigma: float = 1.0
    """The standard deviation of the log of the setting."""

    def __post_init__(self):
        """Check the parameters."""
        object.__setattr__(self, "mu", _finite(self.mu, "LogNormal's mu"))
        object.__setattr__(self, "sigma", _positive(self.sigma, "LogNormal's sigma"))

    @property
    def median(self) -> float:
        r"""The median, :math:`e^\mu`."""
        return math.exp(self.mu)

    def logpdf(self, x: float) -> float:
        r"""Return :math:`-\ln x - \ln \sigma - \tfrac{1}{2} \ln 2\pi - (\ln x - \mu)^2 / 2\sigma^2` (``-inf`` at and below 0)."""
        if not x > 0:
            return -math.inf
        u = math.log(x)
        return self._log_density_unbounded(u) - u

    def _log_density_unbounded(self, u: float) -> float:
        r"""Return the normal log density of the log, :math:`N(\mu, \sigma^2)`."""
        return (
            -math.log(self.sigma)
            - 0.5 * _LOG_2PI
            - 0.5 * ((u - self.mu) / self.sigma) ** 2
        )

    def sample(
        self, n: int, rng: "int | np.random.Generator | None" = None
    ) -> np.ndarray:
        r"""Return `n` draws, :math:`e^{\mu + \sigma \xi}`."""
        normals = np.random.default_rng(rng).standard_normal(n)
        return np.exp(self.mu + self.sigma * normals)


@dataclass(frozen=True)
class HalfNormal(_Positive):
    r"""
    A half-normal prior: the absolute value of a normal of mean 0 and standard deviation `scale`.

    A weak prior on a positive setting that is likely below a few times
    `scale` and may be near zero: its median is :math:`0.674` `scale`, and
    95% of it is below :math:`1.96` `scale`.

    Parameters
    ----------
    scale : float
        The standard deviation of the normal, positive.

    Raises
    ------
    ValueError
        If `scale` is not positive and finite.

    Examples
    --------
    >>> import fips
    >>> prior = fips.distributions.HalfNormal(10.0)
    >>> round(prior.median, 3)
    6.745
    """

    scale: float = 1.0
    """The standard deviation of the normal the setting is the absolute value of."""

    def __post_init__(self):
        """Check the parameter."""
        object.__setattr__(self, "scale", _positive(self.scale, "HalfNormal's scale"))

    @property
    def median(self) -> float:
        r"""The median, :math:`\Phi^{-1}(0.75)` `scale`."""
        return float(ndtri(0.75)) * self.scale

    def logpdf(self, x: float) -> float:
        r"""Return :math:`\ln 2 - \ln \sigma - \tfrac{1}{2} \ln 2\pi - x^2 / 2\sigma^2` for :math:`x \ge 0` (``-inf`` below)."""
        if not x >= 0:
            return -math.inf
        return (
            math.log(2.0)
            - math.log(self.scale)
            - 0.5 * _LOG_2PI
            - 0.5 * (x / self.scale) ** 2
        )

    def sample(
        self, n: int, rng: "int | np.random.Generator | None" = None
    ) -> np.ndarray:
        r"""Return `n` draws, :math:`\lvert \sigma \xi \rvert`."""
        return np.abs(self.scale * np.random.default_rng(rng).standard_normal(n))


@dataclass(frozen=True)
class Uniform(_Distribution):
    r"""
    A uniform prior on an interval, sampled in the logit of its position, :math:`\ln((s - a) / (b - s))`.

    Parameters
    ----------
    low : float
        The lower end of the interval.
    high : float
        The upper end, above `low`.

    Raises
    ------
    ValueError
        If the ends are not finite, or `high` is not above `low`.

    Examples
    --------
    >>> import fips
    >>> prior = fips.distributions.Uniform(1.0, 9.0)
    >>> prior.median, round(prior.logpdf(2.0), 4), prior.logpdf(10.0)
    (5.0, -2.0794, -inf)
    """

    low: float = 0.0
    """The lower end of the interval."""
    high: float = 1.0
    """The upper end of the interval."""

    def __post_init__(self):
        """Check the ends."""
        low = _finite(self.low, "Uniform's low")
        high = _finite(self.high, "Uniform's high")
        if not high > low:
            raise ValueError(f"Uniform's high ({high}) must be above its low ({low}).")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def median(self) -> float:
        """The middle of the interval."""
        return 0.5 * (self.low + self.high)

    def logpdf(self, x: float) -> float:
        r"""Return :math:`-\ln(b - a)` inside the interval, ``-inf`` outside."""
        if not self.low <= x <= self.high:
            return -math.inf
        return -math.log(self.high - self.low)

    def _to_unbounded(self, x: float) -> float:
        """Return the logit of the position of `x` in the interval."""
        return float(logit((x - self.low) / (self.high - self.low)))

    def _from_unbounded(self, u: float) -> float:
        """Return the setting whose position's logit is `u`."""
        return self.low + (self.high - self.low) * math.exp(float(log_expit(u)))

    def _log_density_unbounded(self, u: float) -> float:
        r"""Return :math:`\ln \sigma(u) + \ln \sigma(-u)`, the logistic density of the logit."""
        return float(log_expit(u) + log_expit(-u))

    def sample(
        self, n: int, rng: "int | np.random.Generator | None" = None
    ) -> np.ndarray:
        """Return `n` draws."""
        return np.random.default_rng(rng).uniform(self.low, self.high, n)
