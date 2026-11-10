.. _settings:

Fitting the covariance settings
===============================

Every inversion rests on settings someone chose: the prior error's standard
deviation and correlation length, the model-data mismatch's standard
deviation. The rule: **the settings that make the observations most probable
are their maximum-likelihood estimates, and** :attr:`post.log_evidence
<fips.Posterior.log_evidence>` **is that probability.** Write a function that
builds the problem from the settings, and maximize the evidence over them with
:mod:`scipy.optimize` (Michalak et al., 2005). fips adds no fitting function:
the settings are yours, and so is the function that turns them into a
problem.

.. contents:: On this page
   :local:
   :depth: 1


What the evidence is
--------------------

With the state integrated out, the observations of a linear problem are
Gaussian: :math:`z \sim N(H x_0 + c, G)`, with :math:`G = H S_0 H^T + S_z /
\gamma` the covariance of the innovation (:attr:`~fips.Posterior.innovation_cov`).
The evidence is the density of the observations under it:

.. math::

    \ln p(z) = -\tfrac{1}{2} \left( \chi^2 + \ln \lvert G \rvert + m \ln 2\pi \right),

with :math:`\chi^2 = d^T G^{-1} d` (:attr:`~fips.Posterior.chi2`) and :math:`d
= z - H x_0 - c`. The two terms pull opposite ways: a wider prior or a larger
mismatch lowers :math:`\chi^2` and raises :math:`\ln \lvert G \rvert`, so the
evidence is largest at covariances as large as the innovation needs and no
larger. fips reads :math:`\ln \lvert G \rvert` from what the solve computed (it
is :math:`\ln \lvert S_z / \gamma \rvert` plus twice the
:attr:`~fips.Posterior.information_content`), so the evidence costs one
:math:`\ln \lvert S_z \rvert` beyond the solve, and ``post.with_gamma(g)``
gives another gamma's for a few products. Gamma scales the mismatch, so it is
the same setting as the mismatch's standard deviation: fit one or the other,
not both.


A world at known settings
-------------------------

A hundred and fifty cells along a line, 1 km apart, with a prior of 2 and
errors correlated over a length the fit should find; two hundred observations,
each of one to four neighboring cells:

.. ipython:: python

   import numpy as np
   import pandas as pd
   from scipy.optimize import brentq, minimize
   import fips

   rng = np.random.default_rng(0)
   n_cells, n_obs = 150, 200
   cells = fips.Space({"flux": pd.Index(np.arange(n_cells) + 0.5, name="x")})  # km
   observed = fips.Space({"ch4": pd.RangeIndex(n_obs, name="obs_id")})

.. ipython:: python

   footprint = np.zeros((n_obs, n_cells))
   for i, start in enumerate(rng.integers(0, n_cells - 4, n_obs)):
       footprint[i, start : start + rng.integers(1, 5)] = rng.uniform(0.5, 1.5)

   H = fips.Operator(footprint, observed, cells)
   prior = fips.Vector(np.full(n_cells, 2.0), cells, name="prior")

The function from the settings to the problem is the one a repeated inversion
already has (:ref:`the user guide <repeated-inversions>`). Here the settings
are three numbers:

.. ipython:: python

   def build(settings, obs):
       """The problem at the settings: the prior std, the correlation length (km), the mismatch std."""
       std, length, noise = settings
       return fips.Problem(
           prior=prior,
           prior_cov=fips.Exponential("x", length).with_std(std),
           obs=obs,
           obs_cov=fips.Independent().with_std(noise),
           forward=H,
       )

The world is the problem at the true settings, and
:meth:`~fips.Problem.simulate` draws a truth from its prior and the
observations it would give:

.. ipython:: python

   true = pd.Series({"std": 1.0, "length": 5.0, "noise": 0.5})
   world = build(true, fips.Vector(np.zeros(n_obs), observed))
   truth, obs = world.simulate(rng)


Maximizing the evidence
-----------------------

Minimize the negative log evidence over the logarithms of the settings, which
keeps each positive and puts a factor of two the same distance away on either
side. L-BFGS-B estimates the gradient from differences, at a few solves each:

.. ipython:: python

   def cost(log_settings, obs=obs):
       return -fips.solve(build(np.exp(log_settings), obs)).log_evidence

   guess = pd.Series({"std": 0.3, "length": 1.0, "noise": 2.0})
   fit = minimize(cost, np.log(guess), method="L-BFGS-B")
   fitted = pd.Series(np.exp(fit.x), index=true.index)
   pd.DataFrame({"true": true, "guess": guess, "fitted": fitted}).round(3)

From a guess off by a factor of three to five in each setting, the fit finds
the two standard deviations to 2% and 5% of the truth, and a correlation length of
2.8 km against the true 5. That is not a failure of the fit: one map of 150
cells holds about thirty correlation lengths, and the evidence says little
more about the length than that. Hold the length at each of several values and
fit the two standard deviations (a profile): the evidence changes by less than
1.5 between 2 and 5 km, and a drop of 2 below the maximum is roughly the edge
of a 95% interval (half of 3.84, the 95% point of a chi-squared with one degree
of freedom):

.. ipython:: python

   def profile(length):
       best = minimize(lambda s: cost(np.log([np.exp(s[0]), length, np.exp(s[1])])), np.log([1.0, 0.5]), method="L-BFGS-B")
       return pd.Series({"log_evidence": -best.fun, "std": np.exp(best.x[0]), "noise": np.exp(best.x[1])})

   table = pd.DataFrame({L: profile(L) for L in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)}).T.rename_axis("length")
   table["below_max"] = -fit.fun - table["log_evidence"]
   table.round(3)

The data rule out lengths of 1 km and of 8 km and more, and leave 2 to 5 km
about equally likely. Over other worlds at the same settings the fits scatter
about the truth; their medians, over ten worlds, are within 2% of it:

.. ipython:: python

   fits = []
   for trial in range(10):
       _, obs_k = world.simulate(rng)
       best = minimize(cost, np.log(guess), args=(obs_k,), method="L-BFGS-B")
       fits.append(np.exp(best.x))

   fits = pd.DataFrame(fits, columns=true.index)
   fits.median().round(3)

The scatter of the fits is the uncertainty of the settings, which the
posterior at the fitted settings leaves out: it treats them as known. To carry
it into the flux, the settings need a prior of their own and draws over them:
:doc:`hierarchical` draws them.


Against a reduced chi-squared of 1
----------------------------------

The common alternative is to sweep a setting until the reduced chi-squared,
:math:`\chi^2 / m`, is 1 (:attr:`~fips.Posterior.reduced_chi2`). It is one
equation, so it picks one setting given the others: each guess of the length
and the mismatch has a prior std that passes it. Here are four, each with its
evidence:

.. ipython:: python

   def chi2_of_one(length, noise):
       def excess(log_std):
           return fips.solve(build([np.exp(log_std), length, noise], obs)).reduced_chi2 - 1.0
       std = np.exp(brentq(excess, np.log(0.01), np.log(100.0)))
       evidence = fips.solve(build([std, length, noise], obs)).log_evidence
       return pd.Series({"length": length, "noise": noise, "std": std, "log_evidence": evidence})

   pd.DataFrame([chi2_of_one(L, s) for L, s in ((1.0, 2.0), (2.0, 1.0), (5.0, 0.5), (5.0, 0.3))]).round(3)

Every row has a reduced chi-squared of 1, and their prior stds run from 0.3 to
3.1: the test passes whatever the guess, so it cannot say which guess is right.
Only the row at the true length and mismatch gets the right prior std (1.1),
and its evidence is near the maximum (-268.5); the others are 58 to 163 below
it, odds against them of :math:`e^{58}` and more. The evidence compares the
rows, where the reduced chi-squared cannot.

At the maximum of the evidence the reduced chi-squared is 1 by itself:

.. ipython:: python

   round(fips.solve(build(fitted, obs)).reduced_chi2, 4)

Scaling the prior std and the mismatch std together by :math:`k` scales
:math:`G` by :math:`k^2`, and the evidence is stationary in :math:`k` exactly
when :math:`\chi^2 = m`. So a reduced chi-squared of 1 is a consequence of the
maximum, one condition of three, not a way to find it.


The other methods
-----------------

``"gauss_newton"`` (:doc:`nonlinear`, :doc:`positive`) gives Laplace's
approximation of the evidence: the same formula with the cost at the mode and
the information content of the Gaussian there. For a lognormal model in four
cells with a log-std of 0.8, it was within about 0.1 of an importance-sampling
estimate, a small error next to the differences between settings above; it is
exact for a linear model, and it leaves out the skew of the posterior. Fit
lognormal settings with it as above. ``"bounded"`` (:doc:`bounds`) raises: the
evidence with bounds needs the probability of the bounds under the prior and
under the posterior, which fips does not compute. Fit the settings of a
bounded inversion on its analytic problem. ``"variational"``
(:doc:`variational`) gives the evidence of its low-rank posterior, exact at
full rank. Below it the evidence leaves out the information of the pairs not
kept and is too high, the more so for a larger prior, so a maximum would pick
too large a prior: it raises while the last pair kept still has an eigenvalue
above 1, and is exact only when ``rank`` reaches the problem's.

References
~~~~~~~~~~

Michalak, A. M., Hirsch, A., Bruhwiler, L., Gurney, K. R., Peters, W., and
Tans, P. P. (2005). Maximum likelihood estimation of covariance parameters for
Bayesian atmospheric trace gas surface flux inversions. *Journal of
Geophysical Research*, 110, D24107. https://doi.org/10.1029/2005JD005970
