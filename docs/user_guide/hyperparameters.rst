.. _hyperparameters:
.. _settings:

Fitting the covariance hyperparameters
======================================

Every inversion rests on numbers someone chose: the prior covariance's
standard deviation and correlation length, the observation covariance's
standard deviation. These are the *hyperparameters*. **The hyperparameters that make the
observations most probable are their maximum-likelihood estimates, and**
:attr:`post.log_evidence <fips.Posterior.log_evidence>` **is the log of that
probability**, the *evidence*. Write a function that builds the problem from
the hyperparameters, and maximize the evidence over them with :mod:`scipy.optimize`:

.. code-block:: python

   def cost(log_hyperparameters):
       return -fips.solve(build(np.exp(log_hyperparameters))).log_evidence

   fit = scipy.optimize.minimize(cost, np.log(guess), method="L-BFGS-B")

``np.exp(fit.x)`` is the fitted hyperparameters, and ``fips.solve(build(...))`` at
them is the posterior. fips has no fitting function of its own: the hyperparameters
are yours, and so is the function that turns them into a problem (Michalak et
al., 2005).

.. contents:: On this page
   :local:
   :depth: 1


When to use it
--------------

- **Your hyperparameters are a judgment, and you want the data's view of them.**
  Fit the ones you are least sure of; hold the others.
- **The reduced chi-squared is far from 1.** It says the covariances are too
  small or too large together, not which one is wrong; the evidence decides
  between them (`Against a reduced chi-squared of 1`_).
- **Only gamma, one factor on the observation covariance.**
  ``post.with_gamma(g).log_evidence`` gives the evidence at another gamma for
  a few products, with no new solve, so a sweep or
  :func:`scipy.optimize.minimize_scalar` over gamma is cheap. Gamma scales the
  observation covariance, so it is the same hyperparameter as the observation
  standard deviation: fit one or the other, not both.
- **The fitted hyperparameters are uncertain, and that matters to the answer.** A
  posterior at fitted hyperparameters treats them as known. :doc:`hierarchical`
  draws the hyperparameters from their posterior instead, and carries their
  uncertainty into the state's.


An example
----------

A quantity along a line: 150 state elements one unit apart, with a prior of
2, and 200 observations, each a weighted sum of one to four neighboring
elements.

.. ipython:: python

   import numpy as np
   import pandas as pd
   from scipy.optimize import brentq, minimize
   import fips

   rng = np.random.default_rng(0)
   n, m = 150, 200
   line = fips.Space({"field": pd.Index(np.arange(n) + 0.5, name="x")})
   observed = fips.Space({"obs": pd.RangeIndex(m, name="obs_id")})

.. ipython:: python

   weights = np.zeros((m, n))
   for i, start in enumerate(rng.integers(0, n - 4, m)):
       weights[i, start : start + rng.integers(1, 5)] = rng.uniform(0.5, 1.5)

   H = fips.Operator(weights, observed, line)
   prior = fips.Vector(np.full(n, 2.0), line, name="prior")

The function from the hyperparameters to the problem is the one a
:ref:`repeated inversion <repeated-inversions>` already has. Here the
hyperparameters are three numbers: the prior standard deviation, the correlation
length, and the observation standard deviation.

.. ipython:: python

   def build(hyperparameters, obs):
       """Return the problem at the hyperparameters (prior std, correlation length, observation std)."""
       std, length, noise = hyperparameters
       return fips.Problem(
           prior=prior,
           prior_cov=fips.Exponential("x", length).with_std(std),
           obs=obs,
           obs_cov=fips.Independent().with_std(noise),
           forward=H,
       )

To show that the fit finds the right hyperparameters, the observations here come from
a world whose hyperparameters are known: the problem at the true hyperparameters, whose
:meth:`~fips.Problem.simulate` draws a truth from its prior and the
observations it gives (:doc:`synthetic`).

.. ipython:: python

   true = pd.Series({"std": 1.0, "length": 5.0, "noise": 0.5})
   truth, obs = build(true, fips.Vector(np.zeros(m), observed)).simulate(rng)

Minimize the negative log evidence over the logarithms of the hyperparameters. The
logarithm keeps each hyperparameter positive and puts a factor of two the same
distance away on either side. L-BFGS-B estimates the gradient from
differences, a few solves per step:

.. ipython:: python

   def cost(log_hyperparameters, obs=obs):
       return -fips.solve(build(np.exp(log_hyperparameters), obs)).log_evidence

   guess = pd.Series({"std": 0.3, "length": 1.0, "noise": 2.0})
   fit = minimize(cost, np.log(guess), method="L-BFGS-B")
   fitted = pd.Series(np.exp(fit.x), index=true.index)
   pd.DataFrame({"true": true, "guess": guess, "fitted": fitted}).round(3)

From a guess off by a factor of three to five in each hyperparameter, the fit finds
both standard deviations close to the truth. The correlation length it finds
is shorter than the true one, and the next section shows why that is not a
failure of the fit.


How well the data decide a hyperparameter
-----------------------------------------

**The evidence is a likelihood, so its curvature says how well the data
decide a hyperparameter.** Hold one hyperparameter at each of several values, fit the
others, and read how far the evidence falls below its maximum (a *profile*).
A fall of about 2 is the edge of a 95% interval: half of 3.84, the 95% point
of a chi-squared with one degree of freedom.

.. ipython:: python

   def profile(length):
       """Fit the two standard deviations with the length held, and return the evidence there."""
       best = minimize(lambda s: cost(np.log([np.exp(s[0]), length, np.exp(s[1])])), np.log([1.0, 0.5]), method="L-BFGS-B")
       return pd.Series({"log_evidence": -best.fun, "std": np.exp(best.x[0]), "noise": np.exp(best.x[1])})

   table = pd.DataFrame({L: profile(L) for L in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)}).T.rename_axis("length")
   table["below_max"] = -fit.fun - table["log_evidence"]
   table.round(3)

The data rule out the shortest and the longest lengths and leave the middle
of the range about equally likely: a line of 150 elements holds only a few
dozen correlation lengths, and its observations say little more about the
length than that. A flat profile is the data's answer, not the optimizer's failure;
when it matters to the answer, :doc:`hierarchical` carries it into the
state's uncertainty.


Which methods give the evidence
-------------------------------

The evidence is exact for a Gaussian posterior, Laplace's approximation for a
posterior summarized at its mode, and refused where neither holds:

- ``"analytic"``: exact.
- ``"gauss_newton"`` (:doc:`nonlinear`, :doc:`positive`): Laplace's
  approximation, the same formula with the misfit at the mode and the
  information content of the Gaussian there. It is exact for a linear
  forward model. For a model that bends, it leaves out how the posterior
  departs from that Gaussian (its skew); for a lognormal state of a few
  elements that error is about 0.1, small next to the differences between
  hyperparameters, so fit the hyperparameters of a forward model with it as above.
- ``"variational"`` (:doc:`variational`): the same formula with the
  information content of its low-rank posterior (Laplace's at the mode, with
  a forward model). It is exact at full rank. Below it, the evidence leaves
  out the information of the directions not kept and is too high, the more so
  for a larger prior, so a maximum would pick too large a prior. It raises
  while the last direction kept still has an eigenvalue above 1: raise
  ``rank`` until the last :attr:`~fips.Posterior.eigenvalues` are well below 1.
- ``"bounded"`` (:doc:`bounds`): raises with a finite bound. The evidence
  with bounds needs the probability of the bounds under the prior and under
  the posterior, which fips does not compute. Fit the hyperparameters on the problem
  without bounds, with ``"analytic"``. With no finite bound it is the
  analytic posterior, and gives the exact evidence.
- ``"ensemble"`` (:doc:`ensemble`): raises. For a linear problem, the
  analytic solve of the same problem gives the exact evidence.

A method that cannot give it raises :class:`~fips.UnavailableError`, saying
why. The :ref:`members table <posterior-members>` lists it with the rest.


Against a reduced chi-squared of 1
----------------------------------

The common alternative is to adjust a hyperparameter until the reduced chi-squared,
:math:`\chi^2 / m` (:attr:`~fips.Posterior.reduced_chi2`), is 1. **It is one
equation, so it picks one hyperparameter given the others**: each guess of the
length and the observation standard deviation has a prior standard deviation
that passes it. Here are four guesses, each with the prior standard deviation
that gives a reduced chi-squared of 1, and the evidence there:

.. ipython:: python

   def chi2_of_one(length, noise):
       """Return the prior std at which the reduced chi-squared is 1, and the evidence there."""
       def excess(log_std):
           return fips.solve(build([np.exp(log_std), length, noise], obs)).reduced_chi2 - 1.0
       std = np.exp(brentq(excess, np.log(0.01), np.log(100.0)))
       evidence = fips.solve(build([std, length, noise], obs)).log_evidence
       return pd.Series({"length": length, "noise": noise, "std": std, "log_evidence": evidence})

   pd.DataFrame([chi2_of_one(L, s) for L, s in ((1.0, 2.0), (2.0, 1.0), (5.0, 0.5), (5.0, 0.3))]).round(3)

Every row passes the test, with prior standard deviations a factor of ten
apart, so the test cannot say which guess is right. The evidence can: only
the row at the true length and observation standard deviation comes near its
maximum, and a difference in log evidence is the log of the odds between two
rows. At the maximum of the evidence, the reduced chi-squared is 1 by itself:

.. ipython:: python

   round(fips.solve(build(fitted, obs)).reduced_chi2, 4)

A reduced chi-squared of 1 is one condition the maximum satisfies, not a way
to find it (`How it works`_).


Warnings and checks
-------------------

- **The evidence is a density of the observations**, so it depends on their
  units and their number. Compare it only between problems of the same
  observations.
- **A hyperparameter the data do not decide** has a flat evidence, and the optimizer
  stops anywhere along the flat part. Check a fitted hyperparameter with a profile
  before trusting it.
- **The observation covariance must be positive definite**: the evidence
  needs :math:`\ln \lvert S_z \rvert`, and raises
  :class:`numpy.linalg.LinAlgError` otherwise. A singular prior covariance is
  fine.
- **Gamma and the observation standard deviation are one hyperparameter.** Fitting
  both leaves the optimizer a direction in which the evidence does not
  change.


Cost
----

Each evaluation is one solve. L-BFGS-B with differences takes one more
solve per hyperparameter at each step, so a fit of :math:`p` hyperparameters costs a few
dozen times :math:`p + 1` solves. The evidence itself costs one
:math:`\ln \lvert S_z \rvert` beyond the solve, and
``post.with_gamma(g).log_evidence`` a few products, after one
eigendecomposition that every gamma shares.


.. _hyperparameters-evidence:
.. _settings-evidence:

How it works
------------

With the state integrated out, the observations of a linear problem are
Gaussian, :math:`z \sim N(H x_0 + c, G)`, with :math:`G = H S_0 H^T + S_z /
\gamma` the covariance of the innovation
(:attr:`~fips.Posterior.innovation_cov`). The evidence is the density of the
observations under it:

.. math::

    \ln p(z) = -\tfrac{1}{2} \left( \chi^2 + \ln \lvert G \rvert + m \ln 2\pi \right),

with :math:`\chi^2 = d^T G^{-1} d` (:attr:`~fips.Posterior.chi2`) and
:math:`d = z - H x_0 - c`. The two terms pull opposite ways: a wider prior or
a larger observation error lowers :math:`\chi^2` and raises
:math:`\ln \lvert G \rvert`, so the evidence is largest at covariances as
large as the innovation needs and no larger.

fips reads :math:`\ln \lvert G \rvert` from what the solve computed:
:math:`\lvert G \rvert = \lvert S_z / \gamma \rvert \, \lvert I + \gamma
S_z^{-1} H S_0 H^T \rvert`, and the log of the second factor is twice the
:attr:`~fips.Posterior.information_content`. So

.. math::

    \ln p(z) = -\tfrac{1}{2} \left( \chi^2 + 2 \mathcal{H} + \ln \lvert S_z \rvert
        - m \ln \gamma + m \ln 2\pi \right),

with :math:`\mathcal{H}` the information content: only :math:`\ln \lvert S_z
\rvert` is computed beyond the solve, once per problem. Laplace's
approximation, for a forward model, is the same formula with the misfit at
the mode and the information content of the Gaussian there.

**Why the reduced chi-squared is 1 at the maximum.** Scaling the prior and
the observation standard deviations together by :math:`k` scales :math:`G`
by :math:`k^2`, so the evidence along that direction is
:math:`-\tfrac{1}{2} (\chi^2 / k^2 + m \ln k^2) + \text{const}`. It is
stationary in :math:`k` exactly when :math:`\chi^2 = m`. The maximum over all
the hyperparameters is stationary in that direction too, so its reduced chi-squared
is 1; the other directions (how the variance splits between the prior and the
observations, the correlation length) the reduced chi-squared does not see.

References
----------

Michalak, A. M., Hirsch, A., Bruhwiler, L., Gurney, K. R., Peters, W., and
Tans, P. P. (2005). Maximum likelihood estimation of covariance parameters for
Bayesian atmospheric trace gas surface flux inversions. *Journal of
Geophysical Research*, 110, D24107. https://doi.org/10.1029/2005JD005970
