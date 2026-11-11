.. _hierarchical:

The uncertainty of the hyperparameters
======================================

A posterior at fitted hyperparameters (:doc:`hyperparameters`) treats them as known. **Give
each hyperparameter a prior, and** :func:`fips.hierarchical` **draws the hyperparameters
from their posterior and the state at each of those draws, so the state's
intervals hold the uncertainty of the hyperparameters too.** The call takes the
function that builds the problem from the hyperparameters, and a prior for each
hyperparameter by name:

.. code-block:: python

   result = fips.hierarchical(build, {"std": LogNormal(np.log(0.5), 1.0), ...})

What comes back is a :class:`~fips.Hierarchical`: the kept draws of the
hyperparameters, and one draw of the state at each. Its estimate is the mean of the
state's draws, and its spread and intervals are theirs. It is not a
:class:`~fips.Posterior`: the posterior of the state is a mixture of
Gaussians, one for each draw of the hyperparameters, and what is defined only for
one Gaussian comes from the posterior at one hyperparameter,
:meth:`result.at(hyperparameters) <fips.Hierarchical.at>`.

.. contents:: On this page
   :local:
   :depth: 1


When to use it
--------------

- **Few observations decide the hyperparameters, and the answer depends on them.**
  The hyperparameters' uncertainty then widens the state's intervals. With many
  observations it widens them little, and a fit (:doc:`hyperparameters`) is enough;
  ``result.at(result.mode)`` shows the difference on your own problem.
- **The only hyperparameters are a factor on the prior standard deviation and one
  on the observation standard deviation.** Pass the problem itself and take
  `The fast path`_, which costs about one eigendecomposition for the whole
  run.
- **For one best value of the hyperparameters**, maximize the evidence instead
  (:doc:`hyperparameters`): it is cheaper, and the posterior at the fit has every
  diagnostic.
- **Not for a problem the analytic method cannot solve** (a forward model,
  bounds, a problem too large to factor): each step needs the exact evidence
  and exact draws of the analytic posterior.


An example
----------

The example of :doc:`hyperparameters`: 150 state elements along a line, one unit
apart, 200 observations of one to four neighboring elements, and
observations drawn from a world at a prior standard deviation of 1, a
correlation length of 5 and an observation standard deviation of 0.5.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips
   from fips.distributions import LogNormal

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

The function takes the hyperparameters as a dict, by the names their priors have:

.. ipython:: python

   def build(hyperparameters, obs, H=H):
       """Return the problem at the hyperparameters (prior std, correlation length, observation std)."""
       return fips.Problem(
           prior=prior,
           prior_cov=fips.Exponential("x", hyperparameters["length"]).with_std(hyperparameters["std"]),
           obs=obs,
           obs_cov=fips.Independent().with_std(hyperparameters["noise"]),
           forward=H,
       )

   true = {"std": 1.0, "length": 5.0, "noise": 0.5}
   truth, obs = build(true, fips.Vector(np.zeros(m), observed)).simulate(rng)

Each hyperparameter's prior is a :class:`~fips.distributions.LogNormal` that knows
it to a factor of :math:`e` (a standard deviation of 1 in its log), centered
a factor of two or more from the truth:

.. ipython:: python

   priors = {
       "std": LogNormal(np.log(0.5), 1.0),
       "length": LogNormal(np.log(2.0), 1.0),
       "noise": LogNormal(np.log(1.0), 1.0),
   }
   result = fips.hierarchical(lambda s: build(s, obs), priors, draws=1000, burn_in=300, seed=0)
   result.hyperparameters.quantile([0.025, 0.5, 0.975]).T.assign(true=pd.Series(true), mode=result.mode).round(3)

The two standard deviations are known to within about a fifth, and each
95% interval holds its truth. The correlation length is known only to a factor of about three,
as the profile of :doc:`hyperparameters` found.


Reading the result
------------------

**The hyperparameters.** :attr:`~fips.Hierarchical.hyperparameters` is the kept draws, one
row per draw and one column per hyperparameter; :attr:`~fips.Hierarchical.mode` is
the posterior mode in the coordinates the sampler works in (the log of a
positive hyperparameter, the logit of a bounded one), mapped back: close to the fit
of :doc:`hyperparameters`, but not the peak of the hyperparameters' density itself. Two
numbers say whether the chain can be trusted:
:attr:`~fips.Hierarchical.acceptance`, the fraction of proposals accepted
(near 0.3 when the proposal is tuned), and :attr:`~fips.Hierarchical.ess`,
each hyperparameter's effective sample size: about as many independent draws as the
correlated ones are worth.

.. ipython:: python

   round(result.acceptance, 2), result.ess.round().to_dict()

A hyperparameter's mean is known to its standard deviation over the root of its
effective sample size. For a smaller error, ask for more ``draws``.

**The state.** :attr:`~fips.Hierarchical.estimate`,
:attr:`~fips.Hierarchical.variances`, :meth:`~fips.Hierarchical.cov`,
:meth:`~fips.Hierarchical.uncertainty` and :meth:`~fips.Hierarchical.sample`
read the state, or its aggregates with ``W``, from the draws, as a
:class:`~fips.Posterior`'s members of the same names do. The intervals are the
draws' quantiles, and ``uncertainty`` has no prior columns: the prior of the
state depends on the hyperparameters.

**One hyperparameter.** :meth:`~fips.Hierarchical.at` gives the ordinary posterior
at one value of every hyperparameter: the gain, DOFS, the chi-squared tests, the
evidence. At the mode, its intervals are those of a fit, which leave the
hyperparameters' uncertainty out. Here, the sum over each third of the line, with
and without the hyperparameters' uncertainty:

.. ipython:: python

   x = line.select("field")
   parts = pd.Series(np.select([x < 50, x < 100], ["left", "middle"], "right"), index=x, name="part")
   W = fips.aggregate.sum(line, by=parts)

.. ipython:: python

   def compare(result, truth, level=0.9):
       """Each aggregate's truth, and its interval with the hyperparameters' uncertainty and at their mode."""
       columns = ["posterior_std", "lower", "upper"]
       with_hyperparameters = result.uncertainty(W, level=level)[columns]
       at_mode = result.at(result.mode).uncertainty(W, level=level)[columns]
       true_value = pd.DataFrame({"": (W @ truth).values}, index=with_hyperparameters.index)
       table = pd.concat({"with hyperparameters": with_hyperparameters, "at mode": at_mode, "truth": true_value}, axis=1)
       return table.droplevel("block")

   compare(result, truth).round(2)

The two agree closely: two hundred observations decide these aggregates, and
the correlation length, the hyperparameter known least, moves how an aggregate
splits among its elements more than the aggregate itself.


The priors of the hyperparameters
---------------------------------

A hyperparameter's prior is one of the three distributions of
:mod:`fips.distributions`:

- :class:`~fips.distributions.LogNormal` ``(mu, sigma)``: a positive hyperparameter
  known to a factor, with median :math:`e^{\mu}`. The usual choice for a
  standard deviation or a correlation length.
- :class:`~fips.distributions.HalfNormal` ``(sigma)``: a positive hyperparameter that
  may be near zero.
- :class:`~fips.distributions.Uniform` ``(low, high)``: a hyperparameter known only
  to lie in an interval.

The search for the hyperparameters' mode starts at the priors' medians. **A prior on
a hyperparameter is a claim like any other, and the intervals are only as good as
it.** A prior centered far from the truth pulls the hyperparameters toward it, and
with few observations the data do not undo all of the pull.


The fast path
-------------

When the only hyperparameters are a factor on the prior standard deviation and one
on the observation standard deviation, pass the problem built at factors of
1 instead of a function, and name the hyperparameters:

- ``"prior_scale"``: the prior standard deviation is multiplied by it;
- ``"obs_scale"``: the observation standard deviation is multiplied by it;
  or ``"gamma"`` in its place: the observation covariance is divided by it.

One eigendecomposition then gives the evidence at every step without a
solve, and the state's draws at every kept step in one product, whatever the
size of the problem.

Fast or not, **the hyperparameters' uncertainty matters most when few observations
decide them.** Fifteen observations, all in the left two thirds of the line,
with the correlation length held at 5:

.. ipython:: python

   few = fips.Space({"obs": pd.RangeIndex(15, name="obs_id")})
   sparse = np.zeros((15, n))
   for i, start in enumerate(rng.integers(0, 96, 15)):
       sparse[i, start : start + rng.integers(1, 5)] = rng.uniform(0.5, 1.5)

   H_few = fips.Operator(sparse, few, line)
   truth_few, obs_few = build(true, fips.Vector(np.zeros(15), few), H=H_few).simulate(rng)

.. ipython:: python

   unit = build({"std": 1.0, "length": 5.0, "noise": 1.0}, obs_few, H=H_few)
   scales = {"prior_scale": LogNormal(np.log(0.5), 1.0), "obs_scale": LogNormal(0.0, 1.0)}
   fast = fips.hierarchical(unit, scales, seed=0)
   fast.hyperparameters.quantile([0.025, 0.5, 0.975]).round(3)
   compare(fast, truth_few).round(2)

Here each standard deviation is known only to a factor of several, and the
intervals with the hyperparameters' uncertainty are wider than those at the mode.
Over many such worlds whose hyperparameters are drawn from their priors, the
intervals with the hyperparameters' uncertainty hold the truth about as often as
they claim, and those at the mode less often.


Options
-------

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Option
     - Default
     - What it does
   * - ``build``
     - (required)
     - The function from a dict of hyperparameters to the :class:`~fips.Problem`, or
       to its analytic posterior (``fips.solve(problem, gamma=s["gamma"])``
       makes gamma a hyperparameter). Or the problem itself, for the fast path.
   * - ``hyperparameters``
     - (required)
     - The prior of each hyperparameter, by the name ``build`` reads it by. Their
       order is the order of the columns of ``result.hyperparameters``.
   * - ``draws``
     - 2000
     - The kept steps of the chain, each with one draw of the state.
   * - ``burn_in``
     - 500
     - The steps before them, which tune the proposal and are not kept.
   * - ``seed``
     - 0
     - The seed of the chain and of the state's draws (an integer), so a run
       is reproducible. The two have their own streams: the chain is the same
       whatever the state's draws are.


Warnings and checks
-------------------

- **The log posterior does not curve down in some direction at the mode**
  (a warning): the data and the prior hardly decide a hyperparameter, or a
  combination of them. The proposal takes a variance of 1 there and the
  burn-in scales it, but the chain may be slow: look at the effective sample
  sizes, and give that hyperparameter a narrower prior or hold it fixed.
- **A low acceptance or a small effective sample size**: run more ``draws``
  or a longer ``burn_in``. An ``INFO`` record gives both at the end of each
  run, and another says when the search for the mode stopped early.
- **A problem that does not solve at some hyperparameters** (a covariance that is
  not positive definite) has no density there, and the chain does not step
  to it. At the priors' medians it raises a :class:`ValueError`, because the
  search for the mode starts there.
- **A function that returns a posterior of another method** raises a
  :class:`TypeError`: return the problem, or its analytic posterior. A prior
  that is not one of :mod:`fips.distributions` raises a :class:`TypeError`.
- **The fast path takes only its three names**, and only one of
  ``"obs_scale"`` and ``"gamma"``; any other raises a :class:`ValueError`.
  For other hyperparameters, pass a function.


Cost
----

A step of the chain is one solve of the problem the function builds, so a
run costs ``burn_in + draws`` solves, plus a few dozen for the search for the
mode and its Hessian (more with more hyperparameters). A step of the fast path
costs a sum over the smaller of the number of observations and the number of
state elements, after one eigendecomposition. The draws of the state are kept
as a ``state x draws`` array: 2000 draws of 30,000 elements take 480 MB, so
for a large state ask for fewer draws, or keep the aggregates you need.


How it works
------------

With a prior :math:`p(s)` on the hyperparameters, their posterior is the prior times
the evidence of the problem built at them (:ref:`the evidence <hyperparameters-evidence>`,
exact for a linear problem):

.. math::

    p(s \mid z) \propto p(s) \, p(z \mid s).

The posterior of the state is the posterior at each hyperparameter, averaged over
the hyperparameters' posterior,

.. math::

    p(x \mid z) = \int p(x \mid s, z) \, p(s \mid z) \, ds,

so a draw of the hyperparameters from :math:`p(s \mid z)`, then of the state from
:math:`p(x \mid s, z)` (exactly, as :meth:`~fips.Posterior.sample` draws it),
is a draw of :math:`p(x \mid z)`. Its variance is

.. math::

    \operatorname{var}(x \mid z) = E\big[\hat{S}(s)\big] + \operatorname{var}\big[\hat{x}(s)\big]

over the hyperparameters' posterior: the average of each hyperparameter's posterior
variance, and the spread of the estimates between hyperparameters. A solve at
fitted hyperparameters gives the first term at one hyperparameter and leaves out the
second.

The hyperparameters are drawn by a random-walk Metropolis in unbounded coordinates:
the log of a positive hyperparameter, the logit of a hyperparameter on an interval. The
chain starts at the hyperparameters' posterior mode, found by L-BFGS-B from the
priors' medians. It proposes steps from Laplace's approximation there (the
inverse of a numerical Hessian of the log posterior), and during the burn-in
scales them toward an acceptance of 0.3 (Gelman, Roberts and Gilks, 1996);
the scale is fixed after, so the kept steps are a Markov chain of the exact
posterior. At each kept step it draws the state once, at that step's
hyperparameters. The effective sample size is Geyer's (1992) initial monotone
sequence.

On the fast path, with :math:`s` the prior scale, :math:`o` the observation
scale and :math:`c = o^2 / \gamma`, the innovation covariance is :math:`s^2 H S_0
H^T + c S_z = s^2 (H S_0 H^T + S_z / \gamma')` with :math:`\gamma' = s^2 /
c`. The evidence at every :math:`(s, c)` is the problem's at gamma
:math:`\gamma'`, rescaled, and one eigendecomposition gives it at every gamma.

References
----------

Gelman, A., Roberts, G. O., and Gilks, W. R. (1996). Efficient Metropolis
jumping rules. *Bayesian Statistics 5*, 599-607.

Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
Science*, 7(4), 473-483.
