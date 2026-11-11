.. _comparison:

Comparing ways to keep a quantity positive
==========================================

A state that cannot be negative has four answers in fips: the analytic
posterior, which ignores the bound; ``"bounded"`` with ``lower=0``; and
``"gauss_newton"`` with a :class:`~fips.LogNormalForward` or a
:class:`~fips.SoftplusForward`. This example asks how they differ. It solves
one synthetic problem by all four, with one prior written four ways, on many
sets of synthetic data from several worlds (:doc:`../synthetic`), and scores
each answer against the truth that made the data.

What it finds, for this problem:

- **In its own world, every method is right.** Its intervals hold the truth
  about as often as they claim, and its total is unbiased.
- **For a total of the whole state, the analytic posterior is enough.** Its
  total is unbiased in every world whose truth is correlated as the priors
  say, and its interval covers in every world tried, at a hundredth of the
  others' cost or less.
- **For each element, the lognormal model is best** when the truth is a
  factor of the reference.
- **The bounded method's draws change the total.** Cutting a Gaussian at
  zero raises its mean, so where the data see little, the draws put the total
  far above a reference that is the mean. Its mode is not moved.
- **The softplus model** is right in its own world, and its total is about 5%
  low in the lognormal one.

:doc:`../methods` turns these into advice. The rest of this page is the
experiment: its code, small enough to run when the docs are built, then the
same experiment at 400 elements.

.. contents:: On this page
   :local:
   :depth: 1


The problem
-----------

A field on a 10 by 10 grid, a reference value for each element (the prior
mean), and 40 observations from two sites. Each observation is a weighted sum
of the elements in a narrow plume from its site, pointing in a random
direction:

.. ipython:: python

   import time

   import numpy as np
   import pandas as pd
   import fips

   side, n_obs, n_sites = 10, 40, 2  # elements on a side, observations, sites
   trials, n_draws, burn_in = 40, 100, 50  # trials of each world, draws in each, tuning draws
   rng = np.random.default_rng(0)

.. ipython:: python

   x, y = np.meshgrid(np.arange(side), np.arange(side))
   xy = np.c_[x.ravel(), y.ravel()].astype(float)
   state = fips.Space({"field": pd.MultiIndex.from_arrays([xy[:, 0], xy[:, 1]], names=["x", "y"])})
   observed = fips.Space({"obs": pd.RangeIndex(n_obs, name="obs_id")})
   sites = rng.uniform(0, side - 1, (n_sites, 2))
   at = rng.integers(0, n_sites, n_obs)  # the site of each observation

.. ipython:: python

   footprint = np.zeros((n_obs, side * side))
   for i in range(n_obs):
       angle = rng.uniform(0, 2 * np.pi)
       along, across = ((xy - sites[at[i]]) @ [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).T
       footprint[i] = np.exp(-np.abs(across) - np.maximum(along, 0.0) / 3.0) * (along > -1.0)

.. ipython:: python

   footprint[footprint < 0.02] = 0.0
   footprint *= 5.0 / footprint.sum(axis=1, keepdims=True)
   H = fips.Operator(footprint, observed, state)
   reference = fips.Vector(rng.lognormal(0.0, 0.8, side * side), state)

**One prior, written four ways.** Each element's prior has the reference as
its mean and 131% of it as its standard deviation (the spread of a lognormal
whose logarithm has a standard deviation of 1), and the errors are correlated
over two grid steps. The analytic and bounded methods write it as a Gaussian
of the value. For each transform, ``prior_for`` gives the prior on :math:`u`
whose value has the same mean and standard deviation; the correlation is then
that of :math:`u`. The softplus model bends at the median reference:

.. ipython:: python

   std = np.sqrt(np.e - 1.0) * reference
   lognormal = fips.LogNormalForward(H, reference)
   softplus = fips.SoftplusForward(H, scale=float(np.median(reference.values)))
   priors = {
       "analytic": (reference, std, H),
       "bounded": (reference, std, H),
       "lognormal": (*lognormal.prior_for(reference, std), lognormal),
       "softplus": (*softplus.prior_for(reference, std), softplus),
   }

.. ipython:: python

   def build(method, obs):
       """The problem of the observations `obs`, as `method` writes it."""
       mean, sd, forward = priors[method]
       return fips.Problem(
           prior=mean,
           prior_cov=fips.Exponential(["x", "y"], 2.0).with_std(sd),
           obs=obs,
           obs_cov=fips.Independent().with_std(0.3),
           forward=forward,
       )

   linear = build("analytic", fips.Vector(np.zeros(n_obs), observed))
   start = linear.predict(reference)  # the data the reference gives

Most elements are seen weakly: about half have an averaging kernel below 0.1.

.. ipython:: python

   seen = fips.solve(build("analytic", start)).dofs_per_element.values
   round(float(np.mean(seen < 0.1)), 2)

Each method's solve, and the value of a state. A transform's estimate and
draws are :math:`u`, and ``transform`` maps them to the value:

.. ipython:: python

   def solve(method, obs):
       """The method's posterior of the observations `obs`."""
       if method == "analytic":
           return fips.solve(build(method, obs))
       if method == "bounded":
           return fips.solve(build(method, obs), "bounded", lower=0.0)
       return fips.solve(build(method, obs), "gauss_newton", burn_in=burn_in)

   def value(method, v):
       """The value of a state, or of draws: the transform's, or the state itself."""
       return priors[method][2].transform(v) if method in ("lognormal", "softplus") else v


The worlds
----------

A world makes a truth and the observations it gives. Two kinds are used.

**A lognormal truth.** Each element is the reference times a lognormal factor
whose mean is 1, so the reference is the mean value, and the logarithms of
the factors correlate over two grid steps, as the priors do. The truth has
the priors' mean, standard deviation and correlation; its shape is the
lognormal model's and none of the others'. It is the world of a reference
that is right on average and wrong by a factor in each place:

.. ipython:: python

   factor_cov = fips.Exponential(["x", "y"], 2.0).with_std(1.0).build(state)

   def lognormal_world(rng):
       """A truth lognormal about the reference (its mean), and the observations it gives."""
       truth = reference * np.exp(factor_cov.sample(rng) - 0.5)
       return truth, linear.predict(truth) + linear.obs_cov.sample(rng)

**Each method's own world.** The truth is a draw from the method's own prior,
and the data come through its own model, so an exact posterior's intervals
hold the truth as often as they claim. This checks each method on its own
terms. The lognormal model's own world is the lognormal truth above:

.. ipython:: python

   def own_world(method):
       """Truths from the method's prior and data through its model, the truth as a value."""
       problem = build(method, start)
       def world(rng):
           truth, obs = problem.simulate(rng=rng)
           return value(method, truth), obs
       return world

The bounded method's prior is the Gaussian cut at zero. That is the bounded
posterior of a problem with no data, so its truths are that posterior's
draws:

.. ipython:: python
   :okwarning:

   nothing = fips.Space({"obs": pd.RangeIndex(1, name="obs_id")})
   no_data = fips.Problem(reference, fips.Exponential(["x", "y"], 2.0).with_std(std), fips.Vector([0.0], nothing), fips.Independent().with_std(1.0), fips.Operator(np.zeros((1, side * side)), nothing, state))
   truncated = fips.solve(no_data, "bounded", lower=0.0).sample(trials, rng=100)  # one truth a column

   def truncated_world(rng, truths=iter(truncated.T.to_numpy())):
       """The bounded method's own world: the cut prior's draws, one a trial."""
       truth = fips.Vector(next(truths), state)
       return truth, linear.predict(truth) + linear.obs_cov.sample(rng)


The scoring
-----------

Each trial, the world makes a truth and its data, the method solves the data
(``post.with_data(obs)``), and the posterior is compared with the truth in
the value, for the total of all elements and for each element. Every
posterior is read the same way, from ``n_draws`` of its draws
(:meth:`~fips.Posterior.sample`):

- **the total**: its estimate is the mean of the draws' totals, its interval
  the 2.5% and 97.5% quantiles of the draws' totals;
- **each element**: its interval is the quantiles of its draws, and its error
  is that of the estimate (the mode, or the analytic mean), as a root mean
  square over the elements divided by the reference's: 1 is no better than
  the prior.

This is the loop of :meth:`~fips.Posterior.trials`, written out, because
``trials`` aggregates the state, and a transform's state is :math:`u`: a
total of the value is not a linear function of :math:`u`. One generator makes
every world's data and another every method's draws, so the four methods see
the same data, trial for trial:

.. ipython:: python

   def score(method, world, seed=1):
       """Each trial's error and whether its 95% intervals (the draws' quantiles) hold the truth."""
       post, rows, clock = solve(method, start), [], time.perf_counter()
       g_world, g_draws = np.random.default_rng(seed), np.random.default_rng(seed + 1)
       for _ in range(trials):
           truth, obs = world(g_world)
           trial = post.with_data(obs)
           t, draws = truth.values, value(method, trial.sample(n_draws, rng=g_draws)).to_numpy()
           low, high = np.quantile(draws, [0.025, 0.975], axis=1)
           totals, error = draws.sum(axis=0), value(method, trial.estimate).values - t
           rows.append(
               {
                   "total error": totals.mean() - t.sum(),
                   "total std": totals.std(),
                   "total covered": np.quantile(totals, 0.025) <= t.sum() <= np.quantile(totals, 0.975),
                   "elements covered": np.mean((low <= t) & (t <= high)),
                   "element error": np.sqrt(np.mean(error**2) / np.mean((reference.values - t) ** 2)),
               }
           )
       return pd.DataFrame(rows).assign(seconds=(time.perf_counter() - clock) / trials)

Each row of a summary is one method:

- ``bias`` and ``bias se``: the mean error of the total, and its standard
  error;
- ``spread``: the standard deviation of the total's error over the trials;
- ``claimed``: the standard deviation the posterior claims for the total;
- ``coverage``: how often the total's interval holds the truth;
- ``elements`` and ``elements se``: how often an element's interval holds the
  truth, and its standard error;
- ``element error``: the error of the estimate, as above;
- ``s a trial``: the seconds a trial takes.

The first three are fractions of the reference's total, as is every bias on
this page:

.. ipython:: python

   def summarize(table):
       """The total's bias, spread and coverage, the elements' coverage and error, with standard errors."""
       n, e = len(table), table["total error"] / reference.values.sum()
       return pd.Series(
           {
               "bias": e.mean(),
               "bias se": e.std() / np.sqrt(n),
               "spread": e.std(),
               "claimed": table["total std"].mean() / reference.values.sum(),
               "coverage": table["total covered"].mean(),
               "elements": table["elements covered"].mean(),
               "elements se": table["elements covered"].std() / np.sqrt(n),
               "element error": table["element error"].mean(),
               "s a trial": table["seconds"].iloc[0],
           }
       )


The results
-----------

**The lognormal truth**, 40 trials:

.. ipython:: python
   :okwarning:

   methods = ["analytic", "bounded", "lognormal", "softplus"]
   in_lognormal = {m: score(m, lognormal_world) for m in methods}
   pd.DataFrame({m: summarize(in_lognormal[m]) for m in methods}).T.round(3)

**Each method in its own world** (the lognormal model's is the table above):

.. ipython:: python
   :okwarning:

   own = {m: score(m, own_world(m)) for m in ["analytic", "softplus"]}
   own["bounded"] = score("bounded", truncated_world)
   own["lognormal"] = in_lognormal["lognormal"]
   pd.DataFrame({m: summarize(own[m]) for m in methods}).T.round(3)

Forty trials know a coverage near 0.95 to about 0.034
(:math:`\sqrt{0.95 \cdot 0.05 / 40}`), and a bias to its ``bias se``. Read
these two tables for their large differences. In the lognormal truth, the
bounded draws' total is more than 20% high and its interval rarely holds the
truth, and the lognormal model's element error is the lowest. In its own
world, each method's intervals hold the truth about as often as they claim.
The element coverages there sit a little below 0.95 for every method because
of the draws, not the methods: the 2.5% and 97.5% quantiles of 100 exact
draws hold the truth about 93% of the time (:math:`94/101`), and of 400 draws
about 94.5%. The rest is within the noise.

For the analytic posterior, :meth:`~fips.Posterior.trials` on the same seed
(the same data) gives the bias of the estimate's total and the coverage of
its interval, the estimate plus or minus 1.96 standard deviations. They match
the loop's within the noise of its 100 draws:

.. ipython:: python

   W = fips.Operator(np.ones((1, side * side)), fips.Space({"total": pd.Index(["all"], name="aggregate")}), state)
   table = solve("analytic", start).trials(lognormal_world, trials, rng=1, W=W)
   error = (table["estimate"] - table["truth"]) / reference.values.sum()
   round(float(error.mean()), 3), round(float(((table["lower"] <= table["truth"]) & (table["truth"] <= table["upper"])).mean()), 3)

At 400 elements
^^^^^^^^^^^^^^^

The small run tells the large differences apart; the small ones need a larger
run. The same code with ``side, n_obs, n_sites = 20, 120, 3`` and ``trials,
n_draws, burn_in = 200, 400, 100``, on seed 1, has 400 elements, 69% of them
with an averaging kernel below 0.1. Its coverages near 0.95 are known to
about 0.015 and its element coverages to about 0.002; the standard error of
each bias is in parentheses:

.. list-table:: The lognormal truth (400 elements, 200 trials)
   :header-rows: 1
   :stub-columns: 1

   * -
     - bias (se)
     - spread
     - claimed
     - coverage
     - elements
     - element error
   * - analytic
     - +0.007 (0.010)
     - 0.147
     - 0.165
     - 0.970
     - 0.955
     - 0.763
   * - bounded
     - +0.565 (0.010)
     - 0.136
     - 0.083
     - 0.010
     - 0.837
     - 0.750
   * - lognormal
     - +0.001 (0.008)
     - 0.119
     - 0.115
     - 0.945
     - 0.944
     - 0.714
   * - softplus
     - -0.046 (0.009)
     - 0.130
     - 0.136
     - 0.960
     - 0.976
     - 0.759

.. list-table:: Each method's own world (400 elements, 200 trials; the lognormal model's is the table above)
   :header-rows: 1
   :stub-columns: 1

   * -
     - bias (se)
     - spread
     - claimed
     - coverage
     - elements
     - element error
   * - analytic
     - +0.004 (0.011)
     - 0.162
     - 0.165
     - 0.975
     - 0.946
     - 0.692
   * - bounded
     - -0.005 (0.008)
     - 0.112
     - 0.111
     - 0.955
     - 0.945
     - 0.569
   * - softplus
     - +0.002 (0.010)
     - 0.137
     - 0.133
     - 0.950
     - 0.942
     - 0.698

**A third world: independent factors.** A lognormal truth whose elements
scatter about the reference each on its own, with no correlation (which no
prior here assumes), at 400 elements. Every total's interval is too wide: the
priors say the elements' errors add up, and the truth's mostly cancel. The
analytic, lognormal and softplus totals cover 0.99 to 1.0, and claim 1.4 to
1.6 times their real spread. The bounded total is +0.59 and never covers. The
lognormal model's elements cover 0.86, against the analytic posterior's 0.93:
its prior in :math:`u` holds the elements to each other more than this truth
does. The order of the element errors is the same as above: 0.84 for the
lognormal model, 0.87 for the bounded and softplus ones, 0.88 for the
analytic posterior.

**How the bounded bias grows.** The bounded total's bias grows with the share
of elements the data barely see. In the lognormal truth with 40 trials it is
+0.11 at 36 elements, +0.23 at 100, +0.39 at 144, +0.50 at 196 and +0.58 at
400. With a prior standard deviation of 50% of the reference instead of 131%
(400 elements, 100 trials), the draws' total is still 9% high (standard error
0.6%), and its interval holds the truth in 57% of the trials.


What the results mean
---------------------

**The analytic posterior.** Its total is unbiased in both worlds with
correlated truths (+0.7% and +0.4%, each with a standard error of about 1%),
and its interval covers in every world. Some elements come out below zero,
but a total sums many elements, and the sum of their errors is close to
Gaussian. Its element intervals cover as they claim. It is the cheapest by
far, and enough when the total is what you report.

**The lognormal model.** It makes the best element estimates: an element
error 6% below the analytic posterior's in the lognormal truth (0.714 against
0.763), and 4% below with independent factors. Its total is as good as the
analytic one, with an interval a third narrower that still covers. Its element
intervals are right when the truth is a factor of the reference correlated as
its prior says, and too narrow (0.86) when the truth's factors are less
correlated. An element whose reference is zero stays at zero
(:doc:`../positive`).

**The bounded method.** Its prior is the Gaussian cut at zero, and cutting
raises the mean. A Gaussian cut at zero never has a standard deviation above
its mean, so no cut prior has the reference as its mean and 131% of it as its
standard deviation; and correlated elements that must all be positive are
pushed up together. Here the cut prior's mean total is 2.36 times the
reference's (on a 36-element version, an independent Gibbs sampler gives 2.10
against fips's 2.12). Where the data barely see an element, its posterior is
that prior, so the draws put the total above a truth whose mean is the
reference, by 56% of the reference's total, and their interval almost never
holds it. The mode is not moved: a weakly seen element's mode stays at its
reference, the mode's total is within 2.5% of the truth in the lognormal
worlds, and its element error is about the analytic one (0.750 against
0.763). But the mode's total is not the bounded posterior's: in the bounded
method's own world it is below the truth by 45% of the reference's total, and
the mode plus or minus 1.96 standard deviations holds the total in 2.5% of
the trials (0.955 from the draws' quantiles) and an element in 0.90 (0.945).
Use ``"bounded"`` for element estimates that must not go below zero; read a
total or an interval from its draws only if the cut prior is the prior you
mean. These are the truncated draws; the rectified draws
(:ref:`bounds-rectified`) shift a total less, and were not run here.

**The softplus model.** Its prior is Gaussian where the value is large and
bends near zero. In its own world it is right. In the lognormal truth its
total is 4.6% low (five standard errors), its element intervals are a little
too wide (0.976), and its element error is about the analytic posterior's.

These numbers are for one problem: a prior of 131% correlated over two grid
steps, these footprints and this noise. Another problem's own trials, at its
own size, say how its methods compare; this page is the code to run them.


Cost
----

At 400 elements, a solve takes about ten milliseconds for the analytic and
bounded methods, and tens of milliseconds for ``"gauss_newton"``, which takes
about ten Newton or Gauss-Newton steps to find the mode. The draws are where
they differ. The default 2000 draws take seconds for ``"bounded"`` and tens
of seconds for either transform; the analytic posterior needs none. A trial
with 400 draws takes about ten milliseconds for the analytic posterior, a
fraction of a second to a couple of seconds for ``"bounded"`` (more where
more elements sit near the bound), and about five seconds for a transform, so
200 trials of a transform take a quarter of an hour. The ``s a trial`` column
of the tables above measures the same at this page's size.
