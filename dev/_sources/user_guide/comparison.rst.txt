.. _comparison:

Choosing a method
=================

fips solves a problem whose state cannot be negative (a flux, a
concentration, a rate) in four ways: the analytic posterior, which ignores
the bound, ``"bounded"`` with ``lower=0``, and ``"gauss_newton"`` with a
:class:`~fips.LogNormalForward` or a :class:`~fips.SoftplusForward`. This page
runs the four on one example, a problem shaped like a Salt Lake Valley methane
inversion (the flux of each cell of a grid, with an inventory as its prior,
seen from a few sites), with one prior written four ways, and scores each on
many sets of synthetic data (:doc:`synthetic`). What the example's numbers
say, in short:

- **for a total of the whole state (here, a basin's), the analytic posterior
  is good enough**: its total and its interval held in every world tried
  here, at a hundredth of the others' cost or less;
- **for a map, the lognormal model is best**, and its intervals are right
  when the flux is a factor of the inventory, correlated as its prior says;
- **bounds change the prior**: a Gaussian truncated at zero has a higher mean
  than the Gaussian, so the draws of ``"bounded"`` put the total far above
  an inventory that is the mean. Its mode does not;
- **the softplus model** is right in its own world, and about 5% low on the
  total in a lognormal one.

The rest of the page is the experiment behind them, small enough to run here,
then the same experiment at the Salt Lake Valley's size.

.. contents:: On this page
   :local:
   :depth: 1


The problem
-----------

A grid of cells, two sites, and observations each of which sees the cells
upwind of its site, in that hour's wind direction:

.. ipython:: python

   import time

   import numpy as np
   import pandas as pd
   import fips

   side, n_obs, n_sites = 10, 40, 2  # cells on a side, observations, sites
   trials, n_draws = 40, 100  # trials of each world, draws in each trial
   rng = np.random.default_rng(0)

.. ipython:: python

   x, y = np.meshgrid(np.arange(side), np.arange(side))
   xy = np.c_[x.ravel(), y.ravel()].astype(float)
   state = fips.Space({"flux": pd.MultiIndex.from_arrays([xy[:, 0], xy[:, 1]], names=["x", "y"])})
   observed = fips.Space({"ch4": pd.RangeIndex(n_obs, name="obs_id")})
   sites = rng.uniform(0, side - 1, (n_sites, 2))
   at = rng.integers(0, n_sites, n_obs)  # the site of each observation

.. ipython:: python

   footprint = np.zeros((n_obs, side * side))
   for i in range(n_obs):
       wind = rng.uniform(0, 2 * np.pi)
       along, across = ((xy - sites[at[i]]) @ [[np.cos(wind), -np.sin(wind)], [np.sin(wind), np.cos(wind)]]).T
       footprint[i] = np.exp(-np.abs(across) - np.maximum(along, 0.0) / 3.0) * (along > -1.0)

.. ipython:: python

   footprint[footprint < 0.02] = 0.0
   footprint *= 5.0 / footprint.sum(axis=1, keepdims=True)
   H = fips.Operator(footprint, observed, state)
   inventory = fips.Vector(rng.lognormal(0.0, 0.8, side * side), state)

**One prior, written four ways.** Each cell's prior flux has the inventory as
its mean and 131% of it as its standard deviation (the spread of a lognormal
of log standard deviation 1, the setting of the Salt Lake Valley-shaped tests
in :doc:`positive`), and the errors are correlated over two cells. The
analytic and bounded methods write it as a Gaussian of the flux;
``prior_for`` gives each transform the prior on :math:`u` whose flux has the
same mean and standard deviation, and the correlation is then that of
:math:`u`. The softplus model bends at the median inventory:

.. ipython:: python

   std = np.sqrt(np.e - 1.0) * inventory
   lognormal = fips.LogNormalForward(H, inventory.to_series("flux"))
   softplus = fips.SoftplusForward(H, scale=float(np.median(inventory.values)))
   priors = {
       "analytic": (inventory, std, H),
       "bounded": (inventory, std, H),
       "lognormal": (*lognormal.prior_for(inventory, std), lognormal),
       "softplus": (*softplus.prior_for(inventory, std), softplus),
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
           offset=1900.0,
       )

   linear = build("analytic", fips.Vector(np.zeros(n_obs), observed))
   start = linear.predict(inventory)  # the data the inventory gives

Most cells are seen weakly, as in a valley with a few sites: about half have
an averaging kernel below 0.1.

.. ipython:: python

   seen = fips.solve(build("analytic", start)).dofs_per_element.values
   round(float(np.mean(seen < 0.1)), 2)

Each method's solve, and its flux (a transform's estimate and draws are
:math:`u`, and ``transform`` maps them to the flux). Each trial's posterior is
``post.with_data(obs)``; a ``"gauss_newton"`` one makes 50 draws to tune its
sampler (``burn_in``) before those asked of it:

.. ipython:: python

   def solve(method, obs):
       """The method's posterior of the observations `obs`."""
       if method == "analytic":
           return fips.solve(build(method, obs))
       if method == "bounded":
           return fips.solve(build(method, obs), "bounded", lower=0.0)
       return fips.solve(build(method, obs), "gauss_newton", burn_in=50)

   def flux(method, x):
       """The flux of a state, or of draws: the transform's, or the state itself."""
       return priors[method][2].transform(x) if method in ("lognormal", "softplus") else x


Two worlds
----------

**A lognormal truth.** Each cell's flux is the inventory times a lognormal
factor whose mean is 1, so the inventory is the mean flux, and the
logarithms of the factors correlate over two cells, as the priors do. The
truth's mean, standard deviation and correlations are the priors'; its shape
is the lognormal model's and none of the others'. This is the world of an
inventory that is right on average and wrong by a factor in each place:

.. ipython:: python

   field = fips.Exponential(["x", "y"], 2.0).with_std(1.0).build(state)

   def lognormal_world(rng):
       """A truth lognormal about the inventory (its mean), and the observations it gives."""
       truth = inventory * np.exp(field.sample(rng) - 0.5)
       return truth, linear.predict(truth) + linear.obs_cov.sample(rng)

**Each method's own world.** The truth is a draw from the method's own prior
and the data come through its own model, so an exact posterior's intervals
hold the truth as often as they claim. This checks that each method is right
on its own terms. The lognormal model's own world is the lognormal truth
above. The bounded method's prior is the Gaussian truncated at zero, which is
the bounded posterior of a problem with no data, so its truths are that
posterior's draws:

.. ipython:: python

   def own_world(method):
       """Truths from the method's prior and data through its model, the truth as a flux."""
       problem = build(method, start)
       def world(rng):
           truth, obs = problem.simulate(rng=rng)
           return flux(method, truth), obs
       return world

.. ipython:: python

   nothing = fips.Space({"ch4": pd.RangeIndex(1, name="obs_id")})
   no_data = fips.Problem(inventory, fips.Exponential(["x", "y"], 2.0).with_std(std), fips.Vector([0.0], nothing), fips.Independent().with_std(1.0), fips.Operator(np.zeros((1, side * side)), nothing, state))
   truncated = fips.solve(no_data, "bounded", lower=0.0).sample(trials, rng=100)  # one truth a column

   def truncated_world(rng, truths=iter(truncated.T.to_numpy())):
       """The bounded method's own world: the truncated prior's draws, one a trial."""
       truth = fips.Vector(next(truths), state)
       return truth, linear.predict(truth) + linear.obs_cov.sample(rng)


Scoring
-------

Each trial, the world makes a truth and its data, the method solves them, and
the trial's posterior is compared with the truth, in flux: the total of all
cells and each cell. Every method's posterior is read the same way, from
``n_draws`` draws of it (:meth:`~fips.Posterior.sample`):

- **the total**: its estimate is the mean of the draws' totals, its interval
  the 2.5% and 97.5% quantiles of the draws' totals;
- **each cell**: its interval is the quantiles of its draws, and its error is
  that of the estimate (the mode, or the analytic mean), as a root mean square
  over the cells divided by that of the inventory: 1 is no better than the
  prior.

This is :meth:`~fips.Posterior.trials`' loop, written out, because
``trials`` compares totals of the state, and a transform's state is
:math:`u`: a total of the flux is not a linear function of :math:`u`. Its
intervals are read the same way: ``trials``' ``lower`` and ``upper`` are the
quantiles of each trial's draws for a sampled posterior (``"bounded"``, a
transform) and the estimate plus or minus 1.96 standard deviations for the
analytic one, which is that posterior's interval. Every world draws from one generator and every method's draws from another,
so the four methods get the same data, trial for trial:

.. ipython:: python

   def score(method, world, seed=1):
       """Each trial's error and whether its 95% intervals (the draws' quantiles) hold the truth, in flux."""
       post, rows, clock = solve(method, start), [], time.perf_counter()
       g_world, g_draws = np.random.default_rng(seed), np.random.default_rng(seed + 1)
       for _ in range(trials):
           truth, obs = world(g_world)
           trial = post.with_data(obs)
           t, draws = truth.values, flux(method, trial.sample(n_draws, rng=g_draws)).to_numpy()
           low, high = np.quantile(draws, [0.025, 0.975], axis=1)
           totals, error = draws.sum(axis=0), flux(method, trial.estimate).values - t
           rows.append(
               {
                   "total error": totals.mean() - t.sum(),
                   "total std": totals.std(),
                   "total covered": np.quantile(totals, 0.025) <= t.sum() <= np.quantile(totals, 0.975),
                   "cells covered": np.mean((low <= t) & (t <= high)),
                   "cell error": np.sqrt(np.mean(error**2) / np.mean((inventory.values - t) ** 2)),
               }
           )
       return pd.DataFrame(rows).assign(seconds=(time.perf_counter() - clock) / trials)

The summary has the total's bias, its standard error, the spread of the
total's error over the trials and the spread the posterior claims (each as a
fraction of the inventory's total, as every bias on this page is), and how
often the total's interval holds the truth; then the cells' coverage with its
standard error, the cell error, and the seconds a trial takes:

.. ipython:: python

   def summarize(table):
       """The total's bias, spread and coverage, the cells' coverage and error, with standard errors."""
       n, e = len(table), table["total error"] / inventory.values.sum()
       return pd.Series(
           {
               "bias": e.mean(),
               "bias se": e.std() / np.sqrt(n),
               "spread": e.std(),
               "claimed": table["total std"].mean() / inventory.values.sum(),
               "coverage": table["total covered"].mean(),
               "cells": table["cells covered"].mean(),
               "cells se": table["cells covered"].std() / np.sqrt(n),
               "cell error": table["cell error"].mean(),
               "s a trial": table["seconds"].iloc[0],
           }
       )


The numbers
-----------

The lognormal truth, 40 trials (about a minute):

.. ipython:: python

   methods = ["analytic", "bounded", "lognormal", "softplus"]
   in_lognormal = {m: score(m, lognormal_world) for m in methods}
   pd.DataFrame({m: summarize(in_lognormal[m]) for m in methods}).T.round(3)

Each method in its own world (the lognormal model's is the table above):

.. ipython:: python

   own = {m: score(m, own_world(m)) for m in ["analytic", "softplus"]}
   own["bounded"] = score("bounded", truncated_world)
   own["lognormal"] = in_lognormal["lognormal"]
   pd.DataFrame({m: summarize(own[m]) for m in methods}).T.round(3)

Forty trials know a coverage near 0.95 to about 0.034
(:math:`\sqrt{0.95 \cdot 0.05 / 40}`), and a bias to its ``bias se``, so read
these tables for their large differences, and the next section's for the
small ones. The large ones: in the lognormal truth the bounded draws' total is
more than 20% high and its interval rarely holds the truth, and the lognormal
model's cell error is the lowest; in its own world each method's intervals
hold the truth about as often as they claim. The rest is within the noise.

For the analytic posterior, ``trials`` on the same seed (the same data) gives
the bias of the estimate's total and the coverage of its interval, the
estimate plus or minus 1.96 standard deviations: the loop's, within the noise
of its 100 draws:

.. ipython:: python

   W = fips.Operator(np.ones((1, side * side)), fips.Space({"total": pd.Index(["valley"], name="what")}), state)
   table = solve("analytic", start).trials(lognormal_world, trials, rng=1, W=W)
   error = (table["estimate"] - table["truth"]) / inventory.values.sum()
   round(float(error.mean()), 3), round(float(((table["lower"] <= table["truth"]) & (table["truth"] <= table["upper"])).mean()), 3)


At the Salt Lake Valley's size
------------------------------

The same experiment with 400 cells (20 on a side), 120 observations from
three sites (69% of the cells with an averaging kernel below 0.1), 200 trials,
and 400 draws in each after a ``burn_in`` of 100, on seed 1, was run on
2026-11-06 (an hour and a half on one core). Its tables, with the standard
error of each bias, and 0.015 for a coverage near 0.95:

.. list-table:: The lognormal truth (400 cells, 200 trials)
   :header-rows: 1
   :stub-columns: 1

   * -
     - bias (se)
     - spread
     - claimed
     - coverage
     - cells
     - cell error
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

.. list-table:: Each method's own world (400 cells, 200 trials; the lognormal model's is the table above)
   :header-rows: 1
   :stub-columns: 1

   * -
     - bias (se)
     - spread
     - claimed
     - coverage
     - cells
     - cell error
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

The cells' coverages are known to about 0.002. The small world here shows the
same pattern with less weight: the bounded total's bias grows with the share
of cells the data barely see (in the lognormal truth, with 40 trials: +0.11 at
36 cells, +0.23 at 100, +0.39 at 144, +0.50 at 196 and +0.58 at 400).

**A third world**, a lognormal truth whose cells scatter about the inventory
each on its own (no correlation, which no prior here assumes), at 400 cells:
every total's interval is too wide, since the priors say the cells' errors add
up and the truth's mostly cancel (coverage 0.99 to 1.0 for the analytic,
lognormal and softplus methods, whose claimed spreads are 1.4 to 1.6 times the
real ones); the bounded total is +0.59 and its coverage 0. The lognormal
model's cells cover 0.86, against 0.93 for the analytic posterior's: its prior
in :math:`u` holds the cells to each other more than this truth does. The
order of the maps is the same (cell errors 0.84 lognormal, 0.87 bounded and
softplus, 0.88 analytic).


What the numbers mean
---------------------

**Every method is right on its own terms.** In its own world each one's
intervals hold the truth about as often as they claim, for the total and the
cells, and its total is unbiased. The differences come from what each prior
says when the world is not its own.

**The analytic posterior is good enough for a basin total.** In the lognormal
truth its total is unbiased (+0.7%, with a standard error of 1.0%) and its
interval holds the truth 97% of the time; its cells' intervals cover 95.5%.
The Gaussian posterior puts some cells below zero, but a total sums many
cells, and here the sum of their errors is close enough to Gaussian.
It costs a few milliseconds a solve.

**The lognormal model makes the best map.** Its cell error is 6% lower than
the analytic posterior's in the lognormal truth (0.714 against 0.763) and 4%
lower with independent cells, and its total is as good as the analytic one,
with an interval a third narrower (0.115 against 0.165) that still covers.
Its cells' intervals are right when the flux is a factor of the inventory with
the correlations its prior assumes, and too narrow (0.86) when the truth's
factors are less correlated than its prior says. A cell with no inventory
stays at zero (:doc:`positive`).

**Bounds change the prior, and the change reaches the total.** The bounded
method's prior is the Gaussian prior truncated at zero, and truncation raises
the mean. A Gaussian truncated at zero never has a standard deviation above
its mean, so none has the inventory as its mean and 131% of it as its
standard deviation; and cells that are correlated and must all be positive
are pushed up together. In this problem the truncated prior's mean total is 2.36 times
the inventory's (the mean of the truths of the bounded method's own world; a
Gibbs sampler, independent of fips, gives 2.10 against fips's 2.12 for a
36-cell version). Where the data barely see a cell, its posterior is that
prior, so the draws put the total above the truth by 56% of the inventory's
total when the inventory is the mean, and their interval almost never holds
it. With a standard deviation of 50% of the inventory instead of 131%, the
draws' total was still 9% high (400 cells, 100 trials, a standard error of
0.6%) and its interval held the truth in 57% of the trials. The mode is not
moved: a weakly seen cell's mode stays at its inventory, so the sum of the
mode is within 2.5% of the truth in the lognormal worlds, and the mode's map
is about the analytic one (cell error 0.750 against 0.763). But the mode's
total is not the bounded posterior's: in the bounded method's own world,
whose truths are the truncated prior's, it is below the truth by 45% of the
inventory's total, and an interval of the mode's total plus or minus 1.96
standard deviations held the truth in 2.5% of the trials. So use
``"bounded"`` for a map that must not go below zero, and do not read a total,
or a cell's interval, from its draws unless the truncated prior is the prior
you mean. These tables are of the truncated draws; the rectified draws
(:ref:`bounds-rectified`) shift a total less, and were not run here.

**The softplus model** is a Gaussian prior where the flux is large and bends
near zero. Its own world is right; in the lognormal truth its total is 4.6%
low (five standard errors), its cells' intervals are a little too wide
(0.976), and its map is about the analytic posterior's.

**Which interval to read.** Every interval here is a quantile of the draws.
For the analytic posterior that is the estimate plus or minus 1.96 standard
deviations; for a sampled one, ``"bounded"`` or a transform, the quantiles of
its draws, which is how :meth:`~fips.Posterior.uncertainty` and
:meth:`~fips.Posterior.trials` give its ``lower`` and ``upper``. Read as the
mode plus or minus 1.96 standard deviations instead, the bounded method's
cells covered 0.90 in its own world (0.945 from quantiles), and its total
0.025 (0.955). A total of a transform's flux is not a total of its state:
take it from :meth:`~fips.Posterior.sample`.

These numbers are for one problem: a prior of 131% correlated over two cells,
these footprints and this noise. Another inversion's own trials, at its own
size, say how its methods compare; this page is the code to run them.


What each costs
---------------

At 400 cells, on one core: an analytic solve took 0.014 s, a bounded one
0.016 s, and a ``"gauss_newton"`` one, which finds the mode, 0.05 to 0.06 s (a median of 7 Newton or
Gauss-Newton steps, 9 for the softplus model in its own world). The spread is
where they differ: the default 2000 draws took 6 s for ``"bounded"``, 17 s
for the lognormal model and 23 s for the softplus model, against no draws for
the analytic posterior. A trial with 400 draws took 0.013 s for the analytic
posterior, 0.2 to 1.7 s for ``"bounded"`` (more where more cells sit near the
bound) and 4.4 to 5.2 s for either transform: 200 trials of a transform take
about 17 minutes. The ``s a trial`` column above measures the same at the
page's size.
