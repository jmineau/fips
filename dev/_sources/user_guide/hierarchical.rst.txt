.. _hierarchical:

The uncertainty of the settings
===============================

:doc:`settings` fits the covariance settings and solves at the fit, which
treats them as known. The rule: **give each setting a prior, and**
:func:`fips.hierarchical` **draws the settings from their posterior and the
state at each of those draws, so intervals read from its draws hold the
uncertainty of the settings.** The answer is not one Gaussian posterior but a
mixture of them, one for each draw of the settings, held by a
:class:`~fips.Hierarchical`.

.. contents:: On this page
   :local:
   :depth: 1


The settings' posterior
-----------------------

With a prior :math:`p(s)` on the settings, their posterior is the prior times
the evidence of the problem built at them (:attr:`~fips.Posterior.log_evidence`,
exact for a linear problem):

.. math::

    p(s \mid z) \propto p(s) \, p(z \mid s).

The posterior of the state is the posterior at each setting, averaged over
the settings' posterior,

.. math::

    p(x \mid z) = \int p(x \mid s, z) \, p(s \mid z) \, ds,

so a draw of the settings from :math:`p(s \mid z)`, then of the state from
:math:`p(x \mid s, z)` (exactly, :meth:`~fips.Posterior.sample`), is a draw
of :math:`p(x \mid z)`. Its variance is

.. math::

    \operatorname{var}(x \mid z) = E\big[\hat{S}(s)\big] + \operatorname{var}\big[\hat{x}(s)\big],

over the settings' posterior: the average of each setting's posterior
variance, and the spread of the estimates between settings. A solve at fitted
settings gives the first term at one setting and leaves out the second.

:func:`fips.hierarchical` draws the settings by a random-walk Metropolis, in
the log of each positive setting. It starts at the settings' posterior mode,
proposes steps from Laplace's approximation there (the inverse of a numerical
Hessian of the log posterior), and during the burn-in scales the steps until
about 0.3 of them are accepted. Each step solves one problem; at each kept
step it draws the state once, at that step's settings.


A world at known settings
-------------------------

The world of :doc:`settings`: a hundred and fifty cells along a line, 1 km
apart, two hundred observations of one to four neighboring cells, and a truth
drawn at a prior std of 1, a correlation length of 5 km and a mismatch std of
0.5:

.. ipython:: python

   import numpy as np
   import pandas as pd
   from scipy.special import ndtri
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

The function from the settings to the problem takes them as a dict, by name:

.. ipython:: python

   def build(settings, obs, H=H):
       """The problem at the settings: the prior std, the correlation length (km), the mismatch std."""
       return fips.Problem(
           prior=prior,
           prior_cov=fips.Exponential("x", settings["length"]).with_std(settings["std"]),
           obs=obs,
           obs_cov=fips.Independent().with_std(settings["noise"]),
           forward=H,
       )

   true = pd.Series({"std": 1.0, "length": 5.0, "noise": 0.5})
   world = build(true, fips.Vector(np.zeros(n_obs), observed))
   truth, obs = world.simulate(rng)

Each setting gets a prior that knows it to a factor of :math:`e` (a
:class:`~fips.distributions.LogNormal` with a standard deviation of 1 in the log), about a
guess that is off by a factor of two in the two standard deviations and 2.5 in
the length. :class:`~fips.distributions.HalfNormal` and
:class:`~fips.distributions.Uniform` are the other two
(:mod:`fips.distributions`).

.. ipython:: python

   priors = {
       "std": fips.distributions.LogNormal(np.log(0.5), 1.0),
       "length": fips.distributions.LogNormal(np.log(2.0), 1.0),
       "noise": fips.distributions.LogNormal(np.log(1.0), 1.0),
   }
   result = fips.hierarchical(lambda s: build(s, obs), priors, draws=2000, burn_in=500, seed=0)
   table = result.settings.quantile([0.025, 0.5, 0.975]).T.set_axis(["2.5%", "median", "97.5%"], axis=1)
   table.insert(0, "true", true)
   table.assign(mode=result.mode).round(3)

The two standard deviations are known to about 25% and 15% (their 95%
intervals), each holding the truth. The length is known to a factor of three:
from 1.8 to 5.8 km, about its median of 2.8 km, the fit :doc:`settings` found.
:attr:`~fips.Hierarchical.mode` is the settings' posterior mode, where the
chain started, close to that fit. The chain's diagnostics are its acceptance
and the effective sample size of each setting, about as many independent
draws as the 2000 correlated ones are worth:

.. ipython:: python

   round(result.acceptance, 2), result.ess.round().to_dict()

A standard error of a setting's mean is its standard deviation over the root
of its effective sample size: here about a tenth of its spread. Ask for more
draws to make it smaller.


Totals with and without the settings' uncertainty
-------------------------------------------------

The flux of three regions of 50 km each:
:meth:`~fips.Hierarchical.uncertainty` reads each total's mean, standard
deviation and interval from the draws (here 90%), and
:meth:`~fips.Hierarchical.at` gives the ordinary posterior at one setting, the
mode, whose interval leaves the settings' uncertainty out:

.. ipython:: python

   x = cells.select("flux")
   regions = pd.Series(np.select([x < 50, x < 100], ["west", "middle"], "east"), index=x, name="region")
   W = fips.aggregate.sum(cells, by=regions)

.. ipython:: python

   def intervals(result, truth, level=0.9):
       """Each region's truth, and its interval with the settings' uncertainty and at their mode."""
       mixed = result.uncertainty(W, level=level)
       at_mode = result.at(result.mode)
       estimate = (W @ at_mode.estimate).values
       std = at_mode.uncertainty(W)["posterior_std"].values
       half = ndtri(0.5 + level / 2) * std
       return pd.DataFrame(
           {
               "truth": (W @ truth).values,
               "std": mixed["posterior_std"].values,
               "lower": mixed["lower"].values,
               "upper": mixed["upper"].values,
               "std_at_mode": std,
               "lower_at_mode": estimate - half,
               "upper_at_mode": estimate + half,
           },
           index=W.rows.index.get_level_values("region"),
       )

   intervals(result, truth).round(2)

The two agree to about 2%: two hundred observations pin these totals, and the
length, the setting known least, moves how a total splits between its cells
more than the total itself (the prior std of each region's total is 15.7; the
posterior's is under 2).


Few observations, and the fast path
-----------------------------------

The settings' uncertainty matters when few observations decide them. Fifteen
observations, all in the western two regions:

.. ipython:: python

   few = fips.Space({"ch4": pd.RangeIndex(15, name="obs_id")})
   sparse = np.zeros((15, n_cells))
   for i, start in enumerate(rng.integers(0, 96, 15)):
       sparse[i, start : start + rng.integers(1, 5)] = rng.uniform(0.5, 1.5)

   H_few = fips.Operator(sparse, few, cells)
   few_world = build(true, fips.Vector(np.zeros(15), few), H=H_few)
   truth_few, obs_few = few_world.simulate(rng)

Here the length is held at its true 5 km and the two standard deviations are
the settings. When the only settings are a factor on the prior std and one on
the mismatch's, pass the problem at factors of 1 instead of a function, and
name them ``"prior_scale"`` and ``"obs_scale"`` (or ``"gamma"``): one
eigendecomposition then gives the evidence at every step without a solve, and
the state's draws at every kept step in one product. This run of 2500 steps
takes about a tenth of a second, where the function would take a solve each:

.. ipython:: python

   unit = {"std": 1.0, "length": 5.0, "noise": 1.0}
   scales = {"prior_scale": fips.distributions.LogNormal(np.log(0.5), 1.0), "obs_scale": fips.distributions.LogNormal(0.0, 1.0)}
   fast = fips.hierarchical(build(unit, obs_few, H=H_few), scales, seed=0)
   fast.settings.quantile([0.025, 0.5, 0.975]).round(3)
   intervals(fast, truth_few).round(2)

Each standard deviation is now known to a factor of three to six, and the
totals' standard deviations with their uncertainty are 4% to 9% larger. Whether that is enough
is a question of coverage: over worlds, how often each interval holds the
true total. First in worlds at the true settings, then in worlds whose
settings are drawn from their priors, 200 of each:

.. ipython:: python

   def coverage(make_world, trials=200):
       """The fraction of worlds whose 90% interval holds each region's true total."""
       rows = []
       for k in range(trials):
           truth_k, obs_k = make_world()
           table = intervals(fips.hierarchical(build(unit, obs_k, H=H_few), scales, seed=k), truth_k)
           rows.append(pd.DataFrame({
               "with": table["lower"].le(table["truth"]) & table["truth"].le(table["upper"]),
               "without": table["lower_at_mode"].le(table["truth"]) & table["truth"].le(table["upper_at_mode"]),
           }))
       return pd.concat(rows).groupby(level="region").mean()

.. ipython:: python

   coverage(lambda: few_world.simulate(rng))

.. ipython:: python

   def from_priors():
       """A world whose two standard deviations are drawn from their priors."""
       std, noise = (float(scales[name].sample(1, rng)[0]) for name in ("prior_scale", "obs_scale"))
       world_k = build({"std": std, "length": 5.0, "noise": noise}, fips.Vector(np.zeros(15), few), H=H_few)
       return world_k.simulate(rng)

   coverage(from_priors)

With 200 worlds a coverage is known to about 0.02. In the worlds drawn from
the priors, the intervals with the settings' uncertainty cover at their level,
89% to 93.5%, as a posterior must on average over its own prior; those at the
mode cover 85% to 89%. At the true settings both cover less, 84.5% to 88.5%
with the uncertainty and 82.5% to 84.5% without: there the priors pull each
standard deviation towards their median, the prior std's (0.5) below the
truth and the mismatch std's (1) above it, and fifteen observations do not
undo all of the pull. With the prior std's prior median at the truth instead,
the same worlds cover 87% to 91.5% with the uncertainty (a run not on this
page). A prior on the settings is a claim like any other, and the intervals
are only as good as it.


What it costs, and what it does not do
--------------------------------------

A step of the general path is a solve of the problem the function builds
(here 4 ms for 150 cells and 200 observations, so the 2500 steps above took
about 12 s), and the search for the mode and its Hessian about sixty more. A step of the
fast path is :math:`O(m)` (or :math:`O(n)`, whichever is smaller) after one
eigendecomposition, whatever the size of the problem. The draws of the state
are kept, a ``state x draws`` array: 2000 draws of 30,000 cells are 480 MB.
The settings' estimates have Monte Carlo errors that
:attr:`~fips.Hierarchical.ess` measures; four hundred effective draws give a
setting's mean to a twentieth of its spread.

The posterior of the state is a mixture, so what is defined for one Gaussian
(the gain, DOFS, the chi-squared tests, the evidence) is read from the
posterior at one setting, ``result.at(settings)``. The method is the analytic
one: a function that returns a ``"bounded"`` or ``"gauss_newton"`` posterior
is refused, since neither gives the exact evidence and exact draws each step
needs.

References
~~~~~~~~~~

Gelman, A., Roberts, G. O., and Gilks, W. R. (1996). Efficient Metropolis
jumping rules. *Bayesian Statistics 5*, 599-607.

Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
Science*, 7(4), 473-483.
