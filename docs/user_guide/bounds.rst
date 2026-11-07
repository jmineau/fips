.. _bounds:

Bounds on the state
===================

A flux of methane cannot be negative, yet the Gaussian posterior of a flux
inversion often is in cells where the prior is small and the observations see
little. ``fips.solve(problem, "bounded", lower={"flux": 0.0})`` holds the
state to bounds. The rule: **the bounded estimate is the analytic one moved
onto the bounds along the posterior's correlations.** An element the analytic
estimate puts below its bound goes to the bound, and its neighbors move by what
the posterior says they would if that element were known to be at its bound.
A neighbor the move would push out is held at its bound too. Clipping (setting
the negative cells to zero and nothing else) is not this: it always raises the
total, while moving along the correlations can lower it as well.

.. contents:: On this page
   :local:
   :depth: 1


What the bounded posterior is
-----------------------------

The prior with the bounds is the Gaussian prior set to zero outside them
(whose mean is above the Gaussian's: :doc:`comparison` shows how far), so
the posterior with the bounds is the analytic posterior :math:`N(\hat{x},
\hat{S})` truncated to the box :math:`l \le x \le u`. It is not Gaussian, and
its mean is not its mode:

- the **estimate** is the **mode**, the point of the box where the analytic
  cost is least, which is the point of the box nearest :math:`\hat{x}` in the
  metric of :math:`\hat{S}`. fips finds it with an active set that needs only
  the columns of :math:`\hat{S}` at the elements on a bound, from the analytic
  factors, and checks the optimality conditions of the answer before it
  returns it. ``post.mean`` raises: the mode is not the mean;
- the **uncertainty** comes from **draws** of the truncated posterior, by exact
  Hamiltonian Monte Carlo (Pakman and Paninski, 2014). An element on its bound
  still has spread, on one side of the bound.

The bounds are ``lower`` and ``upper``: a number for every element, a
:class:`~fips.Vector` with an element's own bound (``-inf`` or ``inf`` for
none), or a dict by block. ``{"flux": 0.0}`` bounds the fluxes and leaves the
other blocks free, as a bias block must be. Bounds go to the solve, not to the
:class:`~fips.Problem`: the problem is the Gaussian question, and the bounds
are how the method answers it.


A small problem
---------------

Twenty cells along a line, with a prior of 0.2 and correlated errors, seen
from two sites, each with a bias of its own; the data come from a truth drawn
lognormal, most of it near zero:

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   rng = np.random.default_rng(0)
   n_cells, n_obs = 20, 24
   cells = pd.Index(np.arange(n_cells) + 0.5, name="x")  # km along a line
   sites = pd.Index(["north", "south"], name="site")
   prior = fips.Vector.from_blocks(
       {"flux": pd.Series(0.2, index=cells), "bias": pd.Series(0.0, index=sites)},
       name="prior",
   )

.. ipython:: python

   obs_labels = pd.MultiIndex.from_arrays(
       [np.repeat(sites, n_obs // 2), np.tile(np.arange(n_obs // 2), 2)],
       names=["site", "hour"],
   )
   obs_space = fips.Space({"ch4": obs_labels})
   footprint = rng.uniform(0.0, 1.0, (n_obs, n_cells))
   footprint *= rng.uniform(0.0, 1.0, (n_obs, n_cells)) < 0.3
   south = (obs_labels.get_level_values("site") == "south").astype(int)
   H = np.hstack([footprint, np.eye(2)[south]])  # each site sees its own bias
   forward = fips.Operator(H, obs_space, prior.space)

.. ipython:: python

   def build(obs):
       """The problem of the observations `obs`."""
       return fips.Problem(
           prior=prior,
           prior_cov=fips.Covariance.block_diag(
               flux=fips.Exponential("x", 3.0).with_std(0.4),
               bias=fips.Independent().with_std(0.5),
           ),
           obs=obs,
           obs_cov=fips.Independent().with_std(0.2),
           forward=forward,
       )

.. ipython:: python

   flux = np.exp(rng.normal(np.log(0.15), 1.0, n_cells))
   truth = fips.Vector(np.r_[flux, -0.6, 0.4], prior.space)
   obs = fips.Vector(H @ truth.values + rng.normal(0.0, 0.2, n_obs), obs_space)
   problem = build(obs)

The analytic posterior and the bounded one, side by side. Two cells go below
zero in the analytic estimate; with the bounds they sit at zero, their
neighbors move a little, and the bias block, unbounded, keeps its negative
north bias:

.. ipython:: python

   analytic = fips.solve(problem)
   post = fips.solve(problem, "bounded", lower={"flux": 0.0})
   side = pd.DataFrame(
       {
           "truth": truth.to_series(),
           "analytic": analytic.estimate.to_series(),
           "bounded": post.estimate.to_series(),
           "at_bound": post.at_bound,
       }
   )
   near = post.at_bound.to_numpy()
   near = near | np.roll(near, 1) | np.roll(near, -1)  # on a bound, or next to one
   side[near].round(3)
   side.loc["bias"].round(3)

:attr:`~fips.Posterior.at_bound` says which elements sit on a bound at the
mode: the elements the bounds decide, where the observations decide the
others.


What the posterior says
-----------------------

**The spread comes from draws.** :attr:`~fips.Posterior.variances`,
:meth:`~fips.Posterior.cov` (of the state or of totals) and
:meth:`~fips.Posterior.uncertainty` are those of the draws the solve keeps:
``draws=2000`` of them by default, made at the first need from ``seed=0``, so a
solve is reproducible. They are Monte Carlo estimates (a standard deviation
from 2000 draws is good to about 2%). The total's uncertainty is smaller with
the bounds: they rule out the negative totals the Gaussian allowed.

.. ipython:: python

   W = fips.Operator(
       np.r_[np.ones(n_cells), 0.0, 0.0][None, :],
       fips.Space({"total": pd.Index(["flux"], name="what")}),
       prior.space,
   )
   pd.concat({"analytic": analytic.uncertainty(W), "bounded": post.uncertainty(W)}).round(3)

**The mode is not the mean.** The estimate is the mode; the draws are spread
on the side of the bound away from it, so their mean is above it for a lower
bound. A question about the distribution (a total's mean, the chance it is
below a value) is read off :meth:`~fips.Posterior.sample`, new draws of the
truncated posterior:

.. ipython:: python

   draws = post.sample(2000, rng=1, W=W)
   (W @ post.estimate).values.round(3), round(float(draws.iloc[0].mean()), 3)
   round(float((draws.iloc[0] < 3.0).mean()), 3)

**The gain, DOFS and the averaging kernel are those at the mode**: the
analytic posterior with the elements on a bound held there. An element on a
bound has no DOFS, since the bound, not the data, decides it. The chi-squared
and the misfits are those of the mode (the chi-squared tests' degrees of
freedom and p-values are then approximate):

.. ipython:: python

   post.dofs_per_element.to_series()[post.at_bound].round(3)
   round(analytic.dofs, 3), round(post.dofs, 3)

:meth:`~fips.Posterior.with_gamma` and :meth:`~fips.Posterior.with_data` find
the mode again on the kept factors, without solving again, and their draws are
made when first needed. :meth:`~fips.Posterior.recovery` and the data worth
refuse a bounded posterior, whose estimate is not linear in the data:
:meth:`~fips.Posterior.trials` measures it instead (below).


What it costs
-------------

The mode costs little more than the analytic solve: the active set needs the
columns of :math:`\hat{S}` at the elements on a bound, a product with the
factors for each, and a solve with :math:`\hat{S}` at those elements at each
of its steps. The draws cost more. Each draw follows a path that bounces off
every bound it reaches, each bounce a step of the size of the state, and a
path reaches about as many bounds as there are elements near one. A state of a
thousand elements, most of them near zero, took about 20 ms a draw (40 s for
2000), and one of three thousand about 0.1 s a draw, on one machine; a state
far from its bounds bounces little and costs little. The draws are made only
when a spread is asked for, and a smaller ``draws`` (``fips.solve(problem,
"bounded", lower=0.0, draws=500)``) costs proportionally less, with a larger
Monte Carlo error.

:meth:`~fips.Posterior.trials` asks each trial's posterior for its spread, so
a bounded posterior draws in every trial: keep ``draws`` small there.


In a world with a nonnegative truth
-----------------------------------

Does holding the fluxes at or above zero give a better answer? It depends on
what is asked, and :meth:`~fips.Posterior.trials` (see :doc:`synthetic`)
measures it: the same data, trial for trial (the same seed), solved with and
without the bounds, in a world whose truth is lognormal, most of it near zero,
which neither posterior assumes. The rows of the aggregation are each cell and
the total:

.. ipython:: python

   def world(rng):
       """A lognormal flux, a Gaussian bias, and the observations they give."""
       x = np.r_[np.exp(rng.normal(np.log(0.15), 1.0, n_cells)), rng.normal(0.0, 0.5, 2)]
       return fips.Vector(x, prior.space), fips.Vector(
           H @ x + rng.normal(0.0, 0.2, n_obs), obs_space
       )

.. ipython:: python

   rows = fips.Space({"cell": cells, "total": pd.Index(["flux"], name="what")})
   each_and_total = np.vstack([np.eye(n_cells, n_cells + 2), np.r_[np.ones(n_cells), 0.0, 0.0]])
   W_rows = fips.Operator(each_and_total, rows, prior.space)
   small = fips.solve(problem, "bounded", lower={"flux": 0.0}, draws=300)
   tables = {
       "analytic": analytic.trials(world, 100, rng=7, W=W_rows),
       "bounded": small.trials(world, 100, rng=7, W=W_rows),
   }

.. ipython:: python

   def scores(table):
       """The cells' RMSE, and the total's bias, spread, claimed spread and 95% coverage."""
       error = table["estimate"] - table["truth"]
       cell = error.xs("cell", level="block")
       total = table.xs("total", level="block")
       total_error = total["estimate"] - total["truth"]
       covered = total_error.abs() <= 1.96 * total["posterior_std"]
       return pd.Series(
           {
               "cell RMSE": float(np.sqrt((cell**2).mean())),
               "total bias": total_error.mean(),
               "total spread": total_error.std(),
               "claimed spread": total["posterior_std"].mean(),
               "total coverage": covered.mean(),
           }
       )

.. ipython:: python

   pd.DataFrame({name: scores(table) for name, table in tables.items()}).round(3)

With 100 trials the bias is known to about a tenth of the spread, and a
coverage near 0.9 to about 0.03. In this world the bounds make the map better
(a smaller cell RMSE) and move the total: up, by a few Monte Carlo standard
errors of the bias, with a smaller spread and a smaller claimed spread (the
two coverages are within their Monte Carlo error of each other). That is the
pattern to expect: **bounds mostly improve maps and can move totals**, and since the
estimate is the mode of a skewed posterior, a total read from it is not the
posterior's mean total (above, the draws'). Which matters depends on the
question. For a basin total, read the analytic posterior: the draws' total is
that of the truncated prior wherever the data see little, above an inventory
that is the mean, and :doc:`comparison` measures it far above in a Salt Lake
Valley-shaped problem. The world here is a toy: the trials of an inversion's
own problem, at its own size, say what its bounds do.

References
~~~~~~~~~~

Pakman, A., and Paninski, L. (2014). Exact Hamiltonian Monte Carlo for
truncated multivariate Gaussians. *Journal of Computational and Graphical
Statistics*, 23(2), 518-542. https://doi.org/10.1080/10618600.2013.788448

Miller, S. M., Michalak, A. M., and Levi, P. J. (2014). Atmospheric inverse
modeling with known physical bounds: an example from trace gas emissions.
*Geoscientific Model Development*, 7, 303-315.
https://doi.org/10.5194/gmd-7-303-2014
