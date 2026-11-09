.. _ensemble:

A model with no Jacobian: the ensemble method
=============================================

Some forward models can only be run: a TM5 or WRF configuration has no
:math:`H` matrix and no Jacobian anyone can write by hand. The rule: **draw
members from the prior with** :meth:`problem.ensemble <fips.Problem.ensemble>`,
**run the model on each, and give the runs to** ``fips.solve(problem,
"ensemble", members=members, predicted=modeled)``. It updates the members with
the observations, as CarbonTracker does (a deterministic square-root ensemble
Kalman filter, EnSRF, applied to all the observations at once), and reports
the posterior of the updated members: their mean, their spread, their gain.
Its answer approaches the analytic posterior as the members grow, with an
error that falls as :math:`1/\sqrt{N}`.

**When not to use it**: with an explicit :math:`H` (a STILT footprint is one),
``fips.solve(problem)`` gives the posterior exactly, at the cost of one
factorization; the ensemble only approximates it. With a model whose Jacobian
you can write, ``"gauss_newton"`` finds the mode and draws the posterior exactly
(:doc:`nonlinear`).

.. contents:: On this page
   :local:
   :depth: 1


A model that can only be run
----------------------------

A line of 120 cells, one a kilometer, and 40 observations along it, each of
which sees the cells upwind, the nearest most. ``transport`` stands in for the
transport model: it takes the fluxes of one member and returns the
observations they make. Here it is a line of numpy; in a real inversion it is
a job on a cluster, and the members go to it as files.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   n, m = 120, 40
   cell_x = np.arange(n, dtype=float)  # km
   site_x = np.linspace(2.0, n - 3.0, m)
   upwind = site_x[:, None] - cell_x[None, :]
   footprint = np.where(upwind >= 0, np.exp(-upwind / 3.0), 0.0)

   def transport(flux):
       """Stand-in for a transport model run: the observations a member's fluxes make."""
       return footprint @ flux.to_numpy()

The problem still needs its forward map's labels, so the model is a
:class:`~fips.ForwardModel` with ``__call__`` alone: the ensemble method needs
no Jacobian. When given the runs, the solve never calls it; what does
(:meth:`~fips.Problem.simulate`, :attr:`~fips.Posterior.modeled_obs`) runs it
once. A model that cannot run from Python at all can raise in ``__call__``:
the two-stage solve still works.

.. ipython:: python

   cells = fips.Space({"flux": pd.RangeIndex(n, name="cell")})
   observed = fips.Space({"ch4": pd.RangeIndex(m, name="obs_id")})

   class Transport(fips.ForwardModel):
       def __call__(self, x):
           return fips.Vector(transport(x.to_series()), observed)

   model = Transport(observed, cells)

A prior of 1 in every cell with a standard deviation of 0.5, correlated over
2 km (short: the footprints are longer), and observations of a truth drawn
from it:

.. ipython:: python

   def build(obs, forward=model):
       return fips.Problem(
           prior=fips.Vector(np.ones(n), cells),
           prior_cov=fips.Exponential("cell", 2.0).with_std(0.5),
           obs=obs,
           obs_cov=fips.Independent().with_std(0.2),
           forward=forward,
       )

   truth, obs = build(fips.Vector(np.zeros(m), observed)).simulate(rng=1)
   problem = build(obs)


The two stages
--------------

Stage one, in fips: draw the members. Each is a column, labeled as the prior
is; their mean is the prior exactly.

.. ipython:: python

   members = problem.ensemble(100, rng=0)
   members.shape

Stage two, outside fips: run the model on each member (``members[j]`` is one
member's fluxes; write it out with ``to_csv`` or ``to_xarray``), and put the
runs in a DataFrame, one row per observation, labeled as ``problem.obs.index``,
one column per member, labeled as the members are (in any order). The runs are
the model's output, :math:`F(x_j)`, without the problem's offset.

.. ipython:: python

   modeled = pd.DataFrame(
       {j: transport(members[j]) for j in members.columns},
       index=problem.obs.index,
   )

Stage three, in fips: the update.

.. ipython:: python

   post = fips.solve(problem, "ensemble", members=members, predicted=modeled)
   regions = fips.Operator(
       np.kron(np.eye(4), np.ones((1, n // 4))),
       fips.Space({"total": pd.Index(["W", "WC", "EC", "E"], name="region")}),
       cells,
   )
   post.uncertainty(regions).round(2)

A problem whose forward runs in Python can skip the middle:
``fips.solve(problem, "ensemble", members=100, seed=0)`` draws the members
with that seed and runs the forward (a :class:`~fips.ForwardModel`'s
``__call__``, or an :class:`~fips.Operator`'s product) on each, and gives the
same posterior.

What the posterior is, for the ensemble:

- :attr:`~fips.Posterior.estimate` (and :attr:`~fips.Posterior.mean`) is the
  mean of the updated members; :attr:`~fips.Posterior.variances`,
  :meth:`~fips.Posterior.cov` and :attr:`~fips.Posterior.kalman_gain` are
  those of the members, and :meth:`~fips.Posterior.sample` returns the
  updated members themselves (it raises past their number);
- :attr:`~fips.Posterior.dofs`, :attr:`~fips.Posterior.information_content`,
  :attr:`~fips.Posterior.chi2` and :meth:`~fips.Posterior.chi2_tests` are the
  analytic method's formulas with the members' covariance of the modeled
  observations in place of :math:`H S_0 H^T`;
- :meth:`~fips.Posterior.with_data` and :meth:`~fips.Posterior.with_gamma`
  reuse the runs, so :meth:`~fips.Posterior.trials` runs the model only to make
  each trial's data;
- :meth:`~fips.Posterior.recovery` and :attr:`~fips.Posterior.log_evidence`
  are refused (the evidence of a sample covariance is not the evidence), and
  what needs :math:`H` (:meth:`~fips.Posterior.averaging_kernel`,
  :attr:`~fips.Posterior.dofs_per_element`) raises for a model.


How close it is
---------------

This stand-in is cheap and linear, so its :math:`H` can be built from 120
runs, one a cell, and the exact posterior compared with the ensemble's. A real
transport model is not cheap, which is why it is run per member and not per
cell.

.. ipython:: python

   H = fips.Operator(
       np.column_stack([transport(pd.Series(np.eye(n)[j])) for j in range(n)]),
       observed,
       cells,
   )
   exact = fips.solve(build(obs, forward=H))
   exact.uncertainty(regions).round(2)

The error of the regions' totals over 16 ensembles of each size, the mean's
in posterior standard deviations and the standard deviation's relative to the
exact one:

.. ipython:: python

   exact_totals = (regions @ exact.estimate).values
   exact_std = np.sqrt(exact.cov(regions).variances.values)

   def error(N, seed):
       e = fips.solve(problem, "ensemble", members=N, seed=seed)
       mean = ((regions @ e.estimate).values - exact_totals) / exact_std
       std = np.sqrt(e.cov(regions).variances.values) / exact_std - 1.0
       return np.sqrt(np.mean(mean**2)), np.sqrt(np.mean(std**2))

   sizes = [25, 50, 100, 200, 400, 800, 1600]
   curve = pd.DataFrame(
       [np.sqrt(np.mean([np.square(error(N, s)) for s in range(16)], axis=0)) for N in sizes],
       index=pd.Index(sizes, name="members"),
       columns=["mean", "std"],
   )
   curve.round(3)
   np.polyfit(np.log(sizes), np.log(curve), 1)[0].round(2)  # the slopes

.. ipython:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(5, 3.5))
   ax.loglog(curve.index, curve["mean"], "o-", label="mean (posterior sd)")
   ax.loglog(curve.index, curve["std"], "s-", label="std (relative)")
   ax.loglog(sizes, curve["mean"].iloc[0] * np.sqrt(sizes[0] / np.array(sizes)), "k:", label=r"$1/\sqrt{N}$")
   ax.set_xlabel("members")
   ax.set_ylabel("error of the region totals")
   @savefig ensemble_convergence.png width=5in
   ax.legend()

The mean's error falls as :math:`1/\sqrt{N}` (a slope of about -0.6 here,
a little steeper at the smallest ensembles, where the gain's error has a part
that falls as :math:`1/N`): four times the members for half the error. The
standard deviation's falls faster over these sizes: at small ensembles it is
mostly a bias, a spread too small by a part that falls as :math:`1/N` (the
members' sampling noise fits the data and is taken for information), and the
random part, which falls as :math:`1/\sqrt{N}`, takes over only at large
ensembles. At 100 members the totals are within about four tenths of their
posterior standard deviation of the exact ones, and their standard deviations
about 10% small. (The tests hold the slopes of the mean and of the whole
covariance to about -1/2 from 25 to 3200 members on an explicit-:math:`H`
problem.)


Localization
------------

An ensemble of :math:`N` members has :math:`N - 1` directions, far fewer than
the cells, so its covariance between a cell and an observation far away is
noise, not zero. The update follows that noise: it moves cells the observation
cannot see, and shrinks their spread, so the posterior claims more certainty
than it has. Localization multiplies the ensemble's covariances by weights that
fall to zero with distance (Houtekamer and Mitchell, 2001). fips computes no
distances: the weights are yours, an :class:`~fips.Operator` of the state by
the observations (``localization=``), and one of the observations by
themselves (``obs_localization=``). Here, Gaspari and Cohn's (1999)
fifth-order function, which falls from 1 at no distance to 0 at twice its
half-width:

.. ipython:: python

   def gaspari_cohn(distance, half_width):
       """Gaspari and Cohn's (1999) function of a distance, zero beyond twice the half-width."""
       r = np.abs(distance) / half_width
       w = np.zeros_like(r)
       a, b = r <= 1.0, (r > 1.0) & (r < 2.0)
       w[a] = -r[a]**5 / 4 + r[a]**4 / 2 + 5 * r[a]**3 / 8 - 5 * r[a]**2 / 3 + 1
       w[b] = r[b]**5 / 12 - r[b]**4 / 2 + 5 * r[b]**3 / 8 + 5 * r[b]**2 / 3 - 5 * r[b] + 4 - 2 / (3 * r[b])
       return w

   def localized(N, half_width):
       return fips.solve(
           problem, "ensemble", members=N, seed=3,
           localization=fips.Operator(gaspari_cohn(cell_x[:, None] - site_x[None, :], half_width), cells, observed),
           obs_localization=fips.Operator(gaspari_cohn(site_x[:, None] - site_x[None, :], half_width), observed, observed),
       )

The test is the one of :doc:`synthetic`: 200 sets of data from the problem's
own world, each solved with the same members (``with_data``), and how often
each region's 95% interval holds its true total. The analytic posterior's
holds it 95% of the time; the ensemble's, with 40 members and no
localization, much less, since it claims too small a spread:

.. ipython:: python

   def score(post):
       table = post.trials(n=200, rng=5, W=regions)
       error = table["estimate"] - table["truth"]
       return pd.Series({
           "coverage": (error.abs() <= 1.96 * table["posterior_std"]).mean(),
           "rmse": np.sqrt((error**2).mean()),
           "claimed_std": table["posterior_std"].mean(),
       })

   pd.DataFrame({
       "analytic": score(exact),
       "40 members": score(fips.solve(problem, "ensemble", members=40, seed=3)),
       "40, localized 5 km": score(localized(40, 5.0)),
       "40, localized 10 km": score(localized(40, 10.0)),
       "40, localized 20 km": score(localized(40, 20.0)),
   }).round(2)

Localized, the 40 members' intervals hold the truth about as often as they
claim to, and their totals are closer to the truth. The half-width is a
choice: too short cuts real correlations (the footprints here reach about
10 km upwind, and the prior's correlation 2 km past that), so the update moves
less than it should, and the error and the spread it claims are both larger (5
km); longer comes closer to the analytic posterior (20 km, the best of the
three here), until the noise is back (no localization at all is an infinite
half-width). Choose it with ``trials``, as here, in a world shaped like
yours. With localization the
updated members' covariance is not the localized Kalman posterior (an ensemble
of :math:`N` members cannot hold a covariance of more than :math:`N - 1`
directions), and weights of one everywhere give the unlocalized update
exactly.


The update, and what it costs
-----------------------------

The prior members' deviations :math:`A` and their runs' :math:`B` (each
divided by :math:`\sqrt{N - 1}`) give the ensemble's covariances
:math:`P_{xy} = A B^T` and :math:`P_{yy} = B B^T`, each localized by its
weights. The mean moves by the ensemble gain, :math:`K = P_{xy} (P_{yy} +
S_z/\gamma)^{-1}`, and the deviations by the reduced gain of Whitaker and
Hamill's (2002) batch square-root filter, computed with the observations
whitened by :math:`S_z`: their covariance after the update is the Kalman
posterior of the ensemble's own covariance, exactly. Without localization the
same update is the ensemble transform of the ETKF, :math:`A^a = A (I + S^T
S)^{-1/2}`, which fips computes from the thin SVD of the whitened runs
(:math:`O(m N \min(m, N))`); with localization it eigendecomposes the
:math:`m \times m` localized :math:`P_{yy}` and forms the :math:`n \times m`
localized :math:`P_{xy}`. A non-diagonal :math:`S_z` is factored by Cholesky,
dense. None of it depends on :math:`\gamma` or the observations' values, so
:meth:`~fips.Posterior.with_gamma` and :meth:`~fips.Posterior.with_data` cost
a product. The runs are the cost: :math:`N` of them, once.

What stays outside: assimilating window after window (CarbonTracker's lag
loop) is a loop of ``fips.solve(..., "ensemble")`` calls, each starting from
the last one's updated members (``post.sample(N)``, with the next window's
prior added); no outlier rejection is made.
