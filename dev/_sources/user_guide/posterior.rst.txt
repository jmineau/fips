.. _reading-the-posterior:

Reading the posterior
=====================

:func:`fips.solve` solves a problem and returns a :class:`~fips.Posterior`,
the answer. By default it uses the ``"analytic"`` method, the exact Gaussian
posterior of a linear problem; :doc:`methods` covers the others. This page
reads an analytic posterior: the estimate and its uncertainty, aggregates of
the state, draws, how much the observations tell, and the checks of the
covariances. Every method's posterior has the same members; the
:ref:`members table <posterior-members>` says which each kind of answer gives.

.. contents:: On this page
   :local:
   :depth: 1


Solving
-------

A small problem: a weekly state on a 2 by 2 grid, with a prior correlated in
time and space, and 15 observations.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   state = pd.MultiIndex.from_product(
       [pd.date_range("2024-01-01", periods=4, freq="7D"), [40.5, 40.6], [-112.0, -111.9]],
       names=["time", "lat", "lon"],
   )
   prior_cov = fips.Kronecker(
       fips.Exponential("time", "14D"),
       fips.Exponential(["lat", "lon"], 10.0, metric="haversine"),
   ).with_std(0.5)

.. ipython:: python
   :suppress:

   pd.set_option("display.max_columns", None)  # show every column of a summary() table
   pd.set_option("display.width", 110)

.. ipython:: python

   rng = np.random.default_rng(1)
   obs_labels = pd.MultiIndex.from_product([["conc"], range(15)], names=["block", "obs_id"])
   forward = fips.Operator.from_frame(
       pd.DataFrame(rng.uniform(0, 1, (15, len(state))), index=obs_labels, columns=state),
       col_block="flux",
   )
   problem = fips.Problem(
       prior=fips.Vector.from_series(pd.Series(1.0, index=state), block="flux"),
       prior_cov=prior_cov,
       obs=fips.Vector.from_series(pd.Series(rng.normal(3.0, 1.0, 15), index=obs_labels)),
       obs_cov=fips.Independent().with_std(0.7),
       forward=forward,
   )
   post = fips.solve(problem)

``fips.solve(problem, method, **options)`` takes the method's name and its
options; :func:`fips.available_methods` lists the names, and an unknown method
or option raises with the names it knows. The analytic method's one option is
``gamma``, the regularization factor (below).


The estimate and its uncertainty
--------------------------------

:attr:`~fips.Posterior.estimate` is the posterior mean :math:`\hat{x}`, a
:class:`~fips.Vector` on the prior's labels. :attr:`~fips.Posterior.variances`
is the diagonal of the posterior covariance, and ``post.cov()`` the whole
:math:`\hat{S}`, formed only when asked for. :attr:`~fips.Posterior.modeled_obs`
is :math:`H \hat{x} + c`, and :meth:`~fips.Posterior.summary` puts the scalars
in one row (DOFS, chi-squared, :math:`R^2`, RMSE, ...):

.. ipython:: python

   post.estimate.to_series().head()
   post.variances.to_series().head()
   post.summary()


Aggregates of the state
-----------------------

**Most questions are about an aggregate**: a monthly total, a regional mean,
the difference between two seasons. An aggregation :math:`W` has one row per
aggregate (:func:`fips.aggregate.sum` and :func:`~fips.aggregate.mean` build
it from the labels). With it:

- ``W @ post.estimate`` is the aggregates' estimate;
- ``post.cov(W)`` is their posterior covariance, :math:`W \hat{S} W^T`;
- ``problem.prior_cov.aggregate(W)`` is their prior covariance,
  :math:`W S_0 W^T`;
- ``post.averaging_kernel(W)`` is their averaging kernel, :math:`W A W^+`,
  with :math:`W^+` the pseudo-inverse of :math:`W`;
- :meth:`~fips.Posterior.uncertainty` puts the prior and posterior standard
  deviations of each aggregate side by side, with the fraction the
  observations take off.

None of them forms an :math:`n \times n` matrix. Here, the total of each week:

.. ipython:: python

   W = fips.aggregate.sum(problem.prior.space, by="time")
   post.cov(W).to_frame().round(4)
   post.uncertainty(W).round(3)
   post.averaging_kernel(W).dense().diagonal().round(3)

**A combination of aggregates is a row too.** ``C @ W``, with ``C`` the
weights of the rows of ``W``, carries the covariance between the aggregates.
The change from one week to the next is not the root of the two weeks'
variances summed:

.. ipython:: python

   change = fips.Operator(
       np.array([[-1.0, 1.0, 0.0, 0.0]]),
       fips.Space({"change": pd.Index(["week 2 - week 1"], name="what")}),
       W.rows,
   )
   post.uncertainty(change @ W).round(3)

``dropna=True`` on :func:`fips.aggregate.sum` and :func:`~fips.aggregate.mean`
aggregates part of the state: labels a key cannot place are left out instead
of raising (a Series that maps only one region's elements gives that region's
total), and so are the blocks not named in ``blocks`` (a bias block next to
the fluxes).


.. _posterior-draws:

Draws
-----

**A covariance answers questions about spread; draws answer the rest.** The
chance that an aggregate is below a value, or which week has the largest
total, is a function of the whole distribution.
:meth:`~fips.Posterior.sample` draws from the posterior, one column per draw,
labeled as the estimate is; with ``W`` it draws the aggregates. Apply the
function to each draw and read the answer off the draws:

.. ipython:: python

   draws = post.sample(4000, rng=0, W=W)
   draws.shape
   (draws < 1.0).mean(axis=1).round(2)
   largest = draws.idxmax().map(lambda label: label[1])
   largest.value_counts(normalize=True).sort_index().round(2)

The analytic posterior's draws are exact: each perturbs the prior and the
observations by their errors and solves again. All the draws together cost
one product with each covariance's square root and one with the solve's
factors, never an :math:`n \times n` matrix. The draws are a
``state x draws`` array, so for a large state ask for them in blocks, or for
the aggregates you need.


How much the observations tell
------------------------------

:attr:`~fips.Posterior.dofs` (degrees of freedom for signal) is the trace of
the averaging kernel: the number of independent pieces of information the
observations give about the state. :attr:`~fips.Posterior.dofs_per_element`
is its diagonal, on the state's labels, so the share of a week, a region or a
block is a sum over its labels. :attr:`~fips.Posterior.information_content`
is the Shannon information content, :math:`-\tfrac{1}{2} \ln \lvert I - A
\rvert` in nats. DOFS counts a well-constrained direction as at most 1; the
information content keeps growing as the observations get more precise. Both
come from the solve's factors, without forming the :math:`n \times n` kernel:

.. ipython:: python

   post.dofs_per_element.to_series().groupby(level="time").sum().round(3)
   round(post.dofs, 3), round(post.information_content, 3)


Gamma, other data, and the L-curve
----------------------------------

**gamma scales the observation covariance**: the solve uses :math:`S_z /
\gamma`, so a gamma above 1 trusts the observations more and below 1 the
prior more. It is 1 by default.

**Another gamma or other observations need no new solve.** The analytic
factors depend neither on gamma nor on the observations' values, so
``post.with_gamma(g)`` is ``fips.solve(problem, gamma=g)`` for the cost of a
few products, and ``post.with_data(obs)`` is the posterior of other
observations at the same labels (a bootstrap, or :doc:`synthetic data
<synthetic>`). The rows of :meth:`~fips.Posterior.summary` stack:

.. ipython:: python

   pd.concat([post.with_gamma(g).summary() for g in (0.25, 1.0, 4.0)], ignore_index=True)

**An L-curve** plots the two terms of :attr:`~fips.Posterior.chi2` against
each other over a sweep of gamma. :attr:`~fips.Posterior.data_misfit` is how
far the modeled observations are from the observations, :math:`r^T S_z^{-1}
r`, on :math:`S_z` as given (so it compares across gammas).
:attr:`~fips.Posterior.state_misfit` is how far the estimate is from the
prior, :math:`(\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0)`. As gamma grows the
first falls and the second rises; the corner of the curve (Hansen, 1992) is
the gamma past which fitting the observations more closely costs much more
departure from the prior:

.. ipython:: python

   sweep = [post.with_gamma(g) for g in np.logspace(-2, 2, 9)]
   pd.DataFrame(
       [(p.options.gamma, p.data_misfit, p.state_misfit) for p in sweep],
       columns=["gamma", "data_misfit", "state_misfit"],
   ).round(3)

To choose gamma, or any other covariance setting, from the data instead, see
:doc:`settings`.


Checking the covariances
------------------------

**The posterior is only as right as the covariances.** Four checks:

- :attr:`~fips.Posterior.reduced_chi2` is near 1 when the covariances fit the
  observations.
- :meth:`~fips.Posterior.chi2_tests` makes that a test, with degrees of
  freedom and a p-value (Rodgers, 2000, chapter 12), and tests three more
  vectors against their own covariances. A small p-value says the vector is
  larger than its covariance allows; one near 1, smaller.
- :meth:`~fips.Posterior.desroziers` compares the observation error variances
  the residuals diagnose with those the solve used, per observation or
  averaged over groups.
- :meth:`~fips.Posterior.leverage` says which observations weigh most on a
  state element.

The four rows of ``chi2_tests`` are the innovation :math:`z - H x_0 - c` (the
test of :attr:`~fips.Posterior.chi2`, with :math:`m` degrees of freedom), the
residual :math:`z - H \hat{x} - c` (also :math:`m`), the modeled increment
:math:`H (\hat{x} - x_0)`, and the increment :math:`\hat{x} - x_0` (each with
as many degrees of freedom as there are directions of the state the
observations constrain). When the estimate is linear in the observations, as
it is here, the residual's test gives the innovation's numbers and the
modeled increment's gives the increment's; they differ for a method whose
estimate is not (``"bounded"``, ``"gauss_newton"``). Here the state has more
elements than the 15 observations, which constrain 15 directions of it, so all
four rows are the same:

.. ipython:: python

   post.chi2_tests().round(3)

:attr:`~fips.Posterior.log_evidence`, the log probability of the observations
given the covariances, is the measure to maximize when fitting them:
:doc:`settings`.


What each group of observations adds
------------------------------------

**Data worth** asks how much an aggregate's uncertainty depends on each group
of observations (a site, a platform, a month). :meth:`~fips.Posterior.data_worth`
groups the observations by any of their levels, as
:meth:`~fips.Posterior.desroziers` does, and gives for each group:

- ``std_with`` and ``std_without``: the aggregate's standard deviation with
  all the observations, and without this group;
- ``reduction``: the fraction this group takes off, on top of the others;
- ``dofs``: its share of DOFS;
- ``std_alone`` and ``reduction_alone``: the standard deviation with this
  group's observations only, and the fraction that takes off the prior.

The two readings differ when groups see the same part of the state: a site
that adds little on top of the others may have taught a lot had it been the
only one. They agree only when there is one group. The *alone* columns use
the group's own block of the observation covariance, so they leave out its
correlation with other groups' errors. Each is a rank update of
:math:`\hat{S}` or :math:`S_0`, exact, with no problem solved again.

A problem with two sites, and the worth of each to the total:

.. ipython:: python

   hours = pd.date_range("2024-01-01", periods=48, freq="h")
   labels = pd.MultiIndex.from_product([["A", "B"], hours], names=["site", "obs_time"])
   ch4 = fips.Vector.from_series(pd.Series(rng.normal(2.0, 0.1, 96), index=labels), block="ch4")
   flux = fips.Vector.from_series(pd.Series(1.0, index=pd.Index(["c0", "c1", "c2"], name="cell")), block="flux")
   hourly = fips.Problem(
       prior=flux,
       prior_cov=fips.Independent().with_std(0.5),
       obs=ch4,
       obs_cov=fips.Independent().with_std(0.05),
       forward=fips.Operator(rng.uniform(0, 0.1, (96, 3)), ch4.space, flux.space),
   )

.. ipython:: python

   hourly_post = fips.solve(hourly)
   total = fips.aggregate.sum(hourly.prior.space, by=pd.Series("all", index=hourly.prior.space.select("flux"), name="region"))
   hourly_post.data_worth(by="site", W=total).round(3)

**Candidate observations** (a site or platform you might add) are their rows
of the forward operator and the covariance of their errors, with no values.
:meth:`~fips.Posterior.candidate_worth` adds each group on its own and gives
the same columns except the *alone* ones, so "which candidate helps the total
most" is one call:

.. ipython:: python

   new_sites = fips.Space({"ch4": pd.MultiIndex.from_product([["C", "D"], hours], names=["site", "obs_time"])})
   H_new = fips.Operator(rng.uniform(0, 0.1, (96, 3)), new_sites, flux.space)
   hourly_post.candidate_worth(H_new, fips.Independent().with_std(0.05), by="site", W=total).round(3)

The data worth needs an estimate linear in the observations: the analytic
method gives it, and the members table says which others do.


Plotting
--------

fips draws one plot, :func:`~fips.visualization.plot_comparison` (the ``viz``
extra): several vectors or Series on one axes, with a band of one standard
deviation from each covariance, and an optional truth. Draw maps and anything
else from :meth:`~fips.Vector.to_series` and :meth:`~fips.Vector.to_xarray`
with pandas, xarray or matplotlib:

.. ipython:: python

   from fips.visualization import plot_comparison

   weekly = fips.aggregate.mean(problem.prior.space, by="time")

   @savefig usage_comparison.png width=5in
   fig, ax = plot_comparison(
       (weekly @ problem.prior).to_series().droplevel("block").rename("prior"),
       (weekly @ post.estimate).to_series().droplevel("block").rename("posterior"),
       errors=[problem.prior_cov.aggregate(weekly), post.cov(weekly)],
   )


.. _repeated-inversions:

Repeated inversions, logging and saving
---------------------------------------

**An inversion run many times** (one a month, a sweep over settings) is a
function in your own code that builds the :class:`~fips.Problem` from your
settings, then :func:`fips.solve`:

.. code-block:: python

   import pandas as pd

   import fips

   def build_problem(config) -> fips.Problem:
       """Load the observations and the prior, build H and the covariances."""
       ...

   rows = [fips.solve(build_problem(config)).summary() for config in configs]
   table = pd.concat(rows, ignore_index=True)

**fips prints nothing, and its warnings are silent until you configure
logging.** It logs through the standard :mod:`logging` module, under the
``fips`` logger (the solve under ``fips.methods``), with a
:class:`~logging.NullHandler`, as a library should: nothing reaches the
screen until your program configures logging. Several methods warn about
things that change the answer (a covariance close to singular, a prior wide
against its bounds, draws that mix slowly, iterations that converge slowly),
so configure it before you solve:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.WARNING)  # logging.INFO shows each solve's steps too

**A problem and a posterior pickle** with the standard library
(``pickle.dump(post, file)``). A posterior keeps its problem, its options and
what the method kept from the solve, and rebuilds every output when it is
read.


How the analytic solve works
----------------------------

The analytic method factors one matrix, in one of two forms that give the
same posterior:

- **The whitened form** works in the observations' size. It whitens with the
  Cholesky factor of :math:`S_z` (so :math:`S_z` must be positive definite)
  and factors the :math:`m \times m` matrix :math:`H S_0 H^T` in that metric.
  It computes :math:`H S_0` in the forms of :math:`H` and :math:`S_0`
  (through the factors of a Kronecker covariance), and the estimate, the
  variances, DOFS and chi-squared from the factors, forming nothing
  :math:`n \times n`.
- **The information form** works in the state's size. It factors an
  :math:`n \times n` matrix built from :math:`H`, a square root of
  :math:`S_0` and :math:`S_z^{-1}`, and forms no :math:`m \times m` matrix of
  its own.

fips takes the information form when there are fewer state elements than
observations and :math:`S_z` is dense, diagonal, sparse, block diagonal or
Kronecker; otherwise the whitened form. In both, ``post.cov()``, the gain and
the averaging kernel are formed only when read, once each.

:func:`fips.solve` factors that matrix by Cholesky at its ``gamma``, the
cheapest way to solve at one gamma. The first ``post.with_gamma`` takes its
eigendecomposition instead, which serves every gamma, and keeps it: a sweep
over gamma pays for one eigendecomposition.

References
----------

Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of the
L-curve. *SIAM Review*, 34(4), 561-580.

Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and
Practice*. World Scientific.
