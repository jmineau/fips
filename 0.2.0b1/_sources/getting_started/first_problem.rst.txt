A First Problem
===============

This page solves one small problem from start to finish, meeting fips's
objects in the order you use them: the labels and values of a vector, the
forward operator, the covariances, the problem, and its posterior. The words
(state, observation, forward operator, prior, ...) are defined once in
:doc:`terminology`.

A first problem
---------------

Six rays cross a line of four cells, and each ray's travel time is measured. A
ray's time is the sum, over the cells it crosses, of its length in the cell
times the cell's slowness (the inverse of a speed). The state is the slowness
of the four cells; the observations are the six times; the forward operator
holds the lengths. We know roughly what the slowness is (the prior: 0.25 s/km
everywhere, give or take 0.05) and how well the times are measured (to 0.01 s).

Labels and vectors
------------------

fips keeps the labels of your data beside the numbers and matches everything
by label, never by position. The labels of a state or of the observations are
a :class:`~fips.Space`: named blocks, each labeled by its own pandas index. A
:class:`~fips.Vector` is values on a space. A pandas Series you already have
becomes a vector of one block with :meth:`Vector.from_series
<fips.Vector.from_series>`:

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   cells = pd.Index([0.5, 1.5, 2.5, 3.5], name="x")  # where each cell is, km
   prior = fips.Vector.from_series(pd.Series(0.25, index=cells, name="prior"), block="slowness")
   prior.space

   rays = pd.Index([1, 2, 3, 4, 5, 6], name="ray")
   times = pd.Series([0.53, 0.76, 1.02, 0.25, 0.48, 0.80], index=rays)
   obs = fips.Vector.from_series(times, block="time")
   obs.to_series()

A vector is read-only, and :meth:`~fips.Vector.to_series` gives it back to
pandas, with a ``block`` level first.

The forward operator
--------------------

The forward operator maps a state to the observations it would produce. In
fips it is an :class:`~fips.Operator`: a matrix with a space on each axis, the
observations' on its rows and the state's on its columns. A labeled DataFrame
becomes one with :meth:`Operator.from_frame <fips.Operator.from_frame>`, which
names the block of each axis:

.. ipython:: python

   lengths = pd.DataFrame(
       [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
       index=rays,
       columns=cells,
       dtype=float,
   )
   forward = fips.Operator.from_frame(lengths, row_block="time", col_block="slowness")
   (forward @ prior).to_series()

``forward @ prior`` is the travel times the prior predicts, a vector on the
observations' space. A product checks the labels it meets on: the columns of
``forward`` and the labels of ``prior`` must be the same, in any order.

Covariances
-----------

A covariance says how far a vector may be off, and how its errors go
together. Independent errors with one standard deviation are a diagonal, which
:class:`~fips.Independent` writes down. Errors that are alike for nearby
cells are a correlation model times standard deviations: here
:class:`~fips.Exponential` correlates two cells by their distance along ``x``,
over 1 km. ``with_std`` turns a correlation model into a covariance model:

.. ipython:: python

   obs_cov = fips.Independent().with_std(0.01)
   prior_cov = fips.Exponential("x", 1.0).with_std(0.05)
   prior_cov.build(prior.space).dense().round(5)

A covariance model is a covariance waiting for its labels: ``build`` puts it on
a space and gives a :class:`~fips.Covariance`. You rarely call it yourself,
since a problem builds its covariance models on the prior's and the
observations' spaces. :ref:`Covariance models <covariance-models>` in the User Guide has the other
kernels, Kronecker products and correlations within groups.

The problem
-----------

A :class:`~fips.Problem` puts the inputs together and checks their labels once:
the forward operator, the covariances and the offset must have exactly the
labels of the observations or of the prior, in any order:

.. ipython:: python

   problem = fips.Problem(
       prior=prior, prior_cov=prior_cov, obs=obs, obs_cov=obs_cov, forward=forward
   )
   problem.predict(problem.prior).to_series()

:meth:`~fips.Problem.predict` is :math:`H x + c` for any state. A label on one
side only is an error that names it, so a typo or a missing row cannot be
filled in silently:

.. ipython:: python

   five_rays = fips.Operator.from_frame(lengths.iloc[:5], row_block="time", col_block="slowness")
   try:
       fips.Problem(prior, prior_cov, obs, obs_cov, five_rays)
   except ValueError as error:
       print(error)


Solving
-------

:func:`fips.solve` solves the problem and returns its
:class:`~fips.Posterior`; the problem itself is not changed. The posterior has
the estimate, on the prior's labels, and its uncertainty:

.. ipython:: python

   post = fips.solve(problem)
   post.estimate.to_series().round(3)
   post.variances.to_series().pow(0.5).round(4)
   post.cov().to_frame().round(6)

Nothing larger than the observations is formed until you ask for it: the
estimate and the variances come from a factorization of an :math:`m \times m`
matrix, and ``post.cov()`` forms the :math:`n \times n` covariance only when it
is read. The diagnostics of the solve are attributes, and
:meth:`~fips.Posterior.summary` gives them as one row:

.. ipython:: python

   post.dofs
   post.reduced_chi2
   post.summary()

The method is ``"analytic"`` by default, and its one option is the
regularization factor ``gamma`` (the solve uses :math:`S_z / \gamma`).
:meth:`~fips.Posterior.with_gamma` gives another gamma's posterior from the same
factorization, without solving again:

.. ipython:: python

   fips.solve(problem, gamma=4.0).options
   post.with_gamma(4.0).dofs

A second block
--------------

The rays were timed by two instruments whose clocks may each be off by a
delay. The delays are unknown too, so they are a second block of the state.
:meth:`Vector.from_blocks <fips.Vector.from_blocks>` puts blocks together, each
with its own labels:

.. ipython:: python

   instruments = pd.Index(["east", "west"], name="instrument")
   prior2 = fips.Vector.from_blocks(
       {"slowness": pd.Series(0.25, index=cells), "delay": pd.Series(0.0, index=instruments)}
   )
   prior2.space

The forward operator is built from its blocks, each named by the observation
block it maps to and the state block it maps from, and placed on the spaces by
its labels (:meth:`Operator.from_blocks <fips.Operator.from_blocks>`); a pair
of blocks left out is zero. Rays 1 to 3 were timed east, 4 to 6 west:

.. ipython:: python

   timer = pd.DataFrame(
       [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
       index=rays,
       columns=instruments,
       dtype=float,
   )
   forward2 = fips.Operator.from_blocks(
       {("time", "slowness"): lengths, ("time", "delay"): timer},
       rows=obs.space,
       cols=prior2.space,
   )

The prior covariance has a part for each block, independent of each other:
:meth:`Covariance.block_diag <fips.Covariance.block_diag>` takes them by block
name:

.. ipython:: python

   prior_cov2 = fips.Covariance.block_diag(
       slowness=fips.Exponential("x", 1.0).with_std(0.05),
       delay=fips.Independent().with_std(0.02),
   )
   post2 = fips.solve(fips.Problem(prior2, prior_cov2, obs, obs_cov, forward2))
   post2.estimate["delay"].to_series().round(3)

``post2.estimate["delay"]`` reads one block by name, as a vector of its own.

The posterior of a total
------------------------

Often the question is about an aggregate of the state: the mean slowness of
each half of the line, a regional total, a monthly mean. An aggregation
:math:`W` (:mod:`fips.aggregate`) groups the labels of a space;
``post.cov(W)`` is the posterior covariance of the aggregates, computed without
the full covariance:

.. ipython:: python

   halves = pd.Series(["near", "near", "far", "far"], index=cells, name="half")
   W = fips.aggregate.mean(prior2.space, by=halves, blocks="slowness")
   (W @ post2.estimate).to_series().round(3)
   post2.cov(W)["slowness", "slowness"].to_frame().round(6)

Plotting
--------

fips draws one plot, :func:`~fips.visualization.plot_comparison` (it needs the
``viz`` extra): several vectors on one axes, with a band of one standard
deviation from each covariance. Anything else is drawn from
:meth:`~fips.Vector.to_series` or :meth:`~fips.Vector.to_xarray` with pandas or
xarray.

.. ipython:: python

   from fips.visualization import plot_comparison

   @savefig getting_started_comparison.png width=5in
   fig, ax = plot_comparison(
       problem.prior,
       post.estimate,
       errors=[problem.prior_cov, post.cov()],
   )

What's next
-----------

- :doc:`/user_guide/index`, the User Guide: each object in more depth, covariance models,
  sparse matrices, offsets, aggregated observations, part of the state held at
  its prior, and repeated inversions.
- :doc:`/user_guide/examples/index`: two worked problems, gravity and a regional flux
  inversion.
- :doc:`/reference/index`: every class, function and attribute.
