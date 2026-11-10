User Guide
==========

This guide takes fips's objects one at a time, in the order a problem uses
them, and says what each is for before how to call it. :doc:`/getting_started/first_problem`
solves one problem from start to finish; :doc:`/getting_started/terminology` defines the words;
:doc:`synthetic` tests an inversion on data whose answer is known; :doc:`settings`
fits the covariance settings by maximizing the evidence; :doc:`bounds`
holds the state to bounds (a flux that cannot be negative); :doc:`nonlinear`
solves a problem whose forward is a model that is not linear; :doc:`positive`
keeps a flux positive by solving for its logarithm or its softplus; and
:doc:`comparison` runs the analytic, bounded, lognormal and softplus methods
on one problem, to choose among them.

.. contents:: On this page
   :local:
   :depth: 1


Spaces and vectors
------------------

Every number in fips has a label, and fips matches numbers by their labels,
never by their position. A :class:`~fips.Space` is the labels of a state, of
the observations, or of one axis of a matrix: an ordered set of named
**blocks**, each labeled by its own pandas index. A state can have a ``flux``
block labeled by ``(time, lat, lon)`` and a ``bias`` block labeled by
``site``; the observations a ``station`` block and a ``satellite`` block.

A :class:`~fips.Vector` is values on a space: a prior, the observations, an
estimate. It comes from a Series (one block) or from several (one per block):

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   cells = pd.Index([1, 2, 3], name="cell")
   sites = pd.Index(["north", "south"], name="site")
   prior = fips.Vector.from_blocks(
       {"flux": pd.Series([1.0, 2.0, 0.5], index=cells), "bias": pd.Series(0.0, index=sites)},
       name="prior",
   )
   prior.to_series()
   prior["flux"].to_series()

.. ipython:: python
   :suppress:

   pd.set_option("display.max_columns", None)  # show every column of a summary() table
   pd.set_option("display.width", 110)

Together, the blocks are one :class:`pandas.MultiIndex` with the ``block``
level first (``prior.space.index``). A level that one block has and another
lacks is missing (NaN) on the other's rows, and :class:`~fips.Space` is the one
place that adds and removes those levels: ``prior["flux"]`` comes back labeled
as it was given, integer cells and all. ``prior.space.select("flux")`` gives a
block's labels, and :meth:`~fips.Vector.to_xarray` a DataArray on the levels of
one block (a Dataset for several).

Labels are kept as they are given: fips does not convert strings to numbers or
round floats. Duplicate, missing (NaN) and unnamed labels raise a
:class:`ValueError` that names them. Labels that should match only after
rounding are rounded on purpose, with :meth:`~fips.Vector.round_labels` (also
on a :class:`~fips.Space` and an :class:`~fips.Operator`).

Vectors do arithmetic by label. ``v + w``, ``v - w``, ``v * w`` and ``v / w``
put ``w`` in the order of ``v``'s labels first (a missing or extra label
raises, naming it), and a number or a 1D array combines by position;
``np.exp(v)`` and ``np.maximum(v, w)`` return vectors too. A reduction such as
``np.sum(v)`` is not defined: use ``v.values``.

A vector is read-only: a new value is a new vector. :meth:`~fips.Vector.align`
puts a vector in the order of a space with the same labels, and
:meth:`~fips.Vector.combine_first` fills the labels one vector lacks from
another (the estimate of part of the state, completed by the prior).


Operators
---------

An :class:`~fips.Operator` is a matrix with a space on each axis: the forward
operator (observations by state), an aggregation, a gain. Its matrix is kept
in the cheapest form that is exact (a dense array, a SciPy sparse matrix, a
diagonal, Kronecker factors; see :doc:`/reference/linalg`), and the labels are
beside it. There are three ways in:

- ``Operator(matrix, rows, cols)`` labels a matrix you have, by position: its
  rows are ``rows``'s labels in order, its columns ``cols``'s. A SciPy sparse
  matrix stays sparse.
- :meth:`Operator.from_frame <fips.Operator.from_frame>` reads a labeled
  DataFrame, dense or with pandas-sparse columns.
- :meth:`Operator.from_blocks <fips.Operator.from_blocks>` assembles one from
  its blocks, each named by its row block and its column block: a DataFrame,
  or a tuple ``(matrix, row_labels, col_labels)``. Given the spaces
  (``rows=``, ``cols=``), it places each block by its labels: a label the block
  has no entry for is zero, as a pair of blocks left out is, and a label the
  space does not have raises.

.. ipython:: python

   obs_id = pd.Index(["o1", "o2", "o3"], name="obs_id")
   obs = fips.Vector.from_series(pd.Series([2.0, 3.5, 1.0], index=obs_id), block="co2")
   H = fips.Operator.from_blocks(
       {
           ("co2", "flux"): pd.DataFrame(np.eye(3), index=obs_id, columns=cells),
           ("co2", "bias"): pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], index=obs_id, columns=sites),
       },
       rows=obs.space,
       cols=prior.space,
   )
   H.to_frame()
   (H @ prior).to_series()

``H @ vector`` is a vector, ``H @ other`` an operator, and ``+`` and ``-`` add
operators; each first matches the labels they meet on, in any order, and a
label on one side only raises. ``H["co2", "flux"]`` is one block, ``H.T`` the
transpose, and :meth:`~fips.Operator.to_frame` the dense DataFrame.

**Sparse matrices.** A forward operator from atmospheric transport is mostly
zeros. An operator built
from a :mod:`scipy.sparse` matrix (or from a DataFrame with pandas-sparse
columns) keeps it sparse, and so do the products and sums that stay mostly
zeros: a diagonal or sparse matrix times a sparse one is sparse, and a product
is dense once more than a quarter of its entries are filled, or when either
side is dense. The forward operator, the prior covariance and the observation
covariance can each be dense or sparse:

.. ipython:: python

   import scipy.sparse as sp

   J = sp.random(3, 3, density=0.4, random_state=0, format="csr")
   sparse_H = fips.Operator(J, obs.space, fips.Space({"flux": cells}))
   sparse_H.data

A large product of a sparse matrix and a dense one (:math:`H S_0 H^T` with a
dense :math:`S_0` has one) runs on threads, each multiplying a block of the
sparse matrix's rows, and gives the same numbers as one thread, bit for bit. It
uses the CPUs the process may run on (a Slurm job's ``--cpus-per-task``, not the
node's), and no more than ``OMP_NUM_THREADS`` when that is set, as NumPy's BLAS
does: ``OMP_NUM_THREADS=1`` keeps it on one thread.


Covariances and covariance models
---------------------------------

A :class:`~fips.Covariance` is a symmetric operator from a space to itself:
the prior covariance :math:`S_0` or the observation covariance :math:`S_z`. It
adds what a covariance needs: its :attr:`~fips.Covariance.variances`,
:meth:`~fips.Covariance.solve`, :meth:`~fips.Covariance.logdet`,
:meth:`~fips.Covariance.quadratic_form` (of its inverse) and
:meth:`~fips.Covariance.sqrt`, each done the way its form makes cheap.

A matrix you give (``fips.Covariance(array, space)``,
:meth:`~fips.Operator.from_frame`, :meth:`~fips.Operator.from_blocks`, or the
variances of :meth:`Covariance.diagonal <fips.Covariance.diagonal>`) is checked
once, when the covariance is built: it must be symmetric, to 1e-10 of its
largest entry, and positive semidefinite, with no eigenvalue below zero by
more than roundoff (``n * eps`` times the largest). One that is not, a
covariance estimated or tapered by hand, say, raises, naming its smallest
eigenvalue. The check is the square root a draw is made with
(:meth:`~fips.Covariance.sample`), built and kept: for a dense matrix, one
Cholesky factorization and an ``n x n`` factor held beside it; for a sparse
one, a factor for each group of correlated rows; for a singular matrix (zero
variances, a :class:`~fips.Constant` block), which has no Cholesky factor, an
eigendecomposition. A covariance a model builds is positive semidefinite by
construction and is not checked.

Independent errors are a diagonal, kept as its variances, one number per
element: :meth:`Covariance.diagonal <fips.Covariance.diagonal>` takes a scalar
(one variance for every element), an array (one each) or a vector. Independent
parts go side by side with :meth:`Covariance.block_diag
<fips.Covariance.block_diag>`, each keeping its form:

.. ipython:: python

   S_0 = fips.Covariance.block_diag(
       fips.Covariance.diagonal(0.25, fips.Space({"flux": cells})),
       fips.Covariance.diagonal(0.01, fips.Space({"bias": sites})),
   )
   S_0.variances.to_series()
   S_0.data

.. _covariance-models:

**Covariance models.** Correlated errors are usually written as a correlation
model and the standard deviations. A correlation model says how correlated two
labels are from their values on some levels:

- :class:`~fips.Exponential`, :class:`~fips.Gaussian` and :class:`~fips.Matern`
  decay with the distance along their levels: a datetime level with a time
  scale (``Exponential("time", "30D")``), numeric levels as a straight line,
  or latitude and longitude on the Earth (``metric="haversine"``, the
  great-circle distance in km; ``metric="chordal"``, through the Earth, which a
  Gaussian or a Matérn needs to stay positive-definite). The two levels are
  read by position, not by name, so give latitude first:
  ``Exponential(["lat", "lon"], 10.0, metric="haversine")``;
- :class:`~fips.Constant` is 1 everywhere, and :class:`~fips.Independent` the
  identity (1 for a label with itself, 0 otherwise);
- :class:`~fips.Kronecker` multiplies models on separate levels, kept as its
  factors.

``model.with_std(std)`` is the covariance. With the standard deviations as a
number, or a pandas Series on the block's labels (lined up by label), it is a
covariance model, which ``build(space)`` puts on a space; a plain array is
refused, since it has no labels. A :class:`~fips.Problem` builds its
covariance models on the prior's or the observations' space itself. With a
:class:`~fips.Vector` of standard deviations, the space is known and it is a
:class:`~fips.Covariance` at once. Covariance models add (``a + b``), and
:meth:`~fips.Covariance.block_diag` takes them by block name.

``within=`` keeps a model inside groups of labels and makes it zero between
them, as a sparse matrix: ``within=["site", "obs_time.date"]`` correlates the
observations of one site on one day only (``"obs_time.date"`` is the date of
the ``obs_time`` level), and ``within="time.hour"`` the same hour of day only.
Here an independent instrument error plus an error correlated over 2 hours
within a site's day:

.. ipython:: python

   hours = pd.date_range("2024-01-01 21:00", periods=4, freq="h")
   labels = pd.MultiIndex.from_product([["A", "B"], hours], names=["site", "obs_time"])
   obs_cov = fips.Independent().with_std(0.1) + fips.Exponential(
       "obs_time", "2h", within=["site", "obs_time.date"]
   ).with_std(0.5)
   obs_cov
   obs_cov.build(fips.Space({"co2": labels})).dense().round(3)

A :class:`~fips.Kronecker` model is kept as its factors: its covariance holds a
:class:`~fips.linalg.KroneckerOperator`, which never forms the
:math:`n \times n` matrix. Each factor is evaluated once, on the unique values
of its levels, and the labels can be any subset of the product, in any order:

.. ipython:: python

   state = pd.MultiIndex.from_product(
       [pd.date_range("2024-01-01", periods=4, freq="7D"), [40.5, 40.6], [-112.0, -111.9]],
       names=["time", "lat", "lon"],
   )
   prior_cov = fips.Kronecker(
       fips.Exponential("time", "14D"),
       fips.Exponential(["lat", "lon"], 10.0, metric="haversine"),
   ).with_std(0.5)
   S = prior_cov.build(fips.Space({"flux": state}))
   [factor.shape for factor in S.data.factors]

A new kind of correlation is a subclass of
:class:`~fips.covariance.CorrelationModel`: a frozen dataclass with its
``levels`` and a ``correlation(coords)`` method.


The problem
-----------

A :class:`~fips.Problem` is the question: the prior and its covariance, the
observations and theirs, the forward operator, and an optional offset. It
checks the labels once, when it is built, by an exact join: the forward
operator's rows and the observation covariance must have exactly the labels of
the observations, its columns and the prior covariance exactly those of the
prior, in any order. A block or a label on one side only raises a
:class:`ValueError` naming it, so nothing is filled with zeros and nothing is
rounded without your asking. Each input is then held in the order of the
prior's and the observations' spaces, in its own form.

The **offset** :math:`c` is the part of the observations the state does not
explain (a background): a vector on the observations' space, or one number.
:meth:`~fips.Problem.predict` is :math:`H x + c` for any state, such as the
prior or a truth:

.. ipython:: python

   problem = fips.Problem(
       prior=prior,
       prior_cov=S_0,
       obs=obs,
       obs_cov=fips.Independent().with_std(0.2),
       forward=H,
       offset=0.5,
   )
   problem.predict(problem.prior).to_series()

A problem is not changed by a solve, and makes other problems from itself.

**Aggregated observations.** :meth:`Problem.aggregate <fips.Problem.aggregate>`
gives the problem of aggregated observations: the observations, the forward
operator, the observation covariance and the offset each go through one
aggregation :math:`W` (:math:`W z`, :math:`W H`, :math:`W S_z W^T`,
:math:`W c`). :func:`fips.aggregate.mean` and :func:`fips.aggregate.sum` build
:math:`W` from the labels, grouped by levels, time bins or a mapping. Here,
hourly observations at two sites become daily means:

.. ipython:: python

   rng = np.random.default_rng(0)
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
   daily = hourly.aggregate(
       fips.aggregate.mean(ch4.space, by=["site", pd.Grouper(level="obs_time", freq="1D")])
   )
   daily.obs.to_series()

**Offsets the data should decide.** An offset that is not known (a background
that drifts, a calibration bias) is a block of the state instead:
:func:`fips.offset_block` gives one offset per group of observations, zeros as
their prior, and the forward block that adds each offset to its observations.
Their prior covariance is any model on their labels; here the days of a site
are correlated:

.. ipython:: python

   days = pd.interval_range(hours[0], periods=2, freq="1D", closed="left", name="obs_time")
   offsets, H_offsets = fips.offset_block(ch4.space, by=[days, "site"])
   with_offsets = fips.Problem(
       prior=fips.Vector.from_blocks({"flux": flux, "offset": offsets}),
       prior_cov=fips.Covariance.block_diag(
           flux=fips.Independent().with_std(0.5),
           offset=fips.Exponential("obs_time", "2D", within="site").with_std(0.1),
       ),
       obs=ch4,
       obs_cov=fips.Independent().with_std(0.05),
       forward=fips.Operator.from_blocks(
           {("ch4", "flux"): hourly.forward, ("ch4", "offset"): H_offsets["ch4", "offset"]}
       ),
   )
   fips.solve(with_offsets).estimate["offset"].to_series().round(3)

**Part of the state held at its prior.** :meth:`Problem.restrict
<fips.Problem.restrict>` solves for part of the state (cells no observation
sees well are left out) and holds the rest at its prior: their contribution
:math:`H x_0` moves into the offset. :meth:`~fips.Vector.combine_first` puts
the estimate back on the whole state:

.. ipython:: python

   small = hourly.restrict(fips.Space({"flux": pd.Index(["c0", "c1"], name="cell")}))
   fips.solve(small).estimate.combine_first(hourly.prior).to_series().round(3)


Solving and the posterior
-------------------------

:func:`fips.solve` solves a problem with a method and returns a
:class:`~fips.Posterior`, the answer. The method is ``"analytic"`` by default
(the closed-form Gaussian posterior; :func:`fips.available_methods` lists the
methods), and its one option is the regularization factor ``gamma``.
``"bounded"`` holds the state to bounds (:doc:`bounds`), and ``"gauss_newton"``
finds the mode of a problem whose forward is a model that is not linear
(:doc:`nonlinear`). An unknown method or option raises with the names it knows.

The analytic method factors the problem once, in one of two forms that give
the same posterior. The whitened form works in the observations' size: it
whitens with the Cholesky factor of :math:`S_z` (or the square root of its
diagonal), so :math:`S_z` must be positive-definite, and factors the
:math:`m \times m` matrix :math:`H S_0 H^T` in that metric. It computes
:math:`H S_0` in the forms of :math:`H` and :math:`S_0` (through the factors of
a Kronecker covariance), and the estimate, the variances, the DOFS and the
chi-squared from the factors, without forming anything :math:`n \times n`. The
information form works in the state's size: it factors an
:math:`n \times n` matrix built from :math:`H`, a square root of :math:`S_0` and
:math:`S_z^{-1}`, and forms no :math:`m \times m` matrix of its own (a dense
:math:`S_z` is applied through its Cholesky factor, a Kronecker one through
its factors). It is used when there are fewer state elements than
observations and :math:`S_z` is dense, diagonal, sparse, block diagonal or
Kronecker; otherwise the whitened form. In both, the full posterior
covariance (``post.cov()``), the gain and the averaging kernel are formed
only when they are read, once each. :func:`fips.solve` factors that matrix
by Cholesky at its ``gamma``, the cheapest way to solve at one ``gamma``; the
first ``post.with_gamma`` eigendecomposes it, which holds at any ``gamma``,
and keeps it, so a sweep over ``gamma`` pays for one eigendecomposition.

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
   post.variances.to_series().head()

**How much the observations tell.** :attr:`~fips.Posterior.dofs` is the trace
of the averaging kernel, the number of independent pieces of information the
observations give about the state; :attr:`~fips.Posterior.dofs_per_element` is
its diagonal on the state's labels, so the share of a week, a region or a block
is a sum over its labels. :attr:`~fips.Posterior.information_content` is the
Shannon information content, :math:`-\tfrac{1}{2} \ln \lvert I - A \rvert` in
nats: where DOFS counts a constrained direction as at most 1, it keeps growing
as the observations get more precise. Both are read from the factors, without
the :math:`n \times n` kernel:

.. ipython:: python

   post.dofs_per_element.to_series().groupby(level="time").sum().round(3)
   round(post.dofs, 3), round(post.information_content, 3)

**The posterior of aggregates.** An aggregation :math:`W` of the state (one row
per total or mean) gives the posterior covariance of the aggregates,
``post.cov(W)`` (:math:`W \hat{S} W^T`), their prior covariance,
``problem.prior_cov.aggregate(W)`` (:math:`W S_0 W^T`), and their averaging
kernel, ``post.averaging_kernel(W)`` (:math:`W A W^*`, with :math:`W^*` the
pseudo-inverse of :math:`W`), none forming the :math:`n \times n` matrices.
:meth:`~fips.Posterior.uncertainty` puts the two standard deviations of each
aggregate side by side, with the fraction the observations take off. Here, the
total flux of each week:

.. ipython:: python

   W = fips.aggregate.sum(problem.prior.space, by="time")
   post.cov(W).to_frame().round(4)
   post.uncertainty(W).round(3)
   post.averaging_kernel(W).dense().diagonal().round(3)

A row of :math:`W` is any linear combination of the state, so a combination of
totals is a row too: ``C @ W``, with ``C`` the weights of the rows of ``W``. Its
uncertainty includes the covariance between the totals: the change from one
week to the next is not the root of the two weeks' variances summed:

.. ipython:: python

   change = fips.Operator(
       np.array([[-1.0, 1.0, 0.0, 0.0]]),
       fips.Space({"change": pd.Index(["week 2 - week 1"], name="what")}),
       W.rows,
   )
   post.uncertainty(change @ W).round(3)

**Draws, for a question a covariance cannot answer.** ``post.cov(W)`` gives the
spread of every linear combination, but not the chance that a total is above a
value, or which week's total is the largest.
:meth:`~fips.Posterior.sample` draws from the posterior (exactly, for the
analytic method), one column per draw, labeled as the estimate is; with ``W`` it
draws the totals. Take the function of each draw and read the answer off them.
Here, how likely each week's total is to be below 1, and how often each week
has the largest total:

.. ipython:: python

   draws = post.sample(4000, rng=0, W=W)
   draws.shape
   (draws < 1.0).mean(axis=1).round(2)
   largest = draws.idxmax().map(lambda label: label[1])
   largest.value_counts(normalize=True).sort_index().round(2)

Each draw perturbs the prior and the observations by their errors and solves,
which is the posterior's distribution exactly, and costs one product with each
covariance's root and one with the factors for all the draws together: never an
:math:`n \times n` matrix. The draws are a ``state x draws`` array, so for a
large state ask for them in blocks.

``dropna=True`` on :func:`fips.aggregate.sum` and :func:`~fips.aggregate.mean`
picks part of the state: labels a key cannot place are left out instead of
raising (a Series that maps only a region's cells gives the region's total, an
interval of one year that year's), and so are the blocks not named in
``blocks`` (a bias block next to the fluxes).

**Another gamma or other observations, without a new solve.** The factors
depend neither on gamma nor on the values of the observations, so
``post.with_gamma(g)`` is ``fips.solve(problem, gamma=g)`` at the cost of a few
products, and ``post.with_data(obs)`` the posterior of other observations at
the same labels (a bootstrap, :ref:`synthetic observations of a known truth
<synthetic-data>`). A sweep over gamma factors once, and the rows of
:meth:`~fips.Posterior.summary` stack:

.. ipython:: python

   pd.concat([post.with_gamma(g).summary() for g in (0.25, 1.0, 4.0)], ignore_index=True)

**An L-curve.** :attr:`~fips.Posterior.chi2` is the sum of two misfits:
:attr:`~fips.Posterior.data_misfit`, how far the modeled observations are from
the observations (:math:`r^T S_z^{-1} r`, on :math:`S_z` as given, so that it
compares across gammas), times gamma, and
:attr:`~fips.Posterior.state_misfit`, how far the estimate is from the prior
(:math:`(\hat{x} - x_0)^T S_0^{-1} (\hat{x} - x_0)`). Over a sweep of gamma
the first falls and the second rises; plotted one against the other they
trace the L-curve (Hansen, 1992), whose corner is a gamma past which fitting
the observations more closely costs much more departure from the prior:

.. ipython:: python

   sweep = [post.with_gamma(g) for g in np.logspace(-2, 2, 9)]
   pd.DataFrame(
       [(p.options.gamma, p.data_misfit, p.state_misfit) for p in sweep],
       columns=["gamma", "data_misfit", "state_misfit"],
   ).round(3)

**What each group of observations adds.** :meth:`~fips.Posterior.data_worth`
groups the observations by any of their levels (a site, a platform's block, a
month by ``pd.Grouper``), as :meth:`~fips.Posterior.desroziers` does, and says
for each group how uncertain each aggregate would be without it
(``std_without``) and is with it (``std_with``), the fraction it takes off on
top of the other observations (``reduction``), and its share of DOFS. Removing
a group is a rank update of :math:`\hat{S}`, exact for any model-data
mismatch, so no problem is solved again. Two columns say what the group is
worth by itself: ``std_alone`` is how uncertain the aggregate would be with only
this group's observations, on the prior, and ``reduction_alone`` the fraction
that takes off the prior (a rank update of :math:`S_0`). The two readings
differ when groups see the same part of the state: a site that adds little on
top of the others may have taught a lot had it been the only one. They agree
only when there is one group. The alone columns use the group's own block of the
model-data mismatch, so they leave out its correlation with the other groups'
errors. For the total flux of the hourly problem above, observed at two sites:

.. ipython:: python

   hourly_post = fips.solve(hourly)
   total = fips.aggregate.sum(hourly.prior.space, by=pd.Series("all", index=hourly.prior.space.select("flux"), name="region"))
   hourly_post.data_worth(by="site", W=total).round(3)

Observations that are not in the problem, a candidate site or platform, are
their rows of the forward operator and the covariance of their errors (no
values): :meth:`~fips.Posterior.candidate_worth` adds each group on its own,
by the same kind of update, and gives the same columns but the two alone ones
(a candidate is always added on top of the problem's observations), so a network
design question ("which candidate site helps the total most") is one call:

.. ipython:: python

   new_sites = fips.Space({"ch4": pd.MultiIndex.from_product([["C", "D"], hours], names=["site", "obs_time"])})
   H_new = fips.Operator(rng.uniform(0, 0.1, (96, 3)), new_sites, flux.space)
   hourly_post.candidate_worth(H_new, fips.Independent().with_std(0.05), by="site", W=total).round(3)

**Checking the covariances.** :meth:`~fips.Posterior.desroziers` compares the
observation error variances the residuals diagnose with those the solve used,
per observation or averaged over groups; :attr:`~fips.Posterior.reduced_chi2`
is near 1 when the covariances are right; :meth:`~fips.Posterior.leverage`
says which observations weigh most on a state. :attr:`~fips.Posterior.log_evidence`
is the log probability of the observations given the covariances, with the
state integrated out: the settings that maximize it are the
maximum-likelihood ones, and :doc:`settings` fits them with
:mod:`scipy.optimize`.

:meth:`~fips.Posterior.chi2_tests` makes the reduced chi-squared a test
(Rodgers, 2000, chapter 12), and adds the same test of three more vectors:
each is compared with its own covariance, and has its degrees of freedom and
p-value, small when the vector is larger than the covariances allow and near 1
when it is smaller. The four rows are the innovation :math:`z - H x_0 - c`
(the row of :attr:`~fips.Posterior.chi2`, with :math:`m` degrees of freedom),
the residual :math:`z - H \hat{x} - c` (also :math:`m`), the modeled increment
:math:`H (\hat{x} - x_0)` and the increment :math:`\hat{x} - x_0` (each with
as many degrees of freedom as the directions of the state the observations
constrain). In a linear problem, the only kind fips solves, the rows come in
two pairs of the same numbers: the residual's test gives the innovation's, the
modeled increment's the increment's. They differ only when the estimate is not
linear in the observations; the table has four rows so that it keeps its shape
when such a method comes. In the weekly problem the state has more elements
than its 15 observations, which constrain 15 directions of it, so all four rows
are the same number; in the hourly problem many observations see three state
elements, and the two increments have three degrees of freedom. Every test of
the hourly problem rejects its covariances (p-values of 0): its made-up
observations are much farther from the prior than its covariances allow.

.. ipython:: python

   post.chi2_tests().round(3)
   hourly_post.chi2_tests().round(3)


Synthetic truth and observations
--------------------------------

To test an inversion on data whose answer you know, make the data from the
problem: :meth:`Problem.simulate <fips.Problem.simulate>` returns a truth drawn
from the prior and the observations it would give, ``post.with_data(obs)``
solves them at the cost of a product, and :meth:`post.recovery
<fips.Posterior.recovery>` gives the exact bias, spread and coverage of each
total when the data come from a world, without sampling. The page
:doc:`synthetic` is the guide to these experiments: the four kinds of world,
when the answer is exact and when it needs trials, the loop and its summary,
worlds no problem can describe, two inversions on the same data, and what to
read off the table.


Plotting
--------

fips draws one plot, :func:`~fips.visualization.plot_comparison` (the ``viz``
extra): several vectors or Series on one axes, with a band of one standard
deviation from each covariance, and an optional truth. Maps and anything else
are drawn from :meth:`~fips.Vector.to_series` and
:meth:`~fips.Vector.to_xarray` with pandas, xarray or matplotlib:

.. ipython:: python

   from fips.visualization import plot_comparison

   weekly = fips.aggregate.mean(problem.prior.space, by="time")

   @savefig usage_comparison.png width=5in
   fig, ax = plot_comparison(
       (weekly @ problem.prior).to_series().droplevel("block").rename("prior"),
       (weekly @ post.estimate).to_series().droplevel("block").rename("posterior"),
       errors=[problem.prior_cov.aggregate(weekly), post.cov(weekly)],
   )


.. _flux-inversion:

Example: a flux inversion
-------------------------

fips knows no science domain: an atmospheric flux inversion is a
:class:`~fips.Problem` whose state has a ``flux`` block and whose observations
a ``concentration`` block (any names will do), solved with :func:`fips.solve`
like any problem, with its blocks read by name:

.. code-block:: python

   import fips

   problem = fips.Problem(
       prior=fips.Vector.from_series(prior_fluxes, block="flux"),
       prior_cov=prior_cov,  # a Covariance or a covariance model
       obs=fips.Vector.from_series(concentrations, block="concentration"),
       obs_cov=obs_cov,  # a Covariance or a covariance model
       forward=jacobian,  # an Operator: the transport Jacobian
       offset=background,  # a Vector on the observations, or a number
   )
   post = fips.solve(problem)

   post.estimate["flux"].to_xarray()  # posterior fluxes on (time, lat, lon)
   post.cov()["flux", "flux"]  # their posterior covariance
   post.modeled_obs["concentration"]  # H x̂ + c
   problem.predict(problem.prior)  # H x₀ + c
   post.summary()  # DOFS, chi-squared, R², RMSE, ... as one row

For a STILT Jacobian, `PYSTILT <https://jmineau.github.io/PYSTILT>`_ sums the
footprints onto your flux grid and time bins, as a sparse matrix with its row
and column labels. It has columns only for the cells a footprint touched,
while a problem needs the forward operator on exactly the prior's labels:
:meth:`Operator.from_blocks <fips.Operator.from_blocks>` places it on the whole
state by its labels, and keeps it sparse:

.. code-block:: python

   import stilt

   project = stilt.Project("my_project")
   J = project.jacobian(project.simulations, grid, time_bins)
   jacobian = fips.Operator.from_blocks(
       {("concentration", "flux"): (J.data, J.receptors.rename("receptor"), J.columns)},
       rows=obs.space,
       cols=prior.space,
   )

Label the observations and the prior to match the Jacobian's rows (receptor
ids) and columns (``(time, lon, lat)``).


.. _repeated-inversions:

Repeated inversions, logging and saving
---------------------------------------

A workflow run many times (an inversion per month, a sweep over settings) is a
few functions in your own code: one that builds the :class:`~fips.Problem` from
your settings, then :func:`fips.solve`:

.. code-block:: python

   import pandas as pd

   import fips

   def build_problem(config) -> fips.Problem:
       """Load the observations and the prior, build H and the covariances."""
       ...

   rows = [fips.solve(build_problem(config)).summary() for config in configs]
   table = pd.concat(rows, ignore_index=True)

fips prints nothing. It logs through the standard :mod:`logging` module, under
the ``fips`` logger (the solve under ``fips.methods``), and Python shows only
warnings until logging is configured; to see its ``INFO`` records:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)

A :class:`~fips.Problem` and a :class:`~fips.Posterior` pickle with the
standard library (``pickle.dump(post, file)``): a posterior keeps its problem,
its options and what the method kept from the solve, and rebuilds every output
when it is read.

.. toctree::
   :hidden:

   synthetic
   settings
   bounds
   nonlinear
   positive
   comparison
   examples/index
