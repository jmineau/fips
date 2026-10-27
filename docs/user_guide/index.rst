User Guide
==========

This guide takes fips's objects one at a time, in the order a problem uses
them, and says what each is for before how to call it. :doc:`/getting_started/first_problem`
solves one problem from start to finish; :doc:`/getting_started/terminology` defines the words.

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
  Gaussian or a Matérn needs to stay positive-definite);
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
(the closed-form Gaussian posterior, the one method so far;
:func:`fips.available_methods` lists them), and its one option is the
regularization factor ``gamma``. An unknown method or option raises with the
names it knows.

The analytic method factors the problem once, in one of two forms that give
the same posterior. The whitened form works in the observations' size: it
whitens with the Cholesky factor of :math:`S_z` (or the square root of its
diagonal), so :math:`S_z` must be positive-definite, and eigendecomposes the
:math:`m \times m` matrix :math:`H S_0 H^T` in that metric. It computes
:math:`H S_0` in the forms of :math:`H` and :math:`S_0` (through the factors of
a Kronecker covariance), and the estimate, the variances, the DOFS and the
chi-squared from the factors, without forming anything :math:`n \times n`. The
information form works in the state's size: it eigendecomposes an
:math:`n \times n` matrix built from :math:`H`, a square root of :math:`S_0` and
:math:`S_z^{-1}`, and forms no :math:`m \times m` matrix of its own (a dense
:math:`S_z` is applied through its Cholesky factor). It is used when there
are fewer state elements than observations and :math:`S_z` is dense,
diagonal, sparse or block diagonal; otherwise the whitened form. In both, the
full posterior covariance (``post.cov()``), the gain and the averaging
kernel are formed only when they are read, once each.

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

**The posterior of aggregates.** An aggregation :math:`W` of the state (one row
per total or mean) gives the posterior covariance of the aggregates,
``post.cov(W)`` (:math:`W \hat{S} W^T`), and their averaging kernel,
``post.averaging_kernel(W)`` (:math:`W A W^*`, with :math:`W^*` the
pseudo-inverse of :math:`W`), neither forming the :math:`n \times n`
matrices. Here, the total flux of each week:

.. ipython:: python

   W = fips.aggregate.sum(problem.prior.space, by="time")
   post.cov(W).to_frame().round(4)
   post.averaging_kernel(W).dense().diagonal().round(3)

**Another gamma or other observations, without a new solve.** The factors
depend neither on gamma nor on the values of the observations, so
``post.with_gamma(g)`` is ``fips.solve(problem, gamma=g)`` at the cost of a few
products, and ``post.with_data(obs)`` the posterior of other observations at
the same labels (a bootstrap, synthetic observations of a known truth). A sweep
over gamma factors once, and the rows of :meth:`~fips.Posterior.summary` stack:

.. ipython:: python

   pd.concat([post.with_gamma(g).summary() for g in (0.25, 1.0, 4.0)], ignore_index=True)

**Checking the covariances.** :meth:`~fips.Posterior.desroziers` compares the
observation error variances the residuals diagnose with those the solve used,
per observation or averaged over groups; :attr:`~fips.Posterior.reduced_chi2`
is near 1 when the covariances are right; :meth:`~fips.Posterior.leverage`
says which observations weigh most on a state.


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
       errors=[weekly @ problem.prior_cov @ weekly.T, post.cov(weekly)],
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

   examples/index
