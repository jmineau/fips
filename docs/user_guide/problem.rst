.. _building-a-problem:

Building a problem
==================

A problem is five things: the prior and its covariance, the observations and
their covariance, and the forward operator that maps a state to the
observations it would produce (and, if needed, an offset). fips holds each of
them with its labels, and matches them by label, never by position. This
page builds each piece, then the :class:`~fips.Problem`.

.. contents:: On this page
   :local:
   :depth: 1


Spaces and vectors
------------------

**Every number in fips has a label.** A :class:`~fips.Space` is the labels of
a state, of the observations, or of one axis of a matrix: an ordered set of
named **blocks**, each labeled by its own pandas index. A state can have a
``flux`` block labeled by ``(time, lat, lon)`` and a ``bias`` block labeled by
``site``; the observations a ``station`` block and a ``satellite`` block.

A :class:`~fips.Vector` is values on a space: a prior, the observations, an
estimate. Build one from a Series (one block) or from several (one per block):

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

The blocks together are one :class:`pandas.MultiIndex` whose first level is
``block`` (``prior.space.index``). A level that one block has and another
lacks is NaN on the other's rows. :class:`~fips.Space` adds and removes those
levels, so a block comes back labeled as it was given: ``prior["flux"]`` has
its integer cells. ``prior.space.select("flux")`` gives a block's labels, and
:meth:`~fips.Vector.to_xarray` a DataArray on one block's levels (a Dataset
for several).

**Labels are kept as given.** fips does not convert strings to numbers or
round floats. Duplicate, missing (NaN) and unnamed labels raise a
:class:`ValueError` that names them. To match labels that agree only after
rounding, round them yourself with :meth:`~fips.Vector.round_labels` (also on
a :class:`~fips.Space` and an :class:`~fips.Operator`).

**Arithmetic is by label.** ``v + w``, ``v - w``, ``v * w`` and ``v / w`` put
``w`` in the order of ``v``'s labels first; a label on one side only raises,
naming it. A number or a 1D array combines by position. ``np.exp(v)`` and
``np.maximum(v, w)`` return vectors. A reduction such as ``np.sum(v)`` is not
defined: use ``v.values``.

**A vector is read-only**: a new value is a new vector.
:meth:`~fips.Vector.align` puts a vector in the order of a space with the same
labels, and :meth:`~fips.Vector.combine_first` fills the labels one vector
lacks from another (an estimate of part of the state, completed by the prior).


Operators
---------

**An** :class:`~fips.Operator` **is a matrix with a space on each axis**: the
forward operator (observations by state), an aggregation, a gain. The matrix
is kept in the cheapest exact form (a dense array, a SciPy sparse matrix, a
diagonal, Kronecker factors; :doc:`/reference/linalg`), with the labels beside
it. There are three ways to build one:

- ``Operator(matrix, rows, cols)`` labels a matrix you have, by position: its
  rows are the labels of the space ``rows`` in order, its columns those of
  ``cols``. A SciPy sparse matrix stays sparse.
- :meth:`Operator.from_frame <fips.Operator.from_frame>` reads a labeled
  DataFrame, dense or with pandas-sparse columns.
- :meth:`Operator.from_blocks <fips.Operator.from_blocks>` assembles one from
  blocks, each keyed by its row block and its column block: a DataFrame, or a
  tuple ``(matrix, row_labels, col_labels)``. Given the spaces (``rows=``,
  ``cols=``), it places each block by its labels. A label a block has no entry
  for is zero, as is a pair of blocks left out; a label the space lacks raises.

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

``H @ vector`` is a vector and ``H @ other`` an operator; ``+`` and ``-`` add
operators. Each first matches the labels the two sides meet on, in any order,
and a label on one side only raises. ``H["co2", "flux"]`` is one block,
``H.T`` the transpose, and :meth:`~fips.Operator.to_frame` the dense
DataFrame.

**Sparse matrices stay sparse.** A forward operator from a transport model is
mostly zeros. An operator built from a :mod:`scipy.sparse` matrix (or a
DataFrame with pandas-sparse columns) keeps it sparse, and so do products and
sums that stay mostly zeros. A product turns dense once more than a quarter of
its entries are filled, or when either side is dense. The forward operator and
both covariances can each be dense or sparse:

.. ipython:: python

   import scipy.sparse as sp

   J = sp.random(3, 3, density=0.4, random_state=0, format="csr")
   sparse_H = fips.Operator(J, obs.space, fips.Space({"flux": cells}))
   sparse_H.data

A large product of a sparse matrix and a dense one (:math:`H S_0 H^T` with a
dense :math:`S_0`) runs on threads and gives the same numbers as one thread,
bit for bit. It uses the CPUs the process may run on (a Slurm job's
``--cpus-per-task``, not the whole node's), and no more than
``OMP_NUM_THREADS`` when that is set; ``OMP_NUM_THREADS=1`` keeps it on one
thread.


.. _covariance-models:

Covariances and covariance models
---------------------------------

**A** :class:`~fips.Covariance` **is a symmetric operator from a space to
itself**: the prior covariance :math:`S_0` or the observation covariance
:math:`S_z`. It adds what a covariance needs, each done the way its form makes
cheap: :attr:`~fips.Covariance.variances`, :meth:`~fips.Covariance.solve`,
:meth:`~fips.Covariance.logdet`, :meth:`~fips.Covariance.quadratic_form` (of
its inverse), :meth:`~fips.Covariance.sqrt` and :meth:`~fips.Covariance.sample`.

There are two ways to write one: give the matrix, or write a **covariance
model**, a correlation model times standard deviations, and let the problem
put it on its labels. Most problems use models.

**Independent errors** are a diagonal, kept as its variances.
:meth:`Covariance.diagonal <fips.Covariance.diagonal>` takes one variance for
every element, an array (one each) or a vector. Independent parts go side by
side with :meth:`Covariance.block_diag <fips.Covariance.block_diag>`, each
keeping its form:

.. ipython:: python

   S_0 = fips.Covariance.block_diag(
       fips.Covariance.diagonal(0.25, fips.Space({"flux": cells})),
       fips.Covariance.diagonal(0.01, fips.Space({"bias": sites})),
   )
   S_0.variances.to_series()
   S_0.data

**A correlation model** says how correlated two labels are from their values
on some levels:

- :class:`~fips.Exponential`, :class:`~fips.Gaussian` and :class:`~fips.Matern`
  decay with the distance along their levels. A datetime level takes a time
  scale (``Exponential("time", "30D")``). Numeric levels are measured in a
  straight line. Latitude and longitude are measured on the Earth:
  ``metric="haversine"`` is the great-circle distance in km, and
  ``metric="chordal"`` the distance through the Earth, which a Gaussian or a
  Matérn needs to stay positive definite. The two levels are read by position,
  not by name, so give latitude first:
  ``Exponential(["lat", "lon"], 10.0, metric="haversine")``.
- :class:`~fips.Constant` is 1 everywhere; :class:`~fips.Independent` is the
  identity (1 for a label with itself, 0 otherwise).
- :class:`~fips.Kronecker` multiplies models on separate levels and is kept as
  its factors.

``model.with_std(std)`` is the covariance model. ``std`` is a number or a
pandas Series on the block's labels (matched by label); a plain array is
refused, since it has no labels. A :class:`~fips.Problem` builds a covariance
model on the prior's or the observations' space itself, and ``build(space)``
does it by hand. With a :class:`~fips.Vector` of standard deviations the space
is known, and the result is a :class:`~fips.Covariance` at once. Covariance
models add (``a + b``), and :meth:`~fips.Covariance.block_diag` takes them by
block name.

``within=`` keeps a model inside groups of labels and makes it zero between
them, as a sparse matrix. ``within=["site", "obs_time.date"]`` correlates the
observations of one site on one day only (``"obs_time.date"`` is the date of
the ``obs_time`` level), and ``within="time.hour"`` the same hour of day only.
Here, an independent instrument error plus an error correlated over 2 hours
within a site's day:

.. ipython:: python

   hours = pd.date_range("2024-01-01 21:00", periods=4, freq="h")
   labels = pd.MultiIndex.from_product([["A", "B"], hours], names=["site", "obs_time"])
   obs_cov = fips.Independent().with_std(0.1) + fips.Exponential(
       "obs_time", "2h", within=["site", "obs_time.date"]
   ).with_std(0.5)
   obs_cov
   obs_cov.build(fips.Space({"co2": labels})).dense().round(3)

A :class:`~fips.Kronecker` model never forms the :math:`n \times n` matrix:
its covariance holds a :class:`~fips.linalg.KroneckerOperator`. Each factor is
evaluated once, on the unique values of its levels, and the labels can be any
subset of the product, in any order:

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

**A matrix you give is checked once**, when the covariance is built
(``fips.Covariance(array, space)``, :meth:`~fips.Operator.from_frame`,
:meth:`~fips.Operator.from_blocks`, :meth:`Covariance.diagonal
<fips.Covariance.diagonal>`). It must be symmetric, to 1e-10 of its largest
entry, and positive semidefinite, with no eigenvalue below zero by more than
roundoff (``n * eps`` times the largest). One that fails raises and names its
smallest eigenvalue: a covariance estimated or tapered by hand often does. The
check builds the square root that :meth:`~fips.Covariance.sample` draws with,
and keeps it. A covariance a model builds is positive semidefinite by
construction and is not checked.


The problem
-----------

**A** :class:`~fips.Problem` **is the question**: the prior and its
covariance, the observations and theirs, the forward operator, and an
optional offset. It checks the labels once, when it is built. The forward
operator's rows and the observation covariance must have exactly the labels
of the observations, and its columns and the prior covariance exactly those
of the prior, in any order. A block or a label on one side only raises a
:class:`ValueError` naming it; nothing is filled with zeros or rounded unless
you ask. Each input is then held in the order of the prior's and the
observations' spaces, in its own form.

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

A problem does not change. Solving it leaves it as it was, and each of the
methods below returns a new problem.

**Aggregated observations.** :meth:`Problem.aggregate
<fips.Problem.aggregate>` gives the problem of aggregated observations: the
observations, the forward operator, the observation covariance and the offset
each go through one aggregation :math:`W` (:math:`W z`, :math:`W H`,
:math:`W S_z W^T`, :math:`W c`). :func:`fips.aggregate.mean` and
:func:`fips.aggregate.sum` build :math:`W` from the labels, grouped by levels,
time bins or a mapping. Here, hourly observations at two sites become daily
means:

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

**An offset the data should decide** (a background that drifts, a calibration
bias) is a block of the state instead. :func:`fips.offset_block` gives one
offset per group of observations, zeros as their prior, and the forward block
that adds each offset to its observations. Their prior covariance is any model
on their labels; here a site's days are correlated:

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
<fips.Problem.restrict>` solves for part of the state (leaving out elements no
observation sees well) and holds the rest at its prior: their contribution
:math:`H x_0` moves into the offset. :meth:`~fips.Vector.combine_first` puts
the estimate back on the whole state:

.. ipython:: python

   small = hourly.restrict(fips.Space({"flux": pd.Index(["c0", "c1"], name="cell")}))
   fips.solve(small).estimate.combine_first(hourly.prior).to_series().round(3)

**A forward model that is not linear** goes in the same place: ``forward=``
takes a :class:`~fips.ForwardModel` as well as an operator
(:doc:`nonlinear`).


.. _flux-inversion:

Example: a flux inversion
-------------------------

fips knows no science domain. An atmospheric flux inversion is a
:class:`~fips.Problem` whose state has a ``flux`` block and whose observations
have a ``concentration`` block (any names will do), solved like any other:

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
footprints onto your grid and time bins, as a sparse matrix with its row and
column labels. It has columns only for the cells a footprint touched, while a
problem needs the forward operator on exactly the prior's labels.
:meth:`Operator.from_blocks <fips.Operator.from_blocks>` places it on the
whole state by its labels, and keeps it sparse:

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
