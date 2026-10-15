User Guide
==========

.. contents:: On this page
   :local:
   :depth: 2


Data Model
----------

*fips* keeps labels beside the numbers, never inside them:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Role
   * - :class:`~fips.Space`
     - The labels of a state, of the observations, or of one axis of a
       matrix: named blocks, each labeled by its own pandas index, together
       one :class:`pandas.MultiIndex` with a ``"block"`` level first
   * - :class:`~fips.Vector`
     - Values on a space: a prior, the observations, an estimate
   * - :class:`~fips.Operator`
     - A matrix from one space (its columns) to another (its rows), kept in
       its cheapest form: the forward operator :math:`H`, an aggregation
   * - :class:`~fips.Covariance`
     - A symmetric operator from a space to itself: :math:`S_0`, :math:`S_z`
   * - :class:`~fips.Problem`
     - The inputs of a solve: prior, observations, forward operator,
       covariances and an offset, their labels checked once
   * - :class:`~fips.Posterior`
     - What :func:`fips.solve` returns: the estimate and its diagnostics,
       labeled, from the factorization it keeps

Blocks may have different index levels: a level one block has and another
lacks is missing (NaN) on the other's rows, and :class:`~fips.Space` is the
one place that pads and strips those levels, so a block comes back with the
labels (and dtypes) it was given. A :class:`~fips.Problem` puts each input on
the observation and state labels by an exact join: the labels may come in any
order, but a block or a label that one side has and the other lacks raises a
:class:`ValueError` that names it, and so does a label that appears twice in a
block. Labels are kept as given: fips does not convert them to numbers, and
rounds float labels only when asked (:meth:`Space.round_labels
<fips.Space.round_labels>`).

.. tip::

   pandas objects come in through constructors that read their labels:
   :meth:`Vector.from_series <fips.Vector.from_series>` for the prior and the
   observations, :meth:`Operator.from_frame <fips.Operator.from_frame>` and
   :meth:`Covariance.from_frame <fips.Operator.from_frame>` for matrices.


Building Operators and Covariances
----------------------------------

An :class:`~fips.Operator` is built from its matrix and the two spaces, from
a labeled :class:`pandas.DataFrame` (:meth:`~fips.Operator.from_frame`), or
from its blocks (:meth:`~fips.Operator.from_blocks`), one per pair of a row
block and a column block; a pair left out is zero. ``op.to_frame()`` gives the
DataFrame back, and ``op["co2", "flux"]`` one block.

.. ipython:: python

   import pandas as pd, numpy as np
   import fips

   obs = pd.Index(["o1", "o2", "o3"], name="obs_id")
   cells = pd.Index(["a", "b"], name="cell")
   H = fips.Operator.from_blocks(
       {
           ("co2", "flux"): pd.DataFrame(np.ones((3, 2)), index=obs, columns=cells),
           ("co2", "bias"): pd.DataFrame(1.0, index=obs, columns=pd.Index(["A"], name="site")),
       }
   )
   H.cols
   H.to_frame()

A diagonal covariance (independent errors) has its own constructor,
:meth:`~fips.Covariance.diagonal`. It takes the variances and the space that
labels both axes: a scalar is the same variance for every element, and a 1D
array or list gives one for each. A :class:`~fips.Vector` brings its own space.
The covariance is kept as its diagonal, so it costs one number per element:

.. ipython:: python

   space = fips.Space({"obs": pd.Index(["a", "b", "c"], name="id")})

   # Diagonal: independent errors with the same variance
   fips.Covariance.diagonal(0.25, space).dense()

   # Diagonal: a variance for each element
   S = fips.Covariance.diagonal([0.5, 0.5, 1.0], space)
   S.data

   # Full dense matrix
   fips.Covariance(np.eye(3) * 0.25, space).shape

Independent covariances go side by side with
:meth:`~fips.Covariance.block_diag`, each keeping its form. A covariance also
gives its :attr:`~fips.Covariance.variances`, :meth:`~fips.Covariance.solve`,
:meth:`~fips.Covariance.logdet`, :meth:`~fips.Covariance.quadratic_form` (of
its inverse) and :meth:`~fips.Covariance.sqrt`.

.. _covariance-models:

Covariance models
-----------------

A covariance is usually written as a correlation model and the standard
deviations. A model says how correlated two labels are from their
coordinates: :class:`~fips.Exponential`, :class:`~fips.Gaussian` and
:class:`~fips.Matern` decay with the distance along some levels (a time, a
straight line, or a distance on the Earth), :class:`~fips.Constant` is one,
:class:`~fips.Independent` is the identity, and :class:`~fips.Kronecker`
multiplies models on separate levels. ``model.with_std(std)`` is the
covariance: given the standard deviations as a :class:`~fips.Vector`, it is a
:class:`~fips.Covariance` at once; given a scalar or an array, it is a
:class:`~fips.models.CovarianceModel`, which ``build(space)`` puts on a space.
Models add (``a + b``), and :meth:`~fips.Covariance.block_diag` takes them by
block name.

``within=`` keeps a model inside groups of labels and makes it zero between
them: ``within=["site", "obs_time.date"]`` correlates the observations of one
site on one day only (``"obs_time.date"`` is the date of the ``obs_time``
level), and ``within="time.hour"`` correlates only the same hour of day. The
matrix is then sparse.

.. ipython:: python

   obs = pd.MultiIndex.from_product(
       [["A", "B"], pd.date_range("2024-01-01 21:00", periods=4, freq="h")],
       names=["site", "obs_time"],
   )
   obs_space = fips.Space({"co2": obs})
   S_z = fips.Independent().with_std(0.1) + fips.Exponential(
       "obs_time", "2h", within=["site", "obs_time.date"]
   ).with_std(0.5)
   S_z
   S_z.build(obs_space).dense().round(3)

A :class:`~fips.Kronecker` model is kept as its factors: its covariance holds
a :class:`~fips.kronecker.KroneckerOperator`, which never forms the
:math:`n \times n` matrix. Each factor is evaluated once, on the unique
coordinates of its levels; the rows can be any subset of the product, in any
order. The solve computes :math:`H S_0` and :math:`H S_0 H^T` from the
factors, and ``dense()`` gives the matrix:

.. ipython:: python

   times = pd.date_range("2024-01-01", periods=4, freq="7D")
   state = pd.MultiIndex.from_product(
       [["flux"], times, [40.5, 40.6], [-112.0, -111.9]],
       names=["block", "time", "lat", "lon"],
   )
   model = fips.Kronecker(
       fips.Exponential("time", "14D"),
       fips.Exponential(["lat", "lon"], 10.0, metric="haversine"),
   )
   S_0 = model.with_std(0.5).build(fips.Space.from_index(state))
   [C.shape for C in S_0.data.factors]
   S_0.dense().shape

``metric="haversine"`` measures latitude and longitude along the great
circle, in km; ``metric="chordal"`` through the Earth, which a
:class:`~fips.Gaussian` or :class:`~fips.Matern` needs to stay
positive-definite.

A :class:`~fips.Problem` holds the spaces already, so its ``prior_cov`` and
``obs_cov`` can be given as a model, which it builds on the prior's or the
observations' space: ``prior_cov=model.with_std(0.5)``.


Sparse Matrices
---------------

For large transport Jacobians (e.g. from STILT) with many structural zeros,
sparse storage gives significant memory and speed savings. An operator built
from a :mod:`scipy.sparse` matrix keeps it, never forming the dense one; so
does one read from a DataFrame with pandas-sparse columns, whose entries a
column does not store are zeros.

.. note::

   The forward operator :math:`H` is also called the Jacobian or observation operator.
   See :ref:`inverse-problem-terms` for details.

.. code-block:: python

   import scipy.sparse as sp
   from fips import Operator, Space

   H = Operator(sp.csr_matrix(values), rows=obs_space, cols=state_space)

The same holds for the covariances: the forward operator, the prior error and
the model-data mismatch can each be dense or sparse, in any combination. Each
operator's ``data`` is its form (:mod:`fips.linalg`): a sparse matrix as a
:class:`~fips.linalg.SparseOperator`, and a sparse matrix whose entries are
all on its diagonal as that diagonal alone:

.. ipython:: python

   import scipy.sparse as sp

   S = fips.Covariance(sp.diags([0.5, 1.0, 2.0]), space)
   S.data
   S.data.solve(np.ones(3))

A product keeps the cheapest exact form: a diagonal times a sparse matrix is
sparse, and anything times a dense matrix is dense. :math:`H S_0 H^T` stays
sparse while it is mostly zeros, and is dense once more than a quarter of its
entries are filled, since a dense solve is then faster. Solves divide by a
diagonal, factor a sparse matrix with a sparse LU, and a dense one with
Cholesky. A Kronecker prior error multiplies :math:`H` through its factors, and
solves through each factor's Cholesky factorization. The gain is a dense
:math:`n \times m` array whatever the inputs; the full posterior covariance is a
dense :math:`n \times n` matrix, formed only when it is read (see below).


The Posterior
-------------

:func:`fips.solve` returns a :class:`~fips.Posterior`. The Bayesian solve
whitens the problem once: it eigendecomposes an :math:`m \times m` matrix
built from :math:`H S_0 H^T` and :math:`S_z`, keeps the factors, and derives
the estimate, the posterior variances, DOFS and the reduced chi-squared from
them, without forming anything :math:`n \times n`. The full posterior
covariance (``post.cov()``) and averaging kernel
(``post.averaging_kernel()``) are formed only when read.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips
   from fips import Exponential, Kronecker, Problem, Space, Vector

The state is six weeks of fluxes on four cells, with a prior error correlated
in time and space:

.. ipython:: python

   rng = np.random.default_rng(0)
   times = pd.date_range("2024-01-01", periods=6, freq="7D")
   state = pd.MultiIndex.from_product(
       [["flux"], times, [40.5, 40.6], [-112.0, -111.9]],
       names=["block", "time", "lat", "lon"],
   )
   obs = pd.MultiIndex.from_product([["conc"], range(15)], names=["block", "obs_id"])
   H = pd.DataFrame(rng.uniform(0, 1, (15, len(state))), index=obs, columns=state)
   prior_cov = Kronecker(
       Exponential("time", "14D"),
       Exponential(["lat", "lon"], 10.0, metric="haversine"),
   ).with_std(0.5)
   problem = Problem(
       prior=Vector.from_series(pd.Series(1.0, index=state)),
       prior_cov=prior_cov,
       obs=Vector.from_series(pd.Series(rng.normal(3.0, 1.0, 15), index=obs)),
       obs_cov=fips.Independent().with_std(np.sqrt(0.5)),
       forward=fips.Operator.from_frame(H),
   )
   post = fips.solve(problem)
   post.variances.to_series().head()

An aggregation turns the state into a few totals or means: an
:class:`~fips.Operator` with one row per aggregate and the state's labels as
its columns (``1`` for a sum, ``1/count`` for a mean). ``post.cov(W)`` is the posterior covariance of
the aggregates, :math:`W \hat{S} W^T`, and ``post.averaging_kernel(W)`` their
averaging kernel, :math:`W A W^*` with :math:`W^*` the pseudo-inverse of
:math:`W`; neither forms the :math:`n \times n` matrices. Here, the total
flux of each week:

.. ipython:: python

   weeks = pd.MultiIndex.from_product([["total"], times], names=["block", "week"])
   W = fips.Operator.from_frame(
       pd.DataFrame(
           (state.get_level_values("time").to_numpy() == times.to_numpy()[:, None]).astype(float),
           index=weeks,
           columns=state,
       )
   )
   post.cov(W).to_frame().round(4)
   post.averaging_kernel(W).dense().diagonal().round(3)

The factors do not depend on the regularization factor ``gamma``, so
``post.with_gamma(0.5)`` is the posterior of ``fips.solve(problem, gamma=0.5)``
at the cost of a few products, not a new solve. A sweep over gamma factors
once:

.. ipython:: python

   for gamma in (0.25, 0.5, 1.0, 2.0):
       p = post.with_gamma(gamma)
       print(f"gamma {gamma}: DOFS {p.dofs:.2f}, reduced chi2 {p.reduced_chi2:.2f}")


.. _flux-inversion:

Domain-Specific Problem: Flux Inversion
-----------------------------------------

:class:`~fips.problems.flux.FluxProblem` is a ready-made subclass of
:class:`~fips.Problem` for atmospheric flux estimation using STILT
transport footprints.  It adds named accessors that speak the language of the
application; it is solved with :func:`fips.solve`, as any problem:

.. code-block:: python

   import fips
   from fips import Vector
   from fips.problems.flux import FluxPlotter, FluxProblem

   inversion = FluxProblem(
       obs=Vector.from_series(concentrations, block="concentration"),
       prior=Vector.from_series(prior_fluxes, block="flux"),
       forward=jacobian,  # an Operator: the transport Jacobian
       obs_cov=S_z,       # concentration error covariance (or a model)
       prior_cov=S_0,     # flux error covariance (or a model)
   )
   post = fips.solve(inversion)

   # Domain-aware accessors of the problem
   inversion.concentrations        # observed concentrations
   inversion.prior_fluxes          # prior flux pd.Series
   inversion.prior_concentrations  # H @ prior + c, inversion.predict(inversion.prior)

   # The posterior's blocks
   post.estimate["flux"]           # posterior fluxes, a Vector (.to_series())
   post.cov()["flux", "flux"]      # posterior flux error, a Covariance
   post.modeled_obs["concentration"]  # H @ posterior + c

   # Built-in plots of a posterior (requires cartopy)
   plot = FluxPlotter(post)
   plot.fluxes()
   plot.concentrations()

For a STILT Jacobian, `PYSTILT <https://jmineau.github.io/PYSTILT>`_ sums the
footprints onto your flux grid and time bins, as a sparse matrix with its
labels. Its rows are receptor ids and its columns ``(time, lon, lat)``, so
label the observations and the prior to match. The Jacobian has columns only
for the cells a footprint touched, while a :class:`~fips.Problem` needs the
forward operator on exactly the prior's labels. :meth:`Operator.from_blocks
<fips.Operator.from_blocks>` places a labeled block by its labels: a cell the
Jacobian lacks is a column of zeros, and a label the state lacks raises. The
matrix stays sparse:

.. code-block:: python

   import stilt
   from fips import Operator

   project = stilt.Project("my_project")
   sims = project.simulations
   H = project.jacobian(sims[sims.variant == "hrrr"], grid, time_bins)
   J = (H.data, H.receptors.rename("receptor"), H.columns)  # matrix, row and column labels
   jacobian = Operator.from_blocks(
       {("concentration", "flux"): J}, rows=obs.space, cols=prior.space
   )

A Jacobian given as a DataFrame, dense or pandas-sparse, is placed the same
way: ``Operator.from_blocks({("concentration", "flux"): frame}, rows=obs.space,
cols=prior.space)``.

Background subtraction is supported via the ``offset`` argument to
:class:`~fips.Problem`:

.. code-block:: python

   inversion = FluxProblem(
       ...
       offset=background,  # a Vector on the observations, subtracted from them
   )
   inversion.background    # returns the background pd.Series


Advanced: Reading Blocks of the Results
---------------------------------------

A problem is solved with :func:`fips.solve`, which returns the posterior. The
problem is the question and holds what needs no solve
(:meth:`~fips.Problem.predict`); the posterior is the answer and holds what
depends on the solve. Domain-specific names are functions of their blocks:

.. code-block:: python

   import fips

   post = fips.solve(problem)

   def density(post):
       """Posterior density anomaly [g cm⁻³]."""
       return post.estimate["density"]

   def gravity_residual(post):
       """Observed minus prior-modelled gravity."""
       problem = post.problem
       prior_modeled = problem.predict(problem.prior)
       return problem.obs["gravity"].values - prior_modeled["gravity"].values

All :class:`~fips.Posterior` results are computed when first read, so you only
pay for what you access.


Advanced: Inversion Pipelines
------------------------------

:class:`~fips.pipeline.InversionPipeline` provides a structured, reproducible
workflow.  Subclass it, implement the abstract hooks, then call
:meth:`~fips.pipeline.InversionPipeline.run`:

.. code-block:: python

   from fips import Covariance, Operator, Problem, Vector
   from fips.pipeline import InversionPipeline

   class MyPipeline(InversionPipeline):

       def __init__(self, config):
           super().__init__(config=config, problem=Problem, method="analytic")

       # --- Required hooks ---

       def get_obs(self) -> Vector:
           """Load observations from config / disk / API."""
           ...

       def get_prior(self) -> Vector:
           """Load prior from inventory / model output."""
           ...

       def get_forward_operator(self, obs, prior) -> Operator:
           """Build or load the Jacobian."""
           ...

       def get_prior_error(self, prior) -> Covariance:
           ...

       def get_modeldata_mismatch(self, obs) -> Covariance:
           ...

   post = MyPipeline(config).run()  # a Posterior; the problem is .problem

``run`` logs each step, with its elapsed time, at ``INFO`` level through the
``fips.pipeline`` logger, and prints nothing. Python shows only warnings until
logging is configured, so to follow a long run, or to keep the steps in a batch
job's output, configure it before the call:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)

Optional hooks let you customise the workflow without touching its core:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Hook
     - Purpose
   * - :meth:`~fips.pipeline.InversionPipeline.filter_state_space`
     - Trim or align obs / state space before building operators
   * - :meth:`~fips.pipeline.InversionPipeline.aggregate_obs_space`
     - Aggregate observations (e.g. hourly → daily) after building operators
   * - :meth:`~fips.pipeline.InversionPipeline.get_constant`
     - Provide a background or offset to subtract from observations

For atmospheric flux inversion, :class:`~fips.problems.flux.pipeline.FluxInversionPipeline`
pre-implements several of these hooks (interval filtering, minimum-obs thresholds,
summary reporting) so you only need to supply your data loaders.
