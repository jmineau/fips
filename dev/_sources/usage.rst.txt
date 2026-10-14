User Guide
==========

.. contents:: On this page
   :local:
   :depth: 2


Data Model
----------

*fips* uses a two-level hierarchy for all inputs:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 1-D class
     - 2-D class
     - Role
   * - :class:`~fips.Block`
     - :class:`~fips.MatrixBlock`
     - Single named component — the fundamental unit
   * - :class:`~fips.Vector`
     - :class:`~fips.Matrix`
     - Multi-block composite used directly by :class:`~fips.InverseProblem`
   * - —
     - :class:`~fips.CovarianceMatrix`
     - Semantic subclass of :class:`~fips.Matrix` for error covariances
   * - —
     - :class:`~fips.ForwardOperator`
     - Semantic subclass of :class:`~fips.Matrix` for the Jacobian / :math:`H`

A :class:`~fips.Vector` concatenates :class:`~fips.Block` objects along a
hierarchical index that adds a ``"block"`` level identifying each component.
The same pattern applies in 2-D.  Index alignment across components is handled
automatically: :class:`~fips.InverseProblem` places each matrix on the
observation and state labels, and fills the labels a matrix lacks within a
block with zeros (with a warning). A block name that matches no block of the
state or the observations raises a :class:`ValueError`, and so does a label
that appears twice in the state or the observations, including two labels
that ``round_index`` rounds to the same value.

.. tip::

   Plain :class:`pandas.Series` and :class:`pandas.DataFrame` are accepted
   everywhere — *fips* will promote them to :class:`~fips.Block` /
   :class:`~fips.MatrixBlock` automatically.


Constructing Covariance Matrices
---------------------------------

:class:`~fips.CovarianceMatrix` takes a full :class:`pandas.DataFrame` (or a 2D
array with an ``index``), or a list of :class:`~fips.MatrixBlock` objects. The
index needs a ``"block"`` level.

A diagonal covariance (independent errors) has its own constructor,
:meth:`~fips.CovarianceMatrix.diagonal`. It takes the variances and the index
that labels both axes: a scalar is the same variance for every element, and a
1D array or list gives one for each. A :class:`pandas.Series` or a
:class:`~fips.Vector` brings its own index.

.. ipython:: python

   import pandas as pd, numpy as np
   from fips import CovarianceMatrix

   idx = pd.MultiIndex.from_product([["obs"], ["a", "b", "c"]], names=["block", "id"])

   # Diagonal: independent errors with the same variance
   CovarianceMatrix.diagonal(0.25, index=idx).values

   # Diagonal: a variance for each element
   CovarianceMatrix.diagonal(pd.Series([0.5, 0.5, 1.0], index=idx)).values

   # Full dense matrix
   S = CovarianceMatrix(pd.DataFrame(np.eye(3) * 0.25, index=idx, columns=idx))
   S.shape

With ``sparse=True`` the diagonal is stored sparse and the dense matrix is never
formed, which matters once the index is long:

.. ipython:: python

   CovarianceMatrix.diagonal(0.25, index=idx, sparse=True).is_sparse

:class:`~fips.Matrix`, :class:`~fips.ForwardOperator` and
:class:`~fips.MatrixBlock` have the same ``diagonal`` constructor; a block also
takes its ``row_block`` and ``col_block``. The constructors themselves take 2D
data only: a scalar or 1D input raises an error that points to ``diagonal``.

An :class:`~fips.InverseProblem` holds the indexes already, so its
``prior_error`` and ``modeldata_mismatch`` can be given as a scalar variance, or
as a Series or Vector of variances with a ``"block"`` level.


Sparse Matrices
---------------

For large transport Jacobians (e.g. from STILT) with many structural zeros,
sparse storage gives significant memory and speed savings.  Pass
``sparse=True`` to :class:`~fips.MatrixBlock` at construction time, or convert
an existing operator:

.. note::

   The forward operator :math:`H` is also called the Jacobian or observation operator.
   See :ref:`inverse-problem-terms` for details.

.. code-block:: python

   from fips import MatrixBlock

   # Sparse at construction (recommended for large Jacobians)
   H_block = MatrixBlock(df, row_block="obs", col_block="flux", sparse=True)

   # Or convert an existing operator in-place
   H_sparse = problem.forward_operator.to_sparse(threshold=1e-10)

The same holds for the covariances: the forward operator, the prior error and
the model-data mismatch can each be dense or sparse, in any combination. The
solve computes with each matrix's ``operator``
(:attr:`~fips.base.Structure2D.operator`), which holds a sparse matrix as a
:mod:`scipy.sparse` matrix, never as the dense one, and a sparse matrix whose
entries are all on its diagonal as that diagonal alone
(:mod:`fips.operator`):

.. ipython:: python

   S = CovarianceMatrix.diagonal([0.5, 1.0, 2.0], index=idx, sparse=True)
   S.operator
   S.operator.solve(np.ones(3))

A product keeps the cheapest exact form: a diagonal times a sparse matrix is
sparse, and anything times a dense matrix is dense. :math:`H S_0 H^T` stays
sparse while it is mostly zeros, and is dense once more than a quarter of its
entries are filled, since a dense solve is then faster. Solves divide by a
diagonal, factor a sparse matrix with a sparse LU, and a dense one with
Cholesky. The outputs are dense whatever the inputs: the posterior covariance
is a dense :math:`n \times n` matrix, and the gain :math:`n \times m`.


.. _flux-inversion:

Domain-Specific Problem: Flux Inversion
-----------------------------------------

:class:`~fips.problems.flux.FluxProblem` is a ready-made subclass of
:class:`~fips.InverseProblem` for atmospheric flux estimation using STILT
transport footprints.  It adds named accessors that speak the language of the
application:

.. code-block:: python

   import pandas as pd
   from fips.problems.flux import FluxProblem

   inversion = FluxProblem(
       obs=concentrations,        # pd.Series  — measured concentrations
       prior=prior_fluxes,        # pd.Series  — prior flux inventory
       forward_operator=jacobian, # pd.DataFrame — transport Jacobian
       modeldata_mismatch=S_z,    # concentration error covariance
       prior_error=S_0,           # flux error covariance
   )
   inversion.solve()

   # Domain-aware accessors
   inversion.concentrations        # observed concentrations
   inversion.prior_fluxes          # prior flux pd.Series
   inversion.posterior_fluxes      # posterior flux pd.Series
   inversion.posterior_flux_error  # posterior flux error pd.DataFrame
   inversion.prior_concentrations  # H @ prior
   inversion.posterior_concentrations  # H @ posterior

   # Built-in plots (requires cartopy)
   inversion.plot.fluxes()
   inversion.plot.concentrations()

For a STILT Jacobian, `PYSTILT <https://jmineau.github.io/PYSTILT>`_ sums the
footprints onto your flux grid and time bins. Wrap its matrix in a
:class:`~fips.MatrixBlock`:

.. code-block:: python

   import stilt
   from fips import MatrixBlock

   project = stilt.Project("my_project")
   sims = project.simulations
   H = project.jacobian(sims[sims.variant == "hrrr"], grid, time_bins)
   jacobian = MatrixBlock(
       H.to_frame(sparse=True), row_block="concentration", col_block="flux", sparse=True
   )

Its rows are receptor ids and its columns ``(time, lon, lat)``, so label the
observations and the prior to match.

Background subtraction is supported via the ``constant`` argument to
:class:`~fips.InverseProblem`:

.. code-block:: python

   inversion = FluxProblem(
       ...
       constant=background_series,  # subtracted from obs before inversion
   )
   inversion.background             # returns the background pd.Series


Advanced: Subclassing InverseProblem
-------------------------------------

For full control, subclass :class:`~fips.InverseProblem` directly to add
domain-specific accessors, pre- or post-processing, or alternative solve
strategies:

.. code-block:: python

   from fips import InverseProblem

   class GravityInversion(InverseProblem):
       """Bayesian inversion for subsurface density from gravity data."""

       def solve(self, estimator="bayesian", **kwargs):
           return super().solve(estimator=estimator, **kwargs)

       @property
       def density(self):
           """Posterior density anomaly [g cm⁻³]."""
           return self.posterior["density"]

       @property
       def gravity_residual(self):
           """Observed minus prior-modelled gravity."""
           return self.obs["gravity"] - self.prior_obs["gravity"]

All :class:`~fips.InverseProblem` results are lazily evaluated so you only pay
for what you access.


Advanced: Inversion Pipelines
------------------------------

:class:`~fips.pipeline.InversionPipeline` provides a structured, reproducible
workflow.  Subclass it, implement the abstract hooks, then call
:meth:`~fips.pipeline.InversionPipeline.run`:

.. code-block:: python

   from fips.pipeline import InversionPipeline
   from fips.covariance import CovarianceMatrix
   from fips.operators import ForwardOperator
   from fips.vector import Vector

   class MyPipeline(InversionPipeline):

       def __init__(self, config):
           super().__init__(
               config=config,
               problem=MyInversion,
               estimator="bayesian",
           )

       # --- Required hooks ---

       def get_obs(self) -> Vector:
           """Load observations from config / disk / API."""
           ...

       def get_prior(self) -> Vector:
           """Load prior from inventory / model output."""
           ...

       def get_forward_operator(self, obs, prior) -> ForwardOperator:
           """Build or load the Jacobian."""
           ...

       def get_prior_error(self, prior) -> CovarianceMatrix:
           ...

       def get_modeldata_mismatch(self, obs) -> CovarianceMatrix:
           ...

   solved = MyPipeline(config).run()

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
