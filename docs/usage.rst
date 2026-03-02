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
automatically.

.. tip::

   Plain :class:`pandas.Series` and :class:`pandas.DataFrame` are accepted
   everywhere — *fips* will promote them to :class:`~fips.Block` /
   :class:`~fips.MatrixBlock` automatically.


Constructing Covariance Matrices
---------------------------------

:class:`~fips.CovarianceMatrix` accepts a diagonal :class:`pandas.Series`, a full
:class:`pandas.DataFrame`, or a list of :class:`~fips.MatrixBlock` objects:

.. code-block:: python

   import pandas as pd, numpy as np
   from fips import CovarianceMatrix

   idx = pd.Index(["a", "b", "c"], name="id")

   # Diagonal — independent errors
   S = CovarianceMatrix(pd.Series([0.5, 0.5, 1.0], index=idx))

   # Full dense matrix
   S = CovarianceMatrix(
       pd.DataFrame(np.eye(3) * 0.25, index=idx, columns=idx)
   )


Sparse Forward Operators
------------------------

For large transport Jacobians (e.g. from STILT) with many structural zeros,
sparse storage gives significant memory and speed savings.  Pass
``sparse=True`` to :class:`~fips.MatrixBlock` at construction time, or convert
an existing operator:

.. code-block:: python

   from fips import MatrixBlock

   # Sparse at construction (recommended for large Jacobians)
   H_block = MatrixBlock(df, row_block="obs", col_block="flux", sparse=True)

   # Or convert an existing operator in-place
   H_sparse = problem.forward_operator.to_sparse(threshold=1e-10)

*fips* detects sparsity automatically at solve time and routes through
:mod:`scipy.sparse` for the matrix algebra.


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
       forward_operator=jacobian, # pd.DataFrame — STILT Jacobian
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
