Flux Inversion
==============

FluxProblem
-----------

.. currentmodule:: fips.problems.flux

.. autoclass:: FluxProblem
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~FluxProblem.prior
      ~FluxProblem.prior_cov
      ~FluxProblem.obs
      ~FluxProblem.obs_cov
      ~FluxProblem.forward_op
      ~FluxProblem.spatial_index
      ~FluxProblem.temporal_index

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~FluxProblem.solve
      ~FluxProblem.add_estimator
      ~FluxProblem.available_estimators

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: prior
   .. autoattribute:: prior_cov
   .. autoattribute:: obs
   .. autoattribute:: obs_cov
   .. autoattribute:: forward_op
   .. autoattribute:: spatial_index
   .. autoattribute:: temporal_index
   .. automethod:: solve
   .. automethod:: add_estimator
   .. automethod:: available_estimators


FluxInversionPipeline
---------------------

.. currentmodule:: fips.problems.flux.pipeline

.. autoclass:: FluxInversionPipeline
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~FluxInversionPipeline.problem
      ~FluxInversionPipeline.estimator
      ~FluxInversionPipeline.filters
      ~FluxInversionPipeline.operators
      ~FluxInversionPipeline.aggregators

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~FluxInversionPipeline.add_filter
      ~FluxInversionPipeline.add_operator
      ~FluxInversionPipeline.add_aggregator
      ~FluxInversionPipeline.run
      ~FluxInversionPipeline.run_async

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: problem
   .. autoattribute:: estimator
   .. autoattribute:: filters
   .. autoattribute:: operators
   .. autoattribute:: aggregators
   .. automethod:: add_filter
   .. automethod:: add_operator
   .. automethod:: add_aggregator
   .. automethod:: run
   .. automethod:: run_async


FluxPlotter
-----------

.. currentmodule:: fips.problems.flux.visualization

.. autoclass:: FluxPlotter

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~FluxPlotter.plot_spatial
      ~FluxPlotter.plot_temporal
      ~FluxPlotter.plot_comparison
      ~FluxPlotter.plot_uncertainty

   .. rubric:: Details

   .. automethod:: plot_spatial
   .. automethod:: plot_temporal
   .. automethod:: plot_comparison
   .. automethod:: plot_uncertainty
