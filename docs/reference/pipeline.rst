========
Pipeline
========

.. currentmodule:: fips.pipeline

InversionPipeline
=================

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InversionPipeline

Getting Pipeline Data
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InversionPipeline.get_obs
   InversionPipeline.get_prior
   InversionPipeline.get_forward_operator
   InversionPipeline.get_prior_error
   InversionPipeline.get_modeldata_mismatch
   InversionPipeline.get_constant

Filtering and Aggregating
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InversionPipeline.filter_state_space
   InversionPipeline.aggregate_obs_space

Running the Inversion
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InversionPipeline.get_inputs
   InversionPipeline.run
