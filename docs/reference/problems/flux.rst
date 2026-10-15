==============
Flux Inversion
==============

.. currentmodule:: fips.problems.flux

FluxProblem
===========

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxProblem

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxProblem.prior
   FluxProblem.obs
   FluxProblem.result

Methods
~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxProblem.solve


FluxInversionPipeline
=====================

.. currentmodule:: fips.problems.flux.pipeline

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxInversionPipeline

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxInversionPipeline.problem
   FluxInversionPipeline.method

Methods
~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxInversionPipeline.filter_state_space
   FluxInversionPipeline.run
   FluxInversionPipeline.summarize


FluxPlotter
===========

.. currentmodule:: fips.problems.flux.visualization

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxPlotter

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxPlotter.fluxes
   FluxPlotter.concentrations
