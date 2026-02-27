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
   FluxProblem.estimator

Methods
~~~~~~~
.. autosummary::
   :toctree: ../api/

   FluxProblem.solve
   FluxProblem.get_block


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
   FluxInversionPipeline.estimator

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


Transport Backends
==================

STILT
-----

.. currentmodule:: fips.problems.flux.transport.stilt

.. warning::
   The STILT transport backend is **experimental** and not production-ready.

   This module depends on the ``stilt`` package, which is:

   * Not available on PyPI
   * Only available on GitHub at https://github.com/jmineau/stilt
   * Subject to rapid changes and breaking updates
   * Not guaranteed to be stable or backwards-compatible

   Use this module at your own risk. The API may change without notice.

.. autosummary::
   :toctree: ../api/

   builder.JacobianBuilder
   footprint.load_footprint
   footprint.load_footprints
   simulation.get_sim
   simulation.load_simulation
