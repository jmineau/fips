===============
Inverse Problem
===============

.. currentmodule:: fips

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InverseProblem

Inputs
~~~~~~
.. autosummary::
   :toctree: api/

   InverseProblem.obs
   InverseProblem.prior
   InverseProblem.forward_operator
   InverseProblem.prior_error
   InverseProblem.modeldata_mismatch
   InverseProblem.constant

Solving the Inverse Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InverseProblem.estimator
   InverseProblem.solve

Results
~~~~~~~
.. autosummary::
   :toctree: api/

   InverseProblem.posterior
   InverseProblem.posterior_error
   InverseProblem.prior_obs
   InverseProblem.posterior_obs
   InverseProblem.kalman_gain
   InverseProblem.averaging_kernel

Selection and indexing
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InverseProblem.get_block

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   InverseProblem.from_file
   InverseProblem.to_file
