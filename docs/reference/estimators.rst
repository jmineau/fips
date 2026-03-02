==========
Estimators
==========

.. currentmodule:: fips

Estimator (Base Class)
======================

Inputs
~~~~~~
.. autosummary::
   :toctree: api/

   Estimator.n_z
   Estimator.n_x
   Estimator.z
   Estimator.x_0
   Estimator.H
   Estimator.S_0
   Estimator.S_z
   Estimator.c

Methods
~~~~~~~
.. autosummary::
   :toctree: api/

   Estimator.cost
   Estimator.forward
   Estimator.residual
   Estimator.leverage

Results
~~~~~~~
.. autosummary::
   :toctree: api/

   Estimator.x_hat
   Estimator.S_hat
   Estimator.y_hat
   Estimator.y_0
   Estimator.K
   Estimator.A

Metrics
~~~~~~~
.. autosummary::
   :toctree: api/

   Estimator.reduced_chi2
   Estimator.R2
   Estimator.RMSE
   Estimator.uncertainty_reduction
   Estimator.U_red


BayesianSolver
==============

.. currentmodule:: fips.estimators

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   BayesianSolver

Methods
~~~~~~~
.. autosummary::
   :toctree: api/

   BayesianSolver.cost

Results
~~~~~~~
.. autosummary::
   :toctree: api/

   BayesianSolver.x_hat
   BayesianSolver.S_hat
