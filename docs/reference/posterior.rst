=========
Posterior
=========

.. currentmodule:: fips

:func:`solve` fits an estimator to an :class:`InverseProblem` and returns a
:class:`Posterior`, which labels what the estimator computed with the
problem's state and observation indexes. :meth:`InverseProblem.solve` calls it
and keeps the posterior as :attr:`InverseProblem.result`.

.. seealso::

   See :doc:`../terminology` for the notation (:math:`\hat{x}`,
   :math:`\hat{S}`, :math:`K`, :math:`A`, DOFS).

Solving
~~~~~~~
.. autosummary::
   :toctree: api/

   solve

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior

State estimate
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.estimate
   Posterior.mean
   Posterior.covariance
   Posterior.variances
   Posterior.cov

Observation space
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.modeled_obs
   Posterior.prior_modeled_obs
   Posterior.prior_obs_covariance
   Posterior.desroziers

Operators
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.kalman_gain
   Posterior.averaging_kernel

Diagnostics
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.dofs
   Posterior.reduced_chi2
   Posterior.r2
   Posterior.rmse
   Posterior.uncertainty_reduction

Regularization
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.with_gamma

Inputs
~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.estimator
   Posterior.state_index
   Posterior.obs_index
   Posterior.state_space
   Posterior.obs_space

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.from_file
   Posterior.to_file
