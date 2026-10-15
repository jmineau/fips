=====================
Solving and Posterior
=====================

.. currentmodule:: fips

:func:`solve` solves a :class:`Problem` with a method and returns a
:class:`Posterior`. The problem is the question and holds what needs no solve
and no :math:`m \times m` matrix (:meth:`Problem.predict`); the posterior is
the answer and holds what depends on the solve or on the :math:`m \times m`
matrix it factors (:attr:`Posterior.innovation_cov`). Each method declares its options once (for ``"analytic"``,
:class:`~fips.solve.AnalyticOptions`), and an unknown method or option raises
with the names it knows. The posterior keeps the factorization of the solve:
everything it gives is derived from it and labeled with the problem's spaces,
nothing :math:`n \times n` is formed unless the full covariance or averaging
kernel is asked for, and :meth:`Posterior.with_gamma` and
:meth:`Posterior.with_data` reuse it.

.. seealso::

   See :doc:`../terminology` for the notation (:math:`\hat{x}`,
   :math:`\hat{S}`, :math:`K`, :math:`A`, DOFS).

Solving
~~~~~~~
.. autosummary::
   :toctree: api/

   solve
   available_methods

.. currentmodule:: fips.solve

.. autosummary::
   :toctree: api/

   AnalyticOptions

.. currentmodule:: fips

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior

Inputs
~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.problem
   Posterior.options

State estimate
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.estimate
   Posterior.mean
   Posterior.variances
   Posterior.cov

Observation space
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.modeled_obs
   Posterior.innovation_cov
   Posterior.desroziers
   Posterior.leverage

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
   Posterior.chi2
   Posterior.reduced_chi2
   Posterior.r2
   Posterior.rmse
   Posterior.uncertainty_reduction

Reusing the factorization
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.with_gamma
   Posterior.with_data

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.from_file
   Posterior.to_file
