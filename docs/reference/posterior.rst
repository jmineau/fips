=====================
Solving and Posterior
=====================

.. currentmodule:: fips

:func:`solve` solves a :class:`Problem` with a method and returns a
:class:`Posterior`. The problem is the question and holds what needs no solve
and no :math:`m \times m` matrix (:meth:`Problem.predict`); the posterior is
the answer and holds what depends on the solve or on the :math:`m \times m`
matrix it factors (:attr:`Posterior.innovation_cov`). Each method declares its options once (for ``"analytic"``,
:class:`~fips.methods.analytic.AnalyticOptions`), and an unknown method or option raises
with the names it knows. The posterior labels the answer and the method keeps
its own machinery: the posterior asks what the method kept from the solve (a
:class:`~fips.posterior.Solution`) for unlabeled arrays and labels them with
the problem's spaces. The analytic method keeps the factorization of its
whitened system (:math:`m \times m`), or of its information system
(:math:`n \times n`) when there are fewer state elements than observations
and the model-data mismatch is diagonal, sparse or block diagonal. In the
whitened form nothing :math:`n \times n` is formed unless the full covariance
or averaging kernel is asked for, and :meth:`Posterior.with_gamma` and
:meth:`Posterior.with_data` reuse the factorization in either form.

.. seealso::

   See :doc:`../terminology` for the notation (:math:`\hat{x}`,
   :math:`\hat{S}`, :math:`K`, :math:`A`, DOFS).

Solving
~~~~~~~
.. autosummary::
   :toctree: api/

   solve
   available_methods

.. currentmodule:: fips.methods.analytic

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
   Posterior.summary

Reusing the factorization
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Posterior.with_gamma
   Posterior.with_data

Serialization
~~~~~~~~~~~~~
A posterior pickles with the standard library (``pickle.dump(post, file)``):
it keeps the problem, the options and the method's solution (for the analytic
method, the factorization), and rebuilds every output when it is read.

Writing a method
~~~~~~~~~~~~~~~~
What a :class:`Posterior` asks of a method: its solution answers the
questions of :class:`~fips.posterior.Solution`, and its options have the
regularization factor (:class:`~fips.posterior.Options`). How a method is
added is in the docstring of :mod:`fips.methods`.

.. currentmodule:: fips.posterior

.. autosummary::
   :toctree: api/

   Solution
   Solution.estimate
   Solution.variances
   Solution.cov
   Solution.gain
   Solution.dofs
   Solution.chi2
   Options
   Options.gamma
