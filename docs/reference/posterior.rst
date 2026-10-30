=====================
Solving and Posterior
=====================

.. currentmodule:: fips

:func:`solve` solves a :class:`Problem` with a method and returns a
:class:`Posterior`. The problem is the question and holds what needs no solve
and no :math:`m \times m` matrix (:meth:`Problem.predict`); the posterior is
the answer and holds what depends on the solve or on the :math:`m \times m`
innovation covariance (:attr:`Posterior.innovation_cov`). Each method declares its options once (for ``"analytic"``,
:class:`~fips.methods.analytic.AnalyticOptions`), and an unknown method or option raises
with the names it knows. The posterior labels the answer and the method keeps
its own machinery: the posterior asks what the method kept from the solve (a
:class:`~fips.posterior.Solution`) for unlabeled arrays and labels them with
the problem's spaces. The analytic method keeps the factorization of its
whitened system (:math:`m \times m`), or of its information system
(:math:`n \times n`) when there are fewer state elements than observations
and the model-data mismatch is dense, diagonal, sparse, block diagonal or
Kronecker. In the whitened form nothing :math:`n \times n` is formed unless
the full covariance or averaging kernel is asked for. The solve factors by
Cholesky at its gamma; the first :meth:`Posterior.with_gamma` eigendecomposes
once, and later ones reuse it. :meth:`Posterior.with_data` reuses the
factorization in either form.

.. seealso::

   See :doc:`/getting_started/terminology` for the notation (:math:`\hat{x}`,
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
   Posterior.uncertainty

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

Recovering a known truth
~~~~~~~~~~~~~~~~~~~~~~~~
The exact mean and spread of the estimate's error when the data come from a
world (the problem itself, a fixed truth, or another :class:`Problem` on the
same labels): the answer of a Monte Carlo loop over :meth:`Problem.simulate`
and :meth:`Posterior.with_data`, without sampling.

.. autosummary::
   :toctree: api/

   Posterior.recovery

What groups of observations add
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each group of observations (a site, a platform, a month), in the problem
or a candidate not yet in it, the posterior uncertainty of aggregates of the
state without the group and with it, and the group's share of DOFS: a rank
update of the posterior covariance per group, without solving again and
without an :math:`n \times n` matrix.

.. autosummary::
   :toctree: api/

   Posterior.data_worth
   Posterior.candidate_worth

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
