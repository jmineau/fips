=====================
Solving and Posterior
=====================

.. currentmodule:: fips

:func:`solve` solves a :class:`Problem` by a method and returns a
:class:`Posterior`. The problem is the question; the posterior is the answer,
and holds everything that depends on the solve. Each method declares its
options once, as a dataclass (below), and :func:`solve` checks the options it
is given against them: an unknown method or option raises with the names it
knows. :doc:`/user_guide/methods` says which method to use, and
:doc:`/user_guide/posterior` how to read the answer.

A posterior labels what its method computed: the method keeps what it needs
from the solve (a :class:`~fips.posterior.Solution`), answers with unlabeled
arrays, and the posterior puts the problem's labels on them and derives the
rest (the averaging kernel, the innovation covariance, the diagnostics, the
summary). So every method's posterior has the same members, and the table
below says which each kind of answer can give.

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

.. currentmodule:: fips.methods.bounded

.. autosummary::
   :toctree: api/

   BoundedOptions

.. currentmodule:: fips.methods.gauss_newton

.. autosummary::
   :toctree: api/

   GaussNewtonOptions

.. currentmodule:: fips.methods.ensemble

.. autosummary::
   :toctree: api/

   EnsembleOptions

.. currentmodule:: fips.methods.variational

.. autosummary::
   :toctree: api/

   VariationalOptions

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
   Posterior.sample
   Posterior.at_bound

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
   Posterior.dofs_per_element
   Posterior.information_content
   Posterior.chi2
   Posterior.data_misfit
   Posterior.state_misfit
   Posterior.reduced_chi2
   Posterior.log_evidence
   Posterior.chi2_tests
   Posterior.r2
   Posterior.rmse
   Posterior.uncertainty_reduction
   Posterior.summary
   Posterior.iterations
   Posterior.eigenvalues

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
and :meth:`Posterior.with_data`, without sampling. :meth:`Posterior.trials` is that loop kept trial by
trial, for any world (a :class:`Problem` or a function) and any statistic.

.. autosummary::
   :toctree: api/

   Posterior.recovery
   Posterior.trials

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

.. _posterior-members:

What a posterior can give
~~~~~~~~~~~~~~~~~~~~~~~~~
A posterior's members belong to its kind of answer, not to its method. There
are three kinds: a *Gaussian* posterior (``estimate`` is its mean and the
covariance is known), a *sampled* one (``estimate`` is the mode, and the mean,
the spread and the intervals come from kept draws) and a *low-rank* one (the
covariance is kept as ``rank`` directions). A member a kind cannot give raises
:class:`UnavailableError`, which is both an :class:`AttributeError` and a
:class:`TypeError`, so ``hasattr(post, "at_bound")`` is False where it would
raise.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 16 28 28 28

   * - Member
     - Gaussian
     - Sampled
     - Low-rank
   * - Methods
     - ``"analytic"``; ``"ensemble"``; ``"variational"`` on a linear problem
       with ``rank`` at the problem's rank
     - ``"bounded"`` with a finite bound; ``"gauss_newton"`` with a forward
       model
     - ``"variational"`` with ``rank`` below the problem's rank, or with a
       forward model (the Gaussian at the mode)
   * - ``estimate``
     - the mean
     - the mode
     - the mean (linear problem), the mode (forward model)
   * - ``variances``, ``cov``
     - exact (an ensemble's: those of its members)
     - those of the kept draws (a Monte Carlo estimate)
     - the prior's less ``rank`` pairs: too large below the problem's rank
   * - ``mean``
     - yes
     - the kept draws' mean
     - yes (linear problem); raises (forward model)
   * - ``lower``, ``upper`` (``uncertainty``, ``trials``)
     - the estimate plus or minus ``z`` standard deviations
     - the kept draws' quantiles
     - the estimate plus or minus ``z`` standard deviations
   * - ``prior_std``, ``reduction`` (``uncertainty``),
       ``uncertainty_reduction``
     - against :math:`S_0`; ``"ensemble"``: against its prior members
     - ``"bounded"``: against draws of the prior with the bounds;
       ``"gauss_newton"``: against :math:`S_0`
     - against :math:`S_0`
   * - ``with_gamma``
     - yes
     - ``"bounded"``: yes; ``"gauss_newton"``: raises (its linearization
       depends on gamma)
     - yes (linear problem); raises (forward model)
   * - ``log_evidence``
     - yes; ``"ensemble"``: raises
     - ``"gauss_newton"``: Laplace's; ``"bounded"``: raises
     - yes (Laplace's with a forward model) unless ``rank`` left out pairs
       that still hold information
   * - ``recovery``, ``data_worth``, ``candidate_worth``
     - ``"analytic"`` only: they need an estimate linear in the observations
     - raises
     - raises
   * - ``at_bound``
     - raises
     - ``"bounded"`` with a finite bound
     - raises
   * - ``iterations``
     - ``"variational"`` only
     - ``"gauss_newton"`` only
     - yes
   * - ``eigenvalues``
     - ``"variational"`` only
     - raises
     - yes

``"bounded"`` with no finite bound and ``"gauss_newton"`` with a linear
operator return the analytic solution itself, so they are Gaussian. Every
posterior has ``estimate``, ``variances``, ``cov``, ``uncertainty``,
``sample``, ``with_data``, ``trials``, ``kalman_gain``, ``chi2_tests`` and
``summary``; what needs the forward model as an operator (the averaging kernel,
``innovation_cov``, ``dofs_per_element``, ``leverage``) raises for the
ensemble with a model, and ``"gauss_newton"`` reads the Jacobian of its last
step for them. This table is the one in the :class:`Posterior` docstring.

.. autosummary::
   :toctree: api/

   UnavailableError

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
   Solution.dofs_per_element
   Solution.information
   Solution.chi2
   Solution.misfits
   Solution.chi2_tests
   Options
   Options.gamma
