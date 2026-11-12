==============================
The covariance hyperparameters
==============================

.. currentmodule:: fips

:func:`hierarchical` gives the covariance hyperparameters priors, draws them from
their posterior with the exact evidence of the problem at each
(:attr:`Posterior.log_evidence`), and draws the state at each kept draw
(:meth:`Posterior.sample`): a :class:`Hierarchical` holds both, and the
state's draws are those of the posterior mixed over the hyperparameters. Given a
:class:`Problem` instead of a function, with hyperparameters named
``"prior_scale"`` and ``"obs_scale"`` (or ``"gamma"``), it takes the fast
path: one eigendecomposition for the whole run. The guide is
:doc:`/user_guide/hierarchical`.

:func:`sweep` solves the problem at every combination of a grid of the
hyperparameters' values instead: a :class:`Sweep` holds each member's
diagnostics, estimate and variances, and the mean and spread of the
estimates of the members kept. Its guide is
:ref:`hyperparameters-grid`.

Sampling the hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   hierarchical

The result
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Hierarchical
   Hierarchical.hyperparameters
   Hierarchical.mode
   Hierarchical.acceptance
   Hierarchical.ess
   Hierarchical.estimate
   Hierarchical.variances
   Hierarchical.cov
   Hierarchical.uncertainty
   Hierarchical.sample
   Hierarchical.at

A grid of hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   sweep
   Sweep
   Sweep.table
   Sweep.estimates
   Sweep.variances
   Sweep.uncertainty
   Sweep.at

The priors of the hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: fips.distributions
   :no-members:

.. currentmodule:: fips.distributions

.. autosummary::
   :toctree: api/

   LogNormal
   LogNormal.mu
   LogNormal.sigma
   LogNormal.median
   LogNormal.logpdf
   LogNormal.sample
   HalfNormal
   HalfNormal.sigma
   HalfNormal.median
   HalfNormal.logpdf
   HalfNormal.sample
   Uniform
   Uniform.low
   Uniform.high
   Uniform.median
   Uniform.logpdf
   Uniform.sample

.. currentmodule:: fips

Serialization
~~~~~~~~~~~~~
A result of either pickles when the function that builds the problem does (a
function defined at the top level of a module); one given a problem always does.
