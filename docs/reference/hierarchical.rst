===========================
The settings' uncertainty
===========================

.. currentmodule:: fips

:func:`hierarchical` gives the covariance settings priors, draws them from
their posterior with the exact evidence of the problem at each
(:attr:`Posterior.log_evidence`), and draws the state at each kept draw
(:meth:`Posterior.sample`): a :class:`Hierarchical` holds both, and the
state's draws are those of the posterior mixed over the settings. Given a
:class:`Problem` instead of a function, with settings named
``"prior_scale"`` and ``"obs_scale"`` (or ``"gamma"``), it takes the fast
path: one eigendecomposition for the whole run. The guide is
:doc:`/user_guide/hierarchical`.

Sampling the settings
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   hierarchical

The result
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Hierarchical
   Hierarchical.settings
   Hierarchical.mode
   Hierarchical.acceptance
   Hierarchical.ess
   Hierarchical.estimate
   Hierarchical.variances
   Hierarchical.cov
   Hierarchical.uncertainty
   Hierarchical.sample
   Hierarchical.at

The priors of the settings
~~~~~~~~~~~~~~~~~~~~~~~~~~
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
   HalfNormal.scale
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
A result pickles when the function that builds the problem does (a function
defined at the top level of a module); the fast path's always does.
