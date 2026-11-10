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

The priors
~~~~~~~~~~
Each has a log density and a map to unbounded coordinates (the log of a
positive setting, the logit of a bounded one), where the chain moves.

.. autosummary::
   :toctree: api/

   Prior
   LogNormal
   HalfNormal
   Uniform

Serialization
~~~~~~~~~~~~~
A result pickles when the function that builds the problem does (a function
defined at the top level of a module); the fast path's always does.
