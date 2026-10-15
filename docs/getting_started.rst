Getting Started
===============

*fips* solves the standard Bayesian linear inverse problem:

.. math::

   \hat{x} = x_0 + K(z - Hx_0), \qquad \hat{S} = S_0 - K H S_0

where :math:`x_0` is the prior state, :math:`z` the observations, :math:`H` the
forward (Jacobian) operator, :math:`S_0` the prior-error covariance, and
:math:`S_z` the observation-error covariance.  The Kalman gain is

.. math::

   K = S_0 H^T (H S_0 H^T + S_z)^{-1}.

.. tip::

   See :doc:`terminology` for a complete guide to mathematical notation and abbreviations
   used throughout FIPS. Different fields use different conventions — this reference
   helps clarify the terminology.

The building blocks are:

- :class:`~fips.Vector` — labelled 1-D state and observation vectors, on a
  :class:`~fips.Space` of named blocks
- :class:`~fips.Operator` / :class:`~fips.Covariance` — labelled matrices: a
  :class:`~fips.Space` of labels on each axis, the numbers kept in their
  cheapest form (dense, sparse, diagonal, Kronecker factors)
- :class:`~fips.Problem` — puts everything together, its labels checked once
- :func:`fips.solve` — solves a problem and returns its :class:`~fips.Posterior`

.. note::

   pandas objects you already have come in through constructors that read
   their labels: :meth:`Vector.from_series <fips.Vector.from_series>` and
   :meth:`Operator.from_frame <fips.Operator.from_frame>`.


Minimal Example
---------------

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips
   from fips import Covariance, Operator, Problem, Vector

   # --- Indices ---
   state_idx = pd.Index(["s0", "s1", "s2"], name="state_id")
   obs_idx   = pd.Index(["o0", "o1"],        name="obs_id")

   # --- Prior state vector, one block named "state" ---
   prior = Vector.from_series(pd.Series([1.0, 2.0, 3.0], index=state_idx), block="state")

   # --- Observations ---
   obs = Vector.from_series(pd.Series([1.8, 4.3], index=obs_idx), block="obs")

   # --- Forward operator H (maps state → obs space) ---
   H = Operator(
       np.array([[1.0, 0.5, 0.0],
                 [0.0, 0.5, 1.0]]),
       obs.space,
       prior.space,
   )

   # --- Error covariances ---
   S_0 = Covariance.diagonal([1.0, 1.0, 1.0], prior.space)
   S_z = Covariance.diagonal([0.1, 0.1], obs.space)

   # --- Solve ---
   problem = Problem(prior=prior, prior_cov=S_0, obs=obs, obs_cov=S_z, forward=H)
   post = fips.solve(problem)

   print(post.estimate)               # posterior state as a labelled Vector
   print(post.cov().to_frame())       # posterior error covariance, labelled


Inspecting the Solution
-----------------------

:func:`fips.solve` returns a :class:`~fips.Posterior` and leaves the problem
as it was. Its results are labelled *fips* objects: vectors whose
``to_series()`` is a pandas Series, and operators whose ``to_frame()`` is a
DataFrame:

.. ipython:: python

   # Posterior state and uncertainty
   x_hat = post.estimate            # Vector  (on prior.space)
   S_hat = post.cov()               # Covariance (state × state)
   v_hat = post.variances           # its diagonal, without forming it

   # Modelled observations for prior and posterior
   y_0   = post.prior_modeled_obs   # Vector  (on obs.space)
   y_hat = post.modeled_obs         # Vector  (on obs.space)
   S_y0  = post.prior_obs_cov       # Covariance: H S_0 H^T, the prior's error on y_0

   # Kalman gain and averaging kernel
   K = post.kalman_gain             # Operator  (state × obs)
   A = post.averaging_kernel()      # Operator  (state × state)

.. ipython:: python

   # Scalar diagnostics
   print(f"DOFS:       {post.dofs:.2f}")
   print(f"reduced χ²: {post.reduced_chi2:.3f}")
   print(f"R²:         {post.r2:.3f}")
   print(f"RMSE:       {post.rmse:.4f}")

The method is ``"analytic"`` by default; its one option is the regularization
factor ``gamma``, and ``post.with_gamma(g)`` gives another gamma's posterior
from the same factorization:

.. ipython:: python

   fips.solve(problem, gamma=0.5).options
   post.with_gamma(0.5).dofs


Working with Blocks
-------------------

Use named blocks when your state space is composed of **named components** —
for example, fluxes and a per-site bias correction:

.. ipython:: python

   import pandas as pd
   from fips import Covariance, Operator, Space, Vector

   # --- Two state components ---
   flux_idx = pd.Index(["cell_0", "cell_1"], name="cell_id")
   bias_idx = pd.Index(["site_A"],           name="site_id")
   obs_idx  = pd.Index(["o0", "o1"],         name="obs_id")

   prior = Vector.from_blocks(
       {
           "flux": pd.Series([1.0, 2.0], index=flux_idx),
           "bias": pd.Series([0.1], index=bias_idx),
       },
       name="prior",
   )
   print(prior)

The forward operator is built from its blocks, each named by the observation
block it maps to and the state block it maps from:

.. ipython:: python

   H = Operator.from_blocks(
       {
           ("obs", "flux"): pd.DataFrame(
               [[1.0, 0.5], [0.5, 1.0]], index=obs_idx, columns=flux_idx
           ),
           ("obs", "bias"): pd.DataFrame([[1.0], [1.0]], index=obs_idx, columns=bias_idx),
       }
   )
   print(H.to_frame())

The covariances follow the same pattern; independent components go side by
side with :meth:`~fips.Covariance.block_diag`, each in its own form. Then pass
everything to :class:`~fips.Problem` as before:

.. ipython:: python

   S_0 = Covariance.block_diag(
       Covariance.diagonal(1.0, Space({"flux": flux_idx})),
       Covariance.diagonal(0.01, Space({"bias": bias_idx})),
   )
   S_0.variances.to_series()

Correlated errors are written as a correlation model with standard deviations
(:ref:`covariance-models` in the User Guide has them all). Here the flux errors
of neighboring cells are correlated over 1.5 cells, and the parts are named by
block and put on the prior's space with ``build`` (a :class:`~fips.Problem`
builds a model given as ``prior_cov`` on the prior's space itself):

.. ipython:: python

   from fips import Exponential, Independent

   cells = pd.Index([0.0, 1.0], name="cell_x")
   flux_space = Space({"flux": cells, "bias": bias_idx})
   model = Covariance.block_diag(
       flux=Exponential("cell_x", 1.5).with_std(1.0),
       bias=Independent().with_std(0.1),
   )
   model.build(flux_space).dense().round(3)


What's Next
-----------

- **Domain-specific problems** — see :ref:`flux-inversion` in the User Guide for
  the ready-made :class:`~fips.problems.flux.FluxProblem` subclass and its
  :class:`~fips.problems.flux.pipeline.FluxInversionPipeline`.
- **Pipelines** — learn how to build your own
  :class:`~fips.pipeline.InversionPipeline` subclasses in :doc:`usage`.
- **Full API** — every class and method is documented in :doc:`reference/index`.
