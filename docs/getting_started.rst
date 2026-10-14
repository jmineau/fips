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

- :class:`~fips.Block` / :class:`~fips.Vector` — labelled 1-D state and observation vectors
- :class:`~fips.Operator` / :class:`~fips.Covariance` — labelled matrices: a
  :class:`~fips.Space` of labels on each axis, the numbers kept in their
  cheapest form (dense, sparse, diagonal, Kronecker factors)
- :class:`~fips.InverseProblem` — wires everything together and runs the estimator

.. note::

   Plain :class:`pandas.Series` and :class:`pandas.DataFrame` objects are accepted
   by :class:`~fips.InverseProblem`: it reads their labels, so you can start with
   data you already have.


Minimal Example
---------------

.. ipython:: python

   import numpy as np
   import pandas as pd
   from fips import InverseProblem

   # --- Indices ---
   state_idx = pd.Index(["s0", "s1", "s2"], name="state_id")
   obs_idx   = pd.Index(["o0", "o1"],        name="obs_id")

   # --- Prior state vector ---
   prior = pd.Series([1.0, 2.0, 3.0], index=state_idx, name="state")

   # --- Observations ---
   obs = pd.Series([1.8, 4.3], index=obs_idx, name="obs")

   # --- Forward operator H (maps state → obs space) ---
   H = pd.DataFrame(
       [[1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0]],
       index=obs_idx,
       columns=state_idx,
   )

   # --- Error covariances ---
   S_0 = pd.DataFrame(np.diag([1.0, 1.0, 1.0]), index=state_idx, columns=state_idx)
   S_z = pd.DataFrame(np.diag([0.1, 0.1]),       index=obs_idx,   columns=obs_idx)

   # --- Solve ---
   problem = InverseProblem(
       obs=obs,
       prior=prior,
       forward_operator=H,
       modeldata_mismatch=S_z,
       prior_error=S_0,
   )
   problem.solve("bayesian")

   print(problem.posterior)                   # posterior state as a labelled Vector
   print(problem.posterior_error.to_frame())  # posterior error covariance, labelled


Inspecting the Solution
-----------------------

After calling :meth:`~fips.InverseProblem.solve`, results are available as
labelled *fips* objects: vectors that hold a pandas Series, and operators
whose ``to_frame()`` is a DataFrame:

.. ipython:: python

   # Posterior state and uncertainty
   x_hat = problem.posterior        # Vector  (index = state_idx)
   S_hat = problem.posterior_error  # Covariance (state × state)

   # Modelled observations for prior and posterior
   y_0   = problem.prior_obs        # Vector  (index = obs_idx)
   y_hat = problem.posterior_obs    # Vector  (index = obs_idx)
   S_y0  = problem.prior_obs_error  # Covariance: H S_0 H^T, the prior's error on y_0

   # Kalman gain and averaging kernel
   K = problem.kalman_gain          # Operator  (state × obs)
   A = problem.averaging_kernel     # Operator  (state × state)

   # Scalar diagnostics from the estimator
   est = problem.estimator
   print(f"DOFS:       {est.DOFS:.2f}")
   print(f"reduced χ²: {est.reduced_chi2:.3f}")
   print(f"R²:         {est.R2:.3f}")
   print(f"RMSE:       {est.RMSE:.4f}")

The same results come as one object from :func:`fips.solve`, which returns a
:class:`~fips.Posterior` and leaves the problem as it was.
:meth:`~fips.InverseProblem.solve` calls it and keeps the posterior as
``problem.result``:

.. ipython:: python

   import fips

   post = fips.solve(problem)       # estimator="bayesian" by default
   post.estimate                    # the posterior state, as problem.posterior
   post.variances                   # the diagonal of post.covariance
   print(f"DOFS: {post.dofs:.2f}, reduced χ²: {post.reduced_chi2:.3f}")


Working with Blocks
-------------------

Use named blocks when your state space is composed of **named components** —
for example, fluxes and a per-site bias correction:

.. ipython:: python

   import pandas as pd
   from fips import Block, Covariance, Operator, Space, Vector

   # --- Two state components ---
   flux_idx = pd.Index(["cell_0", "cell_1"], name="cell_id")
   bias_idx = pd.Index(["site_A"],           name="site_id")
   obs_idx  = pd.Index(["o0", "o1"],         name="obs_id")

   flux_block = Block(pd.Series([1.0, 2.0], index=flux_idx, name="flux"))
   bias_block = Block(pd.Series([0.1],      index=bias_idx, name="bias"))
   prior = Vector([flux_block, bias_block], name="prior")
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
everything to :class:`~fips.InverseProblem` as before:

.. ipython:: python

   S_0 = Covariance.block_diag(
       Covariance.diagonal(1.0, Space({"flux": flux_idx})),
       Covariance.diagonal(0.01, Space({"bias": bias_idx})),
   )
   S_0.variances.to_series()


What's Next
-----------

- **Domain-specific problems** — see :ref:`flux-inversion` in the User Guide for
  the ready-made :class:`~fips.problems.flux.FluxProblem` subclass and its
  :class:`~fips.problems.flux.pipeline.FluxInversionPipeline`.
- **Subclassing** — learn how to build your own :class:`~fips.InverseProblem`
   and :class:`~fips.pipeline.InversionPipeline` subclasses in :doc:`usage`.
- **Full API** — every class and method is documented in :doc:`reference/index`.
