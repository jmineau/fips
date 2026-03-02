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

The three building blocks are:

- :class:`~fips.Block` / :class:`~fips.Vector` — labelled 1-D state and observation vectors
- :class:`~fips.MatrixBlock` / :class:`~fips.Matrix` — labelled 2-D operators and covariances
- :class:`~fips.InverseProblem` — wires everything together and runs the estimator

.. note::

   Plain :class:`pandas.Series` and :class:`pandas.DataFrame` objects are accepted
   everywhere. *fips* promotes them to :class:`~fips.Block` / :class:`~fips.MatrixBlock`
   automatically, so you can start with data you already have.


Minimal Example
---------------

.. code-block:: python

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

   print(problem.posterior)       # posterior state as a labelled Vector
   print(problem.posterior_error) # posterior error covariance as a labelled Matrix


Inspecting the Solution
-----------------------

After calling :meth:`~fips.InverseProblem.solve`, results are available as
labelled *fips* objects that behave like pandas Series / DataFrames:

.. code-block:: python

   # Posterior state and uncertainty
   x_hat = problem.posterior        # Vector  (index = state_idx)
   S_hat = problem.posterior_error  # CovarianceMatrix

   # Modelled observations for prior and posterior
   y_0   = problem.prior_obs        # Vector  (index = obs_idx)
   y_hat = problem.posterior_obs    # Vector  (index = obs_idx)

   # Kalman gain and averaging kernel
   K = problem.kalman_gain          # Matrix  (obs × state)
   A = problem.averaging_kernel     # Matrix  (state × state)

   # Scalar diagnostics from the estimator
   est = problem.estimator
   print(f"DOFS:       {est.DOFS:.2f}")
   print(f"χ²:         {est.chi2:.3f}")
   print(f"R²:         {est.R2:.3f}")
   print(f"RMSE:       {est.RMSE:.4f}")


Working with Blocks and MatrixBlocks
-------------------------------------

Use :class:`~fips.Block` and :class:`~fips.MatrixBlock` explicitly when your
state space is composed of **named components** — for example, fluxes and a
per-site bias correction:

.. code-block:: python

   import pandas as pd
   from fips import Block, Vector, MatrixBlock, ForwardOperator, CovarianceMatrix, InverseProblem

   # --- Two state components ---
   flux_idx = pd.Index(["cell_0", "cell_1"], name="cell_id")
   bias_idx = pd.Index(["site_A"],           name="site_id")
   obs_idx  = pd.Index(["o0", "o1"],         name="obs_id")

   flux_block = Block(pd.Series([1.0, 2.0], index=flux_idx, name="flux"))
   bias_block = Block(pd.Series([0.1],      index=bias_idx, name="bias"))
   prior = Vector([flux_block, bias_block])

   # --- Matching forward-operator sub-blocks ---
   H_flux = MatrixBlock(
       pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=obs_idx, columns=flux_idx),
       row_block="obs",
       col_block="flux",
   )
   H_bias = MatrixBlock(
       pd.DataFrame([[1.0], [1.0]], index=obs_idx, columns=bias_idx),
       row_block="obs",
       col_block="bias",
   )
   H = ForwardOperator([H_flux, H_bias])

   # Covariances follow the same pattern …
   # Then pass to InverseProblem as before.


What's Next
-----------

- **Domain-specific problems** — see :ref:`flux-inversion` in the User Guide for
  the ready-made :class:`~fips.problems.flux.FluxProblem` subclass and its
  :class:`~fips.problems.flux.pipeline.FluxInversionPipeline`.
- **Subclassing** — learn how to build your own :class:`~fips.InverseProblem`
   and :class:`~fips.pipeline.InversionPipeline` subclasses in :doc:`usage`.
- **Full API** — every class and method is documented in :doc:`reference/index`.
