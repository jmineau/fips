Terminology
===========

Each field that solves inverse problems has its own words and letters for the
same things. This page defines the words fips uses, once each, with the
symbol, the name in the code, and the other names you may know them by. The
rest of the documentation uses these words and no others.

.. _the-linear-model:

The linear model
----------------

fips solves the linear model

.. math::

   z = H x + c + \varepsilon

for the state :math:`x`, given the observations :math:`z`, the forward
operator :math:`H`, the offset :math:`c`, a prior :math:`x_0` with its prior
covariance :math:`S_0`, and the observation covariance :math:`S_z` (the
covariance of the error :math:`\varepsilon`). With :math:`n` state elements and
:math:`m` observations, the answer is the posterior

.. math::

   \hat{x} = x_0 + K (z - H x_0 - c), \qquad
   \hat{S} = S_0 - K H S_0, \qquad
   K = S_0 H^T \left(H S_0 H^T + S_z / \gamma\right)^{-1}.

The letters follow one convention: lowercase for vectors, uppercase for
matrices, a subscript :math:`0` for the prior, a subscript :math:`z` for the
observations, and a hat for the posterior.

.. _inverse-problem-terms:

The words
---------

.. list-table::
   :header-rows: 1
   :widths: 18 9 23 50

   * - Word
     - Symbol
     - In fips
     - What it is
   * - **state**
     - :math:`x`
     - the prior's labels (``problem.prior.space``)
     - The unknowns the problem solves for (fluxes, densities, slownesses),
       :math:`n` of them. Also called the control vector or the parameters.
   * - **observation**
     - :math:`z`
     - ``obs``
     - The measured data, :math:`m` of them. Also called the data or the
       measurement vector (Rodgers writes :math:`y`).
   * - **forward operator**
     - :math:`H`
     - ``forward``
     - The matrix that maps a state to the observations it would produce,
       :math:`m \times n`. Also called the Jacobian (of a nonlinear model,
       linearized), the observation operator, the sensitivity matrix or, in
       atmospheric transport, the footprints. :meth:`Problem.predict
       <fips.Problem.predict>` gives :math:`H x + c` for any state.
   * - **offset**
     - :math:`c`
     - ``offset``
     - The part of the observations the state does not explain, added to
       :math:`H x`: a background concentration, a known bias. Optional. An
       offset the data should decide is a block of the state instead
       (:func:`~fips.offset_block`).
   * - **prior**
     - :math:`x_0`
     - ``prior``
     - What is known of the state before the observations: an inventory, a
       climatology, a first guess. Also called the *a priori* or the
       background state.
   * - **prior covariance**
     - :math:`S_0`
     - ``prior_cov``
     - The covariance of the prior's error, :math:`n \times n`. Also called
       the prior error covariance (:math:`B` in data assimilation,
       :math:`S_a` in Rodgers, :math:`C_M` in Tarantola).
   * - **observation covariance**
     - :math:`S_z`
     - ``obs_cov``
     - The covariance of the error :math:`\varepsilon` between the
       observations and what the forward operator models, :math:`m \times m`:
       measurement error plus the errors of the model and of representing the
       state. Also called the model-data mismatch or the observation error
       covariance (:math:`R` in data assimilation, :math:`S_\epsilon` in
       Rodgers, :math:`C_D` in Tarantola).
   * - **innovation**
     - :math:`d`
     - ``post.innovation_cov``
     - The observations minus what the prior models,
       :math:`d = z - H x_0 - c`. Its covariance,
       :math:`H S_0 H^T + S_z / \gamma`, is the :math:`m \times m` matrix the
       solve factors.
   * - **posterior**
     - :math:`\hat{x}`, :math:`\hat{S}`
     - ``fips.solve(problem)``, a :class:`~fips.Posterior`
     - The answer: the state estimate :math:`\hat{x}` (``post.estimate``) and
       its posterior covariance :math:`\hat{S}` (``post.cov()``, whose
       diagonal is ``post.variances``). Also called the *a posteriori*, the
       analysis or the retrieval. ``post.modeled_obs`` is
       :math:`H \hat{x} + c`.
   * - **gain**
     - :math:`K`
     - ``post.kalman_gain``
     - The :math:`n \times m` matrix that weighs how much each observation
       moves the estimate (the Kalman gain, the contribution function).
   * - **averaging kernel**
     - :math:`A`
     - ``post.averaging_kernel()``
     - :math:`A = K H`, how the estimate responds to the true state: a
       diagonal near 1 is a state element the observations constrain, near 0
       one left at its prior.
   * - **DOFS**
     - :math:`\operatorname{tr}(A)`
     - ``post.dofs``
     - Degrees of freedom for signal: the number of independent pieces of
       information the observations give about the state.
   * - **gamma**
     - :math:`\gamma`
     - ``gamma``, ``post.with_gamma(g)``
     - The regularization factor: the solve uses :math:`S_z / \gamma`, so a
       gamma above 1 trusts the observations more and below 1 the prior
       more. 1 by default.

The posterior also has these statistics:

.. list-table::
   :header-rows: 1
   :widths: 27 23 50

   * - Statistic
     - In fips
     - What it is
   * - chi-squared
     - ``post.chi2``
     - The misfit of the estimate to the observations and to the prior, each
       weighed by its covariance; ``post.reduced_chi2`` divides it by
       :math:`m`, and is near 1 when the covariances are right.
   * - :math:`R^2`, RMSE
     - ``post.r2``, ``post.rmse``
     - The squared correlation, and the root mean square difference, of the
       observations and :math:`H \hat{x} + c`.
   * - uncertainty reduction
     - ``post.uncertainty_reduction``
     - :math:`1 - \sqrt{\operatorname{tr}(\hat{S})} / \sqrt{\operatorname{tr}(S_0)}`.
   * - Desroziers diagnostic
     - ``post.desroziers(by=...)``
     - The observation error variances the residuals :math:`z - H \hat{x} - c`
       and the innovation diagnose, beside those the solve used.

.. _fips-words:

fips's own words
----------------

**label**
   What names one element of a vector or one row or column of a matrix: a
   time, a cell, a site and a time. A label is a row of a pandas index.

**block**
   A named part of a state or of the observations whose elements share their
   kind of label: the ``flux`` cells and the ``bias`` sites of a state, the
   ``station`` and ``satellite`` blocks of the observations.

**space** (:class:`~fips.Space`)
   The labels of a state, of the observations, or of one axis of a matrix: an
   ordered set of blocks, each with its own pandas index.

**vector** (:class:`~fips.Vector`)
   Values on a space: a prior, the observations, an estimate.

**operator** (:class:`~fips.Operator`)
   A matrix with a space on each axis: the forward operator, an aggregation, a
   gain. A covariance (:class:`~fips.Covariance`) is an operator from a space to
   itself.

**covariance model**
   A covariance written down before its labels exist: a correlation model
   (:class:`~fips.Exponential`, :class:`~fips.Kronecker`, ...) with standard
   deviations, which a :class:`~fips.Problem` puts on the prior's or the
   observations' space.

**aggregation** (:math:`W`)
   An operator from a space to groups of its labels (sums or means): hourly
   observations to daily means, the state to monthly totals
   (:mod:`fips.aggregate`).

**method**
   How :func:`fips.solve` solves a problem. The one method is ``"analytic"``,
   the closed-form Gaussian posterior.

.. seealso::

   - :doc:`first_problem`: a first problem, step by step
   - :doc:`/user_guide/index`: each object in more depth
   - :doc:`/reference/posterior`: the solve and the posterior it returns
