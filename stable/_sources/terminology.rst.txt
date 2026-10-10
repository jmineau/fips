Terminology & Notation
======================

This page defines common abbreviations and the mathematical notation used throughout the FIPS documentation and code. Different scientific fields often use different conventions — this guide helps bridge those gaps.

Mathematical Notation
---------------------

.. _notation-framework:

Notation Framework
~~~~~~~~~~~~~~~~~~

FIPS uses a consistent notation system throughout:

**Dimensionality Convention**

- **Lowercase letters** (:math:`x`, :math:`z`, :math:`y`) represent **1-D vectors**
- **Uppercase letters** (:math:`H`, :math:`S`, :math:`K`, :math:`A`) represent **2-D matrices**

**Hat Notation**

- **Hat** :math:`\hat{\ }` = **posterior** (*a posteriori* estimate after incorporating observations)

  - :math:`\hat{x}` = posterior state
  - :math:`\hat{S}` = posterior covariance
  - :math:`\hat{y}` = posterior modeled observations

**The Forward Model**

The fundamental relationship is:

.. math::

   z - c = y = Hx

where:

- :math:`x` = **state vector** (the unknowns we're solving for)
- :math:`z` = **observations** (measured data)
- :math:`c` = **constant** (background or offset)
- :math:`y` = **modeled observations** (what the forward model predicts)
- :math:`H` = **forward operator** (maps state space → observation space)

**Subscript Conventions**

The subscript system indicates which space or time a variable belongs to:

- **Subscript** :math:`_0` = **prior information** / *a priori* (before incorporating observations)

  - :math:`x_0` = prior state vector
  - :math:`S_0` = prior error covariance matrix

- **Subscript** :math:`_z` = **observation space** (associated with :math:`z`)

  - :math:`S_z` = observation/model-data mismatch error covariance matrix

**Covariance Matrices**

Following the uppercase convention for matrices, covariance matrices are denoted with :math:`S`:

- :math:`S` = any covariance matrix (uppercase because 2-D)
- :math:`S_0` = **prior** error covariance (subscript _0 for *a priori*)
- :math:`S_z` = **observation** error covariance (subscript _z because it's in observation space)
- :math:`\hat{S}` = **posterior** error covariance (hat for *a posteriori*)

This framework applies consistently: any variable with subscript :math:`_0` refers to the prior,
any variable in observation space gets subscript :math:`_z`, and posterior quantities get a hat.

.. _notation-reference:

Quick Reference
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Symbol
     - Name
     - Description
   * - :math:`x`
     - State vector
     - The unknown quantities being estimated (e.g., fluxes, densities)
   * - :math:`x_0`
     - Prior state
     - *A priori* estimate before incorporating observations
   * - :math:`\hat{x}`
     - Posterior state
     - *A posteriori* optimized state estimate after inversion
   * - :math:`z`
     - Observations
     - Measured data (e.g., concentrations, gravity anomalies)
   * - :math:`c`
     - Constant / Background
     - Additive offset or background field
   * - :math:`y`
     - Modeled observations
     - Forward model output :math:`y = Hx + c`
   * - :math:`y_0`
     - Prior observations
     - :math:`y_0 = Hx_0 + c`
   * - :math:`\hat{y}`
     - Posterior observations
     - :math:`\hat{y} = H\hat{x} + c`
   * - :math:`H`
     - Forward operator / Jacobian
     - Operator mapping state space to observation space
   * - :math:`S_0`
     - Prior error covariance
     - Uncertainty in the prior state estimate
   * - :math:`S_z`
     - Observation error covariance
     - Combined measurement error and model representation error
   * - :math:`\hat{S}`
     - Posterior error covariance
     - Reduced uncertainty after incorporating observations
   * - :math:`K`
     - Kalman gain
     - Weighting matrix that determines how observations update the prior
   * - :math:`A`
     - Averaging kernel
     - Shows which states are constrained by observations: :math:`A = KH`

.. _diagnostic-metrics:

Diagnostic Metrics
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Symbol
     - Name
     - Description
   * - DOFS
     - Degrees of Freedom for Signal
     - Number of independent pieces of information from observations. Equal to :math:`\text{Tr}(A)`
   * - :math:`\chi^2`
     - Chi-squared statistic
     - Goodness-of-fit metric comparing observations to model predictions
   * - :math:`R^2`
     - Coefficient of determination
     - Fraction of variance explained by the model (0 to 1)


.. _inverse-problem-terms:

Inverse Problem Terminology
----------------------------

.. glossary::

   **Prior**
      The initial estimate of the state (and its uncertainty) before incorporating observations. Often comes from inventory data, climatology, or a process model.

   **Posterior**
      The updated estimate of the state (and its uncertainty) after incorporating observations through Bayesian inference.

   **Forward Model / Forward Operator**
      The mathematical operator :math:`H` that predicts observations from a given state: :math:`y = Hx + c`. Sometimes called the Jacobian, observation operator, or sensitivity matrix.

   **Jacobian**
      In the linear case, identical to the forward operator :math:`H`. For nonlinear problems, the Jacobian is the local linearization of the forward model.

   **Observation Operator**
      Another name for the forward operator, emphasizing its role in mapping state space to observation space.

   **Kalman Gain**
      The matrix :math:`K` that optimally weights how much each observation updates the prior state. Derived from minimizing posterior uncertainty.

   **Averaging Kernel**
      Matrix :math:`A = KH` showing which true state variables are constrained by the observations. Diagonal elements near 1 indicate strong constraint; near 0 indicates weak constraint.

   **Model-Data Mismatch**
      The combined error in observations and forward model representation, captured in the covariance matrix :math:`S_z`. Includes measurement error, transport error, aggregation error, etc.

   **Covariance Matrix**
      A symmetric positive-definite matrix encoding uncertainties and their correlations. Diagonal elements are variances; off-diagonal elements are covariances.

   **Posterior Error Reduction**
      The decrease in uncertainty from prior to posterior, often expressed as :math:`1 - \text{diag}(\hat{S}) / \text{diag}(S_0)`.

.. seealso::

   - :doc:`getting_started` — Quick introduction to FIPS with minimal example
   - :doc:`usage` — Detailed guide to data structures and workflows
   - :doc:`reference/estimators` — Full mathematical details of estimators
