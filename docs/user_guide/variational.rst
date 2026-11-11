.. _variational:

A problem too large to factor: the variational method
=====================================================

``fips.solve(problem)`` factors an :math:`m \times m` or an :math:`n \times n`
matrix and forms :math:`H S_0`, which is :math:`m \times n`. When both the
state and the observations number in the hundreds of thousands, none of these
fits in memory. **Give** :math:`H` **as a sparse matrix or by its products,
keep the prior covariance as its factors, and solve with**
``fips.solve(problem, "variational", rank=k)``. The method needs only
products: :math:`H v`, :math:`H^T w`, the prior covariance's root through its
factors, and solves with :math:`S_z`. It never forms anything
:math:`n \times n` or :math:`m \times m`. The estimate is the posterior mean,
found by conjugate gradients to a tolerance (the mode, with a forward model).
The uncertainty is a low-rank posterior: the prior, less what the data remove
along the ``k`` directions they inform most.

.. contents:: On this page
   :local:
   :depth: 1


When to use it
--------------

Use it when the problem is too large for ``fips.solve(problem)``, or when the
forward model is known only by a tangent linear and an adjoint (its
:meth:`~fips.ForwardModel.jvp` and :meth:`~fips.ForwardModel.vjp`), whose
Jacobian is never formed.

- When ``fips.solve(problem)`` fits in memory, use it (:doc:`posterior`): it
  is exact, gives every diagnostic, and is often as fast. A large sparse
  :math:`H` with a Kronecker prior covariance often still fits.
- With a forward model whose Jacobian you can write, and a posterior that may
  not be Gaussian, use ``"gauss_newton"`` (:doc:`nonlinear`): it draws the
  posterior, where this method gives the Gaussian at the mode.
- With a model that can only be run, use ``"ensemble"`` (:doc:`ensemble`).
- With bounds on the state, use ``"bounded"`` (:doc:`bounds`): this method
  takes none.


An example
----------

Here the state is a flux on a 16 by 16 grid for each of 24 hours: 6,144
elements. Six sensors each measure four times an hour, 576 observations in
all. Each observation sees a plume upwind of its sensor, in its hour and the
two before, drifting with that hour's wind. ``plume`` writes one
observation's row of :math:`H`: a few hundred elements, so :math:`H` is
sparse.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import scipy.sparse as sp
   import fips

   hours, ny, nx, sites = 24, 16, 16, 6
   n = hours * ny * nx
   rng = np.random.default_rng(0)
   site_y, site_x = rng.integers(4, ny - 4, sites), rng.integers(4, nx - 4, sites)
   wind = rng.normal(0.0, 1.0, (hours, 2))  # grid cells an hour

.. ipython:: python

   Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

   def plume(s, t, jitter):
       """One observation's row of H, at sensor s in hour t: the positions it sees and their weights."""
       cells, weights = [], []
       for back in range(min(3, t + 1)):
           h, width = t - back, 1.0 + back
           cy = site_y[s] - (back + 0.5) * wind[h, 0] + jitter[0]
           cx = site_x[s] - (back + 0.5) * wind[h, 1] + jitter[1]
           w = np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * width**2)) / width**2
           keep = w > 1e-3
           cells.append(h * ny * nx + (Y * nx + X)[keep])
           weights.append(w[keep])
       return np.concatenate(cells), np.concatenate(weights)

.. ipython:: python

   m = sites * hours * 4
   footprints = [plume(s, t, rng.normal(0.0, 0.3, 2)) for s, t, _ in np.ndindex(sites, hours, 4)]
   rows = np.concatenate([np.full(len(c), i) for i, (c, _) in enumerate(footprints)])
   cols = np.concatenate([c for c, _ in footprints])
   H = sp.csr_array((np.concatenate([w for _, w in footprints]), (rows, cols)), shape=(m, n))
   H.nnz / m  # elements each observation sees

The prior is 1 everywhere with a standard deviation of 0.5, correlated over 3
hours and 3 grid cells in each direction. :class:`~fips.Kronecker` keeps it as
three small factors, never the :math:`n \times n` matrix. The observation
error, 1.5, is about a third of the spread the prior gives the observations.

.. ipython:: python

   state = fips.Space({"flux": pd.MultiIndex.from_product(
       [range(hours), range(ny), range(nx)], names=["hour", "y", "x"])})
   observed = fips.Space({"obs": pd.RangeIndex(m, name="obs_id")})

   def build(obs):
       return fips.Problem(
           prior=fips.Vector(np.ones(n), state),
           prior_cov=fips.Kronecker(
               fips.Exponential("hour", 3.0), fips.Exponential("y", 3.0), fips.Exponential("x", 3.0)
           ).with_std(0.5),
           obs=obs,
           obs_cov=fips.Independent().with_std(1.5),
           forward=fips.Operator(H, observed, state),
       )

   truth, obs = build(fips.Vector(np.zeros(m), observed)).simulate(rng=1)
   problem = build(obs)

This problem is small enough for both methods, so the variational answer can
be checked against the exact one. The estimates agree far inside the
uncertainty:

.. ipython:: python

   exact = fips.solve(problem)
   post = fips.solve(problem, "variational", rank=200)
   sd = np.sqrt(exact.variances.values)
   np.abs(post.estimate.values - exact.estimate.values).max() / sd.min()  # in posterior sd
   post.iterations

The conjugate gradients took one solve of under a hundred iterations. Each
is one product with :math:`H` and one with :math:`H^T`. Their number follows
the directions the data inform (DOFS is about 95 here), not the 6,144
elements.


How much rank is enough
~~~~~~~~~~~~~~~~~~~~~~~

**Read** :attr:`post.eigenvalues <fips.Posterior.eigenvalues>`. Each row is
one direction the data inform, from the most informed. ``eigenvalue`` says how
many times more the data know about that direction than the prior did, and
``dofs`` is the share of the prior's variance the data remove along it. When
the last rows' ``dofs`` are near zero, ``rank`` held what the data inform.

.. ipython:: python

   post.eigenvalues.iloc[[0, 24, 49, 99, 149, 199]].round(3)

Below the problem's rank, the directions left out keep the prior's variance.
**So a low rank makes every variance too large, never too small.** The error
of the standard deviations against the exact ones, as ``rank`` grows:

.. ipython:: python
   :okwarning:

   table = []
   for k in [25, 50, 100, 200]:
       low = fips.solve(problem, "variational", rank=k)
       error = np.sqrt(low.variances.values) / sd - 1
       last = low.eigenvalues.iloc[-1]
       table.append({"rank": k, "dofs": low.dofs, "last eigenvalue": last["eigenvalue"],
                     "last dofs": last["dofs"], "median error": np.median(error), "worst error": error.max()})

   pd.DataFrame(table).set_index("rank").round(3)
   exact.dofs

At ranks 25 and 50 the last pair kept still has an eigenvalue above 1, and
the solve warns. At rank 100 it does not, yet the last pair still adds about
four tenths of a degree of freedom and the worst standard deviation is
several percent too large; at 200 the pairs have run out of information. The warning is a
floor: read the last pairs' ``dofs``. Aggregates follow the same
rule, here the sums of each six hours:

.. ipython:: python
   :okwarning:

   six_hours = pd.Series(np.arange(hours) // 6, index=pd.Index(range(hours), name="hour"), name="six_hours")
   W = fips.aggregate.sum(state, by=six_hours)
   pd.DataFrame({
       "exact": exact.uncertainty(W)["posterior_std"],
       "rank 50": fips.solve(problem, "variational", rank=50).uncertainty(W)["posterior_std"],
       "rank 200": post.uncertainty(W)["posterior_std"],
   }).round(3)


A model with an adjoint
~~~~~~~~~~~~~~~~~~~~~~~

A forward model whose Jacobian is too large to hold, but which can be run
forward (the tangent linear model, :math:`J v`) and backward (the adjoint
model, :math:`J^T w`), writes those two products in place of ``jacobian``.
Here :math:`F(x) = H (r \circ e^{x})` on a small problem, the model
:class:`~fips.LogNormalForward` writes for you (:doc:`positive`), written by
its products:

.. ipython:: python

   class LogScale(fips.ForwardModel):
       """F(x) = H (r * exp(x)), known by its products."""
       def __init__(self, H, reference, rows, cols):
           super().__init__(rows, cols)
           self.H, self.reference = H, reference
       def __call__(self, x):
           return fips.Vector(self.H @ (self.reference * np.exp(x.values)), self.rows)
       def jvp(self, x, v):  # the tangent linear model
           return fips.Vector(self.H @ (self.reference * np.exp(x.values) * v.values), self.rows)
       def vjp(self, x, w):  # the adjoint model
           return fips.Vector(self.reference * np.exp(x.values) * (self.H.T @ w.values), self.cols)

   small = fips.Space({"state": pd.RangeIndex(60, name="element")})
   seen = fips.Space({"obs": pd.RangeIndex(40, name="obs_id")})
   H_small = np.abs(rng.normal(size=(40, 60))) * (rng.random((40, 60)) < 0.2)
   model = LogScale(H_small, rng.uniform(0.5, 2.0, 60), seen, small)
   fips.taylortest(model, fips.Vector(rng.normal(0.0, 0.5, 60), small))

:func:`fips.taylortest` checks both products. It checks the tangent linear
against the model: the remainder of :math:`F(x + \epsilon v) - F(x) -
\epsilon J v` falls as :math:`\epsilon^2`. It checks the adjoint against the
tangent linear: :math:`\langle J v, w \rangle = \langle v, J^T w \rangle`.
An adjoint with a term missing fails the second. The variational method then
solves by Gauss-Newton, each step a conjugate-gradient solve of the problem
linearized at the last estimate:

.. ipython:: python

   log_truth = fips.Vector(rng.normal(0.0, 0.5, 60), small)
   lognormal = fips.Problem(
       prior=fips.Vector(np.zeros(60), small),
       prior_cov=fips.Exponential("element", 3.0).with_std(0.5),
       obs=fips.Vector(model(log_truth).values + 0.05 * rng.normal(size=40), seen),
       obs_cov=fips.Independent().with_std(0.05),
       forward=model,
   )
   by_products = fips.solve(lognormal, "variational", rank=40)
   by_products.iterations
   newton = fips.solve(lognormal, "gauss_newton", draws=200)
   np.abs(by_products.estimate.values - newton.estimate.values).max()

The two modes agree to within their stopping rules. ``"gauss_newton"`` forms
the Jacobian (here from the same products, a column at a time) and draws the
posterior. The variational method never forms it, and its uncertainty is the
low-rank Gaussian at the mode, which is not the posterior where the model
bends much over the posterior's spread (:doc:`positive`).


What the posterior reports
--------------------------

The variational posterior is a low-rank one, or a Gaussian one when ``rank``
reaches the problem's rank. The :ref:`members table <posterior-members>` lists
every member for every kind of answer; for this method:

- :attr:`~fips.Posterior.estimate` is the posterior mean to ``cg_tol`` for a
  forward operator, and the mode for a forward model (whose
  :attr:`~fips.Posterior.mean` raises).
- :attr:`~fips.Posterior.variances` and :meth:`~fips.Posterior.cov` are the
  prior's less ``rank`` pairs: too large below the problem's rank, exact at
  it. The covariance of aggregates, :meth:`~fips.Posterior.cov` with ``W``,
  forms nothing :math:`n \times n`.
- :attr:`~fips.Posterior.eigenvalues` lists the pairs and the DOFS each adds;
  :attr:`~fips.Posterior.iterations` lists each Gauss-Newton step (one for a
  forward operator) with its conjugate-gradient iterations.
- :attr:`~fips.Posterior.dofs` (a lower bound),
  :attr:`~fips.Posterior.dofs_per_element`,
  :attr:`~fips.Posterior.kalman_gain`,
  :meth:`~fips.Posterior.averaging_kernel` and
  :attr:`~fips.Posterior.information_content` are those of the ``rank``
  directions kept.
- :meth:`~fips.Posterior.sample` gives exact draws of the low-rank posterior.
- :meth:`~fips.Posterior.with_data` and :meth:`~fips.Posterior.with_gamma`
  cost one conjugate-gradient solve each, keeping the pairs. With a forward
  model, ``with_data`` solves again from the last linearization, and
  ``with_gamma`` raises: the linearization depends on :math:`\gamma`.
- :attr:`~fips.Posterior.log_evidence` is exact at full rank for a forward
  operator, and Laplace's for a forward model. It raises when ``rank`` left
  out pairs that still hold information.
- :meth:`~fips.Posterior.recovery` and the data worth raise: the estimate is
  linear in the observations only to the tolerance.


Options
-------

``fips.solve(problem, "variational", **options)`` takes
(:class:`~fips.methods.variational.VariationalOptions`):

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Option
     - Default
     - What it does
   * - ``rank``
     - 100
     - The number of directions the low-rank posterior keeps (at most the
       smaller of :math:`m` and the columns of the prior's root).
   * - ``cg_tol``
     - 1e-8
     - The tolerance of the iterative solves, below 1: the conjugate
       gradients stop when their residual is below ``cg_tol`` times the right
       side, and Lanczos when each eigenvalue is good to ``cg_tol``.
   * - ``max_cg_iterations``
     - 1000
     - The most conjugate-gradient iterations of one solve.
   * - ``max_iterations``
     - 50
     - For a forward model, the most Gauss-Newton steps, as for
       ``"gauss_newton"``.
   * - ``tol``
     - 1e-6
     - For a forward model, the steps stop when one moves the estimate by
       less than ``tol`` times :math:`n`, measured in the posterior
       covariance (Rodgers' rule, as for ``"gauss_newton"``).
   * - ``seed``
     - 0
     - The seed of Lanczos's starting vector, so that a solve is
       reproducible.
   * - ``gamma``
     - 1.0
     - The regularization factor: the observation covariance is divided by
       it, as for every method (:doc:`posterior`).


Warnings and checks
-------------------

**It warns when** ``rank`` **is too small by a clear margin**: when ``rank``
is below the most pairs the problem has and the last pair kept still has an
eigenvalue above 1 (that direction holds more from the data than from the
prior). Raise ``rank``. The warning is a floor, not a target: read the last
pairs' ``dofs`` in :attr:`~fips.Posterior.eigenvalues`. Each solve also logs,
at INFO, the pairs kept, DOFS and the share of it in the last tenth of the pairs,
and the conjugate-gradient iterations. The warning is a
:class:`~fips.FipsWarning`, which Python shows without any setup; the INFO
lines are silent until you configure logging (:ref:`repeated-inversions`).

It raises :class:`~fips.ConvergenceError` (a :class:`RuntimeError`) when:

- the conjugate gradients have not reached ``cg_tol`` after
  ``max_cg_iterations``: raise either. Observations far more precise than
  the prior make large eigenvalues, and the solve slow;
- Lanczos does not find the pairs: lower ``rank``, or raise ``cg_tol``;
- with a forward model, the steps have not stopped after ``max_iterations``.

With a forward model it raises :class:`RuntimeError` when the cost rises
along a step however short it is made: the products are not the derivative
of the model; check them with :func:`fips.taylortest`.

It raises :class:`ValueError` for an option out of range (``cg_tol`` must be
below 1, ``seed`` a nonnegative integer), and :class:`~fips.UnavailableError` for a member it cannot give
(above).


Cost
----

**Memory grows as** :math:`n k`, **not** :math:`m n`. The method holds the
sparse :math:`H` (or nothing, for products), the prior covariance's factors,
and :math:`n \times k` for the pairs. ``fips.solve(problem)`` holds
:math:`H S_0`, :math:`m \times n` dense.

**Time grows with the directions the data inform, not with** :math:`n`. A
conjugate-gradient iteration is a product with each of :math:`H`,
:math:`H^T`, the prior's root and its transpose, and a solve takes about as
many iterations as there are eigenvalues well above 1. The pairs cost about
two such products each by Lanczos. When ``rank`` is at least half the
columns of the prior's root, the method forms the whitened Hessian from that
many products instead and eigendecomposes it, dense. A forward model costs a
solve and a run of the model per Gauss-Newton step. The conjugate gradients
are preconditioned only by the prior's root, so observations far more precise
than the prior take more iterations. For scale: a few hundred thousand state
elements and tens of thousands of observations, which ``fips.solve(problem)``
cannot hold, take about a minute and a gigabyte or two at ``rank=100``.


How it works
------------

Write the prior covariance by its root, :math:`S_0 = R R^T`, and the state in
the whitened form :math:`x = x_0 + R \xi`. With the innovation :math:`d = z -
H x_0 - c`, the cost is :math:`\xi^T \xi + \gamma (d - H R \xi)^T S_z^{-1}
(d - H R \xi)`, and its minimum solves the whitened normal equations

.. math::

   (I + \tilde{H})\, \xi = \gamma R^T H^T S_z^{-1} d, \qquad
   \tilde{H} = \gamma R^T H^T S_z^{-1} H R,

where :math:`\tilde{H}` is the Hessian of the data misfit whitened by the
prior. Their matrix is the identity plus a positive semidefinite one, so its
eigenvalues are at least 1. Conjugate gradients solve the equations in about
as many products as :math:`\tilde{H}` has eigenvalues standing above 1.

The posterior covariance is :math:`\hat{S} = R (I + \tilde{H})^{-1} R^T`.
The eigenvalues :math:`\tilde\lambda_j` of :math:`\tilde{H}` say how many
times more the data know about each direction than the prior did; along that
direction the data remove the share :math:`\tilde\lambda_j / (1 +
\tilde\lambda_j)` of the prior's variance, and these shares add up to DOFS.
The ``k`` leading pairs, :math:`\tilde{H} v_j = \tilde\lambda_j v_j`, found by
Lanczos from products, give the low-rank posterior

.. math::

   \hat{S}_k = S_0 - G \operatorname{diag}\big(\tilde\lambda_j / (1 +
   \tilde\lambda_j)\big) G^T, \qquad G = R V \ (n \times k).

It is the posterior of the ``k`` directions the data inform most, with the
prior left in the others. The covariance of aggregates is :math:`W S_0 W^T -
(W G) D (W G)^T`, and a draw is :math:`\hat{x} + R (w - V
\operatorname{diag}(1 - (1 + \tilde\lambda_j)^{-1/2}) V^T w)` with :math:`w`
standard normal. The pairs depend neither on :math:`\gamma` (it only scales
:math:`\tilde\lambda`) nor on the observations' values, so another
:math:`\gamma` or other data cost one conjugate-gradient solve.

With a forward model the same solve is the inner step of Gauss-Newton, which
is incremental 4D-Var. At each iterate the problem linearized there, with the
Jacobian known by ``jvp`` and ``vjp``, is solved by conjugate gradients, and
the steps stop by the rule ``"gauss_newton"`` uses (:doc:`nonlinear`). A step
that raises the cost is halved until it does not. The pairs are those of the
last linearization, so the uncertainty is the Gaussian at the mode, Laplace's
approximation.


References
----------

Bui-Thanh, T., Ghattas, O., Martin, J., and Stadler, G. (2013). A
computational framework for infinite-dimensional Bayesian inverse problems.
Part I. *SIAM Journal on Scientific Computing*, 35(6), A2494-A2523.
https://doi.org/10.1137/12089586X

Courtier, P., Thépaut, J.-N., and Hollingsworth, A. (1994). A strategy for
operational implementation of 4D-Var, using an incremental approach.
*Quarterly Journal of the Royal Meteorological Society*, 120(519),
1367-1387. https://doi.org/10.1002/qj.49712051912

Spantini, A., Solonen, A., Cui, T., Martin, J., Tenorio, L., and Marzouk, Y.
(2015). Optimal low-rank approximations of Bayesian linear inverse problems.
*SIAM Journal on Scientific Computing*, 37(6), A2451-A2487.
https://doi.org/10.1137/140977308
