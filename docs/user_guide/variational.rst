.. _variational:

A problem too large to factor: the variational method
=====================================================

``fips.solve(problem)`` factors an :math:`m \times m` or an :math:`n \times n`
matrix and forms :math:`H S_0` (:math:`m \times n`). With an hourly state on a
fine grid and a year of observations, both :math:`n` and :math:`m` pass
:math:`10^5`, and then none of them fits in memory. The rule: **give** :math:`H`
**as a sparse matrix (or as its products), keep the prior as Kronecker
factors, and solve with** ``fips.solve(problem, "variational", rank=k)``. It
needs only products, :math:`H v`, :math:`H^T w`, the prior's root through its
factors and solves with :math:`S_z`, and never forms anything :math:`n \times
n` or :math:`m \times m`. Its estimate is the analytic one, by conjugate
gradients, to ``cg_tol``; its uncertainty is a low-rank posterior: the prior, less
what the ``k`` directions the data inform most remove. Check
:attr:`post.eigenvalues <fips.Posterior.eigenvalues>` to see whether ``k`` was
enough.

**When not to use it**: when ``fips.solve(problem)`` fits in memory, it is
exact, gives every diagnostic at no extra cost, and is often as fast. The
variational method is for the problem it cannot hold. It is also the method for
a forward model with a tangent linear and an adjoint (its products, written as
:meth:`~fips.ForwardModel.jvp` and :meth:`~fips.ForwardModel.vjp`), whose
Jacobian is never formed: that is incremental 4D-Var (`A model with an
adjoint`_).

.. contents:: On this page
   :local:
   :depth: 1


One problem, two ways
---------------------

A day of hourly fluxes on a 20 by 20 grid (9,600 state elements) and eight
sites, each measuring six times an hour (1,152 observations). Each observation
sees a plume upwind of its site, in its hour and the two before, drifting with
that hour's wind: a stand-in for STILT footprints, which are sparse and
overlap from one observation to the next.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import scipy.sparse as sp
   import fips
   from fips.linalg import KroneckerOperator

   hours, ny, nx, sites = 24, 20, 20, 8
   n = hours * ny * nx
   rng = np.random.default_rng(0)
   site_y, site_x = rng.integers(4, ny - 4, sites), rng.integers(4, nx - 4, sites)
   wind = rng.normal(0.0, 1.0, (hours, 2))  # cells an hour
   Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

   def plume(s, t, jitter):
       """The footprint of one observation at site s in hour t: a dict of cell positions and weights."""
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

   m = sites * hours * 6  # six observations an hour at each site
   footprints = [plume(s, t, rng.normal(0.0, 0.3, 2)) for s, t, _ in np.ndindex(sites, hours, 6)]
   rows = np.concatenate([np.full(len(c), i) for i, (c, _) in enumerate(footprints)])
   cols = np.concatenate([c for c, _ in footprints])
   H = sp.csr_array((np.concatenate([w for _, w in footprints]), (rows, cols)), shape=(m, n))
   H.nnz / m  # cells each observation sees

The prior is a flux of 1 with a standard deviation of 0.5, correlated over 3
hours and 3 cells in each direction: a Kronecker product of three
correlations, kept as its factors (:class:`~fips.linalg.KroneckerOperator`).
The footprints are scaled so that the prior's spread, seen through them, is
three times the observations' error of 0.1.

.. ipython:: python

   def exponential(k, length):
       d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
       return np.exp(-d / length)

   state = fips.Space({"flux": pd.MultiIndex.from_product(
       [range(hours), range(ny), range(nx)], names=["hour", "y", "x"])})
   observed = fips.Space({"ch4": pd.RangeIndex(m, name="obs_id")})
   prior_cov = fips.Covariance(KroneckerOperator(
       [exponential(hours, 3.0), exponential(ny, 3.0), exponential(nx, 3.0)], std=0.5), state)
   root = prior_cov.data._root  # the prior's square root, through the factors
   H = H * (3 * 0.1 / (H @ (root @ rng.standard_normal((root.shape[1], 20)))).std())

   world = fips.Problem(
       prior=fips.Vector(np.ones(n), state), prior_cov=prior_cov,
       obs=fips.Vector(np.zeros(m), observed),
       obs_cov=fips.Covariance.diagonal(np.full(m, 0.1**2), observed),
       forward=fips.Operator(H, observed, state),
   )
   truth, obs = world.simulate(rng=1)
   problem = fips.Problem(prior=world.prior, prior_cov=prior_cov, obs=obs,
                          obs_cov=world.obs_cov, forward=world.forward)

This one is small enough for both methods, so the variational answer can be
checked against the exact one:

.. ipython:: python

   exact = fips.solve(problem)
   post = fips.solve(problem, "variational", rank=200)
   sd = np.sqrt(exact.variances.values)
   np.abs(post.estimate.values - exact.estimate.values).max() / sd.min()  # in posterior sd
   post.iterations

The estimate is the analytic one far inside its uncertainty (the conjugate
gradients stop when their residual is below ``cg_tol``, 1e-8 of the right side,
by default). It took one solve of about 120 iterations: the data have about
170 degrees of freedom among the 9,600 cells, and conjugate gradients converge
in about as many steps as there are directions that stand out, not in
:math:`n`. Each iteration is one product with :math:`H` and one with
:math:`H^T` (and the prior's root and its transpose, through the factors).


How much rank is enough
-----------------------

The posterior covariance is :math:`\hat{S} = R (I + \tilde{H})^{-1} R^T`, with
:math:`S_0 = R R^T` and :math:`\tilde{H} = \gamma R^T H^T S_z^{-1} H R`, the
data's information on the state whitened by the prior. Its eigenvalues
:math:`\tilde\lambda_j` say how many times more the data know about each
direction than the prior did; along that direction the data remove the share
:math:`\tilde\lambda_j / (1 + \tilde\lambda_j)` of the prior's variance, and
these shares add up to DOFS. ``rank`` keeps the ``k`` largest (by Lanczos,
from products) and leaves the prior in the other directions:

.. math::

   \hat{S}_k = S_0 - G \operatorname{diag}\big(\tilde\lambda_j / (1 + \tilde\lambda_j)\big) G^T,
   \qquad G = R V \ (n \times k).

So each variance is the prior's less a sum over the pairs: too large when ``k``
is too small, never too small, and exact once ``k`` reaches the number of
directions the data see. :attr:`~fips.Posterior.eigenvalues` lists the pairs:

.. ipython:: python

   post.eigenvalues.iloc[[0, 9, 49, 99, 149, 199]].round(3)

The last pairs still add about a third of a degree of freedom each: more
directions are informed. The error of the standard deviations, against the
exact ones, as ``rank`` grows:

.. ipython:: python

   ranks = [25, 50, 100, 200, 400]
   curve = []
   for k in ranks:
       low = fips.solve(problem, "variational", rank=k)
       error = np.sqrt(low.variances.values) / sd - 1
       curve.append({"rank": k, "dofs": low.dofs, "last pair": low.eigenvalues["dofs"].iloc[-1],
                     "median sd error": np.median(error), "largest sd error": error.max()})

   curve = pd.DataFrame(curve).set_index("rank")
   curve.assign(exact_dofs=exact.dofs).round(4)

.. ipython:: python

   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=(5, 3.5))
   ax.semilogy(curve.index, curve["largest sd error"], "o-", label="largest")
   ax.semilogy(curve.index, curve["median sd error"], "s-", label="median")
   ax.set_xlabel("rank")
   ax.set_ylabel("relative error of the posterior sd")
   @savefig variational_rank.png width=5in
   ax.legend()

The error falls with every pair, from above (the test suite holds that: the
variances never cross the exact ones, and DOFS never falls as ``rank`` grows).
At 200 pairs, the median standard deviation is within a percent of the exact
one and the worst about 5% too large; at 400, within 0.2%. Read the last pairs'
``dofs``: when they are near zero, ``rank`` held what the data inform. A solve
warns when the last pair's eigenvalue is still above 1 (that direction holds
more from the data than from the prior). A pair costs Lanczos two or three
products with :math:`H` and :math:`H^T`; the whole loop above took about ten
seconds on four cores.

Everything the posterior reports comes from the same factors, with nothing
:math:`n \times n`: DOFS (the sum over the pairs, a lower bound),
:attr:`~fips.Posterior.dofs_per_element`, the gain and the averaging kernel of
the ``k`` directions, the information content, :meth:`~fips.Posterior.cov`
of aggregates (:math:`W S_0 W^T - (W G) D (W G)^T`), and exact draws of
:math:`\hat{S}_k` (:meth:`~fips.Posterior.sample`). The standard deviations
of the totals of each six hours, at 50 and 200 pairs:

.. ipython:: python

   quarter = pd.Series(np.arange(hours) // 6, index=pd.Index(range(hours), name="hour"), name="quarter")
   W = fips.aggregate.sum(state, by=quarter)  # the totals of each six hours
   pd.DataFrame({k: np.sqrt(np.diag(fips.solve(problem, "variational", rank=k).cov(W).dense()))
                 for k in (50, 200)}).assign(exact=np.sqrt(np.diag(exact.cov(W).dense()))).round(3)

They are about 10% too large at 50 pairs and 1% at 200, as a typical cell's are.

:meth:`~fips.Posterior.with_data` (and so :meth:`~fips.Posterior.trials`) and
:meth:`~fips.Posterior.with_gamma` cost one conjugate-gradient solve each: the
eigenvectors depend neither on the observations' values nor on
:math:`\gamma`.


What it costs at a size fips.solve cannot hold
----------------------------------------------

Measured on one cluster node (eight cores of an Intel Xeon Gold 6230R, 48 GB),
each solve in its own process with its address space capped at 40 GB, with the plume footprints above on a larger grid
(the same plumes, prior and signal-to-noise, with an error of 0.3 and a
prior's correlations of 3 hours and 4 cells):

.. list-table::
   :header-rows: 1

   * - state :math:`n`
     - observations :math:`m`
     - method
     - solve
     - peak memory
   * - 38,400
     - 2,880
     - ``fips.solve``
     - 9 s
     - 2.1 GB
   * - 38,400
     - 2,880
     - variational, rank 100
     - 3 s
     - 0.3 GB
   * - 38,400
     - 2,880
     - variational, rank 300
     - 13 s
     - 0.5 GB
   * - 38,400
     - 2,880
     - variational, rank 600
     - 45 s
     - 0.9 GB
   * - 259,200
     - 17,280
     - ``fips.solve``
     - out of memory (``MemoryError`` at 38 GB, after 4 minutes)
     - above 40 GB
   * - 259,200
     - 17,280
     - variational, rank 100
     - 41 s
     - 1.2 GB
   * - 259,200
     - 17,280
     - variational, rank 300
     - 160 s
     - 2.8 GB

``fips.solve`` holds :math:`H S_0`, :math:`m \times n` dense, and factors the
:math:`m \times m` innovation covariance: its memory grows as :math:`m n`. The
variational solve holds the sparse :math:`H`, the Kronecker factors and
:math:`G` (:math:`n \times k`): its memory grows as :math:`n k`, and its time
as its products with :math:`\tilde{H}` (each one with :math:`H`, :math:`H^T`
and the prior's root and its transpose): the CG iterations, about 200 at
both sizes here, and two to three for each pair. The low ranks leave the
standard deviations too large where the data inform more directions than
``rank`` keeps (at 38,400 cells the exact DOFS is 459): the median cell's is
9% too large at rank 100, 4% at 300 and 0.5% at 600 (the largest 65%, 21% and
4%).


A model with an adjoint
-----------------------

A forward model whose Jacobian is too large to hold, but which can be run
forward (tangent linear, :math:`J v`) and backward (adjoint, :math:`J^T w`),
writes those two products in place of ``jacobian``. Here a lognormal flux,
:math:`F(x) = H (r \circ e^{x})`, on a small grid:

.. ipython:: python

   class Lognormal(fips.ForwardModel):
       def __init__(self, H, reference, rows, cols):
           super().__init__(rows, cols)
           self.H, self.reference = H, reference
       def __call__(self, x):
           return fips.Vector(self.H @ (self.reference * np.exp(x.values)), self.rows)
       def jvp(self, x, v):  # the tangent linear model
           return fips.Vector(self.H @ (self.reference * np.exp(x.values) * v.values), self.rows)
       def vjp(self, x, w):  # the adjoint model
           return fips.Vector(self.reference * np.exp(x.values) * (self.H.T @ w.values), self.cols)

   small = fips.Space({"flux": pd.RangeIndex(60, name="cell")})
   seen = fips.Space({"ch4": pd.RangeIndex(40, name="obs_id")})
   H_small = np.abs(rng.normal(size=(40, 60))) * (rng.random((40, 60)) < 0.2)
   model = Lognormal(H_small, rng.uniform(0.5, 2.0, 60), seen, small)
   fips.taylortest(model, fips.Vector(rng.normal(0.0, 0.5, 60), small))

:func:`fips.taylortest` checks both products: the tangent linear against the
model (the remainder of :math:`F(x + \epsilon v) - F(x) - \epsilon J v` falls
as :math:`\epsilon^2`) and the adjoint against the tangent linear
(:math:`\langle J v, w \rangle = \langle v, J^T w \rangle`); an adjoint with a
term missing fails the second. The variational method then solves by
Gauss-Newton, each step a conjugate-gradient solve of the problem linearized
at the last estimate, with the same stopping rule as ``"gauss_newton"``
(:doc:`nonlinear`):

.. ipython:: python

   log_truth = fips.Vector(rng.normal(0.0, 0.5, 60), small)
   lognormal = fips.Problem(
       prior=fips.Vector(np.zeros(60), small),
       prior_cov=fips.Exponential("cell", 3.0).with_std(0.5),
       obs=fips.Vector(model(log_truth).values + 0.05 * rng.normal(size=40), seen),
       obs_cov=fips.Independent().with_std(0.05),
       forward=model,
   )
   by_products = fips.solve(lognormal, "variational", rank=40)
   by_products.iterations
   newton = fips.solve(lognormal, "gauss_newton", draws=200)
   np.abs(by_products.estimate.values - newton.estimate.values).max()

The two modes agree to within their stopping rules (both stop when a step is
a thousandth of a posterior standard deviation). ``"gauss_newton"`` forms the Jacobian (here from the same
products, a column at a time) and draws the posterior itself; the variational
method never forms it, and its uncertainty is the low-rank Gaussian at the
mode, Laplace's approximation, which is not the posterior where the model
bends much over the posterior's spread (:doc:`positive`). Its posterior is
the low-rank kind, and :ref:`the table <posterior-members>` says what that
gives (no ``with_gamma`` with a forward model, since the linearization depends
on :math:`\gamma`); ``with_data`` solves again from the last linearization.

Out of this method, for now: bounds on the state, checkpointing an adjoint
model, and a preconditioner for the conjugate gradients beyond the prior's
root (a problem with observations far more precise than its prior has large
eigenvalues, and the solve takes more iterations; ``max_cg_iterations`` caps
them).
