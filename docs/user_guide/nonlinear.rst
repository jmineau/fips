.. _nonlinear:

A nonlinear forward model
=========================

When the observations are not a linear function of the state (fluxes solved
as scale factors of an inventory, :math:`x \mapsto r\, e^{x}`; a sensor that
saturates), the forward map is a model :math:`F` and the observations are
:math:`z = F(x) + c + e`. The rule: **write the model and its Jacobian as a**
:class:`~fips.ForwardModel`, **and** ``fips.solve(problem, "gauss_newton")``
**solves the problem linearized at its last estimate, again and again, until the
estimate stops moving.** It ends at the mode of the posterior, and reports the
Gaussian of the problem linearized there. That Gaussian is the posterior when
the model is close to linear over the posterior's spread; when the model bends
a lot it is an approximation, and :meth:`~fips.Posterior.trials` measures how
good.

.. contents:: On this page
   :local:
   :depth: 1


A model
-------

A :class:`~fips.ForwardModel` is a class with two methods: ``__call__(x)``,
the observations a state models without the offset, :math:`F(x)`, a
:class:`~fips.Vector` on the observations' labels; and ``jacobian(x)``, the
derivatives :math:`J_{ij} = \partial F_i / \partial x_j` at :math:`x`, an
:class:`~fips.Operator`. Its ``__init__`` passes the two spaces to
``super().__init__(rows, cols)``. fips builds no Jacobian: it calls yours.

Here the state is the logarithm of a scale factor of each cell's inventory
:math:`r`, so the flux :math:`r\, e^{x}` cannot be negative and the prior is a
factor of the inventory, not a flux. The model is :math:`F(x) = H (r\, e^{x})`
and its Jacobian :math:`H \operatorname{diag}(r\, e^{x})`:

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips
   from fips.linalg import DiagonalOperator

   class ScaledInventory(fips.ForwardModel):
       """The observations of an inventory scaled by exp(x): F(x) = H (r * exp(x))."""
       def __init__(self, H, reference):
           super().__init__(H.rows, H.cols)
           self.H = H
           self.reference = reference
       def __call__(self, x):
           return self.H @ (self.reference * np.exp(x))
       def jacobian(self, x):
           flux = (self.reference * np.exp(x)).values
           return self.H @ fips.Operator(DiagonalOperator(flux), x.space, x.space)

Ten cells, 25 observations of them over a background of 1900 ppb, a prior of
no change (:math:`x_0 = 0`) with a standard deviation of 0.5 (a factor of
about 1.6) correlated over two cells, and data made from a truth drawn from
that prior:

.. ipython:: python

   rng = np.random.default_rng(0)
   cells = fips.Space({"flux": pd.RangeIndex(10, name="cell")})
   observed = fips.Space({"ch4": pd.RangeIndex(25, name="obs_id")})
   footprint = rng.uniform(0.0, 1.0, (25, 10)) * (rng.uniform(size=(25, 10)) < 0.4)
   H = fips.Operator(footprint, observed, cells)
   inventory = fips.Vector(rng.uniform(0.5, 2.0, 10), cells)
   model = ScaledInventory(H, inventory)

.. ipython:: python

   def build(obs, std=0.5, noise=0.2):
       """The problem of the observations `obs`, with a prior std of `std` in log units."""
       return fips.Problem(
           prior=fips.Vector(np.zeros(10), cells),
           prior_cov=fips.Exponential("cell", 2.0).with_std(std),
           obs=obs,
           obs_cov=fips.Independent().with_std(noise),
           forward=model,
           offset=1900.0,
       )

   truth, obs = build(fips.Vector(np.zeros(25), observed)).simulate(rng=1)
   problem = build(obs)
   problem.predict(problem.prior).values[:4].round(3)  # F(x_0) + c

:meth:`~fips.Problem.predict` and :meth:`~fips.Problem.simulate` run the model,
and :meth:`~fips.Problem.aggregate` follows it with the aggregation (the model
of the aggregated problem is :math:`W F(x)`). :meth:`~fips.Problem.restrict`
refuses a model: holding cells at their prior puts what they add to the
observations into the offset, which is a constant only for a linear forward.


Checking the Jacobian
---------------------

A wrong Jacobian still gives an answer, so check it first.
:func:`fips.taylortest` steps away from a state along a direction and
compares the model with its tangent: the remainder :math:`\lVert F(x +
\epsilon\, \delta x) - F(x) - \epsilon J \delta x \rVert` falls as
:math:`\epsilon^2` when :math:`J` is the derivative, and only as
:math:`\epsilon` when it is not. A Jacobian that forgot the inventory fails:

.. ipython:: python

   fips.taylortest(model, problem.prior)

   class Forgot(ScaledInventory):
       def jacobian(self, x):
           scale = fips.Operator(DiagonalOperator(np.exp(x.values)), x.space, x.space)
           return self.H @ scale

   fips.taylortest(Forgot(H, inventory), problem.prior)

Its debug log has the order of the remainder at each step; a right Jacobian
gives 2. It checks one direction, a random one by default (``dx=`` sets it),
and the state you give: check at the prior and at the estimate.


One linearization
-----------------

:meth:`Problem.linearized(x) <fips.Problem.linearized>` is the linear problem
at a state :math:`x`: the forward operator :math:`J(x)` and the offset
:math:`c + F(x) - J(x)\, x`, so it predicts what the model does at :math:`x`
and near it to first order. Its analytic solve is one Gauss-Newton step from
:math:`x`. ``fips.solve(problem)`` refuses a model rather than linearize it
silently at the prior, since one linearization is not the mode when the data
move the state far:

.. ipython:: python

   linear = problem.linearized(problem.prior)
   linear.forward
   one_step = fips.solve(linear)
   one_step.estimate.values.round(2)


Solving
-------

``"gauss_newton"`` starts from the prior and takes that step from each new
estimate. It stops by Rodgers' (2000) rule: when the step :math:`\delta`,
measured in the posterior covariance of the solve that made it, :math:`d^2 =
\delta^T \hat{S}^{-1} \delta`, is below ``tol`` (default :math:`10^{-6}`) times
the number of state elements. A step that would raise the cost is halved until
it does not; a solve that has not stopped after ``max_iterations`` (50) raises,
naming the steps' :math:`d^2`. :attr:`~fips.Posterior.iterations` keeps each
step's :math:`d^2` and the cost after it:

.. ipython:: python

   post = fips.solve(problem, "gauss_newton")
   post.iterations
   post.estimate.values.round(2)

The first step went most of the way; the next ones moved the estimate by up to
a few posterior standard deviations, which one linearization would have missed:

.. ipython:: python

   moved = (post.estimate.values - one_step.estimate.values) / np.sqrt(post.variances.values)
   np.abs(moved).max().round(2)

Near the mode, :math:`d^2` falls by orders of magnitude a step. When it falls
only by a constant factor, the model bends much over the posterior's spread,
the estimate may be further from the mode than the last step, and the solve
logs a warning saying so.


What the posterior at the mode is
---------------------------------

The **estimate** is the mode: the state where the cost :math:`(x - x_0)^T
S_0^{-1} (x - x_0) + \gamma (z - F(x) - c)^T S_z^{-1} (z - F(x) - c)` is least.
``post.mean`` raises. The rest comes from two places:

- **the model at the mode**: :attr:`~fips.Posterior.modeled_obs` is
  :math:`F(\hat{x}) + c`, and the residuals, :attr:`~fips.Posterior.rmse`,
  :attr:`~fips.Posterior.r2`, :attr:`~fips.Posterior.chi2` (the cost at the
  mode) and its two misfits are the model's;
- **the Gaussian at the mode**: the covariance, the variances, the gain, the
  averaging kernel, DOFS, the information content, the innovation covariance
  and :meth:`~fips.Posterior.sample` are those of the problem linearized at
  the last step, one step of :math:`d^2 < \mathrm{tol} \cdot n` from the mode.
  :meth:`~fips.Posterior.chi2_tests` tests the model's innovation and residual
  against that Gaussian's covariances, so its p-values are approximate.

:attr:`~fips.Posterior.problem` is the problem you solved, with the model.
:meth:`~fips.Posterior.with_data` solves again, from the last linearization,
so :meth:`~fips.Posterior.trials` works; there is no
:meth:`~fips.Posterior.with_gamma` (the linearization depends on gamma), and
:meth:`~fips.Posterior.recovery`, whose exact answer needs an estimate linear
in the data, points to ``trials``.

A statement about the flux, not the log scale factor, goes through the model's
transform: the flux at the mode is :math:`r\, e^{\hat{x}}`, and the total from
draws of the Gaussian at the mode:

.. ipython:: python

   draws = post.sample(4000, rng=2)
   totals = (inventory.values[:, None] * np.exp(draws.values)).sum(axis=0)
   flux_at_mode = (inventory * np.exp(post.estimate)).values.sum()
   pd.Series({"truth": (inventory * np.exp(truth)).values.sum(), "at the mode": flux_at_mode, "mean of draws": totals.mean(), "std of draws": totals.std()}).round(2)


When it is wrong
----------------

The Gaussian at the mode is the posterior of a linear problem. For a model that
bends over the posterior's spread, the posterior is skewed, and that Gaussian
covers the truth less often than it claims. :meth:`~fips.Posterior.trials`
measures it in the problem's own world: each trial draws a truth from the
prior, makes its data through the model, and solves them. Here with a prior
standard deviation of 0.3 (a factor of 1.35 either way) and of 1.5 (a factor
of 4.5), with noisier data. A model that bends this much makes Gauss-Newton
slow (each step a constant factor of the last), so the strong world gets more
steps:

.. ipython:: python

   def own_world(std):
       """Coverage and the mean standardized error in the problem's own world, over 400 trials."""
       _, obs = build(fips.Vector(np.zeros(25), observed), std, 0.5).simulate(rng=3)
       post = fips.solve(build(obs, std, 0.5), "gauss_newton", max_iterations=200)
       table = post.trials(n=400, rng=4)
       z = (table["estimate"] - table["truth"]) / table["posterior_std"]
       return {"coverage_68": (z.abs() <= 1.0).mean(), "coverage_95": (z.abs() <= 1.96).mean(), "mean_z": z.mean(), "std_z": z.std()}

   pd.DataFrame({std: own_world(std) for std in (0.3, 1.5)}).T.round(3)

At 0.3 the model is nearly linear over the posterior, and the Gaussian at the
mode covers as it claims, at both levels; the errors lean a little one way
(about a seventh of a standard deviation: the mode is not the mean of a
posterior that is skewed at all). At 1.5 its 95% intervals cover about 92% of
the time, the errors lean twice as far and spread 15% wider than the
Gaussian says: it is narrower than the posterior on its long side. The same
table grouped by cell, or over totals (``W=``), says where it matters.
