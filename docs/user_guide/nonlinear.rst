.. _nonlinear:

A nonlinear forward model
=========================

When the observations are not a linear function of the state (fluxes solved
as scale factors of an inventory, :math:`x \mapsto r\, e^{x}`; a sensor that
saturates), the forward map is a model :math:`F` and the observations are
:math:`z = F(x) + c + e`. The rule: **write the model and its Jacobian as a**
:class:`~fips.ForwardModel`, **and** ``fips.solve(problem, "gauss_newton")``
**solves the problem linearized at its last estimate, again and again, until the
estimate stops moving.** It ends at the mode of the posterior, and draws the
posterior itself for its uncertainty, by Hamiltonian Monte Carlo started at the
mode and steered by the Gaussian of the problem linearized there. That
Gaussian is the posterior when the model is close to linear over the
posterior's spread; when the model bends a lot it is not, and the draws still
are.

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
and its Jacobian :math:`H \operatorname{diag}(r\, e^{x})` (fips has it ready
made, with the prior that goes with it, as :class:`~fips.LognormalForward`:
:doc:`positive`):

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
logs a warning saying so. A model written by hand gives only its Jacobian, so
its steps are Gauss-Newton's; for :class:`~fips.LognormalForward` and
:class:`~fips.SoftplusForward`, whose second derivatives fips knows, the steps
are Newton's where those are better, and the solve takes about half as many
where Gauss-Newton's crawl (:doc:`positive`).


What the posterior reports
--------------------------

The **estimate** is the mode: the state where the cost :math:`(x - x_0)^T
S_0^{-1} (x - x_0) + \gamma (z - F(x) - c)^T S_z^{-1} (z - F(x) - c)` is least.
``post.mean`` raises. The rest comes from three places:

- **the model at the mode**: :attr:`~fips.Posterior.modeled_obs` is
  :math:`F(\hat{x}) + c`, and the residuals, :attr:`~fips.Posterior.rmse`,
  :attr:`~fips.Posterior.r2`, :attr:`~fips.Posterior.chi2` (the cost at the
  mode) and its two misfits are the model's;
- **draws of the posterior**, :math:`p(x \mid z) \propto e^{-\chi^2(x)/2}`:
  :attr:`~fips.Posterior.variances`, :meth:`~fips.Posterior.cov` (of the
  state, or of aggregates :math:`W x`), :meth:`~fips.Posterior.uncertainty`
  and :meth:`~fips.Posterior.sample`. The variances and covariances are those
  of ``draws`` (2000) kept draws, made at their first use from ``seed``, after
  a ``burn_in`` (200) that tunes the sampler;
- **the Gaussian at the mode**, the analytic posterior of the problem
  linearized at the last step, one step of :math:`d^2 < \mathrm{tol} \cdot n`
  from the mode: the gain, the averaging kernel, DOFS, the information
  content and the innovation covariance. :meth:`~fips.Posterior.chi2_tests`
  tests the model's innovation and residual against that Gaussian's
  covariances, so its p-values are approximate. To compare, that Gaussian is
  ``fips.solve(problem.linearized(post.estimate))``.

:attr:`~fips.Posterior.problem` is the problem you solved, with the model.
:meth:`~fips.Posterior.with_data` solves again, from the last linearization,
and draws again, so :meth:`~fips.Posterior.trials` works (running the sampler
in each trial: ``draws=`` fewer makes it quicker). This is a sampled
posterior, and :ref:`the table <posterior-members>` says what it cannot give:
no :meth:`~fips.Posterior.with_gamma` (the linearization depends on gamma),
and :meth:`~fips.Posterior.recovery`, whose exact answer needs an estimate
linear in the data, points to ``trials``.

A statement about the flux, not the log scale factor, goes through the model's
transform: the flux at the mode is :math:`r\, e^{\hat{x}}`, and the total from
draws of the posterior:

.. ipython:: python

   draws = post.sample(4000, rng=2)
   totals = (inventory.values[:, None] * np.exp(draws.values)).sum(axis=0)
   flux_at_mode = (inventory * np.exp(post.estimate)).values.sum()
   pd.Series({"truth": (inventory * np.exp(truth)).values.sum(), "at the mode": flux_at_mode, "mean of draws": totals.mean(), "std of draws": totals.std()}).round(2)


How the posterior is drawn
--------------------------

The sampler is Hamiltonian Monte Carlo: each draw starts from the last with a
random velocity and follows a path on which the cost is a potential energy,
and a Metropolis test accepts the path's end or keeps the start, which makes
the chain's draws those of the posterior exactly, whatever the paths are. The
paths are steered by the Gaussian at the mode, in two ways (Shahbaba et al.,
2014, split HMC): the velocities are its draws less its mean, and the part of
the cost that Gaussian accounts for is followed exactly, as a rotation about
the mode (a quarter turn makes a draw of that Gaussian independent of its
start); only the rest, the cost less the Gaussian's, is followed in steps,
whose length the ``burn_in`` tunes to accept 90% of the paths. A model that is
linear has no rest: every path is a quarter turn of one step and is accepted,
and the draws are exact and independent. The more the model bends over the
posterior's spread, the more steps a path takes (a median of 16 over 200 data
sets of a Salt Lake Valley-shaped flux problem with a log standard deviation
of 1, at most 38). No inverse of
the prior covariance is formed, so a prior with zero spread somewhere (a
cell with no inventory) works.

Each step evaluates the model and the product of its Jacobian's transpose with
a vector (``fips.LognormalForward`` and ``fips.SoftplusForward`` form no
Jacobian for it; a model of your own forms one with ``jacobian``), and one
product with the covariance of the Gaussian at the mode. The draws are
correlated, a few draws apart for most elements; a solve logs a warning when
some element's draws mix so slowly that its standard deviation is uncertain by
more than 10%, and ``draws=`` asks for more. The tuned step and the burn-in's
acceptance rate are logged at the INFO level. A wrong Jacobian does not make
the draws wrong (the Metropolis test corrects the paths), but the step falls
until a draw takes hundreds of steps, and the solve warns: check the Jacobian
with :func:`fips.taylortest`.

How well the draws cover
------------------------

In the problem's own world the draws' intervals should hold the truth as often
as they claim, for any model: each trial draws a truth from the prior, makes
its data through the model, and solves them. Here with a prior standard
deviation of 0.3 (a factor of 1.35 either way) and of 1.5 (a factor of 4.5),
noisier data, and fewer draws and trials than a study would use, to keep the
page quick; each cell's interval is the quantiles of its draws:

.. ipython:: python

   def own_world(std, trials=40):
       """How often each cell's 68% and 95% intervals from the draws hold the truth, in the problem's own world."""
       _, obs = build(fips.Vector(np.zeros(25), observed), std, 0.5).simulate(rng=3)
       post = fips.solve(build(obs, std, 0.5), "gauss_newton", max_iterations=200, burn_in=50)
       g = np.random.default_rng(4)
       inside = {0.68: [], 0.95: []}
       for _ in range(trials):
           truth, data = post.problem.simulate(rng=g)
           draws = post.with_data(data).sample(150, rng=g)
           for level in inside:
               low, high = draws.quantile(0.5 - level / 2, axis=1), draws.quantile(0.5 + level / 2, axis=1)
               inside[level].append(((low.values <= truth.values) & (truth.values <= high.values)).mean())
       return {f"coverage_{round(100 * level)}": np.mean(v) for level, v in inside.items()}

   pd.DataFrame({std: own_world(std) for std in (0.3, 1.5)}).T.round(3)

Both cover about as they claim (40 trials of 10 cells know a coverage to
about 0.02). :meth:`~fips.Posterior.trials` runs the same loop and reads each
trial's interval as the mode plus or minus a multiple of the draws' standard
deviation: where the posterior is skewed that is not the posterior's interval,
and at 1.5 it held the truth 93% of the time at 95% (the Gaussian at the mode,
read the same way, 92%). For a total, read the draws' quantiles too
(:doc:`positive`); :doc:`comparison` reads every method that way.
