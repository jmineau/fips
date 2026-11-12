.. _nonlinear:

A nonlinear forward model
=========================

When the observations are not a linear function of the state, the forward map
is a model :math:`F`, and the observations are :math:`z = F(x) + c + e`.
Write the model and its Jacobian as a :class:`~fips.ForwardModel`, and
``fips.solve(problem, "gauss_newton")`` solves the problem linearized at its
last estimate, again and again, until the estimate stops moving. **The
estimate is the mode** of the posterior. **The spread comes from draws** of
the posterior itself, which the solve keeps: Hamiltonian Monte Carlo, started
at the mode and steered by the Gaussian at the mode (the analytic posterior of
the problem linearized there).

.. contents:: On this page
   :local:
   :depth: 1


When to use it
--------------

Use ``"gauss_newton"`` when the forward map is a model whose Jacobian you can
write (or compute), and the problem fits in memory as the analytic one does.

Another method is better when:

- **the forward map is linear**: write it as an :class:`~fips.Operator` and use
  ``"analytic"`` (:doc:`posterior`). ``"gauss_newton"`` with an operator
  returns the analytic posterior itself;
- **the model makes a quantity positive** (a factor of a reference, or a value
  that bends away from zero): :class:`~fips.LogNormalForward` and
  :class:`~fips.SoftplusForward` are ready-made models solved this way
  (:doc:`positive`);
- **the model can only be run**, with no Jacobian: ``"ensemble"``
  (:doc:`ensemble`);
- **the problem is too large to factor**, or the Jacobian is known only by
  its products (a tangent linear and an adjoint model): ``"variational"``
  (:doc:`variational`).


An example
----------

As an example, a flux inversion solved for scale factors: the state
:math:`x` is the logarithm of a factor that scales each cell's reference flux
:math:`r` (an inventory), so the flux :math:`r\, e^{x}` cannot be negative.
The model is :math:`F(x) = H\, (r\, e^{x})` and its Jacobian is :math:`H
\operatorname{diag}(r\, e^{x})`. (fips has this model ready-made, with the
prior that goes with it: :class:`~fips.LogNormalForward`. Here it is
written by hand, to show how.)

Writing the model
~~~~~~~~~~~~~~~~~

A :class:`~fips.ForwardModel` is a class with two methods: ``__call__(x)``
returns :math:`F(x)` without the offset, a :class:`~fips.Vector` on the
observations' labels, and ``jacobian(x)`` returns :math:`J_{ij} = \partial
F_i / \partial x_j` at :math:`x`, an :class:`~fips.Operator` in any form.
Its ``__init__`` passes the observations' and the state's spaces to
``super().__init__(rows, cols)``. fips builds no Jacobian: it calls yours.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips
   from fips.linalg import DiagonalOperator

   class ScaledReference(fips.ForwardModel):
       """The observations of a reference scaled by exp(x): F(x) = H (r * exp(x))."""
       def __init__(self, H, reference):
           super().__init__(H.rows, H.cols)
           self.H = H
           self.reference = reference
       def __call__(self, x):
           return self.H @ (self.reference * np.exp(x))
       def jacobian(self, x):
           value = (self.reference * np.exp(x)).values
           return self.H @ fips.Operator(DiagonalOperator(value), x.space, x.space)

Ten cells and 25 observations. The prior is no change (:math:`x_0 = 0`), with
a standard deviation of 0.5 (a factor of about 1.6 either way) correlated
over two cells; the data come from a truth drawn from that prior:

.. ipython:: python

   rng = np.random.default_rng(0)
   cells = fips.Space({"flux": pd.RangeIndex(10, name="cell")})
   observed = fips.Space({"conc": pd.RangeIndex(25, name="obs_id")})
   footprint = rng.uniform(0.0, 1.0, (25, 10)) * (rng.uniform(size=(25, 10)) < 0.4)
   H = fips.Operator(footprint, observed, cells)
   reference = fips.Vector(rng.uniform(0.5, 2.0, 10), cells)
   model = ScaledReference(H, reference)

.. ipython:: python

   def build(obs):
       """The problem of the observations `obs`."""
       return fips.Problem(
           prior=fips.Vector(np.zeros(10), cells),
           prior_cov=fips.Exponential("cell", 2.0).with_std(0.5),
           obs=obs,
           obs_cov=fips.Independent().with_std(0.2),
           forward=model,
           offset=10.0,
       )

   truth, obs = build(fips.Vector(np.zeros(25), observed)).simulate(rng=1)
   problem = build(obs)

:meth:`~fips.Problem.predict` and :meth:`~fips.Problem.simulate` run the
model; :meth:`~fips.Problem.aggregate` follows it with the aggregation.
:meth:`~fips.Problem.restrict` refuses a model: holding elements at their
prior puts what they add to the observations into the offset, which is a
constant only for a linear forward map.

Checking the Jacobian
~~~~~~~~~~~~~~~~~~~~~

**A wrong Jacobian still gives an answer, so check it first.**
:func:`fips.taylortest` steps away from a state and compares the model with
its tangent: the remainder falls as the square of the step when the Jacobian
is the derivative, and only as the step when it is not. It also checks that
the Jacobian's transpose is its adjoint. A Jacobian that forgot the reference
fails:

.. ipython:: python

   fips.taylortest(model, problem.prior)

   class Forgot(ScaledReference):
       def jacobian(self, x):
           scale = fips.Operator(DiagonalOperator(np.exp(x.values)), x.space, x.space)
           return self.H @ scale

   fips.taylortest(Forgot(H, reference), problem.prior)

It checks one direction (a random one by default; ``dx=`` sets it) at the
state you give: check at the prior and again at the estimate.

One linearization
~~~~~~~~~~~~~~~~~

:meth:`problem.linearized(x) <fips.Problem.linearized>` is the linear problem
at a state :math:`x`: the forward operator :math:`J(x)` and the offset
:math:`c + F(x) - J(x)\, x`. Its analytic solve is one Gauss-Newton step from
:math:`x`. ``fips.solve(problem)`` refuses a model rather than linearize it
at the prior without saying so:

.. ipython:: python

   linear = problem.linearized(problem.prior)
   one_step = fips.solve(linear)
   one_step.estimate.values.round(2)

Solving
~~~~~~~

``"gauss_newton"`` starts at the prior and takes that step from each new
estimate. :attr:`~fips.Posterior.iterations` has each step's size
:math:`d^2` (the step measured in the posterior covariance) and the cost
after it; the solve stops when :math:`d^2` is below ``tol`` times the number
of state elements:

.. ipython:: python

   post = fips.solve(problem, "gauss_newton")
   post.iterations
   post.estimate.values.round(2)

The first step went most of the way. The next ones still moved some cells
by more than two posterior standard deviations, which one linearization
would have missed:

.. ipython:: python

   moved = (post.estimate.values - one_step.estimate.values) / np.sqrt(post.variances.values)
   np.abs(moved).max().round(2)

**A statement about the value, not the state, comes from the draws.** The
flux at the mode is :math:`r\, e^{\hat{x}}`; the total flux and its spread
come from mapping each draw of the posterior to the flux, then summing:

.. ipython:: python

   draws = post.sample(4000, rng=2)
   totals = (reference.values[:, None] * np.exp(draws.values)).sum(axis=0)
   pd.Series(
       {
           "truth": (reference * np.exp(truth)).values.sum(),
           "at the mode": (reference * np.exp(post.estimate)).values.sum(),
           "mean of draws": totals.mean(),
           "std of draws": totals.std(),
       }
   ).round(2)


What the posterior reports
--------------------------

A ``"gauss_newton"`` posterior is a *sampled* one: the
:ref:`members table <posterior-members>` lists what it gives. Its members
come from three places:

- **the model at the mode.** :attr:`~fips.Posterior.estimate` is the mode.
  :attr:`~fips.Posterior.modeled_obs` is :math:`F(\hat{x}) + c`; the
  residuals, :attr:`~fips.Posterior.rmse`, :attr:`~fips.Posterior.r2`,
  :attr:`~fips.Posterior.chi2` (the cost at the mode) and the misfits are
  the model's.
- **the kept draws of the posterior.** :attr:`~fips.Posterior.mean`,
  :attr:`~fips.Posterior.variances`, :meth:`~fips.Posterior.cov` (of the
  state, or of aggregates :math:`W x`) and the ``lower`` and ``upper`` of
  :meth:`~fips.Posterior.uncertainty` and :meth:`~fips.Posterior.trials`
  (the draws' quantiles) are read from ``draws`` kept draws, made from
  ``seed`` the first time one is needed. :meth:`~fips.Posterior.sample`
  makes new ones.
- **the Gaussian at the mode**, the analytic posterior of the problem
  linearized at the last step: the gain, the averaging kernel, DOFS, the
  information content and the innovation covariance.
  :meth:`~fips.Posterior.chi2_tests` tests the model's innovation and
  residuals against that Gaussian's covariances, so its p-values are
  approximate. To read it
  whole, solve it: ``fips.solve(problem.linearized(post.estimate))``.

:attr:`~fips.Posterior.iterations` has the steps.
:meth:`~fips.Posterior.with_data` solves again from the last linearization
and draws again, so :meth:`~fips.Posterior.trials` works; it runs the sampler
in each trial. ``log_evidence`` is Laplace's, from the Gaussian at the mode.
:meth:`~fips.Posterior.with_gamma` raises (the linearization depends on
gamma), and so does :meth:`~fips.Posterior.recovery`, which needs an
estimate linear in the data: :meth:`~fips.Posterior.trials` measures what it
would (:doc:`synthetic`).

**Read an interval as the draws' quantiles.** In the problem's own world
(truths drawn from the prior, data made through the model), those intervals
hold the truth about as often as they claim, for any model. Read as the mode
plus or minus a multiple of the standard deviation, an interval is not the
posterior's where the posterior is skewed.


Options
-------

Each is a keyword of ``fips.solve(problem, "gauss_newton", ...)``
(:class:`~fips.methods.gauss_newton.GaussNewtonOptions`).

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Option
     - Default
     - What it does
   * - ``tol``
     - ``1e-6``
     - The iterations stop when a step's :math:`d^2` is below ``tol`` times
       the number of state elements. At the default the last step is about a
       thousandth of a posterior standard deviation.
   * - ``max_iterations``
     - 50
     - The most steps to take; a solve that has not stopped by then raises.
   * - ``draws``
     - 2000
     - The number of draws of the posterior kept for the variances, the
       covariances, the mean and the intervals, from all the chains together.
       Fewer make each trial of :meth:`~fips.Posterior.trials` quicker.
   * - ``burn_in``
     - 200
     - The draws each chain makes from its start and leaves out before those
       it keeps, while it tunes its step.
   * - ``seed``
     - 0
     - The seed of the starts, the tuning and the kept draws.
   * - ``chains``
     - 4
     - The number of chains the draws are split among, each from its own
       draw of the Gaussian at the mode. The solve warns when they disagree.
       ``chains=1`` is the sampler of 0.2.0b2, one chain from the mode, which
       cannot be checked this way.
   * - ``gamma``
     - 1.0
     - The regularization factor: the observation covariance is divided by
       it, as for the analytic method.

There is no option to skip the draws. The Gaussian at the mode, which is not
the posterior where the model bends, is
``fips.solve(problem.linearized(post.estimate))``.


Warnings and checks
-------------------

Each warning is a :class:`~fips.FipsWarning`, which Python shows without any
setup, and is also logged (:ref:`repeated-inversions`). The solve warns
when:

- **it converged slowly**: each step was a large fraction of the last, so the
  estimate may be further from the mode than ``tol`` meant. Lower ``tol``.
  The model bends much over the posterior's spread, which is also where the
  Gaussian at the mode is least right.
- **the sampler takes 200 or more steps a draw**: the posterior bends very
  much over its spread, or the Jacobian is not the derivative of the model.
  A wrong Jacobian leaves the draws exact (the sampler's acceptance test
  corrects for it) but makes each draw cost hundreds of steps. Check it with
  ``fips.taylortest(problem.forward, post.estimate)``.
- **the chains disagree**: an element's split R-hat, which compares the
  spread between the chains with the spread within each, is above 1.05. One
  chain stayed out of a part of the posterior that another reached, so the
  variances are not yet the posterior's. It names the worst element and its
  standard deviation in two chains; raise ``draws``, or read the aggregates
  the data constrain, which mix faster.
- **the kept draws mix slowly**: a standard deviation from them is uncertain
  by more than 10%. It says how many draws would do; raise ``draws``, or read
  the aggregates the data constrain, which mix faster.

**Chains that agree are not proof.** On a posterior that bends along a curve
(an observation of a sum of elements, each with a wide prior), each draw
moves about one width of the Gaussian at the mode, and the curve's far arms
lie several widths along it, so every chain can miss the same tail. In the
guide's two-element examples the chains disagree at the default 2000 draws
(every seed tried, 2026-11-11), but at 20,000 draws an element's standard
deviation was still 28% to 40% low with no warning in half the seeds. The aggregates
the observations constrain stay right. When an element's own spread along
such a curve matters, compare it with another estimate (a grid, for a few
elements). Calls to :meth:`~fips.Posterior.sample` continue the chains from
where their burn-ins ended, so draws from several calls are not a check.

At the INFO level it logs the number of steps it took, and the sampler's
tuned step and acceptance rate.

It raises:

- :class:`~fips.ConvergenceError` (a :class:`RuntimeError`) if it has not
  stopped after ``max_iterations``,
  naming each step's :math:`d^2`. If they are still falling, raise
  ``max_iterations``; if the whole steps do not shrink while the steps taken
  do, check the Jacobian; a very wide prior can also be too nonlinear to
  solve from the prior;
- :class:`RuntimeError` if the cost rises along a step however short: the
  Jacobian is not the derivative of the model, or the model does not give
  finite observations near the estimate;
- :class:`NotImplementedError` if the model has no ``jacobian``.

``fips.solve(problem)`` (the analytic method) and ``"bounded"`` raise
:class:`TypeError` for a problem with a model. A member a sampled posterior
cannot give raises :class:`~fips.UnavailableError`.


Cost
----

**Each step costs an analytic solve** of the problem linearized at the last
estimate, plus a few runs of the model. A mildly nonlinear problem stops in a
few steps; a strongly nonlinear one (a wide prior on a model that bends) can
take tens.

**The draws cost more**, and are made only when a spread is first asked for:
``draws`` of them, plus ``burn_in`` for each of the ``chains``. Each draw follows a path of steps (one for
a linear model, more the more it bends), and each step runs the model, a
product with its Jacobian's transpose and one with the covariance of the
Gaussian at the mode: thousands to tens of thousands of model runs in all.
:meth:`~fips.Posterior.trials` draws in every trial: keep ``draws`` small
there.


How it works
------------

**The mode.** The estimate is the state where the cost

.. math::

    \chi^2(x) = (x - x_0)^T S_0^{-1} (x - x_0)
        + \gamma\, (z - F(x) - c)^T S_z^{-1} (z - F(x) - c)

is least. Gauss-Newton finds it as Rodgers (2000, chapter 5) does: from the
prior, each step solves the problem linearized at the last estimate
:math:`x_i`, with the Jacobian :math:`J_i = J(x_i)` and the gain
:math:`K_i` of that linear problem,

.. math::

    x_{i+1} = x_0 + K_i \big(z - F(x_i) - c + J_i (x_i - x_0)\big).

**The stopping rule** is Rodgers': the step, measured in the posterior
covariance of the solve that made it, is small against the number of state
elements :math:`n`:

.. math::

    d_i^2 = (x_{i+1} - x_i)^T \hat{S}_i^{-1} (x_{i+1} - x_i)
        < \mathrm{tol} \cdot n, \qquad
    \hat{S}_i^{-1} = S_0^{-1} + \gamma J_i^T S_z^{-1} J_i.

Neither :math:`\hat{S}_i^{-1}` nor :math:`S_0^{-1}` is formed: every estimate
is :math:`x_0 + S_0 y`, so the prior term of :math:`d^2` is :math:`\Delta y^T
S_0 \Delta y`, and a prior covariance that is singular works. A step that
would raise the cost is halved until it does not; only a whole step below the
rule stops the iterations. The rule measures the step, not the distance to
the mode: near the mode, :math:`d^2` falls by orders of magnitude a step,
and when it falls only by a constant factor the solve warns.

**Newton's steps, for a transform.** For :class:`~fips.LogNormalForward` and
:class:`~fips.SoftplusForward` fips knows the second derivatives, and each
iteration also takes Newton's step, with the cost's whole Hessian. It goes to
whichever of the two estimates has the lower cost. Where Gauss-Newton's steps
crawl, Newton's converge much faster near the mode. The mode is the same, and
the Gaussian at the mode is still that of the problem linearized there. A
model you write gives only its Jacobian, so its steps are Gauss-Newton's.

**The draws** are Hamiltonian Monte Carlo, split at the Gaussian at the mode
(Shahbaba et al., 2014). Each draw starts from the last with a random
velocity and follows a path on which half the cost is a potential energy; a
Metropolis test accepts the path's end or keeps the start, which makes the
posterior, :math:`p(x \mid z) \propto e^{-\chi^2(x)/2}`, the chain's
distribution exactly, whatever the paths are. Each draw is then exact; how
well a finite chain covers the posterior depends on how well it mixes
(`Warnings and checks`_). The Gaussian at the mode
steers the paths in two ways. The velocities are its draws less its mean. The
part of the cost that it accounts for is followed exactly, as a rotation
about the mode,

.. math::

    x(t) = \hat{x} + (x - \hat{x}) \cos t + v \sin t,

for a quarter period (after which a draw of that Gaussian is independent of
its start), and only the rest of the cost is followed in steps. The burn-in
tunes the step's length to accept 90% of the paths (Hoffman and Gelman,
2014). A linear model has no rest: each path is a quarter turn of one step,
always accepted, and the draws are exact and independent. A path the model
cannot follow (it overflows) is rejected.


References
----------

Hoffman, M. D., and Gelman, A. (2014). The No-U-Turn sampler: adaptively
setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning
Research*, 15, 1593-1623.

Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and
Practice*, chapter 5. World Scientific. https://doi.org/10.1142/3171

Shahbaba, B., Lan, S., Johnson, W. O., and Neal, R. M. (2014). Split
Hamiltonian Monte Carlo. *Statistics and Computing*, 24(3), 339-349.
https://doi.org/10.1007/s11222-012-9373-1
