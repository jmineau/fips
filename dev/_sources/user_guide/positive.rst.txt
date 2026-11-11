.. _positive:

Values that stay positive
=========================

Some quantities cannot be negative (a flux, a concentration, a rate), yet the
Gaussian posterior of a problem that estimates them often is where the prior
is small and the observations see little. One way to keep the state positive
is to hold it to bounds (:doc:`bounds`). The other, on this page: **solve for
a variable** :math:`u` **that can be anything, and make the quantity a
positive function of it.** :class:`fips.LogNormalForward` makes it a factor of
a reference (a prior estimate), :math:`r\, e^{u}`;
:class:`fips.SoftplusForward` makes it :math:`u` itself where it is large,
bending to stay above zero near zero. Each is a forward model
(:doc:`nonlinear`): the state, its prior and its covariance are in
:math:`u`'s units, ``model.prior_for`` gives the prior on :math:`u` whose
transformed value has the mean and standard deviation you mean,
``fips.solve(problem, "gauss_newton")`` finds the mode and draws the
posterior, and ``model.transform`` maps the estimate, or the draws, back to
the value.

The example on this page is a methane flux inversion: the state is the flux
of each cell, and the reference an emission inventory.

.. contents:: On this page
   :local:
   :depth: 1


The lognormal model
-------------------

Forty cells along a line, an inventory with three empty cells, and thirty
observations, each of a few neighboring cells over a background of 1900 ppb:

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   rng = np.random.default_rng(0)
   cells = fips.Space({"flux": pd.RangeIndex(40, name="cell")})
   observed = fips.Space({"ch4": pd.RangeIndex(30, name="obs_id")})

.. ipython:: python

   footprint = np.zeros((30, 40))
   for i, start in enumerate(rng.integers(0, 34, 30)):
       footprint[i, start : start + rng.integers(2, 7)] = rng.uniform(0.5, 1.5)

.. ipython:: python

   H = fips.Operator(footprint, observed, cells)
   inventory = pd.Series(rng.lognormal(0.0, 0.8, 40), index=cells.select("flux"))
   inventory.iloc[[5, 17, 29]] = 0.0
   model = fips.LogNormalForward(H, inventory)
   model

The state on the ``flux`` block (the state's one block here; with a bias block
too, name it with ``block="flux"``) is :math:`u`, the logarithm of the factor
that scales each cell's inventory :math:`r`, and the model is :math:`F(x) =
H\, (r\, e^{u})`. Other blocks pass through: :math:`H` is the forward operator
on the whole state, and the columns of a bias block are used as they are. The
Jacobian is :math:`H` with each column of the block scaled by its value, in the
form of :math:`H`, so a sparse footprint stays sparse.


Choosing the prior on u
-----------------------

A prior standard deviation of 0.5 on :math:`u` is a factor of :math:`e^{0.5} =
1.65` either way. What it means for the value depends on where the prior on
:math:`u` is centered, and there are two conventions:

- **the median**: :math:`u \sim N(0, \sigma^2)`. The inventory is the prior's
  median flux, and its mean is :math:`e^{\sigma^2/2}` times the inventory: 1.65
  times for :math:`\sigma = 1`, so the prior's total is 65% above the
  inventory's;
- **the mean**: :math:`u \sim N(-\sigma^2/2, \sigma^2)`. The inventory is the
  prior's mean flux, and its total the inventory's.

A reference is usually an estimate of the mean (an inventory is), and a total
is a sum of means, so fips's helper takes the mean.
:meth:`~fips.LogNormalForward.prior_for` takes the prior mean :math:`m` and
standard deviation :math:`s` of each element's value and returns the :math:`u`
prior whose value has exactly those:
:math:`\sigma^2 = \log(1 + s^2/m^2)` and :math:`\mu = \log(m/r) - \sigma^2/2`.
An uncertainty of 100% of the inventory is :math:`\sigma = \sqrt{\log 2} =
0.83`:

.. ipython:: python

   mu, sigma = model.prior_for(mean=inventory, std=inventory)
   pd.DataFrame({"inventory": inventory, "mu": mu.to_series("flux"), "sigma": sigma.to_series("flux")}).iloc[3:7].round(3)

The correlations are those of :math:`u`. Write the covariance with them as for
any state (``fips.Exponential("cell", 3.0).with_std(sigma)``), and know that
the values' correlations are weaker: two elements whose :math:`u` correlate
by :math:`\rho` have values that correlate by

.. math::

    \frac{e^{\rho \sigma_i \sigma_j} - 1}{\sqrt{(e^{\sigma_i^2} - 1)(e^{\sigma_j^2} - 1)}},

0.38 for :math:`\rho = 0.5` and :math:`\sigma = 1`.


Solving and reading the flux
----------------------------

The problem, with data made from a truth drawn from that prior:

.. ipython:: python

   def build(obs, model=model, mu=mu, sigma=sigma):
       """The problem of the observations `obs`: the state is u, its prior from prior_for."""
       return fips.Problem(
           prior=mu,
           prior_cov=fips.Exponential("cell", 3.0).with_std(sigma),
           obs=obs,
           obs_cov=fips.Independent().with_std(0.2),
           forward=model,
           offset=1900.0,
       )

   truth, obs = build(fips.Vector(np.zeros(30), observed)).simulate(rng=1)
   post = fips.solve(build(obs), "gauss_newton")
   post.iterations

For a transform, ``"gauss_newton"`` finds the mode by Newton's method: the
transform's second derivative is known, so each iteration also takes the step
of the cost's whole Hessian (the Gauss-Newton matrix less a diagonal term on the
block) and goes to whichever of the two estimates has the lower cost. Far from
the mode, where the Newton matrix is not positive definite, the step is
Gauss-Newton's. Where Gauss-Newton converges fast, as here, its steps are
mostly the lower and the iterations are the same; where it crawls (a wide
prior, below), Newton's take over near the mode and :math:`d^2` falls by
squares. The mode is the one Gauss-Newton alone would reach, and the Gaussian
at the mode is still the problem linearized there (:doc:`nonlinear`).

The estimate is the mode of :math:`u`. :meth:`~fips.LogNormalForward.transform`
maps it to the flux, :math:`r\, e^{\hat{u}}`, the flux at the mode: near each
cell's posterior **median**, not its mean (which is higher, by about
:math:`e^{\hat\sigma^2/2}`). :meth:`~fips.Posterior.sample` draws :math:`u` from
the posterior itself, one column per draw (by Hamiltonian Monte Carlo:
:doc:`nonlinear`), and ``transform`` maps each draw to the flux. A cell's
interval is the quantiles of its draws:

.. ipython:: python

   flux = model.transform(post.estimate).to_series("flux")
   draws = model.transform(post.sample(4000, rng=2))
   low, high = draws.quantile(0.025, axis=1), draws.quantile(0.975, axis=1)
   pd.DataFrame({"truth": model.transform(truth).to_series("flux"), "inventory": inventory, "at the mode": flux, "low": low.to_numpy(), "high": high.to_numpy()}, index=flux.index).iloc[3:7].round(2)

A total is a sum of lognormal fluxes, which has no closed form. The draws do
it: the total of each column is a draw of the total.

.. ipython:: python

   totals = draws.sum()
   pd.Series({"truth": model.transform(truth).values.sum(), "inventory": inventory.sum(), "sum at the mode": flux.sum(), "mean of draws": totals.mean(), "2.5%": totals.quantile(0.025), "97.5%": totals.quantile(0.975)}).round(1)

Sum the draws after the transform: ``post.sample(n, W=W)`` aggregates
:math:`u`, and the transform of a sum of log factors is not a total of fluxes.
:attr:`~fips.Posterior.variances` and :meth:`~fips.Posterior.cov` are those
of the kept draws (``draws=2000`` of them), in :math:`u`.


Elements with no reference
--------------------------

An element whose reference is zero has a zero value, whatever :math:`u` is:
its column of the Jacobian is zero, so no observation moves it, and the data
cannot make it positive. ``prior_for`` gives those elements :math:`\mu = 0`
and :math:`\sigma = 0` (a mean or standard deviation that is not zero there
raises); any prior there gives the other elements the same posterior, since a
variable that no observation sees changes nothing it is correlated with.

.. ipython:: python

   flux.iloc[[5, 17, 29]]

In a flux inversion, an inventory can have empty cells where sources could be
(a new well, a leak no inventory lists). To let the data put a value there,
give the element a reference (a small one grows by the prior's factor, so it
stays small), use :class:`~fips.SoftplusForward` with a prior mean above zero,
or hold the state to bounds instead (:doc:`bounds`).


The softplus model
------------------

:class:`fips.SoftplusForward` solves for the value in its own units, with
:math:`T(u) = s \log(1 + e^{u/s})`: where :math:`u` is several ``scale``
:math:`s` above zero the value is :math:`u`, and the problem is the linear
one; several below, the value is :math:`s\, e^{u/s}`, near zero and positive.
The uncertainty is additive away from zero (:math:`10 \pm 2`) and bends near
it. ``scale`` is in the value's units and says where the bend is: the
transform is not the same in other units, so it has no default: set it near
the prior standard deviation of the elements that could be near zero.

There is no closed form for the prior on :math:`u` that a prior on the value
corresponds to: :meth:`~fips.SoftplusForward.prior_for` computes the value's
mean and standard deviation for a Gaussian :math:`u` by quadrature and matches
them by Newton's method, to 1e-12. Far above the scale it returns the value's
own prior; near zero, a :math:`u` centered lower and wider. The empty cells
get a mean of a tenth of the smallest inventory, since a softplus value is never
zero:

.. ipython:: python

   softplus = fips.SoftplusForward(H, scale=0.5)
   mean = inventory.where(inventory > 0, 0.1 * inventory[inventory > 0].min())
   mu_s, sigma_s = softplus.prior_for(mean, 0.5 * mean)
   pd.DataFrame({"mean": mean, "std": 0.5 * mean, "mu": mu_s.to_series("flux"), "sigma": sigma_s.to_series("flux")}).iloc[[0, 5, 9, 12]].round(3)

.. ipython:: python

   post_s = fips.solve(build(obs, softplus, mu_s, sigma_s), "gauss_newton")
   softplus.transform(post_s.estimate).to_series("flux").iloc[3:7].round(2)

The same ``transform`` reads the median flux at the mode and maps draws.


Why a total comes from draws of the posterior
---------------------------------------------

The posterior of :math:`u` is not Gaussian, and the Gaussian at the mode (the
analytic posterior of the problem linearized at the mode,
``fips.solve(problem.linearized(post.estimate))``) can be far from it for a
function of many elements. Two elements whose total is observed closely, each with a
prior factor of :math:`e = 2.7` either way, show why:

.. ipython:: python

   two = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
   one = fips.Space({"ch4": pd.RangeIndex(1, name="obs_id")})
   pair = fips.LogNormalForward(fips.Operator(np.ones((1, 2)), one, two), 1.0)
   m, s = pair.prior_for(1.0, np.sqrt(np.e - 1.0))  # sigma = 1
   problem = fips.Problem(m, fips.Independent().with_std(s), fips.Vector([2.0], one), fips.Independent().with_std(0.05), pair)
   post2 = fips.solve(problem, "gauss_newton")
   gaussian = pair.transform(fips.solve(problem.linearized(post2.estimate)).sample(2000, rng=0)).sum()
   exact = pair.transform(post2.sample(2000, rng=0)).sum()
   pd.DataFrame({"Gaussian at the mode": gaussian.agg(["mean", "std"]), "posterior": exact.agg(["mean", "std"])}).round(2)

The data say the total is :math:`2.00 \pm 0.05`, and so do the posterior's
draws; the Gaussian at the mode's say :math:`2.6 \pm 0.9`. The states that give
that total lie on a curve, :math:`e^{u_a} + e^{u_b} = 2`; the Gaussian at the
mode lies along the curve's tangent, and on the tangent the sum of the
exponentials is never below 2. The values at the mode are right (1 and 1);
the Gaussian's draws are not. An observation of a sum of elements is this
case (in a flux inversion, a footprint sees a sum of cells), and it grows with
the prior's spread. The sampler uses
the Gaussian at the mode only to move (:doc:`nonlinear`), and its draws are of
the posterior.

In this page's problem, in its own world (each trial draws a truth from the
prior and makes its data through the model), with
:meth:`~fips.Posterior.with_data` for each trial's posterior and fewer kept
draws to keep the page quick:

.. ipython:: python

   def own_world(factor, trials=30):
       """The bias of the total at the mode, and how often the total's 95% interval holds the truth: from the posterior's draws, and from the Gaussian at the mode's."""
       mu, sigma = model.prior_for(inventory, factor * inventory)
       post = fips.solve(build(obs, model, mu, sigma), "gauss_newton", draws=200, burn_in=50)
       g = np.random.default_rng(3)
       rows = []
       for _ in range(trials):
           truth, data = post.problem.simulate(rng=g)
           trial = post.with_data(data)
           true = model.transform(truth).values.sum()
           posterior = model.transform(trial.sample(200, rng=g)).sum()
           at_mode = model.transform(fips.solve(trial.problem.linearized(trial.estimate)).sample(200, rng=g)).sum()
           rows.append([model.transform(trial.estimate).values.sum() - true, posterior.quantile(0.025) <= true <= posterior.quantile(0.975), at_mode.quantile(0.025) <= true <= at_mode.quantile(0.975)])
       bias, posterior, at_mode = np.mean(rows, axis=0)
       return {"bias at the mode": bias / inventory.sum(), "covered, posterior": posterior, "covered, Gaussian": at_mode}

.. ipython:: python

   pd.DataFrame({factor: own_world(factor) for factor in (0.3, 1.0)}).T.round(3)

Thirty trials know a coverage to about 0.04, so read the table for its
contrast: with a prior of 100% (a log standard deviation of 0.83) the Gaussian
at the mode's interval misses the truth about one time in four, the posterior's
holds it about as often as it claims, and the sum of the values at the mode stays
unbiased with either prior.

In the larger world of :doc:`comparison` (400 cells, three sites,
three quarters of the cells with an averaging kernel below 0.1, a log standard
deviation of 1), over 200 trials with 1000 kept draws each, the 95% interval
of the total held the truth in 95.5% of them for ``LogNormalForward`` and 96%
for ``SoftplusForward`` (21% and 40% from the Gaussian at the mode), and each
element's 95% interval, the quantiles of its draws (as
:meth:`~fips.Posterior.trials` reads it), in 94.7% and 94.6%. Read
symmetrically about the mode with the draws' standard deviation, an element's
interval held the truth in 92%: the posterior is skewed there.

So: report the total and its interval from the posterior's draws, and each
element's interval from its draws (or from :attr:`~fips.Posterior.variances`).
The sum of the values at the mode is a point estimate of the total near its
median.


What a draw costs
-----------------

The draws are made at the first use of :attr:`~fips.Posterior.variances`,
:meth:`~fips.Posterior.cov` or :meth:`~fips.Posterior.sample`, after a
``burn_in`` of 200 that tunes the sampler's step, and kept. Each draw follows a
path of steps (one for a model that is linear, more where it bends: medians of
16 and 20 in :doc:`comparison`'s 400-cell world), and each step evaluates the
model and the product of its Jacobian's transpose with a vector (a transform
forms no Jacobian), and multiplies by the covariance of the Gaussian at the
mode. In that world (400 cells, 120 observations) a draw
took 5 to 10 ms (a step about 0.5 ms), so the 2200 of a solve 10 to 20 s, on
one core; the same footprint as a linear model, 0.7 ms (one step). A trial of
:meth:`~fips.Posterior.trials` draws its own. The draws are correlated: most
elements' have an autocorrelation time of 1 to 3 draws there (the slowest, tens),
so 2000 draws give a standard deviation to about 3%; ``draws=`` asks for more.
A solve logs a warning when some element's draws mix so slowly that its
standard deviation is uncertain by more than 10% (in the two elements above, along
the curve: the total, which the data constrain, mixes in a few draws).


Which to use
------------

- :class:`~fips.LogNormalForward` when the uncertainty is a **factor** of a
  reference (an inventory), the same in relative terms at small and large
  values. An element with no reference stays at zero.
- :class:`~fips.SoftplusForward` when the uncertainty is **additive**
  (:math:`10 \pm 2`, whatever the value's size) and the value is near zero
  only in some elements, where it bends instead of going below.
- ``"bounded"`` (:doc:`bounds`) when the prior is the Gaussian one and only
  the negative part is wrong: the posterior is the Gaussian truncated at zero,
  whose total is above the Gaussian's where the data see little.

:doc:`comparison` runs the three and the analytic posterior on one problem.

fips draws each posterior exactly, so in its own world a total's interval
covers.

Both transforms make the problem nonlinear, more so the wider the prior, and
the solve takes more steps: in :doc:`comparison`'s 400-cell world at a log
standard deviation of 1 (200 trials), a median of 7 and at most 14 for the
lognormal model, 9 and 14 for the softplus model (Gauss-Newton's steps alone
took 9 and 23, and 18 and 153). A solve logs a warning when its steps fall
slowly (:doc:`nonlinear`), and one that has not stopped after
``max_iterations`` (50) raises; raise it, or narrow the prior.
