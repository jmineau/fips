.. _positive:

Fluxes that stay positive
=========================

A flux of methane cannot be negative, and the Gaussian posterior of a flux
inversion often is where the prior is small and the observations see little.
One way to keep it positive is to hold the state to bounds (:doc:`bounds`).
The other, on this page: **solve for a variable** :math:`u` **that can be
anything, and make the flux a positive function of it.**
:class:`fips.LognormalForward` makes the flux a factor of an inventory,
:math:`r\, e^{u}`; :class:`fips.SoftplusForward` makes it :math:`u` itself
where it is large, bending to stay above zero near zero. Each is a forward
model (:doc:`nonlinear`): the state, its prior and its covariance are in
:math:`u`'s units, ``model.prior_for`` gives the prior on :math:`u` whose flux
has the mean and standard deviation you mean, ``fips.solve(problem,
"gauss_newton")`` finds the mode, and ``model.transform`` maps the estimate, or
draws of the posterior, back to the flux.

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
   model = fips.LognormalForward(H, inventory)
   model

The state on the ``flux`` block (the state's one block here; with a bias block
too, name it with ``block="flux"``) is :math:`u`, the logarithm of the factor
that scales each cell's inventory :math:`r`, and the model is :math:`F(x) =
H\, (r\, e^{u})`. Other blocks pass through: :math:`H` is the forward operator
on the whole state, and the columns of a bias block are used as they are. The
Jacobian is :math:`H` with each column of the block scaled by its flux, in the
form of :math:`H`, so a sparse footprint stays sparse.


Choosing the prior on u
-----------------------

A prior standard deviation of 0.5 on :math:`u` is a factor of :math:`e^{0.5} =
1.65` either way. What it means for the flux depends on where the prior on
:math:`u` is centered, and there are two conventions:

- **the median**: :math:`u \sim N(0, \sigma^2)`. The inventory is the prior's
  median flux, and its mean is :math:`e^{\sigma^2/2}` times the inventory: 1.65
  times for :math:`\sigma = 1`, so the prior's total is 65% above the
  inventory's;
- **the mean**: :math:`u \sim N(-\sigma^2/2, \sigma^2)`. The inventory is the
  prior's mean flux, and its total the inventory's.

An inventory is an estimate of the mean flux, and a basin total is a sum of
means, so fips's helper takes the mean. :meth:`~fips.LognormalForward.prior_for`
takes the flux's prior mean :math:`m` and standard deviation :math:`s` in each
cell and returns the :math:`u` prior whose flux has exactly those:
:math:`\sigma^2 = \log(1 + s^2/m^2)` and :math:`\mu = \log(m/r) - \sigma^2/2`.
An uncertainty of 100% of the inventory is :math:`\sigma = \sqrt{\log 2} =
0.83`:

.. ipython:: python

   mu, sigma = model.prior_for(mean=inventory, std=inventory)
   pd.DataFrame({"inventory": inventory, "mu": mu.to_series("flux"), "sigma": sigma.to_series("flux")}).iloc[3:7].round(3)

The correlations are those of :math:`u`. Write the covariance with them as for
any state (``fips.Exponential("cell", 3.0).with_std(sigma)``), and know that
the flux's correlations are weaker: two cells whose :math:`u` correlate by
:math:`\rho` have fluxes that correlate by

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
   len(post.iterations)

The estimate is the mode of :math:`u`. :meth:`~fips.LognormalForward.transform`
maps it to the flux, :math:`r\, e^{\hat{u}}`: the **median** of each cell's
flux under the Gaussian at the mode, not its mean (which is :math:`e^{\hat
\sigma^2/2}` times it). The transform is monotone, so a cell's interval in
:math:`u` maps to its interval of the flux:

.. ipython:: python

   flux = model.transform(post.estimate).to_series("flux")
   low, high = (model.transform(post.estimate + k * np.sqrt(post.variances)) for k in (-1.96, 1.96))
   pd.DataFrame({"truth": model.transform(truth).to_series("flux"), "inventory": inventory, "at the mode": flux, "low": low.to_series("flux"), "high": high.to_series("flux")}).iloc[3:7].round(2)

A total is a sum of lognormal fluxes, which has no closed form. Draws do it:
:meth:`~fips.Posterior.sample` draws :math:`u` from the Gaussian at the mode,
one column per draw, ``transform`` maps each draw to the flux, and the total of
each column is a draw of the total:

.. ipython:: python

   draws = model.transform(post.sample(4000, rng=2))
   totals = draws.sum()
   pd.Series({"truth": model.transform(truth).values.sum(), "inventory": inventory.sum(), "sum at the mode": flux.sum(), "mean of draws": totals.mean(), "2.5%": totals.quantile(0.025), "97.5%": totals.quantile(0.975)}).round(1)

Sum the draws after the transform: ``post.sample(n, W=W)`` aggregates
:math:`u`, and the transform of a sum of log factors is not a total of fluxes.
Read the section below before you report that interval.


Cells with no reference
-----------------------

A cell whose inventory is zero has a zero flux, whatever :math:`u` is: its
column of the Jacobian is zero, so no observation moves it, and the inversion
can put no source there. ``prior_for`` gives those cells :math:`\mu = 0` and
:math:`\sigma = 0` (a mean or standard deviation that is not zero there
raises); any prior there gives the other cells the same posterior, since a
variable that no observation sees changes nothing it is correlated with.

.. ipython:: python

   flux.iloc[[5, 17, 29]]

Salt Lake Valley inventories have empty cells where sources could be (a new
well, a leak no inventory lists). To let the data put a source there, give the
cell a reference (a small one grows by the prior's factor, so it stays small),
use :class:`~fips.SoftplusForward` with a prior mean above zero, or hold the
flux to bounds instead (:doc:`bounds`).


The softplus model
------------------

:class:`fips.SoftplusForward` solves for the flux in its own units, with
:math:`T(u) = s \log(1 + e^{u/s})`: where :math:`u` is several ``scale``
:math:`s` above zero the flux is :math:`u`, and the problem is the linear one;
several below, the flux is :math:`s\, e^{u/s}`, near zero and positive. The
uncertainty is additive away from zero (a flux of :math:`10 \pm 2`) and bends
near it. ``scale`` is in the flux's units and says where the bend is: the
transform is not the same in other units, so set it, near the prior standard
deviation of the cells that could be near zero.

There is no closed form for the prior on :math:`u` that a flux prior
corresponds to: :meth:`~fips.SoftplusForward.prior_for` computes the flux's
mean and standard deviation for a Gaussian :math:`u` by quadrature and matches
them by Newton's method, to 1e-12. Far above the scale it returns the flux's
own prior; near zero, a :math:`u` centered lower and wider. The empty cells
get a mean of a tenth of the smallest inventory, since a softplus flux is never
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


When the Gaussian at the mode misreads a total
----------------------------------------------

The posterior of :math:`u` is not Gaussian, and the Gaussian at the mode can be
far from it for a function of many cells. Two cells whose total is observed
closely, each with a prior factor of :math:`e = 2.7` either way, show why:

.. ipython:: python

   two = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
   one = fips.Space({"ch4": pd.RangeIndex(1, name="obs_id")})
   pair = fips.LognormalForward(fips.Operator(np.ones((1, 2)), one, two), 1.0)
   m, s = pair.prior_for(1.0, np.sqrt(np.e - 1.0))  # sigma = 1
   problem = fips.Problem(m, fips.Independent().with_std(s), fips.Vector([2.0], one), fips.Independent().with_std(0.05), pair)
   totals = pair.transform(fips.solve(problem, "gauss_newton").sample(20000, rng=0)).sum()
   round(totals.mean(), 2), round(totals.std(), 2)

The data say the total is :math:`2.00 \pm 0.05`; the draws say :math:`2.6 \pm
0.9`. The states that give that total lie on a curve, :math:`e^{u_a} +
e^{u_b} = 2`; the Gaussian at the mode lies along the curve's tangent, and on
the tangent the sum of the exponentials is never below 2. The flux at the
mode is right (1 and 1); the draws are not. A footprint sees a sum of cells, so
this is the case of a flux inversion, and it grows with the prior's spread.
In this page's problem, in its own world (each trial draws a truth from the
prior and makes its data through the model), with :meth:`~fips.Posterior.with_data`
for each trial's posterior:

.. ipython:: python

   def own_world(factor, trials=200):
       """The bias of two totals and the coverage of the draws' 95% interval, for a prior standard deviation of `factor` times the inventory."""
       mu, sigma = model.prior_for(inventory, factor * inventory)
       post = fips.solve(build(obs, model, mu, sigma), "gauss_newton")
       g = np.random.default_rng(3)
       rows = []
       for _ in range(trials):
           truth, data = post.problem.simulate(rng=g)
           trial = post.with_data(data)
           totals = model.transform(trial.sample(400, rng=g)).sum()
           true = model.transform(truth).values.sum()
           rows.append([model.transform(trial.estimate).values.sum() - true, totals.mean() - true, totals.quantile(0.025) <= true <= totals.quantile(0.975)])
       bias_mode, bias_draws, covered = np.mean(rows, axis=0)
       return {"bias, sum at the mode": bias_mode / inventory.sum(), "bias, mean of draws": bias_draws / inventory.sum(), "95% coverage": covered}

.. ipython:: python

   pd.DataFrame({factor: own_world(factor) for factor in (0.3, 1.0)}).T.round(3)

With a prior of 30% the draws' interval covers about as it claims. With 100%
(a log standard deviation of 0.83) the mean of the draws is above the truth on
average and the interval misses it about one time in four, while the sum of
the flux at the mode stays unbiased. In a larger world shaped like the Salt
Lake Valley's (400 cells, three sites, three quarters of the cells with an
averaging kernel below 0.1, a log standard deviation of 1), the draws' 95%
interval held the true total in 29% of 200 trials for ``LognormalForward`` and
40% for ``SoftplusForward``, the mean of the draws was 30% high, and the sum
of the flux at the mode was within 3%; each cell's 95% interval held its truth
91 to 92% of the time.

So: report the **sum of the flux at the mode** as the total, read each
**cell's interval** from the Gaussian at the mode, and treat an interval of a
total from these draws as a lower bound on the uncertainty, to be checked in
your problem's own world with :meth:`~fips.Posterior.trials` and a loop like
the one above. For a total with an interval that covers, :doc:`bounds`
samples its posterior exactly.


Which to use
------------

- :class:`~fips.LognormalForward` when the uncertainty is a **factor** of an
  inventory, the same in relative terms at small and large fluxes. A cell with
  no inventory stays at zero.
- :class:`~fips.SoftplusForward` when the uncertainty is **additive** (a flux
  of :math:`10 \pm 2`, whatever its size) and the flux is near zero only in
  some cells, where it bends instead of going below.
- ``"bounded"`` (:doc:`bounds`) when the prior is the Gaussian one and only
  the negative part is wrong: the posterior is the Gaussian truncated at zero,
  and fips draws from it exactly, so a total's interval covers.

Both transforms make the problem nonlinear, more so the wider the prior:
Gauss-Newton takes more steps (in the Salt Lake Valley-shaped world at a log
standard deviation of 1, from the prior: a median of 9 and at most 38 for the
lognormal model; a median of 17 for the softplus model, and more than 50 in 3
of 200 trials), and logs a warning when its steps fall slowly
(:doc:`nonlinear`). A solve that has not stopped after
``max_iterations`` (50) raises; raise it, or narrow the prior.
