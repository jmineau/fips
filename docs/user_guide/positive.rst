.. _positive:

Values that stay positive
=========================

Some quantities cannot be negative (a rate, a concentration, an amount), yet
the Gaussian posterior of a problem that estimates them often is negative
where the prior is small and the observations see little. One way to keep
them positive is to hold the state to bounds (:doc:`bounds`). The other, on
this page: **solve for a variable** :math:`u` **that can be anything, and
make the value a positive function of it**, :math:`T(u)`. Two forward models
do this:

- :class:`fips.LogNormalForward`: the value is a factor of a reference,
  :math:`r\, e^{u}`;
- :class:`fips.SoftplusForward`: the value is :math:`u` itself where it is
  large, and bends to stay above zero near zero.

The state, its prior and its covariance are in :math:`u`'s units.
``model.prior_for`` gives the prior on :math:`u` whose value has the mean and
standard deviation you mean. ``fips.solve(problem, "gauss_newton")`` finds
the mode of :math:`u` and draws its posterior (:doc:`nonlinear`).
``model.transform`` maps the estimate to the value at the mode, and the draws
to draws of the value.

.. contents:: On this page
   :local:
   :depth: 1


When to use it
--------------

- :class:`~fips.LogNormalForward` when the uncertainty is a **factor** of a
  reference ("a factor of 2"), the same in relative terms at small and large
  values. An element whose reference is zero stays zero.
- :class:`~fips.SoftplusForward` when the uncertainty is **additive**
  ("10 ± 2", whatever the value's size), and the value is near zero only in
  some elements, where it bends instead of going below.
- ``"bounded"`` (:doc:`bounds`) instead, when the Gaussian prior is the one
  you mean and only its negative part is wrong.
- The analytic posterior (``fips.solve(problem)``), when what you need is an
  aggregate of many elements: it ignores the bound, but its aggregate is
  often close to Gaussian, and it costs least.

Each puts a different prior on the state; :ref:`methods-positive` compares them, and :doc:`examples/comparison` runs them
on one problem.


An example
----------

As an example, a flux inversion: the value is the flux of each of forty
cells along a line, and the reference is an emission inventory with three
empty cells. Thirty observations each see a few neighboring cells:

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   rng = np.random.default_rng(0)
   cells = fips.Space({"flux": pd.RangeIndex(40, name="cell")})
   observed = fips.Space({"conc": pd.RangeIndex(30, name="obs_id")})
   footprint = np.zeros((30, 40))
   for i, start in enumerate(rng.integers(0, 34, 30)):
       footprint[i, start : start + rng.integers(2, 7)] = rng.uniform(0.5, 1.5)

.. ipython:: python

   H = fips.Operator(footprint, observed, cells)
   inventory = pd.Series(rng.lognormal(0.0, 0.8, 40), index=cells.select("flux"))
   inventory.iloc[[5, 17, 29]] = 0.0
   model = fips.LogNormalForward(H, inventory)
   model

The state on the ``flux`` block is :math:`u`, the logarithm of the factor
that scales each cell's inventory, and the model is :math:`F(x) = H\, (r\,
e^{u})`. :math:`H` is the forward operator on the whole state: with a bias
block too, name the transformed block with ``block="flux"``, and the bias
passes through as it is.

The prior on u
~~~~~~~~~~~~~~

:meth:`~fips.LogNormalForward.prior_for` takes the prior mean and standard
deviation of each element's **value** and returns the mean :math:`\mu` and
standard deviation :math:`\sigma` of :math:`u` that give exactly those. Here
the inventory is the prior mean, with an uncertainty of 100%:

.. ipython:: python

   mu, sigma = model.prior_for(mean=inventory, std=inventory)
   pd.DataFrame(
       {"inventory": inventory, "mu": mu.to_series("flux"), "sigma": sigma.to_series("flux")}
   ).iloc[3:7].round(3)

An uncertainty of 100% is :math:`\sigma = \sqrt{\log 2} = 0.83`, a factor of
2.3 either way. :math:`\mu` is below zero so that the prior **mean** of the
value is the inventory; a prior of :math:`\mu = 0` would make the inventory
the prior median instead, and the prior's total higher than the inventory's
(How it works, below). The correlations are those of :math:`u`: write the
covariance with them as for any state,
``fips.Exponential("cell", 3.0).with_std(sigma)``.

Solving and reading the value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
           offset=10.0,
       )

   truth, obs = build(fips.Vector(np.zeros(30), observed)).simulate(rng=1)
   post = fips.solve(build(obs), "gauss_newton")
   post.iterations

The estimate is the mode of :math:`u`. :meth:`~fips.LogNormalForward.transform`
maps it to the flux at the mode, which is near each cell's posterior median,
not its mean. The same ``transform`` maps draws of :math:`u`
(:meth:`~fips.Posterior.sample`, one column each) to draws of the flux, and a
cell's interval is the quantiles of its draws:

.. ipython:: python

   flux = model.transform(post.estimate).to_series("flux")
   draws = model.transform(post.sample(4000, rng=2))
   low, high = draws.quantile(0.025, axis=1).to_numpy(), draws.quantile(0.975, axis=1).to_numpy()
   true_flux = model.transform(truth).to_series("flux")
   table = pd.DataFrame({"truth": true_flux, "inventory": inventory, "at the mode": flux})
   table.assign(low=low, high=high).iloc[3:7].round(2)

**An aggregate of the value comes from the draws**: transform each draw,
then aggregate it. The total of each column is a draw of the total:

.. ipython:: python

   totals = draws.sum()
   pd.Series(
       {
           "truth": true_flux.sum(),
           "inventory": inventory.sum(),
           "sum at the mode": flux.sum(),
           "mean of draws": totals.mean(),
           "2.5%": totals.quantile(0.025),
           "97.5%": totals.quantile(0.975),
       }
   ).round(1)

Elements with no reference
~~~~~~~~~~~~~~~~~~~~~~~~~~

An element whose reference is zero has a value of zero, whatever :math:`u`
is. Its column of the Jacobian is zero, so no observation moves it.
``prior_for`` gives those elements :math:`\mu = 0` and :math:`\sigma = 0`;
any prior there gives the other elements the same posterior.

.. ipython:: python

   flux.iloc[[5, 17, 29]]

To let the data put a value there, give the element a reference (a small one
stays small, since it grows by the prior's factor), use
:class:`~fips.SoftplusForward` with a prior mean above zero, or use bounds
(:doc:`bounds`).

The softplus model
~~~~~~~~~~~~~~~~~~

:class:`fips.SoftplusForward` solves for the value in its own units: the
value is :math:`u` where :math:`u` is several ``scale`` above zero, and bends
to stay positive near zero. ``scale`` is in the value's units and says where
the bend is, so it has no default: set it near the prior standard deviation
of the elements that could be near zero. A softplus value is never zero, so
here the empty cells get a prior mean of a tenth of the smallest inventory:

.. ipython:: python

   softplus = fips.SoftplusForward(H, scale=0.5)
   mean = inventory.where(inventory > 0, 0.1 * inventory[inventory > 0].min())
   mu_s, sigma_s = softplus.prior_for(mean, 0.5 * mean)
   pd.DataFrame(
       {"mean": mean, "std": 0.5 * mean, "mu": mu_s.to_series("flux"), "sigma": sigma_s.to_series("flux")}
   ).iloc[[0, 5, 9, 12]].round(3)

Far above the scale, :math:`u`'s prior is the value's own; near zero it is
centered lower and wider. Solving and reading are the same as for the
lognormal model:

.. ipython:: python

   post_s = fips.solve(build(obs, softplus, mu_s, sigma_s), "gauss_newton")
   softplus.transform(post_s.estimate).to_series("flux").iloc[3:7].round(2)


What the posterior reports
--------------------------

The posterior is that of ``"gauss_newton"``, a *sampled* one, in the units of
:math:`u` (:doc:`nonlinear`, "What the posterior reports"; the
:ref:`members table <posterior-members>`). For the value:

- ``model.transform(post.estimate)`` is the **value at the mode**, near each
  element's posterior median. The mean is higher: for the lognormal model,
  by about :math:`e^{\hat\sigma^2/2}`, with :math:`\hat\sigma` the posterior
  standard deviation of :math:`u`.
- ``model.transform(post.sample(n))`` gives **draws of the value**. An
  element's interval is the quantiles of its draws; an aggregate is the
  aggregate of each draw, read off the draws.
- :attr:`~fips.Posterior.variances`, :meth:`~fips.Posterior.cov` and
  :meth:`~fips.Posterior.uncertainty` are those of :math:`u`, from the kept
  draws. ``post.sample(n, W=W)`` and :meth:`~fips.Posterior.trials`
  aggregate :math:`u`, which is not an aggregate of the value: write the loop
  over the draws yourself (:ref:`methods-transformed`).

The draws are of the posterior itself, not of the Gaussian at the mode, so in
the problem's own world a total's interval holds the truth about as often as
it claims. An element's own spread is harder for the chains: where the
observations see only a sum, the solve warns when its chains disagree, and
chains that agree can still all miss a far arm of the curve
(:doc:`nonlinear`, "Warnings and checks"). The Gaussian at the mode can be far from the posterior for a
total of many elements (How it works, below).


Options
-------

The options are the models' arguments. The solve's options (``tol``,
``max_iterations``, ``draws``, ``burn_in``, ``seed``, ``chains``, ``gamma``) are those of
``"gauss_newton"`` (:doc:`nonlinear`).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - What it is
   * - ``H``
     - The forward operator on the whole state, an :class:`~fips.Operator`:
       the transformed block and any other. Its form is kept (sparse stays
       sparse).
   * - ``reference`` (:class:`~fips.LogNormalForward`)
     - The reference :math:`r` of each element of the block, not negative: a
       :class:`~fips.Vector` on the block, a Series on its labels, or one
       number for every element.
   * - ``scale`` (:class:`~fips.SoftplusForward`, required)
     - The scale :math:`s` of the bend, positive, in the value's units.
   * - ``block``
     - The block that is :math:`u`. Default ``None``: the state's one block.
       Every other block passes through as it is.

Each model has two methods:

- ``prior_for(mean, std)``: the prior on :math:`u` (:math:`\mu` and
  :math:`\sigma`, two Vectors on the block) whose value has the mean and the
  standard deviation given. Each argument is a Vector, a Series or a number.
  For the lognormal model a mean and a standard deviation are zero exactly
  where the reference is zero; for the softplus model a mean is positive.
- ``transform(x)``: a state (a Vector, such as ``post.estimate``) or draws (a
  DataFrame, such as ``post.sample(n)``) with the block as the value, on the
  same labels.


Warnings and checks
-------------------

The models raise when they are built:

- :class:`TypeError` if ``H`` is not an Operator, or ``reference`` is an
  array (it has no labels);
- :class:`ValueError` if the state has several blocks and ``block`` is not
  given, ``reference`` does not have exactly the block's labels or is
  negative or not finite, or ``scale`` is not a positive, finite number;
- :class:`KeyError` if ``block`` is not a block of the state.

``prior_for`` raises :class:`ValueError` if a mean or a standard deviation is
negative or not finite, or does not have the block's labels; for the
lognormal model, if a mean is zero where the reference is not (a lognormal
value is never zero), or not zero where the reference is (a positive value
there needs a reference, or the softplus model); for the softplus model, if a
mean is not positive. The softplus ``prior_for`` raises
:class:`RuntimeError` if it cannot match some element.

``transform`` and the model raise :class:`ValueError` if the value
overflows: a :math:`u` that large is far outside any prior on it; check the
prior's units.

The solve warns and raises as ``"gauss_newton"`` does (:doc:`nonlinear`): a
wide prior makes the problem more nonlinear, and a solve that has not
stopped after ``max_iterations`` raises :class:`~fips.ConvergenceError`:
raise ``max_iterations``, or narrow the prior. Each warning is a
:class:`~fips.FipsWarning`, which Python shows without any setup
(:ref:`repeated-inversions`).


Cost
----

The cost is that of ``"gauss_newton"`` (:doc:`nonlinear`): an analytic solve
per step, then the draws. A transform is cheaper than a model written by
hand: its Jacobian keeps the form of :math:`H` (the sampler never forms it),
and its known second derivatives give Newton's steps where Gauss-Newton's
would crawl. The wider the prior on :math:`u`, the more the model bends, and
the more steps both the solve and each draw take. The softplus ``prior_for``
is cheap next to the solve.


How it works
------------

**The models.** The observations are :math:`F(x) = H\, [T(u); \text{the
other blocks}] + c`, and the Jacobian is :math:`H \operatorname{diag}(T'(u),
1)`. The lognormal model has :math:`T(u) = r\, e^{u}`; the softplus model
:math:`T(u) = s \log(1 + e^{u/s})`, which is :math:`u` for :math:`u \gg s`
and :math:`s\, e^{u/s}`, near zero and positive, for :math:`u \ll -s`.

**The lognormal prior.** A prior standard deviation of :math:`\sigma` on
:math:`u` is a factor of :math:`e^{\sigma}` either way. Where the prior on
:math:`u` is centered sets what the reference is:

- with :math:`u \sim N(0, \sigma^2)` the reference is the prior **median** of
  the value, and its mean is :math:`e^{\sigma^2/2}` times the reference (1.65
  times for :math:`\sigma = 1`);
- with :math:`u \sim N(-\sigma^2/2, \sigma^2)` the reference is the prior
  **mean** of the value.

A reference is usually an estimate of the mean, and a total is a sum of
means, so ``prior_for`` matches the mean. For a mean :math:`m` and standard
deviation :math:`s` of the value,

.. math::

    \sigma^2 = \log\left(1 + \frac{s^2}{m^2}\right), \qquad
    \mu = \log\frac{m}{r} - \frac{\sigma^2}{2}.

Two elements whose :math:`u` correlate by :math:`\rho` have values that
correlate less,

.. math::

    \frac{e^{\rho \sigma_i \sigma_j} - 1}{\sqrt{(e^{\sigma_i^2} - 1)(e^{\sigma_j^2} - 1)}},

0.38 for :math:`\rho = 0.5` and :math:`\sigma = 1`.

**The softplus prior** has no closed form. ``prior_for`` computes the value's
mean and standard deviation for a Gaussian :math:`u` by quadrature, and
matches them to the ones asked for by Newton's method, to :math:`10^{-12}`.
Far above the scale it returns the value's own prior; near zero, a :math:`u`
centered lower and wider. Its correlations are those of :math:`u`, and the
value's are the same far above the scale, weaker near zero.

**Why an aggregate comes from the posterior's draws.** The posterior of
:math:`u` is not Gaussian, and the Gaussian at the mode can be far from it
for a function of many elements. Two elements whose sum is observed closely,
each with a prior factor of :math:`e` either way, show it:

.. ipython:: python

   two = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
   one = fips.Space({"conc": pd.RangeIndex(1, name="obs_id")})
   pair = fips.LogNormalForward(fips.Operator(np.ones((1, 2)), one, two), 1.0)
   m, s = pair.prior_for(1.0, np.sqrt(np.e - 1.0))  # sigma = 1
   sums = fips.Problem(m, fips.Independent().with_std(s), fips.Vector([2.0], one), fips.Independent().with_std(0.05), pair)
   post2 = fips.solve(sums, "gauss_newton")
   gaussian = pair.transform(fips.solve(sums.linearized(post2.estimate)).sample(2000, rng=0)).sum()
   exact = pair.transform(post2.sample(2000, rng=0)).sum()
   pd.DataFrame({"Gaussian at the mode": gaussian.agg(["mean", "std"]), "posterior": exact.agg(["mean", "std"])}).round(2)

The data say the sum is :math:`2.00 \pm 0.05`, and so do the posterior's
draws; the Gaussian at the mode's do not. The states that give that sum lie
on a curve, :math:`e^{u_a} + e^{u_b} = 2`. The Gaussian at the mode lies
along the curve's tangent, and on the tangent the sum of the exponentials is
never below 2. The values at the mode are right (1 and 1); the Gaussian's
draws are not. Any observation of a sum of elements is this case, more so
the wider the prior. The sampler uses the Gaussian at the mode only to move,
and its draws are of the posterior (:doc:`nonlinear`).

**The solve** is ``"gauss_newton"``'s, with Newton's steps, since the
transform's second derivatives are known (:doc:`nonlinear`, How it works).
