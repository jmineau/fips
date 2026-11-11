.. _bounds:

Bounds on the state
===================

Some quantities cannot leave a range: a rate cannot be negative, a fraction
stays between 0 and 1. The Gaussian posterior of a linear problem does not
know this, and where the prior is near a bound and the observations see
little, its estimate crosses it. ``fips.solve(problem, "bounded",
lower={"flux": 0.0})`` solves the same problem with the state held to its
bounds (here, a block named ``flux`` at or above zero). **The estimate is the
mode**: the analytic estimate moved onto the bounds along the posterior's
correlations. **The spread comes from draws** of the analytic posterior
truncated to the bounds, which the solve keeps.

.. contents:: On this page
   :local:
   :depth: 1


When to use it
--------------

Use ``"bounded"`` when each element's estimate must respect its bounds (a map
with no negative values) and the Gaussian prior cut at the bounds is the
prior you mean. The problem is the linear one: a forward operator, a
Gaussian prior and Gaussian observation errors.

Another method is better when:

- **you want an aggregate** (a sum or a mean over many elements): the analytic
  posterior's aggregate is often close to Gaussian, its estimate is unbiased,
  and it costs least (``fips.solve(problem)``; :ref:`methods-positive`);
- **the uncertainty is a factor of a reference, or additive away from zero**:
  solve for a transform of the state that is always positive
  (:class:`~fips.LogNormalForward`, :class:`~fips.SoftplusForward`;
  :doc:`positive`);
- **the forward map is a model, not a matrix**: ``"bounded"`` refuses it.
  Write the model on a state that cannot leave the bounds (the logarithm of a
  positive value) and solve with ``"gauss_newton"`` (:doc:`nonlinear`).


An example
----------

As an example, a flux inversion: twenty cells along a line, a prior flux of
0.2 in each with correlated errors, and two sites that see the cells, each
with a bias of its own. The bias block must stay free.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   rng = np.random.default_rng(0)
   n, m = 20, 24
   cells = pd.Index(np.arange(n) + 0.5, name="x")  # km along a line
   sites = pd.Index(["north", "south"], name="site")
   prior = fips.Vector.from_blocks(
       {"flux": pd.Series(0.2, index=cells), "bias": pd.Series(0.0, index=sites)},
       name="prior",
   )

.. ipython:: python

   obs_labels = pd.MultiIndex.from_arrays(
       [np.repeat(sites, m // 2), np.tile(np.arange(m // 2), 2)], names=["site", "hour"]
   )
   obs_space = fips.Space({"conc": obs_labels})
   footprint = rng.uniform(0.0, 1.0, (m, n))
   footprint *= rng.uniform(0.0, 1.0, (m, n)) < 0.3
   south = (obs_labels.get_level_values("site") == "south").astype(int)
   H = np.hstack([footprint, np.eye(2)[south]])  # each site sees its own bias
   forward = fips.Operator(H, obs_space, prior.space)

The data come from a truth that is mostly near zero:

.. ipython:: python

   flux = np.exp(rng.normal(np.log(0.15), 1.0, n))
   truth = fips.Vector(np.r_[flux, -0.6, 0.4], prior.space)
   obs = fips.Vector(H @ truth.values + rng.normal(0.0, 0.2, m), obs_space)
   problem = fips.Problem(
       prior=prior,
       prior_cov=fips.Covariance.block_diag(
           flux=fips.Exponential("x", 3.0).with_std(0.4),
           bias=fips.Independent().with_std(0.5),
       ),
       obs=obs,
       obs_cov=fips.Independent().with_std(0.2),
       forward=forward,
   )

The analytic estimate and the bounded one, at the cells on a bound and their
neighbors:

.. ipython:: python

   analytic = fips.solve(problem)
   post = fips.solve(problem, "bounded", lower={"flux": 0.0})
   side = pd.DataFrame(
       {
           "truth": truth.to_series(),
           "analytic": analytic.estimate.to_series(),
           "bounded": post.estimate.to_series(),
           "at_bound": post.at_bound,
       }
   )
   near = post.at_bound.to_numpy()
   near = near | np.roll(near, 1) | np.roll(near, -1)  # on a bound, or next to one
   side[near].round(3)

The cells the analytic estimate puts below zero sit at zero, and their
neighbors move by what the posterior's correlations say. Clipping (setting
the negative values to zero and nothing else) would always raise the total;
moving along the correlations can lower it too. The bias block has no bound
and keeps its negative north bias:

.. ipython:: python

   side.loc["bias"].round(3)

The total of the flux, with its 95% interval. With the bounds, the interval
is the quantiles of the draws. It is narrower, since the bounds rule out the
negative totals the Gaussian allowed, and higher: where the data see little,
the draws are those of the prior cut at zero, whose mean is above the prior
mean. Here it misses the true total, which the analytic interval holds.

.. ipython:: python

   W = fips.Operator(
       np.r_[np.ones(n), 0.0, 0.0][None, :],
       fips.Space({"total": pd.Index(["flux"], name="what")}),
       prior.space,
   )
   (W @ truth).values.round(3)
   pd.concat({"analytic": analytic.uncertainty(W), "bounded": post.uncertainty(W)}).round(3)

**The mode is not the mean.** The draws spread on the side of the bound away
from the mode, so their mean is above it. A question about the distribution
(the mean of a total, the chance it is below a value) is read off new draws,
:meth:`~fips.Posterior.sample`:

.. ipython:: python

   draws = post.sample(2000, rng=1, W=W)
   (W @ post.estimate).values.round(3), round(float(draws.iloc[0].mean()), 3)
   round(float((draws.iloc[0] < 3.0).mean()), 3)


What the posterior reports
--------------------------

A bounded posterior is a *sampled* one: the
:ref:`members table <posterior-members>` lists what it gives.

- :attr:`~fips.Posterior.estimate` is the **mode**, and
  :attr:`~fips.Posterior.at_bound` says which elements sit on a bound there:
  the bounds decide those, and the observations the others.
- :attr:`~fips.Posterior.variances`, :meth:`~fips.Posterior.cov` (of the
  state or of aggregates), :attr:`~fips.Posterior.mean` and the ``lower`` and
  ``upper`` of :meth:`~fips.Posterior.uncertainty` and
  :meth:`~fips.Posterior.trials` are those of the **kept draws**: ``draws``
  of them, made from ``seed`` the first time one is needed, so a solve is
  reproducible. The interval is the draws' quantiles, not the mode plus or
  minus a multiple of the standard deviation. An element on its bound still
  has spread, on one side of the bound.
- :meth:`~fips.Posterior.sample` makes new draws, for any function of the
  state.
- **The gain, the averaging kernel, DOFS and the information content are
  those at the mode**: the analytic posterior with the elements on a bound
  held there. An element on a bound has no DOFS: the bound decides it, not
  the data.
- :attr:`~fips.Posterior.chi2` and the misfits are those of the mode. The
  degrees of freedom and p-values of :meth:`~fips.Posterior.chi2_tests` are
  then approximate.
- :meth:`~fips.Posterior.with_gamma` and :meth:`~fips.Posterior.with_data`
  find the mode again on the kept factors, without factoring again; their
  draws are made when first needed.
- :meth:`~fips.Posterior.recovery` and the data worth raise: they need an
  estimate linear in the data, and the mode is not.
  :meth:`~fips.Posterior.trials` measures what ``recovery`` would
  (:doc:`synthetic`). ``log_evidence`` raises too; the evidence of the
  problem without bounds is ``fips.solve(problem).log_evidence``.

With no finite bound, the solve returns the analytic posterior itself, a
Gaussian one.


Options
-------

Each is a keyword of ``fips.solve(problem, "bounded", ...)``
(:class:`~fips.methods.bounded.BoundedOptions`).

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Option
     - Default
     - What it does
   * - ``lower``, ``upper``
     - ``None``
     - The bounds: ``lower <= x <= upper``. A number, a
       :class:`~fips.Vector`, or a dict by block (below). ``None`` is no
       bound.
   * - ``distribution``
     - ``"truncated"``
     - What the draws are draws of: the posterior truncated to the bounds, or
       ``"rectified"``, the bounded estimate with the prior mean and the data
       drawn again (below).
   * - ``draws``
     - 2000
     - The number of draws kept for the variances, the covariances, the mean
       and the intervals. The Monte Carlo error of a standard deviation falls
       as one over its square root.
   * - ``burn_in``
     - 20
     - The draws of the truncated chain made and left out before those kept.
       Rectified draws are not a chain, and leave none out.
   * - ``seed``
     - 0
     - The seed of the kept draws.
   * - ``gamma``
     - 1.0
     - The regularization factor: the observation covariance is divided by
       it, as for the analytic method.

Giving the bounds
~~~~~~~~~~~~~~~~~

The bounds go to the solve, not to the :class:`~fips.Problem`: the problem is
the Gaussian question, and the bounds are how this method answers it. Each
of ``lower`` and ``upper`` is one of:

- **a number**, for every state element. On a state of several blocks that
  bounds the bias block too, which is rarely meant, and the solve warns;
- **a** :class:`~fips.Vector` with exactly the state's labels, one bound per
  element, ``-inf`` (or ``inf`` for ``upper``) where an element has none;
- **a dict by block**, such as ``{"flux": 0.0}``: each value a number or a
  Vector on that block's labels. Blocks it does not name are free.

.. _bounds-rectified:

Truncated or rectified draws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Truncated draws** (the default) are draws of the analytic posterior
truncated to the bounds: the Bayesian posterior of the Gaussian prior cut at
the bounds. A truncated draw is never exactly on a bound.

**Rectified draws** (``distribution="rectified"``) are the bounded estimate
itself, with the inputs drawn again: each is the bounded problem solved for
one draw of the prior mean from :math:`N(x_0, S_0)` and of the observations
from :math:`N(z, S_z)`. An element that the draw puts outside its bounds is
exactly on its bound, as at the mode. They are the distribution of the
bounded estimate over inputs as likely as these. They are not the truncated
posterior, and not the Bayesian posterior of any prior fips writes down.

The estimate, ``at_bound``, the gain and DOFS are the same either way; the
draws differ, and with them the variances, the mean, the intervals and the
aggregates. Where the data see little, the truncated draws' mean sits above
the prior mean at a lower bound, and the rectified draws' less far above:

.. ipython:: python

   rectified = fips.solve(problem, "bounded", lower={"flux": 0.0}, distribution="rectified")
   first = post.at_bound[post.at_bound].index[0]  # the first cell on its bound
   on_bound = {"truncated": post, "rectified": rectified}
   {name: float((p.sample(2000, rng=1).loc[first] == 0).mean()) for name, p in on_bound.items()}
   pd.concat({"truncated": post.uncertainty(W), "rectified": rectified.uncertainty(W)}).round(3)

About half the rectified draws hold that cell at exactly zero, and none of
the truncated ones do. The rectified total sits lower, and here its interval
holds the true total. **For an aggregate, read the analytic posterior**;
rectified draws suit the elements of a map near a bound, where the truth may
be exactly on it. :doc:`examples/comparison` compares the methods on one
problem.


Warnings and checks
-------------------

fips logs its warnings through Python's :mod:`logging`, which is silent
until you configure it (:ref:`repeated-inversions`). The solve logs a
warning when:

- **a number bounds a state of several blocks.** It bounds every block, a
  bias block too. Give a dict to bound only some: ``lower={"flux": 0.0}``.
- **the prior is wide against its bounds**: a prior standard deviation over
  half the distance to a bound, for more than a tenth of a block's bounded
  elements. Where the data see little, the draws' mean, intervals and
  aggregates are those of the prior as the draws treat it, cut at the bounds
  (or, for rectified draws, set to them), and so above the prior mean at a
  lower bound. The warning names the block and says by how much its
  aggregate sits above the prior's, as a share of its distance from the
  bounds. The estimate is not affected. For an aggregate, read the analytic
  posterior or use :class:`~fips.LogNormalForward`.
- **the kept draws are too few**: a standard deviation from them is
  uncertain by more than 10%. It says how many draws would do; raise
  ``draws``. Truncated draws mix slowest at an element whose analytic
  estimate is one to a few standard deviations outside its bound; rectified
  draws are too few at an element on its bound in nearly every draw.

It raises:

- :class:`TypeError` if the problem's forward is a model, or a bound is not a
  number, a Vector, a dict or ``None``;
- :class:`ValueError` if ``lower`` is not below ``upper`` at some element
  (an element held at one value is known, not bounded), a bound is NaN or
  one no value can satisfy (a ``lower`` of ``inf``), a Vector's labels are
  not the state's (or its block's), or a dict names a block the state does
  not have;
- :class:`numpy.linalg.LinAlgError` if elements on a bound cannot be held
  there: an element whose posterior variance is zero, or elements whose
  posterior errors are tied together. Check the bounds of the elements the
  prior or the observations fix.

A member a sampled posterior cannot give raises
:class:`~fips.UnavailableError` (:ref:`the table <posterior-members>`).


Cost
----

**The mode costs little more than the analytic solve**: the factors, plus a
column of the posterior covariance :math:`\hat{S}` for each element that ends
on a bound.

**The draws cost more**, and are made only when a spread is first asked for.
A truncated draw follows a path that bounces off each bound it meets, and
each bounce costs a step of the size of the state, so the cost grows with the
number of elements near a bound: for a thousand elements mostly near zero,
the default 2000 draws take tens of seconds, and a state far from its bounds
bounces little. A rectified draw costs an active set, as the mode does. Both
keep the columns of :math:`\hat{S}` at every element a draw holds on a bound,
up to all of them (:math:`n \times n`). The cost is proportional to
``draws``; :meth:`~fips.Posterior.trials` draws in every trial, so keep
``draws`` small there.


How it works
------------

**The posterior.** The prior with the bounds is the Gaussian prior set to
zero outside them. The likelihood is unchanged, so the posterior is the
analytic posterior :math:`N(\hat{x}, \hat{S})` truncated to the box
:math:`l \le x \le u`. It is not Gaussian, and its mean is not its mode.

**The mode** is the point of the box where the analytic cost is least. The
cost is quadratic, so that is the point of the box nearest :math:`\hat{x}` in
the metric of the posterior:

.. math::

    x^\ast = \arg\min_{l \le x \le u} (x - \hat{x})^T \hat{S}^{-1} (x - \hat{x}).

With :math:`a` the set of elements held on a bound :math:`b_a`, the answer is
:math:`x^\ast = \hat{x} + \hat{S}_{:a} \lambda_a`, with :math:`\lambda_a =
\hat{S}_{aa}^{-1} (b_a - \hat{x}_a)`: the held elements go to their bounds,
and the others move by what the posterior says they would if those elements
were known to be there. An active set finds :math:`a`, exchanging at each
step every element to hold or release (Nocedal and Wright, 2006; Lawson and
Hanson, 1974, for nonnegative least squares), and needs :math:`\hat{S}` only
at the columns of :math:`a`, from the analytic factors. The solve checks the
optimality conditions of the answer before it returns it.

**The diagnostics at the mode** are those of the analytic posterior with the
elements of :math:`a` held at their bounds. With :math:`C = \hat{S}_{aa}`,

.. math::

    K_a = K - \hat{S}_{:a} C^{-1} K_{a:}, \qquad
    A_a = K_a H = A - \hat{S}_{:a} C^{-1} A_{a:},

and an element on a bound has a row of zeros in both.

**Truncated draws** are made by exact Hamiltonian Monte Carlo (Pakman and
Paninski, 2014). Each draw starts from the last, :math:`x_0`, with a velocity
:math:`v \sim N(0, \hat{S})`, and follows

.. math::

    x(t) = \hat{x} + (x_0 - \hat{x}) \cos t + v \sin t

for a quarter period, :math:`t = \pi/2`. Each element's time to reach its
bound has a closed form; at the first bound reached, the velocity reflects
and the path goes on. With no bound in the way, :math:`x(\pi/2) = \hat{x} +
v` is a draw independent of the start. There is no step size and no
rejection. The draws are correlated only through the bounces: an element
away from its bounds has about one independent draw per draw (a standard
deviation from 2000 is good to about 1.5%), while one whose analytic
estimate is one to four standard deviations outside its bound piles against
it and mixes an order of magnitude slower.

**Rectified draws** use one fact: the analytic estimate of the problem with a
drawn prior mean and drawn observations is an exact draw :math:`x_s` of
:math:`N(\hat{x}, \hat{S})`, and that problem has the same :math:`\hat{S}`.
Its bounded estimate is therefore :math:`x_s` moved onto the bounds by the
same active set as the mode, with :math:`x_s` in place of :math:`\hat{x}`.
So draw ``k`` of ``rectified.sample(N, rng=5)`` is draw ``k`` of
``fips.solve(problem).sample(N, rng=5)``, unchanged where no bound binds. The
draws are independent. Miller, Michalak and Levi (2014, section 3.3.1)
propose realizations this way and then accept or reject each by a
Metropolis-Hastings test, which fips does not; Bardsley and Fox (2012) solve
a nonnegativity-constrained problem for each sample; Everink, Dong and
Andersen (2023) call the resulting distribution a regularized Gaussian.


References
----------

Bardsley and Fox (2012). *Inverse Problems in Science and
Engineering*, 20(4), 477-498.

Everink, Dong and Andersen (2023). *Inverse Problems*.
https://doi.org/10.1088/1361-6420/acf9c5

Lawson, C. L., and Hanson, R. J. (1974). *Solving Least Squares Problems*.
Prentice-Hall.

Miller, S. M., Michalak, A. M., and Levi, P. J. (2014). Atmospheric inverse
modeling with known physical bounds: an example from trace gas emissions.
*Geoscientific Model Development*, 7, 303-315.
https://doi.org/10.5194/gmd-7-303-2014

Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, second
edition, chapter 16. Springer.

Pakman, A., and Paninski, L. (2014). Exact Hamiltonian Monte Carlo for
truncated multivariate Gaussians. *Journal of Computational and Graphical
Statistics*, 23(2), 518-542. https://doi.org/10.1080/10618600.2013.788448
