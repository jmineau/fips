.. _synthetic-data:

Synthetic experiments
=====================

An inversion returns an estimate and a claim about how uncertain it is. The
one way to check both is to give it data whose answer you know: make a truth,
make the observations it would produce, solve them, and compare. Atmospheric
science calls this an OSSE (an observing system simulation experiment),
oceanography a twin experiment, and statistics a simulation study. fips uses
the statistician's parts, because every such experiment has the same four:

.. list-table::
   :header-rows: 1
   :widths: 16 44 40

   * - Part
     - What it is
     - In fips
   * - **world**
     - What makes the data: a truth and the observations it gives. Also
       called the data-generating mechanism or the nature run.
     - A :class:`~fips.Problem`, through :meth:`~fips.Problem.simulate`; or
       any function of a generator that returns a truth and observations.
   * - **totals**
     - What you compare on: monthly totals, a regional total, a seasonal
       difference, or each state element. Also called the estimands.
     - An aggregation ``W`` (:func:`fips.aggregate.sum`), one row per total.
   * - **inversion**
     - The method under test, with its own assumptions. Also called the
       estimator.
     - A :class:`~fips.Posterior`; ``post.with_data(obs)`` runs it on other data.
   * - **performance**
     - How the estimate's error behaves over worlds: its bias, its spread, the
       spread the inversion claims, and how often the claimed interval holds
       the truth.
     - The columns of :meth:`~fips.Posterior.recovery`, exactly; or the same
       columns summarized from a table of trials.

The inversion's assumptions and the world's may agree or differ, and that is
the whole design space. When the world is the problem itself, the experiment
asks whether the inversion is honest. When one input of the world differs from
what the inversion assumes, it asks what that input costs. This page builds a
small problem, makes worlds for it, reads the exact answer where there is one,
and writes the loop where there is not.

.. contents:: On this page
   :local:
   :depth: 1

A problem, and the function that builds it
------------------------------------------

Three months of fluxes in four cells along a line, observed by 24
measurements, each seeing the cells of its own month with random weights. The
state has a ``month`` level and an ``x`` level, so monthly totals are a sum over
``x``:

.. ipython:: python

   import numpy as np
   import pandas as pd
   from scipy.stats import norm
   import fips

   rng = np.random.default_rng(0)
   months = pd.date_range("2024-01-01", periods=3, freq="MS", name="month")
   cells = pd.Index([0.5, 1.5, 2.5, 3.5], name="x")  # km
   state = pd.MultiIndex.from_product([months, cells])
   prior = fips.Vector.from_series(pd.Series(1.0, index=state, name="prior"), block="flux")
   obs_labels = pd.Index(range(24), name="obs_id")
   obs_space = fips.Space({"conc": obs_labels})

.. ipython:: python

   weights = rng.uniform(0.0, 1.0, (24, 12))
   sees_month = np.repeat(np.arange(3), 8)[:, None] == np.repeat(np.arange(3), 4)[None, :]
   forward = fips.Operator(weights * sees_month, obs_space, prior.space)

The builder is the function that makes a :class:`~fips.Problem` from its
inputs. Every repeated inversion has one (:ref:`repeated-inversions`), and a
synthetic experiment leans on it: a world with one input changed is the same
call with one argument changed. The observations are zeros, because a world
makes the real ones and the inversion only needs their labels:

.. ipython:: python

   def build_problem(obs_std=0.3, offset=0.0, prior_std=0.5):
       """Return the problem, with the given mismatch, background and prior spread."""
       prior_cov = fips.Kronecker(
           fips.Exponential("month", "45D"), fips.Exponential("x", 1.0)
       ).with_std(prior_std)
       return fips.Problem(
           prior=prior,
           prior_cov=prior_cov,
           obs=fips.Vector(np.zeros(24), obs_space, name="conc"),
           obs_cov=fips.Independent().with_std(obs_std),
           forward=forward,
           offset=offset,
       )

.. ipython:: python

   problem = build_problem()
   post = fips.solve(problem)
   W = fips.aggregate.sum(prior.space, by="month")  # the totals: one row per month

A world makes the data
----------------------

A :class:`~fips.Problem` is a world through :meth:`~fips.Problem.simulate`: it
draws a truth from its prior, :math:`x \sim N(x_0, S_0)`, and the observations
that truth gives, :math:`z = H x + c + e` with :math:`e \sim N(0, S_z)`, and
returns the two as vectors. :meth:`~fips.Problem.predict` is the same without
the error. The draws are :meth:`Covariance.sample <fips.Covariance.sample>`,
which draws from :math:`N(0, S)` without forming :math:`S` where its form has a
cheaper root (a Kronecker covariance draws through its factors). The inversion
then runs on the world's data with ``post.with_data(obs)``, which costs one
product, not a new solve, because the factorization depends on where the
observations are and how uncertain they are, not on their values:

.. ipython:: python

   truth, obs = problem.simulate(rng)
   trial = post.with_data(obs)
   miss = (W @ trial.estimate - W @ truth).values
   claimed = np.sqrt(post.cov(W).variances.values)
   pd.DataFrame({"miss": miss, "claimed_std": claimed}, index=months).round(3)

One trial says little; its point is the shape of every experiment: the world
makes the data, the inversion solves them with its own assumptions, and the
totals of the two are compared. There are four kinds of world, and each is a
``Problem`` or a given truth:

- **The problem itself**, ``problem.simulate(rng)``: the inversion's
  assumptions are true. The test is whether it is honest.
- **A chosen truth**, ``problem.simulate(rng, truth=inventory)``: another
  inventory, or 1.5 times the prior, on the prior's labels in any order. Only
  the observation error is drawn. The test is how much of that truth is
  recovered.
- **One input wrong**, ``build_problem(obs_std=1.2).simulate(rng)``: your
  builder with one argument changed (the mismatch, the background, the prior
  spread, a forward operator scaled by a transport error). The inversion keeps
  its assumptions; the test is what the wrong input costs.
- **Another observing system**: a separate ``Problem`` with other observations
  and its own solve, compared on the uncertainty of the totals
  (:meth:`~fips.Posterior.uncertainty`, :meth:`~fips.Posterior.data_worth`).
  No synthetic data is needed for that comparison.

A world whose truth is not a Gaussian draw, or whose error changes from trial
to trial in a way no covariance describes, is a function: :ref:`function-worlds`.

The exact answer
----------------

The analytic method's estimate is linear in the observations,
:math:`\hat{x} = x_0 + K (z - H x_0 - c)`, so when a Gaussian world makes the
data, the error of every total is Gaussian too, and its mean and covariance
have a closed form. :meth:`~fips.Posterior.recovery` returns that form, one
row per total: no trials, no sampling noise.

.. ipython:: python

   honest = post.recovery(W=W).droplevel("block")
   honest.round(3)

The columns are the performance measures of a simulation study:

- ``bias``, the mean of ``estimate - truth``: zero when the inversion's
  assumptions hold at ``gamma`` 1.
- ``std``, the real spread of the error over worlds (the empirical standard
  error).
- ``posterior_std``, the spread the inversion claims (the model standard
  error, :math:`\sqrt{\operatorname{diag} W \hat{S} W^T}`).
- ``coverage``, the probability that the posterior's interval at ``level``
  holds the truth: 0.95 for an honest inversion at the default level.

In its own world the bias is zero, ``std`` equals ``posterior_std``, and the
coverage is the level. Each other world is the same call with the world as the
first argument. The world's mismatch four times what the inversion assumes
makes the inversion overconfident: the real spread is larger than the claimed
one, and the 95% interval holds the truth far less often than claimed:

.. ipython:: python

   noisy = build_problem(obs_std=1.2)
   post.recovery(noisy, W=W).droplevel("block")[["bias", "std", "posterior_std", "coverage"]].round(3)

A chosen truth fixes the state and leaves only the observation error random.
With 1.5 times the prior, the bias is the part of the change the inversion does
not recover, and ``(estimate - prior) / (truth - prior)`` is the recovered
fraction. A second inversion, whose prior is trusted five times more, shows
what regularization costs: it pulls the estimate back toward the prior it
believes, and recovers about three quarters of a change it did not expect
where the first recovers nearly all of it:

.. ipython:: python

   bigger = 1.5 * prior
   post_tight = fips.solve(build_problem(prior_std=0.1))

   def recovered(post):
       """Return the fraction of the change from the prior to `bigger` that `post` recovers."""
       table = post.recovery(truth=bigger, W=W).droplevel("block")
       return (table["estimate"] - table["prior"]) / (table["truth"] - table["prior"])

   pd.DataFrame({"loose prior": recovered(post), "tight prior": recovered(post_tight)}).round(3)

``recovery`` takes a world with exactly the posterior's state and observation
labels, a method whose estimate is linear in the observations, and a Gaussian
or fixed truth. Everything else is a loop.

When you need trials
--------------------

Four things have no closed form: a statistic ``recovery`` does not give (the
reduced chi-squared of each trial), a world whose truth is not Gaussian or
whose error is not additive, a world on other state labels (the whole state
where the solve was restricted, a monthly truth for an annual inversion), and a
method whose estimate is not linear in the observations. For those, write the
loop. It is short, and it is yours: the world makes the data, the inversion
solves them, and each trial adds its rows to a table.

.. ipython:: python

   def trials(world, post, n, rng, W):
       """Return one row per trial and total: the truth, the estimate, the claimed std, chi2."""
       rng = np.random.default_rng(rng)
       make = world.simulate if isinstance(world, fips.Problem) else world
       claimed = np.sqrt(post.cov(W).variances.values)
       rows = []
       for i in range(n):
           truth, obs = make(rng)
           trial = post.with_data(obs)
           rows.append(pd.DataFrame({"trial": i, "truth": (W @ truth).values,
                                     "estimate": (W @ trial.estimate).values,
                                     "posterior_std": claimed, "chi2": trial.reduced_chi2},
                                    index=W.rows.index))
       return pd.concat(rows).set_index("trial", append=True)

The table is long: one row per trial and total, so pandas does the rest. The
same performance measures come from a ``groupby`` over the totals, with the
Monte Carlo standard error of each (Morris, White and Crowther, 2019): with
``n`` trials a bias is known to ``std / sqrt(n)`` and a coverage near 0.95 to
about ``0.22 / sqrt(n)``.

.. ipython:: python

   def performance(table, level=0.95):
       """Return recovery's columns from a table of trials, with their Monte Carlo standard errors."""
       z = norm.ppf(1 - (1 - level) / 2)
       error = table["estimate"] - table["truth"]
       inside = error.abs() <= z * table["posterior_std"]
       by = table.index.names[:-1]
       n = error.groupby(level=by).size()
       out = pd.DataFrame({
           "bias": error.groupby(level=by).mean(),
           "std": error.groupby(level=by).std(),
           "posterior_std": table["posterior_std"].groupby(level=by).first(),
           "coverage": inside.groupby(level=by).mean(),
           "chi2": table["chi2"].groupby(level=by).mean(),
       })
       out["bias_se"] = out["std"] / np.sqrt(n)
       out["coverage_se"] = np.sqrt(out["coverage"] * (1 - out["coverage"]) / n)
       return out

The honest world, sampled and exact, side by side. The sampled coverage is
within its standard error of 0.95, and the reduced chi-squared averages 1, the
one number the exact table does not have:

.. ipython:: python

   sampled = performance(trials(problem, post, 400, 0, W)).droplevel("block")
   pd.DataFrame({"sampled": sampled["coverage"], "exact": honest["coverage"],
                 "se": sampled["coverage_se"]}).round(3)
   sampled["chi2"].round(3)

The noisy world, in the loop, gives the chi-squared that says in which
direction the mismatch is wrong, beside the coverage that says what it costs:

.. ipython:: python

   performance(trials(noisy, post, 400, 0, W)).droplevel("block")[["coverage", "chi2"]].round(3)

Use the exact table where it applies and the loop for what it lacks; where
both apply, the loop should agree with the table within its standard errors,
which is also the check that your loop is right.

.. _function-worlds:

Worlds no Problem can describe
------------------------------

A world is anything that takes a generator and returns a truth and
observations. ``Problem.simulate`` is one; a function of your own is another,
and the loop above takes either. Two worlds a real experiment needs and a
``Problem`` cannot describe: a truth that is lognormal about the prior (so it
stays positive and its spread scales with the prior), and an error the
inversion cannot see, such as a transport error that scales every observation
of a trial by one factor. Each is a few lines, since vectors do arithmetic by label:

.. ipython:: python

   field = fips.Kronecker(fips.Exponential("month", "45D"), fips.Exponential("x", 1.0)).with_std(0.4)
   field_cov = field.build(prior.space)
   half_var = 0.5 * field_cov.variances  # so the truth's mean is the prior

   def lognormal_world(rng):
       """A lognormal truth about the prior, and a transport error of 10% the inversion cannot see."""
       truth = prior * np.exp(field_cov.sample(rng) - half_var)
       scale = np.exp(rng.normal(0.0, 0.10))
       obs = problem.predict(truth) * scale + problem.obs_cov.sample(rng)
       return truth, obs

   performance(trials(lognormal_world, post, 400, 0, W)).droplevel("block")[["bias", "std", "posterior_std", "coverage"]].round(3)

The error the inversion cannot see lands in the totals: the real spread is
twice the claimed one, and the 95% interval holds the truth two times in three.
The bias is small and positive because the transport factor averages a little
above one (it is the exponential of a zero-mean normal).

A truth with a chosen amplitude, a seasonal cycle or a sector-by-sector factor
is the same kind of function. What the world does is yours to write; what fips
gives is the draw from any of its covariance models (:meth:`Covariance.sample
<fips.Covariance.sample>`), the noise-free observations of any truth
(:meth:`~fips.Problem.predict`), and the solve of any data at the cost of a
product (:meth:`~fips.Posterior.with_data`).

Two inversions on the same data
-------------------------------

To compare two inversions, give them the same data. The world does not depend
on the inversion, so the same seed gives both the same truths and the same
observations, trial for trial; ``pd.concat`` with a key pairs them. Here the
two inversions above, the loose prior and the tight one, are compared on the
honest world of the first:

.. ipython:: python

   both = pd.concat({"loose": trials(problem, post, 400, 0, W),
                     "tight": trials(problem, post_tight, 400, 0, W)}, names=["inversion"])
   performance(both).droplevel("block")[["bias", "std", "posterior_std", "coverage"]].round(3)

In a world drawn from the loose prior, the tight inversion has no bias on
average but claims a third of the spread it has, so its 95% interval holds the
truth less than half the time: the prior it trusts is not the one the world
drew from. When the two inversions have different states (a gridded and a basin
inversion, a monthly and an annual one), each takes its own ``W`` to the same
totals, with the same row labels, and the loop compares them as above;
``recovery`` needs the world on the posterior's own labels.

Reading the table
-----------------

Two questions a real experiment asks are read off the table with pandas.
**Where does the real retrieval fall among worlds consistent with the prior?**
Run the honest world and find the quantile of the real estimate in the
``estimate`` column of each total: a real number that almost no honest world
reaches says the truth was not within the prior's spread. **What truth is
consistent with what was retrieved?** Regress the truth on the estimate over
the trials and read the fit at the real number; the quantiles of the residuals
are its interval. For the tight inversion the slope is above 1, because its
estimates move less than the truth does, and the fit undoes that shrinkage:

.. ipython:: python

   tight = both.xs("tight", level="inversion")
   X = np.column_stack([np.ones(len(tight)), tight["estimate"]])
   beta, *_ = np.linalg.lstsq(X, tight["truth"], rcond=None)
   residual = tight["truth"] - X @ beta
   np.round(beta, 3), np.round(np.percentile(residual, [16, 84]), 3)

With two inversions paired on the same worlds (a gridded and a basin one, say),
the same regression on both estimates reads the truth off the retrieved pair.

Randomness
----------

Every draw takes ``rng``: a seed, a :class:`numpy.random.Generator`, or None
for fresh entropy, through :func:`numpy.random.default_rng`, as SciPy's
``rng`` argument does. One generator passed to every call of a loop gives
independent trials; a seed gives the same draws again for the same covariance
forms (a Kronecker covariance on part of its grid draws on the whole grid, so
changing a model's form changes the draws). For trials in parallel, spawn one
generator per worker, ``np.random.SeedSequence(seed).spawn(k)``, and keep each
worker's trials in the table with a ``worker`` key, so the run is reproducible
whatever the scheduling.
