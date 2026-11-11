.. _synthetic-data:

Synthetic experiments
=====================

An inversion returns an estimate and a claim about how uncertain it is. **The
one way to check both is to give it data whose answer you know**: make a
truth, make the observations it would give, solve them, and compare.
:meth:`problem.simulate(rng) <fips.Problem.simulate>` makes a truth and its
observations, and :meth:`post.with_data(obs) <fips.Posterior.with_data>`
solves them for the cost of a product. :meth:`post.recovery(world, W=W)
<fips.Posterior.recovery>` gives the answer of every such trial at once,
exactly: the bias, the spread and the coverage of each aggregate, one row
each. :meth:`post.trials(world, n, rng, W=W) <fips.Posterior.trials>` gives
the same by sampling, trial by trial, for what has no exact answer.

Atmospheric science calls this an observing system simulation experiment
(OSSE), oceanography a twin experiment, and statistics a simulation study.
fips uses the statistician's parts, because every such experiment has the
same four:

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
   * - **aggregates**
     - What you compare on: a monthly total, a regional mean, a difference
       between seasons, or each state element. Also called the estimands.
     - An aggregation ``W`` (:func:`fips.aggregate.sum`,
       :func:`~fips.aggregate.mean`), one row per aggregate.
   * - **inversion**
     - The method under test, with its own assumptions. Also called the
       estimator.
     - A :class:`~fips.Posterior`; ``post.with_data(obs)`` runs it on other
       data.
   * - **performance**
     - How the estimate's error behaves over worlds: its bias, its spread,
       the spread the inversion claims, and how often the claimed interval
       holds the truth.
     - The columns of :meth:`~fips.Posterior.recovery`, exactly; or the same
       columns summarized from the table of :meth:`~fips.Posterior.trials`.

**The design is whether the world's assumptions and the inversion's agree.**
When the world is the problem itself, the experiment asks whether the
inversion is honest. When one input of the world differs from what the
inversion assumes, it asks what that input costs.

.. contents:: On this page
   :local:
   :depth: 1


An example problem
------------------

Three months of a quantity at four positions along a line, observed 24
times, each observation a weighted sum of the positions in its own month. The
state has a ``month`` level and an ``x`` level, so a monthly total is a sum
over ``x``:

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   rng = np.random.default_rng(0)
   months = pd.date_range("2024-01-01", periods=3, freq="MS", name="month")
   positions = pd.Index([0.5, 1.5, 2.5, 3.5], name="x")
   state = pd.MultiIndex.from_product([months, positions])
   prior = fips.Vector.from_series(pd.Series(1.0, index=state, name="prior"), block="field")
   obs_space = fips.Space({"obs": pd.RangeIndex(24, name="obs_id")})

.. ipython:: python

   weights = rng.uniform(0.0, 1.0, (24, 12))
   own_month = np.repeat(np.arange(3), 8)[:, None] == np.repeat(np.arange(3), 4)[None, :]
   forward = fips.Operator(weights * own_month, obs_space, prior.space)

**A synthetic experiment leans on the function that builds the problem**
(every repeated inversion has one, :ref:`repeated-inversions`): a world with
one input changed is the same call with one argument changed. The
observations are zeros, because a world makes the real ones and the
inversion needs only their labels:

.. ipython:: python

   def build_problem(obs_std=0.3, offset=0.0, prior_std=0.5):
       """Return the problem, with the given observation error, offset and prior spread."""
       prior_cov = fips.Kronecker(
           fips.Exponential("month", "45D"), fips.Exponential("x", 1.0)
       ).with_std(prior_std)
       return fips.Problem(
           prior=prior,
           prior_cov=prior_cov,
           obs=fips.Vector(np.zeros(24), obs_space, name="obs"),
           obs_cov=fips.Independent().with_std(obs_std),
           forward=forward,
           offset=offset,
       )

.. ipython:: python

   problem = build_problem()
   post = fips.solve(problem)
   W = fips.aggregate.sum(prior.space, by="month")  # one row per monthly total


A world makes the data
----------------------

**A** :class:`~fips.Problem` **is a world.** :meth:`~fips.Problem.simulate`
draws a truth from its prior, :math:`x \sim N(x_0, S_0)`, and the
observations that truth gives, :math:`z = H x + c + e` with :math:`e \sim
N(0, S_z)`, and returns the two as vectors (:meth:`~fips.Problem.predict` is
the same without the error). The inversion then solves the world's data with
``post.with_data(obs)``, which costs one product, not a new solve:

.. ipython:: python

   truth, obs = problem.simulate(rng)
   trial = post.with_data(obs)
   miss = (W @ trial.estimate - W @ truth).values
   claimed = np.sqrt(post.cov(W).variances.values)
   pd.DataFrame({"miss": miss, "claimed_std": claimed}, index=months).round(3)

One trial says little. Its point is the shape of every experiment: the world
makes the data, the inversion solves them with its own assumptions, and the
aggregates of the two are compared. There are four kinds of world:

- **The problem itself**, ``problem.simulate(rng)``: the inversion's
  assumptions are true. The test is whether it is honest.
- **A chosen truth**, ``problem.simulate(rng, truth=x)``: another prior
  estimate, or 1.5 times the prior, on the prior's labels in any order. Only
  the observation error is drawn. The test is how much of that truth the
  inversion recovers.
- **One input wrong**, ``build_problem(obs_std=1.2).simulate(rng)``: the
  builder with one argument changed (the observation error, the offset, the
  prior spread, a forward operator scaled by a model error). The inversion
  keeps its own assumptions; the test is what the wrong input costs.
- **Another observing system**: a separate problem with other observations
  and its own solve, compared on the uncertainty of the aggregates
  (:meth:`~fips.Posterior.uncertainty`, :meth:`~fips.Posterior.data_worth`).
  That comparison needs no synthetic data.

A world whose truth is not a Gaussian draw, or whose error changes from trial
to trial in a way no covariance describes, is a function:
:ref:`function-worlds`.


The exact answer
----------------

**For the analytic method,** :meth:`~fips.Posterior.recovery` **gives the
answer of every trial at once.** The estimate is linear in the observations,
so when a Gaussian world makes the data, the error of every aggregate is
Gaussian too, with a mean and a spread in closed form: no trials, no
sampling noise. One row per aggregate:

.. ipython:: python

   honest = post.recovery(W=W).droplevel("block")
   honest.round(3)

The columns are the performance measures of a simulation study:

- ``bias``: the mean of ``estimate - truth``, zero when the inversion's
  assumptions hold at a gamma of 1;
- ``std``: the real spread of the error over worlds (the empirical standard
  error);
- ``posterior_std``: the spread the inversion claims (the model standard
  error, :math:`\sqrt{\operatorname{diag} W \hat{S} W^T}`);
- ``coverage``: the probability that the posterior's interval at ``level``
  (0.95 by default) holds the truth.

In its own world, as here, the bias is zero, ``std`` equals
``posterior_std``, and the coverage is the level. **Any other world is the
same call with the world as the first argument.** A world whose observation
error is four times what the inversion assumes makes the inversion
overconfident: the real spread is larger than the claimed one, and the
interval holds the truth far less often than it claims:

.. ipython:: python

   noisy = build_problem(obs_std=1.2)
   post.recovery(noisy, W=W).droplevel("block")[["bias", "std", "posterior_std", "coverage"]].round(3)

**A chosen truth** fixes the state and leaves only the observation error
random. With 1.5 times the prior, the bias is the part of the change the
inversion does not recover, and ``(estimate - prior) / (truth - prior)`` is
the fraction it does. A second inversion, whose prior spread is five times
smaller, shows what a tight prior costs: it pulls the estimate back toward a
prior it trusts, and recovers less of a change it did not expect:

.. ipython:: python

   bigger = 1.5 * prior
   post_tight = fips.solve(build_problem(prior_std=0.1))

   def recovered(post):
       """Return the fraction of the change from the prior to `bigger` that `post` recovers."""
       table = post.recovery(truth=bigger, W=W).droplevel("block")
       return (table["estimate"] - table["prior"]) / (table["truth"] - table["prior"])

   pd.DataFrame({"loose prior": recovered(post), "tight prior": recovered(post_tight)}).round(3)

``recovery`` needs a world whose forward map is an operator, with the
posterior's observation labels, and a Gaussian or fixed truth. It needs an
estimate linear in the observations, which only the analytic method gives
(the :ref:`members table <posterior-members>`); any other method raises
:class:`~fips.UnavailableError`. A world whose state has other labels is
compared on aggregates both states add up to (:ref:`other-state-labels`).
Everything else is a loop of trials.


When you need trials
--------------------

**Three things have no exact answer**: a statistic ``recovery`` does not give
(the reduced chi-squared of each trial), a world whose truth is not Gaussian
or whose error is not additive, and a method whose estimate is not linear in
the observations. For those, sample. :meth:`post.trials <fips.Posterior.trials>`
is the loop ``recovery`` answers exactly, kept trial by trial. It takes the
same world, ``W`` and ``W_world``, plus the number of trials and a seed. In
each trial the world makes the data, the inversion solves them with
``with_data``, and the aggregates of the two go into one row per aggregate:

.. ipython:: python

   table = post.trials(problem, 400, rng=0, W=W)
   table.head(6)

The table is long: one row per trial and aggregate, with the spread the
posterior claims, its interval (``lower`` and ``upper``, as
:meth:`~fips.Posterior.uncertainty` gives them: the kept draws' quantiles for
a sampled posterior) and the trial's reduced chi-squared. pandas does the
rest. The performance measures are a ``groupby`` over the aggregates, each
with its Monte Carlo standard error (Morris, White and Crowther, 2019): with
``n`` trials a bias is known to ``std / sqrt(n)``, and a coverage near 0.95 to
about ``0.22 / sqrt(n)``.

.. ipython:: python

   def performance(table):
       """Return recovery's columns from a table of trials, with their Monte Carlo standard errors."""
       by = [name for name in table.index.names if name != "trial"]  # the aggregates
       def per_row(values):  # dropna=False: a level a block lacks is NaN
           return values.groupby(level=by, dropna=False)
       error = table["estimate"] - table["truth"]
       inside = (table["lower"] <= table["truth"]) & (table["truth"] <= table["upper"])
       out = pd.DataFrame({
           "bias": per_row(error).mean(),
           "std": per_row(error).std(),
           "posterior_std": per_row(table["posterior_std"]).mean(),
           "coverage": per_row(inside).mean(),
           "reduced_chi2": per_row(table["reduced_chi2"]).mean(),
       })
       n = per_row(error).size()
       return out.assign(bias_se=out["std"] / np.sqrt(n), coverage_se=np.sqrt(out["coverage"] * (1 - out["coverage"]) / n))

The honest world, sampled and exact, side by side. The sampled coverage is
within its standard error of the exact one, and the reduced chi-squared
averages 1, the one number the exact table does not have:

.. ipython:: python

   sampled = performance(table).droplevel("block")
   pd.DataFrame({"sampled": sampled["coverage"], "exact": honest["coverage"], "se": sampled["coverage_se"]}).round(3)
   sampled["reduced_chi2"].round(3)

In the noisy world, the reduced chi-squared says which way the observation
error is wrong, beside the coverage that says what it costs:

.. ipython:: python

   performance(post.trials(noisy, 400, rng=0, W=W)).droplevel("block")[["coverage", "reduced_chi2"]].round(3)

Use the exact table where it applies and ``trials`` for what it lacks. Where
both apply they agree within the standard errors.


.. _function-worlds:

Worlds no Problem can describe
------------------------------

**A world is anything that takes a generator and returns a truth and
observations.** ``Problem.simulate`` is one; a function of your own is
another, and ``trials`` takes either. Two worlds an experiment often needs
and a problem cannot describe: a truth that is lognormal about the prior (so
it stays positive and its spread scales with the prior), and an error the
inversion cannot see, such as a model error that scales every observation of
a trial by one factor. Each is a few lines, because vectors do arithmetic by
label:

.. ipython:: python

   field = fips.Kronecker(fips.Exponential("month", "45D"), fips.Exponential("x", 1.0)).with_std(0.4)
   field_cov = field.build(prior.space)
   half_var = 0.5 * field_cov.variances  # so the truth's mean is the prior

   def lognormal_world(rng):
       """Return a lognormal truth about the prior, and its observations with a 10% model error the inversion cannot see."""
       truth = prior * np.exp(field_cov.sample(rng) - half_var)
       scale = np.exp(rng.normal(0.0, 0.10))
       obs = problem.predict(truth) * scale + problem.obs_cov.sample(rng)
       return truth, obs

   performance(post.trials(lognormal_world, 400, rng=0, W=W)).droplevel("block")[["bias", "std", "posterior_std", "coverage"]].round(3)

The error the inversion cannot see lands in the aggregates: the real spread
is well above the claimed one, and the interval holds the truth much less
often than it claims.

A truth with a chosen amplitude, a seasonal cycle or a factor per part of the
state is the same kind of function. What the world does is yours to write.
What fips gives is a draw from any of its covariance models
(:meth:`Covariance.sample <fips.Covariance.sample>`), the noise-free
observations of any truth (:meth:`~fips.Problem.predict`), and the solve of
any data for the cost of a product (:meth:`~fips.Posterior.with_data`).


Two inversions on the same data
-------------------------------

**To compare two inversions, give them the same data.** The world does not
depend on the inversion, so the same seed gives both the same truths and the
same observations, trial for trial, and ``pd.concat`` with a key pairs them.
Here the loose prior and the tight one, in the honest world of the loose
one:

.. ipython:: python

   both = pd.concat({"loose": post.trials(problem, 400, rng=0, W=W), "tight": post_tight.trials(problem, 400, rng=0, W=W)}, names=["inversion"])
   performance(both).droplevel("block")[["bias", "std", "posterior_std", "coverage"]].round(3)

The tight inversion has no bias on average, but it claims much less spread
than it has, so its interval holds the truth far less often than it claims:
the prior it trusts is not the one the world drew from.
:doc:`examples/comparison` pairs four methods this way.


.. _other-state-labels:

A world on other state labels
-----------------------------

**Two inversions whose states differ are compared on aggregates both add up
to.** A fine and a coarse grid, a monthly and an annual state, the whole state
and the part a solve was restricted to: each state has its own aggregation
to the same rows. ``recovery`` and ``trials`` take the world's as
``W_world``, and the error of each aggregate, :math:`W \hat{x} - W_w x`, keeps
its closed form.

For example: does an annual inversion, fed the observations of a truth with a
season, recover the mean, or does it alias the season through weights that
are larger in some months? The annual inversion has one value per position,
and its forward operator is the monthly one with each position's three
columns added. The aggregates are each position's value: the identity on the
annual state, the mean over the months on the monthly one.

.. ipython:: python

   annual_space = fips.Space({"field": positions})
   per_position = np.kron(np.ones((3, 1)), np.eye(4))  # a monthly element -> its position
   annual = fips.Problem(
       prior=fips.Vector(np.ones(4), annual_space, name="prior"),
       prior_cov=fips.Exponential("x", 1.0).with_std(0.5),
       obs=fips.Vector(np.zeros(24), obs_space, name="obs"),
       obs_cov=fips.Independent().with_std(0.3),
       forward=fips.Operator((weights * own_month) @ per_position, obs_space, annual_space),
   )
   post_annual = fips.solve(annual)
   W_annual = fips.aggregate.sum(annual_space, by="x")  # each position's value
   W_monthly = fips.aggregate.mean(prior.space, by="x")  # the mean over the months

A truth of 1.5 times the prior, flat or with a season (one month at 1.3 times
its mean, the other two at 0.85), with no observation error, against the same
truth recovered by the monthly inversion and averaged over the months:

.. ipython:: python

   def aliasing(season):
       """Return the bias of each position's value from the annual and the monthly inversion."""
       truth = fips.Vector(1.5 * prior.values * np.repeat(season, 4), prior.space)
       annual_bias = post_annual.recovery(problem, truth=truth, W=W_annual, W_world=W_monthly)["bias"]
       monthly_bias = post.recovery(truth=truth, W=W_monthly)["bias"]
       return pd.DataFrame({"annual": annual_bias, "monthly, averaged": monthly_bias}).droplevel("block")

   pd.concat({"flat": aliasing([1.0, 1.0, 1.0]), "first month high": aliasing([1.3, 0.85, 0.85])}, axis=1).round(3)

With a flat truth both inversions are nearly unbiased. With a season, the
annual inversion's biases grow well past its claimed spread while the monthly
inversion's stay small: the annual state cannot tell a strong month from a
strong position, because the weights differ between the months. The honest
world of the monthly problem says the same of the annual inversion's claimed
spread:

.. ipython:: python

   post_annual.recovery(problem, W=W_annual, W_world=W_monthly).droplevel("block")[["std", "posterior_std", "coverage"]].round(3)

The annual inversion's prior has no room for the month-to-month variation the
world draws, so its intervals are too narrow. ``trials`` takes the same
``W_world``, and agrees:

.. ipython:: python

   performance(post_annual.trials(problem, 400, rng=0, W=W_annual, W_world=W_monthly)).droplevel("block")[["std", "posterior_std", "coverage"]].round(3)


Reading the table
-----------------

Two more questions an experiment asks are read off the table of trials with
pandas.

- **Where does the real estimate fall among worlds consistent with the
  prior?** Run the honest world and find the quantile of the real estimate
  in each aggregate's ``estimate`` column. A real value that almost no honest
  world reaches says the truth was not within the prior's spread.
- **What truth is consistent with the estimate?** Regress the truth on the
  estimate over the trials and read the fit at the real value; the quantiles
  of the residuals are its interval. For the tight inversion the slope is
  above 1, because its estimates move less than the truth does, and the fit
  undoes that shrinkage:

.. ipython:: python

   tight = both.xs("tight", level="inversion")
   X = np.column_stack([np.ones(len(tight)), tight["estimate"]])
   beta, *_ = np.linalg.lstsq(X, tight["truth"], rcond=None)
   residual = tight["truth"] - X @ beta
   np.round(beta, 3), np.round(np.percentile(residual, [16, 84]), 3)

With two inversions paired on the same worlds, the same regression on both
estimates reads the truth off the pair.


Randomness
----------

Every draw takes ``rng``: a seed, a :class:`numpy.random.Generator`, or None
for fresh entropy, through :func:`numpy.random.default_rng`, as SciPy's
``rng`` argument does. One generator passed to every call of a loop gives
independent trials. A seed gives the same draws again for the same covariance
forms; a Kronecker covariance on part of its grid draws on the whole grid, so
changing a model's form changes the draws. For trials in parallel, spawn one
generator per worker, ``np.random.SeedSequence(seed).spawn(k)``, and keep each
worker's trials in the table with a ``worker`` key, so the run is reproducible
whatever the scheduling.


How it works
------------

The analytic estimate is linear in the observations, :math:`\hat{x} = x_0 + K
(z - H x_0 - c)`, with the gain :math:`K` of the posterior (its problem's
:math:`H`, :math:`c`, :math:`S_0` and :math:`S_z / \gamma`). When a world
makes the data, with a truth :math:`x \sim N(\mu, P)` (its prior, or a given
truth with :math:`P = 0`) and observations :math:`z = H_w x + c_w + e`,
:math:`e \sim N(0, R_w)` (its forward operator, offset and observation
covariance), the error :math:`\hat{x} - x` is Gaussian, with mean and
covariance

.. math::

    b = (x_0 - \mu) + K (H_w \mu + c_w - H x_0 - c), \qquad
    V = (K H_w - I) P (K H_w - I)^T + K R_w K^T.

``recovery`` reports :math:`W b` and the root of the diagonal of :math:`W V
W^T`, and the coverage of an error :math:`N(\text{bias}, \text{std}^2)` by
the interval :math:`\pm z_{level}` ``posterior_std``. It computes them from
:math:`W K` and :math:`W K H_w - W`, with :math:`P` and :math:`R_w` applied
in their own forms, so it forms nothing :math:`n \times n` or :math:`m \times
m` beyond what the posterior holds. When the world is the problem itself and
gamma is 1, :math:`b = 0` and :math:`V = \hat{S}`. With ``W_world``, the
error of an aggregate is :math:`W \hat{x} - W_w x`, and the same formula
holds with each side's own :math:`W`.

``trials`` makes the data with the world, solves them with
:meth:`~fips.Posterior.with_data` (one product with the factors, because they
depend on where the observations are and how uncertain they are, not on
their values) and keeps each trial's aggregates. For a method whose estimate
is not linear in the observations, each trial's claimed spread and interval
are its own posterior's.

References
----------

Morris, T. P., White, I. R., and Crowther, M. J. (2019). Using simulation
studies to evaluate statistical methods. *Statistics in Medicine*, 38(11),
2074-2102. https://doi.org/10.1002/sim.8086
