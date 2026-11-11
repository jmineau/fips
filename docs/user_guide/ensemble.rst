.. _ensemble:

A model without a Jacobian: the ensemble method
===============================================

Some forward models can only be run. A transport or weather model maps a
state to the observations it makes, but it has no matrix :math:`H` and no
Jacobian anyone can write. **Draw members from the prior with**
:meth:`problem.ensemble(N) <fips.Problem.ensemble>`, **run the model on each,
and give the runs to** ``fips.solve(problem, "ensemble", members=members,
predicted=predicted)``. The solve updates the members with the observations.
The estimate is the mean of the updated members, and the uncertainty is their
spread. With a linear model the answer approaches the analytic posterior as
the members grow, with an error that falls as :math:`1/\sqrt{N}`.

.. contents:: On this page
   :local:
   :depth: 1


When to use it
--------------

Use it when the forward model can only be run, and a run costs enough that
you can afford tens or hundreds of them, not one per state element.

- With a forward operator :math:`H` (a matrix), use ``"analytic"``
  (:doc:`posterior`): it is exact, and the ensemble only approximates it.
- With a forward model whose Jacobian you can write, use ``"gauss_newton"``
  (:doc:`nonlinear`): it finds the mode and draws the posterior itself.
- With a model that has a tangent linear and an adjoint, use
  ``"variational"`` (:doc:`variational`).
- With bounds on the state, use ``"bounded"`` (:doc:`bounds`): the ensemble
  method takes none.


An example
----------

Here the state is a flux along a line of 120 elements, one a kilometer, and
40 sensors along the line each see the elements behind them, the nearest most.
``run_model`` stands in for the model: it takes one member and returns the
observations that member makes. Here it is a line of numpy; a real model is a
job on a cluster, and the members go to it as files.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   n, m = 120, 40
   position = np.arange(n, dtype=float)  # km
   site = np.linspace(2.0, n - 3.0, m)
   behind = site[:, None] - position[None, :]
   weights = np.where(behind >= 0, np.exp(-behind / 3.0), 0.0)

   def run_model(x):
       """Stand-in for one run of the model: the observations a member makes."""
       return weights @ np.asarray(x, dtype=float)

The problem still needs the labels of the state and of the observations, so
the model is a :class:`~fips.ForwardModel` that writes ``__call__`` alone: the
ensemble method needs no Jacobian. Given the runs, the solve never calls the
model, so a model that cannot run from Python may raise in ``__call__``. The
prior is 1 in every element with a standard deviation of 0.5, correlated over
2 km; the observations are of a truth drawn from it.

.. ipython:: python

   state = fips.Space({"flux": pd.RangeIndex(n, name="element")})
   observed = fips.Space({"obs": pd.RangeIndex(m, name="obs_id")})

   class Model(fips.ForwardModel):
       def __call__(self, x):
           return fips.Vector(run_model(x.values), self.rows)

   def build(obs, forward=Model(observed, state)):
       return fips.Problem(
           prior=fips.Vector(np.ones(n), state),
           prior_cov=fips.Exponential("element", 2.0).with_std(0.5),
           obs=obs,
           obs_cov=fips.Independent().with_std(0.2),
           forward=forward,
       )

   truth, obs = build(fips.Vector(np.zeros(m), observed)).simulate(rng=1)
   problem = build(obs)

**Stage one, in fips: draw the members.** Each member is a column, labeled as
the prior is. Their mean is the prior exactly.

.. ipython:: python

   members = problem.ensemble(100, rng=0)
   members.shape

**Stage two, outside fips: run the model on each member.** ``members[j]`` is
one member; write it out with ``to_csv`` or ``to_xarray``. Put the runs in a
DataFrame with one row per observation, labeled as ``problem.obs.index``, and
one column per member, labeled as the members are (in any order). A run is
the model's output, :math:`F(x_j)`, without the problem's offset.

.. ipython:: python

   predicted = pd.DataFrame(
       {j: run_model(members[j]) for j in members.columns},
       index=problem.obs.index,
   )

**Stage three, in fips: update the members.** Here the posterior of the sum of
each quarter of the line:

.. ipython:: python

   post = fips.solve(problem, "ensemble", members=members, predicted=predicted)
   quarter = pd.Series(np.arange(n) // 30, index=pd.RangeIndex(n, name="element"), name="quarter")
   W = fips.aggregate.sum(state, by=quarter)
   post.uncertainty(W).round(2)

When the forward runs in Python (an :class:`~fips.Operator`, or a
:class:`~fips.ForwardModel`'s ``__call__``), one call does all three stages:
``members=100, seed=0`` draws the members with that seed and runs the forward
on each. It gives the same posterior.

.. ipython:: python

   one_call = fips.solve(problem, "ensemble", members=100, seed=0)
   np.allclose(one_call.estimate.values, post.estimate.values)

This stand-in model is linear, so the exact posterior is at hand: its matrix
is ``weights``. (A real model is run per member, not per element, because it
cannot be run as often as there are elements.)

.. ipython:: python

   exact = fips.solve(build(obs, forward=fips.Operator(weights, observed, state)))
   pd.DataFrame({
       "exact": exact.uncertainty(W)["posterior_std"],
       "100 members": post.uncertainty(W)["posterior_std"],
   }).round(2)

At 100 members the standard deviations of the sums are about a tenth too
small. A small ensemble's spread is too small, not only noisy: the members'
sampling noise fits the data and is taken for information. The error falls as
:math:`1/\sqrt{N}`, so four times the members halve it.


Localization
~~~~~~~~~~~~

An ensemble of :math:`N` members spans :math:`N - 1` directions, far fewer
than the state's elements. Its covariance between an element and an
observation far away is noise, not zero, and the update follows that noise:
it moves elements the observation cannot see and shrinks their spread, so the
posterior claims more certainty than it has. **Localization multiplies the
ensemble's covariances by weights that fall to zero with distance.** fips
computes no distances: the weights are yours, an :class:`~fips.Operator` of
the state by the observations (``localization=``) and one of the observations
by themselves (``obs_localization=``). Give both or neither. Here the weights
are Gaspari and Cohn's fifth-order function, which falls from 1 at no distance
to 0 at twice its half-width:

.. ipython:: python

   def gaspari_cohn(distance, half_width):
       """Gaspari and Cohn's function of a distance, zero beyond twice the half-width."""
       r = np.abs(distance) / half_width
       w = np.zeros_like(r)
       a, b = r <= 1.0, (r > 1.0) & (r < 2.0)
       w[a] = -r[a]**5 / 4 + r[a]**4 / 2 + 5 * r[a]**3 / 8 - 5 * r[a]**2 / 3 + 1
       w[b] = r[b]**5 / 12 - r[b]**4 / 2 + 5 * r[b]**3 / 8 + 5 * r[b]**2 / 3 - 5 * r[b] + 4 - 2 / (3 * r[b])
       return w

   def localized(N, half_width):
       return fips.solve(
           problem, "ensemble", members=N, seed=3,
           localization=fips.Operator(gaspari_cohn(position[:, None] - site[None, :], half_width), state, observed),
           obs_localization=fips.Operator(gaspari_cohn(site[:, None] - site[None, :], half_width), observed, observed),
       )

The check is the one of :doc:`synthetic`: 200 sets of data from the problem's
own world, each solved with the same members (``trials`` reuses them), and how
often each sum's 95% interval holds its true value. The analytic posterior's
holds it about 95% of the time.

.. ipython:: python

   def coverage(post):
       """How often the 95% interval of each sum holds its true value, over 200 sets of data."""
       table = post.trials(n=200, rng=5, W=W)
       return table["truth"].between(table["lower"], table["upper"]).mean()

   pd.Series({
       "analytic": coverage(exact),
       "40 members": coverage(fips.solve(problem, "ensemble", members=40, seed=3)),
       "40, localized 10 km": coverage(localized(40, 10.0)),
       "40, localized 20 km": coverage(localized(40, 20.0)),
   })

Without localization, 40 members claim too small a spread, and their intervals
miss the truth far more often than they say. Localized, they hold it about as
often as they claim. The half-width is a choice. Too short cuts real
correlations (here an observation sees about 10 km behind it, and the prior's
correlation reaches 2 km past that), so the update moves less than it should.
Too long lets the noise back in (no localization is an infinite half-width).
Choose it with ``trials``, as here, in a synthetic world shaped like yours.


What the posterior reports
--------------------------

The ensemble's posterior is a Gaussian one, written from the updated members.
The :ref:`members table <posterior-members>` lists every member for every kind
of answer; for the ensemble:

- :attr:`~fips.Posterior.estimate` (and :attr:`~fips.Posterior.mean`) is the
  mean of the updated members. :attr:`~fips.Posterior.variances`,
  :meth:`~fips.Posterior.cov` and :attr:`~fips.Posterior.kalman_gain` are
  those of the updated members.
- :meth:`~fips.Posterior.sample` returns the updated members themselves, so
  it raises when asked for more than there are.
- :attr:`~fips.Posterior.dofs`, :attr:`~fips.Posterior.information_content`,
  :attr:`~fips.Posterior.chi2` and :meth:`~fips.Posterior.chi2_tests` are the
  analytic method's formulas, with the members' covariance of the modeled
  observations in place of :math:`H S_0 H^T`.
- :meth:`~fips.Posterior.with_data` and :meth:`~fips.Posterior.with_gamma`
  reuse the runs. :meth:`~fips.Posterior.trials` runs the model only to make
  each trial's data, so it needs a model that runs in Python.
- :attr:`~fips.Posterior.log_evidence` raises: the evidence of a sample
  covariance is not the evidence. :meth:`~fips.Posterior.recovery` and the
  data worth raise: the estimate is not claimed linear in the observations.
- What needs :math:`H` (:meth:`~fips.Posterior.averaging_kernel`,
  :attr:`~fips.Posterior.dofs_per_element`, the leverage, the innovation
  covariance) works when the forward is an :class:`~fips.Operator`, and
  raises :class:`TypeError` for a model.


Options
-------

``fips.solve(problem, "ensemble", **options)`` takes
(:class:`~fips.methods.ensemble.EnsembleOptions`):

.. list-table::
   :header-rows: 1
   :widths: 20 14 66

   * - Option
     - Default
     - What it does
   * - ``members``
     - required
     - The prior members: the DataFrame :meth:`~fips.Problem.ensemble`
       returns (one row per state element, one column per member, at least
       two), or a number of members to draw. The update uses the members'
       own mean and covariance as the prior, not ``problem.prior`` and
       ``problem.prior_cov``: members from an earlier update carry it
       forward.
   * - ``predicted``
     - None
     - The model's output for each member, without the offset: one row per
       observation, the members' columns. Given, fips runs no model; it needs
       ``members`` as a DataFrame. None runs the problem's forward on each
       member.
   * - ``localization``
     - None
     - Weights on the ensemble's covariance of the state with the modeled
       observations: an :class:`~fips.Operator`, state by observations.
   * - ``obs_localization``
     - None
     - Weights on the ensemble's covariance of the modeled observations: an
       :class:`~fips.Operator`, observations by observations, symmetric.
   * - ``seed``
     - 0
     - The seed of the members drawn when ``members`` is a number.
   * - ``gamma``
     - 1.0
     - The regularization factor: the observation covariance is divided by
       it, as for every method (:doc:`posterior`).


Warnings and checks
-------------------

The method warns about nothing. **A spread too small for the number of
members comes with no message**: check it with
:meth:`~fips.Posterior.trials` in a synthetic world, as above. When fips runs
the forward itself, it logs one INFO line, naming the model and the number of
members; INFO lines are silent until you configure logging
(:ref:`repeated-inversions`).

It raises:

- :class:`ValueError` when ``members`` is missing or holds fewer than two
  members, when two members share a label, when ``predicted`` comes with a
  number of members, or when only one of the two localizations is given;
- :class:`ValueError` when the rows of ``members`` or ``predicted`` are not
  the labels of the prior or of the observations (it names the labels on one
  side only), when the columns of ``predicted`` are not the members' labels,
  or when a value, given or run, is not finite;
- :class:`ValueError` when ``obs_localization`` is not symmetric, or when its
  weights make the localized covariance of the modeled observations clearly
  negative (Gaspari and Cohn's function never does);
- :class:`TypeError` when ``members`` is neither a number nor a DataFrame,
  ``predicted`` is not a DataFrame, or a localization is not an
  :class:`~fips.Operator`;
- :class:`numpy.linalg.LinAlgError` when the observation covariance is not
  positive definite: the update whitens the observations with its Cholesky
  factor.


Cost
----

**The cost is the runs**: :math:`N` runs of the model, made by you, once.
Everything fips does is small beside them. Without localization, the update
is a thin SVD of the whitened runs, :math:`O(m N \min(m, N))`, and products
with the :math:`n \times N` members. With localization, it eigendecomposes an
:math:`m \times m` matrix and holds an :math:`n \times m` dense one. A
non-diagonal observation covariance is factored once, dense. Another
:math:`\gamma` or other observation values cost a product, with no more runs.


How it works
------------

The update is a deterministic square-root ensemble Kalman filter (EnSRF)
applied to every observation at once, as CarbonTracker does. Write :math:`X`
for the members (:math:`n \times N`) and :math:`Y` for their runs
(:math:`m \times N`, :math:`Y_{\cdot j} = F(x_j)`). Their deviations from
their means, scaled, are

.. math::

   A = (X - \bar{x}) / \sqrt{N - 1}, \qquad B = (Y - \bar{y}) / \sqrt{N - 1},

and they give the ensemble's covariances :math:`P_{xy} = A B^T` and
:math:`P_{yy} = B B^T`. With localization, each is multiplied element by
element by its weights, :math:`\rho_{xy} \circ P_{xy}` and :math:`\rho_{yy}
\circ P_{yy}`. With the innovation :math:`d = z - c - \bar{y}` and
:math:`R = S_z / \gamma`, the mean moves by the ensemble's gain:

.. math::

   \bar{x}^a = \bar{x} + K d, \qquad K = P_{xy} (P_{yy} + R)^{-1}.

The deviations move by the reduced gain of Whitaker and Hamill's batch
square-root filter, :math:`A^a = A - \tilde{K} B`. With :math:`S_z = L L^T`
and :math:`M = L^{-1} P_{yy} L^{-T}`,

.. math::

   \tilde{K} = P_{xy} L^{-T} \, G^{-1/2} (G^{1/2} + \gamma^{-1/2} I)^{-1},
   \qquad G = M + I / \gamma,

chosen so that the covariance of the updated deviations, :math:`A^a A^{aT}`,
is the Kalman posterior of the ensemble's own covariance :math:`A A^T`,
exactly. Without localization the same update is the ensemble transform of
the ETKF, :math:`A^a = A (I + S^T S)^{-1/2}` with :math:`S = \sqrt{\gamma}
L^{-1} B`, which fips computes from the thin SVD of :math:`L^{-1} B`. With
localization it eigendecomposes the localized :math:`M`. The updated
deviations' covariance is then not the localized Kalman posterior:
:math:`N` members cannot hold a covariance of more than :math:`N - 1`
directions; weights of one everywhere give the unlocalized update exactly.
None of the factors depends on :math:`\gamma` or on the observations' values,
which is why another :math:`\gamma` or other data cost a product.

Assimilating window after window is a loop of ``fips.solve(...,
"ensemble")`` calls, each starting from the members the last one updated
(``post.sample(N)``), with the next window's prior added. The method rejects
no outliers.


References
----------

Gaspari, G., and Cohn, S. E. (1999). Construction of correlation functions in
two and three dimensions. *Quarterly Journal of the Royal Meteorological
Society*, 125(554), 723-757. https://doi.org/10.1002/qj.49712555417

Houtekamer, P. L., and Mitchell, H. L. (2001). A sequential ensemble Kalman
filter for atmospheric data assimilation. *Monthly Weather Review*, 129(1),
123-137. https://doi.org/10.1175/1520-0493(2001)129<0123:ASEKFF>2.0.CO;2

Peters, W., Miller, J. B., Whitaker, J., Denning, A. S., Hirsch, A., Krol,
M. C., Zupanski, D., Bruhwiler, L., and Tans, P. P. (2005). An ensemble data
assimilation system to estimate CO2 surface fluxes from atmospheric trace gas
observations. *Journal of Geophysical Research*, 110, D24304.
https://doi.org/10.1029/2005JD006157

Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., and Whitaker,
J. S. (2003). Ensemble square root filters. *Monthly Weather Review*, 131(7),
1485-1490. https://doi.org/10.1175/1520-0493(2003)131<1485:ESRF>2.0.CO;2

Whitaker, J. S., and Hamill, T. M. (2002). Ensemble data assimilation without
perturbed observations. *Monthly Weather Review*, 130(7), 1913-1924.
https://doi.org/10.1175/1520-0493(2002)130<1913:EDAWPO>2.0.CO;2
