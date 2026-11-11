.. _methods:

Choosing a method
=================

``fips.solve(problem, method, **options)`` solves one :class:`~fips.Problem`
by one of five methods. The problem is the same for all of them: a prior and
its covariance, observations and theirs, and a forward map. The methods
differ in what they allow (bounds, a forward model that is not linear, a
model that can only be run, a problem too large to factor) and so in what
their answer is. **Start with** ``"analytic"``, **the default**: it is exact,
it gives every diagnostic, and every other method gives its answer when the
problem is linear and nothing else is asked of it.

.. contents:: On this page
   :local:
   :depth: 1


The methods
-----------

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 14 26 18 22 20

   * - Method
     - For a problem with
     - The estimate is
     - The uncertainty comes from
     - Guide
   * - ``"analytic"``
     - a forward operator :math:`H`, small enough to factor an
       :math:`m \times m` or :math:`n \times n` matrix
     - the posterior mean
     - the exact posterior covariance
     - :doc:`posterior`
   * - ``"bounded"``
     - a forward operator, and bounds on the state (``lower=``, ``upper=``)
     - the posterior mode
     - kept draws of the bounded posterior
     - :doc:`bounds`
   * - ``"gauss_newton"``
     - a forward model :math:`F(x)` and its Jacobian
     - the posterior mode
     - kept draws of the posterior
     - :doc:`nonlinear`, :doc:`positive`
   * - ``"ensemble"``
     - a forward model that can only be run, on members drawn from the prior
     - the mean of the updated members
     - the spread of the updated members
     - :doc:`ensemble`
   * - ``"variational"``
     - a forward operator or model given by its products, too large to
       factor
     - the posterior mean (the mode, for a forward model)
     - a low-rank posterior covariance
     - :doc:`variational`

:func:`fips.available_methods` lists the names, and each method's options are
in :doc:`/reference/posterior`.


Which one
---------

Ask in this order:

1. **Can the forward map be written as a matrix, and does the problem fit in
   memory?** Use ``"analytic"``. Most problems stop here, and a large sparse
   forward operator with a Kronecker prior covariance still fits.
2. **Must part of the state stay at or above a bound** (a quantity that
   cannot be negative)? Use ``"bounded"``, or solve for a transform of the
   state that is always positive (:class:`~fips.LogNormalForward`,
   :class:`~fips.SoftplusForward`) with ``"gauss_newton"``. The two change
   the prior in different ways: `Keeping a quantity positive`_ below.
3. **Is the forward map a model that is not linear, with a Jacobian you can
   write?** Use ``"gauss_newton"``, after checking the Jacobian with
   :func:`fips.taylortest`.
4. **Can the model only be run** (no matrix, no Jacobian)? Use
   ``"ensemble"``: draw members, run the model on each, and update them.
5. **Is the problem too large to factor**, or is the model given by a tangent
   linear and an adjoint? Use ``"variational"``.

Two questions apply to every method: whether the covariance settings (the
prior's standard deviation and correlation length, the observation error)
are right, which :doc:`settings` answers from the data, and whether their
uncertainty matters to the answer, which :doc:`hierarchical` carries into it.


Three kinds of answer
---------------------

**Every method returns a** :class:`~fips.Posterior`, **but not every posterior
can answer every question.** There are three kinds:

- a **Gaussian** posterior: the estimate is the mean and the covariance is
  known (``"analytic"``, ``"ensemble"``, and ``"variational"`` when its rank
  holds every direction the data inform);
- a **sampled** posterior: the estimate is the mode, and the mean, the
  variances and the intervals come from draws the solve keeps (``"bounded"``,
  ``"gauss_newton"``);
- a **low-rank** posterior: the covariance is the prior's less the ``rank``
  directions the observations inform most (``"variational"``).

**Which interval you get follows from the kind.** For a Gaussian or low-rank
posterior, the ``lower`` and ``upper`` of :meth:`~fips.Posterior.uncertainty`
and :meth:`~fips.Posterior.trials` are the estimate plus or minus ``z``
standard deviations. For a sampled one they are quantiles of the draws, which
follow the posterior's skew. **The** ``reduction`` **columns compare against the problem's Gaussian
prior.** For ``"bounded"`` they include what the bounds remove, and for
``"ensemble"`` the members' sampling error, so they are not zero even when
the observations tell nothing; DOFS is the measure of what the observations
told. A member a kind cannot give raises
:class:`~fips.UnavailableError`; for example, the data worth needs an
estimate linear in the observations, which only the analytic method gives.
The :ref:`members table <posterior-members>` lists every member for every
kind.



.. _methods-positive:

Keeping a quantity positive
---------------------------

A quantity that cannot be negative has three answers in fips, and they are
not the same answer: each puts a different prior on the state.

- **The analytic posterior** ignores the bound. Some elements may come out
  negative where the observations see little, but an aggregate of many
  elements is often close to Gaussian, and its estimate and interval can be
  trusted. It is also by far the cheapest.
- ``"bounded"`` **with** ``lower=0`` keeps the Gaussian prior and cuts it at
  the bound. Its mode is the analytic estimate moved onto the bound along the
  posterior's correlations. Its draws are of the cut posterior, whose mean is
  higher than the Gaussian's wherever the data are weak, so an aggregate read
  from its draws sits above a prior mean you meant as the mean. Use it for an
  element-by-element estimate that must not cross the bound; read an aggregate
  from its draws only if the cut prior is the prior you mean.
  :ref:`Rectified draws <bounds-rectified>` shift an aggregate less.
- :class:`~fips.LogNormalForward` solves for the logarithm of a factor of a
  reference value, so the uncertainty is relative ("a factor of 2"), and an
  element whose reference is zero stays zero.
  :class:`~fips.SoftplusForward` solves for a value that is itself where it is
  large and bends to stay positive near zero, so the uncertainty is additive
  ("10 ± 2"). Both keep the prior's mean and standard deviation as you give
  them (``prior_for``), and draw the posterior exactly.

:doc:`examples/comparison` runs all four on one synthetic problem, in worlds
that match each prior and in one that matches only the lognormal. Its
conclusions, for that problem: every method's intervals hold the truth about
as often as they claim in its own world; the analytic posterior's aggregate is
unbiased wherever the truth is correlated as the priors say, and its interval
covers in every world tried; the lognormal transform gives the best element
estimates when the truth is a factor of the reference; and the bounded
method's draws put an aggregate well above a truth whose mean is the prior's.
Run the same comparison on your own problem with :doc:`synthetic`.



.. _methods-transformed:

Reading a transformed state
---------------------------

With :class:`~fips.LogNormalForward` or :class:`~fips.SoftplusForward`, the
state is the variable :math:`u`, not the value. ``model.transform`` maps the
estimate, or each draw, to the value. **An aggregate of the value comes from
the draws**: transform the draws, then aggregate. ``post.sample(n, W=W)``
aggregates :math:`u`, which is not an aggregate of the value. The same holds
for :meth:`~fips.Posterior.trials`, which aggregates the state: write the loop
over draws yourself, as :doc:`examples/comparison` does.


Warnings
--------

Each method warns when something it found changes how far its answer
can be trusted (a prior wide against its bounds, draws that mix slowly, steps
that converge slowly, rank too small to hold what the data inform). Each
warning is a :class:`~fips.FipsWarning`, which **Python shows without any
setup**, once for each line of your code that leads to it; make them errors
or hide them with :func:`warnings.filterwarnings`. fips also logs each one
under the ``fips`` logger (:ref:`repeated-inversions`).


Cost
----

The analytic method factors one matrix once; a few thousand state elements or
observations take a fraction of a second, and a gamma sweep or new data reuse
the factors. ``"bounded"`` finds its mode for little more, but its draws, made
the first time a spread is asked for, cost a step of the state's size for each
time a draw meets a bound. ``"gauss_newton"`` costs a solve per iteration
(typically a handful) and then its draws, each several runs of the model and
its Jacobian's transpose. ``"ensemble"`` costs one run of the model per member,
done by you. ``"variational"`` costs products with :math:`H` and its transpose:
one per conjugate-gradient step, and ``rank`` more for the uncertainty. Each
method's page says what it scales with.
