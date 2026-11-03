=======
Problem
=======

.. currentmodule:: fips

A :class:`Problem` holds the inputs of a linear inverse problem: the prior and
its covariance, the observations and theirs, the forward operator and an
optional offset. It checks their labels once, when it is built, by an exact
join: the forward operator, a covariance or the offset must have exactly the
labels of the prior or of the observations, in any order, or it raises
naming the blocks and labels on one side only. A covariance model is built on
the prior's or the observations' space. :func:`solve` solves a problem.
:meth:`Problem.restrict` holds part of the state at its prior (the cells no
observation sees well), and :meth:`Problem.aggregate` aggregates the
observations (see :doc:`aggregate`).

.. seealso::

   See :doc:`/getting_started/terminology` for definitions of prior, posterior, forward operator,
   and other inverse problem terminology.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   cells = pd.Index(["a", "b"], name="cell")
   obs_id = pd.Index(["o1", "o2", "o3"], name="obs_id")
   prior = fips.Vector.from_series(pd.Series(0.0, index=cells), block="flux")
   obs = fips.Vector.from_series(pd.Series([1.0, 2.0, 3.0], index=obs_id), block="co2")
   H = fips.Operator(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), obs.space, prior.space)
   problem = fips.Problem(
       prior=prior,
       prior_cov=fips.Independent().with_std(1.0),
       obs=obs,
       obs_cov=fips.Independent().with_std(0.3),
       forward=H,
   )
   problem
   problem.predict(problem.prior)

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Problem

Inputs
~~~~~~
.. autosummary::
   :toctree: api/

   Problem.prior
   Problem.prior_cov
   Problem.obs
   Problem.obs_cov
   Problem.forward
   Problem.offset

Without a solve
~~~~~~~~~~~~~~~
What needs no solve and no :math:`m \times m` matrix is on the problem: what
the problem predicts, a truth and its synthetic observations
(:ref:`how to use them <synthetic-data>`), and the problems made from it (the
observations aggregated, part of the state held at its prior). What depends on
the solve is on the :class:`Posterior`.

.. autosummary::
   :toctree: api/

   Problem.predict
   Problem.simulate
   Problem.aggregate
   Problem.restrict

Serialization
~~~~~~~~~~~~~
A problem pickles with the standard library (``pickle.dump(problem, file)``).
