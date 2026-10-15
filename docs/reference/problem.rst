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

.. seealso::

   See :doc:`../terminology` for definitions of prior, posterior, forward operator,
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

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Problem.from_file
   Problem.to_file
