=======================
Aggregation and Offsets
=======================

.. currentmodule:: fips

:func:`aggregate.sum` and :func:`aggregate.mean` build an aggregation
:math:`W`: an :class:`Operator` from a :class:`Space` (its columns) to the
groups of its labels (its rows), sparse. On the observations,
:meth:`Problem.aggregate` gives the problem of :math:`W z`, :math:`W H`,
:math:`W S_z W^T` and :math:`W c` (hourly observations to daily means); on the
state, :meth:`Posterior.cov` and :meth:`Posterior.averaging_kernel` take it for
the posterior of totals or means. The groups are given by ``by``: level names,
a :class:`pandas.Grouper` with a ``freq`` (time bins at a frequency), an
:class:`pandas.IntervalIndex` named after a level (time bins given), or a
:class:`pandas.Series` that maps a level's values to groups (sites to networks,
cells to regions).

:func:`offset_block` uses the same groups for a block of unknown offsets in the
state: one per group of observations (a site's background in a month), with
the forward block that adds each to its observations. Its prior covariance is
any model on the offsets' labels, so ``Exponential("obs_time", "30D",
within="site")`` correlates the months of a site.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   hours = pd.date_range("2024-01-01", periods=6, freq="8h")
   labels = pd.MultiIndex.from_product([["A", "B"], hours], names=["site", "obs_time"])
   obs = fips.Vector.from_series(pd.Series(np.arange(12.0), index=labels), block="ch4")
   W = fips.aggregate.mean(obs.space, by=["site", pd.Grouper(level="obs_time", freq="1D")])
   (W @ obs).to_series()
   offsets, forward = fips.offset_block(obs.space, by=pd.Grouper(level="obs_time", freq="1D"))
   offsets.space

Aggregation
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   aggregate.sum
   aggregate.mean

Offsets
~~~~~~~
.. autosummary::
   :toctree: api/

   offset_block
