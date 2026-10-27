===========
Covariances
===========

.. currentmodule:: fips

A :class:`Covariance` is a symmetric :class:`Operator` from a :class:`Space`
to itself: the prior covariance or the observation covariance of a problem.
It keeps its matrix in the cheapest form (a diagonal covariance is its
variances, a Kronecker one its factors, independent blocks each their own
form) and adds what a covariance needs: its variances, solves, the
log-determinant, the quadratic form of its inverse, a square root and a draw
from it. A dense or sparse matrix must be symmetric, to 1e-10 of its largest
entry; one that is not (an upper triangle, say) raises, naming the two entries.

.. ipython:: python

   import pandas as pd
   import fips

   flux = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
   bias = fips.Space({"bias": pd.Index(["north"], name="site")})
   S_0 = fips.Covariance.block_diag(
       fips.Covariance.diagonal(4.0, flux),
       fips.Covariance.diagonal(1.0, bias),
       name="prior_cov",
   )
   S_0
   S_0.variances.to_series()
   S_0.logdet()

Covariance
==========

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Covariance
   Covariance.diagonal
   Covariance.block_diag

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Covariance.space
   Covariance.variances

Linear algebra
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Covariance.solve
   Covariance.logdet
   Covariance.quadratic_form
   Covariance.sqrt

Random draws
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Covariance.sample


Covariance models
=================

A correlation model says how correlated two labels are, from the coordinates
on some levels of their space; ``model.with_std(std)`` makes it a covariance.
With the standard deviations as a :class:`Vector`, that is a :class:`Covariance`
at once; with a number, or a pandas Series on the block's labels (lined up by
label; a plain array is refused), a :class:`~fips.covariance.CovarianceModel`,
which :meth:`~fips.covariance.CovarianceModel.build` puts on a space. Models add,
and :meth:`Covariance.block_diag` takes them by block name.

.. ipython:: python

   prior = fips.Kronecker(
       fips.Exponential("time", "30D"),
       fips.Exponential(["lat", "lon"], 20.0, metric="haversine"),
   )
   prior
   mdm = fips.Independent().with_std(0.01) + fips.Exponential(
       "obs_time", "2.8h", within=["site", "obs_time.date"]
   ).with_std(0.02)
   mdm

Kernels
~~~~~~~
.. autosummary::
   :toctree: api/

   Exponential
   Gaussian
   Matern
   Constant

Combinations
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Independent
   Kronecker

Parameters
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Exponential.levels
   Exponential.scale
   Exponential.metric
   Exponential.within
   Matern.nu
   Constant.within
   Kronecker.factors
   Kronecker.levels

.. currentmodule:: fips.covariance

The base of every model
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CorrelationModel
   CorrelationModel.levels
   CorrelationModel.correlation
   CorrelationModel.with_std

A covariance waiting for its space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceModel
   CovarianceModel.build
   CovarianceModel.__add__

Distances
~~~~~~~~~
The kernels measure distance with these functions of
:mod:`fips.covariance.distances`. Points on the Earth are latitude and
longitude in degrees, and their distances are in kilometers on a sphere of
radius 6371 km.

.. currentmodule:: fips.covariance.distances

.. autosummary::
   :toctree: api/

   haversine_matrix
   chordal_matrix
   time_diff_matrix
