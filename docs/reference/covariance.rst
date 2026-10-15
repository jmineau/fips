===========
Covariances
===========

.. currentmodule:: fips

A :class:`Covariance` is a symmetric :class:`Operator` from a :class:`Space`
to itself: the prior error or the model-data mismatch of an inverse problem.
It keeps its matrix in the cheapest form (a diagonal covariance is its
variances, a Kronecker one its factors, independent blocks each their own
form) and adds what a covariance needs: its variances, solves, the
log-determinant, the quadratic form of its inverse and a square root.

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


Covariance models
=================

A correlation model says how correlated two labels are, from the coordinates
on some levels of their space; ``model.with_std(std)`` makes it a covariance.
With the standard deviations as a :class:`Vector`, that is a :class:`Covariance`
at once; with a scalar or an array, a :class:`~fips.models.CovarianceModel`,
which :meth:`~fips.models.CovarianceModel.build` puts on a space. Models add,
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

.. currentmodule:: fips.correlation

The base of every model
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CorrelationModel
   CorrelationModel.levels
   CorrelationModel.correlation
   CorrelationModel.with_std

.. currentmodule:: fips.models

A covariance waiting for its space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceModel
   CovarianceModel.build
   CovarianceModel.__add__
