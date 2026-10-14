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


.. currentmodule:: fips.covariance

ErrorComponents
===============

.. autosummary::
   :toctree: api/

   ErrorComponent
   DiagonalError
   BlockDecayError
   KroneckerError

Build covariance matrix
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ErrorComponent.build

Binary operations
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

    ErrorComponent.__add__


CovarianceBuilder
=================

Constructor
~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    CovarianceBuilder

Build covariance matrix
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    CovarianceBuilder.build

Binary operations
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    CovarianceBuilder.__add__
    CovarianceBuilder.__radd__
