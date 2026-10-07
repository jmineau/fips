===================
Covariance Matrices
===================

.. currentmodule:: fips

Covariance matrices are used in FIPS to represent uncertainty in the data provided to the inverse problem.


CovarianceMatrix
================

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceMatrix

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceMatrix.name
   CovarianceMatrix.data
   CovarianceMatrix.index
   CovarianceMatrix.columns
   CovarianceMatrix.shape
   CovarianceMatrix.values
   CovarianceMatrix.variances
   CovarianceMatrix.blocks

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceMatrix.copy
   CovarianceMatrix.to_frame
   CovarianceMatrix.to_numpy

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceMatrix.xs
   CovarianceMatrix.reindex
   CovarianceMatrix.round_index

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceMatrix.from_file
   CovarianceMatrix.to_file

Sparse support
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceMatrix.is_sparse
   CovarianceMatrix.to_sparse
   CovarianceMatrix.to_dense

Computations
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CovarianceMatrix.force_symmetry


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
