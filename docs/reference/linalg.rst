==============
Linear Algebra
==============

.. currentmodule:: fips.linalg

.. automodule:: fips.linalg
   :no-members:

Forms
~~~~~
.. autosummary::
   :toctree: api/

   LinearOperator
   DenseOperator
   SparseOperator
   DiagonalOperator
   BlockDiagonalOperator

Building
~~~~~~~~
.. autosummary::
   :toctree: api/

   as_linear_operator

Linear algebra
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   LinearOperator.dot
   LinearOperator.solve
   LinearOperator.logdet
   LinearOperator.sqrt
   LinearOperator.dense
   LinearOperator.diagonal
   LinearOperator.trace

Stored data
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DenseOperator.array
   SparseOperator.matrix
   DiagonalOperator.values
   BlockDiagonalOperator.blocks

Testing
~~~~~~~
.. autosummary::
   :toctree: api/

   dottest

Kronecker factors
=================

.. automodule:: fips.linalg.kronecker
   :no-members:

.. currentmodule:: fips.linalg

.. autosummary::
   :toctree: api/

   KroneckerOperator

Linear algebra
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   KroneckerOperator.solve
   KroneckerOperator.logdet
   KroneckerOperator.dense
   KroneckerOperator.diagonal

Stored data
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   KroneckerOperator.factors
   KroneckerOperator.positions
   KroneckerOperator.std
