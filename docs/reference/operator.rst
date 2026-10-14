================
Linear Operators
================

.. currentmodule:: fips.operator

.. automodule:: fips.operator
   :no-members:

Forms
~~~~~
.. autosummary::
   :toctree: api/

   Operator
   DenseOperator
   SparseOperator
   DiagonalOperator

Building
~~~~~~~~
.. autosummary::
   :toctree: api/

   as_operator

Linear algebra
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Operator.dot
   Operator.solve
   Operator.dense
   Operator.diagonal
   Operator.trace
   Operator.to_frame

Stored data
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DenseOperator.array
   SparseOperator.matrix
   DiagonalOperator.values

Testing
~~~~~~~
.. autosummary::
   :toctree: api/

   dottest
