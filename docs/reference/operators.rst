=========
Operators
=========

.. currentmodule:: fips

Operators perfom mathematical operations on blocks and vectors, such as addition, multiplication, etc.
The `ForwardOperator` class represents the forward operator in an inverse problem, and provides methods for applying the operator and its adjoint to vectors.


ForwardOperator
===============

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardOperator

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardOperator.name
   ForwardOperator.data
   ForwardOperator.index
   ForwardOperator.obs_index
   ForwardOperator.columns
   ForwardOperator.state_index
   ForwardOperator.shape
   ForwardOperator.values
   ForwardOperator.blocks

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardOperator.copy
   ForwardOperator.to_frame
   ForwardOperator.to_numpy


Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardOperator.xs
   ForwardOperator.reindex
   ForwardOperator.round_index

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardOperator.from_file
   ForwardOperator.to_file

Sparse support
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardOperator.is_sparse
   ForwardOperator.to_sparse
   ForwardOperator.to_dense

Computations
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardOperator.convolve


Convolution
===========

.. currentmodule:: fips.operators

The `convolve` method applies the forward operator to a vector, and can be used to compute the predicted observations given a model state.

.. autosummary::
   :toctree: api/

   convolve
