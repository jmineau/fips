======================
Multi-Block Structures
======================

.. currentmodule:: fips

Vector
======

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector.name
   Vector.data
   Vector.index
   Vector.shape
   Vector.values
   Vector.blocks

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector.copy
   Vector.to_series
   Vector.to_numpy
   Vector.to_xarray

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector.__getitem__
   Vector.xs
   Vector.reindex
   Vector.round_index

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector.from_file
   Vector.to_file


Matrix
======

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix.name
   Matrix.data
   Matrix.index
   Matrix.columns
   Matrix.shape
   Matrix.values

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix.copy
   Matrix.to_frame
   Matrix.to_numpy

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix.__getitem__
   Matrix.xs
   Matrix.reindex
   Matrix.round_index

Computations
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix.scale

Binary operations
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix.__add__

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix.from_file
   Matrix.to_file

Sparse support
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Matrix.is_sparse
   Matrix.to_sparse
   Matrix.to_dense
