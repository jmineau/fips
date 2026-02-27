================
Block Structures
================

.. currentmodule:: fips


Block
=====

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Block

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Block.name
   Block.data
   Block.index

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Block.copy
   Block.to_series
   Block.to_numpy
   Block.to_xarray

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Block.xs
   Block.reindex
   Block.round_index

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Block.from_file
   Block.to_file



MatrixBlock
===========

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatrixBlock

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatrixBlock.name
   MatrixBlock.data
   MatrixBlock.row_block
   MatrixBlock.col_block
   MatrixBlock.index
   MatrixBlock.columns
   MatrixBlock.shape
   MatrixBlock.values

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatrixBlock.copy
   MatrixBlock.to_frame
   MatrixBlock.to_numpy

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatrixBlock.xs
   MatrixBlock.reindex
   MatrixBlock.round_index

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatrixBlock.from_file
   MatrixBlock.to_file

Sparse support
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatrixBlock.is_sparse
   MatrixBlock.to_sparse
   MatrixBlock.to_dense
