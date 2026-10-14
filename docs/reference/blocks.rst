================
Block Structures
================

.. currentmodule:: fips


Block
=====

.. deprecated:: 0.2.0b1
   :class:`Block` will be removed; build a :class:`Field` instead (see
   :doc:`space`).

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
