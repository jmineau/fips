======================
Multi-Block Structures
======================

.. currentmodule:: fips

Vector
======

.. deprecated:: 0.2.0b1
   :class:`Vector` will be removed; build a :class:`Field` instead (see
   :doc:`space`).

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
