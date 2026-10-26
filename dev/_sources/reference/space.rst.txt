==================
Spaces and Vectors
==================

.. currentmodule:: fips

A :class:`Space` is the labels of a state, of the observations, or of one
axis of a matrix: an ordered set of named blocks, each labeled by its own
pandas index. A :class:`Vector` is values on a space, one for each label.
fips keeps labels as they are given: it does not convert them to numbers or
round them, and it refuses duplicate and missing labels.

.. ipython:: python

   import pandas as pd
   import fips

   prior = fips.Vector.from_blocks(
       {
           "flux": pd.Series([1.0, 2.0], index=pd.Index([1, 2], name="cell")),
           "bias": pd.Series([0.0], index=pd.Index(["A"], name="site")),
       },
       name="prior",
   )
   prior.space
   prior.to_series()
   prior["flux"].space.select("flux")

A vector of one block is a single component of the state or of the
observations: ``fips.Vector.from_series(series, block="flux")``.

Space
=====

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Space
   Space.from_index

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Space.names
   Space.levels
   Space.index
   Space.offsets

Selection
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Space.select
   Space.slice

Alignment and labels
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Space.align
   Space.round_labels

Vector
======

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector
   Vector.from_series
   Vector.from_blocks

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector.values
   Vector.space
   Vector.name
   Vector.index

Selection and conversion
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector.__getitem__
   Vector.to_series
   Vector.to_xarray

Alignment and labels
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Vector.align
   Vector.combine_first
   Vector.round_labels
