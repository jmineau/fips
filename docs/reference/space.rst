=================
Spaces and Fields
=================

.. currentmodule:: fips

A :class:`Space` is the labels of a state, of the observations, or of one
axis of a matrix: an ordered set of named blocks, each labeled by its own
pandas index. A :class:`Field` is values on a space, one for each label.
fips keeps labels as they are given: it does not convert them to numbers or
round them, and it refuses duplicate and missing labels.

.. ipython:: python

   import pandas as pd
   import fips

   prior = fips.Field.from_blocks(
       {
           "flux": pd.Series([1.0, 2.0], index=pd.Index([1, 2], name="cell")),
           "bias": pd.Series([0.0], index=pd.Index(["A"], name="site")),
       },
       name="prior",
   )
   prior.space
   prior.to_series()
   prior["flux"].space.select("flux")

:class:`Block` and :class:`Vector` are deprecated in favor of :class:`Field`;
they are assembled and checked with a :class:`Space` already.

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

Field
=====

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Field
   Field.from_series
   Field.from_blocks

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Field.values
   Field.space
   Field.name
   Field.index

Selection and conversion
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Field.__getitem__
   Field.to_series

Alignment and labels
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Field.align
   Field.round_labels
