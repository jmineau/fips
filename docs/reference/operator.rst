=========
Operators
=========

.. currentmodule:: fips

An :class:`Operator` is a matrix with a :class:`Space` of labels on each
axis: a forward operator from the state to the observations, an aggregation,
a gain. Its matrix is kept in the cheapest form that is exact (a dense array,
a SciPy sparse matrix, a diagonal, Kronecker factors; see :doc:`linalg`), and
the labels are beside it. Products, sums and selections check the labels: two
operators combine only when the spaces they meet on have the same labels, in
any order. A DataFrame is a way in (:meth:`Operator.from_frame`) and a way
out (:meth:`Operator.to_frame`).

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips

   obs = pd.Index(["o1", "o2"], name="obs_id")
   H = fips.Operator.from_blocks(
       {
           ("co2", "flux"): pd.DataFrame(
               [[1.0, 0.5], [0.0, 2.0]],
               index=obs,
               columns=pd.Index(["a", "b"], name="cell"),
           ),
           ("co2", "bias"): pd.DataFrame(
               1.0, index=obs, columns=pd.Index(["A"], name="site")
           ),
       },
       name="forward",
   )
   H
   H["co2", "flux"].to_frame()
   x = fips.Vector(np.array([2.0, 1.0, 0.5]), H.cols)
   (H @ x).to_series()

Operator
========

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Operator
   Operator.from_frame
   Operator.from_blocks

Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Operator.data
   Operator.rows
   Operator.cols
   Operator.name
   Operator.shape
   Operator.T

Selection and labels
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Operator.__getitem__
   Operator.align
   Operator.round_labels

Arithmetic
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Operator.__matmul__
   Operator.__add__
   Operator.__sub__
   Operator.__mul__
   Operator.__truediv__
   Operator.__neg__

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Operator.dense
   Operator.to_frame
