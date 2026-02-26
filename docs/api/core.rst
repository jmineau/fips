Core Data Structures
====================

Block
-----

.. currentmodule:: fips

.. autoclass:: Block

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~Block.name
      ~Block.data
      ~Block.index
      ~Block.dtype

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~Block.copy
      ~Block.sel
      ~Block.where
      ~Block.to_numpy
      ~Block.to_series

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: name
   .. autoattribute:: data
   .. autoattribute:: index
   .. autoattribute:: dtype
   .. automethod:: copy
   .. automethod:: sel
   .. automethod:: where
   .. automethod:: to_numpy
   .. automethod:: to_series


Vector
------

.. autoclass:: Vector

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~Vector.blocks
      ~Vector.index
      ~Vector.shape
      ~Vector.size

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~Vector.copy
      ~Vector.sel
      ~Vector.where
      ~Vector.to_numpy
      ~Vector.to_dataframe
      ~Vector.concat

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: blocks
   .. autoattribute:: index
   .. autoattribute:: shape
   .. autoattribute:: size
   .. automethod:: copy
   .. automethod:: sel
   .. automethod:: where
   .. automethod:: to_numpy
   .. automethod:: to_dataframe
   .. automethod:: concat


MatrixBlock
-----------

.. autoclass:: MatrixBlock

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~MatrixBlock.row_name
      ~MatrixBlock.col_name
      ~MatrixBlock.data
      ~MatrixBlock.row_index
      ~MatrixBlock.col_index
      ~MatrixBlock.shape

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~MatrixBlock.copy
      ~MatrixBlock.sel
      ~MatrixBlock.to_numpy
      ~MatrixBlock.to_dataframe
      ~MatrixBlock.T
      ~MatrixBlock.dot

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: row_name
   .. autoattribute:: col_name
   .. autoattribute:: data
   .. autoattribute:: row_index
   .. autoattribute:: col_index
   .. autoattribute:: shape
   .. automethod:: copy
   .. automethod:: sel
   .. automethod:: to_numpy
   .. automethod:: to_dataframe
   .. autoattribute:: T
   .. automethod:: dot


Matrix
------

.. autoclass:: Matrix

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~Matrix.blocks
      ~Matrix.row_index
      ~Matrix.col_index
      ~Matrix.shape

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~Matrix.copy
      ~Matrix.sel
      ~Matrix.to_numpy
      ~Matrix.to_dataframe
      ~Matrix.T
      ~Matrix.dot

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: blocks
   .. autoattribute:: row_index
   .. autoattribute:: col_index
   .. autoattribute:: shape
   .. automethod:: copy
   .. automethod:: sel
   .. automethod:: to_numpy
   .. automethod:: to_dataframe
   .. autoattribute:: T
   .. automethod:: dot


CovarianceMatrix
----------------

.. autoclass:: CovarianceMatrix

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~CovarianceMatrix.blocks
      ~CovarianceMatrix.index
      ~CovarianceMatrix.shape

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~CovarianceMatrix.copy
      ~CovarianceMatrix.sel
      ~CovarianceMatrix.to_numpy
      ~CovarianceMatrix.to_dataframe
      ~CovarianceMatrix.sqrt
      ~CovarianceMatrix.inv
      ~CovarianceMatrix.dot

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: blocks
   .. autoattribute:: index
   .. autoattribute:: shape
   .. automethod:: copy
   .. automethod:: sel
   .. automethod:: to_numpy
   .. automethod:: to_dataframe
   .. automethod:: sqrt
   .. automethod:: inv
   .. automethod:: dot


ForwardOperator
---------------

.. autoclass:: ForwardOperator

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~ForwardOperator.blocks
      ~ForwardOperator.row_index
      ~ForwardOperator.col_index
      ~ForwardOperator.shape

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~ForwardOperator.copy
      ~ForwardOperator.sel
      ~ForwardOperator.to_numpy
      ~ForwardOperator.to_dataframe
      ~ForwardOperator.T
      ~ForwardOperator.dot
      ~ForwardOperator.apply
      ~ForwardOperator.apply_adjoint

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: blocks
   .. autoattribute:: row_index
   .. autoattribute:: col_index
   .. autoattribute:: shape
   .. automethod:: copy
   .. automethod:: sel
   .. automethod:: to_numpy
   .. automethod:: to_dataframe
   .. autoattribute:: T
   .. automethod:: dot
   .. automethod:: apply
   .. automethod:: apply_adjoint
