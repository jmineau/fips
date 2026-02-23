API Reference
=============

.. contents:: On this page
   :local:
   :depth: 1


Core Data Structures
--------------------

Block
~~~~~

.. autoclass:: fips.Block
   :members:
   :undoc-members:
   :special-members: __init__

Vector
~~~~~~

.. autoclass:: fips.Vector
   :members:
   :undoc-members:
   :special-members: __init__

MatrixBlock
~~~~~~~~~~~

.. autoclass:: fips.MatrixBlock
   :members:
   :undoc-members:
   :special-members: __init__

Matrix
~~~~~~

.. autoclass:: fips.Matrix
   :members:
   :undoc-members:
   :special-members: __init__

CovarianceMatrix
~~~~~~~~~~~~~~~~

.. autoclass:: fips.CovarianceMatrix
   :members:
   :undoc-members:
   :special-members: __init__

ForwardOperator
~~~~~~~~~~~~~~~

.. autoclass:: fips.ForwardOperator
   :members:
   :undoc-members:
   :special-members: __init__


Inverse Problem
---------------

.. autoclass:: fips.InverseProblem
   :members:
   :undoc-members:
   :special-members: __init__


Estimators
----------

.. autoclass:: fips.Estimator
   :members:
   :undoc-members:
   :special-members: __init__

.. automodule:: fips.estimators
   :members:
   :exclude-members: Estimator, EstimatorRegistry


Pipeline
--------

.. autoclass:: fips.pipeline.InversionPipeline
   :members:
   :undoc-members:
   :special-members: __init__


Flux Inversion
--------------

.. autoclass:: fips.problems.flux.FluxInversion
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: fips.problems.flux.pipeline.FluxInversionPipeline
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: fips.problems.flux.visualization.FluxPlotter
   :members:
   :undoc-members:


Utilities
---------

convolve
~~~~~~~~

.. autofunction:: fips.convolve

Indexes
~~~~~~~

.. automodule:: fips.indexes
   :members:
   :undoc-members:

Kernels
~~~~~~~

.. automodule:: fips.kernels
   :members:
   :undoc-members:

Filters
~~~~~~~

.. automodule:: fips.filters
   :members:
   :undoc-members:

Metrics
~~~~~~~

.. automodule:: fips.metrics
   :members:
   :undoc-members:

Aggregators
~~~~~~~~~~~

.. automodule:: fips.aggregators
   :members:
   :undoc-members:
