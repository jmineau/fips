User Guide
==========

The guide follows the order of an inversion: build the problem, solve it,
read the answer. Most problems need only the first two pages. The rest are for
problems the linear Gaussian model does not fit, and for checking the
inversion itself. :doc:`/getting_started/first_problem` solves one problem
from start to finish, and :doc:`/getting_started/terminology` defines the
words.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - If you want to
     - Read
   * - put your data, your forward operator and your covariances into a
       problem
     - :doc:`problem`
   * - read the estimate, its uncertainty, aggregates, and the diagnostics
     - :doc:`posterior`
   * - pick a method for a problem that is not linear, has bounds, or is too
       large to factor
     - :doc:`methods`
   * - keep part of the state at or above zero
     - :doc:`bounds`, :doc:`positive`
   * - solve with a forward model that is not linear, and its Jacobian
     - :doc:`nonlinear`
   * - solve with a model that can only be run
     - :doc:`ensemble`
   * - solve a problem too large to factor, or with an adjoint model
     - :doc:`variational`
   * - choose the covariance settings from the data
     - :doc:`settings`
   * - carry the uncertainty of those settings into the answer
     - :doc:`hierarchical`
   * - test an inversion on data whose answer you know
     - :doc:`synthetic`
   * - see whole problems worked from start to finish
     - :doc:`examples/index`

.. toctree::
   :maxdepth: 1
   :caption: Every problem

   problem
   posterior

.. toctree::
   :maxdepth: 1
   :caption: Beyond the linear Gaussian problem

   methods
   bounds
   positive
   nonlinear
   ensemble
   variational

.. toctree::
   :maxdepth: 1
   :caption: Checking and tuning an inversion

   settings
   hierarchical
   synthetic

.. toctree::
   :maxdepth: 1

   examples/index
