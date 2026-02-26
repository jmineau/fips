Inverse Problem
===============

InverseProblem
--------------

.. currentmodule:: fips

.. autoclass:: InverseProblem

   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

      ~InverseProblem.prior
      ~InverseProblem.prior_cov
      ~InverseProblem.obs
      ~InverseProblem.obs_cov
      ~InverseProblem.forward_op

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~InverseProblem.solve
      ~InverseProblem.add_estimator
      ~InverseProblem.available_estimators

   .. rubric:: Details

   .. automethod:: __init__
   .. autoattribute:: prior
   .. autoattribute:: prior_cov
   .. autoattribute:: obs
   .. autoattribute:: obs_cov
   .. autoattribute:: forward_op
   .. automethod:: solve
   .. automethod:: add_estimator
   .. automethod:: available_estimators
