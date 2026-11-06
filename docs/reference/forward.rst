==============
Forward models
==============

.. currentmodule:: fips

A forward map that is not linear, :math:`z = F(x) + c + e`, is a
:class:`ForwardModel`: a class you write with two methods, the model itself
and its Jacobian at a state (an :class:`Operator`). A :class:`Problem` takes
it as ``forward=``; :meth:`Problem.linearized` is the linear problem at a
state, and ``fips.solve(problem, "gauss_newton")`` finds the mode of the
posterior (:class:`~fips.methods.gauss_newton.GaussNewtonOptions`).
:func:`taylortest` checks the Jacobian against the model, as
:func:`fips.linalg.dottest` checks an operator's transpose. The guide is
:doc:`/user_guide/nonlinear`.

.. ipython:: python

   import numpy as np
   import pandas as pd
   import fips
   from fips.linalg import DiagonalOperator

   class Exp(fips.ForwardModel):
       def __init__(self, H):
           super().__init__(H.rows, H.cols)
           self.H = H
       def __call__(self, x):
           return self.H @ np.exp(x)
       def jacobian(self, x):
           return self.H @ fips.Operator(DiagonalOperator(np.exp(x.values)), x.space, x.space)

   state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
   observed = fips.Space({"co2": pd.RangeIndex(3, name="obs_id")})
   model = Exp(fips.Operator(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), observed, state))
   fips.taylortest(model, fips.Vector([0.1, -0.2], state))

The model
~~~~~~~~~
.. autosummary::
   :toctree: api/

   ForwardModel
   ForwardModel.rows
   ForwardModel.cols
   ForwardModel.__call__
   ForwardModel.jacobian

Checking the Jacobian
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   taylortest

Serialization
~~~~~~~~~~~~~
A problem with a model pickles when the model does: a class defined at the
top level of a module, holding data that pickles.
