==============
Forward models
==============

.. currentmodule:: fips

A forward map that is not linear, :math:`z = F(x) + c + e`, is a
:class:`ForwardModel`: a class you write with two methods, the model itself
and its Jacobian at a state (an :class:`Operator`). A :class:`Problem` takes
it as ``forward=``; :meth:`Problem.linearized` is the linear problem at a
state, and ``fips.solve(problem, "gauss_newton")`` finds the mode of the
posterior (:class:`~fips.methods.gauss_newton.GaussNewtonOptions`). A
model solved with ``"ensemble"`` needs only ``__call__``
(:doc:`/user_guide/ensemble`). A model with a tangent linear and an adjoint
writes :meth:`ForwardModel.jvp` and :meth:`ForwardModel.vjp` in place of the
Jacobian, and ``"variational"`` solves with those products alone
(:doc:`/user_guide/variational`).
:func:`taylortest` checks the Jacobian against the model, and its transpose
against it, as :func:`fips.linalg.dottest` checks an operator's transpose. The
guide is :doc:`/user_guide/nonlinear`.

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
   ForwardModel.jvp
   ForwardModel.vjp

Either kind of forward
~~~~~~~~~~~~~~~~~~~~~~
An :class:`Operator` is a forward model whose Jacobian is itself:
``H(x)`` is ``H @ x`` and ``H.jacobian(x)`` is ``H``. So a problem's
:attr:`~Problem.forward` is one type, :class:`~fips.forward.Forward`, which
both an operator and a :class:`ForwardModel` answer, and code that calls
``problem.forward(x)`` and ``problem.forward.jacobian(x)`` holds for a linear
problem and for a model.

.. currentmodule:: fips.forward

.. autosummary::
   :toctree: api/

   Forward
   Forward.rows
   Forward.cols
   Forward.__call__
   Forward.jacobian

.. currentmodule:: fips

Checking the Jacobian
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   taylortest

Values that stay positive
~~~~~~~~~~~~~~~~~~~~~~~~~
Two models that solve for a variable :math:`u` whose transform, on one block,
is a positive value; the other blocks pass through. The guide is
:doc:`/user_guide/positive`.

.. autosummary::
   :toctree: api/

   LogNormalForward
   LogNormalForward.H
   LogNormalForward.block
   LogNormalForward.reference
   LogNormalForward.__call__
   LogNormalForward.jacobian
   LogNormalForward.transform
   LogNormalForward.prior_for
   SoftplusForward
   SoftplusForward.H
   SoftplusForward.block
   SoftplusForward.scale
   SoftplusForward.__call__
   SoftplusForward.jacobian
   SoftplusForward.transform
   SoftplusForward.prior_for

Serialization
~~~~~~~~~~~~~
A problem with a model pickles when the model does: a class defined at the
top level of a module, holding data that pickles.
