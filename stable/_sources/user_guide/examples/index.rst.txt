========
Examples
========

Worked examples of the inverse problems fips solves, as Jupyter notebooks. They
run when the documentation is built, so the output you see is what the current
code produces.

.. toctree::
   :maxdepth: 1

   gravity_2d
   flux_regional

:doc:`gravity_2d`
   The density of buried prisms from the gravity measured above them: a small,
   self-contained problem with a correlated prior kept as Kronecker factors, and
   a sweep over the regularization factor.

:doc:`flux_regional`
   A year of monthly methane fluxes on a grid from daily concentrations at three
   sites: a sparse forward operator, a background as the offset, maps from
   ``to_xarray``, and the posterior of monthly totals.
