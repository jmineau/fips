=============
Visualization
=============

.. currentmodule:: fips.visualization

fips draws one plot, :func:`plot_comparison`: several vectors or Series on
one axes, with a band (or error bars) of one standard deviation from each
covariance, and an optional truth. It needs the ``viz`` extra
(``pip install "fips[viz]"``, matplotlib). Maps and other figures are drawn
from :meth:`fips.Vector.to_series` and :meth:`fips.Vector.to_xarray` with
pandas, xarray or matplotlib.

.. autosummary::
   :toctree: api/

   plot_comparison
