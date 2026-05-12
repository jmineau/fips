Installation
============

Requirements
------------

- Python 3.10 or higher
- `NumPy <https://numpy.org>`_
- `pandas <https://pandas.pydata.org>`_
- `xarray <https://xarray.pydata.org>`_
- `scipy <https://scipy.org>`_
- `joblib <https://joblib.readthedocs.io>`_

From PyPI
---------

.. code-block:: bash

   pip install fips

Optional extras
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Atmospheric flux inversion (STILT transport, cartopy plotting)
   pip install "fips[flux]"

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/jmineau/fips.git
   cd fips
   python -m pip install --upgrade pip
   uv sync --dev
   pre-commit install
