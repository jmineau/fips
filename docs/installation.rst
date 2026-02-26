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

From GitHub
---------

.. code-block:: bash

   pip install git+https://github.com/jmineau/fips

Optional extras
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Atmospheric flux inversion (STILT transport, cartopy plotting)
   pip install "fips[flux]"

   # Documentation
   pip install "fips[docs]"

   # Development (linting, testing, type-checking)
   pip install "fips[dev]"

From Source
-----------

.. code-block:: bash

   git clone https://github.com/jmineau/fips.git
   cd fips
   pip install -e .

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/jmineau/fips.git
   cd fips
   python -m pip install --upgrade pip
   pip install -e ".[dev,docs]"
   pre-commit install
