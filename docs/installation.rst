Installation
============

From Source
-----------

To install from source:

.. code-block:: bash

   git clone https://github.com/jmineau/fips.git
   cd fips
   pip install -e .

Development Installation
------------------------

For development, install with the development dependencies:

.. code-block:: bash

   git clone https://github.com/jmineau/fips.git
   cd fips
   python -m pip install --upgrade pip
   pip install -e ".[dev,docs]"
   pre-commit install

Requirements
------------

- Python 3.10 or higher
