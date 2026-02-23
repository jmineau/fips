fips
====

.. image:: https://github.com/jmineau/fips/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/jmineau/fips/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://github.com/jmineau/fips/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/jmineau/fips/actions/workflows/docs.yml
   :alt: Documentation

.. image:: https://github.com/jmineau/fips/actions/workflows/quality.yml/badge.svg
   :target: https://github.com/jmineau/fips/actions/workflows/quality.yml
   :alt: Code Quality

.. image:: https://codecov.io/gh/jmineau/fips/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/jmineau/fips
   :alt: Code Coverage

.. image:: https://badge.fury.io/py/fips.svg
   :target: https://badge.fury.io/py/fips
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/fips.svg
   :target: https://pypi.org/project/fips/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

.. image:: https://img.shields.io/badge/pyright-checked-brightgreen.svg
   :target: https://github.com/microsoft/pyright
   :alt: Pyright

**Flexible Inverse Problem Solver** — a Pythonic framework for solving linear
inverse problems using Bayesian estimation.

*fips* is built around familiar :mod:`pandas` and :mod:`numpy` data structures,
and integrates naturally with the broader PyData ecosystem. It handles the
bookkeeping - labelled vectors, covariance matrices, forward operators, and
diagnostics - so you can focus on your science.

.. note::

   Inverse problems arise when you want to infer causes from observed
   effects: recovering a signal from noisy measurements, estimating surface
   greenhouse-gas fluxes from atmospheric concentrations, or inferring
   subsurface structure from geophysical data. A thorough theoretical treatment
   can be found in `Tarantola (2005) <https://doi.org/10.1137/1.9780898717921>`_
   and `Rodgers (2000) <https://doi.org/10.1142/3171>`_. *fips* currently focuses on
   the **linear Bayesian** case and is not intended as an authoritative reference.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api


.. include:: installation.rst

.. include:: usage.rst

.. include:: api.rst

Contributing
============

See the `CONTRIBUTING.md <https://github.com/jmineau/fips/blob/main/CONTRIBUTING.md>`_ file for guidelines on how to contribute to this project.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
