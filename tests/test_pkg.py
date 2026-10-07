"""Test package metadata."""

from importlib.metadata import version

import fips


def test_version():
    """`__version__` is the installed distribution's version."""
    assert isinstance(fips.__version__, str)
    assert fips.__version__ == version("fips")
