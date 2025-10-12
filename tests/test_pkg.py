"""Test basic functionality of fips."""

import fips


def test_version():
    """Test that version is defined."""
    assert hasattr(fips, "__version__")
    assert isinstance(fips.__version__, str)


def test_author():
    """Test that author is defined."""
    assert hasattr(fips, "__author__")
    assert isinstance(fips.__author__, str)


def test_email():
    """Test that email is defined."""
    assert hasattr(fips, "__email__")
    assert isinstance(fips.__email__, str)
