"""Test package metadata."""

from importlib.metadata import version
from pathlib import Path

import fips


def test_version():
    """`__version__` is the installed distribution's version."""
    assert isinstance(fips.__version__, str)
    assert fips.__version__ == version("fips")


def test_no_blanket_type_ignore():
    """The source silences pyrefly only with ``# pyrefly: ignore[<code>]`` and a reason, never ``# type: ignore``."""
    src = Path(fips.__file__).parent
    found = [
        f"{path.relative_to(src)}:{number}"
        for path in sorted(src.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if "# type: ignore" in line
    ]
    assert not found, found
