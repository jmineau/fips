"""Tests for the Pickleable mixin."""

import pytest

from fips.base import Pickleable


class DummyPickleable(Pickleable):
    """
    Dummy class to test the Pickleable mixin.
    """

    def __init__(self, val):
        self.val = val

    def __getstate__(self):
        """Return state for pickling."""
        return {"val": self.val}

    def __setstate__(self, state):
        """Restore state from pickling."""
        self.val = state["val"]


def test_pickleable(tmp_path):
    """
    Test serialization and deserialization using the Pickleable mixin.
    """
    obj = DummyPickleable(42)
    file_path = tmp_path / "test.pkl"

    obj.to_file(file_path)
    assert file_path.exists()

    loaded = DummyPickleable.from_file(file_path)
    assert loaded.val == 42

    # Test invalid extension
    with pytest.raises(ValueError):
        obj.to_file(tmp_path / "test.txt")

    with pytest.raises(ValueError):
        DummyPickleable.from_file(tmp_path / "test.txt")

    with pytest.raises(FileNotFoundError):
        DummyPickleable.from_file(tmp_path / "nonexistent.pkl")
