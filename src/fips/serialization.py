"""Serialization utilities for pickling and file I/O.

This module provides the Pickleable mixin for adding to_file/from_file methods
with .pkl/.pickle extension validation, and load_or_pass for transparent pickle loading.
"""

import pickle
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class Pickleable:
    """Mixin to add to_file() and from_file() methods for pickle serialization.
    
    Provides automatic pickle file I/O with extension validation (.pkl or .pickle).
    """

    VALID_EXTENSIONS = {".pkl", ".pickle"}

    def to_file(self, path: str | Path) -> None:
        """Save object to a pickle file.

        Parameters
        ----------
        path : str or Path
            File path where object will be saved.
        """
        path = Path(path)
        if path.suffix not in self.VALID_EXTENSIONS:
            raise ValueError(
                f"File extension must be one of {self.VALID_EXTENSIONS}, got {path.suffix}"
            )
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, path: str | Path):
        """Load object from a pickle file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        object
            The unpickled object.
        """
        path = Path(path)
        if path.suffix not in cls.VALID_EXTENSIONS:
            raise ValueError(
                f"File extension must be one of {cls.VALID_EXTENSIONS}, got {path.suffix}"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)


def load_or_pass(obj: str | Path | T) -> T:
    """Load an object from a pickle file if path, otherwise pass through.

    Parameters
    ----------
    obj : str, Path, or object
        Either a file path to a pickled object, or an object to return as-is.

    Returns
    -------
    object
        The unpickled object or the input object.
    """
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return obj
