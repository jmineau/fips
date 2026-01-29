"""Serialization utilities for pickling and file I/O.

This module provides:
- Pickleable: A mixin class for easy pickle file save/load
- load_or_pass: A utility to load pickle files or pass through objects
"""

import pickle
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class Pickleable:
    """Mixin to add to_file() and from_file() methods for pickle serialization."""

    VALID_EXTENSIONS = {".pkl", ".pickle"}

    def to_file(self, path: str | Path) -> None:
        """
        Save object to a pickle file.

        Parameters
        ----------
        path : str | Path
            File path where object will be saved. Must have .pkl or .pickle extension.

        Raises
        ------
        ValueError
            If file extension is not .pkl or .pickle.
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
        """
        Load object from a pickle file.

        Parameters
        ----------
        path : str | Path
            File path to load from. Must have .pkl or .pickle extension.

        Returns
        -------
        cls
            The unpickled object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If file extension is not .pkl or .pickle.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix not in cls.VALID_EXTENSIONS:
            raise ValueError(
                f"File extension must be one of {cls.VALID_EXTENSIONS}, got {path.suffix}"
            )
        with open(path, "rb") as f:
            return pickle.load(f)


def load_or_pass(obj: str | Path | T) -> T:
    """
    Load an object from a pickle file if obj is a file path, otherwise return as-is.

    Parameters
    ----------
    obj : str | Path | T
        Either a file path (str or Path) to a pickled object, or an object to pass through.

    Returns
    -------
    T
        The unpickled object or the input object.

    Raises
    ------
    FileNotFoundError
        If the path doesn't exist.
    pickle.UnpicklingError
        If the file cannot be unpickled.
    """
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return obj
