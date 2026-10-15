"""
The serialization mixin of the objects a solve keeps.

:class:`Pickleable` gives a class ``to_file`` and ``from_file`` for pickle
files.
"""

import logging
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Pickleable:
    """
    Mixin to add to_file() and from_file() methods for pickle serialization.

    Provides automatic pickle file I/O with extension validation (.pkl or .pickle).
    """

    VALID_EXTENSIONS = {".pkl", ".pickle"}

    def to_file(self, path: str | Path) -> None:
        """
        Save object to a pickle file.

        Parameters
        ----------
        path : str or Path
            File path where object will be saved.
        """
        path = Path(path)
        logger.debug(f"Serializing {type(self).__name__} to {path}")
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
        path : str or Path
            File path to load from.

        Returns
        -------
        object
            The unpickled object.
        """
        path = Path(path)
        logger.debug(f"Deserializing {cls.__name__} from {path}")
        if path.suffix not in cls.VALID_EXTENSIONS:
            raise ValueError(
                f"File extension must be one of {cls.VALID_EXTENSIONS}, got {path.suffix}"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    @abstractmethod
    def __getstate__(self) -> dict[str, Any]:
        """Return state as dict for pickling."""
        pass

    @abstractmethod
    def __setstate__(self, state: dict[str, Any]):
        """Restore state from dict for unpickling."""
        pass
