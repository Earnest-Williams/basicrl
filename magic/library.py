"""Utilities for tracking magical Works known by the player or game entities.

This module provides a small in-memory library that keeps track of "Works" -
the fundamental units of magic in the game world. A work can represent a
spell, ritual, or any other piece of arcane knowledge.

The :class:`MagicLibrary` acts as a registry of all works an entity has
learned. Two convenience functions, :func:`learn_work` and
:func:`research_work`, expose a simple command-style API which mirrors other
parts of the codebase. These commands will likely be hooked into a larger
command/console system in the future but are kept intentionally lightweight for
now so they can be used directly in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Work:
    """Representation of a magical Work.

    Parameters
    ----------
    name:
        Unique name for the work.
    description:
        Optional human readable description of the work.
    """

    name: str
    description: str = ""


class MagicLibrary:
    """Container for Works known to an entity.

    The library simply maps work names to :class:`Work` instances. Only the
    information necessary for the tests is implemented; additional metadata can
    be added in the future without changing the public API.
    """

    def __init__(self) -> None:
        self._works: Dict[str, Work] = {}

    def learn(self, work: Work) -> None:
        """Add a work to the library.

        If a work with the same name already exists it is replaced.
        """

        self._works[work.name] = work

    def knows(self, work_name: str) -> bool:
        """Return ``True`` if the work is present in the library."""

        return work_name in self._works

    def research(self, work_name: str) -> Optional[Work]:
        """Retrieve a work from the library.

        Parameters
        ----------
        work_name:
            Name of the work to look up.

        Returns
        -------
        Optional[Work]
            The corresponding :class:`Work` if it is known, otherwise ``None``.
        """

        return self._works.get(work_name)


# A module level instance makes the commands easy to use without manual setup
_library = MagicLibrary()


def learn_work(name: str, description: str = "") -> None:
    """Command to learn a new work.

    Parameters
    ----------
    name:
        Name of the work.
    description:
        Optional description of the work.
    """

    work = Work(name=name, description=description)
    _library.learn(work)


def research_work(name: str) -> Optional[Work]:
    """Command to look up a work by name.

    Returns the :class:`Work` instance if the work has been learnt, otherwise
    ``None``.
    """

    return _library.research(name)


__all__ = [
    "Work",
    "MagicLibrary",
    "learn_work",
    "research_work",
]
