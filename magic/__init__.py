
"""Utilities for simple magic system involving Works, Wards and Counterseals, and a simple in-memory library of Works."""

# ``Work`` is defined in multiple modules.  The executor's ``Work`` represents the
# runtime action that can be performed, while the library module uses a simpler
# data container for tracking knowledge.  Export the executor's ``Work`` as the
# public class and alias the library variant to avoid confusion.
from .library import MagicLibrary, Work as LibraryWork, learn_work, research_work
from .models import *  # noqa: F401,F403
from .executor import Work, execute_work
from .wards import Ward, Counterseal, is_blocked

__all__ = [
    "MagicLibrary",
    "LibraryWork",
    "Work",
    "learn_work",
    "research_work",
    "execute_work",
    "Ward",
    "Counterseal",
    "is_blocked",
]
