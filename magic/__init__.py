"""Utilities for simple magic system involving Works, Wards and Counterseals, and a simple in-memory library of Works."""
from .library import MagicLibrary, Work, learn_work, research_work
from .models import *  # noqa: F401,F403
from .executor import Work, execute_work
from .wards import Ward, Counterseal, is_blocked

__all__ = [
    "MagicLibrary", 
    "Work", 
    "learn_work", 
    "research_work",
    "execute_work",
    "Ward",
    "Counterseal",
    "is_blocked",
]
