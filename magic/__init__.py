"""Utilities for simple magic system involving Works, Wards and Counterseals."""

from .models import *  # noqa: F401,F403
from .executor import Work, execute_work
from .wards import Ward, Counterseal, is_blocked

__all__ = [
    "Work",
    "execute_work",
    "Ward",
    "Counterseal",
    "is_blocked",
]


