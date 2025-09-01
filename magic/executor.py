"""Execution helpers for Works.

The :func:`execute_work` function is a thin wrapper around calling a Work's
``perform`` method.  Before execution it checks any active :class:`Ward`
instances and optional :class:`Counterseal`s to ensure the Work is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Set

from .wards import Counterseal, Ward, is_blocked


@dataclass
class Work:
    """Simple container representing a magical Work.

    Only the ``art`` and ``substances`` attributes are required for interacting
    with Wards.  ``func`` holds the callable that performs the actual work when
    executed.
    """

    art: str
    substances: Set[str] = field(default_factory=set)
    func: Callable[[], object] | None = None

    def perform(self) -> object:
        """Execute the underlying callable if present."""

        if self.func is None:
            return None
        return self.func()


def execute_work(
    work: Work,
    *,
    wards: Iterable[Ward] = (),
    counterseals: Iterable[Counterseal] = (),
) -> bool:
    """Execute ``work`` if it is not blocked by a Ward.

    ``True`` is returned when the Work is executed.  If a Ward blocks the Work
    (and no Counterseal allows it) ``False`` is returned and the work's callable
    is not invoked.
    """

    if is_blocked(work, wards, counterseals):
        return False
    work.perform()
    return True


__all__ = ["Work", "execute_work"]
