"""Ward and Counterseal definitions.

This module provides small data containers used by the ``magic`` package.  A
:class:`Ward` blocks Works that employ particular Arts or Substances.  A
:class:`Counterseal` can lift the restriction for matching Arts or Substances.

The implementation here intentionally keeps the rules light‑weight – the
project's tests exercise only simple set based matching and do not require the
complex rule systems that a full game might implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Set


class WorkLike(Protocol):
    """Protocol describing the pieces of a Work needed by wards.

    Any object with ``art`` and ``substances`` attributes can be checked by a
    :class:`Ward` or :class:`Counterseal`.  ``substances`` is expected to be an
    iterable of strings (a set is ideal).
    """

    art: str
    substances: Iterable[str]


@dataclass(frozen=True)
class Ward:
    """Blocks Works that match any of the configured Arts/Substances."""

    arts: Set[str] = field(default_factory=set)
    substances: Set[str] = field(default_factory=set)

    def blocks(self, work: WorkLike) -> bool:
        """Return ``True`` if this Ward prevents ``work`` from executing."""

        if work.art in self.arts:
            return True
        # ``substances`` may be any iterable; convert to ``set`` for intersection
        return bool(self.substances.intersection(set(work.substances)))


@dataclass(frozen=True)
class Counterseal:
    """Allows Works through Wards when the Art/Substance matches."""

    arts: Set[str] = field(default_factory=set)
    substances: Set[str] = field(default_factory=set)

    def allows(self, work: WorkLike) -> bool:
        """Return ``True`` if this Counterseal negates a blocking Ward."""

        if work.art in self.arts:
            return True
        return bool(self.substances.intersection(set(work.substances)))


def is_blocked(
    work: WorkLike,
    wards: Iterable[Ward],
    counterseals: Iterable[Counterseal] = (),
) -> bool:
    """Determine whether ``work`` is stopped by any of ``wards``.

    If a Ward blocks the Work and no Counterseal allows it, ``True`` is
    returned.  Otherwise ``False`` is returned, meaning the work may proceed.
    """

    for ward in wards:
        if ward.blocks(work):
            # See if a counterseal negates the block
            for seal in counterseals:
                if seal.allows(work):
                    break
            else:  # No counterseal matched
                return True
    return False


__all__ = ["Ward", "Counterseal", "is_blocked", "WorkLike"]
